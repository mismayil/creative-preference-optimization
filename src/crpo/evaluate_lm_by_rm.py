import argparse
from tqdm import tqdm
import pathlib
from collections import defaultdict
from statistics import mean
from dotenv import load_dotenv

load_dotenv()

from utils import (
    read_json,
    write_json,
    find_files,
    make_lm_conv,
    wandb_log_run,
    prepare_metrics_for_wandb,
    get_model_name,
    load_rm,
)
from metrics import compute_quality


def compute_rm_metrics(results, config=None, reward_model=None, reward_tokenizer=None):
    metrics = results.get("metrics") or {}
    if "by_task" not in metrics:
        metrics["by_task"] = defaultdict(dict)
    metrics_by_task = metrics["by_task"]

    data = [result for result in results["data"] if "output" in result]

    data_by_task = defaultdict(list)

    for result in data:
        data_by_task[result["task"]].append(result)

    eval_data = []

    for sample in data:
        prompt = sample["user_prompt"]
        if isinstance(prompt, list):
            prompt = prompt[0]
        eval_data.append(make_lm_conv(prompt, sample["output"]))

    quality_scores = compute_quality(
        eval_data, reward_model, reward_tokenizer, batch_size=config["batch_size"]
    )

    for sample, score in zip(data, quality_scores):
        if "metrics" not in sample:
            sample["metrics"] = {}
        if "rewards" not in sample["metrics"]:
            sample["metrics"]["rewards"] = []
        model_name = get_model_name(config["reward_model"])
        existing_reward = [
            reward
            for reward in sample["metrics"]["rewards"]
            if reward["model"] == model_name
        ]
        if existing_reward:
            existing_reward[0]["score"] = score
        else:
            sample["metrics"]["rewards"].append({"model": model_name, "score": score})

    def _aggregate_metrics(metric_data):
        agg_metrics = {}
        model_score_map = defaultdict(list)
        for result in metric_data:
            if "rewards" in result["metrics"]:
                for reward in result["metrics"]["rewards"]:
                    model_score_map[reward["model"]].append(reward["score"])
        if model_score_map:
            agg_metrics["avg_rewards"] = {
                model: mean(scores) for model, scores in model_score_map.items()
            }

        return agg_metrics

    metrics.update(_aggregate_metrics(data))

    for task, task_data in data_by_task.items():
        metrics_by_task[task].update(_aggregate_metrics(task_data))

    return metrics


def report_rm_metrics(
    results_files, config=None, reward_model=None, reward_tokenizer=None
):
    for results_file in results_files:
        results = read_json(results_file)

        try:
            if "data" in results:
                print(f"Reporting RM metrics for: {results_file}")
                metrics = compute_rm_metrics(
                    results,
                    config=config,
                    reward_model=reward_model,
                    reward_tokenizer=reward_tokenizer,
                )
                results["metadata"]["rm_metrics_config"] = config
                results["metrics"] = metrics

                if config["wandb"]:
                    run_metrics = prepare_metrics_for_wandb(
                        metrics, exclude_prefixes=["num_", "usage", "cost"]
                    )
                    metadata = results["metadata"]
                    model_name = metadata.get("model_name", metadata["model"])
                    previous_run_id = metadata.get("eval_wandb_run_id")
                    dataset = metadata.get("dataset")

                    if not dataset:
                        dataset = metadata["source"].split("/")[-1]
                        metadata["dataset"] = dataset

                    wandb_run = wandb_log_run(
                        name=model_name,
                        project=config["wandb_project"],
                        metrics=run_metrics,
                        config=metadata,
                        run_id=previous_run_id,
                    )
                    metadata["eval_wandb_run_id"] = wandb_run.id

                write_json(results, results_file, ensure_ascii=False)
        except Exception as e:
            print(results_file)
            raise e


def main():
    parser = argparse.ArgumentParser(
        prog="evaluate_lm_by_rm",
        description="Evaluate language model outputs using reward model",
    )
    parser.add_argument(
        "-r",
        "--results-path",
        type=str,
        help="Path to evaluation results file in json or directory",
        required=True,
    )
    parser.add_argument(
        "-rm",
        "--reward-model",
        type=str,
        help="Reward model to use for evaluation",
        default="Skywork/Skywork-Reward-Gemma-2-27B-v0.2",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    parser.add_argument("-w", "--wandb", type=bool, default=True)
    parser.add_argument(
        "-p", "--wandb-project", type=str, default="project-cpo-lm-evals"
    )
    parser.add_argument(
        "-c", "--config", type=str, help="Path to config file", default=None
    )

    args = parser.parse_args()

    config = vars(args)

    if args.config:
        config.update(read_json(args.config))

    files_to_process = []

    results_path = pathlib.Path(config["results_path"])

    if results_path.is_file():
        files_to_process.append(config["results_path"])
    else:
        files_to_process.extend(find_files(config["results_path"], extension="json"))

    reward_model, reward_tokenizer = (
        load_rm(config["reward_model"]) if config.get("reward_model") else (None, None)
    )

    report_rm_metrics(
        files_to_process,
        config=config,
        reward_model=reward_model,
        reward_tokenizer=reward_tokenizer,
    )


if __name__ == "__main__":
    main()
