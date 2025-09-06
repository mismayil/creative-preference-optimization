import argparse
from dotenv import load_dotenv
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
import pathlib

load_dotenv()

from utils import none_or_int, read_json, write_json, find_files


def _get_metric_value(sample, metric):
    if "." in metric:
        metric_parts = metric.split(".")
        metric_value = sample["metrics"]
        for part in metric_parts:
            if isinstance(metric_value, list):
                metric_value = metric_value[0]
            if part in metric_value:
                metric_value = metric_value[part]
            else:
                return 0
        return metric_value
    else:
        return sample["metrics"].get(metric, 0)


def normalize_scores(scores):
    max_score = max(scores)
    min_score = min(scores)
    normalized_scores = [
        (
            (score - min_score) / (max_score - min_score)
            if max_score != min_score
            else 0.5
        )
        for score in scores
    ]
    return normalized_scores


def prepare_human_eval_data(
    input_paths,
    tasks=None,
    num_prompts=None,
    num_responses=None,
    extra_input_paths=None,
    metrics=None,
    prompt_ids=None,
):
    if prompt_ids is None:
        prompt_ids = []

    if metrics is None:
        metrics = []

    all_input_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    all_extra_input_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for input_path in tqdm(input_paths, desc="Processing input data"):
        input_data = read_json(input_path)
        model = input_data["metadata"]["model_name"]
        for sample in input_data["data"]:
            task = sample["task"]
            sample_id = sample["id"]

            if prompt_ids and sample_id not in prompt_ids:
                continue

            if tasks is not None and task not in tasks:
                continue

            all_input_data[task][sample_id][model].append(
                (sample, input_data["metadata"])
            )

    if extra_input_paths:
        for extra_input_path in tqdm(
            extra_input_paths, desc="Processing extra input data"
        ):
            extra_input_data = read_json(extra_input_path)
            model = extra_input_data["metadata"]["model_name"]
            for sample in extra_input_data["data"]:
                task = sample["task"]
                sample_id = sample["id"]

                if prompt_ids and sample_id not in prompt_ids:
                    continue

                if tasks is not None and task not in tasks:
                    continue

                all_extra_input_data[task][sample_id][model].append(
                    (sample, extra_input_data["metadata"])
                )

    if num_prompts is not None:
        all_input_data = {
            task: dict(list(task_data.items())[:num_prompts])
            for task, task_data in all_input_data.items()
        }

    if num_responses is not None:
        for task, task_data in all_input_data.items():
            for sample_id, sample_responses in task_data.items():
                for model, responses in sample_responses.items():
                    if not metrics:
                        all_input_data[task][sample_id][model] = random.sample(
                            responses, min(num_responses, len(responses))
                        )
                    else:
                        scores = np.array(
                            [
                                normalize_scores(
                                    [_get_metric_value(r[0], metric) for r in responses]
                                )
                                for metric in metrics
                            ]
                        )
                        scores = np.sum(scores, axis=0)
                        responses_with_scores = sorted(
                            zip(responses, scores), key=lambda rs: rs[1], reverse=True
                        )
                        all_input_data[task][sample_id][model] = responses_with_scores[
                            :num_responses
                        ]

    output_data = {
        "metadata": {
            "source": input_paths,
            "tasks": tasks,
            "num_prompts_per_task": num_prompts,
            "num_responses_per_prompt": num_responses,
        },
        "data": [],
    }

    response_set = set()

    for task, task_data in all_input_data.items():
        for sample_id, sample_data in task_data.items():
            for model, responses_with_scores in sample_data.items():
                for (response, metadata), response_score in responses_with_scores:
                    if response["output"].lower() in response_set:
                        print(f"Duplicate response found: {response['output']}")
                        if (
                            task in all_extra_input_data
                            and sample_id in all_extra_input_data[task]
                            and model in all_extra_input_data[task][sample_id]
                        ):
                            extra_samples = all_extra_input_data[task][sample_id].get(
                                model, []
                            )
                            for i, extra_sample in enumerate(extra_samples):
                                if (
                                    extra_sample[0]["output"].lower()
                                    != response["output"].lower()
                                ):
                                    print(
                                        f"Replacing with extra sample: {extra_sample[0]['output']}"
                                    )
                                    response = extra_sample[0]
                                    del extra_samples[i]
                                    break

                    output_sample = {
                        "task": task,
                        "model": model,
                        "prompt": response["user_prompt"],
                        "response": response["output"],
                        "prompt_id": sample_id,
                        "response_id": response["result_id"],
                        "model_args": (
                            metadata["model_args"]
                            if "model_args" in metadata
                            else response.get("model_args")
                        ),
                    }

                    if metrics:
                        for metric in metrics:
                            output_sample[metric] = _get_metric_value(response, metric)
                        output_sample["final_score"] = response_score

                    output_data["data"].append(output_sample)
                    response_set.add(response["output"].lower())

    output_data["metadata"]["num_samples"] = len(output_data["data"])

    return output_data


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-paths",
        type=str,
        nargs="+",
        required=True,
        help="Input paths (can be files or directories).",
    )
    parser.add_argument("-o", "--output-path", type=str, required=True)
    parser.add_argument(
        "-t",
        "--tasks",
        type=str,
        nargs="*",
        default=None,
        help="Tasks to include. If not specified, all tasks will be included.",
    )
    parser.add_argument(
        "-np",
        "--num-prompts",
        type=none_or_int,
        default=None,
        help="Number of prompts to include per task. If not specified, all prompts will be included.",
    )
    parser.add_argument(
        "-nr",
        "--num-responses",
        type=none_or_int,
        default=None,
        help="Number of responses to include per prompt. If not specified, all responses will be included.",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=42, help="Random seed for sampling."
    )
    parser.add_argument(
        "-ei",
        "--extra-input-paths",
        type=str,
        nargs="*",
        default=None,
        help="Extra input paths to use replace duplicated entries.",
    )
    parser.add_argument(
        "-fs",
        "--file-substrings",
        type=str,
        nargs="*",
        default=[],
        help="Substrings to filter files by. If not specified, all files will be included.",
    )
    parser.add_argument(
        "-m", "--metrics", type=str, nargs="*", help="Metrics for choosing the samples."
    )
    parser.add_argument(
        "-pids",
        "--prompt-ids",
        type=str,
        nargs="*",
        default=[],
        help="Specific prompt IDs to choose the samples from.",
    )

    args = parser.parse_args()

    np.random.seed(args.seed)
    input_paths = []

    for input_path in args.input_paths:
        if pathlib.Path(input_path).is_dir():
            json_files = find_files(input_path, "json")
            if args.file_substrings:
                json_files = [
                    f
                    for f in json_files
                    if any(substring in f for substring in args.file_substrings)
                ]
            input_paths.extend(json_files)
        else:
            input_paths.append(input_path)

    extra_input_paths = []
    if args.extra_input_paths:
        for extra_input_path in args.extra_input_paths:
            if pathlib.Path(extra_input_path).is_dir():
                extra_input_paths.extend(find_files(extra_input_path, "json"))
            else:
                extra_input_paths.append(extra_input_path)

    output_data = prepare_human_eval_data(
        input_paths=input_paths,
        tasks=args.tasks,
        num_prompts=args.num_prompts,
        num_responses=args.num_responses,
        extra_input_paths=extra_input_paths,
        metrics=args.metrics,
        prompt_ids=args.prompt_ids,
    )

    output_path = pathlib.Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_data, args.output_path)
    print(f"Output written to {args.output_path}")


if __name__ == "__main__":
    main()
