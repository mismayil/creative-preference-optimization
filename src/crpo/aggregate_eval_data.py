import argparse
from dotenv import load_dotenv
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
import pathlib

load_dotenv()

from crpo.utils import none_or_int, read_json, write_json, find_files


def aggregate_eval_data(
    input_paths, output_path, tasks=None, num_prompts=None, num_responses=None
):
    # we assume we are aggregating data from the same model
    all_input_data = defaultdict(lambda: defaultdict(list))
    models = []
    model_names = []

    for input_path in tqdm(input_paths, desc="Processing input data"):
        input_data = read_json(input_path)
        models.append(input_data["metadata"]["model"])
        model_names.append(input_data["metadata"]["model_name"])
        for sample in input_data["data"]:
            task = sample["task"]
            sample_id = sample["id"]

            if tasks is not None and task not in tasks:
                continue

            all_input_data[task][sample_id].append((sample, input_data["metadata"]))

    if num_prompts is not None:
        all_input_data = {
            task: dict(list(task_data.items())[:num_prompts])
            for task, task_data in all_input_data.items()
        }

    if num_responses is not None:
        for task, task_data in all_input_data.items():
            for sample_id, sample_responses in task_data.items():
                all_input_data[task][sample_id] = random.sample(
                    sample_responses, min(num_responses, len(sample_responses))
                )

    if len(set(models)) > 1:
        raise ValueError(
            "Input data contains multiple models. Please provide data from a single model."
        )

    if len(set(model_names)) > 1:
        raise ValueError(
            "Input data contains multiple model names. Please provide data from a single model."
        )

    model = models[0]
    model_name = model_names[0]

    output_data = {
        "metadata": {
            "source": input_paths,
            "tasks": tasks,
            "num_prompts_per_task": num_prompts,
            "num_responses_per_prompt": num_responses,
            "dataset": pathlib.Path(output_path).stem,
            "model": model,
            "model_name": model_name,
        },
        "metrics": {},
        "data": [],
    }

    for task, task_data in all_input_data.items():
        for sample_id, results in task_data.items():
            for result, metadata in results:
                output_data["data"].append(
                    {
                        **{
                            key: value
                            for key, value in result.items()
                            if key != "metrics"
                        },
                        "model_args": metadata["model_args"],
                    }
                )

    output_data["metadata"]["num_samples"] = len(output_data["data"])

    return output_data


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Aggregate evaluation data from multiple JSON files for one model."
    )
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
        "-fs",
        "--file-substrings",
        type=str,
        nargs="*",
        default=[],
        help="Substrings to filter files by. If not specified, all files will be included.",
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

    output_data = aggregate_eval_data(
        input_paths=input_paths,
        output_path=args.output_path,
        tasks=args.tasks,
        num_prompts=args.num_prompts,
        num_responses=args.num_responses,
    )

    output_path = pathlib.Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_data, args.output_path)
    print(f"Output written to {args.output_path}")


if __name__ == "__main__":
    main()
