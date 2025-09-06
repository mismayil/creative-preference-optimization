import argparse
from dotenv import load_dotenv
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
import pathlib
import pandas as pd

load_dotenv()

from utils import none_or_int, read_json, write_json, find_files


def prepare_ext_eval_data(
    input_paths, tasks=None, num_prompts=None, num_responses=None, metric="cap"
):
    all_input_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for input_path in tqdm(input_paths, desc="Processing input data"):
        input_data = read_json(input_path)
        model = input_data["metadata"]["model_name"]
        for sample in input_data["data"]:
            task = sample["task"]
            sample_id = sample["id"]

            if tasks is not None and task not in tasks:
                continue

            all_input_data[task][sample_id][model].append(
                (sample, input_data["metadata"])
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
                    all_input_data[task][sample_id][model] = random.sample(
                        responses, min(num_responses, len(responses))
                    )

    output_data = []

    for task, task_data in all_input_data.items():
        output_task_data = []
        for sample_id, sample_data in task_data.items():
            for model, responses in sample_data.items():
                for response, metadata in responses:
                    response_len_suffix = (
                        "Please limit your response to a few sentences"
                    )
                    prompt = (
                        response["user_prompt"].replace(response_len_suffix, "").strip()
                    )

                    if task == "Alternate Uses of Objects Task":
                        prefix = "Come up with an original and creative use for the following object:"
                        assert prompt.startswith(prefix)
                        prompt = prompt.replace(prefix, "").strip().strip(".").strip()
                    elif task == "Stories":
                        prefix = "Come up with an original and creative story which includes the following 3 words, make it around 5 sentences long:"
                        assert prompt.startswith(prefix)
                        prompt = "-".join(
                            [
                                w.strip().strip(".")
                                for w in prompt.replace(prefix, "").split(",")
                            ]
                        ).strip()

                    output_sample = {
                        "task": task,
                        "model": model,
                        "response": response["output"],
                        "prompt_id": sample_id,
                        "response_id": response["result_id"],
                    }

                    if metric == "cap":
                        output_sample["item"] = prompt
                    else:
                        output_sample["prompt"] = prompt

                    output_task_data.append(output_sample)

        output_data.append(output_task_data)

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
    parser.add_argument("-o", "--output-dir", type=str, required=True)
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
    parser.add_argument(
        "-sf",
        "--suffix",
        type=str,
        default="",
        help="Suffix to add to the output files.",
    )
    parser.add_argument(
        "-m",
        "--metric",
        type=str,
        default="cap",
        choices=["cap", "ocsai"],
        help="Metric to use for evaluation. Default is 'cap'.",
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

    output_data_lst = prepare_ext_eval_data(
        input_paths=input_paths,
        tasks=args.tasks,
        num_prompts=args.num_prompts,
        num_responses=args.num_responses,
        metric=args.metric,
    )

    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for output_data in output_data_lst:
        df = pd.DataFrame(output_data)
        output_file = (
            output_path
            / f"{args.metric}_eval_{output_data[0]['task'].lower().replace(' ', '_')}{args.suffix}.csv"
        )
        df.to_csv(output_file, index=False)
        print(f"Output written to {output_file}")


if __name__ == "__main__":
    main()
