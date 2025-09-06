import argparse
from dotenv import load_dotenv
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
import pathlib
from itertools import product

load_dotenv()

from utils import none_or_int, read_json, write_json, find_files


def prepare_human_pref_eval_data(input_paths, aggregate_responses=False):
    all_input_data_by_prompt = []

    for input_path in tqdm(input_paths, desc="Processing input data"):
        input_data = read_json(input_path)
        input_data_by_prompt = defaultdict(lambda: defaultdict(list))
        for sample in input_data["data"]:
            input_data_by_prompt[sample["prompt_id"]][sample["model"]].append(sample)
        all_input_data_by_prompt.append(input_data_by_prompt)

    all_pref_data_by_prompt = defaultdict(list)
    for input_data_by_prompt in all_input_data_by_prompt:
        for prompt_id, model_data in input_data_by_prompt.items():
            model_samples_lst = []
            for model, samples in model_data.items():
                model_samples_lst.append(
                    sorted(samples, key=lambda x: x["final_score"])
                )
            all_pref_data_by_prompt[prompt_id].append(model_samples_lst)

    pref_data_by_prompt = {}

    for prompt_id, model_samples_lst in all_pref_data_by_prompt.items():
        pref_data_by_prompt[prompt_id] = list(product(*model_samples_lst))

    pref_data = []

    for prompt_id, pairs_lst in pref_data_by_prompt.items():
        for samples_lst in pairs_lst:
            preferences = []
            if aggregate_responses:
                preference = []
                for samples in samples_lst:
                    preference.append(
                        {
                            "task": samples[0]["task"],
                            "prompt": samples[0]["prompt"],
                            "prompt_id": prompt_id,
                            "model": samples[0]["model"],
                            "response": "\n".join(
                                [
                                    f"{i+1}. {sample['response']}"
                                    for i, sample in enumerate(samples)
                                ]
                            ),
                            "response_id": "-".join(
                                [sample["response_id"] for sample in samples]
                            ),
                        }
                    )
                preferences.append(preference)
            else:
                preferences = zip(*samples_lst)

            for pref_samples in preferences:
                pref_data_sample = {
                    "task": pref_samples[0]["task"],
                    "prompt": pref_samples[0]["prompt"],
                    "prompt_id": prompt_id,
                }
                for i, sample in enumerate(pref_samples):
                    pref_data_sample[f"response_{i+1}"] = sample["response"]
                    pref_data_sample[f"model_{i+1}"] = sample["model"]
                    pref_data_sample[f"response_id_{i+1}"] = sample["response_id"]

                pref_data.append(pref_data_sample)

    output_data = {
        "metadata": {
            "source": input_paths,
            "num_samples": len(pref_data),
            "aggregate_responses": aggregate_responses,
        },
        "data": pref_data,
    }

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
        "-a", "--aggregate-responses", action="store_true", help="Aggregate responses."
    )
    parser.add_argument(
        "-f",
        "--output-format",
        type=str,
        default="csv",
        choices=["json", "csv"],
        help="Output format.",
    )

    args = parser.parse_args()

    input_paths = []

    for input_path in args.input_paths:
        if pathlib.Path(input_path).is_dir():
            json_files = find_files(input_path, "json")
            input_paths.extend(json_files)
        else:
            input_paths.append(input_path)

    output_data = prepare_human_pref_eval_data(
        input_paths=input_paths,
        aggregate_responses=args.aggregate_responses,
    )

    output_path = pathlib.Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.output_format == "csv":
        import pandas as pd

        df = pd.DataFrame(output_data["data"])
        df.to_csv(output_path, index=False)
    elif args.output_format == "json":
        write_json(output_data, args.output_path)

    print(f"Output written to {args.output_path}")


if __name__ == "__main__":
    main()
