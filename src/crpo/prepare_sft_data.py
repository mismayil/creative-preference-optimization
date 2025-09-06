import argparse
from dotenv import load_dotenv
import numpy as np

load_dotenv()

from datasets import DatasetDict, load_dataset
from data import (
    get_data_dict,
    flatten_data,
    prepare_sft_data,
    convert_to_trl_sft_dataset,
)

from utils import none_or_int


def prepare_cpo_sft_data(
    input_dataset,
    dry_run=False,
    ignored_tasks=None,
    sft_topk=None,
    sft_template="chat",
    sft_min_score=None,
):
    sft_data = {}
    splits = list(input_dataset.keys())

    for split in splits:
        sft_data[split] = input_dataset[split].to_pandas()
        sft_data[split] = get_data_dict(sft_data[split], ignored_tasks=ignored_tasks)
        sft_data[split] = prepare_sft_data(
            sft_data[split], topk=sft_topk, min_score=sft_min_score
        )
        sft_data[split] = flatten_data(sft_data[split])

    for split in splits:
        print(f"{split} data: {len(sft_data[split])} samples")

    if dry_run:
        return

    # convert the sft data to trl sft dataset
    for split in splits:
        sft_data[split] = convert_to_trl_sft_dataset(
            sft_data[split], template=sft_template
        )

    sft_dataset = DatasetDict(sft_data)

    return sft_dataset


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-dataset",
        type=str,
        default="CNCL-Penn-State/cpo_en_multitask_text_fullprompt",
    )
    parser.add_argument("-o", "--output-dataset", type=str, required=True)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-d", "--dry-run", action="store_true")
    parser.add_argument(
        "-it",
        "--ignored-tasks",
        type=str,
        nargs="+",
        default=[],
        help="Tasks to ignore",
    )
    parser.add_argument(
        "--sft-topk",
        type=none_or_int,
        default=None,
        help="Top-k responses to sample per prompt for SFT",
    )
    parser.add_argument(
        "--sft-min-score", type=none_or_int, default=None, help="Minimum score for SFT"
    )
    parser.add_argument(
        "-st", "--sft-template", type=str, default="chat", help="Template for SFT"
    )

    args = parser.parse_args()

    np.random.seed(args.seed)

    input_dataset = load_dataset(args.input_dataset)

    sft_dataset = prepare_cpo_sft_data(
        input_dataset,
        dry_run=args.dry_run,
        ignored_tasks=args.ignored_tasks,
        sft_topk=args.sft_topk,
        sft_template=args.sft_template,
        sft_min_score=args.sft_min_score,
    )

    if sft_dataset:
        sft_dataset.push_to_hub(args.output_dataset, private=True)
        print(f"Pushed the dataset to {args.output_dataset}")


if __name__ == "__main__":
    main()
