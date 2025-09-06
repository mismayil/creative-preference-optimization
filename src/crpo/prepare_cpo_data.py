import argparse
from dotenv import load_dotenv
import numpy as np
from collections import defaultdict
from tqdm import tqdm

load_dotenv()

from datasets import DatasetDict, load_dataset, concatenate_datasets, Dataset
from data import (
    get_data_dict,
    sample_data,
    prepare_preference_data,
    prepare_split_data_by_prompt,
    flatten_data,
    convert_to_trl_pref_dataset,
    sample_pref_dataset_by_task,
)

from utils import none_or_int


def prepare_cpo_data(
    input_dataset,
    max_num_responses=100,
    min_score_diff=10,
    max_score_diff=None,
    rm_split=0.8,
    pref_template="chat",
    pref_format="implicit",
    eval_sample_sizes=[1024, 4096],
    eval_sampling_method="uniform",
    dry_run=False,
    max_matching=None,
    strict_label_checking=False,
    min_score=None,
    length_balancing=False,
    max_length_balancing_size=None,
    ignored_tasks=None,
):
    cpo_data = {}
    splits = list(input_dataset.keys())

    for split in splits:
        cpo_data[split] = input_dataset[split].to_pandas()
        cpo_data[split] = get_data_dict(cpo_data[split], ignored_tasks=ignored_tasks)
        if max_num_responses:
            cpo_data[split] = sample_data(cpo_data[split], max_num_responses)
        cpo_data[split] = prepare_preference_data(
            cpo_data[split],
            min_margin=min_score_diff,
            max_margin=max_score_diff,
            max_matching=max_matching,
            strict_label_checking=strict_label_checking,
            min_score=min_score,
            length_balancing=length_balancing,
            max_length_balancing_size=max_length_balancing_size,
        )

    if rm_split > 0:
        # prepare rm and po train data by splitting on the prompt
        rm_train_pref_data, po_train_pref_data = prepare_split_data_by_prompt(
            cpo_dataset["train"], split=rm_split
        )
        cpo_dataset["rm_train"] = rm_train_pref_data
        cpo_dataset["po_train"] = po_train_pref_data

    # flatten the preference data
    for split in splits:
        cpo_data[split] = flatten_data(cpo_data[split])

    # print the sizes of the preference data
    for split in splits:
        print(f"{split} data: {len(cpo_data[split])} samples")

    if dry_run:
        return

    # convert the preference data to trl preference dataset
    for split in splits:
        cpo_data[split] = convert_to_trl_pref_dataset(
            cpo_data[split], template=pref_template, template_format=pref_format
        )

    for split in ["val", "test"]:
        for eval_sample_size in eval_sample_sizes:
            cpo_data[f"{split}_sample{eval_sample_size}"] = sample_pref_dataset_by_task(
                cpo_data[split],
                sample_size=eval_sample_size,
                sampling_method=eval_sampling_method,
            )

    cpo_dataset = DatasetDict(cpo_data)

    return cpo_dataset


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
    parser.add_argument("-mnr", "--max-num-responses", type=none_or_int, default=None)
    parser.add_argument("-msd", "--min-score-diff", type=none_or_int, default=None)
    parser.add_argument("-xsd", "--max-score-diff", type=none_or_int, default=None)
    parser.add_argument("-ms", "--min-score", type=none_or_int, default=None)
    parser.add_argument("-rs", "--rm-split", type=float, default=0.0)
    parser.add_argument("-pt", "--pref-template", type=str, default="chat")
    parser.add_argument("-pf", "--pref-format", type=str, default="implicit")
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument(
        "-ess", "--eval-sample-sizes", type=int, nargs="+", default=[1024, 4096]
    )
    parser.add_argument("-esm", "--eval-sampling-method", type=str, default="uniform")
    parser.add_argument("-mm", "--max-matching", type=none_or_int, default=None)
    parser.add_argument("-d", "--dry-run", action="store_true")
    parser.add_argument(
        "-slc",
        "--strict_label_checking",
        action="store_true",
        help="Force scoring labels to match exactly",
    )
    parser.add_argument(
        "-lb",
        "--length-balancing",
        action="store_true",
        help="Balance the number of responses w.r.t their lengths",
    )
    parser.add_argument(
        "-mlbs", "--max-length-balancing-size", type=none_or_int, default=None
    )
    parser.add_argument(
        "-it",
        "--ignored-tasks",
        type=str,
        nargs="+",
        default=[],
        help="Tasks to ignore",
    )

    args = parser.parse_args()

    np.random.seed(args.seed)

    input_dataset = load_dataset(args.input_dataset)

    cpo_dataset = prepare_cpo_data(
        input_dataset,
        max_num_responses=args.max_num_responses,
        min_score_diff=args.min_score_diff,
        max_score_diff=args.max_score_diff,
        rm_split=args.rm_split,
        pref_template=args.pref_template,
        pref_format=args.pref_format,
        eval_sample_sizes=args.eval_sample_sizes,
        eval_sampling_method=args.eval_sampling_method,
        dry_run=args.dry_run,
        max_matching=args.max_matching,
        min_score=args.min_score,
        strict_label_checking=args.strict_label_checking,
        length_balancing=args.length_balancing,
        max_length_balancing_size=args.max_length_balancing_size,
        ignored_tasks=args.ignored_tasks,
    )

    if cpo_dataset:
        cpo_dataset.push_to_hub(args.output_dataset, private=True)
        print(f"Pushed the dataset to {args.output_dataset}")


if __name__ == "__main__":
    main()
