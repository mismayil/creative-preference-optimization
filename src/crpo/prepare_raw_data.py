import argparse
from dotenv import load_dotenv
import numpy as np

load_dotenv()

from datasets import DatasetDict, load_dataset, disable_caching
from data import prepare_full_prompt


def prepare_raw_data(input_dataset, modalities=None, languages=None, tasks=None):
    if modalities is None:
        modalities = []
    if languages is None:
        languages = []
    if tasks is None:
        tasks = []

    modalities = [modality.lower() for modality in modalities]
    languages = [language.lower() for language in languages]
    tasks = [task.lower() for task in tasks]

    print("Input dataset:", input_dataset)

    input_dataset = input_dataset["train"]
    text_only = modalities == ["text"]

    if text_only:
        # this makes preprocessing faster
        print("Removing non-text modalities from the dataset")
        input_dataset = input_dataset.remove_columns(["image"])

    if modalities:
        print("Filtering dataset by modalities")
        input_dataset = input_dataset.filter(
            lambda x: x["TaskType"].lower() in modalities
        )
        print("Dataset after filtering for modalities:", input_dataset)

    if languages:
        print("Filtering dataset by languages")
        input_dataset = input_dataset.filter(
            lambda x: x["Language"].lower() in languages
        )
        print("Dataset after filtering for languages:", input_dataset)

    if tasks:
        print("Filtering dataset by tasks")
        input_dataset = input_dataset.filter(
            lambda x: x["TasksNamesFull"].lower() in tasks
        )
        print("Dataset after filtering for tasks:", input_dataset)

    print("Adding FullPrompt column")
    input_dataset = input_dataset.map(
        lambda x: {
            "FullPrompt": prepare_full_prompt(
                x["TasksNamesFull"], x["Dataset"], x["Prompt"], x["Language"]
            )
        }
    )

    print("Creating dataset splits")
    train_dataset = input_dataset.filter(lambda x: x["Set"] == "Training")
    val_dataset = input_dataset.filter(lambda x: x["Set"] == "Validation")
    test_dataset = input_dataset.filter(lambda x: x["Set"] == "Test")
    heldout_item_dataset = input_dataset.filter(lambda x: x["Set"] == "HeldoutItem")
    heldout_task_dataset = input_dataset.filter(lambda x: x["Set"] == "HeldoutTask")
    heldout_language_dataset = input_dataset.filter(
        lambda x: x["Set"] == "HeldoutLanguage"
    )

    print(train_dataset[0])

    splits = [
        ("train", train_dataset),
        ("val", val_dataset),
        ("test", test_dataset),
        ("heldout_item", heldout_item_dataset),
        ("heldout_task", heldout_task_dataset),
        ("heldout_language", heldout_language_dataset),
    ]

    cpo_data = {}

    for split_name, split_dataset in splits:
        if len(split_dataset) > 0:
            cpo_data[split_name] = split_dataset

    cpo_dataset = DatasetDict(cpo_data)

    return cpo_dataset


def main():
    load_dotenv()
    # to make sure the dataset is updated
    disable_caching()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-dataset",
        type=str,
        default="CNCL-Penn-State/MultitaskDataset_Complete",
        help="Input dataset name",
    )
    parser.add_argument(
        "-o", "--output-dataset", type=str, required=True, help="Output dataset name"
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "-m",
        "--modalities",
        type=str,
        nargs="*",
        default=[],
        help="Modalities to include. If empty, all modalities are included.",
    )
    parser.add_argument(
        "-l",
        "--languages",
        type=str,
        nargs="*",
        default=[],
        help="Languages to include. If empty, all languages are included.",
    )
    parser.add_argument(
        "-t",
        "--tasks",
        type=str,
        nargs="*",
        default=[],
        help="Tasks to include. If empty, all tasks are included.",
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="If set, the dataset will not be pushed to the hub.",
    )

    args = parser.parse_args()

    np.random.seed(args.seed)

    input_dataset = load_dataset(args.input_dataset)

    cpo_dataset = prepare_raw_data(
        input_dataset,
        modalities=args.modalities,
        languages=args.languages,
        tasks=args.tasks,
    )

    print(cpo_dataset)

    if args.dry_run:
        print("Dry run mode. Not pushing the dataset to the hub.")
        return

    if cpo_dataset:
        cpo_dataset.push_to_hub(args.output_dataset, private=True)
        print(f"Pushed the dataset to {args.output_dataset}")


if __name__ == "__main__":
    main()
