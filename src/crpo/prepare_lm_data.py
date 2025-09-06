import argparse
import pathlib
from dotenv import load_dotenv

from tqdm import tqdm
from datasets import load_dataset

from crpo.utils import write_json, generate_unique_id

load_dotenv()


def prepare_lm_data(dataset, dataset_name, split):
    lm_data = []
    full_prompt_set = set()

    for i, row in tqdm(
        enumerate(dataset), total=len(dataset), desc="Preparing LM data"
    ):
        full_prompt = row["FullPrompt"]
        if full_prompt:
            if full_prompt in full_prompt_set:
                continue
            full_prompt_set.add(full_prompt)
            lm_data.append(
                {
                    "id": f"{dataset_name}-{split}-{generate_unique_id()}",
                    "original_prompt": row["Prompt"],
                    "full_prompt": row["FullPrompt"],
                    "dataset": row["Dataset"],
                    "task": row["TasksNamesFull"],
                    "dataset_split": row["DatasetsSplit"],
                    "language": row["Language"],
                }
            )

    return lm_data


def main():
    parser = argparse.ArgumentParser(description="Prepare LM raw data")
    parser.add_argument(
        "-i", "--input-dataset", type=str, help="Path to input dataset", required=True
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory path. Defaults to input directory path.",
    )
    parser.add_argument(
        "-c",
        "--subset-name",
        type=str,
        default="default",
        help="Name of the subset to create.",
    )

    args = parser.parse_args()
    input_dataset = load_dataset(args.input_dataset, name=None if args.subset_name == "default" else args.subset_name)
    splits = input_dataset.keys()

    for split in splits:
        dataset = input_dataset[split]
        dataset_name = args.input_dataset.split("/")[-1]
        lm_data = prepare_lm_data(dataset, dataset_name, split)
        output_path = f"{dataset_name}_{args.subset_name}_{split}_lm_data.json"
        output_dir = (
            pathlib.Path(args.output_dir)
            if args.output_dir is not None
            else pathlib.Path(args.input_dataset).parent
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = output_dir / output_path
        output_data = {
            "metadata": {
                "source": args.input_dataset,
                "subset": args.subset_name,
                "split": split,
                "dataset_name": dataset_name,
                "size": len(lm_data),
            },
            "data": lm_data,
        }
        write_json(output_data, output_file_path)
        print(f"Output data saved to {output_file_path}")


if __name__ == "__main__":
    main()
