import argparse
from dotenv import load_dotenv
import numpy as np
from collections import defaultdict
from tqdm import tqdm

load_dotenv()

from datasets import load_dataset, Dataset, DatasetDict

from utils import read_json, load_lm, load_rm, make_lm_conv, none_or_int
from data import get_prompt_and_responses_from_trl_sample
from metrics import (
    DEF_EMB_MODEL,
    DEF_EMB_TYPE,
    DEF_DIST_FN,
    DEF_EMB_STRATEGY,
    compute_inverse_homogenization,
    compute_novelty,
    compute_perplexity,
    compute_dsi,
    compute_quality,
)


def extend_cpo_dataset_with_diversity(
    input_dataset,
    config=None,
):
    data_by_prompt = defaultdict(lambda: defaultdict(float))

    for sample in input_dataset:
        prompt, chosen_response, rejected_response = (
            get_prompt_and_responses_from_trl_sample(sample)
        )
        data_by_prompt[prompt][chosen_response] = 0

        if config.get("include_rejected", False):
            data_by_prompt[prompt][rejected_response] = 0

    for prompt, prompt_data in tqdm(
        data_by_prompt.items(), desc="Computing diversity scores per prompt"
    ):
        responses = list(prompt_data.keys())
        if len(responses) > 1:
            diversity_scores = compute_inverse_homogenization(
                responses,
                emb_model=config.get("emb_model", DEF_EMB_MODEL),
                emb_type=config.get("emb_type", DEF_EMB_TYPE),
                emb_strategy=config.get("emb_strategy", DEF_EMB_STRATEGY),
                distance_fn=config.get("distance_fn", DEF_DIST_FN),
            )
            max_diversity_score = max(diversity_scores)
            min_diversity_score = min(diversity_scores)
            diversity_scores = [
                (
                    (score - min_diversity_score)
                    / (max_diversity_score - min_diversity_score)
                    if max_diversity_score != min_diversity_score
                    else 0.5
                )
                for score in diversity_scores
            ]

            if config.get("normalize_by_sum", False):
                diversity_scores = [
                    score / sum(diversity_scores) for score in diversity_scores
                ]

            for response, diversity_score in zip(responses, diversity_scores):
                data_by_prompt[prompt][response] = diversity_score
        else:
            # If there is only one response, set the diversity score to 0
            for response in responses:
                data_by_prompt[prompt][response] = 0.0

    extended_data = []

    for sample in input_dataset:
        prompt, chosen_response, rejected_response = (
            get_prompt_and_responses_from_trl_sample(sample)
        )
        extended_sample = {
            **sample,
            "diversity_chosen": data_by_prompt[prompt][chosen_response],
        }

        if config.get("include_rejected", False):
            extended_sample["diversity_rejected"] = data_by_prompt[prompt][
                rejected_response
            ]
        extended_data.append(extended_sample)

    return Dataset.from_generator(lambda: extended_data)


def extend_cpo_dataset_with_novelty(
    input_dataset,
    config=None,
):
    preprocessing_args = {
        "lower": config["lower"],
        "remove_punct": config["remove_punct"],
        "remove_stopwords": config["remove_stopwords"],
        "lemmatize": config["lemmatize"],
        "dominant_k": config["dominant_k"],
        "unique": config["unique"],
        "no_spacy": config["no_spacy"],
    }
    data_by_prompt = defaultdict(lambda: defaultdict(float))
    external_references = config.get("external_references", [])
    external_data_by_prompt = defaultdict(list)

    if external_references:
        print("Using external references for novelty computation")
        for reference in external_references:
            if reference.endswith(".json"):
                reference_data = read_json(reference)
                for sample in reference_data["data"]:
                    prompt, response = sample["full_prompt"], sample["output"]
                    external_data_by_prompt[prompt].append(response)
            else:
                raise ValueError(
                    f"Unknown external reference type: {reference}. Only .json files are supported."
                )

    for sample in input_dataset:
        prompt, chosen_response, rejected_response = (
            get_prompt_and_responses_from_trl_sample(sample)
        )
        data_by_prompt[prompt][chosen_response] = 0
        if config.get("include_rejected", False):
            data_by_prompt[prompt][rejected_response] = 0

    for prompt, prompt_data in tqdm(
        data_by_prompt.items(), desc="Computing novelty scores per prompt"
    ):
        responses = list(prompt_data.keys())

        novelty_scores = []

        if external_references:
            external_prompt_responses = external_data_by_prompt[prompt]
            external_corpus = " ".join(external_prompt_responses)
            external_corpus_dsi = compute_dsi(
                external_corpus,
                emb_model=config.get("emb_model", DEF_EMB_MODEL),
                emb_type=config.get("emb_type", DEF_EMB_TYPE),
                distance_fn=config.get("distance_fn", DEF_DIST_FN),
                preprocessing_args=preprocessing_args,
            )
            for response in responses:
                response_dsi = compute_dsi(
                    response,
                    emb_model=config.get("emb_model", DEF_EMB_MODEL),
                    emb_type=config.get("emb_type", DEF_EMB_TYPE),
                    distance_fn=config.get("distance_fn", DEF_DIST_FN),
                    preprocessing_args=preprocessing_args,
                )
                novelty_score = 2 * abs(response_dsi - external_corpus_dsi)
                novelty_scores.append(novelty_score)
        else:
            if len(responses) > 1:
                novelty_scores = compute_novelty(
                    responses,
                    emb_model=config.get("emb_model", DEF_EMB_MODEL),
                    emb_type=config.get("emb_type", DEF_EMB_TYPE),
                    distance_fn=config.get("distance_fn", DEF_DIST_FN),
                    preprocessing_args=preprocessing_args,
                )
            else:
                # If there is only one response, set the novelty score to 0
                novelty_scores = [0.0] * len(responses)

        if novelty_scores:
            max_novelty_score = max(novelty_scores)
            min_novelty_score = min(novelty_scores)
            novelty_scores = [
                (
                    (score - min_novelty_score)
                    / (max_novelty_score - min_novelty_score)
                    if max_novelty_score != min_novelty_score
                    else 0.5
                )
                for score in novelty_scores
            ]

            if config.get("normalize_by_sum", False):
                novelty_scores = [
                    score / sum(novelty_scores) for score in novelty_scores
                ]

            for response, novelty_score in zip(responses, novelty_scores):
                data_by_prompt[prompt][response] = novelty_score

    extended_data = []

    for sample in input_dataset:
        prompt, chosen_response, rejected_response = (
            get_prompt_and_responses_from_trl_sample(sample)
        )
        extended_sample = {
            **sample,
            "novelty_chosen": data_by_prompt[prompt][chosen_response],
        }

        if config.get("include_rejected", False):
            extended_sample["novelty_rejected"] = data_by_prompt[prompt][
                rejected_response
            ]

        extended_data.append(extended_sample)

    return Dataset.from_generator(lambda: extended_data)


def extend_cpo_dataset_with_surprise(
    input_dataset,
    config=None,
):
    print(
        "This script only supports instruction-tuned reference model. Make sure to use a model that is instruction-tuned."
    )

    data_by_prompt = defaultdict(lambda: defaultdict(float))

    reference_model, reference_tokenizer = load_lm(
        config["reference_model"], device=config.get("device")
    )

    for sample in input_dataset:
        prompt, chosen_response, rejected_response = (
            get_prompt_and_responses_from_trl_sample(sample)
        )
        data_by_prompt[prompt][chosen_response] = 0

        if config.get("include_rejected", False):
            data_by_prompt[prompt][rejected_response] = 0

    for prompt, prompt_data in tqdm(
        data_by_prompt.items(), desc="Computing surprise scores per prompt"
    ):
        responses = list(prompt_data.keys())
        perplexity_data = [make_lm_conv(prompt, response) for response in responses]
        surprise_scores = compute_perplexity(
            perplexity_data,
            model=reference_model,
            tokenizer=reference_tokenizer,
            batch_size=config.get("batch_size", 8),
        )
        max_surprise_score = max(surprise_scores)
        min_surprise_score = min(surprise_scores)
        surprise_scores = [
            (
                (score - min_surprise_score) / (max_surprise_score - min_surprise_score)
                if max_surprise_score != min_surprise_score
                else 0.5
            )
            for score in surprise_scores
        ]

        if config.get("normalize_by_sum", False):
            surprise_scores = [
                score / sum(surprise_scores) for score in surprise_scores
            ]

        for response, surprise_score in zip(responses, surprise_scores):
            data_by_prompt[prompt][response] = surprise_score

    extended_data = []

    for sample in input_dataset:
        prompt, chosen_response, rejected_response = (
            get_prompt_and_responses_from_trl_sample(sample)
        )
        extended_sample = {
            **sample,
            "surprise_chosen": data_by_prompt[prompt][chosen_response],
        }

        if config.get("include_rejected", False):
            extended_sample["surprise_rejected"] = data_by_prompt[prompt][
                rejected_response
            ]

        extended_data.append(extended_sample)

    return Dataset.from_generator(lambda: extended_data)


def extend_cpo_dataset_with_quality(
    input_dataset,
    config=None,
):
    data_by_prompt = defaultdict(lambda: defaultdict(float))

    quality_model, quality_tokenizer = load_rm(
        config["quality_model"], device=config.get("device")
    )

    for sample in input_dataset:
        prompt, chosen_response, rejected_response = (
            get_prompt_and_responses_from_trl_sample(sample)
        )
        data_by_prompt[prompt][chosen_response] = 0

        if config.get("include_rejected", False):
            data_by_prompt[prompt][rejected_response] = 0

    for prompt, prompt_data in tqdm(
        data_by_prompt.items(), desc="Computing quality scores per prompt"
    ):
        responses = list(prompt_data.keys())
        quality_data = [make_lm_conv(prompt, response) for response in responses]
        quality_scores = compute_quality(
            quality_data,
            model=quality_model,
            tokenizer=quality_tokenizer,
            batch_size=config.get("batch_size", 8),
        )
        max_quality_score = max(quality_scores)
        min_quality_score = min(quality_scores)
        quality_scores = [
            (
                (score - min_quality_score) / (max_quality_score - min_quality_score)
                if max_quality_score != min_quality_score
                else 0.5
            )
            for score in quality_scores
        ]

        if config.get("normalize_by_sum", False):
            quality_scores = [score / sum(quality_scores) for score in quality_scores]

        for response, quality_score in zip(responses, quality_scores):
            data_by_prompt[prompt][response] = quality_score

    extended_data = []

    for sample in input_dataset:
        prompt, chosen_response, rejected_response = (
            get_prompt_and_responses_from_trl_sample(sample)
        )
        extended_sample = {
            **sample,
            "quality_chosen": data_by_prompt[prompt][chosen_response],
        }

        if config.get("include_rejected", False):
            extended_sample["quality_rejected"] = data_by_prompt[prompt][
                rejected_response
            ]

        extended_data.append(extended_sample)

    return Dataset.from_generator(lambda: extended_data)


def extend_cpo_dataset(
    input_dataset,
    metrics=None,
    config=None,
):
    if metrics is None:
        metrics = []

    for metric in metrics:
        if metric == "diversity":
            input_dataset = extend_cpo_dataset_with_diversity(
                input_dataset, config=config
            )
        elif metric == "novelty":
            input_dataset = extend_cpo_dataset_with_novelty(
                input_dataset, config=config
            )
        elif metric == "surprise":
            input_dataset = extend_cpo_dataset_with_surprise(
                input_dataset, config=config
            )
        elif metric == "quality":
            input_dataset = extend_cpo_dataset_with_quality(
                input_dataset, config=config
            )
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return input_dataset


def extend_cpo_data(
    input_dataset,
    metrics=None,
    config=None,
):
    splits = list(input_dataset.keys())
    extended_dataset = {}

    for split in splits:
        extended_dataset[split] = extend_cpo_dataset(
            input_dataset[split],
            metrics=metrics,
            config=config,
        )
    return DatasetDict(extended_dataset)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-dataset",
        type=str,
        default=None,
        help="Input preference dataset",
        required=True,
    )
    parser.add_argument(
        "-c", "--config", type=str, help="Path to config file", default=None
    )
    parser.add_argument("-o", "--output-dataset", type=str, default=None, required=True)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument(
        "-m",
        "--metrics",
        type=str,
        nargs="+",
        default=[],
        choices=["diversity", "novelty", "surprise", "quality"],
        help="Metrics to compute",
    )
    parser.add_argument(
        "-rfm",
        "--reference-model",
        type=str,
        help="Reference model",
        default="google/gemma-2-27b-it",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    parser.add_argument(
        "-er",
        "--external-references",
        type=str,
        nargs="*",
        default=[],
        help="External references for the metric computation",
    )
    parser.add_argument("-ir", "--include-rejected", action="store_true", default=False)
    parser.add_argument(
        "-qm",
        "--quality-model",
        type=str,
        help="Quality model",
        default="Skywork/Skywork-Reward-Gemma-2-27B-v0.2",
    )
    parser.add_argument(
        "-nbs", "--normalize-by-sum", action="store_true", default=False
    )

    emb_group = parser.add_argument_group("Embedding arguments")
    emb_group.add_argument(
        "-em",
        "--emb-model",
        type=str,
        help="Sentence embedding model",
        default=DEF_EMB_MODEL,
    )
    emb_group.add_argument(
        "-et", "--emb-type", type=str, help="Embedding type", default=DEF_EMB_TYPE
    )
    emb_group.add_argument(
        "-es",
        "--emb-strategy",
        type=str,
        help="Embedding strategy",
        default=DEF_EMB_STRATEGY,
    )
    emb_group.add_argument(
        "-d", "--distance-fn", type=str, help="Distance function", default=DEF_DIST_FN
    )

    pp_group = parser.add_argument_group("Preprocessing arguments")
    pp_group.add_argument(
        "--no-spacy",
        type=bool,
        default=False,
        help="Turn off all processing using spacy, including metrics that depend on it. This should always be used for multilingual models, since spacy doesn't support all ORACL languages.",
    )
    pp_group.add_argument(
        "-l", "--lower", type=bool, help="Lowercase text", default=True
    )
    pp_group.add_argument(
        "-rp", "--remove-punct", type=bool, help="Remove punctuation", default=True
    )
    pp_group.add_argument(
        "-rs", "--remove-stopwords", type=bool, help="Remove stopwords", default=False
    )
    pp_group.add_argument(
        "-lm", "--lemmatize", type=bool, help="Lemmatize text", default=False
    )
    pp_group.add_argument(
        "-dk",
        "--dominant-k",
        type=none_or_int,
        help="Number of dominant words to consider",
        default=None,
    )
    pp_group.add_argument(
        "-q", "--unique", type=bool, help="Unique words", default=True
    )

    args = parser.parse_args()

    config = vars(args)

    if args.config:
        config.update(read_json(args.config))

    np.random.seed(args.seed)

    input_dataset = load_dataset(args.input_dataset)

    cpo_dataset = extend_cpo_data(
        input_dataset,
        metrics=args.metrics,
        config=config,
    )

    if cpo_dataset:
        cpo_dataset.push_to_hub(args.output_dataset, private=True)
        print(f"Pushed the dataset to {args.output_dataset}")


if __name__ == "__main__":
    main()
