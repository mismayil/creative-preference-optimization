import argparse
from tqdm import tqdm
import pathlib
from collections import defaultdict
from statistics import mean, median
from dotenv import load_dotenv

load_dotenv()

from crpo.utils import (
    read_json,
    write_json,
    find_files,
    compute_usage,
    wandb_log_run,
    none_or_int,
    none_or_str,
    prepare_metrics_for_wandb,
    detect_outliers_iqr,
)
from crpo.inference_lm import load_model
from crpo.metrics import (
    compute_inverse_homogenization,
    compute_novelty,
    compute_theme_uniqueness,
    compute_dsi,
    compute_n_gram_diversity,
    DEF_EMB_MODEL,
    DEF_EMB_TYPE,
    DEF_EMB_STRATEGY,
    DEF_DIST_FN,
    DEF_SPACY_LANG,
    DEF_PREPROCESSING_ARGS,
    get_words,
    get_sentences,
    compute_dependency_complexity,
    compute_constituency_complexity,
    compute_flesch_readability_scores,
    compute_pos_complexity,
    compute_perplexity,
)


def remove_outliers(
    data,
    one_word_only=False,
    preprocessing_args=DEF_PREPROCESSING_ARGS,
    iqr_threshold=1.5,
):
    data_by_task = defaultdict(list)

    for result in data:
        data_by_task[result["task"]].append(result)

    new_data_ids = []

    for task, task_data in data_by_task.items():
        lengths_in_words = []
        lengths_in_concepts = []

        for result in task_data:
            all_words = get_words(
                result["output"],
                lower=False,
                remove_punct=True,
                remove_stopwords=False,
                lemmatize=False,
                unique=False,
                dominant_k=None,
            )
            concepts = get_words(result["output"], **preprocessing_args)
            lengths_in_words.append(len(all_words))
            lengths_in_concepts.append(len(concepts))

        # remove outliers
        if not one_word_only:
            lengths_in_words_outliers = detect_outliers_iqr(
                lengths_in_words, multiplier=iqr_threshold
            )
        else:
            lengths_in_words_outliers = []

        for i, result in enumerate(task_data):
            if i not in lengths_in_words_outliers and lengths_in_concepts[i] > 1:
                new_data_ids.append(result["result_id"])

    new_data = [result for result in data if result["result_id"] in new_data_ids]
    outlier_data = [
        result for result in data if result["result_id"] not in new_data_ids
    ]

    return new_data, outlier_data


def compute_metrics(
    results,
    config=None,
    reference_model=None,
    reference_tokenizer=None,
):
    metrics = {}

    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    cost = {"input": 0, "output": 0, "total": 0}

    preprocessing_args = {
        "lower": config["lower"],
        "remove_punct": config["remove_punct"],
        "remove_stopwords": config["remove_stopwords"],
        "lemmatize": config["lemmatize"],
        "dominant_k": config["dominant_k"],
        "unique": config["unique"],
        "no_spacy": config["no_spacy"],
    }

    metrics_by_task = defaultdict(dict)

    data = [result for result in results["data"] if "output" in result]

    if config["remove_outliers"] or config.get("remove_1word_outliers_only"):
        print("Removing outliers...")
        data, outlier_data = remove_outliers(
            data,
            one_word_only=config.get("remove_1word_outliers_only"),
            preprocessing_args=preprocessing_args,
            iqr_threshold=config.get("iqr_threshold", 1.5),
        )

    data_by_task = defaultdict(list)

    for result in data:
        data_by_task[result["task"]].append(result)

    def _compute_global_metrics(result_data):
        if len(result_data) > 1:
            texts = [res["output"] for res in result_data]
            inv_homogen = compute_inverse_homogenization(
                texts,
                config["emb_model"],
                config["emb_type"],
                config["emb_strategy"],
                config["distance_fn"],
                preprocessing_args,
            )
            novelty = compute_novelty(
                texts,
                config["emb_model"],
                config["emb_type"],
                config["distance_fn"],
                preprocessing_args,
            )
            theme_uniqueness, theme_clusters = compute_theme_uniqueness(
                texts,
                config["emb_model"],
                config["emb_type"],
                config["emb_strategy"],
                config["cluster_linkage"],
                config["cluster_dist_threshold"],
                preprocessing_args,
            )
            if not config["no_spacy"]:
                corpus_n_gram_diversity, corpus_n_gram_frequency = (
                    compute_n_gram_diversity("".join(texts), config["max_n_gram"])
                )
            else:
                corpus_n_gram_diversity = None
                corpus_n_gram_frequency = None
            return {
                "inv_homogen": inv_homogen,
                "novelty": novelty,
                "theme_uniqueness": theme_uniqueness,
                "theme_clusters": theme_clusters,
                "corpus_n_gram_diversity": corpus_n_gram_diversity,
                "corpus_n_gram_frequency": corpus_n_gram_frequency,
            }

    for task, task_data in tqdm(
        data_by_task.items(), desc="Computing corpus metrics by task"
    ):
        print(f"Computing corpus metrics for task: {task}")
        if len(task_data) > 1:
            task_metrics = _compute_global_metrics(task_data)
            metrics_by_task[task]["corpus_n_gram_diversity"] = task_metrics[
                "corpus_n_gram_diversity"
            ]
            metrics_by_task[task]["num_themes"] = len(
                set(task_metrics["theme_clusters"])
            )

            for (
                task_result,
                inv_homogen_score,
                novelty_score,
                theme_uniqueness_score,
                theme_cluster,
            ) in zip(
                task_data,
                task_metrics["inv_homogen"],
                task_metrics["novelty"],
                task_metrics["theme_uniqueness"],
                task_metrics["theme_clusters"],
            ):
                if "metrics" not in task_result:
                    task_result["metrics"] = {}
                task_result["metrics"]["inv_homogen"] = inv_homogen_score
                task_result["metrics"]["novelty"] = novelty_score
                task_result["metrics"]["theme_uniqueness"] = theme_uniqueness_score
                task_result["metrics"]["theme_cluster"] = theme_cluster

    # compute metrics per prompt
    data_by_id = defaultdict(list)
    for result in data:
        data_by_id[result["id"]].append(result)

    for _, prompt_data in tqdm(
        data_by_id.items(), desc="Computing corpus metrics by sample"
    ):
        if len(prompt_data) > 1:
            sample_metrics = _compute_global_metrics(prompt_data)
            for (
                task_result,
                inv_homogen_score,
                novelty_score,
                theme_uniqueness_score,
                theme_cluster,
            ) in zip(
                prompt_data,
                sample_metrics["inv_homogen"],
                sample_metrics["novelty"],
                sample_metrics["theme_uniqueness"],
                sample_metrics["theme_clusters"],
            ):
                if "metrics" not in task_result:
                    task_result["metrics"] = {}
                task_result["metrics"]["per_prompt_inv_homogen"] = inv_homogen_score
                task_result["metrics"]["per_prompt_novelty"] = novelty_score
                task_result["metrics"][
                    "per_prompt_theme_uniqueness"
                ] = theme_uniqueness_score
                task_result["metrics"]["per_prompt_theme_cluster"] = theme_cluster

    for result in tqdm(data, desc="Computing local metrics"):
        if "metrics" not in result:
            result["metrics"] = {}

        all_words = get_words(
            result["output"],
            preprocessing_args["no_spacy"],
            lower=False,
            remove_punct=True,
            remove_stopwords=False,
            lemmatize=False,
            unique=False,
            dominant_k=None,
        )
        unique_words = list(set([word.lower() for word in all_words if all_words]))
        concepts = get_words(
            result["output"],
            preprocessing_args["no_spacy"],
            lower=True,
            remove_punct=True,
            remove_stopwords=True,
            lemmatize=True,
            unique=True,
            dominant_k=None,
        )
        if not preprocessing_args["no_spacy"]:
            sentences = get_sentences(result["output"])
            sentence_words = [
                get_words(
                    sentence,
                    preprocessing_args["no_spacy"],
                    lower=False,
                    remove_punct=True,
                    remove_stopwords=False,
                    lemmatize=False,
                    unique=False,
                    dominant_k=None,
                )
                for sentence in sentences
            ]
            sentence_unique_words = [
                list(set([word.lower() for word in words])) for words in sentence_words
            ]

        # basic metrics
        result["metrics"]["length_in_chars"] = len(result["output"])
        result["metrics"]["length_in_words"] = len(all_words)
        result["metrics"]["length_in_unique_words"] = len(unique_words)
        result["metrics"]["length_in_concepts"] = len(concepts)
        result["metrics"]["type_token_ratio"] = (
            len(unique_words) / len(all_words) if all_words else 0
        )
        result["metrics"]["avg_word_length_in_chars"] = mean(
            [len(word) for word in all_words]
        )

        if not preprocessing_args["no_spacy"]:
            result["metrics"]["length_in_sentences"] = len(sentences)
            result["metrics"]["avg_sentence_length_in_chars"] = mean(
                [len(sentence) for sentence in sentences]
            )
            result["metrics"]["avg_sentence_length_in_words"] = mean(
                [len(words) for words in sentence_words]
            )
            result["metrics"]["avg_sentence_length_in_unique_words"] = mean(
                [len(words) for words in sentence_unique_words]
            )
            result["metrics"]["length_in_first_person_singular"] = len(
                [
                    word
                    for word in all_words
                    if word.lower() in ["i", "me", "my", "mine", "myself"]
                ]
            )
            result["metrics"]["length_in_first_person_plural"] = len(
                [
                    word
                    for word in all_words
                    if word.lower() in ["we", "us", "our", "ours", "ourselves"]
                ]
            )
            result["metrics"]["length_in_second_person"] = len(
                [
                    word
                    for word in all_words
                    if word.lower()
                    in ["you", "your", "yours", "yourself", "yourselves"]
                ]
            )
            result["metrics"]["length_in_third_person_singular"] = len(
                [
                    word
                    for word in all_words
                    if word.lower()
                    in [
                        "he",
                        "him",
                        "his",
                        "himself",
                        "she",
                        "her",
                        "hers",
                        "herself",
                        "it",
                        "its",
                        "itself",
                    ]
                ]
            )
            result["metrics"]["length_in_third_person_plural"] = len(
                [
                    word
                    for word in all_words
                    if word.lower() in ["they", "them", "their", "theirs", "themselves"]
                ]
            )
            result["metrics"]["length_in_first_person"] = (
                result["metrics"]["length_in_first_person_singular"]
                + result["metrics"]["length_in_first_person_plural"]
            )
            result["metrics"]["length_in_third_person"] = (
                result["metrics"]["length_in_third_person_singular"]
                + result["metrics"]["length_in_third_person_plural"]
            )

        # complex metrics
        result["metrics"]["dsi"] = compute_dsi(
            result["output"],
            config["emb_model"],
            config["emb_type"],
            config["distance_fn"],
            preprocessing_args,
        )
        result["metrics"]["n_gram_diversity"], _ = compute_n_gram_diversity(
            result["output"],
            config["max_n_gram"],
            no_spacy=preprocessing_args["no_spacy"],
        )
        if not config["no_spacy"]:
            dependency_paths, dependency_num_clauses = compute_dependency_complexity(
                result["output"]
            )
            result["metrics"]["avg_dep_num_clauses"] = mean(dependency_num_clauses)
            result["metrics"]["max_dep_num_clauses"] = max(dependency_num_clauses)
            result["metrics"]["avg_dep_path_length"] = mean(
                [
                    mean([len(path) for path, freq in path_counter.items()])
                    for path_counter in dependency_paths
                ]
            )
            result["metrics"]["max_dep_path_length"] = max(
                [
                    max([len(path) for path, freq in path_counter.items()])
                    for path_counter in dependency_paths
                ]
            )

            constituency_complexity = compute_constituency_complexity(result["output"])
            result["metrics"]["avg_constituency_tree_depth"] = mean(
                constituency_complexity
            )
            result["metrics"]["max_constituency_tree_depth"] = max(
                constituency_complexity
            )

            flesch_ease, flesch_kincaid = compute_flesch_readability_scores(
                result["output"], no_spacy=preprocessing_args["no_spacy"]
            )
            result["metrics"]["readability_flesch_ease"] = flesch_ease
            result["metrics"]["readability_flesch_kincaid"] = flesch_kincaid

            pos_complexity = compute_pos_complexity(result["output"])
            for pos, pos_comps in pos_complexity.items():
                result["metrics"][f"avg_pos_{pos.lower()}_ratio"] = (
                    mean(pos_comps) if pos_comps else 0
                )

        if config["report_usage"]:
            sample_usage, sample_cost = compute_usage(
                result, results["metadata"]["model"]
            )

            if sample_usage:
                usage["input_tokens"] += sample_usage["input_tokens"]
                usage["output_tokens"] += sample_usage["output_tokens"]
                usage["total_tokens"] += (
                    sample_usage["input_tokens"] + sample_usage["output_tokens"]
                )

            if sample_cost:
                cost["input"] += sample_cost["input"]
                cost["output"] += sample_cost["output"]
                cost["total"] += sample_cost["total"]

    if reference_model:
        is_instruct_model = (
            "instruct" in config["reference_model"].lower()
            or "-it" in config["reference_model"].lower()
        )
        if is_instruct_model:
            perplexity_data = [
                [
                    {"role": "user", "content": result["user_prompt"][0] if isinstance(result["user_prompt"], list) else result["user_prompt"]},
                    {"role": "assistant", "content": result["output"]},
                ]
                for result in data
            ]
        else:
            perplexity_data = [
                f'{result["user_prompt"][0]}\n{result["output"]}' if isinstance(result["user_prompt"], list) else f'{result["user_prompt"]}\n{result["output"]}' for result in data
            ]
        perplexities = compute_perplexity(
            perplexity_data,
            reference_model,
            reference_tokenizer,
            batch_size=config["batch_size"],
        )
        for result, perplexity in zip(data, perplexities):
            result["metrics"]["perplexity"] = perplexity
        metrics["avg_perplexity"] = mean(perplexities)

    def _aggregate_metrics(metric_data, no_spacy=False):
        # basic metrics
        agg_metrics = {}
        agg_metrics["avg_length_in_chars"] = mean(
            [result["metrics"]["length_in_chars"] for result in metric_data]
        )
        agg_metrics["median_length_in_chars"] = median(
            [result["metrics"]["length_in_chars"] for result in metric_data]
        )
        agg_metrics["avg_length_in_words"] = mean(
            [result["metrics"]["length_in_words"] for result in metric_data]
        )
        agg_metrics["median_length_in_words"] = median(
            [result["metrics"]["length_in_words"] for result in metric_data]
        )
        agg_metrics["avg_length_in_unique_words"] = mean(
            [result["metrics"]["length_in_unique_words"] for result in metric_data]
        )
        agg_metrics["median_length_in_unique_words"] = median(
            [result["metrics"]["length_in_unique_words"] for result in metric_data]
        )
        agg_metrics["avg_length_in_concepts"] = mean(
            [result["metrics"]["length_in_concepts"] for result in metric_data]
        )
        agg_metrics["median_length_in_concepts"] = median(
            [result["metrics"]["length_in_concepts"] for result in metric_data]
        )

        agg_metrics["avg_word_length_in_chars"] = mean(
            [result["metrics"]["avg_word_length_in_chars"] for result in metric_data]
        )

        if not no_spacy:
            agg_metrics["avg_length_in_sentences"] = mean(
                [result["metrics"]["length_in_sentences"] for result in metric_data]
            )
            agg_metrics["median_length_in_sentences"] = median(
                [result["metrics"]["length_in_sentences"] for result in metric_data]
            )
            agg_metrics["avg_sentence_length_in_chars"] = mean(
                [
                    result["metrics"]["avg_sentence_length_in_chars"]
                    for result in metric_data
                ]
            )
            agg_metrics["avg_sentence_length_in_words"] = mean(
                [
                    result["metrics"]["avg_sentence_length_in_words"]
                    for result in metric_data
                ]
            )
            agg_metrics["avg_sentence_length_in_unique_words"] = mean(
                [
                    result["metrics"]["avg_sentence_length_in_unique_words"]
                    for result in metric_data
                ]
            )

        # complex metrics
        inv_homogen = [
            result["metrics"]["inv_homogen"]
            for result in metric_data
            if "inv_homogen" in result["metrics"]
        ]
        if inv_homogen:
            agg_metrics["avg_inv_homogen"] = mean(inv_homogen)
            agg_metrics["median_inv_homogen"] = median(inv_homogen)

        novelty = [
            result["metrics"]["novelty"]
            for result in metric_data
            if "novelty" in result["metrics"]
        ]
        if novelty:
            agg_metrics["avg_novelty"] = mean(novelty)
            agg_metrics["median_novelty"] = median(novelty)

        theme_uniqueness = [
            result["metrics"]["theme_uniqueness"]
            for result in metric_data
            if "theme_uniqueness" in result["metrics"]
        ]
        if theme_uniqueness:
            agg_metrics["avg_theme_uniqueness"] = mean(theme_uniqueness)
            agg_metrics["median_theme_uniqueness"] = median(theme_uniqueness)

        # per prompt metrics
        per_prompt_inv_homogen = [
            result["metrics"]["per_prompt_inv_homogen"]
            for result in metric_data
            if "per_prompt_inv_homogen" in result["metrics"]
        ]
        if per_prompt_inv_homogen:
            agg_metrics["avg_per_prompt_inv_homogen"] = mean(per_prompt_inv_homogen)
            agg_metrics["median_per_prompt_inv_homogen"] = median(
                per_prompt_inv_homogen
            )

        per_prompt_novelty = [
            result["metrics"]["per_prompt_novelty"]
            for result in metric_data
            if "per_prompt_novelty" in result["metrics"]
        ]
        if per_prompt_novelty:
            agg_metrics["avg_per_prompt_novelty"] = mean(per_prompt_novelty)
            agg_metrics["median_per_prompt_novelty"] = median(per_prompt_novelty)

            metric_data_by_id = defaultdict(list)
            for result in metric_data:
                if "per_prompt_novelty" in result["metrics"]:
                    metric_data_by_id[result["id"]].append(
                        result["metrics"]["per_prompt_novelty"]
                    )

            # sort items
            for key in metric_data_by_id:
                metric_data_by_id[key].sort(reverse=True)

            agg_metrics["top1_per_prompt_novelty"] = mean(
                [novelties[0] for key, novelties in metric_data_by_id.items()]
            )
            agg_metrics["top3_per_prompt_novelty"] = mean(
                [mean(novelties[:3]) for key, novelties in metric_data_by_id.items()]
            )
            agg_metrics["top5_per_prompt_novelty"] = mean(
                [mean(novelties[:5]) for key, novelties in metric_data_by_id.items()]
            )

        per_prompt_theme_uniqueness = [
            result["metrics"]["per_prompt_theme_uniqueness"]
            for result in metric_data
            if "per_prompt_theme_uniqueness" in result["metrics"]
        ]
        if per_prompt_theme_uniqueness:
            agg_metrics["avg_per_prompt_theme_uniqueness"] = mean(
                per_prompt_theme_uniqueness
            )
            agg_metrics["median_per_prompt_theme_uniqueness"] = median(
                per_prompt_theme_uniqueness
            )

        agg_metrics["avg_dsi"] = mean(
            [result["metrics"]["dsi"] for result in metric_data]
        )
        agg_metrics["median_dsi"] = median(
            [result["metrics"]["dsi"] for result in metric_data]
        )

        agg_metrics["avg_n_gram_diversity"] = []
        for n_gram_len in range(1, config["max_n_gram"] + 1):
            n_gram_diversity = [
                result["metrics"]["n_gram_diversity"][n_gram_len - 1]
                for result in metric_data
                if len(result["metrics"]["n_gram_diversity"]) >= n_gram_len
            ]
            if n_gram_diversity:
                agg_metrics["avg_n_gram_diversity"].append(mean(n_gram_diversity))

        if not no_spacy:
            agg_metrics["avg_dep_num_clauses"] = mean(
                [result["metrics"]["avg_dep_num_clauses"] for result in metric_data]
            )
            agg_metrics["avg_max_dep_num_clauses"] = mean(
                [result["metrics"]["max_dep_num_clauses"] for result in metric_data]
            )
            agg_metrics["avg_dep_path_length"] = mean(
                [result["metrics"]["avg_dep_path_length"] for result in metric_data]
            )
            agg_metrics["avg_max_dep_path_length"] = mean(
                [result["metrics"]["max_dep_path_length"] for result in metric_data]
            )
        perplexity = [
            result["metrics"]["perplexity"]
            for result in metric_data
            if "perplexity" in result["metrics"]
        ]
        if perplexity:
            agg_metrics["avg_perplexity"] = mean(perplexity)
            agg_metrics["median_perplexity"] = median(perplexity)

        return agg_metrics

    metrics.update(_aggregate_metrics(data, preprocessing_args["no_spacy"]))

    for task, task_data in data_by_task.items():
        metrics_by_task[task].update(
            _aggregate_metrics(task_data, preprocessing_args["no_spacy"])
        )

    metrics["usage"] = usage
    metrics["cost"] = cost
    metrics["num_total_samples"] = len(results["data"])
    metrics["num_samples"] = len(data)
    metrics["num_tasks"] = len(data_by_task)
    metrics["num_samples_by_task"] = {
        task: len(task_data) for task, task_data in data_by_task.items()
    }
    metrics["by_task"] = metrics_by_task

    return metrics


def report_metrics(
    results_files,
    config=None,
    reference_model=None,
    reference_tokenizer=None,
):
    for results_file in results_files:
        results = read_json(results_file)

        try:
            if "data" in results:
                print(f"Reporting metrics for: {results_file}")
                metrics = compute_metrics(
                    results,
                    config=config,
                    reference_model=reference_model,
                    reference_tokenizer=reference_tokenizer,
                )
                results["metadata"]["metrics_config"] = config
                results["metrics"] = metrics

                output_file = results_file

                if config["output_dir"]:
                    # If output_dir is specified, save the results in that directory
                    output_dir = pathlib.Path(config["output_dir"])
                    output_dir.mkdir(parents=True, exist_ok=True)
                    suffix = config.get("suffix", "")
                    dataset_name = f"{pathlib.Path(results_file).stem}{suffix}"
                    output_file = output_dir / f"{dataset_name}.json"
                    results["metadata"]["dataset"] = dataset_name
                    results["metadata"]["eval_wandb_run_id"] = None

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

                write_json(results, output_file, ensure_ascii=False)
        except Exception as e:
            print(results_file)
            raise e


def main():
    parser = argparse.ArgumentParser(
        prog="evaluate_lm", description="Evaluate language model outputs"
    )
    parser.add_argument(
        "-r",
        "--results-path",
        type=str,
        help="Path to evaluation results file in json or directory",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="If not specified, the results will be saved in the same directory as the input file.",
    )
    parser.add_argument(
        "-s",
        "--suffix",
        type=str,
        default=None,
        help="Suffix to add to the output file name.",
    )
    parser.add_argument(
        "-c", "--config", type=str, help="Path to config file", default=None
    )
    parser.add_argument(
        "-u", "--report-usage", type=bool, help="Report usage metrics", default=True
    )
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    parser.add_argument("-w", "--wandb", type=bool, default=True)
    parser.add_argument(
        "-p", "--wandb-project", type=str, default="project-cpo-lm-evals"
    )
    parser.add_argument(
        "-sl",
        "--spacy-lang",
        type=str,
        help="Spacy language model. Ignored when using --no-spacy",
        default=DEF_SPACY_LANG,
    )
    parser.add_argument(
        "-ng", "--max-n-gram", type=int, help="Maximum n-gram to consider", default=5
    )
    parser.add_argument(
        "-mf",
        "--max-frequency",
        type=int,
        help="Maximum n-gram frequency to consider",
        default=10,
    )
    parser.add_argument(
        "-rfm",
        "--reference-model",
        type=none_or_str,
        help="Reference model",
        default="google/gemma-2-27b-it",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--remove-outliers",
        type=bool,
        default=False,
        help="Remove outliers from the data (including 1-word outliers)",
    )
    parser.add_argument(
        "--remove-1word-outliers-only",
        type=bool,
        default=False,
        help="Remove only 1-word outliers from the data",
    )
    parser.add_argument(
        "--iqr-threshold",
        type=float,
        default=1.5,
        help="IQR threshold for outlier detection",
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

    cl_group = parser.add_argument_group("Clustering arguments")
    cl_group.add_argument(
        "-cl", "--cluster-linkage", type=str, help="Cluster linkage", default="ward"
    )
    cl_group.add_argument(
        "-cdt",
        "--cluster-dist-threshold",
        type=float,
        help="Cluster distance threshold",
        default=0.5,
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

    if not files_to_process:
        print(f"No files found in {config['results_path']}")
        return

    reference_model, reference_tokenizer = (
        load_model(config["reference_model"], device=config["device"])
        if config.get("reference_model")
        else (None, None)
    )

    report_metrics(
        files_to_process,
        config=config,
        reference_model=reference_model,
        reference_tokenizer=reference_tokenizer,
    )


if __name__ == "__main__":
    main()
