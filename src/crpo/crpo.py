import torch
import warnings
from typing import Tuple

STRATEGIES = ["add", "mult"]


def compute_creativity_scores(
    chosen_diversity_scores: torch.FloatTensor = None,
    chosen_novelty_scores: torch.FloatTensor = None,
    chosen_surprise_scores: torch.FloatTensor = None,
    chosen_quality_scores: torch.FloatTensor = None,
    rejected_diversity_scores: torch.FloatTensor = None,
    rejected_novelty_scores: torch.FloatTensor = None,
    rejected_surprise_scores: torch.FloatTensor = None,
    rejected_quality_scores: torch.FloatTensor = None,
    lambda_diversity: float = 0.0,
    lambda_novelty: float = 0.0,
    lambda_surprise: float = 0.0,
    lambda_quality: float = 0.0,
    expected_shape: Tuple[int, int] = None,
    strategy="add",
):
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Invalid strategy '{strategy}'. Expected one of {STRATEGIES}."
        )

    creativity_scores = 0

    if strategy == "mult":
        creativity_scores = 1.0

    if lambda_diversity > 0.0:
        if chosen_diversity_scores is None:
            warnings.warn(
                "Diversity scores are not provided. Setting lambda_diversity to 0.0."
            )
            lambda_diversity = 0.0
        else:
            if expected_shape:
                assert (
                    chosen_diversity_scores.shape == expected_shape
                ), f"Diversity scores shape mismatch: {chosen_diversity_scores.shape} != {expected_shape}"
            rejected_diversity_scores = (
                rejected_diversity_scores
                if rejected_diversity_scores is not None
                else 0.0
            )
            diversity_scores_diff = chosen_diversity_scores - rejected_diversity_scores
            diversity_scores_mask = diversity_scores_diff >= 0
            diversity_scores_diff = diversity_scores_diff * diversity_scores_mask
            diversity_scores = lambda_diversity * diversity_scores_diff

            if strategy == "add":
                creativity_scores += diversity_scores
            elif strategy == "mult":
                creativity_scores *= diversity_scores

    if lambda_novelty > 0.0:
        if chosen_novelty_scores is None:
            warnings.warn(
                "Novelty scores are not provided. Setting lambda_novelty to 0.0."
            )
            lambda_novelty = 0.0
        else:
            if expected_shape:
                assert (
                    chosen_novelty_scores.shape == expected_shape
                ), f"Novelty scores shape mismatch: {chosen_novelty_scores.shape} != {expected_shape}"
            rejected_novelty_scores = (
                rejected_novelty_scores if rejected_novelty_scores is not None else 0.0
            )
            novelty_scores_diff = chosen_novelty_scores - rejected_novelty_scores
            novelty_scores_mask = novelty_scores_diff >= 0
            novelty_scores_diff = novelty_scores_diff * novelty_scores_mask
            novelty_scores = lambda_novelty * novelty_scores_diff

            if strategy == "add":
                creativity_scores += novelty_scores
            elif strategy == "mult":
                creativity_scores *= novelty_scores

    if lambda_surprise > 0.0:
        if chosen_surprise_scores is None:
            warnings.warn(
                "Surprise scores are not provided. Setting lambda_surprise to 0.0."
            )
            lambda_surprise = 0.0
        else:
            if expected_shape:
                assert (
                    chosen_surprise_scores.shape == expected_shape
                ), f"Surprise scores shape mismatch: {chosen_surprise_scores.shape} != {expected_shape}"
            rejected_surprise_scores = (
                rejected_surprise_scores
                if rejected_surprise_scores is not None
                else 0.0
            )
            surprise_scores_diff = chosen_surprise_scores - rejected_surprise_scores
            surprise_scores_mask = surprise_scores_diff >= 0
            surprise_scores_diff = surprise_scores_diff * surprise_scores_mask
            surprise_scores = lambda_surprise * surprise_scores_diff

            if strategy == "add":
                creativity_scores += surprise_scores
            elif strategy == "mult":
                creativity_scores *= surprise_scores

    if lambda_quality > 0.0:
        if chosen_quality_scores is None:
            warnings.warn(
                "Quality scores are not provided. Setting lambda_quality to 0.0."
            )
            lambda_quality = 0.0
        else:
            if expected_shape:
                assert (
                    chosen_quality_scores.shape == expected_shape
                ), f"Quality scores shape mismatch: {chosen_quality_scores.shape} != {expected_shape}"
            rejected_quality_scores = (
                rejected_quality_scores if rejected_quality_scores is not None else 0.0
            )
            quality_scores_diff = chosen_quality_scores - rejected_quality_scores
            quality_scores_mask = quality_scores_diff >= 0
            quality_scores_diff = quality_scores_diff * quality_scores_mask
            quality_scores = lambda_quality * quality_scores_diff

            if strategy == "add":
                creativity_scores += quality_scores
            elif strategy == "mult":
                creativity_scores *= quality_scores

    apply_creativity = (
        lambda_diversity > 0.0
        or lambda_novelty > 0.0
        or lambda_surprise > 0.0
        or lambda_quality > 0.0
    )
    creativity_scores = apply_creativity * creativity_scores + (1 - apply_creativity)

    return creativity_scores
