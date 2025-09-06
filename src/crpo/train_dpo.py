# adapted from https://github.com/huggingface/trl/blob/main/trl/scripts/dpo.py

import argparse
from dotenv import load_dotenv
from dataclasses import dataclass, field
import warnings
from typing import Union, Literal, Any
import scipy.stats as stats

load_dotenv()

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.data.data_collator import DataCollatorMixin

from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    FDivergenceType,
    FDivergenceConstants,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE, cap_exp, pad, empty_cache

from utils import find_latest_checkpoint
from cpo import compute_creativity_scores


@dataclass
class CDPOConfig(DPOConfig):
    lambda_diversity: float = field(
        default=0.0,
        metadata={"help": "Coefficient for the diversity injection."},
    )
    lambda_novelty: float = field(
        default=0.0,
        metadata={"help": "Coefficient for the novelty injection."},
    )
    lambda_surprise: float = field(
        default=0.0,
        metadata={"help": "Coefficient for the surprise injection."},
    )
    lambda_quality: float = field(
        default=0.0,
        metadata={"help": "Coefficient for the quality injection."},
    )
    cpo_strategy: str = field(
        default="add",
        metadata={
            "help": "Strategy for combining creativity scores. Options: 'add' or 'mult'."
        },
    )


@dataclass
class DataCollatorForPreferenceWithCreativity(DataCollatorMixin):
    """
    Data collator used for preference data with creativity scores. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        pad_token_id (`int`):
            Token ID to use for padding.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of Tensor to return. Only `"pt"` is currently supported.

    Examples:
    ```python
    >>> from trl import DataCollatorForPreference
    >>> collator = DataCollatorForPreference(pad_token_id=0)
    >>> examples = [
    ...     {"prompt_input_ids": [1, 2, 3], "chosen_input_ids": [4, 5], "rejected_input_ids": [6], "diversity_chosen": 0.5, "novelty_chosen": 0.7, "surprise_chosen": 0.9},
    ...     {"prompt_input_ids": [7, 8], "chosen_input_ids": [9, 10], "rejected_input_ids": [11, 12, 13], "diversity_chosen": 0.6, "novelty_chosen": 0.8, "surprise_chosen": 0.95},
    ... ]
    >>> collator(examples)
    {'prompt_input_ids': tensor([[1, 2, 3],
                                 [0, 7, 8]]),
     'prompt_attention_mask': tensor([[1, 1, 1],
                                      [0, 1, 1]]),
     'chosen_input_ids': tensor([[ 4,  5],
                                 [ 9, 10]]),
     'chosen_attention_mask': tensor([[1, 1],
                                      [1, 1]]),
     'rejected_input_ids': tensor([[ 6,  0,  0],
                                   [11, 12, 13]]),
     'rejected_attention_mask': tensor([[1, 0, 0],
                                        [1, 1, 1]]),
    'diversity_chosen': tensor([0.5000, 0.6000]),
    'novelty_chosen': tensor([0.7000, 0.8000]),
    'surprise_chosen': tensor([0.9000, 0.9500]),
    }
    ```
    """

    pad_token_id: int
    return_tensors: str = "pt"

    def torch_call(
        self, examples: list[Union[list[int], Any, dict[str, Any]]]
    ) -> dict[str, Any]:
        # Convert to tensor
        prompt_input_ids = [
            torch.tensor(example["prompt_input_ids"]) for example in examples
        ]
        prompt_attention_mask = [
            torch.ones_like(input_ids) for input_ids in prompt_input_ids
        ]
        chosen_input_ids = [
            torch.tensor(example["chosen_input_ids"]) for example in examples
        ]
        chosen_attention_mask = [
            torch.ones_like(input_ids) for input_ids in chosen_input_ids
        ]
        rejected_input_ids = [
            torch.tensor(example["rejected_input_ids"]) for example in examples
        ]
        rejected_attention_mask = [
            torch.ones_like(input_ids) for input_ids in rejected_input_ids
        ]
        if "pixel_values" in examples[0]:
            pixel_values = [
                torch.tensor(example["pixel_values"]) for example in examples
            ]
        if "pixel_attention_mask" in examples[0]:
            pixel_attention_mask = [
                torch.tensor(example["pixel_attention_mask"]) for example in examples
            ]
        if "ref_chosen_logps" in examples[0] and "ref_rejected_logps" in examples[0]:
            ref_chosen_logps = torch.tensor(
                [example["ref_chosen_logps"] for example in examples]
            )
            ref_rejected_logps = torch.tensor(
                [example["ref_rejected_logps"] for example in examples]
            )

        # Pad
        output = {}
        output["prompt_input_ids"] = pad(
            prompt_input_ids, padding_value=self.pad_token_id, padding_side="left"
        )
        output["prompt_attention_mask"] = pad(
            prompt_attention_mask, padding_value=0, padding_side="left"
        )
        output["chosen_input_ids"] = pad(
            chosen_input_ids, padding_value=self.pad_token_id
        )
        output["chosen_attention_mask"] = pad(chosen_attention_mask, padding_value=0)
        output["rejected_input_ids"] = pad(
            rejected_input_ids, padding_value=self.pad_token_id
        )
        output["rejected_attention_mask"] = pad(
            rejected_attention_mask, padding_value=0
        )
        if "pixel_values" in examples[0]:
            output["pixel_values"] = pad(pixel_values, padding_value=0.0)
        if "pixel_attention_mask" in examples[0]:
            output["pixel_attention_mask"] = pad(pixel_attention_mask, padding_value=0)
        if "image_sizes" in examples[0]:
            output["image_sizes"] = torch.tensor(
                [example["image_sizes"] for example in examples]
            )
        if "ref_chosen_logps" in examples[0] and "ref_rejected_logps" in examples[0]:
            output["ref_chosen_logps"] = ref_chosen_logps
            output["ref_rejected_logps"] = ref_rejected_logps

        if "diversity_chosen" in examples[0]:
            output["diversity_chosen"] = torch.tensor(
                [example["diversity_chosen"] for example in examples]
            )

        if "novelty_chosen" in examples[0]:
            output["novelty_chosen"] = torch.tensor(
                [example["novelty_chosen"] for example in examples]
            )

        if "surprise_chosen" in examples[0]:
            output["surprise_chosen"] = torch.tensor(
                [example["surprise_chosen"] for example in examples]
            )
        if "quality_chosen" in examples[0]:
            output["quality_chosen"] = torch.tensor(
                [example["quality_chosen"] for example in examples]
            )
        if "diversity_rejected" in examples[0]:
            output["diversity_rejected"] = torch.tensor(
                [example["diversity_rejected"] for example in examples]
            )
        if "novelty_rejected" in examples[0]:
            output["novelty_rejected"] = torch.tensor(
                [example["novelty_rejected"] for example in examples]
            )
        if "surprise_rejected" in examples[0]:
            output["surprise_rejected"] = torch.tensor(
                [example["surprise_rejected"] for example in examples]
            )
        if "quality_rejected" in examples[0]:
            output["quality_rejected"] = torch.tensor(
                [example["quality_rejected"] for example in examples]
            )

        return output

class CDPOTrainer(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def tokenize_row(
        features,
        processing_class,
        max_prompt_length,
        max_completion_length,
        add_special_tokens,
    ):
        """
        Tokenize a row of the dataset.

        Args:
            features (`dict[str, str]`):
                Row of the dataset, should contain the keys `"prompt"`, `"chosen"`, and `"rejected"`.
            processing_class (`PreTrainedTokenizerBase`):
                Processing class used to process the data.
            max_prompt_length (`int` or `None`):
                Maximum length of the prompt sequence. If `None`, the prompt sequence is not truncated.
            max_completion_length (`int` or `None`):
                Maximum length of the completion sequences. If `None`, the completion sequences are not truncated.
            add_special_tokens (`bool`):
                Whether to add special tokens to the sequences. Typically used for encoder-decoder models. If `True`,
                the prompt sequence will have a bos token prepended and an eos token appended. In any case, the
                completion sequences will have an eos token appended.

        Returns:
            `dict[str, list[int]]`:
                Tokenized sequences with the keys `"prompt_input_ids"`, `"chosen_input_ids"`, and
                `"rejected_input_ids".

        Example:
        ```python
        >>> from transformers import GPT2Tokenizer
        >>> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        >>> features = {"prompt": "The sky is", "chosen": " blue", "rejected": " green"}
        >>> DPOTrainer.tokenize_row(
        ...     features, tokenizer, max_prompt_length=3, max_completion_length=3, add_special_tokens=False
        ... )
        {'prompt_input_ids': [464, 6766, 318], 'chosen_input_ids': [4171, 50256], 'rejected_input_ids': [4077, 50256]}
        ```
        """
        tokenizer = processing_class  # the processing class is a tokenizer
        prompt_input_ids = tokenizer(features["prompt"], add_special_tokens=False)[
            "input_ids"
        ]
        chosen_input_ids = tokenizer(features["chosen"], add_special_tokens=False)[
            "input_ids"
        ]
        rejected_input_ids = tokenizer(features["rejected"], add_special_tokens=False)[
            "input_ids"
        ]

        # Add special tokens (typically for encoder-decoder models)
        if add_special_tokens:
            if tokenizer.bos_token_id is not None:
                prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
            if tokenizer.eos_token_id is not None:
                prompt_input_ids = prompt_input_ids + [tokenizer.eos_token_id]
        chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
        rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

        # Truncate prompt and completion sequences
        if max_prompt_length is not None:
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        if max_completion_length is not None:
            chosen_input_ids = chosen_input_ids[:max_completion_length]
            rejected_input_ids = rejected_input_ids[:max_completion_length]

        tokenized_row = {
            "prompt_input_ids": prompt_input_ids,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
        }

        diversity_chosen = features.get("diversity_chosen")
        novelty_chosen = features.get("novelty_chosen")
        surprise_chosen = features.get("surprise_chosen")
        quality_chosen = features.get("quality_chosen")
        diversity_rejected = features.get("diversity_rejected")
        novelty_rejected = features.get("novelty_rejected")
        surprise_rejected = features.get("surprise_rejected")
        quality_rejected = features.get("quality_rejected")

        if diversity_chosen is not None:
            tokenized_row["diversity_chosen"] = diversity_chosen
        if novelty_chosen is not None:
            tokenized_row["novelty_chosen"] = novelty_chosen
        if surprise_chosen is not None:
            tokenized_row["surprise_chosen"] = surprise_chosen
        if quality_chosen is not None:
            tokenized_row["quality_chosen"] = quality_chosen
        if diversity_rejected is not None:
            tokenized_row["diversity_rejected"] = diversity_rejected
        if novelty_rejected is not None:
            tokenized_row["novelty_rejected"] = novelty_rejected
        if surprise_rejected is not None:
            tokenized_row["surprise_rejected"] = surprise_rejected
        if quality_rejected is not None:
            tokenized_row["quality_rejected"] = quality_rejected

        return tokenized_row

    def dpo_loss(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
        chosen_diversity_scores: torch.FloatTensor = None,
        chosen_novelty_scores: torch.FloatTensor = None,
        chosen_surprise_scores: torch.FloatTensor = None,
        chosen_quality_scores: torch.FloatTensor = None,
        rejected_diversity_scores: torch.FloatTensor = None,
        rejected_novelty_scores: torch.FloatTensor = None,
        rejected_surprise_scores: torch.FloatTensor = None,
        rejected_quality_scores: torch.FloatTensor = None,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            chosen_logps (`torch.FloatTensor`):
                Log probabilities of the model for the chosen responses. Shape: `(batch_size,)`.
            rejected_logps (`torch.FloatTensor`):
                Log probabilities of the model for the rejected responses. Shape: `(batch_size,)`.
            ref_chosen_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the chosen responses. Shape: `(batch_size,)`.
            ref_rejected_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the rejected responses. Shape: `(batch_size,)`.
            chosen_diversity_scores (`torch.FloatTensor`, *optional*):
                Diversity scores for the chosen responses. Shape: `(batch_size,)`.
            chosen_novelty_scores (`torch.FloatTensor`, *optional*):
                Novelty scores for the chosen responses. Shape: `(batch_size,)`.
            chosen_surprise_scores (`torch.FloatTensor`, *optional*):
                Surprise scores for the chosen responses. Shape: `(batch_size,)`.
            chosen_quality_scores (`torch.FloatTensor`, *optional*):
                Quality scores for the chosen responses. Shape: `(batch_size,)`.
            rejected_diversity_scores (`torch.FloatTensor`, *optional*):
                Diversity scores for the rejected responses. Shape: `(batch_size,)`.
            rejected_novelty_scores (`torch.FloatTensor`, *optional*):
                Novelty scores for the rejected responses. Shape: `(batch_size,)`.
            rejected_surprise_scores (`torch.FloatTensor`, *optional*):
                Surprise scores for the rejected responses. Shape: `(batch_size,)`.
            rejected_quality_scores (`torch.FloatTensor`, *optional*):
                Quality scores for the rejected responses. Shape: `(batch_size,)`.

        Returns:
            A tuple of three tensors: `(losses, chosen_rewards, rejected_rewards)`.
            The losses tensor contains the DPO loss for each example in the batch.
            The `chosen_rewards` and `rejected_rewards` tensors contain the rewards for the chosen and rejected
            responses, respectively.
        """
        device = self.accelerator.device

        # Get the log ratios for the chosen and rejected responses
        chosen_logratios = chosen_logps.to(device) - (
            not self.reference_free
        ) * ref_chosen_logps.to(device)
        rejected_logratios = rejected_logps.to(device) - (
            not self.reference_free
        ) * ref_rejected_logps.to(device)

        if self.f_divergence_type == FDivergenceType.ALPHA_DIVERGENCE.value:
            # The alpha-divergence formula: (1 - u^-alpha) / alpha
            # The divergence difference between the chosen and rejected sample is:
            #     (1 - u[w]^-alpha) / alpha - (1 - u[l]^-alpha) / alpha
            #        = (u[l]^-alpha - u[w]^-alpha) / alpha
            # where u[w] and u[l] are the policy/reference probability ratios
            # for the chosen and rejected samples, respectively.
            alpha_coef = FDivergenceConstants.ALPHA_DIVERGENCE_COEF_DEFAULT
            if (
                self.f_divergence_params
                and FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY
                in self.f_divergence_params
            ):
                alpha_coef = float(
                    self.f_divergence_params[
                        FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY
                    ]
                )
            logits = (
                cap_exp(rejected_logratios * -alpha_coef)
                - cap_exp(chosen_logratios * -alpha_coef)
            ) / alpha_coef
        else:
            logratios = chosen_logps - rejected_logps
            if self.reference_free:
                ref_logratios = torch.tensor(
                    [0], dtype=logratios.dtype, device=logratios.device
                )
            else:
                ref_logratios = ref_chosen_logps - ref_rejected_logps

            logratios = logratios.to(self.accelerator.device)
            ref_logratios = ref_logratios.to(self.accelerator.device)
            logits = logratios - ref_logratios

            if self.f_divergence_type == FDivergenceType.JS_DIVERGENCE.value:
                # The js-divergence formula: log(2 * u / (1 + u))
                # The divergence difference between the chosen and rejected sample is:
                #     log(2 * u[w] / (1 + u[w])) - log(2 * u[l] / (1 + u[l]))
                #       = log(u[w]) - log(u[l]) - (log(1 + u[w]) - log(1 + u[l]))
                # where u[w] and u[l] are the policy/reference probability ratios
                # for the chosen and rejected samples, respectively.
                logits -= F.softplus(chosen_logratios) - F.softplus(rejected_logratios)

        # compute creativity scores
        creativity_scores = compute_creativity_scores(
            chosen_diversity_scores=chosen_diversity_scores,
            chosen_novelty_scores=chosen_novelty_scores,
            chosen_surprise_scores=chosen_surprise_scores,
            chosen_quality_scores=chosen_quality_scores,
            rejected_diversity_scores=rejected_diversity_scores,
            rejected_novelty_scores=rejected_novelty_scores,
            rejected_surprise_scores=rejected_surprise_scores,
            rejected_quality_scores=rejected_quality_scores,
            lambda_diversity=self.args.lambda_diversity,
            lambda_novelty=self.args.lambda_novelty,
            lambda_surprise=self.args.lambda_surprise,
            lambda_quality=self.args.lambda_quality,
            expected_shape=logits.shape,
            strategy=self.args.cpo_strategy,
        )

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the
        # labels and calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            losses = (
                -creativity_scores
                * F.logsigmoid(self.beta * logits)
                * (1 - self.label_smoothing)
                - creativity_scores
                * F.logsigmoid(-self.beta * logits)
                * self.label_smoothing
            )

        elif self.loss_type == "robust":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                + F.logsigmoid(-self.beta * logits) * self.label_smoothing
            ) / (1 - 2 * self.label_smoothing)

        elif self.loss_type == "exo_pair":
            # eqn (16) of the EXO paper: https://huggingface.co/papers/2402.00856
            import math

            if self.label_smoothing == 0:
                self.label_smoothing = 1e-3
            losses = (self.beta * logits).sigmoid() * (
                F.logsigmoid(self.beta * logits) - math.log(1 - self.label_smoothing)
            ) + (-self.beta * logits).sigmoid() * (
                F.logsigmoid(-self.beta * logits) - math.log(self.label_smoothing)
            )

        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)

        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2

        elif self.loss_type == "bco_pair":
            chosen_logratios = chosen_logps - ref_chosen_logps
            rejected_logratios = rejected_logps - ref_rejected_logps
            chosen_rewards = self.beta * chosen_logratios
            rejected_rewards = self.beta * rejected_logratios
            rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
            self.running.update(rewards)
            delta = self.running.mean
            losses = -F.logsigmoid(
                (self.beta * chosen_logratios) - delta
            ) - F.logsigmoid(-(self.beta * rejected_logratios - delta))

        elif self.loss_type == "sppo_hard":
            # In the paper (https://huggingface.co/papers/2405.00675), SPPO employs a soft probability approach,
            # estimated using the PairRM score. The probability calculation is conducted outside of the trainer class.
            # The version described here is the hard probability version, where P in Equation (4.7) of Algorithm 1 is
            # set to 1 for the winner and 0 for the loser.
            a = chosen_logps - ref_chosen_logps
            b = rejected_logps - ref_rejected_logps
            losses = (a - 0.5 / self.beta) ** 2 + (b + 0.5 / self.beta) ** 2

        elif self.loss_type == "nca_pair":
            chosen_rewards = (chosen_logps - ref_chosen_logps) * self.beta
            rejected_rewards = (rejected_logps - ref_rejected_logps) * self.beta
            losses = (
                -F.logsigmoid(chosen_rewards)
                - 0.5 * F.logsigmoid(-chosen_rewards)
                - 0.5 * F.logsigmoid(-rejected_rewards)
            )

        elif self.loss_type == "aot_pair":
            chosen_logratios = chosen_logps - ref_chosen_logps
            rejected_logratios = rejected_logps - ref_rejected_logps
            chosen_logratios_sorted, _ = torch.sort(chosen_logratios, dim=0)
            rejected_logratios_sorted, _ = torch.sort(rejected_logratios, dim=0)
            delta = chosen_logratios_sorted - rejected_logratios_sorted
            losses = (
                -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * delta) * self.label_smoothing
            )

        elif self.loss_type == "aot":
            logratios = chosen_logps - rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            logratios_sorted, _ = torch.sort(logratios, dim=0)
            ref_logratios_sorted, _ = torch.sort(ref_logratios, dim=0)
            delta = logratios_sorted - ref_logratios_sorted
            losses = (
                -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * delta) * self.label_smoothing
            )

        elif self.loss_type == "apo_zero":
            # Eqn (7) of the APO paper (https://huggingface.co/papers/2408.06266)
            # Use this loss when you believe the chosen outputs are better than your model's default output
            losses_chosen = 1 - F.sigmoid(
                self.beta * chosen_logratios
            )  # Increase chosen likelihood
            losses_rejected = F.sigmoid(
                self.beta * rejected_logratios
            )  # Decrease rejected likelihood
            losses = losses_chosen + losses_rejected

        elif self.loss_type == "apo_down":
            # Eqn (8) of the APO paper (https://huggingface.co/papers/2408.06266)
            # Use this loss when you believe the chosen outputs are worse than your model's default output.
            # Decrease chosen likelihood and decrease rejected likelihood more
            losses_chosen = F.sigmoid(self.beta * chosen_logratios)
            losses_rejected = 1 - F.sigmoid(
                self.beta * (chosen_logratios - rejected_logratios)
            )
            losses = losses_chosen + losses_rejected

        elif self.loss_type == "discopop":
            # Eqn (5) of the DiscoPOP paper (https://huggingface.co/papers/2406.08414)
            # This loss was discovered with LLM discovery
            logratios = chosen_logps - rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            logits = logratios - ref_logratios
            logits = logits * self.beta
            # Modulate the mixing coefficient based on the log ratio magnitudes
            log_ratio_modulation = torch.sigmoid(logits / self.args.discopop_tau)
            logistic_component = -F.logsigmoid(logits)
            exp_component = torch.exp(-logits)
            # Blend between logistic and exponential component based on log ratio modulation
            losses = (
                logistic_component * (1 - log_ratio_modulation)
                + exp_component * log_ratio_modulation
            )

        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'exo_pair', "
                "'nca_pair', 'robust', 'bco_pair', 'sppo_hard', 'aot', 'aot_pair', 'discopop', 'apo_zero', 'apo_down']"
            )

        chosen_rewards = (
            self.beta * (chosen_logps.to(device) - ref_chosen_logps.to(device)).detach()
        )
        rejected_rewards = (
            self.beta
            * (rejected_logps.to(device) - ref_rejected_logps.to(device)).detach()
        )

        return losses, chosen_rewards, rejected_rewards

    def get_batch_loss_metrics(
        self,
        model,
        batch: dict[str, Union[list, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        model_output = self.concatenated_forward(model, batch)

        # if ref_chosen_logps and ref_rejected_logps in batch use them, otherwise use the reference model
        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            model_output["chosen_logps"],
            model_output["rejected_logps"],
            ref_chosen_logps,
            ref_rejected_logps,
            chosen_diversity_scores=batch.get("diversity_chosen"),
            chosen_novelty_scores=batch.get("novelty_chosen"),
            chosen_surprise_scores=batch.get("surprise_chosen"),
            chosen_quality_scores=batch.get("quality_chosen"),
            rejected_diversity_scores=batch.get("diversity_rejected"),
            rejected_novelty_scores=batch.get("novelty_rejected"),
            rejected_surprise_scores=batch.get("surprise_rejected"),
            rejected_quality_scores=batch.get("quality_rejected"),
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            losses = (
                losses + self.args.rpo_alpha * model_output["nll_loss"]
            )  # RPO loss from V3 of the paper

        if self.use_weighting:
            losses = losses * model_output["policy_weights"]

        if self.aux_loss_enabled:
            losses = losses + self.aux_loss_coef * model_output["aux_loss"]

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = (
            self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
        )
        metrics[f"{prefix}rewards/rejected"] = (
            self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
        )
        metrics[f"{prefix}rewards/accuracies"] = (
            self.accelerator.gather_for_metrics(reward_accuracies).mean().item()
        )
        metrics[f"{prefix}rewards/margins"] = (
            self.accelerator.gather_for_metrics(chosen_rewards - rejected_rewards)
            .mean()
            .item()
        )
        metrics[f"{prefix}logps/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["chosen_logps"])
            .detach()
            .mean()
            .item()
        )
        metrics[f"{prefix}logps/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["rejected_logps"])
            .detach()
            .mean()
            .item()
        )
        metrics[f"{prefix}logits/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["mean_chosen_logits"])
            .detach()
            .mean()
            .item()
        )
        metrics[f"{prefix}logits/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["mean_rejected_logits"])
            .detach()
            .mean()
            .item()
        )
        if self.args.rpo_alpha is not None:
            metrics[f"{prefix}nll_loss"] = (
                self.accelerator.gather_for_metrics(model_output["nll_loss"])
                .detach()
                .mean()
                .item()
            )
        if self.aux_loss_enabled:
            metrics[f"{prefix}aux_loss"] = (
                self.accelerator.gather_for_metrics(model_output["aux_loss"])
                .detach()
                .mean()
                .item()
            )
        
        def _set_reward_correlation_metrics(batch, rewards, metrics, metric, prefix="", metric_type="chosen"):
            if f"{metric}_{metric_type}" in batch:
                rewards = self.accelerator.gather_for_metrics(rewards)
                metric_results = self.accelerator.gather_for_metrics(
                    batch[f"{metric}_{metric_type}"]
                )
                pearson_stat, pearson_pvalue = stats.pearsonr(
                    rewards.cpu().numpy(), metric_results.cpu().numpy()
                )
                spearman_stat, spearman_pvalue = stats.spearmanr(
                    rewards.cpu().numpy(), metric_results.cpu().numpy()
                )
                metrics[f"{prefix}margins/pearson/stat/{metric}_{metric_type}"] = pearson_stat
                metrics[f"{prefix}margins/pearson/pval/{metric}_{metric_type}"] = pearson_pvalue
                metrics[f"{prefix}margins/spearman/stat/{metric}_{metric_type}"] = spearman_stat
                metrics[f"{prefix}margins/spearman/pval/{metric}_{metric_type}"] = spearman_pvalue

        # # Compute correlation between rewards and creativity scores
        # _set_reward_correlation_metrics(
        #     batch,
        #     chosen_rewards - rejected_rewards,
        #     metrics,
        #     "diversity",
        #     metric_type="chosen",
        #     prefix=prefix,
        # )
        # _set_reward_correlation_metrics(
        #     batch,
        #     chosen_rewards - rejected_rewards,
        #     metrics,
        #     "novelty",
        #     metric_type="chosen",
        #     prefix=prefix,
        # )
        # _set_reward_correlation_metrics(
        #     batch,
        #     chosen_rewards - rejected_rewards,
        #     metrics,
        #     "surprise",
        #     metric_type="chosen",
        #     prefix=prefix,
        # )
        # _set_reward_correlation_metrics(
        #     batch,
        #     chosen_rewards - rejected_rewards,
        #     metrics,
        #     "quality",
        #     metric_type="chosen",
        #     prefix=prefix,
        # )

        return losses.mean(), metrics


def main(script_args, training_args, model_args):
    ################
    # Model & Tokenizer
    ###################

    # Find the latest checkpoint if available
    checkpoint_dir = training_args.output_dir
    if training_args.resume_from_checkpoint:
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            training_args.resume_from_checkpoint = latest_checkpoint
            print(f"Resuming from checkpoint: {latest_checkpoint}")
        else:
            print(f"No checkpoint found in {checkpoint_dir}. Starting from scratch.")
            training_args.resume_from_checkpoint = None

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=(
            "eager"
            if "gemma" in model_args.model_name_or_path
            else model_args.attn_implementation
        ),
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )
    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
    else:
        ref_model = None
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    ##########
    # Training
    ################
    trainer = CDPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != "no"
            else None
        ),
        processing_class=tokenizer,
        peft_config=peft_config,
        data_collator=DataCollatorForPreferenceWithCreativity(
            pad_token_id=tokenizer.pad_token_id
        ),
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, CDPOConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser(
            "dpo", help="Run the DPO training script", dataclass_types=dataclass_types
        )
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
