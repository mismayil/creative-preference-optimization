# adapted from https://github.com/huggingface/trl/blob/main/trl/scripts/sft.py

import argparse
from dotenv import load_dotenv

load_dotenv()

from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
)

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from crpo.utils import find_latest_checkpoint


def main(script_args, training_args, model_args):
    if True:
        return 
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

    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # Create model
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    valid_image_text_architectures = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values()

    if config.architectures and any(
        arch in valid_image_text_architectures for arch in config.architectures
    ):
        from transformers import AutoModelForImageTextToText

        model_kwargs.pop("use_cache", None)  # Image models do not support cache
        model = AutoModelForImageTextToText.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )
    else:
        if "gemma" in model_args.model_name_or_path.lower():
            model_kwargs["attn_implementation"] = "eager"
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
    )

    if (
        tokenizer.chat_template is None
        and model_args.model_name_or_path == "meta-llama/Llama-3.1-8B"
    ):
        # Use the instruct model's chat template
        tokenizer.chat_template = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct"
        ).chat_template

    # Adjust tokenizer padding for causal modeling
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        if "llama" in model_args.model_name_or_path.lower():
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
        elif (
            "ministral" in model_args.model_name_or_path.lower()
            or "mistral-small" in model_args.model_name_or_path.lower()
        ):
            tokenizer.pad_token = "<pad>"  # based on the vocabulary
        elif "mistral-7b" in model_args.model_name_or_path.lower():
            tokenizer.pad_token = "<unk>"
        else:
            raise ValueError(
                "Tokenizer does not have a pad token. Please decide what token to set as a padding token, note that it should be different from the eos token."
            )

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != "no"
            else None
        ),
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser(
            "sft", help="Run the SFT training script", dataclass_types=dataclass_types
        )
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
