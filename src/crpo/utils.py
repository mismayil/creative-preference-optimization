import os, sys
from string import Formatter
import json
import uuid
import os
import glob
import pandas as pd
import dataclasses
from typing import Optional, List, NewType, Any, Tuple
from dataclasses import dataclass
from transformers import HfArgumentParser
import wandb
from dotenv import load_dotenv
import multiprocessing
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import numpy as np

load_dotenv()

MODEL_COSTS = {
    "gpt-3.5-turbo": {"input": 0.0000015, "output": 0.000002},
    "gpt-4": {"input": 30e-6, "output": 60e-6},
    "gpt-4o": {"input": 2.5e-6, "output": 10e-6},
    "gpt-4o-mini": {"input": 0.15e-6, "output": 0.6e-6},
    "gpt-4-0125-preview": {"input": 10e-6, "output": 30e-6},
    "gpt-4o-2024-08-06": {"input": 2.5e-6, "output": 10e-6},
    "text-davinci-003": {"input": 0.00002, "output": 0.00002},
    "gemini-1.5-flash": {"input": 3.5e-7, "output": 1.05e-6},
    "gemini-1.5-pro": {"input": 3.5e-6, "output": 10.5e-6},
    "claude-3-5-sonnet-20240620": {"input": 3e-6, "output": 15e-6},
    "claude-3-5-haiku-20241022": {"input": 1e-6, "output": 5e-6},
    "claude-3-opus-20240229": {"input": 15e-6, "output": 75e-6},
    "claude-3-sonnet-20240229": {"input": 3e-6, "output": 15e-6},
    "claude-3-haiku-20240307": {"input": 0.25e-6, "output": 1.25e-6},
}

MODEL_ENCODINGS = {
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "text-davinci-003": "p50k_base",
    "gpt-4o": "o200k_base",
}

# claude, gpt and gemini model sizes based on https://arxiv.org/abs/2412.19260 and https://lifearchitect.substack.com/p/the-memo-special-edition-claude-3
MODEL_SIZE_MAP = {
    "llama-3.2-3b-instruct": 3e9,
    "llama-3.1-8b-instruct": 8e9,
    "llama-3.1-70b-instruct": 70e9,
    "llama-3.1-405b-instruct": 405e9,
    "qwen-2.5-coder-32b-instruct": 32e9,
    "wizardlm-2-8x22b": 22e9,
    "gemma-2-27b": 27e9,
    "gemma-2-9b": 9e9,
    "gemma-2b": 2e9,
    "deepseek-llm-chat-67b": 67e9,
    "mythomax-l2-13b": 13e9,
    "mistral-7b-instruct-v0.3": 7e9,
    "mixtral-8x7b-instruct": 13e9,
    "mixtral-8x22b-instruct": 39e9,
    "nous-hermes-2-mixtral-8x7b-dpo": 7e9,
    "qwen-2.5-7b-instruct": 7e9,
    "qwen-2.5-72b-instruct": 72e9,
    "stripedhyena-nous-7b": 7e9,
    "solar-10.7b-instruct-v1.0": 10.7e9,
    "nemotron-4-340b-instruct": 340e9,
    "yi-large": 34e9,
    "granite-34b-code-instruct": 34e9,
    "granite-8b-code-instruct": 8e9,
    "mistral-nemo-12b-instruct": 12e9,
    "baichuan2-13b-chat": 13e9,
    "nemotron-mini-4b-instruct": 4e9,
    "zamba2-7b-instruct": 7e9,
    "granite-3.0-8b-instruct": 8e9,
    "dbrx-instruct": 132e9,
    "gemma-2-2b-it": 2e9,
    "yi-1.5-34b-chat": 34e9,
    "yi-1.5-9b-chat": 9e9,
    "stablelm-2-12b-chat": 12e9,
    "stablelm-zephyr-3b": 3e9,
    "olmo-2-7b": 7e9,
    "olmo-2-13b": 13e9,
    "persimmon-8b-chat": 8e9,
    "mpt-7b-8k-chat": 7e9,
    "mpt-30b-chat": 30e9,
    "llama-3.2-1b-instruct": 1e9,
    "deepseek-llm-7b-chat": 7e9,
    "baichuan2-7b-chat": 7e9,
    "zamba2-2.7b-instruct": 2.7e9,
    "zamba2-1.2b-instruct": 1.2e9,
    "granite-3.0-2b-instruct": 2e9,
    "gpt-3.5-turbo": 175e9,
    "grok-beta": 314e9,
    "reka-core": 67e9,
    "reka-edge": 7e9,
    "reka-flash": 21e9,
    "glm-4-0520": 130e9,
    "jamba-1.5-mini": 12e9,
    "jamba-1.5-large": 94e9,
    "Phi-3-mini-4k-instruct": 3.8e9,
    "Phi-3-small-8k-instruct": 7e9,
    "Phi-3-medium-4k-instruct": 14e9,
    "Phi-3.5-MoE-instruct": 6.6e9,
    "command-r-plus": 104e9,
    "c4ai-aya-expanse-8b": 8e9,
    "c4ai-aya-expanse-32b": 32e9,
    "mistral-large-latest": 123e9,
    "ministral-3b-latest": 3e9,
    "ministral-8b-latest": 8e9,
    "mistral-small-latest": 22e9,
    "lfm-40b": 40e9,
    "claude-3-5-haiku-20241022": 20e9,
    "claude-3-5-sonnet-20240620": 175e9,
    "claude-3-opus-20240229": 500e9,
    "gpt-4": 500e9,
    "gpt-4o": 200e9,
    "gemini-1.5-flash": 500e9,
    "gemini-1.5-pro": 500e9,
}


def find_latest_checkpoint(checkpoint_dir):
    if os.path.exists(checkpoint_dir):
        checkpoints = [
            os.path.join(checkpoint_dir, d)
            for d in os.listdir(checkpoint_dir)
            if d.startswith("checkpoint-")
        ]
        if not checkpoints:
            return None
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        return latest_checkpoint
    return None


def num_tokens_from_string(text, model):
    import tiktoken

    if model not in MODEL_ENCODINGS:
        return 0
    encoding_name = MODEL_ENCODINGS[model]
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


def write_json(data, path, ensure_ascii=True, indent=4):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


def generate_unique_id():
    return str(uuid.uuid4()).split("-")[-1]


def find_files(directory, extension="json"):
    return glob.glob(f"{directory}/**/*.{extension}", recursive=True)


def concatenate(lists):
    return [item for sublist in lists for item in sublist]


def levenshtein_distance(s, t):
    m = len(s)
    n = len(t)
    d = [[0] * (n + 1) for i in range(m + 1)]

    for i in range(1, m + 1):
        d[i][0] = i

    for j in range(1, n + 1):
        d[0][j] = j

    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if s[i - 1] == t[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(
                d[i - 1][j] + 1,  # deletion
                d[i][j - 1] + 1,  # insertion
                d[i - 1][j - 1] + cost,
            )  # substitution

    return d[m][n]


def batched(lst, size=4):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def compute_usage(
    sample,
    model,
    input_attrs=["system_prompt", "user_prompt"],
    output_attrs=["output"],
    max_input_tokens=None,
    max_output_tokens=None,
):
    if model not in MODEL_COSTS:
        return None, None

    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    usage = sample.get("usage")

    if not usage:
        input_tokens = 0
        output_tokens = 0

        if max_input_tokens:
            input_tokens = max_input_tokens
        else:
            for attr in input_attrs:
                if attr in sample:
                    input_tokens += num_tokens_from_string(sample[attr], model)

        if max_output_tokens:
            output_tokens = max_output_tokens
        else:
            for attr in output_attrs:
                if attr in sample:
                    output_tokens += num_tokens_from_string(sample[attr], model)

        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    input_cost = usage["input_tokens"] * MODEL_COSTS[model]["input"]
    output_cost = usage["output_tokens"] * MODEL_COSTS[model]["output"]

    return usage, {
        "input": input_cost,
        "output": output_cost,
        "total": input_cost + output_cost,
    }


def get_template_keys(template):
    return [i[1] for i in Formatter().parse(template) if i[1] is not None]


def is_immutable(obj):
    return isinstance(obj, (str, int, float, bool, tuple, type(None)))


def cache(cache_dict):
    def decorator_cache(func):
        def wrapper(*args, **kwargs):
            if all(is_immutable(arg) for arg in args) and all(
                is_immutable(val) for val in kwargs.values()
            ):
                key = (args, frozenset(kwargs.items()))
                if key in cache_dict:
                    return cache_dict[key]
                result = func(*args, **kwargs)
                cache_dict[key] = result
            else:
                result = func(*args, **kwargs)
            return result

        return wrapper

    return decorator_cache


def concat_dfs(df_lst):
    shared_columns = None

    for df in df_lst:
        if shared_columns is None:
            shared_columns = set(df.columns)
        else:
            shared_columns.intersection_update(df.columns)

    shared_columns = list(shared_columns)
    return pd.concat([df[shared_columns] for df in df_lst]).reset_index()


def build_baichuan_chat_input(model, tokenizer, messages, max_new_tokens=0):
    import torch

    def _parse_messages(messages, split_role="user"):
        system, rounds = "", []
        round = []
        for i, message in enumerate(messages):
            if message["role"] == "system":
                assert i == 0
                system = message["content"]
                continue
            if message["role"] == split_role and round:
                rounds.append(round)
                round = []
            round.append(message)
        if round:
            rounds.append(round)
        return system, rounds

    max_new_tokens = max_new_tokens or model.generation_config.max_new_tokens
    max_input_tokens = model.config.model_max_length - max_new_tokens
    system, rounds = _parse_messages(messages, split_role="user")
    system_tokens = tokenizer.encode(system)
    max_history_tokens = max_input_tokens - len(system_tokens)

    history_tokens = []
    for round in rounds[::-1]:
        round_tokens = []
        for message in round:
            if message["role"] == "user":
                round_tokens.append(model.generation_config.user_token_id)
            else:
                round_tokens.append(model.generation_config.assistant_token_id)
            round_tokens.extend(tokenizer.encode(message["content"]))
        if (
            len(history_tokens) == 0
            or len(history_tokens) + len(round_tokens) <= max_history_tokens
        ):
            history_tokens = round_tokens + history_tokens  # concat left
            if len(history_tokens) < max_history_tokens:
                continue
        break

    input_tokens = system_tokens + history_tokens
    if messages[-1]["role"] != "assistant":
        input_tokens.append(model.generation_config.assistant_token_id)
    input_tokens = input_tokens[-max_input_tokens:]  # truncate left
    return torch.LongTensor([input_tokens]).to(model.device)


DataClassType = NewType("DataClassType", Any)


class H4ArgumentParser(HfArgumentParser):
    def parse_yaml_and_args(
        self, yaml_arg: str, other_args: Optional[List[str]] = None
    ) -> List[dataclass]:
        """
        Parse a YAML file and overwrite the default/loaded values with the values provided to the command line.

        Args:
            yaml_arg (`str`):
                The path to the config file used
            other_args (`List[str]`, *optional`):
                A list of strings to parse as command line arguments, e.g. ['--arg=val', '--arg2=val2'].

        Returns:
            [`List[dataclass]`]: a list of dataclasses with the values from the YAML file and the command line
        """
        arg_list = self.parse_yaml_file(os.path.abspath(yaml_arg))

        outputs = []
        # strip other args list into dict of key-value pairs
        other_args = {
            arg.split("=")[0].strip("-"): arg.split("=")[1] for arg in other_args
        }
        used_args = {}

        # overwrite the default/loaded value with the value provided to the command line
        # adapted from https://github.com/huggingface/transformers/blob/d0b5002378daabf62769159add3e7d66d3f83c3b/src/transformers/hf_argparser.py#L327
        for data_yaml, data_class in zip(arg_list, self.dataclass_types):
            keys = {f.name for f in dataclasses.fields(data_yaml) if f.init}
            inputs = {k: v for k, v in vars(data_yaml).items() if k in keys}
            for arg, val in other_args.items():
                # add only if in keys

                if arg in keys:
                    base_type = data_yaml.__dataclass_fields__[arg].type
                    inputs[arg] = val

                    # cast type for ints, floats (default to strings)
                    if base_type in [int, float]:
                        inputs[arg] = base_type(val)

                    if base_type == List[str]:
                        inputs[arg] = [str(v) for v in val.split(",")]

                    # bool of a non-empty string is True, so we manually check for bools
                    if base_type is bool:
                        if val in ["true", "True"]:
                            inputs[arg] = True
                        else:
                            inputs[arg] = False

                    # add to used-args so we can check if double add
                    if arg not in used_args:
                        used_args[arg] = val
                    else:
                        raise ValueError(
                            f"Duplicate argument provided: {arg}, may cause unexpected behavior"
                        )

            obj = data_class(**inputs)
            outputs.append(obj)

        return outputs

    def parse(self) -> DataClassType | Tuple[DataClassType]:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
            # If we pass only one argument to the script and it's the path to a YAML file,
            # let's parse it to get our arguments.
            output = self.parse_yaml_file(os.path.abspath(sys.argv[1]))
        # parse command line args and yaml file
        elif len(sys.argv) > 2 and sys.argv[1].endswith(".yaml"):
            output = self.parse_yaml_and_args(
                os.path.abspath(sys.argv[1]), sys.argv[2:]
            )
        # parse command line args only
        else:
            output = self.parse_args_into_dataclasses()

        if len(output) == 1:
            output = output[0]
        return output


def make_lm_conv(prompt, response):
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def wandb_log_run(name, config=None, metrics=None, project=None, run_id=None):
    if run_id is not None:
        run = get_wandb_run(run_id, project=project)
        if run:
            run.delete()
    run = wandb.init(name=name, project=project, config=config)
    if metrics:
        run.log(metrics)
    run.finish()
    return run


def get_wandb_run(run_id, entity=None, project=None):
    entity = os.getenv("WANDB_ENTITY") if entity is None else entity
    project = os.getenv("WANDB_PROJECT") if project is None else project
    try:
        return wandb.Api().run(f"{entity}/{project}/{run_id}")
    except Exception as e:
        print(f"Error getting run {run_id}: {str(e)}")
        return None


def none_or_int(value):
    if value.lower() == "none":
        return None
    return int(value)


def none_or_str(value):
    if value.lower() == "none":
        return None
    return value


def parallelize(func, data, batch_size=None, unpack_args=False):
    """
    Apply a function to each item in a list in parallel and return the results in the same order.

    Args:
        func (callable): The function to apply to each item.
        data (list): The list of items to process.
        batch_size (int, optional): The size of each batch. If None, it will be set to the number of CPU cores.
        unpack_args (bool, optional): Whether to unpack the arguments when calling. Defaults to False.

    Returns:
        list: The results of applying the function to each item in the list.
    """
    if batch_size is None:
        batch_size = multiprocessing.cpu_count() // 2

    with multiprocessing.Pool(batch_size) as pool:
        if unpack_args:
            results = list(pool.starmap(func, data, chunksize=batch_size))
        else:
            results = list(pool.map(func, data, chunksize=batch_size))

    return results


def prepare_metrics_for_wandb(metrics, exclude_prefixes=None):
    if exclude_prefixes is None:
        exclude_prefixes = []

    wandb_metrics = {}

    for key, value in metrics.items():
        if any(key.startswith(prefix) for prefix in exclude_prefixes):
            continue

        if isinstance(value, dict):
            wandb_metrics[key] = prepare_metrics_for_wandb(value, exclude_prefixes)
        else:
            if is_immutable(value):
                wandb_metrics[key] = value

    return wandb_metrics


def get_model_name(model_path):
    model_path_parts = model_path.split("/")
    model_name = model_path_parts[-1].lower()

    if model_name.startswith("checkpoint"):
        model_name = model_path_parts[-2].lower()

    return model_name


def load_lm(model_path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer


def load_rm(model_path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        num_labels=1,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    if not model.config.pad_token_id:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def detect_outliers_iqr(data, multiplier=1.5):
    """
    Detect outliers in a 1D array using the IQR method.

    Parameters:
        data (array-like): The input data.
        multiplier (float): The IQR multiplier, usually 1.5 (can use 3.0 for more extreme outliers).

    Returns:
        outlier_indices (list): Indices of outlier elements in the data.
    """
    data = np.array(data)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    outlier_indices = np.where((data < lower_bound) | (data > upper_bound))[0]
    return outlier_indices

def is_evaluable(expr):
    try:
        compile(expr, "<string>", "eval")
        return True
    except Exception:
        return False