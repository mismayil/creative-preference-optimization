import argparse
import pathlib
from tqdm import tqdm

from crpo.utils import read_json, write_json, get_template_keys, find_files


def make_short_response_template(sample):
    if sample["task"] in [
        "Real-Life Creative Problem Solving",
        "Malevolent Problems",
        "Alternate Uses of Objects Task",
        "Design Solutions",
        "Consequences",
        "Instances of Common Concepts",
        "Experiment Design",
        "Hypothesis Generation",
        "Research Questions",
        "Associations",
        "Metaphors",
    ]:
        return "{full_prompt}\nPlease limit your response to a few sentences."

    return "{full_prompt}"


def make_constrained_response_template(sample):
    if sample["task"] in [
        "Real-Life Creative Problem Solving",
        "Malevolent Problems",
        "Stories",
        "Experiment Design",
    ]:
        return "{full_prompt}\nPlease limit your response to about 5 sentences and at most 150 words."

    if sample["task"] in [
        "Alternate Uses of Objects Task",
        "Metaphors",
        "Sentence Completion",
        "Research Questions",
        "Question Asking",
        "Consequences",
    ]:
        return "{full_prompt}\nPlease limit your response to about 1 sentence and at most 25 words."

    if sample["task"] in [
        "Design Solutions",
        "Hypothesis Generation",
        "Associations",
        "Instances of Common Concepts",
    ]:
        return "{full_prompt}\nPlease limit your response to about 3 sentences and at most 50 words."

    if sample["task"] in ["Essays", "Poems"]:
        return "{full_prompt}\nPlease limit your response to about 20 sentences and at most 500 words."

    return "{full_prompt}"


# specific values here have been computed based on the results of cpo models in experiments/data/anova_eval_data_v1.json
# median number of sentences and words for each task
def make_specific_constrained_response_template(sample):
    if sample["task"] == "Real-Life Creative Problem Solving":
        return "{full_prompt}\nPlease limit your response to 4 sentences and at most 75 words."

    if sample["task"] == "Alternate Uses of Objects Task":
        return "{full_prompt}\nPlease limit your response to 1 sentence and at most 17 words."

    if sample["task"] == "Design Solutions":
        return "{full_prompt}\nPlease limit your response to 2 sentences and at most 36 words."

    if sample["task"] == "Hypothesis Generation":
        return "{full_prompt}\nPlease limit your response to 1 sentence and at most 22 words."

    if sample["task"] == "Metaphors":
        return "{full_prompt}\nPlease limit your response to 1 sentence and at most 10 words."

    if sample["task"] == "Poems":
        return "{full_prompt}\nPlease limit your response to 5 sentences and at most 150 words."

    if sample["task"] == "Associations":
        return "{full_prompt}\nPlease limit your response to 1 sentence and at most 10 words."

    return "{full_prompt}"


CREATIVE_MATH_TEMPLATE = """
Criteria for evaluating the difference between two mathematical
solutions include:
1. If the methods used to arrive at the solutions are fundamentally
different, such as algebraic manipulation versus geometric reasoning,
they can be considered distinct;
2. Even if the final results are the same, if the intermediate steps or
processes involved in reaching those solutions vary significantly, the
solutions can be considered different;
3. If two solutions rely on different assumptions or conditions, they
are likely to be distinct;
4. A solution might generalize to a broader class of problems, while
another solution might be specific to certain conditions. In such
cases, they are considered distinct;
5. If one solution is significantly simpler or more complex than the
other, they can be regarded as essentially different, even if they lead
to the same result.

Given the following mathematical problem:
{problem}

And some typical solutions:
{solutions}

Please output a novel solution distinct from the given ones for
this math problem.
"""

USER_INSTRUCTION_TEMPLATES = {
    "default": "{full_prompt}",
    "short_response": make_short_response_template,
    "constrained_response": make_constrained_response_template,
    "scr": make_specific_constrained_response_template,
    "cmath": CREATIVE_MATH_TEMPLATE,
    "aut_bs_scr": make_specific_constrained_response_template
}


def prepare_template_value(value):
    if isinstance(value, list):
        return ", ".join(value)
    return value


def prepare_template(sample, template):
    template_keys = get_template_keys(template)
    format_args = {
        k: prepare_template_value(sample[k])
        for k in template_keys
        if sample.get(k) is not None
    }
    return template.format(**format_args)


def prepare_user_instruction(sample, template):
    instruction_template = USER_INSTRUCTION_TEMPLATES[template]
    if callable(instruction_template):
        instruction_template = instruction_template(sample)
    return prepare_template(sample, instruction_template)


def prepare_cmath_user_instruction(sample, template):
    instruction_template = USER_INSTRUCTION_TEMPLATES[template]
    problem = sample["problem"]
    solutions = sample["solutions"]
    solutions = "\n".join([f"{key}. {s}" for key, s in solutions.items()])
    return prepare_template(
        {"problem": problem, "solutions": solutions}, instruction_template
    )

def prepare_aut_bs_user_instruction(sample, template):
    object = sample["full_prompt"].replace("Come up with an original and creative use for the following object: ", "").strip().strip(".").strip()
    user_prompt = [
        prepare_user_instruction(sample, template),
        f"str(f'Name one or more advantages to using a {object} for that purpose. Please keep your response concise and to the point.')",
        f"str(f'Name one or more drawbacks to using a {object} for that purpose. Please keep your response concise and to the point.')",
        f"str(f'Based on these advantages and drawbacks, do you think using a {object} for that purpose is a good idea? If Yes, output \"{{responses[0]}}\" as your final answer, otherwise suggest a different creative and good use for a {object}.')",
        f"str(f'If someone suggested using a {object} for the use in your last response, would you be surprised and think it was a novel idea? If Yes, output \"{{responses[3]}}\" as your final answer, otherwise suggest a final creative, novel and surprising use for a {object}.')",
    ]
    return user_prompt

USER_INSTRUCTION_PROCESSORS = {
    "default": prepare_user_instruction,
    "cmath": prepare_cmath_user_instruction,
    "aut_bs": prepare_aut_bs_user_instruction,
    "aut_bs_scr": prepare_aut_bs_user_instruction,
}

SHOT_PROCESSORS = {
    "default": lambda *args, **kwargs: "",
}


def prepare_sample_for_eval(sample, template, num_shots=1, shot_data=None):
    user_instr_processor = USER_INSTRUCTION_PROCESSORS.get(
        template, USER_INSTRUCTION_PROCESSORS["default"]
    )
    shot_processor = SHOT_PROCESSORS.get(template, SHOT_PROCESSORS["default"])

    eval_data = []

    user_prompt = user_instr_processor(sample, template)
    shot_prompt = shot_processor(
        sample, template, num_shots=num_shots, shot_data=shot_data
    )

    if shot_prompt:
        if isinstance(user_prompt, list):
            user_prompt += ["\n\n" + shot_prompt]
        else:
            user_prompt += "\n\n" + shot_prompt

    if isinstance(user_prompt, list):
        user_prompt = [p.strip() for p in user_prompt]
    else:
        user_prompt = user_prompt.strip()

    eval_data.append(
        {**sample, "user_prompt": user_prompt, "template": template}
    )

    return eval_data


def main():
    parser = argparse.ArgumentParser(description="Prepare LM evaluation data")
    parser.add_argument(
        "-d",
        "--datapath",
        type=str,
        help="Path to task data in json or directory",
        required=True,
    )
    parser.add_argument(
        "-t", "--template", type=str, default="default", help="Template name"
    )
    parser.add_argument(
        "-s",
        "--suffix",
        type=str,
        default="",
        help="Custom suffix for output file path.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory path. Defaults to input directory path.",
    )
    parser.add_argument(
        "-sp",
        "--shot-path",
        type=str,
        default=None,
        help="Path to shot examples in json",
    )
    parser.add_argument(
        "-n",
        "--num-shots",
        type=int,
        default=1,
        help="Number of shot examples to include",
    )

    args = parser.parse_args()

    datapaths = []

    datapath = pathlib.Path(args.datapath)

    if datapath.is_file():
        datapaths.append(args.datapath)
    else:
        datapaths.extend(find_files(args.datapath, "json"))

    for datapath in datapaths:
        input_data = read_json(datapath)
        shot_data = read_json(args.shot_path) if args.shot_path is not None else None

        eval_data = []

        for sample in tqdm(
            input_data["data"], desc=f"Preparing {datapath} for evaluation"
        ):
            eval_data.extend(
                prepare_sample_for_eval(
                    sample,
                    template=args.template,
                    num_shots=args.num_shots,
                    shot_data=shot_data,
                )
            )

        datapath = pathlib.Path(datapath)
        output_dir = (
            pathlib.Path(args.output_dir)
            if args.output_dir is not None
            else datapath.parent
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        eval_data_path = (
            output_dir / f"{datapath.stem}_eval_{args.template}{args.suffix}.json"
        )

        output_data = {
            "metadata": {
                "source": str(datapath),
                "template": args.template,
                "size": len(eval_data),
                "shot_path": args.shot_path,
            },
            "data": eval_data,
        }
        write_json(output_data, eval_data_path)

        print(f"Output data saved to {eval_data_path}")


if __name__ == "__main__":
    main()
