from collections import defaultdict, Counter
import random
from tqdm import tqdm
from datasets import Dataset
import numpy as np


def prepare_full_prompt(task, dataset, prompt, language=None):
    full_prompt = prompt
    lang_instruction = ""

    if language and language.lower() != "english":
        lang_instruction = f"Please respond in {language.lower().capitalize()}.\n"

    if task == "Consequences":
        full_prompt = f"Come up with an original and creative consequence for the following scenario:\n{prompt}".strip()

    elif task == "Alternate Uses of Objects Task":
        full_prompt = f"Come up with an original and creative use for the following object: {prompt}".strip()

    elif task == "Design Solutions":
        if dataset == "OE_ratings_scored":
            assert prompt.startswith("Develop as many design ideas as you can")
            prompt = prompt.replace(
                "Develop as many design ideas as you can",
                "Come up with an original and creative solution",
            )
            full_prompt = prompt.strip()
        elif dataset == "Rater_Sheets_RL_MASTER":
            assert prompt.startswith(
                "Please list all of the unusual, creative, and uncommon engineering solutions"
            )
            prompt = prompt.replace(
                "Please list all of the unusual, creative, and uncommon engineering solutions",
                "Come up with an original and creative solution",
            )
            full_prompt = prompt.strip()
        elif dataset == "Environment_topic_Idea_ratings_unpub_data_M_Baas":
            full_prompt = f"Come up with an original and creative solution to {prompt.lower()}.".strip()

    elif task == "Real-Life Creative Problem Solving":
        if dataset in ["AverieThesis2024cpst", "averie2024thesis"]:
            return None
        elif dataset in ["WE_kapoor", "EE_kapoor"]:
            assert (
                "Think of as many ethical, legal, and moral, but also original ways"
                in prompt
            )
            prompt = prompt.replace(
                "Think of as many ethical, legal, and moral, but also original ways",
                "Think of an ethical, legal, and moral, but also original way",
            )
            full_prompt = f"Come up with an original and creative solution for the following real-world problem:\n{prompt}".strip()
        else:
            full_prompt = f"Come up with an original and creative solution for the following real-world problem:\n{prompt}.".strip()

    elif task == "Metaphors":
        if dataset == "DiStefano2024crj_Meta":
            full_prompt = f"Come up with an original and creative metaphor to describe the following concept: {prompt}.".strip()
        elif dataset == "MersealDissertation":
            full_prompt = f"Finish the sentence with an original and creative metaphor: {prompt}.".strip()
        elif dataset == "yuhua2024cogsci_meta":
            full_prompt = f"Come up with an original and creative metaphoric equivalent for the concept described below:\n{prompt}.".strip()

    elif task == "Research Questions":
        full_prompt = f"Come up with an original and creative scientific research question for the following scenario:\n{prompt}".strip()

    elif task == "Hypothesis Generation":
        if dataset == "s1_sctt_goecke_HG":
            full_prompt = f"Come up with an original and creative scientific hypothesis for the following concept:\n{prompt}.".strip()
        else:
            full_prompt = f"Come up with an original and creative scientific hypothesis for the following scenario:\n{prompt}".strip()

    elif task == "Experiment Design":
        if dataset == "s1_sctt_goecke_ED":
            full_prompt = f"Come up with an original and creative experiment about the following concept: {prompt}.".strip()
        else:
            full_prompt = f"Come up with an original and creative experiment to test the following hypothesis or research question:\n{prompt}.".strip()

    elif task == "Malevolent Problems":
        if dataset in ["WU_kapoor", "EU_kapoor"]:
            assert (
                "Think of as many unethical, illegal, and immoral, but also original ways"
                in prompt
            )
            prompt = prompt.replace(
                "Think of as many unethical, illegal, and immoral, but also original ways",
                "Think of an unethical, illegal, and immoral, but also original way",
            )
            full_prompt = f"Come up with an original and creative way to react to the following unfair scenario and get back at or sabotage the wrongdoer:\n{prompt}".strip()
        elif dataset in [
            "Creativity_Lying_Collated_Final_Ratings_Creativ",
            "Malevolent_1_fromAgnoliMalevolent2024",
            "Malevolent_2_fromAgnoliMalevolent2024",
        ]:
            full_prompt = f"Come up with an original and creative way to react to the following unfair scenario and get back at or sabotage the wrongdoer:\n{prompt}".strip()

    elif task == "Question Asking":
        if dataset == "AQTCreativity":
            full_prompt = f"Come up with an original and creative question you could ask about the following object: {prompt}.".strip()

    elif task == "Sentence Completion":
        full_prompt = f"Finish the sentence with an original and creative ending: {prompt}.".strip()

    elif task == "Instances of Common Concepts":
        if dataset == "motesf_0_instances":
            full_prompt = f"Come up with an original and creative instance of the following concept: {prompt}.".strip()

    elif task == "Analogies":
        if dataset == "s2_sctt_goecke_AN":
            concepts = prompt.split("_")
            full_prompt = "Come up with an original and creative analogy to explain the following concept or concepts: {concepts}.".format(
                concepts=", ".join(concepts)
            ).strip()

    elif task == "Stories":
        if dataset in [
            "luchini2024paca_english",
            "luchini2024paca_multilingual_english",
            "Apollinariia2024stories",
            "luchini2024paca_multilingual_arabic",
            "SWU_short_stroy_Qunlin_Chen",
            "luchini2024paca_multilingual_chinese",
            "luchini2024paca_multilingual_dutch",
            "luchini2024paca_multilingual_french",
            "luchini2024paca_multilingual_german",
            "luchini2024paca_multilingual_hebrew",
            "luchini2024paca_multilingual_italian",
            "luchini2024paca_multilingual_polish",
            "luchini2024paca_multilingual_russian",
            "luchini2024paca_multilingual_spanish",
        ]:
            words = prompt.split("-")
            full_prompt = "Come up with an original and creative story which includes the following 3 words, make it around 5 sentences long: {words}.".format(
                words=", ".join(words)
            ).strip()

    elif task == "Associations":
        if dataset == "s5_data_long":
            full_prompt = f"Come up with an original and creative word that is associated with the following word: {prompt}.".strip()

    elif task == "Essays":
        if dataset in ["Etown2012", "Etown2011"]:
            full_prompt = "Come up with an original and creative essay describing a project you would like to complete in your field of study."

    elif task == "Poems":
        full_prompt = f"Come up with an original and creative poem about the following concept: {prompt}.".strip()

    elif task == "Emotions in Everyday Situations":
        full_prompt = f"Come up with an original and creative sentence describing the emotions that the following everyday situation triggers in you: {prompt}.".strip()

    elif task == "Evoking Emotional Responses from People":
        full_prompt = f"Come up with an original and creative way to evoke the following emotional reaction in people as a television show producer: {prompt}.".strip()

    elif task == "Emotional Trials":
        full_prompt = f"Come up with an original and creative way to combine three emotional terms that you associate with the following situation: {prompt}.".strip()

    elif task == "Invent Nicknames":
        full_prompt = f"Come up with an original and creative nickname for the following thing: {prompt}.".strip()

    elif task == "Composites":
        full_prompt = "Come up with an original and creative compound word that describes the following emotion, by using the following emotion as part of the compound word: {prompt}.".strip()

    elif task == "Alternate Titles Generation":
        full_prompt = f"Come up with an original and creative for the following well-known book or movie: {prompt}.".strip()

    elif task == "Situation Redescription":
        full_prompt = f"Come up with an original and creative way to reevaluate the following situation and see it in a positive light: {prompt}.".strip()

    elif task == "Plot Titles Generation":
        full_prompt = f"Come up with an original and creative title for the following passage: {prompt}.".strip()

    elif task == "Combining Objects":
        full_prompt = f"Come up with two objects that can be combined in an original and creative way to achieve the following purpose: {prompt}.".strip()

    else:
        print(
            f"Task {task} and dataset {dataset} not recognized, returning the original prompt."
        )

    full_prompt = f"{full_prompt}\n{lang_instruction}".strip()

    return full_prompt


def add_full_prompt(data_df):
    data_df["FullPrompt"] = data_df.apply(
        lambda x: prepare_full_prompt(
            x["TasksNamesFull"], x["Dataset"], x["Prompt"], x["Language"]
        ),
        axis=1,
    )
    return data_df


def get_data_dict(data_df, ignored_tasks=None):
    if ignored_tasks is None:
        ignored_tasks = []

    data_dict = defaultdict(lambda: defaultdict(list))
    for i, row in data_df.iterrows():
        if row["FullPrompt"] is not None:
            task = row["TasksNamesFull"]
            if task not in ignored_tasks:
                data_dict[row["TasksNamesFull"]][row["FullPrompt"]].append(
                    row.to_dict()
                )
    return data_dict


def sample_data(data_dict, num_samples=50):
    sampled_data = {}
    for task, task_data in data_dict.items():
        sampled_data[task] = {}
        for k, v in task_data.items():
            if len(v) < 2:
                continue
            if len(v) < num_samples:
                sampled_data[task][k] = v
            else:
                sampled_data[task][k] = random.sample(v, num_samples)
    return sampled_data


def prepare_sft_data(data, topk=None, min_score=None):
    sft_data = {}
    for task, task_data in data.items():
        sft_data[task] = {}
        for prompt, evals in tqdm(
            task_data.items(), desc=f"Preparing SFT data for {task}"
        ):
            if min_score is not None:
                evals = [ev for ev in evals if ev["facsco_grm_scaled"] >= min_score]

            evals = sorted(evals, key=lambda x: x["facsco_grm_scaled"], reverse=True)
            evals = evals[:topk] if topk else evals
            sft_data[task][prompt] = []

            for ev in evals:
                sft_data[task][prompt].append(
                    {
                        "prompt": prompt,
                        "dataset": ev["Dataset"],
                        "task": ev["TasksNamesFull"],
                        "score_label": ev["RatingLabel"],
                        "response": ev["Response"],
                        "score": ev["facsco_grm_scaled"],
                    }
                )

    return sft_data


def prepare_preference_data(
    data,
    min_margin=None,
    max_margin=None,
    max_matching=None,
    strict_label_checking=False,
    min_score=None,
    length_balancing=False,
    max_length_balancing_size=None,
):
    preference_data = {}
    for task, task_data in data.items():
        preference_data[task] = {}
        for prompt, evals in tqdm(
            task_data.items(), desc=f"Preparing preference data for {task}"
        ):
            preference_data[task][prompt] = []
            eval_counter = Counter()
            for i in range(len(evals)):
                for j in range(i + 1, len(evals)):
                    p1_score = evals[i]["facsco_grm_scaled"]
                    p2_score = evals[j]["facsco_grm_scaled"]

                    if min_score is not None and (
                        p1_score < min_score or p2_score < min_score
                    ):
                        continue

                    if min_margin is not None and abs(p1_score - p2_score) < min_margin:
                        continue

                    if max_margin is not None and abs(p1_score - p2_score) > max_margin:
                        continue

                    p1_dataset = evals[i]["Dataset"]
                    p2_dataset = evals[j]["Dataset"]
                    p1_task = evals[i]["TasksNamesFull"]
                    p2_task = evals[j]["TasksNamesFull"]
                    p1_response = evals[i]["Response"]
                    p2_response = evals[j]["Response"]
                    p1_score_label = evals[i]["RatingLabel"]
                    p2_score_label = evals[j]["RatingLabel"]

                    if max_matching:
                        if (
                            eval_counter.get(i, 0) + 1 > max_matching
                            or eval_counter.get(j, 0) + 1 > max_matching
                        ):
                            continue
                        eval_counter.update([i, j])

                    assert p1_task == p2_task
                    if strict_label_checking:
                        assert p1_score_label == p2_score_label
                    else:
                        pass
                        # assert (
                        #     p1_score_label == "creativity"
                        #     or p1_score_label == "originality"
                        # )
                        # assert (
                        #     p2_score_label == "creativity"
                        #     or p2_score_label == "originality"
                        # )

                    if p1_score > p2_score:
                        preference_data[task][prompt].append(
                            {
                                "prompt": prompt,
                                "dataset": p1_dataset,
                                "task": p1_task,
                                "score_label": p1_score_label,
                                "response_chosen": p1_response,
                                "response_rejected": p2_response,
                                "score_chosen": p1_score,
                                "score_rejected": p2_score,
                            }
                        )
                    else:
                        preference_data[task][prompt].append(
                            {
                                "prompt": prompt,
                                "dataset": p1_dataset,
                                "task": p1_task,
                                "score_label": p1_score_label,
                                "response_chosen": p2_response,
                                "response_rejected": p1_response,
                                "score_chosen": p2_score,
                                "score_rejected": p1_score,
                            }
                        )

            if length_balancing:
                evals = preference_data[task][prompt]
                length_biased_evals = [
                    ev
                    for ev in evals
                    if len(ev["response_chosen"].split())
                    > len(ev["response_rejected"].split())
                ]
                non_length_biased_evals = [
                    ev
                    for ev in evals
                    if len(ev["response_chosen"].split())
                    <= len(ev["response_rejected"].split())
                ]
                sample_size = min(
                    len(length_biased_evals), len(non_length_biased_evals)
                )
                if max_length_balancing_size:
                    sample_size = min(sample_size, max_length_balancing_size)
                preference_data[task][prompt] = random.sample(
                    length_biased_evals, sample_size
                ) + random.sample(non_length_biased_evals, sample_size)

    return preference_data


def prepare_split_data_by_prompt(data, split=0.8):
    split1_data = {}
    split2_data = {}

    for task, task_data in tqdm(data.items(), desc="Splitting data by prompt"):
        task_split1_data = {}
        task_split2_data = {}

        # If there is only one prompt, add it to split1
        if len(task_data) == 1:
            split1_data[task] = task_data
            split2_data[task] = {}
            continue

        # If there are two prompts, split them
        if len(task_data) == 2:
            for i, (prompt, evals) in enumerate(task_data.items()):
                if i == 0:
                    task_split1_data[prompt] = evals
                else:
                    task_split2_data[prompt] = evals
            split1_data[task] = task_split1_data
            split2_data[task] = task_split2_data
            continue

        # If there are more than two prompts, split them
        num_total_samples = sum([len(v) for v in task_data.values()])
        split_size = int(num_total_samples * split)
        task_data = sorted(
            [(k, v) for k, v in task_data.items()],
            key=lambda x: len(x[1]),
            reverse=True,
        )

        for prompt, evals in task_data:
            if (
                sum([len(v) for v in task_split1_data.values()]) + len(evals)
                <= split_size
            ):
                task_split1_data[prompt] = evals
            else:
                task_split2_data[prompt] = evals

        split1_data[task] = task_split1_data
        split2_data[task] = task_split2_data

    return split1_data, split2_data


def flatten_data(data):
    flat_data = []
    for task, task_data in tqdm(data.items(), desc="Flattening data"):
        for prompt, evals in task_data.items():
            flat_data.extend(evals)
    return flat_data


def convert_to_trl_sft_dataset(data, template="chat"):
    trl_data = []

    for d in tqdm(data, desc="Converting preference data to TRL SFT dataset"):
        core_data = {
            "dataset": d["dataset"],
            "task": d["task"],
            "score_label": d["score_label"],
            "score": d["score"],
        }

        if template == "chat":
            core_data["messages"] = [
                {"role": "user", "content": d["prompt"]},
                {"role": "assistant", "content": d["response"]},
            ]
        elif template == "instruction":
            core_data["prompt"] = d["prompt"]
            core_data["completion"] = d["response"]

        trl_data.append(core_data)

    return Dataset.from_generator(lambda: trl_data)


def convert_to_trl_pref_dataset(data, template="chat", template_format="implicit"):
    trl_data = []

    for d in tqdm(data, desc="Converting preference data to TRL preference dataset"):
        core_data = {
            "dataset": d["dataset"],
            "task": d["task"],
            "score_label": d["score_label"],
            "score_chosen": d["score_chosen"],
            "score_rejected": d["score_rejected"],
        }

        if template == "std":
            if template_format == "implicit":
                core_data["chosen"] = d["prompt"] + " " + d["response_chosen"]
                core_data["rejected"] = d["prompt"] + " " + d["response_rejected"]
            else:
                core_data["prompt"] = d["prompt"]
                core_data["chosen"] = d["response_chosen"]
                core_data["rejected"] = d["response_rejected"]
        elif template == "chat":
            if template_format == "implicit":
                core_data["chosen"] = [
                    {"role": "user", "content": d["prompt"]},
                    {"role": "assistant", "content": d["response_chosen"]},
                ]
                core_data["rejected"] = [
                    {"role": "user", "content": d["prompt"]},
                    {"role": "assistant", "content": d["response_rejected"]},
                ]
            else:
                core_data["prompt"] = [{"role": "user", "content": d["prompt"]}]
                core_data["chosen"] = [
                    {"role": "assistant", "content": d["response_chosen"]}
                ]
                core_data["rejected"] = [
                    {"role": "assistant", "content": d["response_rejected"]}
                ]
        trl_data.append(core_data)

    return Dataset.from_generator(lambda: trl_data)


def stratified_sampling(data, num_samples=100):
    """
    Perform stratified sampling ensuring at least one sample per category.

    :param data: Dict where keys are category names and values are lists of items.
    :param num_samples: Total number of samples to draw.
    :return: List of sampled items.
    """
    N = len(data)  # Number of data
    if num_samples < N:
        raise ValueError("Number of samples must be at least the number of data.")

    # Step 1: Compute ideal allocation
    total_items = sum(len(items) for items in data.values())
    allocation = {
        cat: max(1, round(num_samples * (len(items) / total_items)))
        for cat, items in data.items()
    }

    # Step 2: Adjust allocation to sum to num_samples
    while sum(allocation.values()) > num_samples:
        max_cat = max(allocation, key=allocation.get)
        if allocation[max_cat] > 1:
            allocation[max_cat] -= 1
    while sum(allocation.values()) < num_samples:
        min_cat = min(allocation, key=allocation.get)
        allocation[min_cat] += 1

    # Step 3: Sample items based on allocation
    sampled_items = []
    for cat, count in allocation.items():
        random.shuffle(data[cat])
        sampled_items.extend(np.random.choice(data[cat], count, replace=False))

    return sampled_items


def uniform_sampling(data, num_samples=100):
    """
    Perform uniform sampling ensuring at least one sample per category.

    :param data: Dict where keys are category names and values are lists of items.
    :param num_samples: Total number of samples to draw.
    :return: List of sampled items.
    """
    # uniform sampling per category
    sampled_items = []
    sample_size = num_samples // len(data)
    for cat, items in data.items():
        random.shuffle(items)
        sampled_items.extend(
            np.random.choice(items, min(len(items), sample_size), replace=False)
        )
    return sampled_items


def sample_pref_dataset_by_task(data, sample_size=1000, sampling_method="uniform"):
    data_by_task = defaultdict(list)

    for sample in data:
        task = sample["task"]
        data_by_task[task].append(sample)

    if sampling_method == "stratified":
        sampled_data = stratified_sampling(data_by_task, sample_size)
    else:
        sampled_data = uniform_sampling(data_by_task, sample_size)

    return Dataset.from_generator(lambda: sampled_data)


# currently supports only chat-implicit template
def get_prompt_and_responses_from_trl_sample(trl_sample):
    return (
        trl_sample["chosen"][0]["content"],
        trl_sample["chosen"][1]["content"],
        trl_sample["rejected"][1]["content"],
    )
