import spacy
import string
from collections import Counter
from statistics import mean
import re
from tqdm import tqdm
from functools import partial
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
import warnings

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import (
    cos_sim,
    dot_score,
    euclidean_sim,
    manhattan_sim,
    pairwise_cos_sim,
    pairwise_dot_score,
    pairwise_euclidean_sim,
    pairwise_manhattan_sim,
)
from sklearn.cluster import AgglomerativeClustering

from crpo.utils import cache, load_lm, load_rm

DEF_EMB_MODEL = "jinaai/jina-embeddings-v3"
DEF_EMB_TYPE = "sentence_embedding"
DEF_EMB_STRATEGY = "direct"
DEF_DIST_FN = "cosine"
DEF_SPACY_LANG = "en_core_web_sm"
DEF_PREPROCESSING_ARGS = {
    "lower": True,
    "remove_punct": True,
    "remove_stopwords": False,
    "lemmatize": False,
    "dominant_k": None,
    "unique": True,
    "no_spacy": False,
}

SPACY_ENGINE_CACHE = {}
EMB_MODEL_CACHE = {}
EMBEDDING_CACHE = {}
SPACY_CACHE = {}
DEF_PERPLEXITY_MODEL = "google/gemma-2-27b-it"
DEF_QUALITY_MODEL = "Skywork/Skywork-Reward-Gemma-2-27B-v0.2"


@cache(cache_dict=SPACY_ENGINE_CACHE)
def load_spacy_engine(
    language=DEF_SPACY_LANG, include_syllables=False, include_constituency=False
):
    print(f"Loading spacy engine: {language}")
    engine = spacy.load(language)
    if include_syllables:
        from spacy_syllables import SpacySyllables

        engine.add_pipe("syllables", after="tagger")
    if include_constituency:
        import benepar

        engine.add_pipe("benepar", config={"model": "benepar_en3"})
    return engine


@cache(cache_dict=EMB_MODEL_CACHE)
def load_emb_model(model=DEF_EMB_MODEL):
    print(f"Loading embedding model: {model}")
    return SentenceTransformer(model, trust_remote_code=True)


@cache(cache_dict=EMBEDDING_CACHE)
def get_embedding(text, model=DEF_EMB_MODEL, emb_type=DEF_EMB_TYPE):
    emb_model = load_emb_model(model)

    output_value = "sentence_embedding"

    if "token" in emb_type:
        output_value = "token_embeddings"

    embeddings = emb_model.encode(text, output_value=output_value)

    if embeddings.ndim == 2:
        embeddings = embeddings.mean(axis=0)

    return embeddings


@cache(cache_dict=SPACY_CACHE)
def get_spacy_doc(text, include_syllables=False, include_constituency=False):
    spacy_engine = load_spacy_engine(
        include_syllables=include_syllables, include_constituency=include_constituency
    )
    return spacy_engine(text)


def compute_sem_dis(emb1, emb2, distance_fn=DEF_DIST_FN):
    if distance_fn == "cosine":
        return (1 - cos_sim(emb1, emb2)).item()
    elif distance_fn == "dot":
        return (1 - dot_score(emb1, emb2)).item()
    elif distance_fn == "euclidean":
        return (-euclidean_sim(emb1, emb2)).item()
    elif distance_fn == "manhattan":
        return (-manhattan_sim(emb1, emb2)).item()
    else:
        raise ValueError(f"Invalid distance function: {distance_fn}")


def compute_pairwise_sem_dis(embs1, embs2, distance_fn=DEF_DIST_FN):
    if distance_fn == "cosine":
        return (1 - pairwise_cos_sim(np.array(embs1), np.array(embs2))).tolist()
    elif distance_fn == "dot":
        return (1 - pairwise_dot_score(np.array(embs1), np.array(embs2))).tolist()
    elif distance_fn == "euclidean":
        return (-pairwise_euclidean_sim(np.array(embs1), np.array(embs2))).tolist()
    elif distance_fn == "manhattan":
        return (-pairwise_manhattan_sim(np.array(embs1), np.array(embs2))).tolist()
    else:
        raise ValueError(f"Invalid distance function: {distance_fn}")


def get_sentences(text):
    doc = get_spacy_doc(text)
    return [sent.text for sent in doc.sents]


def get_words(
    text,
    no_spacy=False,
    lower=True,
    remove_punct=True,
    remove_stopwords=True,
    lemmatize=True,
    unique=True,
    dominant_k=None,
):
    if no_spacy:
        # default to a naive word tokenizer that splits on spaces, best we can do if spacy doesn't support the language
        words = text.split(" ")
        if remove_punct:
            words = [
                w.translate(str.maketrans("", "", string.punctuation)) for w in words
            ]
        if lower:
            words = [w.lower() for w in words]
    else:
        doc = get_spacy_doc(text)
        tokens = [token for token in doc]

        if remove_punct:
            tokens = [token for token in tokens if not token.is_punct]

        if remove_stopwords:
            tokens = [token for token in tokens if not token.is_stop]

        words = [token.text for token in tokens]

        if lemmatize:
            words = [token.lemma_ for token in tokens]

        if lower:
            words = [word.lower() for word in words]

    if dominant_k is None or dominant_k == 0 or dominant_k >= len(words):
        if unique:
            return list(set(words))
        return words

    word_freq = Counter(words)

    return [w[0] for w in word_freq.most_common(dominant_k)]


def get_syllables(text):
    doc = get_spacy_doc(text, include_syllables=True)
    return [
        (token._.syllables, token._.syllables_count)
        for token in doc
        if token._.syllables_count
    ]


def get_pos_tags(text, remove_punct=True):
    doc = get_spacy_doc(text)
    return [token.pos_ for token in doc if not (remove_punct and token.is_punct)]


def compute_text_embedding(
    text,
    emb_model=DEF_EMB_MODEL,
    emb_type=DEF_EMB_TYPE,
    emb_strategy=DEF_EMB_STRATEGY,
    preprocessing_args=DEF_PREPROCESSING_ARGS,
):
    if emb_strategy == "direct":
        return get_embedding(text, emb_model, emb_type)
    elif emb_strategy == "by_word":
        words = get_words(text, **preprocessing_args)
        return get_embedding(words, emb_model, emb_type)
    elif emb_strategy == "by_sentence":
        sentences = get_sentences(text)
        return get_embedding(sentences, emb_model, emb_type)
    else:
        raise ValueError(f"Invalid embedding strategy: {emb_strategy}")


def compute_avg_pairwise_distances(embeddings, distance_fn=DEF_DIST_FN):
    if len(embeddings) <= 1:
        return [0]

    avg_pairwise_distances = []

    distance_cache = {}

    for i in tqdm(range(len(embeddings)), desc="Computing average pairwise distances"):
        cached_pairs = [
            (i, j) for j in range(len(embeddings)) if (i, j) in distance_cache
        ]
        cached_pairs = cached_pairs + [
            (j, i) for j in range(len(embeddings)) if (j, i) in distance_cache
        ]
        pairs = [
            (i, j)
            for j in range(len(embeddings))
            if i != j and (i, j) not in distance_cache and (j, i) not in distance_cache
        ]

        pairwise_distances = []

        if pairs:
            embs1 = [embeddings[pair[0]] for pair in pairs]
            embs2 = [embeddings[pair[1]] for pair in pairs]
            pairwise_distances = compute_pairwise_sem_dis(embs1, embs2, distance_fn)

            for pair, distance in zip(pairs, pairwise_distances):
                distance_cache[pair] = distance

        pairwise_distances = pairwise_distances + [
            distance_cache[pair] for pair in cached_pairs
        ]
        avg_pairwise_distances.append(mean(pairwise_distances))

    return avg_pairwise_distances


def compute_avg_sem_dis(
    text,
    emb_model=DEF_EMB_MODEL,
    emb_type=DEF_EMB_TYPE,
    distance_fn=DEF_DIST_FN,
    preprocessing_args=DEF_PREPROCESSING_ARGS,
):
    words = get_words(text, **preprocessing_args)
    embeddings = [
        get_embedding(word, emb_model, emb_type)
        for word in tqdm(words, desc="Computing word embeddings")
    ]
    return mean(compute_avg_pairwise_distances(embeddings, distance_fn))


def compute_inverse_homogenization(
    texts,
    emb_model=DEF_EMB_MODEL,
    emb_type=DEF_EMB_TYPE,
    emb_strategy=DEF_EMB_STRATEGY,
    distance_fn=DEF_DIST_FN,
    preprocessing_args=DEF_PREPROCESSING_ARGS,
):
    text_embeddings = [
        compute_text_embedding(
            text,
            emb_model=emb_model,
            emb_type=emb_type,
            emb_strategy=emb_strategy,
            preprocessing_args=preprocessing_args,
        )
        for text in tqdm(texts, desc="Computing text embeddings")
    ]
    return compute_avg_pairwise_distances(text_embeddings, distance_fn)


def compute_novelty(
    texts,
    emb_model=DEF_EMB_MODEL,
    emb_type=DEF_EMB_TYPE,
    distance_fn=DEF_DIST_FN,
    preprocessing_args=DEF_PREPROCESSING_ARGS,
):
    corpus = " ".join(texts)
    corpus_avg_sem_dis = compute_avg_sem_dis(
        corpus, emb_model, emb_type, distance_fn, preprocessing_args
    )

    novelty_scores = []

    for text in texts:
        text_avg_sem_dis = compute_avg_sem_dis(
            text, emb_model, emb_type, distance_fn, preprocessing_args
        )
        novelty_scores.append(2 * abs(text_avg_sem_dis - corpus_avg_sem_dis))

    return novelty_scores


def compute_theme_uniqueness(
    texts,
    emb_model=DEF_EMB_MODEL,
    emb_type=DEF_EMB_TYPE,
    emb_strategy=DEF_EMB_STRATEGY,
    cluster_linkage="ward",
    cluster_distance_threshold=0.5,
    preprocessing_args=DEF_PREPROCESSING_ARGS,
):
    text_embeddings = [
        compute_text_embedding(
            text,
            emb_model=emb_model,
            emb_type=emb_type,
            emb_strategy=emb_strategy,
            preprocessing_args=preprocessing_args,
        )
        for text in tqdm(texts, desc="Computing text embeddings")
    ]
    clustering = AgglomerativeClustering(
        n_clusters=None,
        linkage=cluster_linkage,
        distance_threshold=cluster_distance_threshold,
    )
    cluster_labels = clustering.fit_predict(text_embeddings)
    cluster_freq = Counter(cluster_labels)
    return [1 / cluster_freq[cluster_labels[i]] for i in range(len(texts))], [
        int(label) for label in cluster_labels
    ]


def compute_dsi(
    text,
    emb_model=DEF_EMB_MODEL,
    emb_type=DEF_EMB_TYPE,
    distance_fn=DEF_DIST_FN,
    preprocessing_args=DEF_PREPROCESSING_ARGS,
):
    return compute_avg_sem_dis(
        text, emb_model, emb_type, distance_fn, preprocessing_args
    )


def compute_n_gram_diversity(
    text,
    max_n_gram=5,
    remove_punct=True,
    no_spacy=False,
):
    words = get_words(
        text,
        no_spacy,
        lower=True,
        remove_punct=remove_punct,
        remove_stopwords=False,
        lemmatize=False,
        unique=False,
    )
    all_n_grams = []

    for n in range(1, max_n_gram + 1):
        all_n_grams.append([tuple(words[i : i + n]) for i in range(len(words) - n + 1)])

    all_n_gram_freqs = [Counter(n_grams) for n_grams in all_n_grams]
    n_gram_diversity = [
        len(n_gram_freqs) / len(n_grams)
        for n_grams, n_gram_freqs in zip(all_n_grams, all_n_gram_freqs)
        if n_grams
    ]

    return n_gram_diversity, all_n_gram_freqs


def compute_pos_diversity(text, max_n_gram=5, remove_punct=True):
    pos_tags = get_pos_tags(text, remove_punct=remove_punct)
    all_pos_tags = []

    for n in range(1, max_n_gram + 1):
        all_pos_tags.append(
            [tuple(pos_tags[i : i + n]) for i in range(len(pos_tags) - n + 1)]
        )

    all_pos_tag_freqs = [Counter(p_tags) for p_tags in all_pos_tags]
    pos_tag_diversity = [
        len(pos_freqs) / len(p_tags)
        for p_tags, pos_freqs in zip(all_pos_tags, all_pos_tag_freqs)
        if p_tags
    ]

    return pos_tag_diversity, all_pos_tag_freqs


def _get_dep_paths(token):
    if not list(token.children):
        return [(token.dep_,)]
    paths = []
    for child in token.children:
        child_paths = _get_dep_paths(child)
        for path in child_paths:
            paths.append((token.dep_,) + path)
    return paths


def compute_dependency_complexity(text):
    # see labels https://github.com/clir/clearnlp-guidelines/blob/master/md/specifications/dependency_labels.md
    # see https://euroslajournal.org/articles/10.22599/jesla.63
    clause_labels = ["conj", "cc", "preconj", "ccomp", "xcomp", "acl", "relcl", "advcl"]
    dep_num_clauses = []
    dep_paths = []
    sentences = get_sentences(text)

    for sentence in sentences:
        doc = get_spacy_doc(sentence)
        num_clauses = 0
        sent_path_counter = Counter()
        for token in doc:
            paths = _get_dep_paths(token)
            sent_path_counter.update(paths)
            if token.dep_ in clause_labels:
                num_clauses += 1
        dep_num_clauses.append(num_clauses)
        dep_paths.append(sent_path_counter)

    return dep_paths, dep_num_clauses


def compute_pos_complexity(text):
    sentences = get_sentences(text)
    pos_complexity = {
        "NOUN": [],
        "VERB": [],
        "ADJ": [],
        "ADV": [],
        "PRON": [],
        "DET": [],
        "ADP": [],
    }

    for sentence in sentences:
        pos_tags = get_pos_tags(sentence)
        pos_freq = Counter(pos_tags)
        num_words = len(pos_tags)
        if num_words > 0:
            for pos in pos_complexity:
                pos_complexity[pos].append(pos_freq.get(pos, 0) / num_words)

    return pos_complexity


def compute_flesch_readability_scores(text, no_spacy=False):
    # see https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
    num_words = len(
        get_words(
            text,
            no_spacy,
            lower=True,
            remove_punct=False,
            remove_stopwords=False,
            lemmatize=False,
            unique=False,
        )
    )
    num_sentences = len(get_sentences(text))
    num_syllables = sum([n_syllables for _, n_syllables in get_syllables(text)])
    flesch_ease = (
        206.835
        - 1.015 * (num_words / num_sentences)
        - 84.6 * (num_syllables / num_words)
    )
    flesch_kincaid = (
        0.39 * (num_words / num_sentences) + 11.8 * (num_syllables / num_words) - 15.59
    )
    return flesch_ease, flesch_kincaid


def compute_constituency_complexity(text):
    text = re.sub(r"\s+", " ", text)

    def _get_height(sent):
        if not list(sent._.children):
            return 1
        return 1 + max([_get_height(child) for child in sent._.children])

    try:
        doc = get_spacy_doc(text, include_constituency=True)
    except ValueError as e:
        text_len = len(text.split())
        warnings.warn(
            "Text is too long for benepar model (over 512 tokens), truncating to half. This may affect the results.",
        )
        text = " ".join(text.split()[: (text_len // 2)])
        return compute_constituency_complexity(text)

    return [_get_height(sent) for sent in doc.sents]


# adapted from https://huggingface.co/spaces/evaluate-metric/perplexity/blob/main/perplexity.py
def compute_perplexity(
    predictions,
    model=DEF_PERPLEXITY_MODEL,
    tokenizer=None,
    batch_size: int = 16,
    add_start_token: bool = True,
    device=None,
    max_length=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(model, str):
        model, tokenizer = load_lm(model, device=device)

    # if batch_size > 1 (which generally leads to padding being required), and
    # if there is not an already assigned pad_token, assign an existing
    # special token to also be the padding token
    if tokenizer.pad_token is None and batch_size > 1:
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        # check that the model already has at least one special token defined
        assert (
            len(existing_special_tokens) > 0
        ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
        # assign one of the special tokens to also be the pad token
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    if add_start_token and max_length:
        # leave room for <BOS> token to be added:
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    if tokenizer.chat_template:
        predictions = [
            tokenizer.apply_chat_template(p, tokenize=False) for p in predictions
        ]

    encodings = tokenizer(
        predictions,
        add_special_tokens=False,
        padding=True,
        truncation=True if max_tokenized_len else False,
        max_length=max_tokenized_len,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    # check that each input is long enough:
    if add_start_token:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 1)
        ), "Each input text must be at least one token long."
    else:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 2)
        ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")

    for start_index in tqdm(range(0, len(encoded_texts), batch_size)):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

        if add_start_token:
            bos_tokens_tensor = torch.tensor(
                [[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)
            ).to(device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [
                    torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device),
                    attn_mask,
                ],
                dim=1,
            )

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        perplexity_batch = torch.exp(
            (
                loss_fct(shift_logits.transpose(1, 2), shift_labels)
                * shift_attention_mask_batch
            ).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        ppls += perplexity_batch.tolist()

    return ppls


# only supports instruction tuned reward models
def compute_quality(
    predictions,
    model=DEF_QUALITY_MODEL,
    tokenizer=None,
    batch_size: int = 8,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(model, str):
        model, tokenizer = load_rm(model, device=device)

    batches = [
        predictions[i : i + batch_size] for i in range(0, len(predictions), batch_size)
    ]

    quality_scores = []

    for batch in tqdm(batches, desc="Computing quality", unit="batch"):
        batch_formatted = [
            tokenizer.apply_chat_template(conv, tokenize=False) for conv in batch
        ]
        batch_tokenized = tokenizer(
            batch_formatted, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        with torch.no_grad():
            batch_scores = model(**batch_tokenized).logits[:, 0].cpu()

        quality_scores += batch_scores.tolist()

    return quality_scores
