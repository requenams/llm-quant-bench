"""Accuracy evaluations that keep per item results, so variants can be compared pairwise."""

import numpy as np
import torch
from datasets import load_dataset
from tqdm.auto import tqdm

LETTERS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------- perplexity

def wikitext_text(split="test"):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    return "\n\n".join(ds["text"])


def calibration_texts(n=64, min_chars=500):
    """Paragraphs from the WikiText-2 train split (never from the test split)."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if len(t) >= min_chars]
    return texts[:n]


@torch.no_grad()
def window_nll(model, tok, text, seq_len=1024, max_windows=None, device="cuda"):
    """Split the text into non overlapping windows and return the summed negative
    log likelihood of each window plus its number of predicted tokens.

    Keeping one value per window lets us bootstrap perplexity afterwards, and
    every variant sees exactly the same windows.
    """
    ids = tok(text, return_tensors="pt").input_ids[0]
    n = ids.numel() // seq_len
    if max_windows:
        n = min(n, max_windows)
    nll, ntok = [], []
    for i in tqdm(range(n), desc="perplexity"):
        x = ids[i * seq_len:(i + 1) * seq_len].unsqueeze(0).to(device)
        loss = model(x, labels=x).loss.float().item()  # mean over seq_len - 1 tokens
        nll.append(loss * (seq_len - 1))
        ntok.append(seq_len - 1)
    return np.array(nll), np.array(ntok)


def perplexity(nll, ntok):
    return float(np.exp(nll.sum() / ntok.sum()))


# ---------------------------------------------------------------------- MMLU

def mmlu_sample(n=1000, seed=0):
    """Fixed random subset of the MMLU test set (same questions for every variant)."""
    ds = load_dataset("cais/mmlu", "all", split="test")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(ds), size=min(n, len(ds)), replace=False))
    return ds.select(idx.tolist()), idx


def _mmlu_prompt(row):
    subject = row["subject"].replace("_", " ")
    lines = [f"The following is a multiple choice question about {subject}.", "", row["question"].strip()]
    lines += [f"{l}. {c}" for l, c in zip(LETTERS, row["choices"])]
    lines.append("Answer:")
    return "\n".join(lines)


def _letter_token_ids(tok):
    ids = []
    for l in LETTERS:
        enc = tok.encode(" " + l, add_special_tokens=False)
        ids.append(enc[-1])
    return ids


@torch.no_grad()
def mmlu_eval(model, tok, rows, device="cuda"):
    """Zero shot MMLU scored by comparing the logits of the answer letters.

    Returns per question correctness and the probability given to the right answer.
    """
    letter_ids = _letter_token_ids(tok)
    correct, p_right, preds = [], [], []
    for row in tqdm(rows, desc="mmlu"):
        x = tok(_mmlu_prompt(row), return_tensors="pt").input_ids.to(device)
        logits = model(x).logits[0, -1].float()
        probs = torch.softmax(logits[letter_ids], dim=-1)
        pred = int(torch.argmax(probs))
        preds.append(pred)
        correct.append(pred == int(row["answer"]))
        p_right.append(float(probs[int(row["answer"])]))
    return np.array(correct, dtype=bool), np.array(p_right), np.array(preds)
