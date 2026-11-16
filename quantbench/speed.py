"""Throughput and memory measurements."""

import time

import numpy as np
import torch

PROMPT = (
    "Explain in simple terms how a large language model generates text, "
    "and why running it on a GPU is faster than on a CPU."
)


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def decode_throughput(model, tok, new_tokens=128, warmup=2, repeats=5, device="cuda"):
    """Tokens per second when generating with batch size 1 and greedy decoding."""
    inputs = tok(PROMPT, return_tensors="pt").to(device)
    kwargs = dict(max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False,
                  pad_token_id=tok.pad_token_id)
    for _ in range(warmup):
        model.generate(**inputs, **kwargs)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rates = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        model.generate(**inputs, **kwargs)
        _sync(device)
        rates.append(new_tokens / (time.perf_counter() - t0))
    peak = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else None
    return {"decode_tok_s_mean": float(np.mean(rates)), "decode_tok_s_std": float(np.std(rates)),
            "decode_tok_s_runs": rates, "peak_mem_gb_generate": peak}


@torch.no_grad()
def prefill_throughput(model, tok, seq_len=1024, warmup=2, repeats=5, device="cuda"):
    """Tokens per second when processing a long prompt in a single forward pass."""
    vocab = model.config.vocab_size
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, min(vocab, 30000), (1, seq_len), generator=g).to(device)
    for _ in range(warmup):
        model(x)
    rates = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        model(x)
        _sync(device)
        rates.append(seq_len / (time.perf_counter() - t0))
    return {"prefill_tok_s_mean": float(np.mean(rates)), "prefill_tok_s_std": float(np.std(rates))}
