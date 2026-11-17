"""Benchmark one precision variant of a model and save the raw results.

Example:
    python run_benchmark.py --variant fp16
    python run_benchmark.py --variant nf4
"""

import argparse
import gc
import json
import platform
import time
from pathlib import Path

import torch
import transformers

from quantbench import evals, speed
from quantbench.models import ALL_VARIANTS, default_device, load_model, memory_footprint_gb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--variant", required=True, choices=list(ALL_VARIANTS))
    ap.add_argument("--n-mmlu", type=int, default=1000)
    ap.add_argument("--ppl-windows", type=int, default=80)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-speed", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--mlflow", action="store_true", help="also log the run to MLflow")
    args = ap.parse_args()

    device = default_device()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model {args.model} | variant {args.variant} | device {device}")

    calib = evals.calibration_texts() if args.variant.startswith("mo_") else None
    t0 = time.time()
    model, tok, simulated = load_model(args.model, args.variant, device, calib)
    load_s = time.time() - t0

    res = {
        "model": args.model,
        "variant": args.variant,
        "description": ALL_VARIANTS[args.variant],
        "simulated_quantization": simulated,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor() or "cpu",
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "config": vars(args),
        "load_seconds": load_s,
        "footprint_gb": memory_footprint_gb(model),
    }

    if not args.skip_speed and not simulated:
        res["speed"] = {
            **speed.decode_throughput(model, tok, args.new_tokens, device=device),
            **speed.prefill_throughput(model, tok, args.seq_len, device=device),
        }

    nll, ntok = evals.window_nll(model, tok, evals.wikitext_text(), args.seq_len, args.ppl_windows, device)
    res["perplexity"] = {"value": evals.perplexity(nll, ntok), "window_nll": nll.tolist(),
                         "window_tokens": ntok.tolist()}

    rows, idx = evals.mmlu_sample(args.n_mmlu, args.seed)
    correct, p_right, preds = evals.mmlu_eval(model, tok, rows, device)
    res["mmlu"] = {"accuracy": float(correct.mean()), "question_index": idx.tolist(),
                   "subject": list(rows["subject"]), "correct": correct.tolist(),
                   "p_right": p_right.tolist(), "pred": preds.tolist()}

    path = out_dir / f"{args.variant}.json"
    path.write_text(json.dumps(res, indent=1))
    print(f"Saved {path}")
    if args.mlflow:
        from quantbench.tracking import log_result
        print(f"  MLflow run {log_result(res, path)}")
    print(f"  footprint {res['footprint_gb']:.2f} GB | PPL {res['perplexity']['value']:.3f} | "
          f"MMLU {res['mmlu']['accuracy']:.3f}")
    if "speed" in res:
        print(f"  decode {res['speed']['decode_tok_s_mean']:.1f} tok/s | "
              f"prefill {res['speed']['prefill_tok_s_mean']:.0f} tok/s")

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
