"""Compare every variant against the baseline and produce a table and two plots.

Example:
    python analyze.py --baseline fp16
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quantbench import stats

ORDER = ["fp32", "bf16", "fp16", "int8", "mo_int8_sq", "nf4", "fp4", "mo_int4_awq"]


def load_results(folder):
    res = {}
    for p in Path(folder).glob("*.json"):
        r = json.loads(p.read_text())
        res[r["variant"]] = r
    return dict(sorted(res.items(), key=lambda kv: ORDER.index(kv[0]) if kv[0] in ORDER else 99))


def check_paired(base, other):
    if base["mmlu"]["question_index"] != other["mmlu"]["question_index"]:
        raise ValueError(f"{other['variant']} was evaluated on different MMLU questions")
    if base["perplexity"]["window_tokens"] != other["perplexity"]["window_tokens"]:
        raise ValueError(f"{other['variant']} was evaluated on different text windows")


def build_rows(results, baseline):
    base = results[baseline]
    rows = []
    for name, r in results.items():
        check_paired(base, r)
        acc, acc_lo, acc_hi = stats.bootstrap_mean_ci(r["mmlu"]["correct"])
        row = {
            "variant": name,
            "simulated": r["simulated_quantization"],
            "footprint_gb": r["footprint_gb"],
            "decode": r.get("speed", {}).get("decode_tok_s_mean"),
            "prefill": r.get("speed", {}).get("prefill_tok_s_mean"),
            "ppl": r["perplexity"]["value"],
            "acc": acc, "acc_lo": acc_lo, "acc_hi": acc_hi,
        }
        if name != baseline:
            ratio, lo, hi = stats.perplexity_ratio_ci(
                base["perplexity"]["window_nll"], r["perplexity"]["window_nll"],
                base["perplexity"]["window_tokens"])
            d, dlo, dhi = stats.paired_accuracy_diff(base["mmlu"]["correct"], r["mmlu"]["correct"])
            p, b, c = stats.mcnemar_exact(base["mmlu"]["correct"], r["mmlu"]["correct"])
            row.update(ppl_change=(ratio - 1, lo - 1, hi - 1), acc_diff=(d, dlo, dhi),
                       mcnemar_p=p, flips=(b, c))
        rows.append(row)
    return rows


def fmt_pct(x):
    return f"{100 * x:+.2f}%"


def to_markdown(rows, baseline, model):
    head = ("| Variant | Memory (GB) | Decode (tok/s) | Prefill (tok/s) | Perplexity | "
            "PPL change [95% CI] | MMLU acc [95% CI] | Acc change [95% CI] | McNemar p |")
    sep = "|" + "---|" * 9
    lines = [f"Model: `{model}` · baseline: `{baseline}`", "", head, sep]
    for r in rows:
        speed_d = f"{r['decode']:.1f}" if r["decode"] else "n/a"
        speed_p = f"{r['prefill']:.0f}" if r["prefill"] else "n/a"
        mem = "n/a (sim.)" if r["simulated"] else f"{r['footprint_gb']:.2f}"
        acc = f"{100*r['acc']:.1f}% [{100*r['acc_lo']:.1f}, {100*r['acc_hi']:.1f}]"
        if r["variant"] == baseline:
            ppl_c, acc_c, p = "baseline", "baseline", ""
        else:
            c, lo, hi = r["ppl_change"]
            ppl_c = f"{fmt_pct(c)} [{fmt_pct(lo)}, {fmt_pct(hi)}]"
            d, dlo, dhi = r["acc_diff"]
            acc_c = f"{100*d:+.1f} pp [{100*dlo:+.1f}, {100*dhi:+.1f}]"
            p = f"{r['mcnemar_p']:.3f}"
        name = r["variant"] + (" *" if r["simulated"] else "")
        lines.append(f"| {name} | {mem} | {speed_d} | {speed_p} | {r['ppl']:.3f} | {ppl_c} | {acc} | {acc_c} | {p} |")
    if any(r["simulated"] for r in rows):
        lines += ["", "\\* Simulated quantization (NVIDIA Model Optimizer): accuracy is realistic, "
                      "memory and speed are not measured."]
    return "\n".join(lines)


def plot(rows, out_dir):
    real = [r for r in rows if not r["simulated"]]
    for key, label, fname in [("footprint_gb", "Model memory (GB)", "accuracy_vs_memory.png"),
                              ("decode", "Decode throughput (tokens/s)", "accuracy_vs_speed.png")]:
        pts = [r for r in real if r.get(key)]
        if not pts:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for r in pts:
            yerr = [[100 * (r["acc"] - r["acc_lo"])], [100 * (r["acc_hi"] - r["acc"])]]
            ax.errorbar(r[key], 100 * r["acc"], yerr=yerr, fmt="o", capsize=4)
            ax.annotate(r["variant"], (r[key], 100 * r["acc"]), textcoords="offset points", xytext=(6, 4))
        ax.set_xlabel(label)
        ax.set_ylabel("MMLU accuracy (%)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(Path(out_dir) / fname, dpi=150)
        plt.close(fig)


def plot_paired(rows, baseline, out_dir):
    """Paired changes against the baseline, which is what actually tells real losses from noise."""
    q = [r for r in rows if r["variant"] != baseline]
    if not q:
        return
    names = [r["variant"] for r in q]
    y = np.arange(len(q))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.8 * len(q) + 1.6), sharey=True)
    for ax, key, scale, label in [(axes[0], "acc_diff", 100, "MMLU accuracy change (pp)"),
                                  (axes[1], "ppl_change", 100, "Perplexity change (%)")]:
        pts = np.array([r[key] for r in q]) * scale
        err = [pts[:, 0] - pts[:, 1], pts[:, 2] - pts[:, 0]]
        ax.errorbar(pts[:, 0], y, xerr=err, fmt="o", capsize=4)
        ax.axvline(0, color="grey", lw=1, ls="--")
        ax.set_xlabel(label)
        ax.grid(alpha=0.3, axis="x")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    fig.suptitle(f"Change against {baseline} (paired 95% CI)")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "paired_differences.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--baseline", default="fp16")
    args = ap.parse_args()

    results = load_results(args.results)
    if args.baseline not in results:
        raise SystemExit(f"Baseline {args.baseline} not found in {args.results}")
    rows = build_rows(results, args.baseline)
    md = to_markdown(rows, args.baseline, results[args.baseline]["model"])
    (Path(args.results) / "summary.md").write_text(md + "\n")
    plot(rows, args.results)
    plot_paired(rows, args.baseline, args.results)
    print(md)


if __name__ == "__main__":
    main()
