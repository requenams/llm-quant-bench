"""Optional experiment tracking with MLflow.

Every benchmark run becomes an MLflow run with its configuration as parameters,
the headline numbers as metrics and the raw per item results as an artifact.
"""

import json
from pathlib import Path

EXPERIMENT = "llm-quant-bench"
DEFAULT_URI = "sqlite:///mlflow.db"  # local database; view it with `mlflow ui --backend-store-uri sqlite:///mlflow.db`


def _flatten_metrics(res):
    m = {
        "footprint_gb": res["footprint_gb"],
        "load_seconds": res["load_seconds"],
        "perplexity": res["perplexity"]["value"],
        "mmlu_accuracy": res["mmlu"]["accuracy"],
    }
    speed = res.get("speed") or {}
    for key in ("decode_tok_s_mean", "decode_tok_s_std", "prefill_tok_s_mean",
                "prefill_tok_s_std", "peak_mem_gb_generate"):
        if speed.get(key) is not None:
            m[key] = speed[key]
    return m


def log_result(res, json_path=None, tracking_uri=None, experiment=EXPERIMENT):
    """Log one result dict (as written by run_benchmark.py) to MLflow."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri or DEFAULT_URI)
    mlflow.set_experiment(experiment)

    cfg = res.get("config", {})
    params = {
        "model": res["model"],
        "variant": res["variant"],
        "method": res.get("description", ""),
        "simulated_quantization": res.get("simulated_quantization", False),
        "device": res.get("device", ""),
        "n_mmlu": cfg.get("n_mmlu"),
        "ppl_windows": cfg.get("ppl_windows"),
        "seq_len": cfg.get("seq_len"),
        "new_tokens": cfg.get("new_tokens"),
        "seed": cfg.get("seed"),
        **{f"version_{k}": v for k, v in res.get("versions", {}).items()},
    }

    with mlflow.start_run(run_name=f"{res['model'].split('/')[-1]}-{res['variant']}") as run:
        mlflow.log_params({k: v for k, v in params.items() if v is not None})
        mlflow.log_metrics(_flatten_metrics(res))
        mlflow.set_tags({"variant": res["variant"], "model": res["model"]})
        if json_path:
            mlflow.log_artifact(str(json_path), artifact_path="raw")
        return run.info.run_id


def log_folder(folder="results", tracking_uri=None, experiment=EXPERIMENT):
    """Log every JSON result in a folder. Useful to track runs made on Colab or Kaggle."""
    ids = {}
    for p in sorted(Path(folder).glob("*.json")):
        res = json.loads(p.read_text())
        ids[res["variant"]] = log_result(res, p, tracking_uri, experiment)
    return ids
