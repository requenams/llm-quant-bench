import json
from pathlib import Path

import pytest

mlflow = pytest.importorskip("mlflow")

from quantbench.tracking import log_folder  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def test_log_folder_creates_one_run_per_variant(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # keep artifacts out of the repo
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    ids = log_folder(RESULTS, tracking_uri=uri, experiment="test")
    assert set(ids) >= {"fp16", "int8", "nf4", "fp4"}
    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    run = client.get_run(ids["nf4"])
    expected = json.loads((RESULTS / "nf4.json").read_text())
    assert run.data.params["variant"] == "nf4"
    assert run.data.metrics["mmlu_accuracy"] == pytest.approx(expected["mmlu"]["accuracy"])
