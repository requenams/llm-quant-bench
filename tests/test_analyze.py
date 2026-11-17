import copy
import json
from pathlib import Path

import pytest

import analyze

RESULTS = Path(__file__).resolve().parents[1] / "results"


@pytest.fixture
def results():
    return analyze.load_results(RESULTS)


def test_results_are_loaded_in_order(results):
    assert list(results)[0] == "fp16"
    assert {"fp16", "int8", "nf4", "fp4"} <= set(results)


def test_rows_have_paired_comparisons(results):
    rows = {r["variant"]: r for r in analyze.build_rows(results, "fp16")}
    assert "acc_diff" not in rows["fp16"]
    for v in ("int8", "nf4", "fp4"):
        d, lo, hi = rows[v]["acc_diff"]
        assert lo <= d <= hi
        assert 0 <= rows[v]["mcnemar_p"] <= 1


def test_mismatched_questions_are_rejected(results):
    broken = copy.deepcopy(results)
    broken["int8"]["mmlu"]["question_index"][0] += 1
    with pytest.raises(ValueError):
        analyze.build_rows(broken, "fp16")


def test_markdown_and_plots(results, tmp_path):
    rows = analyze.build_rows(results, "fp16")
    md = analyze.to_markdown(rows, "fp16", results["fp16"]["model"])
    assert "| fp16 |" in md and "McNemar" in md
    analyze.plot(rows, tmp_path)
    analyze.plot_paired(rows, "fp16", tmp_path)
    for name in ("accuracy_vs_memory.png", "accuracy_vs_speed.png", "paired_differences.png"):
        assert (tmp_path / name).stat().st_size > 0
