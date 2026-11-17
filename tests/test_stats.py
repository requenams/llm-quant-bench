import numpy as np
import pytest

from quantbench import stats


def test_mcnemar_known_case():
    # 10 questions only the baseline gets right, 2 only the quantized model gets right
    base = np.array([1] * 10 + [0] * 2 + [1] * 50 + [0] * 38, bool)
    quant = np.array([0] * 10 + [1] * 2 + [1] * 50 + [0] * 38, bool)
    p, b, c = stats.mcnemar_exact(base, quant)
    assert (b, c) == (10, 2)
    assert p == pytest.approx(0.03857, abs=1e-4)


def test_mcnemar_identical_answers():
    x = np.array([1, 0, 1, 1, 0], bool)
    assert stats.mcnemar_exact(x, x) == (1.0, 0, 0)


def test_paired_diff_ci_contains_point_and_is_negative():
    base = np.array([1] * 10 + [0] * 2 + [1] * 50 + [0] * 38, bool)
    quant = np.array([0] * 10 + [1] * 2 + [1] * 50 + [0] * 38, bool)
    d, lo, hi = stats.paired_accuracy_diff(base, quant)
    assert d == pytest.approx(-0.08)
    assert lo <= d <= hi
    assert hi < 0


def test_perplexity_ratio_identity():
    nll = np.array([100.0, 120.0, 90.0])
    ntok = np.array([50, 50, 50])
    ratio, lo, hi = stats.perplexity_ratio_ci(nll, nll, ntok)
    assert ratio == pytest.approx(1.0)
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)


def test_perplexity_ratio_uniform_shift():
    base = np.array([100.0, 120.0, 90.0])
    ntok = np.array([50, 50, 50])
    quant = base + 0.1 * ntok  # +0.1 nats per token everywhere
    ratio, lo, hi = stats.perplexity_ratio_ci(base, quant, ntok)
    assert ratio == pytest.approx(np.exp(0.1))
    assert lo == pytest.approx(np.exp(0.1)) and hi == pytest.approx(np.exp(0.1))
