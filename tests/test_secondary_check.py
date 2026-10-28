"""Tests for src/secondary_check.py (final value vs average of the last k)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import secondary_check as sc  # noqa: E402


def make_curves(*seed_values):
    """One curve per seed; each curve is a list of {'mae': value} points."""
    return {str(i): [{"mae": v} for v in values] for i, values in enumerate(seed_values)}


def test_final_value_uses_the_last_entry():
    curves = make_curves([10, 20, 30], [20, 40, 50])
    result = sc.final_value(curves, "mae")
    assert result["mean"] == pytest.approx(40.0)   # mean of 30 and 50
    assert result["std"] == pytest.approx(14.142135, rel=1e-5)  # sample std of 30, 50


def test_last_k_mean_averages_only_the_last_k():
    # First two points are huge and must be ignored when k = 3.
    curves = make_curves([1000, 1000, 10, 20, 30], [1000, 1000, 20, 30, 40])
    result = sc.last_k_mean(curves, "mae", k=3)
    assert result["mean"] == pytest.approx(25.0)   # seed means are 20 and 30


def test_last_k_mean_works_when_curve_is_shorter_than_k():
    curves = make_curves([10, 20], [30, 40])
    result = sc.last_k_mean(curves, "mae", k=10)
    assert result["mean"] == pytest.approx(25.0)   # seed means are 15 and 35
