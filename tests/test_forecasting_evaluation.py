"""Tests for forecasting evaluation metrics."""

import pandas as pd
import pytest

from vectramark.domain.forecasting.evaluation import evaluate_forecast


def test_evaluation_metrics_are_calculated_correctly() -> None:
    actual = pd.Series([100.0, 200.0])
    predicted = pd.Series([90.0, 220.0])

    result = evaluate_forecast(actual, predicted)

    assert result["mae"] == pytest.approx(15.0)
    assert result["rmse"] == pytest.approx((250.0) ** 0.5)
    assert result["mape"] == pytest.approx(10.0)


def test_unequal_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        evaluate_forecast(
            pd.Series([100.0, 200.0]),
            pd.Series([110.0]),
        )


def test_zero_actual_prices_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        evaluate_forecast(
            pd.Series([100.0, 0.0]),
            pd.Series([105.0, 110.0]),
        )


def test_non_finite_predictions_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        evaluate_forecast(
            pd.Series([100.0, 110.0]),
            pd.Series([105.0, float("nan")]),
        )


def test_empty_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluate_forecast(pd.Series(dtype=float), pd.Series(dtype=float))
