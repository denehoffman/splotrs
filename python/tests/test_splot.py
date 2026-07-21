from typing import Any, cast

import numpy as np
import pytest

from splotrs import SPlotError, SPlotResult, ShapeParameter, splot


def normal(mean: float, sigma: float):
    calls = []

    def pdf(events: np.ndarray, _parameters: dict[str, float]) -> np.ndarray:
        calls.append(events.shape)
        z = (events[:, 0] - mean) / sigma
        return np.exp(-0.5 * z**2) / (sigma * np.sqrt(2.0 * np.pi))

    pdf.calls = calls  # ty:ignore[unresolved-attribute]
    return pdf


def test_one_component_result_and_vectorized_callback():
    data = np.array([[-1.0], [0.0], [0.5], [2.0]], dtype=np.float64)
    pdf = normal(0.0, 1.0)

    result = splot(data, [pdf])

    assert isinstance(result, SPlotResult)
    assert pdf.calls == [data.shape]
    np.testing.assert_allclose(result.yields, [len(data)], atol=1e-6)
    np.testing.assert_allclose(result.sweights, np.ones((len(data), 1)), atol=1e-8)
    assert result.covariance.shape == (1, 1)
    assert result.success
    assert result.objective_evaluations > 0


def test_pdf_exception_is_preserved():
    data = np.ones((2, 1), dtype=np.float64)

    def broken(_events, _parameters):
        raise LookupError('pdf failed')

    with pytest.raises(LookupError, match='pdf failed'):
        splot(data, [broken])


def test_pdf_must_be_callable():
    data = np.ones((2, 1), dtype=np.float64)
    with pytest.raises(ValueError, match='PDF 0 must be callable'):
        splot(data, [cast(Any, object())])


def test_pdf_output_shape_is_validated():
    data = np.ones((2, 1), dtype=np.float64)
    with pytest.raises(ValueError, match='returned 1 values, expected 2'):
        splot(data, [lambda _events, _parameters: np.ones(1, dtype=np.float64)])


def test_signed_weights_enter_fit_covariance_and_sweights():
    data = np.array([[-1.0], [0.0], [1.0]], dtype=np.float64)
    weights = np.array([2.0, -1.0, 2.0])
    result = splot(data, [normal(0.0, 1.0)], weights=weights)

    np.testing.assert_allclose(result.yields, [weights.sum()], atol=1e-6)
    np.testing.assert_allclose(result.covariance, [[weights.sum()]], atol=1e-6)
    np.testing.assert_allclose(result.sweights[:, 0], weights, atol=1e-6)


def test_identical_pdfs_raise_splot_error():
    data = np.array([[-1.0], [0.0], [1.0]], dtype=np.float64)
    with pytest.raises(SPlotError, match='event-summed yield information matrix'):
        splot(data, [normal(0.0, 1.0), normal(0.0, 1.0)])


def test_shape_parameter_is_fitted_before_sweights():
    data = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]], dtype=np.float64)
    weights = np.array([1.0, 1.0, 1.0, 1.0, 5.0])
    calls = []

    def gaussian_with_mean(events, parameters):
        calls.append(parameters.copy())
        residual = events[:, 0] - parameters['mean']
        return np.exp(-0.5 * residual**2) / np.sqrt(2.0 * np.pi)

    result = splot(
        data,
        [gaussian_with_mean],
        shape_parameters=[ShapeParameter('mean', 0.7, -5.0, 5.0)],
        weights=weights,
    )

    assert result.shape_parameters['mean'] == pytest.approx(
        np.average(data[:, 0], weights=weights), abs=1e-5
    )
    assert set(result.shape_errors) == {'mean'}
    np.testing.assert_allclose(result.yields, [weights.sum()], atol=1e-6)
    np.testing.assert_allclose(result.covariance, [[weights.sum()]], atol=1e-6)
    np.testing.assert_allclose(result.sweights[:, 0], weights, atol=1e-6)
    assert result.fit_covariance.shape == (2, 2)
    assert len(calls) > 1


def test_shape_parameter_defaults_to_infinite_bounds():
    parameter = ShapeParameter('unbounded', 0.0)
    assert parameter.lower == -np.inf
    assert parameter.upper == np.inf
