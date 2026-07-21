# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib>=3.8",
#     "numpy>=1.26",
#     "splotrs",
# ]
#
# [tool.uv.sources]
# splotrs = { path = ".." }
# ///

"""Fit a bump in one variable and use sWeights to reveal species in another."""

from math import erf, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from splotrs import ShapeParameter, splot

LOWER = -5.0
UPPER = 5.0


def truncated_gaussian(events: np.ndarray, parameters: dict[str, float]) -> np.ndarray:
    mean = parameters['mean']
    sigma = parameters['sigma']
    z = (events[:, 0] - mean) / sigma
    integral = 0.5 * (
        erf((UPPER - mean) / (sigma * sqrt(2.0))) - erf((LOWER - mean) / (sigma * sqrt(2.0)))
    )
    return np.exp(-0.5 * z**2) / (sigma * np.sqrt(2.0 * np.pi) * integral)


def quadratic_background(events: np.ndarray, parameters: dict[str, float]) -> np.ndarray:
    linear = parameters['background_linear']
    quadratic = parameters['background_quadratic']
    coordinate = 2.0 * (events[:, 0] - LOWER) / (UPPER - LOWER) - 1.0
    legendre_2 = 0.5 * (3.0 * coordinate**2 - 1.0)
    return (1.0 + linear * coordinate + quadratic * legendre_2) / (UPPER - LOWER)


def sample_background(
    rng: np.random.Generator, size: int, linear: float, quadratic: float
) -> np.ndarray:
    samples = []
    envelope = 1.0 + abs(linear) + abs(quadratic)
    while len(samples) < size:
        proposals = rng.uniform(LOWER, UPPER, 2 * (size - len(samples)))
        coordinate = 2.0 * (proposals - LOWER) / (UPPER - LOWER) - 1.0
        shape = 1.0 + linear * coordinate + quadratic * 0.5 * (3.0 * coordinate**2 - 1.0)
        accepted = proposals[rng.uniform(0.0, envelope, len(proposals)) < shape]
        samples.extend(accepted.tolist())
    return np.asarray(samples[:size])


def main() -> None:
    rng = np.random.default_rng(42)
    true_signal_yield = 500
    true_background_yield = 1_500
    true_shapes = [0.6, 0.45, -0.20, 0.25]
    true_mean, true_sigma, true_linear, true_quadratic = true_shapes

    signal_mass = rng.normal(true_mean, true_sigma, true_signal_yield)
    signal_mass = signal_mass[(signal_mass >= LOWER) & (signal_mass <= UPPER)]
    background_mass = sample_background(rng, true_background_yield, true_linear, true_quadratic)

    # The control variable is not used by either fit PDF. Within each species it is generated
    # independently of the discriminating mass, which is the condition needed for sPlot.
    signal_control = rng.normal(1.2, 0.55, len(signal_mass))
    background_control = rng.normal(-0.8, 1.1, true_background_yield)
    signal = np.column_stack([signal_mass, signal_control])
    background = np.column_stack([background_mass, background_control])

    data = np.concatenate([signal, background])
    labels = np.concatenate(
        [np.ones(len(signal), dtype=bool), np.zeros(len(background), dtype=bool)]
    )
    order = rng.permutation(len(data))
    data = data[order]
    labels = labels[order]

    result = splot(
        data,
        [truncated_gaussian, quadratic_background],
        shape_parameters=[
            ShapeParameter('mean', 0.2, -1.0, 2.0),
            ShapeParameter('sigma', 0.7, 0.1, 1.5),
            ShapeParameter('background_linear', 0.0, -0.35, 0.35),
            ShapeParameter('background_quadratic', 0.0, -0.35, 0.35),
        ],
    )

    weighted_control_means = [
        np.average(data[:, 1], weights=result.sweights[:, component]) for component in range(2)
    ]
    print(f'converged: {result.success} ({result.message})')
    print(
        'yields [signal, background]:',
        np.round(result.yields, 2),
        f'(truth: [{len(signal)}, {len(background)}])',
    )
    print(
        'shapes [mean, sigma, linear, quadratic]:',
        {name: round(value, 3) for name, value in result.shape_parameters.items()},
        f'(truth: {true_shapes})',
    )
    print('sum of sWeights:', np.round(result.sweights.sum(axis=0), 2))
    print(
        'sWeighted control means [signal, background]:',
        np.round(weighted_control_means, 3),
    )
    print('True control means [signal, background]: [1.200, -0.800]')

    mass_bins = np.linspace(LOWER, UPPER, 61)
    centers = 0.5 * (mass_bins[1:] + mass_bins[:-1])
    width = mass_bins[1] - mass_bins[0]
    grid = np.column_stack([centers, np.zeros_like(centers)])
    signal_curve = result.yields[0] * truncated_gaussian(grid, result.shape_parameters) * width
    background_curve = (
        result.yields[1] * quadratic_background(grid, result.shape_parameters) * width
    )

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(data[:, 0], bins=mass_bins, histtype='step', color='black', label='data')
    axes[0].plot(centers, signal_curve + background_curve, label='total fit')
    axes[0].plot(centers, signal_curve, '--', label='Gaussian signal')
    axes[0].plot(centers, background_curve, ':', label='quadratic background')
    axes[0].set(xlabel='discriminating variable', ylabel='events / bin')
    axes[0].legend()

    control_bins = np.linspace(-4.0, 4.0, 50)
    axes[1].hist(
        data[:, 1],
        bins=control_bins,
        weights=result.sweights[:, 0],
        histtype='step',
        linewidth=2,
        label='signal sWeights',
    )
    axes[1].hist(
        data[:, 1],
        bins=control_bins,
        weights=result.sweights[:, 1],
        histtype='step',
        linewidth=2,
        label='background sWeights',
    )
    axes[1].hist(
        data[labels, 1],
        bins=control_bins,
        histtype='step',
        linestyle='--',
        label='signal truth',
    )
    axes[1].hist(
        data[~labels, 1],
        bins=control_bins,
        histtype='step',
        linestyle='--',
        label='background truth',
    )
    axes[1].set(xlabel='control variable', ylabel='weighted events / bin')
    axes[1].legend()

    figure.tight_layout()
    output_path = Path(__file__).parent.parent / 'target/examples/bump.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f'Plot saved to: {output_path}')


if __name__ == '__main__':
    main()
