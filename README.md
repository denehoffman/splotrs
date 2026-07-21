# splotrs

`splotrs` fits the yields and shared shape parameters of normalized component probability-density
functions, then calculates per-event sWeights. The numerical core is written in Rust and uses
[Ganesh](https://crates.io/crates/ganesh); the Python API accepts vectorized NumPy callbacks.

The library is intended for unbinned mixture fits where a discriminating variable separates
components and the resulting sWeights are used to study another, component-independent variable.
It supports signed input event weights and exposes both the sPlot yield covariance and the full
joint fit covariance.

## Features

- Joint extended maximum-likelihood fits of component yields and shared PDF shape parameters.
- Vectorized Python PDF callbacks with runtime validation of their output type and shape.
- A native Rust API for allocation-conscious fitting without Python.
- Per-event sWeights in the same row order as the input data.
- Signed input event weights.
- Fit diagnostics, uncertainties, and covariance matrices.

## Installation

Install the Python package from PyPI:

```console
python -m pip install splotrs
```

Or add the Rust crate from crates.io:

```console
cargo add splotrs
```

Python 3.10 or newer is required.

## Python quick start

Each component PDF must be callable as
`pdf(events: NDArray[np.float64], parameters: dict[str, float]) -> NDArray[np.float64]`. It receives
the complete two-dimensional event array and must return one finite, nonnegative `float64` density
per event.

```python
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from splotrs import ShapeParameter, splot

Pdf = Callable[
    [NDArray[np.float64], dict[str, float]],
    NDArray[np.float64],
]


def signal(events: NDArray[np.float64], parameters: dict[str, float]) -> NDArray[np.float64]:
    mean = parameters['mean']
    residual = events[:, 0] - mean
    return np.exp(-0.5 * residual**2) / np.sqrt(2.0 * np.pi)


def background(
    events: NDArray[np.float64], _parameters: dict[str, float]
) -> NDArray[np.float64]:
    residual = (events[:, 0] - 2.5) / 0.7
    return np.exp(-0.5 * residual**2) / (0.7 * np.sqrt(2.0 * np.pi))


data = np.array([[-1.2], [-0.3], [0.2], [2.1], [2.8]], dtype=np.float64)
pdfs: list[Pdf] = [signal, background]

result = splot(
    data,
    pdfs,
    shape_parameters=[ShapeParameter('mean', 0.0, -5.0, 5.0)],
)

print(result.yields)
print(result.shape_parameters)
print(result.sweights)  # (n_events, n_components)
```

`shape_parameters` are fitted jointly with the yields. Every PDF receives the complete parameter
dictionary, so multiple components can share the same parameter.

### Inputs

- `data`: finite values with shape `(n_events, n_features)`.
- `pdfs`: one callable per mixture component, in output-column order.
- `shape_parameters`: optional `ShapeParameter` objects with initial values and bounds.
- `initial_yields`: optional nonnegative starting yields, one per PDF.
- `weights`: optional signed event weights with shape `(n_events,)`.
- `max_steps` and `tolerance`: optimizer controls.

### Result

`splot` returns an immutable `SPlotResult` containing:

- `yields` and `yield_errors`;
- `sweights` with shape `(n_events, n_components)`;
- `covariance`, the event-summed sPlot yield covariance;
- `shape_parameters` and `shape_errors` keyed by parameter name;
- `fit_covariance`, the full joint yield-and-shape covariance;
- convergence status, message, minimum negative log-likelihood, and evaluation counts.

Invalid inputs and callback contracts raise `ValueError`. Numerical PDF, optimizer, and covariance
failures raise `SPlotError`. Exceptions raised inside a PDF callback are preserved.

## Complete example

The included example fits a Gaussian bump over a quadratic background and uses sWeights to recover
the component distributions of a control variable:

```console
just examples
```

It writes the resulting plot to `target/examples/bump.png`.

## Rust usage

Implement `ParametricPdf` or use compatible closures, then call `splot` with event rows, PDFs,
shape parameters, and an `SPlotConfig`:

```rust
use splotrs::{ParametricPdf, SPlotConfig, ShapeParameter, ShapeParameters, splot};

let data = vec![vec![-1.2], vec![-0.3], vec![0.2], vec![2.1], vec![2.8]];
let signal = |event: &[f64], parameters: &ShapeParameters| {
    let residual = event[0] - parameters["mean"];
    (-0.5 * residual * residual).exp() / std::f64::consts::TAU.sqrt()
};
let background = |event: &[f64], _parameters: &ShapeParameters| {
    let residual = (event[0] - 2.5) / 0.7;
    (-0.5 * residual * residual).exp() / (0.7 * std::f64::consts::TAU.sqrt())
};
let pdfs: [&dyn ParametricPdf; 2] = [&signal, &background];
let shapes = [ShapeParameter::new("mean", 0.0)?.with_bounds(-5.0, 5.0)?];

let result = splot(&data, &pdfs, &shapes, SPlotConfig::default())?;
println!("yields: {:?}", result.yields);
# Ok::<(), splotrs::SPlotError>(())
```

## Statistical assumptions

- Component PDFs must be normalized over the fitted discriminating variables.
- Components must be identifiable; linearly dependent PDFs produce a singular covariance error.
- A variable studied with sWeights should be independent of the discriminating variables within
  each component.
- Signed input weights are supported, but some weighted datasets can produce an indefinite or
  singular information matrix.

## Development

Enter the reproducible Nix development shell:

```console
nix develop
```

The shell supplies Rust, Cargo, Clippy, rustfmt, Python, uv, and Just. It synchronizes the locked
Python development environment on entry. Run `just` to list all recipes; the main workflows are:

```console
just build       # Build the Rust crate and Python distributions
just test        # Run Rust and Python tests
just lint        # Run Clippy, rustfmt, Ruff, ty, and Yamloom checks
just fmt         # Format Rust and Python sources
just examples    # Run the complete plotting example
just clean       # Remove generated build and test artifacts
```

Without Nix, install uv, Rust, and Just, then run `uv sync` before using the same recipes.

Before publishing, inspect both artifacts locally:

```console
cargo publish --dry-run
uv build
```

## License

Licensed under either of the following, at your option:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))
