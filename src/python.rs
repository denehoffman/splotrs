use pyo3::pymodule;

/// Joint extended-likelihood fitting and sWeight calculation.
#[pymodule]
pub mod splotrs {
    use crate::{
        ParametricPdf, SPlotConfig, SPlotError as RustSPlotError, SPlotResult, ShapeParameter,
        ShapeParameters, splot as fit_splot,
    };
    use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyUntypedArrayMethods};
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use pyo3::inspect::PyStaticExpr;
    use pyo3::prelude::*;
    use pyo3::types::{PyAny, PyDict};
    use pyo3::{Borrowed, type_hint_identifier, type_hint_subscript};
    use std::sync::Mutex;

    pyo3::create_exception!(splotrs, SPlotError, PyRuntimeError);

    /// Shared shape parameter used by the joint yield-and-shape fit.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Unique name identifying the parameter.
    /// initial : float
    ///     Initial value supplied to the optimizer.
    /// lower : float, optional
    ///     Inclusive lower bound. Defaults to negative infinity.
    /// upper : float, optional
    ///     Inclusive upper bound. Defaults to positive infinity.
    ///
    /// Attributes
    /// ----------
    /// name : str
    ///     Parameter name.
    /// initial : float
    ///     Initial optimizer value.
    /// lower : float
    ///     Lower optimizer bound.
    /// upper : float
    ///     Upper optimizer bound.
    ///
    /// Notes
    /// -----
    /// The initial value must be finite and lie within the configured bounds.
    /// Parameter names must be unique within a call to `splot`.
    #[pyclass(name = "ShapeParameter", frozen, module = "splotrs")]
    pub struct PyShapeParameter {
        #[pyo3(get)]
        name: String,
        #[pyo3(get)]
        initial: f64,
        #[pyo3(get)]
        lower: f64,
        #[pyo3(get)]
        upper: f64,
    }

    #[pymethods]
    impl PyShapeParameter {
        /// Create a shape parameter.
        ///
        /// Parameters
        /// ----------
        /// name : str
        ///     Unique name identifying the parameter.
        /// initial : float
        ///     Initial value supplied to the optimizer.
        /// lower : float, optional
        ///     Inclusive lower bound. Defaults to negative infinity.
        /// upper : float, optional
        ///     Inclusive upper bound. Defaults to positive infinity.
        ///
        /// Returns
        /// -------
        /// ShapeParameter
        ///     Immutable shape-parameter specification.
        #[new]
        #[pyo3(signature = (name, initial, lower=f64::NEG_INFINITY, upper=f64::INFINITY))]
        fn new(name: String, initial: f64, lower: f64, upper: f64) -> Self {
            Self {
                name,
                initial,
                lower,
                upper,
            }
        }

        fn __repr__(&self) -> String {
            format!(
                "ShapeParameter(name={:?}, initial={}, lower={}, upper={})",
                self.name, self.initial, self.lower, self.upper
            )
        }
    }

    impl TryFrom<&PyShapeParameter> for ShapeParameter {
        type Error = PyErr;
        fn try_from(parameter: &PyShapeParameter) -> Result<Self, Self::Error> {
            Ok(Self::new(parameter.name.clone(), parameter.initial)?
                .with_bounds(parameter.lower, parameter.upper)?)
        }
    }

    /// Result of a joint yield-and-shape fit and its sWeight calculation.
    ///
    /// Attributes
    /// ----------
    /// yields : numpy.ndarray
    ///     Fitted component yields with shape ``(n_components,)``.
    /// yield_errors : numpy.ndarray
    ///     Yield uncertainties derived from the full joint covariance, with
    ///     shape ``(n_components,)``.
    /// covariance : numpy.ndarray
    ///     Event-summed sPlot covariance matrix for the fitted yields, with shape
    ///     ``(n_components, n_components)``.
    /// sweights : numpy.ndarray
    ///     Per-event sWeights with shape ``(n_events, n_components)``.
    /// shape_parameters : dict[str, float]
    ///     Fitted shape-parameter values keyed by parameter name.
    /// shape_errors : dict[str, float]
    ///     Shape-parameter uncertainties keyed by parameter name.
    /// fit_covariance : numpy.ndarray
    ///     Full covariance matrix from the joint fit. Yields precede shape
    ///     parameters along both axes.
    /// minimum_nll : float
    ///     Minimum joint negative log-likelihood.
    /// success : bool
    ///     Whether the optimizer reported successful convergence.
    /// message : str
    ///     Optimizer termination message.
    /// objective_evaluations : int
    ///     Number of objective-function evaluations.
    /// gradient_evaluations : int
    ///     Number of gradient evaluations.
    /// hessian_evaluations : int
    ///     Number of Hessian evaluations.
    #[pyclass(name = "SPlotResult", frozen, module = "splotrs")]
    pub struct PySPlotResult {
        n_components: usize,
        #[pyo3(get)]
        yields: Py<PyArray1<f64>>,
        #[pyo3(get)]
        yield_errors: Py<PyArray1<f64>>,
        #[pyo3(get)]
        covariance: Py<PyArray2<f64>>,
        #[pyo3(get)]
        sweights: Py<PyArray2<f64>>,
        #[pyo3(get)]
        shape_parameters: ShapeParameters,
        #[pyo3(get)]
        shape_errors: ShapeParameters,
        #[pyo3(get)]
        fit_covariance: Py<PyArray2<f64>>,
        #[pyo3(get)]
        minimum_nll: f64,
        #[pyo3(get)]
        success: bool,
        #[pyo3(get)]
        message: String,
        #[pyo3(get)]
        objective_evaluations: usize,
        #[pyo3(get)]
        gradient_evaluations: usize,
        #[pyo3(get)]
        hessian_evaluations: usize,
    }

    #[pymethods]
    impl PySPlotResult {
        fn __repr__(&self) -> String {
            format!(
                "SPlotResult(yields=<{} components>, success={}, minimum_nll={:.6})",
                self.n_components, self.success, self.minimum_nll
            )
        }
    }

    struct CachedEvaluation {
        parameters: ShapeParameters,
        values: Vec<f64>,
    }

    struct PythonPdf(Py<PyAny>);

    struct PythonArray1(Py<PyArray1<f64>>);

    struct PythonArray2(Py<PyArray2<f64>>);

    impl FromPyObject<'_, '_> for PythonArray1 {
        type Error = PyErr;

        const INPUT_TYPE: PyStaticExpr = type_hint_subscript!(
            type_hint_identifier!("numpy.typing", "NDArray"),
            type_hint_identifier!("numpy", "float64")
        );

        fn extract(object: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
            Ok(Self(object.cast::<PyArray1<f64>>()?.to_owned().unbind()))
        }
    }

    impl FromPyObject<'_, '_> for PythonArray2 {
        type Error = PyErr;

        const INPUT_TYPE: PyStaticExpr = type_hint_subscript!(
            type_hint_identifier!("numpy.typing", "NDArray"),
            type_hint_identifier!("numpy", "float64")
        );

        fn extract(object: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
            Ok(Self(object.cast::<PyArray2<f64>>()?.to_owned().unbind()))
        }
    }

    impl FromPyObject<'_, '_> for PythonPdf {
        type Error = PyErr;

        const INPUT_TYPE: PyStaticExpr = type_hint_subscript!(
            type_hint_identifier!("collections.abc", "Callable"),
            PyStaticExpr::List {
                elts: &[
                    type_hint_subscript!(
                        type_hint_identifier!("numpy.typing", "NDArray"),
                        type_hint_identifier!("numpy", "float64")
                    ),
                    type_hint_subscript!(
                        type_hint_identifier!("builtins", "dict"),
                        type_hint_identifier!("builtins", "str"),
                        type_hint_identifier!("builtins", "float")
                    )
                ]
            },
            type_hint_subscript!(
                type_hint_identifier!("numpy.typing", "NDArray"),
                type_hint_identifier!("numpy", "float64")
            )
        );

        fn extract(object: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
            Ok(Self(object.to_owned().unbind()))
        }
    }

    struct PythonParametricPdf {
        callable: Py<PyAny>,
        data: Py<PyArray2<f64>>,
        component: usize,
        n_events: usize,
        cache: Mutex<Option<CachedEvaluation>>,
    }

    impl ParametricPdf for PythonParametricPdf {
        fn evaluate(
            &self,
            event: &[f64],
            shape_parameters: &ShapeParameters,
        ) -> Result<f64, RustSPlotError> {
            let event_index = event[0] as usize;
            let mut cache = self.cache.lock().map_err(|_| {
                RustSPlotError::PdfEvaluation("Python PDF cache lock was poisoned".into())
            })?;
            let needs_evaluation = cache
                .as_ref()
                .is_none_or(|cached| cached.parameters != *shape_parameters);
            if needs_evaluation {
                let values = Python::attach(|py| -> Result<Vec<f64>, RustSPlotError> {
                    let parameters = parameters_to_python(py, shape_parameters)
                        .map_err(|error| RustSPlotError::PdfEvaluation(error.to_string()))?;
                    let output = self
                        .callable
                        .bind(py)
                        .call1((self.data.bind(py), parameters))
                        .map_err(|error| RustSPlotError::PdfEvaluation(error.to_string()))?;
                    let array = output.cast::<PyArray1<f64>>().map_err(|_| {
                        RustSPlotError::PdfEvaluation(format!(
                            "PDF {} must return a one-dimensional float64 NumPy array",
                            self.component
                        ))
                    })?;
                    let readonly: PyReadonlyArray1<'_, f64> = array.readonly();
                    if readonly.len() != self.n_events {
                        return Err(RustSPlotError::PdfEvaluation(format!(
                            "PDF {} returned {} values, expected {}",
                            self.component,
                            readonly.len(),
                            self.n_events
                        )));
                    }
                    Ok(readonly.as_array().to_vec())
                })?;
                *cache = Some(CachedEvaluation {
                    parameters: shape_parameters.clone(),
                    values,
                });
            }
            cache
                .as_ref()
                .and_then(|cached| cached.values.get(event_index))
                .copied()
                .ok_or_else(|| {
                    RustSPlotError::PdfEvaluation(format!(
                        "event index {event_index} is outside the Python PDF output"
                    ))
                })
        }
    }

    fn matrix_to_numpy<'py>(
        py: Python<'py>,
        values: &[Vec<f64>],
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        PyArray2::from_vec2(py, values).map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    fn parameters_to_python<'py>(
        py: Python<'py>,
        parameters: &ShapeParameters,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dictionary = PyDict::new(py);
        for (name, value) in parameters {
            dictionary.set_item(name, value)?;
        }
        Ok(dictionary)
    }

    fn result_to_python(py: Python<'_>, result: SPlotResult) -> PyResult<PySPlotResult> {
        let covariance = matrix_to_numpy(py, &result.covariance)?.unbind();
        let sweights = matrix_to_numpy(py, &result.sweights)?.unbind();
        let fit_covariance = matrix_to_numpy(py, &result.fit_covariance)?.unbind();
        Ok(PySPlotResult {
            n_components: result.yields.len(),
            yields: PyArray1::from_vec(py, result.yields).unbind(),
            yield_errors: PyArray1::from_vec(py, result.yield_errors).unbind(),
            covariance,
            sweights,
            shape_parameters: result.shape_parameters,
            shape_errors: result.shape_errors,
            fit_covariance,
            minimum_nll: result.minimum_nll,
            success: result.success,
            message: result.message,
            objective_evaluations: result.evaluations.objective,
            gradient_evaluations: result.evaluations.gradient,
            hessian_evaluations: result.evaluations.hessian,
        })
    }

    impl From<RustSPlotError> for PyErr {
        fn from(error: RustSPlotError) -> PyErr {
            match error {
                RustSPlotError::InvalidInput(_) => PyValueError::new_err(error.to_string()),
                RustSPlotError::PdfEvaluation(_)
                | RustSPlotError::Optimization(_)
                | RustSPlotError::SingularCovariance(_)
                | RustSPlotError::Ganesh(_) => SPlotError::new_err(error.to_string()),
            }
        }
    }

    /// Fit component yields and shared shape parameters, then calculate sWeights.
    ///
    /// The yields and shape parameters are determined in a single joint extended
    /// maximum-likelihood fit. After optimization, the yield information matrix is
    /// evaluated at the fitted point and inverted to obtain the sPlot covariance.
    ///
    /// Parameters
    /// ----------
    /// data : array_like
    ///     Two-dimensional event data with shape ``(n_events, n_features)``.
    ///     Values must be finite, and both dimensions must be nonzero.
    /// pdfs : list of callable
    ///     Component probability-density functions. Each callable must have the
    ///     signature ``pdf(data, shape_parameters)`` and return a one-dimensional
    ///     ``float64`` NumPy array of length ``n_events`` containing finite,
    ///     nonnegative density values.
    /// shape_parameters : list of ShapeParameter or None, optional
    ///     Shared shape parameters used by the component PDFs. Each PDF receives
    ///     the complete parameter dictionary keyed by name. Defaults to no free
    ///     shape parameters.
    /// initial_yields : list of float or None, optional
    ///     Initial component yields. The list must contain one finite,
    ///     nonnegative value per PDF and have a positive sum. When omitted, equal
    ///     yields summing to the event-weight sum are used.
    /// weights : array_like or None, optional
    ///     Signed event weights with shape ``(n_events,)``. These weights enter the
    ///     likelihood, yield information matrix, and final sWeights. Unit weights
    ///     are used when omitted.
    /// max_steps : int or None, optional
    ///     Maximum number of optimizer steps. Passing ``None`` disables the
    ///     explicit step limit. Defaults to 1000.
    /// tolerance : float, optional
    ///     Positive finite convergence tolerance used for the objective and
    ///     gradient termination criteria. Defaults to ``1e-8``.
    ///
    /// Returns
    /// -------
    /// SPlotResult
    ///     Joint fit parameters, covariance matrices, diagnostics, and per-event
    ///     sWeights.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the data, weights, initial yields, shape parameters, or optimizer
    ///     configuration are invalid.
    /// SPlotError
    ///     If a PDF raises an exception, returns invalid values, the optimizer
    ///     fails numerically, or the yield information matrix is singular.
    ///
    /// Notes
    /// -----
    /// PDF callables are vectorized over the complete data array. Their outputs
    /// are cached for each shape-parameter point to avoid reevaluating a PDF once
    /// per event.
    ///
    /// The rows of ``sweights`` follow the input event order, and its columns
    /// follow the order of ``pdfs``.
    ///
    /// Signed input weights are supported, but they can make the information
    /// matrix indefinite or singular for some datasets.
    #[pyfunction(name = "splot")]
    #[pyo3(signature = (
        data,
        pdfs,
        *,
        shape_parameters: "list[ShapeParameter] | None" = None,
        initial_yields: "list[float] | None" = None,
        weights = None,
        max_steps: "int | None" = Some(1000),
        tolerance: "float" = 1e-8
    ) -> "SPlotResult")]
    #[allow(clippy::too_many_arguments)]
    fn splot(
        py: Python<'_>,
        data: PythonArray2,
        pdfs: Vec<PythonPdf>,
        shape_parameters: Option<Vec<Py<PyShapeParameter>>>,
        initial_yields: Option<Vec<f64>>,
        weights: Option<PythonArray1>,
        max_steps: Option<usize>,
        tolerance: f64,
    ) -> PyResult<PySPlotResult> {
        let data = data.0.bind(py);
        let data_view = data.readonly();
        let shape = data_view.shape();
        if shape[0] == 0 || shape[1] == 0 {
            return Err(PyValueError::new_err(
                "data must have shape (n_events, n_features) with both dimensions nonzero",
            ));
        }
        if pdfs.is_empty() {
            return Err(PyValueError::new_err("at least one PDF is required"));
        }

        let config = SPlotConfig {
            initial_yields,
            event_weights: weights
                .map(|weights| weights.0.bind(py).to_vec())
                .transpose()?,
            max_steps,
            tolerance,
        };
        let parameters: Vec<ShapeParameter> = shape_parameters
            .unwrap_or_default()
            .iter()
            .map(|parameter| ShapeParameter::try_from(&*parameter.bind(py).borrow()))
            .collect::<Result<_, _>>()?;
        let indexed_rows: Vec<Vec<f64>> = data_view
            .as_array()
            .rows()
            .into_iter()
            .enumerate()
            .map(|(index, row)| {
                let mut indexed = Vec::with_capacity(row.len() + 1);
                indexed.push(index as f64);
                indexed.extend(row.iter().copied());
                indexed
            })
            .collect();
        let initial_parameters: ShapeParameters = parameters
            .iter()
            .map(|parameter| (parameter.name.clone(), parameter.initial))
            .collect();
        let initial_parameters_python = parameters_to_python(py, &initial_parameters)?;
        let mut initial_values = Vec::with_capacity(pdfs.len());
        for (index, pdf) in pdfs.iter().enumerate() {
            let pdf = pdf.0.bind(py);
            if !pdf.is_callable() {
                return Err(PyValueError::new_err(format!(
                    "PDF {index} must be callable"
                )));
            }
            let output = pdf.call1((data, initial_parameters_python.clone()))?;
            let array = output.cast::<PyArray1<f64>>().map_err(|_| {
                PyValueError::new_err(format!(
                    "PDF {index} must return a one-dimensional float64 NumPy array"
                ))
            })?;
            let readonly: PyReadonlyArray1<'_, f64> = array.readonly();
            if readonly.len() != shape[0] {
                return Err(PyValueError::new_err(format!(
                    "PDF {index} returned {} values, expected {}",
                    readonly.len(),
                    shape[0]
                )));
            }
            initial_values.push(readonly.as_array().to_vec());
        }
        let adapters: Vec<PythonParametricPdf> = pdfs
            .iter()
            .zip(initial_values)
            .enumerate()
            .map(|(component, (callable, values))| PythonParametricPdf {
                callable: callable.0.clone_ref(py),
                data: data.clone().unbind(),
                component,
                n_events: shape[0],
                cache: Mutex::new(Some(CachedEvaluation {
                    parameters: initial_parameters.clone(),
                    values,
                })),
            })
            .collect();
        let references: Vec<&dyn ParametricPdf> = adapters
            .iter()
            .map(|pdf| pdf as &dyn ParametricPdf)
            .collect();
        let result = fit_splot(&indexed_rows, &references, &parameters, config)?;
        result_to_python(py, result)
    }

    #[pymodule_init]
    fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add("SPlotError", module.py().get_type::<SPlotError>())?;
        Ok(())
    }
}
