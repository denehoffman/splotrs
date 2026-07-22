use ganesh::algorithms::gradient::{
    LBFGSB, LBFGSBConfig, LBFGSBFTerminator, LBFGSBGTerminator, LBFGSBInfNormGTerminator,
};
use ganesh::core::{Callbacks, Matrix, MaxSteps, Vector};
use ganesh::error::GaneshError;
use ganesh::traits::{Algorithm, CostFunction, Gradient};
use std::collections::BTreeMap;
use thiserror::Error;

/// Shape-parameter values keyed by their user-defined names.
///
/// The map is ordered by name for deterministic iteration. PDF implementations
/// should access values by name, for example `parameters["mean"]`.
pub type ShapeParameters = BTreeMap<String, f64>;

/// A normalized component probability density that depends on a shared set of
/// shape parameters.
///
/// Every component receives the same shape-parameter map, allowing individual
/// parameters to be shared by any subset of the component PDFs.
pub trait ParametricPdf: Send + Sync {
    /// Evaluates the PDF for one event at the supplied shape-parameter values.
    ///
    /// Implementations must return a finite, nonnegative density.
    fn evaluate(
        &self,
        event: &[f64],
        shape_parameters: &ShapeParameters,
    ) -> Result<f64, SPlotError>;
}

impl<F> ParametricPdf for F
where
    F: Fn(&[f64], &ShapeParameters) -> f64 + Send + Sync,
{
    fn evaluate(
        &self,
        event: &[f64],
        shape_parameters: &ShapeParameters,
    ) -> Result<f64, SPlotError> {
        Ok(self(event, shape_parameters))
    }
}

/// Describes one scalar shape parameter in the joint likelihood fit.
///
/// Parameter names must be unique. Bounds may be infinite, but the initial
/// value must be finite and lie within the configured interval.
#[derive(Clone, Debug, PartialEq)]
pub struct ShapeParameter {
    /// Human-readable name used to identify the parameter.
    pub name: String,
    /// Starting value supplied to the optimizer.
    pub initial: f64,
    /// Lower optimizer bound.
    pub lower: f64,
    /// Upper optimizer bound.
    pub upper: f64,
}

impl ShapeParameter {
    /// Creates an unconstrained shape parameter with the given initial value.
    ///
    /// The parameter initially has bounds of negative and positive infinity.
    pub fn new(name: impl Into<String>, initial: f64) -> Result<Self, SPlotError> {
        let name = name.into();
        if name.is_empty() || !initial.is_finite() {
            return Err(SPlotError::InvalidInput(
                "shape parameter has an invalid name or initial value".into(),
            ));
        }
        Ok(Self {
            name,
            initial,
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY,
        })
    }

    /// Assigns lower and upper bounds to this parameter.
    ///
    /// The lower bound must be strictly smaller than the upper bound, and the
    /// initial value must lie within the resulting interval.
    pub fn with_bounds(mut self, lower: f64, upper: f64) -> Result<Self, SPlotError> {
        if lower >= upper {
            return Err(SPlotError::InvalidInput(format!(
                "lower bound must be strictly less than upper bound: lower={lower}, upper={upper}"
            )));
        }
        if lower.is_nan() || upper.is_nan() {
            return Err(SPlotError::InvalidInput(format!(
                "lower or upper bound is NaN: lower={lower}, upper={upper}"
            )));
        }
        if self.initial < lower || self.initial > upper {
            return Err(SPlotError::InvalidInput(format!(
                "bounds must contain initial value: initial={}, bounds=({}, {})",
                self.initial, lower, upper
            )));
        }
        self.lower = lower;
        self.upper = upper;
        Ok(self)
    }
}

/// Configures the joint yield-and-shape fit and subsequent sWeight calculation.
#[derive(Clone, Debug)]
pub struct SPlotConfig {
    /// Initial values for the component yields.
    ///
    /// The vector must contain one nonnegative value per PDF and have a
    /// positive sum. When omitted, all components receive equal initial yields
    /// whose sum equals the sum of the event weights.
    pub initial_yields: Option<Vec<f64>>,

    /// Optional signed weight for each input event.
    ///
    /// These weights are applied consistently in the likelihood, yield
    /// information matrix, and final sWeights. When omitted, every event has
    /// unit weight.
    pub event_weights: Option<Vec<f64>>,

    /// Maximum number of optimizer steps.
    ///
    /// A value of `None` disables the explicit step limit.
    pub max_steps: Option<usize>,

    /// Absolute convergence tolerance used by the objective and gradient
    /// terminators.
    ///
    /// This value must be finite and strictly positive.
    pub tolerance: f64,
}

impl Default for SPlotConfig {
    fn default() -> Self {
        Self {
            initial_yields: None,
            event_weights: None,
            max_steps: Some(1_000),
            tolerance: 1e-8,
        }
    }
}

/// Counts objective, gradient, and Hessian evaluations performed by the
/// optimizer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EvaluationCounts {
    /// Number of objective-function evaluations.
    pub objective: usize,
    /// Number of gradient evaluations.
    pub gradient: usize,
    /// Number of Hessian evaluations.
    pub hessian: usize,
}

/// Contains the joint fit result and the sWeights evaluated at its optimum.
///
/// Yields and shape parameters come from one joint optimization. The sPlot
/// yield covariance is then constructed from the event-summed yield
/// information matrix without performing another fit.
#[derive(Clone, Debug)]
pub struct SPlotResult {
    /// Fitted component yields, ordered as the input PDFs.
    pub yields: Vec<f64>,

    /// Yield uncertainties reported by the joint optimizer
    /// covariance ([`Self::fit_covariance`]).
    pub yield_errors: Vec<f64>,

    /// Event-summed sPlot covariance matrix for the component yields.
    ///
    /// Shape parameters are held at their fitted values while this matrix is
    /// constructed.
    pub covariance: Vec<Vec<f64>>,

    /// Per-event sWeights.
    ///
    /// The outer vector follows event order, while each inner vector follows
    /// component order.
    pub sweights: Vec<Vec<f64>>,

    /// Fitted shape-parameter values keyed by parameter name.
    pub shape_parameters: ShapeParameters,

    /// Shape-parameter uncertainties reported by the joint optimizer
    /// covariance ([`Self::fit_covariance`]).
    pub shape_errors: ShapeParameters,

    /// Full covariance matrix from the joint fit.
    ///
    /// Rows and columns are ordered as all component yields followed by all
    /// shape parameters.
    pub fit_covariance: Vec<Vec<f64>>,

    /// Minimum joint negative log-likelihood value.
    pub minimum_nll: f64,

    /// Whether the optimizer reported successful convergence.
    pub success: bool,

    /// Human-readable optimizer termination message.
    pub message: String,

    /// Objective, gradient, and Hessian evaluation counts.
    pub evaluations: EvaluationCounts,
}

/// An error produced while validating inputs, evaluating PDFs, fitting the
/// likelihood, or constructing the sPlot covariance.
#[derive(Debug, Error)]
pub enum SPlotError {
    /// The supplied data, parameters, weights, or configuration are invalid.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// A component PDF failed or returned an invalid density.
    #[error("PDF evaluation failed: {0}")]
    PdfEvaluation(String),

    /// The likelihood or optimizer encountered a numerical failure.
    #[error("optimization failed: {0}")]
    Optimization(String),

    /// The yield information matrix could not produce a valid covariance.
    #[error("singular yield covariance: {0}")]
    SingularCovariance(String),

    /// An error propagated from the ganesh optimization library.
    #[error(transparent)]
    Ganesh(#[from] GaneshError),
}

struct JointLikelihood<'a> {
    data: &'a [Vec<f64>],
    pdfs: &'a [&'a dyn ParametricPdf],
    shape_parameter_names: Vec<String>,
    event_weights: &'a [f64],
    n_components: usize,
}

#[derive(Clone, Debug)]
struct EvaluatedModel {
    pdf_values: Vec<f64>,
    event_weights: Vec<f64>,
    n_events: usize,
    n_components: usize,
}

impl JointLikelihood<'_> {
    fn named_shape_parameters(&self, values: &[f64]) -> ShapeParameters {
        self.shape_parameter_names
            .iter()
            .cloned()
            .zip(values.iter().copied())
            .collect()
    }

    fn evaluate_model(&self, shape_parameters: &[f64]) -> Result<EvaluatedModel, SPlotError> {
        let shape_parameters = self.named_shape_parameters(shape_parameters);
        let mut pdf_values = Vec::with_capacity(self.data.len() * self.n_components);

        for (event_index, event) in self.data.iter().enumerate() {
            let start = pdf_values.len();

            for (component, pdf) in self.pdfs.iter().enumerate() {
                let value = pdf.evaluate(event, &shape_parameters)?;

                if !value.is_finite() || value < 0.0 {
                    return Err(SPlotError::PdfEvaluation(format!(
                        "PDF {component} returned an invalid value for event {event_index}"
                    )));
                }

                pdf_values.push(value);
            }

            if pdf_values[start..].iter().all(|value| *value == 0.0) {
                return Err(SPlotError::PdfEvaluation(format!(
                    "all PDFs are zero for event {event_index}"
                )));
            }
        }

        Ok(EvaluatedModel {
            pdf_values,
            event_weights: self.event_weights.to_vec(),
            n_events: self.data.len(),
            n_components: self.n_components,
        })
    }
}

impl CostFunction<f64, ganesh::NalgebraProvider, (), SPlotError> for JointLikelihood<'_> {
    fn evaluate(&self, parameters: &Vector<f64>, _: &()) -> Result<f64, SPlotError> {
        let parameter_values = parameters.to_vec();
        let shape_parameters = self.named_shape_parameters(&parameter_values[self.n_components..]);

        let mut nll: f64 = (0..self.n_components)
            .map(|component| parameters.get(component))
            .sum();

        for (event_index, event) in self.data.iter().enumerate() {
            let mut denominator = 0.0;
            let mut has_nonzero_pdf = false;

            for (component, pdf) in self.pdfs.iter().enumerate() {
                let value = pdf.evaluate(event, &shape_parameters)?;

                if !value.is_finite() || value < 0.0 {
                    return Err(SPlotError::PdfEvaluation(format!(
                        "PDF {component} returned an invalid value for event {event_index}"
                    )));
                }

                has_nonzero_pdf |= value > 0.0;
                denominator += parameters.get(component) * value;
            }

            if !has_nonzero_pdf {
                return Err(SPlotError::PdfEvaluation(format!(
                    "all PDFs are zero for event {event_index}"
                )));
            }

            if !denominator.is_finite() {
                return Err(SPlotError::Optimization(format!(
                    "mixture density is not finite for event {event_index}"
                )));
            }

            // Protect the logarithm against floating-point underflow.
            let safe_denominator = denominator.max(1e-300);

            nll -= self.event_weights[event_index] * safe_denominator.ln();
        }

        if !nll.is_finite() {
            return Err(SPlotError::Optimization(
                "negative log-likelihood is not finite".into(),
            ));
        }

        Ok(nll)
    }
}

impl Gradient<f64, ganesh::NalgebraProvider, (), SPlotError> for JointLikelihood<'_> {}

impl EvaluatedModel {
    fn denominator(&self, event: usize, yields: &[f64]) -> f64 {
        let offset = event * self.n_components;

        (0..self.n_components)
            .map(|component| yields[component] * self.pdf_values[offset + component])
            .sum()
    }

    fn yield_information(&self, yields: &[f64]) -> Result<Matrix<f64>, SPlotError> {
        let mut information = Matrix::zeros(self.n_components, self.n_components);

        for event in 0..self.n_events {
            let denominator = self.denominator(event, yields);

            if !denominator.is_finite() || denominator <= 0.0 {
                return Err(SPlotError::Optimization(format!(
                    "mixture density is not positive for event {event}"
                )));
            }

            let offset = event * self.n_components;
            let denominator_squared = denominator * denominator;
            let event_weight = self.event_weights[event];

            for row in 0..self.n_components {
                for column in 0..self.n_components {
                    let value = information.get(row, column)
                        + event_weight
                            * self.pdf_values[offset + row]
                            * self.pdf_values[offset + column]
                            / denominator_squared;

                    information.set(row, column, value);
                }
            }
        }

        Ok(information)
    }

    fn calculate_sweights(
        &self,
        yields: &[f64],
        covariance: &Matrix<f64>,
    ) -> Result<Vec<Vec<f64>>, SPlotError> {
        let mut sweights = vec![vec![0.0; self.n_components]; self.n_events];

        for (event, event_sweights) in sweights.iter_mut().enumerate() {
            let denominator = self.denominator(event, yields);

            if !denominator.is_finite() || denominator <= 0.0 {
                return Err(SPlotError::Optimization(format!(
                    "mixture density is not positive for event {event}"
                )));
            }

            let offset = event * self.n_components;

            for (component, sweight) in event_sweights.iter_mut().enumerate() {
                let numerator: f64 = (0..self.n_components)
                    .map(|column| {
                        covariance.get(component, column) * self.pdf_values[offset + column]
                    })
                    .sum();

                *sweight = self.event_weights[event] * numerator / denominator;
            }
        }

        Ok(sweights)
    }
}

fn validate_inputs(
    data: &[Vec<f64>],
    pdfs: &[&dyn ParametricPdf],
    shape_parameters: &[ShapeParameter],
    config: &SPlotConfig,
) -> Result<(Vec<f64>, Vec<f64>), SPlotError> {
    if data.is_empty() {
        return Err(SPlotError::InvalidInput("data must not be empty".into()));
    }

    if pdfs.is_empty() {
        return Err(SPlotError::InvalidInput(
            "at least one PDF is required".into(),
        ));
    }

    let dimension = data[0].len();

    if dimension == 0 || data.iter().any(|event| event.len() != dimension) {
        return Err(SPlotError::InvalidInput(
            "events must have one consistent, nonzero dimension".into(),
        ));
    }

    if data.iter().flatten().any(|value| !value.is_finite()) {
        return Err(SPlotError::InvalidInput(
            "event coordinates must be finite".into(),
        ));
    }

    if config.max_steps == Some(0) {
        return Err(SPlotError::InvalidInput(
            "max_steps must be positive".into(),
        ));
    }

    if !config.tolerance.is_finite() || config.tolerance <= 0.0 {
        return Err(SPlotError::InvalidInput(
            "tolerance must be finite and positive".into(),
        ));
    }

    let event_weights = match &config.event_weights {
        Some(weights) => {
            if weights.len() != data.len() {
                return Err(SPlotError::InvalidInput(format!(
                    "event_weights has length {}, expected {}",
                    weights.len(),
                    data.len()
                )));
            }

            if weights.iter().any(|weight| !weight.is_finite()) {
                return Err(SPlotError::InvalidInput(
                    "event weights must be finite".into(),
                ));
            }

            weights.clone()
        }
        None => vec![1.0; data.len()],
    };

    for (index, parameter) in shape_parameters.iter().enumerate() {
        if shape_parameters[..index]
            .iter()
            .any(|other| other.name == parameter.name)
        {
            return Err(SPlotError::InvalidInput(format!(
                "shape parameter name {:?} is duplicated",
                parameter.name
            )));
        }
    }

    let mut initial_yields = config.initial_yields.clone().unwrap_or_else(|| {
        let total_weight = event_weights.iter().sum::<f64>();
        vec![total_weight / pdfs.len() as f64; pdfs.len()]
    });

    if initial_yields.len() != pdfs.len()
        || initial_yields
            .iter()
            .any(|yield_| !yield_.is_finite() || *yield_ < 0.0)
        || initial_yields.iter().sum::<f64>() <= 0.0
    {
        return Err(SPlotError::InvalidInput(
            "initial_yields must match the PDFs, be finite and nonnegative, \
             and have positive sum"
                .into(),
        ));
    }

    // The lower optimizer bound is positive because the extended likelihood
    // is undefined when all yields vanish. Convert user-supplied zeros into
    // an effective floating-point zero.
    for yield_ in &mut initial_yields {
        *yield_ = yield_.max(f64::EPSILON);
    }

    Ok((initial_yields, event_weights))
}

fn covariance_is_valid(
    information: &Matrix<f64>,
    covariance: &Matrix<f64>,
    tolerance: f64,
) -> bool {
    let dimension = information.rows();
    let residual_tolerance = (10.0 * tolerance).max(1e-8);

    for row in 0..dimension {
        for column in 0..dimension {
            let product: f64 = (0..dimension)
                .map(|index| information.get(row, index) * covariance.get(index, column))
                .sum();

            let expected = if row == column { 1.0 } else { 0.0 };

            if !product.is_finite() || (product - expected).abs() > residual_tolerance {
                return false;
            }
        }
    }

    true
}

/// Fits component yields and shared shape parameters in a single joint
/// extended-likelihood optimization, then computes sWeights at the fitted
/// point.
///
/// Each PDF receives the complete shape-parameter map, allowing parameters
/// to be shared across arbitrary subsets of components. After the fit, the
/// PDFs are evaluated at the fitted shape values and the yield-only information
/// matrix is inverted to obtain the sPlot covariance. No secondary yield fit is
/// performed.
///
/// # Parameters
///
/// - `data`: Event vectors. Every event must have the same nonzero dimension
///   and contain only finite coordinates.
/// - `pdfs`: Normalized component PDFs.
/// - `shape_parameters`: Shared shape-parameter declarations. Their order sets
///   the shape-parameter axes in [`SPlotResult::fit_covariance`], while PDFs
///   access their current values by name.
/// - `config`: Initial yields, event weights, and optimizer controls.
///
/// # Errors
///
/// Returns [`SPlotError::InvalidInput`] for malformed inputs,
/// [`SPlotError::PdfEvaluation`] for invalid PDF values,
/// [`SPlotError::Optimization`] for numerical fit failures, or
/// [`SPlotError::SingularCovariance`] when the component yields are not
/// independently identifiable.
pub fn splot(
    data: &[Vec<f64>],
    pdfs: &[&dyn ParametricPdf],
    shape_parameters: &[ShapeParameter],
    config: SPlotConfig,
) -> Result<SPlotResult, SPlotError> {
    let (mut initial_yields, event_weights) =
        validate_inputs(data, pdfs, shape_parameters, &config)?;

    // Ganesh's L-BFGS-B expects to take at least one step before its
    // terminators run. With no shape parameters, the default yield start can
    // be exactly stationary, particularly for a one-component model.
    if shape_parameters.is_empty() {
        let displacement = config.tolerance.sqrt().clamp(1e-6, 0.5);

        for yield_ in &mut initial_yields {
            *yield_ *= 1.0 - displacement;
            *yield_ = yield_.max(f64::EPSILON);
        }
    }

    let mut initial = initial_yields;
    initial.extend(shape_parameters.iter().map(|parameter| parameter.initial));

    let mut bounds = vec![(f64::EPSILON, f64::INFINITY); pdfs.len()];

    bounds.extend(
        shape_parameters
            .iter()
            .map(|parameter| (parameter.lower, parameter.upper)),
    );

    let problem = JointLikelihood {
        data,
        pdfs,
        shape_parameter_names: shape_parameters
            .iter()
            .map(|parameter| parameter.name.clone())
            .collect(),
        event_weights: &event_weights,
        n_components: pdfs.len(),
    };

    // Validate PDFs and the mixture at the initial point before optimization.
    problem.evaluate(&Vector::from_vec(initial.clone()), &())?;

    let optimizer_config = LBFGSBConfig::<f64>::default().with_bounds(bounds)?;

    let mut callbacks = Callbacks::empty()
        .with_terminator(LBFGSBFTerminator::new(config.tolerance)?)
        .with_terminator(LBFGSBGTerminator::new(config.tolerance)?)
        .with_terminator(LBFGSBInfNormGTerminator::new(config.tolerance)?);

    if let Some(max_steps) = config.max_steps {
        callbacks = callbacks.with_terminator(MaxSteps(max_steps));
    }

    // The only optimizer invocation.
    let joint = LBFGSB::<f64>::default().process(
        &problem,
        &(),
        Vector::from_vec(initial),
        optimizer_config,
        callbacks,
    )?;

    let fitted_parameters = joint.x.to_vec();
    let yields = fitted_parameters[..pdfs.len()].to_vec();
    let fitted_shapes = fitted_parameters[pdfs.len()..].to_vec();

    // Evaluate the PDFs once at the joint optimum. These values are used only
    // for the covariance and sWeight calculations, not for another fit.
    let evaluated_model = problem.evaluate_model(&fitted_shapes)?;

    let information = evaluated_model.yield_information(&yields)?;

    let splot_covariance = information.lu_inverse().ok_or_else(|| {
        SPlotError::SingularCovariance(
            "the event-summed yield information matrix is not invertible".into(),
        )
    })?;

    if !covariance_is_valid(&information, &splot_covariance, config.tolerance) {
        return Err(SPlotError::SingularCovariance(
            "the component PDFs are not independently identifiable".into(),
        ));
    }

    let sweights = evaluated_model.calculate_sweights(&yields, &splot_covariance)?;

    let covariance: Vec<Vec<f64>> = (0..pdfs.len())
        .map(|row| {
            (0..pdfs.len())
                .map(|column| splot_covariance.get(row, column))
                .collect()
        })
        .collect();

    let total_dimension = pdfs.len() + shape_parameters.len();

    let fit_covariance = (0..total_dimension)
        .map(|row| {
            (0..total_dimension)
                .map(|column| joint.covariance.get(row, column))
                .collect()
        })
        .collect();

    let joint_std = joint.std.to_vec();
    let yield_errors = joint_std[..pdfs.len()].to_vec();
    let shape_errors = shape_parameters
        .iter()
        .map(|parameter| parameter.name.clone())
        .zip(joint_std[pdfs.len()..].iter().copied())
        .collect();
    let fitted_shapes = shape_parameters
        .iter()
        .map(|parameter| parameter.name.clone())
        .zip(fitted_shapes)
        .collect();

    Ok(SPlotResult {
        yields,
        yield_errors,
        covariance,
        sweights,
        shape_parameters: fitted_shapes,
        shape_errors,
        fit_covariance,
        minimum_nll: joint.fx,
        success: joint.message.success(),
        message: joint.message.to_string(),
        evaluations: EvaluationCounts {
            objective: joint.evals.f(),
            gradient: joint.evals.g(),
            hessian: joint.evals.h(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gaussian(mean: f64, sigma: f64) -> impl ParametricPdf {
        move |event: &[f64], _parameters: &ShapeParameters| {
            let z = (event[0] - mean) / sigma;

            (-0.5 * z * z).exp() / (sigma * std::f64::consts::TAU.sqrt())
        }
    }

    #[test]
    fn one_component_has_event_count_yield_and_unit_weights() {
        let data = vec![vec![-1.0], vec![0.0], vec![0.5], vec![2.0]];

        let pdf = gaussian(0.0, 1.0);

        let result = splot(&data, &[&pdf], &[], SPlotConfig::default()).unwrap();

        assert!((result.yields[0] - data.len() as f64).abs() < 1e-6);

        assert!((result.covariance[0][0] - data.len() as f64).abs() < 1e-6);

        assert!(
            result
                .sweights
                .iter()
                .all(|weights| (weights[0] - 1.0).abs() < 1e-8)
        );
    }

    #[test]
    fn default_initial_yields_sum_to_event_weight_sum() {
        let data = vec![vec![-1.0], vec![0.0], vec![1.0]];
        let first = gaussian(-1.0, 1.0);
        let second = gaussian(1.0, 1.0);
        let config = SPlotConfig {
            event_weights: Some(vec![0.5, 1.0, 2.0]),
            ..SPlotConfig::default()
        };

        let (initial_yields, _) = validate_inputs(&data, &[&first, &second], &[], &config).unwrap();

        assert_eq!(initial_yields, vec![1.75, 1.75]);
    }

    #[test]
    fn identical_pdfs_are_rejected_as_singular() {
        let data = vec![vec![-1.0], vec![0.0], vec![1.0]];

        let first = gaussian(0.0, 1.0);
        let second = gaussian(0.0, 1.0);

        let error = splot(&data, &[&first, &second], &[], SPlotConfig::default()).unwrap_err();

        assert!(matches!(error, SPlotError::SingularCovariance(_)));
    }

    #[test]
    fn two_components_recover_yields_and_sweight_sum_rule() {
        let mut data = Vec::new();

        for index in 0..20 {
            data.push(vec![-2.0 + 0.02 * index as f64]);
        }

        for index in 0..30 {
            data.push(vec![2.0 + 0.02 * index as f64]);
        }

        let first = gaussian(-2.0, 0.4);
        let second = gaussian(2.2, 0.4);

        let result = splot(&data, &[&first, &second], &[], SPlotConfig::default()).unwrap();

        assert!((result.yields[0] - 20.0).abs() < 1e-4);
        assert!((result.yields[1] - 30.0).abs() < 1e-4);
        assert_eq!(result.sweights.len(), data.len());

        for component in 0..2 {
            let weight_sum: f64 = result
                .sweights
                .iter()
                .map(|weights| weights[component])
                .sum();

            assert!((weight_sum - result.yields[component]).abs() < 1e-4);
        }
    }

    #[test]
    fn signed_weights_enter_fit_covariance_and_sweights() {
        let data = vec![vec![-1.0], vec![0.0], vec![1.0]];
        let pdf = gaussian(0.0, 1.0);
        let weights = vec![2.0, -1.0, 2.0];

        let config = SPlotConfig {
            event_weights: Some(weights.clone()),
            ..SPlotConfig::default()
        };

        let result = splot(&data, &[&pdf], &[], config).unwrap();

        assert!((result.yields[0] - 3.0).abs() < 1e-6);
        assert!((result.covariance[0][0] - 3.0).abs() < 1e-6);

        for (event_weights, input_weight) in result.sweights.iter().zip(weights) {
            assert!((event_weights[0] - input_weight).abs() < 1e-6);
        }
    }

    #[test]
    fn joint_fit_estimates_shape_and_uses_same_fit_yield() {
        let data = vec![vec![-2.0], vec![-1.0], vec![0.0], vec![1.0], vec![2.0]];

        let gaussian = |event: &[f64], parameters: &ShapeParameters| {
            let residual = event[0] - parameters["mean"];

            (-0.5 * residual * residual).exp() / std::f64::consts::TAU.sqrt()
        };

        let parameter = ShapeParameter::new("mean", 0.7)
            .unwrap()
            .with_bounds(-5.0, 5.0)
            .unwrap();

        let result = splot(&data, &[&gaussian], &[parameter], SPlotConfig::default()).unwrap();

        assert!(result.shape_parameters["mean"].abs() < 1e-5);

        assert!((result.yields[0] - data.len() as f64).abs() < 1e-6);

        assert!((result.covariance[0][0] - data.len() as f64).abs() < 1e-6);

        assert_eq!(result.fit_covariance.len(), 2);

        assert!(
            result
                .sweights
                .iter()
                .all(|weights| (weights[0] - 1.0).abs() < 1e-8)
        );
    }

    #[test]
    fn shape_parameter_defaults_to_infinite_bounds() {
        let parameter = ShapeParameter::new("mean", 0.0).unwrap();

        assert_eq!(parameter.lower, f64::NEG_INFINITY);

        assert_eq!(parameter.upper, f64::INFINITY);
    }
}
