//! Extended maximum-likelihood fits and sWeights for parametric component PDFs.

mod fit;
mod python;

pub use fit::{
    EvaluationCounts, ParametricPdf, SPlotConfig, SPlotError, SPlotResult, ShapeParameter,
    ShapeParameters, splot,
};
