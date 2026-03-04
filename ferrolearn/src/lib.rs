//! # ferrolearn
//!
//! A scikit-learn equivalent for Rust.
//!
//! This is the top-level re-export crate that provides a unified API
//! for all ferrolearn functionality. Individual sub-crates can also
//! be used directly for finer-grained dependency control.

pub use ferrolearn_core as core;
pub use ferrolearn_linear as linear;
pub use ferrolearn_metrics as metrics;
pub use ferrolearn_model_sel as model_selection;
pub use ferrolearn_preprocess as preprocess;
pub use ferrolearn_sparse as sparse;

// Also re-export the most common items at the top level.
pub use ferrolearn_core::pipeline::Pipeline;
pub use ferrolearn_core::{
    Dataset, FerroError, FerroResult, Fit, FitTransform, Predict, Transform,
};
