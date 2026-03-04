//! # ferrolearn-preprocess
//!
//! Data preprocessing transformers for the ferrolearn machine learning framework.
//!
//! This crate provides standard scalers, encoders, and other data preprocessing
//! utilities that follow the ferrolearn `Fit`/`Transform` trait pattern.
//!
//! ## Scalers
//!
//! All scalers are generic over `F: Float + Send + Sync + 'static` and implement
//! [`Fit<Array2<F>, ()>`](ferrolearn_core::Fit) (returning a `Fitted*` type) and
//! [`FitTransform<Array2<F>>`](ferrolearn_core::FitTransform). The fitted types
//! implement [`Transform<Array2<F>>`](ferrolearn_core::Transform).
//!
//! - [`StandardScaler`] — zero-mean, unit-variance scaling
//! - [`MinMaxScaler`] — scale features to a given range (default `[0, 1]`)
//! - [`RobustScaler`] — median / IQR-based scaling, robust to outliers
//!
//! ## Encoders
//!
//! - [`OneHotEncoder`] — encode `Array2<usize>` categorical columns as binary columns
//! - [`LabelEncoder`] — map `Array1<String>` labels to integer indices
//!
//! ## Pipeline Integration
//!
//! `StandardScaler<f64>`, `MinMaxScaler<f64>`, and `RobustScaler<f64>` each
//! implement [`PipelineTransformer`](ferrolearn_core::pipeline::PipelineTransformer)
//! so they can be used as steps inside a
//! [`Pipeline`](ferrolearn_core::pipeline::Pipeline).
//!
//! # Examples
//!
//! ```
//! use ferrolearn_preprocess::StandardScaler;
//! use ferrolearn_core::traits::FitTransform;
//! use ndarray::array;
//!
//! let x = array![[1.0_f64, 10.0], [2.0, 20.0], [3.0, 30.0]];
//! let scaled = StandardScaler::<f64>::new().fit_transform(&x).unwrap();
//! // scaled columns now have mean ≈ 0 and std ≈ 1
//! ```

pub mod label_encoder;
pub mod min_max_scaler;
pub mod one_hot_encoder;
pub mod robust_scaler;
pub mod standard_scaler;

// Re-exports
pub use label_encoder::{FittedLabelEncoder, LabelEncoder};
pub use min_max_scaler::{FittedMinMaxScaler, MinMaxScaler};
pub use one_hot_encoder::{FittedOneHotEncoder, OneHotEncoder};
pub use robust_scaler::{FittedRobustScaler, RobustScaler};
pub use standard_scaler::{FittedStandardScaler, StandardScaler};
