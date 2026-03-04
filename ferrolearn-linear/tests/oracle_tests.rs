//! Oracle tests that compare ferrolearn linear models against scikit-learn
//! reference outputs stored in `fixtures/*.json`.

use approx::assert_relative_eq;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::{Fit, Predict};
use ndarray::{Array1, Array2};

/// Helper: parse a JSON nested array into an `Array2<f64>`.
fn json_to_array2(value: &serde_json::Value) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = value
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect()
        })
        .collect();
    let n_rows = rows.len();
    let n_cols = rows[0].len();
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    Array2::from_shape_vec((n_rows, n_cols), flat).unwrap()
}

/// Helper: parse a JSON flat array into an `Array1<f64>`.
fn json_to_array1(value: &serde_json::Value) -> Array1<f64> {
    let vec: Vec<f64> = value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    Array1::from_vec(vec)
}

/// Helper: assert two f64 slices are element-wise equal within `epsilon`,
/// with an informative panic message on failure.
fn assert_array_close(actual: &[f64], expected: &[f64], epsilon: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        let denom = e.abs().max(1.0);
        assert!(
            diff / denom <= epsilon,
            "{label}[{i}]: actual={a}, expected={e}, rel_diff={rel}",
            rel = diff / denom,
        );
    }
}

// ---------------------------------------------------------------------------
// LinearRegression oracle test
// ---------------------------------------------------------------------------

#[test]
fn test_linear_regression_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/linear_regression.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let y = json_to_array1(&fixture["input"]["y"]);

    let expected_coefs = json_to_array1(&fixture["expected"]["coefficients"]);
    let expected_intercept = fixture["expected"]["intercept"].as_f64().unwrap();
    let expected_preds = json_to_array1(&fixture["expected"]["predictions"]);

    let fit_intercept = fixture["params"]["fit_intercept"].as_bool().unwrap();

    let model = ferrolearn_linear::LinearRegression::<f64>::new().with_fit_intercept(fit_intercept);
    let fitted = model.fit(&x, &y).unwrap();

    // Compare coefficients.
    assert_array_close(
        fitted.coefficients().as_slice().unwrap(),
        expected_coefs.as_slice().unwrap(),
        1e-6,
        "LinearRegression coefficients",
    );

    // Compare intercept.
    assert_relative_eq!(fitted.intercept(), expected_intercept, epsilon = 1e-6);

    // Compare predictions.
    let preds = fitted.predict(&x).unwrap();
    assert_array_close(
        preds.as_slice().unwrap(),
        expected_preds.as_slice().unwrap(),
        1e-6,
        "LinearRegression predictions",
    );
}

// ---------------------------------------------------------------------------
// Ridge oracle test
// ---------------------------------------------------------------------------

#[test]
fn test_ridge_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/ridge.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let y = json_to_array1(&fixture["input"]["y"]);

    let expected_coefs = json_to_array1(&fixture["expected"]["coefficients"]);
    let expected_intercept = fixture["expected"]["intercept"].as_f64().unwrap();
    let expected_preds = json_to_array1(&fixture["expected"]["predictions"]);

    let alpha = fixture["params"]["alpha"].as_f64().unwrap();
    let fit_intercept = fixture["params"]["fit_intercept"].as_bool().unwrap();

    let model = ferrolearn_linear::Ridge::<f64>::new()
        .with_alpha(alpha)
        .with_fit_intercept(fit_intercept);
    let fitted = model.fit(&x, &y).unwrap();

    // Compare coefficients.
    assert_array_close(
        fitted.coefficients().as_slice().unwrap(),
        expected_coefs.as_slice().unwrap(),
        1e-5,
        "Ridge coefficients",
    );

    // Compare intercept.
    assert_relative_eq!(fitted.intercept(), expected_intercept, epsilon = 1e-5);

    // Compare predictions.
    let preds = fitted.predict(&x).unwrap();
    assert_array_close(
        preds.as_slice().unwrap(),
        expected_preds.as_slice().unwrap(),
        1e-5,
        "Ridge predictions",
    );
}

// ---------------------------------------------------------------------------
// Lasso oracle test
// ---------------------------------------------------------------------------

#[test]
fn test_lasso_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/lasso.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let y = json_to_array1(&fixture["input"]["y"]);

    let expected_coefs = json_to_array1(&fixture["expected"]["coefficients"]);
    let expected_intercept = fixture["expected"]["intercept"].as_f64().unwrap();
    let expected_preds = json_to_array1(&fixture["expected"]["predictions"]);

    let alpha = fixture["params"]["alpha"].as_f64().unwrap();
    let fit_intercept = fixture["params"]["fit_intercept"].as_bool().unwrap();

    // Use tighter convergence tolerance to match scikit-learn more closely.
    let model = ferrolearn_linear::Lasso::<f64>::new()
        .with_alpha(alpha)
        .with_fit_intercept(fit_intercept)
        .with_max_iter(10_000)
        .with_tol(1e-8);
    let fitted = model.fit(&x, &y).unwrap();

    // Compare coefficients (looser tolerance for coordinate descent).
    assert_array_close(
        fitted.coefficients().as_slice().unwrap(),
        expected_coefs.as_slice().unwrap(),
        1e-3,
        "Lasso coefficients",
    );

    // Compare intercept (looser tolerance).
    assert_relative_eq!(fitted.intercept(), expected_intercept, epsilon = 1e-3);

    // Compare predictions (looser tolerance).
    let preds = fitted.predict(&x).unwrap();
    assert_array_close(
        preds.as_slice().unwrap(),
        expected_preds.as_slice().unwrap(),
        1e-2,
        "Lasso predictions",
    );
}
