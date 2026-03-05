# Ferrolearn Performance Report

**ferrolearn vs. scikit-learn — benchmark comparison**

Measured on Linux 6.6.87 (WSL2), AMD64. Rust benchmarks use [Criterion](https://github.com/bls12-381/criterion.rs) with statistical analysis; Python benchmarks use median wall-clock time over 20 iterations. scikit-learn 1.7.2, ferrolearn 0.1.0, Rust 1.85 (edition 2024).

All comparisons use identical dataset sizes, hyperparameters, and random seeds. Ferrolearn results updated 2026-03-05 after Phase 3 & 4 merge (1,932 tests, 25 new algorithm modules).

---

## Why Ferrolearn?

scikit-learn is the gold standard for classical machine learning — its API design is excellent. But it's built on Python with C/Fortran extensions, which means:

- **Python overhead** on every call boundary, especially painful for small-to-medium datasets
- **No compile-time safety** — calling `predict()` on an unfitted model is a runtime error
- **GIL contention** limits true parallelism
- **Deployment complexity** — shipping a Python runtime, NumPy, and scikit-learn to production

Ferrolearn is a ground-up Rust implementation of classical ML with a scikit-learn-compatible API. It gives you:

1. **Compile-time correctness** — unfitted models don't have a `predict()` method. Period.
2. **True parallelism** — Rayon-based data parallelism with no GIL
3. **Single binary deployment** — no runtime, no pip, no conda
4. **Python bindings included** — drop-in replacement via `pip install ferrolearn`
5. **Modular crates** — depend only on the algorithms you need

And as the benchmarks below show: it's fast.

---

## Benchmark Results

### Regressors

| Algorithm | Dataset | sklearn | ferrolearn | Speedup |
|-----------|---------|---------|------------|---------|
| **LinearRegression** | 50 x 5 | 178 us | 0.84 us | **212x** |
| | 1K x 10 | 228 us | 20 us | **11x** |
| | 10K x 100 | 24.8 ms | 4.7 ms | **5.3x** |
| **Ridge** | 50 x 5 | 213 us | 0.80 us | **266x** |
| | 1K x 10 | 257 us | 20 us | **13x** |
| | 10K x 100 | 7.4 ms | 4.9 ms | **1.5x** |
| **Lasso** | 50 x 5 | 200 us | 2.9 us | **69x** |
| | 1K x 10 | 235 us | 82 us | **2.9x** |
| | 10K x 100 | 15.0 ms | 12.0 ms | **1.3x** |
| **ElasticNet** | 50 x 5 | 199 us | 2.3 us | **87x** |
| | 1K x 10 | 230 us | 69 us | **3.3x** |
| | 10K x 100 | 16.0 ms | 9.4 ms | **1.7x** |

At small-to-medium scale, ferrolearn eliminates Python call overhead entirely — **up to 266x faster** for Ridge regression on small data. At 10K samples, where the actual linear algebra dominates, ferrolearn is still 1.3-5.3x faster.

### Classifiers

| Algorithm | Dataset | sklearn | ferrolearn | Speedup |
|-----------|---------|---------|------------|---------|
| **LogisticRegression** | 50 x 5 | 616 us | 16 us | **38x** |
| | 1K x 10 | 972 us | 433 us | **2.2x** |
| | 10K x 100 | 1,168 ms | 11.3 ms | **103x** |
| **DecisionTree** | 50 x 5 | 212 us | 5.1 us | **42x** |
| | 1K x 10 | 5.2 ms | 245 us | **21x** |
| | 10K x 100 | 1,033 ms | 33.1 ms | **31x** |
| **RandomForest** | 50 x 5 | 51.4 ms | 1.5 ms | **34x** |
| | 1K x 10 | 139 ms | 2.7 ms | **51x** |
| | 10K x 100 | 6,219 ms | 63.6 ms | **98x** |
| **GaussianNB** | 50 x 5 | 189 us | 0.62 us | **305x** |
| | 1K x 10 | 277 us | 10 us | **28x** |
| | 10K x 100 | 3.8 ms | 1.2 ms | **3.2x** |

RandomForest training at 10K x 100 goes from **6.2 seconds to 64 milliseconds** — a 98x speedup driven by Rayon's work-stealing thread pool vs. scikit-learn's joblib.

LogisticRegression at 10K x 100 shows a **103x speedup** — ferrolearn's L-BFGS optimizer runs entirely in Rust without Python callback overhead.

### Transformers

| Algorithm | Dataset | sklearn | ferrolearn | Speedup |
|-----------|---------|---------|------------|---------|
| **StandardScaler** fit | 50 x 5 | 77 us | 0.18 us | **428x** |
| | 1K x 10 | 130 us | 7.3 us | **18x** |
| | 10K x 100 | 2.9 ms | 0.84 ms | **3.5x** |
| **StandardScaler** transform | 50 x 5 | 23 us | 0.22 us | **105x** |
| | 1K x 10 | 35 us | 8.6 us | **4.1x** |
| | 10K x 100 | 1.3 ms | 1.2 ms | **1.1x** |
| **PCA** fit | 50 x 5 | 151 us | 2.1 us | **72x** |
| | 1K x 10 | 174 us | 33 us | **5.3x** |
| | 10K x 100 | 40 ms | 103 ms | 0.39x |
| **PCA** transform | 50 x 5 | 22 us | 0.34 us | **65x** |
| | 1K x 10 | 34 us | 8.7 us | **3.9x** |
| | 10K x 100 | 4.0 ms | 1.1 ms | **3.6x** |

PCA fit at 10K x 100 is the one case where sklearn wins — its LAPACK-backed SVD is heavily optimized for large dense matrices. Ferrolearn uses a Jacobi eigendecomposition which trades peak throughput for portability (no LAPACK/BLAS dependency). A pluggable BLAS backend is now available in `ferrolearn-core` (feature `blas`).

### Clustering

| Algorithm | Dataset | sklearn | ferrolearn | Speedup |
|-----------|---------|---------|------------|---------|
| **KMeans** fit | 50 x 5 | 1.75 ms | 8.0 us | **219x** |
| | 1K x 10 | 4.6 ms | 12.5 ms | 0.37x |
| | 10K x 100 | 281 ms | 18.9 ms | **14.9x** |

KMeans shows an interesting scaling profile: sklearn's C implementation is competitive at 1K but ferrolearn pulls ahead dramatically at 10K samples thanks to Rayon parallelism.

### Metrics

| Metric | Size | sklearn | ferrolearn | Speedup |
|--------|------|---------|------------|---------|
| **accuracy_score** | 1K | 112 us | 0.57 us | **196x** |
| | 10K | 142 us | 5.3 us | **27x** |
| | 100K | 608 us | 51 us | **12x** |
| **f1_score** | 1K | 567 us | 5.0 us | **113x** |
| | 10K | 705 us | 54 us | **13x** |
| | 100K | 2.3 ms | 613 us | **3.8x** |
| **mean_squared_error** | 1K | 78 us | 0.51 us | **153x** |
| | 10K | 86 us | 5.0 us | **17x** |
| | 100K | 185 us | 49 us | **3.8x** |
| **r2_score** | 1K | 105 us | 1.2 us | **88x** |
| | 10K | 117 us | 11.9 us | **9.8x** |
| | 100K | 292 us | 121 us | **2.4x** |

Metric computation shows massive wins at small scale (the Python function call overhead is significant relative to the actual computation) and solid 2-12x wins at 100K samples. The f1_score calculation improved **2x** since Phase 2 thanks to internal refactoring.

---

## Ferrolearn-only Benchmarks (no sklearn comparison)

These algorithms were added in Phase 3 & 4 and don't yet have sklearn comparison numbers.

### Classifiers — Absolute Timings

| Algorithm | Operation | 50 x 5 | 1K x 10 | 10K x 100 |
|-----------|-----------|--------|---------|-----------|
| **KNeighborsClassifier** | fit | 6.8 us | 398 us | 15.5 ms |
| | predict | 90 us | 36.3 ms | 2.27 s |
| **GaussianNB** | predict | 1.6 us | 69 us | 7.2 ms |
| **DecisionTreeClassifier** | predict | 0.20 us | 3.9 us | 40.8 us |
| **RandomForestClassifier** | predict | 23 us | 802 us | 8.7 ms |

### Regressors — Predict Timings

| Algorithm | 50 x 5 | 1K x 10 | 10K x 100 |
|-----------|--------|---------|-----------|
| **LinearRegression** | 0.11 us | 2.4 us | 139 us |
| **Ridge** | 0.11 us | 2.4 us | 140 us |
| **Lasso** | 0.11 us | 2.4 us | 139 us |
| **ElasticNet** | 0.11 us | 2.3 us | 138 us |

### Clustering — Predict Timings

| Algorithm | 50 x 5 | 1K x 10 | 10K x 100 |
|-----------|--------|---------|-----------|
| **KMeans** | 0.66 us | 507 us | 968 us |

---

## Where sklearn wins

Ferrolearn is not faster at everything — honesty matters:

- **PCA fit on large dense matrices** (0.39x) — sklearn delegates to LAPACK's highly-tuned SVD. Ferrolearn's pure-Rust Jacobi decomposition trades speed for zero native dependencies.
- **KNeighborsClassifier predict on large data** — sklearn uses a ball tree with optimized Cython distance computations. Ferrolearn's ball tree implementation is competitive at small scale but needs optimization for high-dimensional data (10K x 100 takes 2.27s).
- **KMeans fit at medium scale** (0.37x) — sklearn's Cython k-means inner loop is very tight. Ferrolearn's Rayon parallelism pays off at larger scale.

The architecture supports pluggable backends — the `blas` feature flag in `ferrolearn-core` provides a BLAS/LAPACK backend for PCA and other decompositions.

---

## Beyond Performance

Speed is only part of the story. Ferrolearn offers qualitative advantages that benchmarks can't capture:

### Compile-time safety

```rust
let model = Ridge::<f64>::new();
// model.predict(&x);  // COMPILE ERROR: Ridge does not implement Predict
let fitted = model.fit(&x, &y)?;
let y_hat = fitted.predict(&x)?;  // OK: FittedRidge implements Predict
```

In scikit-learn, calling `predict()` before `fit()` raises a runtime `NotFittedError`. In ferrolearn, it's a **compile error**. The type system guarantees that every prediction comes from a fitted model.

### Zero-dependency deployment

A ferrolearn model compiles to a single static binary. No Python runtime, no NumPy wheels, no conda environments, no Docker images with 2GB of dependencies. This matters for edge deployment, embedded systems, and serverless functions.

### True thread safety

Every ferrolearn model is `Send + Sync`. You can share fitted models across threads without locks, run predictions in parallel without a GIL, and scale to all cores without joblib's multiprocessing overhead.

### Modular dependency tree

Need only linear models? `cargo add ferrolearn-linear` brings in just what you need. Your binary stays small, your compile times stay fast, and your attack surface stays minimal.

### Python compatibility

For teams that need both worlds, `pip install ferrolearn` provides a scikit-learn-compatible Python API backed by the Rust implementation. Same `fit`/`predict` interface, same NumPy arrays, Rust speed.

---

## Reproducing these benchmarks

```bash
# Rust benchmarks (requires Rust 1.85+)
cargo bench -p ferrolearn-bench

# Python comparison
cd ferrolearn-bench
python3 sklearn_bench.py
```

---

## Summary

On the **medium-scale benchmark (10K samples, 100 features)** — the most representative real-world scenario:

| Category | Geometric mean speedup |
|----------|----------------------|
| Regressors (fit) | **2.0x** |
| Classifiers (fit) | **32x** |
| Transformers | **1.5x** |
| Clustering | **15x** |
| Metrics | **4.5x** |

Ferrolearn delivers scikit-learn's ergonomics with Rust's performance, safety, and deployment story. It's not a wrapper around C extensions pretending to be Python — it's ML done right, from the ground up.

---

*ferrolearn is licensed under MIT or Apache 2.0. Contributions welcome at [github.com/dollspace-gay/ferrolearn](https://github.com/dollspace-gay/ferrolearn).*
