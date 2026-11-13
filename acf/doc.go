// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package acf computes deterministic, direct sample autocorrelation over a
// fixed window of finite, equally spaced observations. It exposes two distinct
// estimator conventions and returns lag-indexed coefficients with their actual
// observation-pair counts; it neither models timestamps nor infers stationarity.
//
// Domain and public surface:
//   - Compute is the canonical entry point; Options selects centering and the
//     denominator convention explicitly. The zero Options value selects
//     centered CommonDenominator.
//   - ComputeDefault, ComputeLagAdjusted, ComputeUncentered, and
//     ComputeUncenteredLagAdjusted are policy-only wrappers around Compute.
//   - Result contains caller-owned Values and PairCounts slices, a fitted Mean,
//     and the effective Denominator; Clone deep-copies both slices. Result.IsNil
//     implements the reflect-free core.Nilable pointer-presence convention.
//
// Mathematical contract:
//   - For n observations, one full-window mean is fitted and shared by every
//     lag unless Options.NoCenter is true. Let c[i] = values[i]-mean when centered,
//     or c[i] = values[i] otherwise. Let S[k] be the sum of c[i]*c[i-k] for
//     i=k,...,n-1, and let S[0] be the sum of c[i]*c[i].
//   - CommonDenominator returns r[k] = S[k]/S[0]. With zero extension of c,
//     its finite Toeplitz matrices are positive semidefinite in exact arithmetic.
//   - LagAdjusted returns r[k] = (n/(n-k))*(S[k]/S[0]). Its lag-specific pair
//     correction generally breaks the Toeplitz PSD guarantee; magnitudes above
//     one are valid results and must never be clipped or reinterpreted as errors.
//   - NoCenter uses raw second moments. Result.Mean is exactly zero in this mode,
//     even for nonzero constant observations; it does not report a fitted mean.
//
// Shape and result laws:
//   - The input slice must be non-nil, nonempty, and contain only finite float64
//     values. maxLag is inclusive and must satisfy 0 <= maxLag < len(values).
//   - On success Values and PairCounts have length maxLag+1, Values[0] is exactly
//     1, and PairCounts[k] is exactly n-k. Pair counts are structural evidence,
//     not statistical degrees of freedom or confidence bounds.
//   - Centered constant windows (including a singleton) have zero variance and
//     are rejected. Uncentered all-zero windows are rejected; uncentered nonzero
//     constants, including singletons, have well-defined second-moment ratios.
//
// Numeric and determinism laws:
//   - The direct kernel selects an overflow-aware midpoint, normalizes the
//     full-window deviations, and uses compensated mean and cross-product
//     reductions. It does not materialize potentially overflowing raw S[0].
//   - A rounded Result.Mean is descriptive metadata: internal centering uses a
//     scaled mean shift and may preserve information lost by values[i]-Mean.
//     The computed coefficients are finite on success, but they are not claimed
//     to be correctly rounded for every adversarial float64 sequence.
//   - Input extrema, mean terms, lag indices, and valid endpoint pairs are
//     traversed in ascending sample-index order. There is no randomness, map
//     iteration, parallel reduction, hidden estimator selection, or fallback.
//   - Exact-arithmetic Toeplitz PSD does not certify that rounded float64
//     coefficients produce a numerically PSD matrix at every dimension.
//
// Error and partial-result laws:
//   - Validation order is invalid Denominator / negative maxLag, nil input,
//     empty input, maxLag >= n, nonfinite observation, zero denominator, then
//     arithmetic evaluation.
//   - ErrInvalidOption, ErrNilInput, ErrEmptyInput, ErrNonFinite,
//     ErrZeroVariance, and ErrOverflow classify the corresponding failures.
//     Use errors.Is; never classify error messages by their text.
//   - All failures return a nil *Result and a non-nil error. No partial lag
//     coefficients, automatically removed samples, or substituted defaults
//     are published on error.
//
// Ownership, concurrency, and complexity:
//   - Inputs are read-only and are never retained or mutated. Returned slices
//     are detached and may be independently changed by the caller. Clone makes
//     separate copies of both slices; IsNil safely handles typed nil receivers.
//   - Independent Compute calls can run concurrently over stable input slices.
//     Concurrent caller mutation of inputs, or unsynchronized writes to a
//     shared Result, are unsupported. No mutable global state or locks exist.
//   - For n=len(values) and m=maxLag, direct computation costs O(n*(m+1)) time
//     and O(n+m) additional memory, including scaled deviations and output.
//     There is no FFT fast path; near-full-lag searches cost quadratic time.
//
// Non-goals and integration boundaries:
//   - A lag measures sample positions, not seconds, timestamps, or calendar
//     intervals. Callers must establish equally spaced, complete observations
//     (for example with series) before using the result as a temporal ACF.
//   - The estimator fits the entire supplied window; unlike causal per-sample
//     series transforms, adding future observations may change all prior lags.
//   - The package does not resample, impute missing values, infer ARMA orders,
//     calculate PACF, spectral density, p-values, confidence bands, or certify
//     stationarity. Downstream PSD-sensitive solvers must select
//     CommonDenominator explicitly and enforce their own numeric requirements.
//
// AI-Hints:
//   - Do not replace the single global mean with lag-specific pairwise means:
//     doing so computes a different estimator and invalidates the PSD proof.
//   - Do not clamp LagAdjusted to [-1,1] or silently pass it to a Toeplitz
//     solver. The pair-count multiplier changes the mathematical objective.
//   - Do not square unscaled deviations: finite samples near 1e154 or 1e308
//     can overflow before a perfectly representable ratio is calculated.
//   - For holdout or incident analysis, fix the observation window and preserve
//     the Denominator and PairCounts metadata alongside the coefficient vector.
package acf
