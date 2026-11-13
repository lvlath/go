// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package acf implements deterministic direct sample autocorrelation estimators.
//
// Domain: fixed, regularly indexed finite float64 observation windows. Values
// are observed in input order; no timestamps, missing-value fill, or streaming
// state are inferred. Compute is the canonical facade and Result the artifact.
//
// Mathematical law: one global sample mean centers every lag unless NoCenter
// is true. CommonDenominator uses S_k/S_0 and is Toeplitz-PSD compatible in
// exact arithmetic. LagAdjusted uses n*S_k/((n-k)*S_0) and can exceed [-1,1].
// Float64 rounding is not a certificate of numerical Toeplitz PSD.
//
// Shape: the sample is non-nil and non-empty; 0 <= maxLag < n. The result
// contains maxLag+1 coefficients and pair counts n-k; lag zero is exactly 1.
//
// Numeric policy: NaN and infinity are rejected. Scaled deviations and
// compensated reductions prevent avoidable overflow in unscaled squares.
// Zero dispersion returns ErrZeroVariance; unrepresentable results ErrOverflow.
// Validation order: options, nil, empty, lag shape, finite data, domain.
// Any error returns a nil result; no partial artifact is published.
//
// Ownership and concurrency: input is never mutated or retained. Result slices
// belong to callers. Independent calls are concurrency-safe provided their
// inputs are not concurrently mutated. There is no shared workspace or lock.
//
// Complexity: direct O(n*(maxLag+1)) time and O(n+maxLag) extra memory.
// No FFT, PACF, spectral density, stationarity test, or confidence interval
// is computed by this package.
//
// AI-Hints: never silently replace CommonDenominator with LagAdjusted for
// Levinson-Durbin; never clamp LagAdjusted; never fit per-lag pairwise means.
package acf
