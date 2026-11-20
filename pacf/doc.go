// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pacf estimates finite-window sample partial autocorrelation by
// Durbin-Levinson recursion over biased sample autocovariance.
// It provides reflection coefficients and the prediction-error variance curve
// needed to assess linear dependence after controlling for all shorter lags.
//
// # Mathematical charter
//
// Given a window x[0:n] and c[t]=x[t]-mean when Center=true (or c[t]=x[t]
// when Center=false), the only covariance estimator is
//
//	gamma[k] = (1/n) * sum(t=k..n-1, c[t]*c[t-k]).
//
// Every lag shares the same divisor n. Extend centered/raw observations by
// zero outside the finite input window. A quadratic form over the resulting
// symmetric Toeplitz covariances equals a sum of squares of shifted samples,
// divided by n: it is therefore positive semidefinite in exact arithmetic.
// A nonzero finite sequence gives linearly independent finite shifts and a
// positive-definite Toeplitz system in exact arithmetic, but this is not a
// guarantee of numerical conditioning in rounded float64 arithmetic.
// Changing the divisor to n-k destroys the general Gram property. The
// solver computes order-m reflection coefficients and variances by
//
//	kappa[m] = (gamma[m] - sum(j=1..m-1, a[m-1,j]*gamma[m-j])) / V[m-1],
//	a[m,j]  = a[m-1,j] - kappa[m]*a[m-1,m-j],
//	V[m]    = V[m-1]*(1-abs(kappa[m]))*(1+abs(kappa[m])).
//
// The previous coefficient row is never modified while it is being read:
// two disjoint coefficient buffers exchange their roles only after a full
// order update. Values[1] is mathematically gamma[1]/gamma[0] when that
// order is requested, whereas subsequent orders remove shorter-lag effects.
// A single shared scale avoids premature overflow in squared observations;
// physical variances are recovered using separate binary exponents. Centered
// means are fitted once on the entire supplied training window using bounded
// midpoint coordinates and compensated reductions.
//
// # Public surface and result laws
//
// Compute is the canonical facade. ComputeCentered and ComputeUncentered
// explicitly select one of the two center policies; ComputeAtLag projects a
// single coefficient after evaluating the complete canonical prefix through
// its requested lag, with no direct single-lag solver or O(n) shortcut. The
// wrappers preserve Compute's validation precedence, sentinel identity,
// and failure atomicity. No wrapper changes the denominator or numeric kernel.
//
// Options{MaxLag:0, Center:false} is the zero-value policy: only uncentered lag
// zero is requested. MaxLag is inclusive and must lie in [0,n-1]. A successful
// Result has Count=n, Values[0]=1, two separately owned slices of length L+1,
// and positive finite PredictionVariance[m] in squared input units. A failed
// operation returns nil Result, including on context cancellation; no usable
// prefix is published. The two output slices never alias each other, inputs,
// or future computations. A Result supports nil-safe IsNil (the core.Nilable
// pointer-presence protocol) and detached Clone; an allocated zero-value
// Result is non-nil but not a valid completed computation.
//
// # Error and numeric policy
//
// Context precedence is nil context (ErrInvalidOption), canceled context
// (ctx.Err), nil input (ErrNilInput), empty input (ErrEmptyInput), invalid lag
// (ErrInvalidOption), nonfinite observations (ErrNonFinite), then computation.
// Zero centered energy or raw second moment returns ErrZeroVariance. An
// invalid Levinson step (abs(kappa)>=1 or nonpositive next variance) returns
// ErrSingular; nonfinite arithmetic and unrepresentable public prediction
// variances return ErrOverflow. Errors are classified with errors.Is, never
// error-string inspection. A positive variance that underflows to zero is not
// silently clamped. Workspace integer byte products are checked before make;
// runtime out-of-memory failures are not converted into ordinary errors.
//
// # Ownership, concurrency and determinism
//
// Inputs are read-only and never retained or mutated. Independently invoked
// calculations on the same immutable slice are safe; concurrent mutation of
// inputs or exported Result fields requires caller synchronization. There is
// no mutable package-global state, RNG, reflection, panic, or recovery.
// Samples are scanned by increasing index, covariances by increasing lag and
// endpoint, and coefficient updates by increasing order and coefficient index.
// The same binary64 inputs and execution environment yield the same output.
//
// # Complexity and non-goals
//
// For n observations and inclusive MaxLag=L, time is O(n*(L+1)+L^2) and
// additional memory, including both returned slices, is O(L+1). No n-sized
// deviation vector is constructed. The package does not estimate Burg or
// regression-based PACF, confidence intervals, statistical significance,
// automatic AR order, nonstationarity, or out-of-sample forecast quality.
// The theoretical PACF of a stationary AR(p) process cuts off above p;
// finite-sample estimates fluctuate and have no exact sample cutoff. This
// package does not fit AR coefficients, set confidence thresholds, or select
// arima.Order. Input lags are positions, not elapsed durations; timestamp
// cadence, complete-window selection, training/holdout separation, and any
// model deployment policy belong to callers. A near-singular but theoretically
// positive Toeplitz system can conservatively fail in binary64 instead of
// publishing numerically unreliable coefficients.
//
// AI-Hints:
//   - Never change the n denominator to n-k or infer Center from observations.
//   - Never update both mirrored AR coefficients in one buffer in place.
//   - Treat V[m] as in-sample one-step linear residual variance, not a
//     confidence bound, forecast error estimate, or fitted AR coefficient.
//   - Compare acf.CommonDenominator and PACF only with matching centering;
//     do not pass lag-adjusted ACF ratios into a Toeplitz solver.
//   - Keep centering within the training window and distinguish train/holdout data.
package pacf
