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
// Every lag shares the same divisor n. In exact arithmetic, this finite-sample
// autocovariance is a Gram sequence, hence its symmetric Toeplitz matrix is
// positive semidefinite. A divisor n-k generally lacks that guarantee. The
// solver computes order-m reflection coefficients and variances by
//
//	kappa[m] = (gamma[m] - sum(j=1..m-1, a[m-1,j]*gamma[m-j])) / V[m-1],
//	a[m,j]  = a[m-1,j] - kappa[m]*a[m-1,m-j],
//	V[m]    = V[m-1]*(1-abs(kappa[m]))*(1+abs(kappa[m])).
//
// The previous coefficient row is never modified while it is being read.
// A single shared scale avoids premature overflow in squared observations;
// physical variances are recovered using separate binary exponents. Centered
// means are fitted once on the entire supplied training window using bounded
// midpoint coordinates and compensated reductions.
//
// # Public surface and result laws
//
// Compute is the canonical facade. ComputeCentered and ComputeUncentered
// explicitly select one of the two center policies; ComputeAtLag projects a
// single coefficient from a complete canonical calculation. They never change
// denominator convention or use a different numerical kernel.
//
// Options{MaxLag:0, Center:false} is the zero-value policy: only uncentered lag
// zero is requested. MaxLag is inclusive and must lie in [0,n-1]. A successful
// Result has Count=n, Values[0]=1, two separately owned slices of length L+1,
// and positive finite PredictionVariance[m] in squared input units. A failed
// operation returns nil Result, including on context cancellation; no usable
// prefix is published. A Result supports nil-safe IsNil and detached Clone.
//
// # Error and numeric policy
//
// Context precedence is nil context (ErrInvalidOption), canceled context
// (ctx.Err), nil input (ErrNilInput), empty input (ErrEmptyInput), invalid lag
// (ErrInvalidOption), nonfinite observations (ErrNonFinite), then computation.
// Zero centered energy or raw second moment returns ErrZeroVariance. An
// invalid Levinson step returns ErrSingular; nonrepresentable arithmetic and
// prediction variances return ErrOverflow. Errors are classified with
// errors.Is. A positive variance that underflows to zero is not silently
// clamped. OOM is not converted into a normal error; integer allocation
// products are validated before the workspace is created.
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
// The theoretical PACF of an AR(p) process cuts off above p; finite-sample
// estimates are noisy and do not have an exact sample cutoff. A near-singular
// but theoretically positive Toeplitz system can conservatively fail in
// binary64 rather than publish numerically unreliable coefficients.
//
// AI-Hints:
//   - Never change the n denominator to n-k or infer Center from observations.
//   - Never update both mirrored AR coefficients in one buffer in place.
//   - Treat variance as a squared-unit predictive moment, not a confidence bound.
//   - Keep centering within the training window and distinguish train/holdout data.
package pacf
