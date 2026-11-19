// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package arima estimates and forecasts fixed-order univariate nonseasonal ARIMA(p,d,q)
// models using explicitly conditional Gaussian likelihood, bounded local optimization,
// strict Schur root checks and anchored inverse differencing.
//
// Domain and operations:
//   - Order fixes p,q in [0,8] and d in [0,2]; Order{} denotes white noise.
//   - Fit estimates c, AR, MA and sigma^2 from a finite, equally spaced sample.
//   - New constructs an immutable model from complete validated physical parameters.
//   - Model.Forecast predicts levels and marginal future-innovation variances.
//   - FitWithDrift, FitWithoutDrift, FitARMA and FitIntegrated select explicit common
//     configurations; ForecastNext is the same canonical forecast with horizon one.
//
// Mathematical laws:
//   - y_t=Delta^d(x_t)=c+sum(phi_i*y_{t-i})+eps_t+sum(theta_j*eps_{t-j}).
//   - Drift is c in differenced coordinates, not the unconditional level mean.
//   - Residuals before b=max(p,q) are exactly zero and excluded from SSE; only
//     N_eff=(n-d)-b observations enter conditional Gaussian likelihood.
//   - The profile variance is SSE/N_eff and zero SSE returns ErrSingular.
//   - All AR/MA polynomial roots must lie strictly outside the unit circle;
//     Schur step-down rejects unit roots without clipping.
//   - At d=1, forecast levels integrate from the last actual level. At d=2,
//     integration also retains the last actual first difference.
//   - Future innovations have conditional expectation zero. Variance is
//     sigma^2 times cumulative squared integrated impulse responses, excluding
//     fitted-parameter uncertainty, with no dense horizon covariance matrix.
//
// Numeric, context and error laws:
//   - Nil context -> ErrInvalidOption; canceled context -> ctx.Err().
//   - Context, options, presence, sample length and finiteness are checked in
//     documented order before the numerical kernel runs.
//   - ErrNilInput, ErrEmptyInput, ErrInsufficientData, ErrNonFinite,
//     ErrNonStationary, ErrNonInvertible, ErrSingular and ErrOverflow distinguish
//     independently actionable failure categories; use errors.Is.
//   - Except for ErrNotConverged (last accepted model and Converged=false),
//     Fit returns no partial result on error. Forecast never returns a partial result.
//
// Determinism, ownership and concurrency:
//   - Residuals, polynomial lags, gradients and optimizer corrections are visited
//     in ascending deterministic order; no RNG, wall clock or global mutable state.
//   - All input slices are read-only and never retained. Every result slice is
//     detached, and Model.Parameters deep-copies both coefficient vectors.
//   - Models are immutable and safe for independent concurrent Forecast calls.
//     Caller mutation of an input slice during the call is unsupported.
//   - FitResult and ForecastResult implement pointer-only nilability with IsNil.
//
// Complexity and non-goals:
//   - Fit: O((I+1)*N*(p+q+1)^2) time and O(N*(p+q+1)+I) space.
//   - Forecast: O((N+H)*(p+q+d+1)) time and O(N+H) space, H forecast steps.
//   - No auto-order search, seasonal model, regressors, Kalman diffuse likelihood,
//     online retraining, parameter-uncertainty bands or production risk strategy.
//
// AI-Hints:
//   - Never treat a drift of c as the long-run growth when AR terms exist:
//     stationary differenced mean equals c/(1-sum(phi)), when defined.
//   - Do not initialize integrated forecasts at zero; retain historical anchors.
//   - Do not replace conditional initialization with exact Kalman MLE without
//     introducing a separately versioned, explicit method.
//   - Numerical correctness comes from external analytical oracles, not from
//     comparing two pathways through this package's same internal kernel.
package arima
