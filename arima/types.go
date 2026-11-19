// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package arima declares the typed public contract of fixed-order ARIMA models.
// This file specifies result ownership, fitted parameters, diagnostics and nil-state rules.
package arima

// Order specifies the polynomial degrees of one univariate ARIMA(p,d,q) model.
// The three degrees are independent; an all-zero order denotes white noise,
// never automatic identification or a default seasonal configuration.
// Implementation:
//   - Stage 1: callers select finite integer degrees through public fields.
//   - Stage 2: Fit/New validate p,q in [0,8] and d in [0,2] before arithmetic.
//   - Stage 3: the accepted order is preserved in Model.Parameters.
//
// Behavior highlights:
//   - Differencing is always nonseasonal and causal, with unit observation spacing.
//
// Inputs:
//   - AR: p, number of lagged differenced observations.
//   - Difference: d, repeated first differences of the original level series.
//   - MA: q, number of lagged conditional innovations.
//
// Returns:
//   - Value-copy order used by fitted models and forecasts.
//
// Errors:
//   - ErrInvalidOption from Fit/New for an out-of-range degree.
//
// Determinism:
//   - Lag coefficients always use increasing lag order starting at one.
//
// Complexity:
//   - Time O(1), Space O(1) for the value itself.
//
// Notes:
//   - The zero value is a valid, explicit ARIMA(0,0,0).
//
// AI-Hints:
//   - Never reinterpret Order{} as automatic order selection.
type Order struct {
	AR         int // p: lags of differenced observations, 0..8.
	Difference int // d: first-difference applications, 0..2.
	MA         int // q: lags of past conditional residuals, 0..8.
}

// Options defines fixed-order conditional Gaussian estimation policy.
// It changes model coefficients through explicit drift inclusion and solver limits,
// without selecting an alternative likelihood or automatic order search.
// Implementation:
//   - Stage 1: collect caller-selected order and optimization controls.
//   - Stage 2: canonical Fit finalizes zeros and rejects invalid controls.
//   - Stage 3: a local immutable policy is passed to the fitting kernel.
//
// Behavior highlights:
//   - MaxIterations=0 means 200 and Tolerance=0 means 1e-7; negative values reject.
//
// Inputs:
//   - Order: fixed p,d,q; IncludeDrift: fit c if true, else c=0 exactly.
//   - MaxIterations: nonnegative limit of accepted L-BFGS steps.
//   - Tolerance: nonnegative finite relative gradient threshold.
//
// Returns:
//   - A value-copy configuration; no mutable global options exist.
//
// Errors:
//   - ErrInvalidOption if order, iteration budget or tolerance is invalid.
//
// Determinism:
//   - No random restarts, data-driven order selection or hidden fallback method.
//
// Complexity:
//   - Time O(1), Space O(1) to finalize.
//
// Notes:
//   - When p=q=0 the solution is analytic and MaxIterations does not constrain it.
//
// AI-Hints:
//   - Change numerical budgets through Fit, not by changing a returned Model.
type Options struct {
	Order         Order   // Fixed ARIMA(p,d,q) order (zero means white noise).
	IncludeDrift  bool    // Fit c in the differenced equation; false fixes c=0.
	MaxIterations int     // Accepted L-BFGS steps; zero selects 200.
	Tolerance     float64 // Relative gradient threshold; zero selects 1e-7.
}

// Parameters contains physical coefficients of an initialized ARIMA model.
// AR and MA coefficients follow y[t]=c+sum(phi*yLag)+eps[t]+sum(theta*epsLag).
// The positive innovation variance governs future shocks, not parameter uncertainty.
// Implementation:
//   - Stage 1: New validates coefficient lengths and Schur stability domains.
//   - Stage 2: New owns copies of both coefficient slices.
//   - Stage 3: Parameters returns detached slices to callers.
//
// Behavior highlights:
//   - Drift is an intercept in differenced coordinates, not an original-level mean.
//
// Inputs:
//   - AR/MA: increasing-lag physical coefficients matching Order.AR/Order.MA.
//   - Drift: finite c; InnovationVariance: finite sigma^2 strictly greater than zero.
//
// Returns:
//   - A coefficient snapshot whose slices can safely be mutated by the caller.
//
// Errors:
//   - ErrInvalidOption, ErrNonFinite, ErrSingular, ErrNonStationary, ErrNonInvertible from New.
//
// Determinism:
//   - Fixed coefficient positions, no random initialization for New.
//
// Complexity:
//   - Time/Space O(p+q) to clone the coefficient slices.
//
// Notes:
//   - Never mutate caller-owned coefficient slices concurrently with New.
//
// AI-Hints:
//   - For d=1 and nonzero AR, average growth is c/(1-sum(phi)), not c.
type Parameters struct {
	AR                 []float64 // phi_1..phi_p; 1-sum(phi_i*z^i) has roots outside |z|=1.
	MA                 []float64 // theta_1..theta_q; 1+sum(theta_j*z^j) has roots outside |z|=1.
	Drift              float64   // c: intercept of Delta^d x, not the level-series mean.
	InnovationVariance float64   // sigma^2 > 0: conditional innovation variance.
	Order              Order     // Fixed polynomial and differencing degrees.
}

// Diagnostics records the accepted conditional objective trajectory during Fit.
// Objectives are negative profiled conditional Gaussian log likelihood values;
// each accepted step adds one element after the initial objective.
// Implementation:
//   - Stage 1: record the finite initial candidate objective.
//   - Stage 2: append only Armijo-accepted objectives and count those iterations.
//   - Stage 3: detach the trajectory on publication, including nonconvergence.
//
// Behavior highlights:
//   - len(Objective)=Iterations+1; none of the burn-in observations enters SSE.
//
// Inputs:
//   - Produced by Fit; callers do not supply this structure to kernels.
//
// Returns:
//   - Caller-owned Objective slice, independent of optimizer workspace.
//
// Errors:
//   - Converged=false and ErrNotConverged signal accepted but unconverged fitting.
//
// Determinism:
//   - Objective values follow accepted line searches in chronological order.
//
// Complexity:
//   - Time/Space O(I) to detach I+1 accepted values.
//
// Notes:
//   - Iterations excludes rejected line-search evaluations.
//
// AI-Hints:
//   - Do not count conditioning observations when computing EffectiveN.
type Diagnostics struct {
	Iterations int       // Number of accepted L-BFGS steps (zero for analytic fits).
	Converged  bool      // True only when the relative gradient criterion was satisfied.
	Objective  []float64 // Initial and accepted negative profiled log likelihoods.
	EffectiveN int       // len(values)-d-max(p,q) points in the conditional objective.
}

// Model is an immutable, versioned collection of validated ARIMA parameters.
// It holds no historical observations and may be shared by concurrent forecasts
// that each supply a separate, caller-owned observation prefix.
// Implementation:
//   - Stage 1: New validates a complete Parameters value.
//   - Stage 2: the model deep-copies coefficient slices and computes a content ID.
//   - Stage 3: read-only methods publish detached result arrays.
//
// Behavior highlights:
//   - A zero-value Model is uninitialized and cannot be used for Forecast.
//
// Inputs:
//   - Construct through New or obtain from a successful/partially valid Fit.
//
// Returns:
//   - An immutable read-only object with a stable model fingerprint.
//
// Errors:
//   - Forecast returns ErrInvalidOption for an uninitialized Model.
//
// Determinism:
//   - Identical parameters and method version produce identical IDs/forecasts.
//
// Complexity:
//   - Storage O(p+q); Forecast allocates its own O(N+H) buffers.
//
// Notes:
//   - Concurrent mutation of input history by callers is not supported.
//
// AI-Hints:
//   - Do not construct &Model{} directly; New enforces root and variance constraints.
type Model struct {
	parameters Parameters // Owned physical coefficients, never returned by reference.
	id         string     // SHA-256 of canonical coefficients and method identifier.
}

// FitResult combines a fitted immutable model with diagnostics and log likelihood.
// On ErrNotConverged the model is the last accepted finite candidate and may
// be inspected, but Converged=false must remain visible to downstream decisions.
// Implementation:
//   - Stage 1: optimizer assembles owned accepted coefficients and diagnostics.
//   - Stage 2: New freezes the accepted parameters and provides the model ID.
//   - Stage 3: Fit returns either success or explicitly marked nonconvergence.
//
// Behavior highlights:
//   - Other errors return nil result, even when temporary workspaces existed.
//
// Inputs:
//   - Constructed by Fit; all exported values are caller-owned outputs.
//
// Returns:
//   - Model: immutable; Diagnostics.Objective: detached; LogLikelihood: scalar.
//
// Errors:
//   - ErrNotConverged permits partial result; other errors do not.
//
// Determinism:
//   - The result reflects the latest accepted optimizer step, not the latest trial.
//
// Complexity:
//   - Result storage O(p+q+I); IsNil O(1).
//
// Notes:
//   - IsNil is pointer nilability, not a validation of user-built struct fields.
//
// AI-Hints:
//   - Never treat a returned ErrNotConverged model as proven converged.
type FitResult struct {
	Model         *Model      // Validated immutable model; non-nil on success/nonconvergence.
	Diagnostics   Diagnostics // Accepted optimizer steps and owned objective history.
	LogLikelihood float64     // Conditional Gaussian log likelihood at the fitted profile variance.
}

// ForecastResult contains causal level predictions and their marginal variances.
// Both arrays are detached and aligned by horizon index h-1; a failed atomic
// forecast never publishes a partially populated result.
// Implementation:
//   - Stage 1: reconstruct conditional innovations at the supplied history origin.
//   - Stage 2: integrate predictions using latest observed historical anchors.
//   - Stage 3: accumulate squared integrated impulse response for each horizon.
//
// Behavior highlights:
//   - Mean/Variance lengths both equal Horizon; variances exclude parameter risk.
//
// Inputs:
//   - Produced by Forecast; never aliases its history or Model.Parameters.
//
// Returns:
//   - Caller-owned arrays; a nil receiver is classified by IsNil.
//
// Errors:
//   - No partial ForecastResult is returned on error.
//
// Determinism:
//   - Horizon indices ascend from first future time to last.
//
// Complexity:
//   - Result storage O(H); IsNil O(1).
//
// Notes:
//   - A predicted marginal variance is not a confidence interval by itself.
//
// AI-Hints:
//   - For interval calculations add a separately justified distributional quantile.
type ForecastResult struct {
	Mean     []float64 // Estimated original levels at horizons 1..H.
	Variance []float64 // Marginal future innovation error variances for levels 1..H.
	Horizon  int       // H, equal to len(Mean) and len(Variance) on success.
}

// IsNil reports whether the FitResult pointer is nil, including typed-nil interfaces.
// This implements the pointer-only core.Nilable convention without inspecting Model.
// Implementation:
//   - Stage 1: compare the receiver pointer to nil.
//   - Stage 2: return the result without dereferencing any fields.
//
// Behavior highlights:
//   - A caller-constructed &FitResult{} is non-nil but not a successful fit.
//
// Inputs:
//   - r: possibly nil *FitResult receiver.
//
// Returns:
//   - bool: true if and only if r==nil.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pure pointer identity check.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Do not infer convergence from IsNil; inspect Diagnostics.Converged and error.
//
// AI-Hints:
//   - Nilability must not depend on publicly mutable Model/Diagnostics fields.
func (r *FitResult) IsNil() bool { return r == nil }

// IsNil reports whether the ForecastResult pointer is nil, including typed-nil interfaces.
// Its semantics agree with the pointer-only nilability convention of other results.
// Implementation:
//   - Stage 1: compare the receiver pointer to nil.
//   - Stage 2: return without reading any potentially modified exported fields.
//
// Behavior highlights:
//   - A non-nil but empty manually assembled ForecastResult is not a valid forecast.
//
// Inputs:
//   - r: possibly nil *ForecastResult receiver.
//
// Returns:
//   - bool: true if and only if r==nil.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pure pointer identity check.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - IsNil does not attempt to validate Mean/Variance shapes.
//
// AI-Hints:
//   - Do not use this predicate in place of checking the Forecast error.
func (r *ForecastResult) IsNil() bool { return r == nil }

// Compile-time nilability compatibility, without coupling arithmetic to core.Graph.
var (
	_ interface{ IsNil() bool } = (*FitResult)(nil)
	_ interface{ IsNil() bool } = (*ForecastResult)(nil)
)
