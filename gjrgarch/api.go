// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gjrgarch defines the public model, filtering, forecasting, and QMLE facades.
// Each wrapper delegates to a single canonical kernel without changing numeric policy.
package gjrgarch

import (
	"context"
	"math"
)

// New constructs an immutable stationary GJR-GARCH(1,1) model.
// It binds the validated parameters to a policy-versioned SHA-256 identity.
//
// Implementation:
//   - Stage 1: validate finite coefficients and the asymmetric nonnegative shock law.
//   - Stage 2: validate symmetric-innovation stationarity and long-run variance range.
//   - Stage 3: publish a model storing value-copied parameters and deterministic identity.
//
// Behavior highlights:
//   - Gamma can be negative when Alpha+Gamma >= 0.
//   - No QMLE variance floor is imposed by New; Fit applies its own stricter policy.
//
// Inputs:
//   - p: finite GJR-GARCH coefficients, with Omega in squared-return units.
//
// Returns:
//   - *Model: immutable, concurrently readable model, or nil on error.
//
// Errors:
//   - ErrNonFinite: a caller-supplied parameter is NaN or infinite.
//   - ErrDomain: shock coefficients or stationarity are invalid.
//   - ErrOverflow: a derived coefficient or unconditional variance is unrepresentable.
//
// Determinism:
//   - Parameters are validated in fixed field order; IDs hash canonical IEEE bits.
//
// Complexity:
//   - Time O(1), Space O(1), including the fixed-size SHA-256 digest.
//
// Notes:
//   - The constructor does not modify or retain a pointer to caller data.
//
// AI-Hints:
//   - Do not reject negative Gamma unless Alpha+Gamma is negative.
//   - Never use model IDs as an estimate of statistical adequacy.
func New(p Parameters) (*Model, error) {
	rho, err := stationary(p)
	if err != nil {
		return nil, err
	}
	return &Model{p: p, id: modelIdentity(p), rho: rho}, nil
}

// ID returns the parameter- and numeric-policy-bound SHA-256 model identity.
// It is suitable for rejecting stale or foreign persisted posterior states.
//
// Implementation:
//   - Stage 1: test the receiver for nil.
//   - Stage 2: return the precomputed immutable string without re-hashing.
//
// Behavior highlights:
//   - Nil receiver maps to the empty identity without dereferencing.
//
// Inputs:
//   - m: immutable model receiver, possibly nil.
//
// Returns:
//   - string: 64 lowercase hexadecimal digits, or an empty string for a nil model.
//
// Errors:
//   - None; nil receiver is a defined empty value.
//
// Determinism:
//   - The identity is fixed at New; normalized negative zero hashes as positive zero.
//
// Complexity:
//   - Time O(1), Space O(1); no output allocation.
//
// Notes:
//   - The ID binds parameters and numeric-policy version, not persisted-state authenticity.
//
// AI-Hints:
//   - Compare state.ModelID against ID rather than trusting user-supplied state fields.
func (m *Model) ID() string {
	if m == nil {
		return ""
	}
	return m.id
}

// Parameters returns a value copy of the model's validated coefficients.
// Changing that copy never mutates the model or its identity.
//
// Implementation:
//   - Stage 1: Check receiver.
//   - Stage 2: Copy the five stored float64 fields to a value.
//
// Behavior highlights:
//   - Changing the returned scalar value cannot alter model state.
//
// Inputs:
//   - m: immutable model receiver, possibly nil.
//
// Returns:
//   - Parameters: immutable model values copied into caller-owned scalar fields;
//     the zero Parameters value for a nil receiver.
//
// Errors:
//   - None; nil receiver is represented by the zero value.
//
// Determinism:
//   - Field values and signed-zero representation are preserved exactly.
//
// Complexity:
//   - Time O(1), Space O(1); no heap ownership transfer.
//
// Notes:
//   - The zero value returned for nil is not a valid stationary model.
//
// AI-Hints:
//   - Construct a new Model after changing the returned Parameters value.
func (m *Model) Parameters() Parameters {
	if m == nil {
		return Parameters{}
	}
	return m.p
}

// UnconditionalVariance returns Omega/(1-rho) for the symmetric-innovation law.
// It is the documented stationary initialization for Gaussian QMLE.
//
// Implementation:
//   - Stage 1: reject a nil model.
//   - Stage 2: divide its validated intercept by the positive stationarity gap.
//
// Behavior highlights:
//   - This is the stationary expected variance, not a data-driven seed.
//
// Inputs:
//   - m: validated model receiver, or nil.
//
// Returns:
//   - float64: positive finite long-run conditional variance, or zero on failure.
//
// Errors:
//   - ErrStateVersion: the model receiver is nil.
//
// Determinism:
//   - One immutable parameter read and one arithmetic division.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This is an unconditional mean, not a realized variance or a VaR estimate.
//
// AI-Hints:
//   - Use this to initialize Filter when a sample-independent stationary start is desired.
func (m *Model) UnconditionalVariance() (float64, error) {
	if m == nil {
		return 0, ErrStateVersion
	}
	return m.p.Omega / (1 - m.rho), nil
}

// InitialState records the first observed return at a caller-supplied h[0].
// The resulting posterior state contains epsilon[0] and h[0], not h[1].
//
// Implementation:
//   - Stage 1: check model presence and finite return/variance values.
//   - Stage 2: require positive variance and compute epsilon[0]=r[0]-Mean.
//   - Stage 3: bind the state to ModelID with Steps=1.
//
// Behavior highlights:
//   - The observed residual is retained for the next transition, not used to revise initial variance.
//
// Inputs:
//   - observedReturn: first finite return, in the model's return units.
//   - variance: strictly positive finite h[0], in squared-return units.
//
// Returns:
//   - State: detached value corresponding to exactly one consumed observation.
//
// Errors:
//   - ErrStateVersion: nil model or nonpositive initial variance.
//   - ErrNonFinite: a caller-supplied return or variance is nonfinite.
//   - ErrOverflow: subtraction of valid finite values overflows.
//
// Determinism:
//   - ModelID, variance, and residual are recorded in fixed order.
//
// Complexity:
//   - Time O(1), Space O(1); no heap allocation.
//
// Notes:
//   - The function does not infer h[0] from any later return.
//
// AI-Hints:
//   - For stationary initialization use UnconditionalVariance, then InitialState.
func (m *Model) InitialState(observedReturn, variance float64) (State, error) {
	if m == nil {
		return State{}, ErrStateVersion
	}
	if !finite(observedReturn) || !finite(variance) {
		return State{}, ErrNonFinite
	}
	if variance <= 0 {
		return State{}, ErrStateVersion
	}
	return m.initialStateKernel(observedReturn, variance)
}

// Step advances the posterior state by one observed return without looking ahead.
// It computes h[t+1] from stored epsilon[t], h[t] before recording epsilon[t+1].
//
// Implementation:
//   - Stage 1: validate ModelID, prior state, new return, and counter capacity.
//   - Stage 2: apply the canonical variance recurrence to the PRIOR residual.
//   - Stage 3: store the new residual and return the updated independent State.
//
// Behavior highlights:
//   - Failed calls publish the zero State and cannot mutate the caller-held value.
//
// Inputs:
//   - state: caller-owned State for observation t; Steps>=1.
//   - observedReturn: finite observation at t+1, in return units.
//
// Returns:
//   - State: new h[t+1], epsilon[t+1] and Steps+1; zero State on error.
//
// Errors:
//   - ErrStateVersion: foreign ID, invalid variance, or corrupted prior state.
//   - ErrNonFinite: the new observed return is NaN or infinite.
//   - ErrOverflow: step counter, innovation, or variance is unrepresentable.
//
// Determinism:
//   - One previous-state transition; the indicator uses epsilon[t] < 0 exactly.
//
// Complexity:
//   - Time O(1), Space O(1), with zero successful-path allocations.
//
// Notes:
//   - The input state is copied; independent models support concurrent read-only use.
//
// AI-Hints:
//   - Do not recompute h[t+1] from observedReturn-Mean: that leaks the new shock.
func (m *Model) Step(state State, observedReturn float64) (State, error) {
	if err := m.stateValid(state); err != nil {
		return State{}, err
	}
	if !finite(observedReturn) {
		return State{}, ErrNonFinite
	}
	if state.Steps == ^uint64(0) {
		return State{}, ErrOverflow
	}
	return m.stepKernel(state, observedReturn)
}

// Filter computes the conditional variance assigned to each observed return.
// The first output is exactly initialVariance; return[t] affects only h[t+1].
//
// Implementation:
//   - Stage 1: enforce context, model, input presence, finiteness, and initial variance.
//   - Stage 2: traverse observations in ascending index order using the Step kernel.
//   - Stage 3: publish an owning variance slice only on complete success.
//
// Behavior highlights:
//   - A singleton still validates its first residual; runtime errors discard the entire output.
//
// Inputs:
//   - ctx: non-nil, active context; cancellation is checked between observations.
//   - returns: non-nil, nonempty, finite, caller-owned read-only observations.
//   - initialVariance: positive finite h[0] in squared-return units.
//
// Returns:
//   - []float64: detached len(returns) sequence h[0]..h[T-1], or nil on failure.
//
// Errors:
//   - ErrInvalidOption and ctx.Err(): invalid or canceled context.
//   - ErrStateVersion: nil model.
//   - ErrNilInput/ErrEmptyInput/ErrNonFinite: invalid observations.
//   - ErrDomain: nonpositive initialVariance.
//   - ErrOverflow: unrepresentable residual or variance during evaluation.
//
// Determinism:
//   - Strictly increasing return index; no map traversal, RNG, or concurrency.
//
// Complexity:
//   - Time O(T), Space O(T), including the detached output allocation.
//
// Notes:
//   - Concurrent mutation of caller-supplied returns is unsupported.
//
// AI-Hints:
//   - Never put the current observation's squared shock into its own h[t].
func (m *Model) Filter(ctx context.Context, returns []float64, initialVariance float64) ([]float64, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	if m == nil {
		return nil, ErrStateVersion
	}
	if err := checkReturns(returns); err != nil {
		return nil, err
	}
	if !finite(initialVariance) {
		return nil, ErrNonFinite
	}
	if initialVariance <= 0 {
		return nil, ErrDomain
	}
	return m.filterKernel(ctx, returns, initialVariance)
}

// FilterStationary filters returns beginning at the model's unconditional variance.
// It is a convenience wrapper for UnconditionalVariance followed by Filter.
//
// Implementation:
//   - Stage 1: validate context and model.
//   - Stage 2: obtain Omega/(1-rho) without using any future return.
//   - Stage 3: delegate input validation, causality, and allocation to Filter.
//
// Behavior highlights:
//   - No sample moment is estimated to choose h[0].
//
// Inputs:
//   - ctx: non-nil context; returns: caller-owned finite nonempty series.
//
// Returns:
//   - []float64: independent conditional-variance series, nil on error.
//
// Errors:
//   - Exactly the errors and precedence of Filter after the model/context checks.
//
// Determinism:
//   - Same index order and numerical recurrence as Filter.
//
// Complexity:
//   - Time O(T), Space O(T).
//
// Notes:
//   - A stationary start is an assumption; it is not a reconstructed pre-sample posterior.
//
// AI-Hints:
//   - Stationary initialization is a modeling choice, not a learned pre-sample state.
func (m *Model) FilterStationary(ctx context.Context, returns []float64) ([]float64, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	initialVariance, err := m.UnconditionalVariance()
	if err != nil {
		return nil, err
	}
	return m.Filter(ctx, returns, initialVariance)
}

// FilterVolatility converts causal variances to their standard-deviation scale.
// It computes sqrt(h[t]), not a realized return or a confidence interval.
//
// Implementation:
//   - Stage 1: delegate all validation and recurrence work to Filter.
//   - Stage 2: transform the private detached variance slice in place.
//
// Behavior highlights:
//   - Returns standard deviation per observation, not an annualized volatility estimate.
//
// Inputs:
//   - ctx, returns, initialVariance: identical to Filter.
//
// Returns:
//   - []float64: owning conditional standard deviations in return units.
//
// Errors:
//   - The identical sentinel and cancellation contract of Filter.
//
// Determinism:
//   - Result indices preserve the chronological Filter order.
//
// Complexity:
//   - Time O(T), Space O(T); no additional slice allocation.
//
// Notes:
//   - Returned values are conditional sigmas, not predictive return quantiles.
//
// AI-Hints:
//   - For percentage-point volatility multiply the returned decimal result by 100.
func (m *Model) FilterVolatility(ctx context.Context, returns []float64, initialVariance float64) ([]float64, error) {
	variances, err := m.Filter(ctx, returns, initialVariance)
	if err != nil {
		return nil, err
	}
	for i := range variances {
		variances[i] = math.Sqrt(variances[i])
	}
	return variances, nil
}

// ForecastVariance forecasts positive conditional variances for H future observations.
// The first horizon knows epsilon[T]; later horizons use the symmetric expectation rho.
//
// Implementation:
//   - Stage 1: reject foreign/invalid State, nonpositive horizon, and allocation overflow.
//   - Stage 2: apply the known-shock canonical variance transition for horizon one.
//   - Stage 3: advance expected variance with Omega+rho*h for remaining horizons.
//
// Behavior highlights:
//   - Only horizon one incorporates the known last residual sign.
//
// Inputs:
//   - state: valid caller-owned posterior state bound to this model.
//   - horizon: number of future observations, at least one.
//
// Returns:
//   - []float64: independent h[T+1]..h[T+H], or nil on error.
//
// Errors:
//   - ErrStateVersion: invalid state or nil model.
//   - ErrDomain: horizon < 1.
//   - ErrOverflow: allocation dimension or variance overflow.
//
// Determinism:
//   - The first known shock and subsequent horizons are processed in index order.
//
// Complexity:
//   - Time O(H), Space O(H), including the detached result allocation.
//
// Notes:
//   - These forecasts exclude parameter uncertainty and innovation tails.
//
// AI-Hints:
//   - Do not generate or assume future residuals for horizons two and later.
func (m *Model) ForecastVariance(state State, horizon int) ([]float64, error) {
	if err := m.stateValid(state); err != nil {
		return nil, err
	}
	if horizon < 1 {
		return nil, ErrDomain
	}
	if horizon > math.MaxInt/8 {
		return nil, ErrOverflow
	}
	return m.forecastKernel(state, horizon)
}

// ForecastNextVariance returns the exact next conditional variance without a slice.
// It is a zero-allocation wrapper over the same known-shock kernel as forecasting.
//
// Implementation:
//   - Stage 1: check model and posterior state identity.
//   - Stage 2: apply the canonical nextVariance transition once.
//
// Behavior highlights:
//   - A point-query wrapper without horizon-slice allocation.
//
// Inputs:
//   - state: valid posterior state containing epsilon[t] and h[t].
//
// Returns:
//   - float64: positive h[t+1], or zero on error.
//
// Errors:
//   - ErrStateVersion, ErrOverflow according to state and variance checks.
//
// Determinism:
//   - Identical first-horizon variance to ForecastVariance(state,1).
//
// Complexity:
//   - Time O(1), Space O(1); no heap allocation.
//
// Notes:
//   - Does not advance State or incorporate an unobserved future return.
//
// AI-Hints:
//   - Prefer this wrapper in hot-path margin-monitoring without needing a forecast slice.
func (m *Model) ForecastNextVariance(state State) (float64, error) {
	if err := m.stateValid(state); err != nil {
		return 0, err
	}
	return nextVariance(m.p, state.LastResidual, state.LastVariance)
}

// ForecastVolatility returns square roots of conditional EXPECTED future variances.
// It provides an intuitive volatility term structure in return units.
//
// Implementation:
//   - Stage 1: delegate validations and forecast mathematics to ForecastVariance.
//   - Stage 2: square-root each element of its private detached slice in place.
//
// Behavior highlights:
//   - Computes sqrt(E[h]), not E[sqrt(h)].
//
// Inputs:
//   - state, horizon: exactly the inputs of ForecastVariance.
//
// Returns:
//   - []float64: variance-implied volatility for each future horizon, or nil on error.
//
// Errors:
//   - All errors of ForecastVariance, without reclassification.
//
// Determinism:
//   - Output order matches ForecastVariance's horizon order exactly.
//
// Complexity:
//   - Time O(H), Space O(H); one forecast allocation.
//
// Notes:
//   - sqrt(E[h]) is not E[sqrt(h)] and must not be interpreted as such.
//
// AI-Hints:
//   - Convert to percentages in the presentation layer; the package keeps input units.
func (m *Model) ForecastVolatility(state State, horizon int) ([]float64, error) {
	variances, err := m.ForecastVariance(state, horizon)
	if err != nil {
		return nil, err
	}
	for i := range variances {
		variances[i] = math.Sqrt(variances[i])
	}
	return variances, nil
}

// Fit estimates feasible GJR-GARCH parameters by deterministic projected Gaussian QMLE.
// It starts exclusively from Options.Initial and never applies a likelihood variance clamp.
//
// Implementation:
//   - Stage 1: validate context, options, and the complete training series in that order.
//   - Stage 2: normalize return units, then run analytic-gradient projected BFGS.
//   - Stage 3: publish the accepted last candidate and detached objective history.
//
// Behavior highlights:
//   - Maximum 20 step halvings; infeasible candidates are rejected, not repaired.
//   - Boundary solutions Alpha=0 and Alpha+Gamma=0 are permitted.
//   - The projected-gradient norm, never objective stagnation alone, proves convergence.
//
// Inputs:
//   - ctx: non-nil training context, polled during optimization and likelihood scans.
//   - returns: finite, nonempty read-only observations in consistent return units.
//   - options: explicit valid Initial, optimizer budget/tolerance, variance floor.
//
// Returns:
//   - *FitResult: complete or last validated candidate, with detached Diagnostics.Objective.
//   - error: nil on convergence, ErrNotConverged on budget/stall, or context error.
//
// Errors:
//   - ErrInvalidOption: absent context or invalid options/initialization.
//   - ErrNilInput/ErrEmptyInput/ErrNonFinite: training data validation failures.
//   - ErrDomain/ErrOverflow: infeasible or non-representable numerical operations.
//   - ErrNotConverged: valid candidate available but convergence unproven.
//   - ctx.Err(): no result before evaluation; optional validated candidate after it.
//
// Determinism:
//   - Ascending observation order, active-set masks, BFGS coordinates and Armijo trials.
//
// Complexity:
//   - Time O(T + I*(21*T + P)), Space O(T+I); P is bounded active-set QR
//     work in five-parameter space and I is the accepted iteration count.
//
// Notes:
//   - The input is not retained; a successful fit does not certify out-of-sample risk.
//
// AI-Hints:
//   - Use errors.Is(err,ErrNotConverged), never string comparisons.
//   - Do not interpret non-nil Model as Diagnostics.Converged=true.
func Fit(ctx context.Context, returns []float64, options Options) (*FitResult, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	resolved, err := prepareOptions(options)
	if err != nil {
		return nil, err
	}
	if err := checkReturns(returns); err != nil {
		return nil, err
	}
	return fitKernel(ctx, returns, resolved)
}

// FitWithInitial fits with explicitly supplied parameters and default QMLE controls.
// It removes options boilerplate without ever selecting an implicit starting model.
//
// Implementation:
//   - Stage 1: build Options with the caller-provided Initial.
//   - Stage 2: delegate all validation and optimization to canonical Fit.
//
// Behavior highlights:
//   - Default iteration/floor/tolerance controls are selected by Fit, not this wrapper.
//
// Inputs:
//   - ctx: non-nil training context; returns: read-only finite training series.
//   - initial: caller-selected stationary and floor-feasible starting model.
//
// Returns:
//   - *FitResult and error with exactly the same partial-result law as Fit.
//
// Errors:
//   - All errors of Fit, including ErrNotConverged and context cancellation.
//
// Determinism:
//   - The same initial point, constraint order, and iteration path as Fit.
//
// Complexity:
//   - Identical to Fit; O(1) additional configuration work.
//
// Notes:
//   - Prefer Fit when an explicit training budget/floor is required.
//
// AI-Hints:
//   - This convenience wrapper still requires an explicit economically defensible Initial.
func FitWithInitial(ctx context.Context, returns []float64, initial Parameters) (*FitResult, error) {
	return Fit(ctx, returns, Options{Initial: initial})
}

// EvaluateGaussianQMLE computes a conditional Gaussian objective and analytic score.
// It evaluates a FIXED parameter candidate; it does not train or project a model.
//
// Implementation:
//   - Stage 1: validate context, variance floor, model coefficients and return series.
//   - Stage 2: initialize h[0]=Omega/(1-rho), then differentiate each causal update.
//   - Stage 3: publish objective, total likelihood and gradient in original units.
//
// Behavior highlights:
//   - A zero varianceFloor chooses the same 1e-12 default as Fit.
//   - Every h[t] must satisfy the floor without post-hoc clamping.
//   - The gradient includes the derivatives of the stationary starting variance.
//
// Inputs:
//   - ctx: non-nil evaluation context.
//   - returns: non-nil, nonempty, finite, caller-owned training return slice.
//   - parameters: valid stationary coefficients to evaluate without modifying them.
//   - varianceFloor: finite, nonnegative lower variance bound in squared-return units.
//
// Returns:
//   - *QMLEEvaluation: detached scalar metrics and derivative, or nil on error.
//
// Errors:
//   - ErrInvalidOption: absent context or negative/nonfinite varianceFloor.
//   - ErrNonFinite/ErrDomain/ErrOverflow: invalid coefficients or evaluation arithmetic.
//   - ErrNilInput/ErrEmptyInput: missing observations.
//   - ctx.Err(): early or mid-evaluation cancellation with nil result.
//
// Determinism:
//   - Analytic derivative coordinates follow Mean,Omega,Alpha,Gamma,Beta order;
//     return rows and recursion stages are evaluated in ascending index order.
//
// Complexity:
//   - Time O(T), Space O(1) in addition to the returned small result.
//
// Notes:
//   - The gradient is in original units, unlike Fit's private normalized workspace.
//   - A finite objective does not imply statistical identification or distribution fit.
//
// AI-Hints:
//   - Central-difference checks must use interior points with unchanged shock signs.
//   - Compare the gradient to the derivative of Objective, not of LogLikelihood.
func EvaluateGaussianQMLE(ctx context.Context, returns []float64, parameters Parameters, varianceFloor float64) (*QMLEEvaluation, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	if !finite(varianceFloor) || varianceFloor < 0 {
		return nil, ErrInvalidOption
	}
	if varianceFloor == 0 {
		varianceFloor = defaultVarianceFloor
	}
	if _, err := stationary(parameters); err != nil {
		return nil, err
	}
	if err := checkReturns(returns); err != nil {
		return nil, err
	}
	value, gradient, err := objective(ctx, returns, vectorFromParameters(parameters), varianceFloor)
	if err != nil {
		return nil, err
	}
	logLikelihood := -float64(len(returns)) * value
	if !finite(logLikelihood) {
		return nil, ErrOverflow
	}
	return &QMLEEvaluation{
		Objective:     value,
		LogLikelihood: logLikelihood,
		Gradient:      parametersFromVector(gradient),
		Observations:  len(returns),
	}, nil
}
