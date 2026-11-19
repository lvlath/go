// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gjrgarch implements constrained Gaussian QMLE with deterministic
// return scaling, projected BFGS, and a bounded Armijo line search.
package gjrgarch

import (
	"context"
	"errors"
	"math"
)

// fitWorkspace contains normalized caller-independent observations and policy.
// All four fields are computed deterministically before the first QMLE evaluation.
//
// Implementation:
//   - Stage 1: Prepare scaling once before optimization.
//   - Stage 2: Store private normalized returns and floor.
//
// Behavior highlights:
//   - One fixed scale is shared by every line-search candidate.
//
// Inputs:
//   - returns: owned normalized observation slice; scale: positive return unit; logScale: log(scale); floor: normalized squared-unit bound.
//
// Returns:
//   - fitWorkspace is privately owned by the optimizer, never exposed to callers.
//
// Errors:
//   - Preparation failures are handled before workspace publication.
//
// Determinism:
//   - Training observations preserve original chronological order.
//
// Complexity:
//   - Storage O(T), not O(I*T).
//
// Notes:
//   - No pointer to caller return data survives workspace creation.
//
// AI-Hints:
//   - Changing the scaling point per iteration would redefine the objective and break BFGS curvature.
type fitWorkspace struct {
	returns  []float64 // Owned training series normalized into computational units.
	scale    float64   // Positive return-unit scale, fixed for the entire fit.
	logScale float64   // log(scale), the Gaussian likelihood Jacobian correction.
	floor    float64   // Variance floor expressed in squared normalized-return units.
}

// optimizer stores exactly the last accepted feasible BFGS candidate.
// No previous candidate, RNG state, mutable global policy, or caller array is retained.
//
// Implementation:
//   - Stage 1: Initialize from explicit candidate.
//   - Stage 2: Retain only accepted point/gradient/Hessian/history.
//
// Behavior highlights:
//   - Rejected line-search trials do not replace the stored candidate.
//
// Inputs:
//   - ctx: cooperative cancellation; point: normalized parameter vector; history: original-unit accepted NLL values.
//
// Returns:
//   - run eventually publishes a detached FitResult or classified failure.
//
// Errors:
//   - Partial publication follows finish/handleSolverError, never an implicit success.
//
// Determinism:
//   - Fixed coordinate and Armijo trial order, independent of RNG.
//
// Complexity:
//   - State O(I) for history plus O(T) referenced normalized data.
//
// Notes:
//   - optimizer is local to one Fit and not shared among goroutines.
//
// AI-Hints:
//   - Do not let a failed trial mutate point or accepted history.
type optimizer struct {
	ctx        context.Context
	data       fitWorkspace
	point      vector
	gradient   vector
	objective  float64
	inverseH   [coordinateCount][coordinateCount]float64
	history    []float64
	iterations int
}

// scaleParameters expresses return-dependent parameters in normalized units.
// Mean scales by s and Omega by s^2; dimensionless weights are untouched.
//
// Implementation:
//   - Stage 1: Divide Mean by scale and Omega by scale twice.
//   - Stage 2: Leave dimensionless weights unchanged.
//
// Behavior highlights:
//   - Scaling is a change of coordinates for optimization, not a new GJR model.
//
// Inputs:
//   - p: physical-unit Parameters; scale: positive fixed return scale.
//
// Returns:
//   - Parameters: computational-unit value, without modifying the input.
//
// Errors:
//   - None here; the caller validates scale and checks the result for finiteness.
//
// Determinism:
//   - Field order is fixed and weights are copied verbatim.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - A scale near underflow may fail later representability gates.
//
// AI-Hints:
//   - Always convert the variance intercept with square of return scale, not linear scale.
func scaleParameters(p Parameters, scale float64) Parameters {
	return Parameters{
		Mean:  p.Mean / scale,
		Omega: p.Omega / scale / scale,
		Alpha: p.Alpha,
		Gamma: p.Gamma,
		Beta:  p.Beta,
	}
}

// unscaleParameters maps a candidate into original units without premature s*s.
// A scaled product is used for Omega to avoid intermediate overflow when s*s
// is unrepresentable but the actual product Omega*s^2 is representable.
//
// Implementation:
//   - Stage 1: Restore Omega with a range-preserving weighted square.
//   - Stage 2: Restore Mean with one multiply.
//   - Stage 3: Validate outputs.
//
// Behavior highlights:
//   - Avoids premature overflow in a separately formed scale*scale.
//
// Inputs:
//   - p: normalized parameter candidate; scale: positive return-unit scale.
//
// Returns:
//   - Parameters: physical-unit candidate or zero value on error.
//
// Errors:
//   - ErrOverflow for final unrepresentable mean/intercept or nonpositive intercept.
//
// Determinism:
//   - Five fields are mapped in fixed declared order.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Restoration is required before model ID, final likelihood and public diagnostics.
//
// AI-Hints:
//   - Do not re-evaluate a different scaled objective when publishing the final candidate.
func unscaleParameters(p Parameters, scale float64) (Parameters, error) {
	omega, err := weightedResidualSquare(p.Omega, scale)
	if err != nil {
		return Parameters{}, err
	}
	p.Mean *= scale
	p.Omega = omega
	if !finite(p.Mean) || !finite(p.Omega) || p.Omega <= 0 {
		return Parameters{}, ErrOverflow
	}
	return p, nil
}

// prepareFitWorkspace chooses one deterministic coordinate scale for all candidates.
// The algorithm never recalibrates the scale based on accepted BFGS parameters.
//
// Implementation:
//   - Stage 1: find the largest magnitude of observed returns and initial parameters.
//   - Stage 2: ensure the variance intercept and floor have a nonzero scale anchor.
//   - Stage 3: create a private normalized data slice and normalized floor.
//
// Behavior highlights:
//   - The entire optimization shares one private fixed-coordinate return series.
//
// Inputs:
//   - returns: read-only training samples; options: validated Initial and variance floor.
//
// Returns:
//   - fitWorkspace: owner of normalized slice and fixed scale/Jacobian metadata.
//
// Errors:
//   - ErrOverflow if the scaling transformation cannot be represented in float64.
//
// Determinism:
//   - One ascending scan for magnitude, then one ascending copy.
//
// Complexity:
//   - Time O(T), Space O(T) with one private normalized return slice.
//
// Notes:
//   - No stochastic initialization or candidate-dependent normalization occurs.
//
// AI-Hints:
//   - Fit must not use a candidate-dependent scale inside the line-search loop.
func prepareFitWorkspace(returns []float64, options Options) (fitWorkspace, error) {
	scale := math.Max(math.Sqrt(options.Initial.Omega), math.Abs(options.Initial.Mean))
	scale = math.Max(scale, math.Sqrt(options.VarianceFloor))
	for _, observed := range returns {
		scale = math.Max(scale, math.Abs(observed))
	}
	if !finite(scale) || scale <= 0 {
		return fitWorkspace{}, ErrOverflow
	}

	// Normalize once so all accepted candidates use the same objective coordinates.
	normalized := make([]float64, len(returns))
	for i, observed := range returns {
		normalized[i] = observed / scale
	}
	floor := options.VarianceFloor / scale / scale
	if !finite(floor) || floor <= 0 {
		return fitWorkspace{}, ErrOverflow
	}
	return fitWorkspace{returns: normalized, scale: scale, logScale: math.Log(scale), floor: floor}, nil
}

// newOptimizer evaluates only the caller's explicitly requested initial model.
// It does not publish a result until objective and analytic gradient both succeed.
//
// Implementation:
//   - Stage 1: Convert explicit Initial to normalized point.
//   - Stage 2: Evaluate loss and gradient.
//   - Stage 3: Set identity inverse Hessian.
//
// Behavior highlights:
//   - No optimizer result exists before the initial objective has passed all numeric gates.
//
// Inputs:
//   - ctx: nonnil active; data: owning normalized workspace; initial: validated physical Parameters.
//
// Returns:
//   - *optimizer: first accepted candidate, with initial score in history.
//
// Errors:
//   - ErrOverflow for invalid conversion; objective/context errors forwarded atomically.
//
// Determinism:
//   - No random seed, unordered search or fit-derived initial guess.
//
// Complexity:
// O(T) time and O(1) additional optimizer metadata.
//
// Notes:
//   - State and objective history are private to this Fit.
//
// AI-Hints:
//   - A zero initial vector must never be treated as a request to initialize from data.
func newOptimizer(ctx context.Context, data fitWorkspace, initial Parameters) (*optimizer, error) {
	point := vectorFromParameters(scaleParameters(initial, data.scale))
	if !allFinite(point) || point[1] < data.floor {
		return nil, ErrOverflow
	}
	value, gradient, err := objective(ctx, data.returns, point, data.floor)
	if err != nil {
		return nil, err
	}
	return &optimizer{
		ctx: ctx, data: data, point: point, gradient: gradient,
		objective: value, inverseH: identityHessian(),
		history: []float64{value + data.logScale},
	}, nil
}

// publishpublish materializes the last accepted original-unit QMLE model.
// Diagnostics are copied; neither user input nor the private objective history leaks.
//
// Implementation:
//   - Stage 1: unscale the candidate in a range-preserving order.
//   - Stage 2: revalidate the model and the total likelihood in physical units.
//   - Stage 3: publish a detached history with a truthful convergence flag.
//
// Behavior highlights:
//   - A candidate is marked converged only when the caller passed the projected test.
//
// Inputs:
//   - converged: exact outcome of the projected-gradient check.
//
// Returns:
//   - *FitResult: validated immutable Model and detached acceptance history.
//
// Errors:
//   - ErrOverflow on unscale/likelihood publication; New errors retain their root.
//
// Determinism:
//   - Final scalars are tied to the last accepted point and fixed initial sample count.
//
// Complexity:
// O(iterations) time and storage for the diagnostic history copy.
//
// Notes:
//   - Publishing does not re-optimize or silently mark a partial result successful.
//
// AI-Hints:
//   - Do not print a norm or likelihood from a rejected Armijo trial.
func (o *optimizer) publish(converged bool) (*FitResult, error) {
	params, err := unscaleParameters(parametersFromVector(o.point), o.data.scale)
	if err != nil {
		return nil, err
	}
	model, err := New(params)
	if err != nil {
		return nil, err
	}
	average := o.objective + o.data.logScale
	logLikelihood := -float64(len(o.data.returns)) * average
	if !finite(average) || !finite(logLikelihood) {
		return nil, ErrOverflow
	}
	return &FitResult{
		Model: model,
		Diagnostics: Diagnostics{
			Iterations: o.iterations,
			Converged:  converged,
			Objective:  append([]float64(nil), o.history...),
		},
		LogLikelihood: logLikelihood,
	}, nil
}

// finishfinish preserves a validated candidate only for documented failures.
// ErrNotConverged and late context cancellation accompany the last valid model;
// unrelated arithmetic/matrix failures remain atomic nil failures.
//
// Implementation:
//   - Stage 1: Publish the last accepted feasible candidate as nonconverged.
//   - Stage 2: Return the supplied reason.
//
// Behavior highlights:
//   - Only called for explicit nonconvergence or eligible late cancellation.
//
// Inputs:
//   - reason: ErrNotConverged or a context cancellation reason.
//
// Returns:
//   - Last validated *FitResult plus nonnil reason, unless publication itself fails.
//
// Errors:
//   - ErrNotConverged/context cause preserved; publication errors abort atomically.
//
// Determinism:
//   - History and iteration count remain as of the last accepted candidate.
//
// Complexity:
//   - Time/Space O(I) for detached history.
//
// Notes:
//   - Valid partial candidates are not automatically deployable.
//
// AI-Hints:
//   - Always retain the error alongside the candidate in audit logs.
func (o *optimizer) finish(reason error) (*FitResult, error) {
	result, err := o.publish(false)
	if err != nil {
		return nil, err
	}
	return result, reason
}

// searchDirectionsearchDirection projects an inverse-BFGS gradient step.
// If the inverse Hessian fails to produce descent, it restarts at identity
// and retries a projected steepest-descent direction before declaring stagnation.
//
// Implementation:
//   - Stage 1: Build inverse-BFGS proposed point.
//   - Stage 2: Project.
//   - Stage 3: Check descent.
//   - Stage 4: Retry projected steepest descent after identity reset.
//
// Behavior highlights:
//   - Restart changes search metric only, not objective or accepted candidate.
//
// Inputs:
//   - o: current optimizer with point, analytic gradient and inverse Hessian.
//
// Returns:
//   - Direction and finite negative slope or classified failure.
//
// Errors:
//   - ErrNotConverged if neither projected direction descends; KKT errors forwarded.
//
// Determinism:
//   - Fixed 5x5 loops, then at most two projections.
//
// Complexity:
// O(5^2) fixed linear algebra plus at most two active-set projections.
//
// Notes:
//   - Projection is of parameters, not computed conditional variances.
//
// AI-Hints:
//   - An unconstrained -H*g is not automatically a feasible descent step.
func (o *optimizer) searchDirection() (vector, float64, error) {
	// First try the existing quasi-Newton search geometry.
	trial := o.point
	for i := range trial {
		for j := range trial {
			trial[i] -= o.inverseH[i][j] * o.gradient[j]
		}
	}
	projected, err := project(o.ctx, trial, o.data.floor)
	if err != nil {
		return vector{}, 0, err
	}
	direction := subtractVector(projected, o.point)
	slope := dot(o.gradient, direction)
	if finite(slope) && slope < 0 {
		return direction, slope, nil
	}

	// Curvature may become unsuitable near a binding inequality face.
	// Restart only the search metric: objective and last accepted point stay fixed.
	o.inverseH = identityHessian()
	for i := range trial {
		trial[i] = o.point[i] - o.gradient[i]
	}
	projected, err = project(o.ctx, trial, o.data.floor)
	if err != nil {
		return vector{}, 0, err
	}
	direction = subtractVector(projected, o.point)
	slope = dot(o.gradient, direction)
	if !finite(slope) || slope >= 0 {
		return vector{}, 0, ErrNotConverged
	}
	return direction, slope, nil
}

// subtractVector forms a-b in the fixed five-parameter coordinate order.
// It is shared by line search and secant update to keep arithmetic ordering stable.
//
// Implementation:
//   - Stage 1: Subtract five fixed-order coordinates element by element.
//
// Behavior highlights:
//   - Keeps secant and line-search direction arithmetic in the same coordinate basis.
//
// Inputs:
//   - a,b: normalized five-coordinate values.
//
// Returns:
//   - vector: a-b independent fixed array.
//
// Errors:
//   - None; arithmetic overflow is checked by downstream kernels.
//
// Determinism:
//   - Increasing coordinate order.
//
// Complexity:
//   - Time O(5), Space O(5) fixed stack value.
//
// Notes:
//   - No allocations or aliasing; callers validate finite results where needed.
//
// AI-Hints:
//   - Do not mix normalized and physical parameter vectors.
func subtractVector(a, b vector) vector {
	var difference vector
	for i := range difference {
		difference[i] = a[i] - b[i]
	}
	return difference
}

// armijoarmijo evaluates at most 21 feasible line-search fractions.
// It never changes a candidate's computed h[t] to satisfy the variance floor.
//
// Implementation:
//   - Stage 1: begin at the full projected descent direction.
//   - Stage 2: evaluate 1, 1/2, ..., 1/2^20 in that fixed order.
//   - Stage 3: return the first sufficient-decrease candidate, or a stalled status.
//
// Behavior highlights:
//   - Rejects failed candidate evaluations without touching the last accepted optimizer state.
//
// Inputs:
//   - direction: projected descent displacement; slope: negative directional gradient product.
//
// Returns:
//   - Candidate/gradient/loss, accepted bool, or false with optional cancellation error.
//
// Errors:
//   - ctx.Err() cancels immediately without modifying optimizer state.
//   - Infeasible/overflowing candidate evaluations are rejected trials, not success.
//
// Determinism:
//   - Trials ordered 1,1/2,...,1/2^20; no stochastic restarts.
//
// Complexity:
// At most 21 O(T) likelihood-and-gradient evaluations, O(1) workspace.
//
// Notes:
//   - The floor applies as candidate feasibility, never loss clamping.
//
// AI-Hints:
//   - When all trials fail, report not converged; do not silently accept a tiny loss change.
func (o *optimizer) armijo(direction vector, slope float64) (vector, vector, float64, bool, error) {
	step := 1.0
	for attempt := 0; attempt <= maxLineSearchHalvings; attempt++ {
		if err := o.ctx.Err(); err != nil {
			return vector{}, vector{}, 0, false, err
		}
		var candidate vector
		for i := range candidate {
			candidate[i] = o.point[i] + step*direction[i]
		}
		value, gradient, err := objective(o.ctx, o.data.returns, candidate, o.data.floor)
		if err == nil && value <= o.objective+armijoCoefficient*step*slope {
			return candidate, gradient, value, true, nil
		}
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return vector{}, vector{}, 0, false, err
		}
		step *= 0.5
	}
	return vector{}, vector{}, 0, false, nil
}

// acceptaccept commits a validated candidate and updates BFGS curvature.
// The history grows only after acceptance; rejected line-search trials leave no trace.
//
// Implementation:
//   - Stage 1: Compute secant movement and gradient difference.
//   - Stage 2: Update inverse Hessian.
//   - Stage 3: Commit point/gradient/loss.
//   - Stage 4: Append history.
//
// Behavior highlights:
//   - Only a successful Armijo trial is accepted; all rejected trials are invisible.
//
// Inputs:
//   - next,nextGradient,nextObjective: validated new candidate tuple.
//
// Returns:
//   - No direct result; o is updated with one more accepted iterate.
//
// Errors:
//   - None; the caller already validated all candidate outputs.
//
// Determinism:
//   - Commit and history append occur in a fixed order.
//
// Complexity:
// O(5^2) for the inverse update; O(1) amortized history append.
//
// Notes:
//   - A curvature failure restarts H but retains the accepted model.
//
// AI-Hints:
//   - Never update history before the Armijo sufficient-decrease gate.
func (o *optimizer) accept(next, nextGradient vector, nextObjective float64) {
	movement := subtractVector(next, o.point)
	gradientChange := subtractVector(nextGradient, o.gradient)
	bfgsInverseUpdate(&o.inverseH, movement, gradientChange)
	o.point = next
	o.gradient = nextGradient
	o.objective = nextObjective
	o.iterations++
	o.history = append(o.history, nextObjective+o.data.logScale)
}

// runrun iterates the projected-gradient stopping and BFGS transitions.
// It does not infer convergence from a small objective decrease or step size.
//
// Implementation:
//   - Stage 1: check cancellation and projected stationarity at each accepted point.
//   - Stage 2: find a descent direction and take a bounded Armijo step.
//   - Stage 3: after the budget, recheck stationarity before publishing a result.
//
// Behavior highlights:
//   - Only projected-gradient stationarity, not objective stagnation, marks Converged.
//
// Inputs:
//   - maxIterations: accepted-move limit; tolerance: relative projected-gradient threshold.
//
// Returns:
//   - FitResult and nil error on convergence; optional candidate/error for unfinished budget.
//
// Errors:
//   - ErrNotConverged, context cancellation or fatal KKT/numeric error as documented.
//
// Determinism:
//   - Fixed observation, active-face and line-search traversal order.
//
// Complexity:
// O(I*T*21+I*K) time and O(I) history, K bounded 3x3 KKT work.
//
// Notes:
//   - Iterations counts accepted candidate moves, not evaluations.
//
// AI-Hints:
//   - A budget exhausted exactly at convergence still gets a final projected-gradient check.
func (o *optimizer) run(maxIterations int, tolerance float64) (*FitResult, error) {
	for o.iterations < maxIterations {
		if err := o.ctx.Err(); err != nil {
			return o.finish(err)
		}
		converged, err := projectedStationary(o.ctx, o.point, o.gradient, o.data.floor, tolerance)
		if err != nil {
			return o.handleSolverError(err)
		}
		if converged {
			return o.publish(true)
		}

		direction, slope, err := o.searchDirection()
		if err != nil {
			return o.handleSolverError(err)
		}
		next, nextGradient, nextObjective, accepted, err := o.armijo(direction, slope)
		if err != nil {
			return o.handleSolverError(err)
		}
		if !accepted {
			return o.finish(ErrNotConverged)
		}
		o.accept(next, nextGradient, nextObjective)
	}

	// A budget boundary can coincide exactly with convergence: check once more.
	converged, err := projectedStationary(o.ctx, o.point, o.gradient, o.data.floor, tolerance)
	if err != nil {
		return o.handleSolverError(err)
	}
	if converged {
		return o.publish(true)
	}
	return o.finish(ErrNotConverged)
}

// handleSolverErrorhandleSolverError preserves the declared partial-result protocol.
// Cancellation and nonconvergence retain the last valid model; all other
// numerical/dependency failures return nil without a misleading partial artifact.
//
// Implementation:
//   - Stage 1: Identify cancellation/nonconvergence through errors.Is.
//   - Stage 2: Retain candidate only for those classes.
//
// Behavior highlights:
//   - Other matrix/arithmetic failures remain atomic and return nil result.
//
// Inputs:
//   - err: classified solver or context cause.
//
// Returns:
//   - Optional last validated FitResult with same err, or nil on fatal failure.
//
// Errors:
//   - Preserves context and ErrNotConverged roots; other errors passed through.
//
// Determinism:
//   - Fixed errors.Is precedence, no string inspection.
//
// Complexity:
//   - Time O(I) on partial publication, O(1) for fatal fast-return.
//
// Notes:
//   - This boundary is the single source for optimizer partial-result behavior.
//
// AI-Hints:
//   - Do not turn a rank/overflow failure into successful completion.
func (o *optimizer) handleSolverError(err error) (*FitResult, error) {
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) ||
		errors.Is(err, ErrNotConverged) {
		return o.finish(err)
	}
	return nil, err
}

// fitKernel performs the validated training pipeline without redefining the API.
// It first owns a normalized data snapshot, then owns all optimizer state until
// the final FitResult is published or a classified error aborts execution.
//
// Implementation:
//   - Stage 1: Prepare fixed normalized workspace.
//   - Stage 2: Evaluate Initial.
//   - Stage 3: Run deterministic projected BFGS.
//
// Behavior highlights:
//   - Owns one training slice snapshot and delegates kernel mathematics to objective/project.
//
// Inputs:
//   - ctx: active context; returns: prevalidated caller training data; options: resolved policy.
//
// Returns:
//   - FitResult and error according to the partial-result contract.
//
// Errors:
//   - Propagates preparation, initial evaluation and solver errors.
//
// Determinism:
//   - The requested Initial defines all subsequent deterministic work.
//
// Complexity:
// O(T+I*21*T) time, O(T+I) space and bounded KKT matrices.
//
// Notes:
//   - No nested go.mod, RNG, wall-clock state or observation-retaining model.
//
// AI-Hints:
//   - Never insert a heuristic initial-parameter fit ahead of newOptimizer.
func fitKernel(ctx context.Context, returns []float64, options Options) (*FitResult, error) {
	workspace, err := prepareFitWorkspace(returns, options)
	if err != nil {
		return nil, err
	}
	opt, err := newOptimizer(ctx, workspace, options.Initial)
	if err != nil {
		return nil, err
	}
	return opt.run(options.MaxIterations, options.Tolerance)
}
