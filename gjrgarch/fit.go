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
type fitWorkspace struct {
	returns  []float64 // Owned training series normalized into computational units.
	scale    float64   // Positive return-unit scale, fixed for the entire fit.
	logScale float64   // log(scale), the Gaussian likelihood Jacobian correction.
	floor    float64   // Variance floor expressed in squared normalized-return units.
}

// optimizer stores exactly the last accepted feasible BFGS candidate.
// No previous candidate, RNG state, mutable global policy, or caller array is retained.
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
// Errors:
//   - ErrOverflow if the scaling transformation cannot be represented in float64.
//
// Complexity:
//   - Time O(T), Space O(T) with one private normalized return slice.
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
// Complexity: O(T) time and O(1) additional optimizer metadata.
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

// (o *optimizer).publish materializes the last accepted original-unit QMLE model.
// Diagnostics are copied; neither user input nor the private objective history leaks.
//
// Implementation:
//   - Stage 1: unscale the candidate in a range-preserving order.
//   - Stage 2: revalidate the model and the total likelihood in physical units.
//   - Stage 3: publish a detached history with a truthful convergence flag.
//
// Complexity: O(iterations) time and storage for the diagnostic history copy.
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

// (o *optimizer).finish preserves a validated candidate only for documented failures.
// ErrNotConverged and late context cancellation accompany the last valid model;
// unrelated arithmetic/matrix failures remain atomic nil failures.
func (o *optimizer) finish(reason error) (*FitResult, error) {
	result, err := o.publish(false)
	if err != nil {
		return nil, err
	}
	return result, reason
}

// (o *optimizer).searchDirection projects an inverse-BFGS gradient step.
// If the inverse Hessian fails to produce descent, it restarts at identity
// and retries a projected steepest-descent direction before declaring stagnation.
//
// Complexity: O(5^2) fixed linear algebra plus at most two active-set projections.
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
func subtractVector(a, b vector) vector {
	var difference vector
	for i := range difference {
		difference[i] = a[i] - b[i]
	}
	return difference
}

// (o *optimizer).armijo evaluates at most 21 feasible line-search fractions.
// It never changes a candidate's computed h[t] to satisfy the variance floor.
//
// Implementation:
//   - Stage 1: begin at the full projected descent direction.
//   - Stage 2: evaluate 1, 1/2, ..., 1/2^20 in that fixed order.
//   - Stage 3: return the first sufficient-decrease candidate, or a stalled status.
//
// Errors:
//   - ctx.Err() cancels immediately without modifying optimizer state.
//   - Infeasible/overflowing candidate evaluations are rejected trials, not success.
//
// Complexity: At most 21 O(T) likelihood-and-gradient evaluations, O(1) workspace.
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

// (o *optimizer).accept commits a validated candidate and updates BFGS curvature.
// The history grows only after acceptance; rejected line-search trials leave no trace.
//
// Complexity: O(5^2) for the inverse update; O(1) amortized history append.
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

// (o *optimizer).run iterates the projected-gradient stopping and BFGS transitions.
// It does not infer convergence from a small objective decrease or step size.
//
// Implementation:
//   - Stage 1: check cancellation and projected stationarity at each accepted point.
//   - Stage 2: find a descent direction and take a bounded Armijo step.
//   - Stage 3: after the budget, recheck stationarity before publishing a result.
//
// Complexity: O(I*T*21+I*K) time and O(I) history, K bounded 3x3 KKT work.
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

// (o *optimizer).handleSolverError preserves the declared partial-result protocol.
// Cancellation and nonconvergence retain the last valid model; all other
// numerical/dependency failures return nil without a misleading partial artifact.
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
// Complexity: O(T+I*21*T) time, O(T+I) space and bounded KKT matrices.
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
