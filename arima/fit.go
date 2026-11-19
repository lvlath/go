// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package arima implements fixed-order conditional Gaussian ARIMA estimation.
// This file prepares the estimation coordinates and delegates optimization to L-BFGS.
package arima

import (
	"context"
	"errors"
	"math"
)

// mean computes the arithmetic sample mean without accumulating an unbounded sum.
// If subtracting opposite-sign finite magnitudes overflows, the increment is
// evaluated as two weighted terms so their finite average remains representable.
// Time O(N), Space O(1); callers must supply nonempty finite observations.
func mean(y []float64) float64 {
	average := 0.0
	for i, value := range y {
		weight := 1 / float64(i+1)
		delta := value - average
		if finite(delta) {
			average += delta * weight
		} else {
			average += value*weight - average*weight
		}
	}
	return average
}

// evaluate maps unrestricted optimizer coordinates to valid physical coefficients.
// It differentiates a profiled Gaussian objective through strict Schur step-up maps,
// preserving the public theta sign instead of reinterpreting MA coefficients.
//
// Implementation:
//   - Stage 1: decode tanh reflection coordinates and reject invalid polynomial domains.
//   - Stage 2: evaluate conditional residuals, profile variance and physical derivatives.
//   - Stage 3: apply AR/MA step-up Jacobians and publish finite coordinate gradients.
//
// Behavior highlights:
//   - The drift derivative is omitted from the optimization vector when c is fixed zero.
//   - The MA chain-rule contribution is negated because theta=-stablePolynomial.
//
// Inputs:
//   - ctx: active context; y: finite ordered differenced samples, read-only.
//   - u: unrestricted coordinates [AR reflections, MA reflections, optional c].
//   - opts: validated order and fixed drift-inclusion policy.
//
// Returns:
//   - objective: negative profiled conditional log likelihood; parameters: valid physical model.
//   - gradient: owned dObjective/du vector, or nil on an error.
//
// Errors:
//   - ErrNonStationary/ErrNonInvertible: rejected Schur step-up candidates.
//   - ErrSingular/ErrOverflow: undefined innovation variance or computed arithmetic.
//   - ctx.Err(): conditional scoring was canceled.
//
// Determinism:
//   - AR then MA Jacobian rows/coordinates are summed in increasing index order.
//
// Complexity:
//   - Time O(N*(p+q)^2+(p+q)^3), Space O(N*(p+q)+(p+q)^2).
//
// Notes:
//   - Every candidate is validated before it is eligible for Armijo acceptance.
//
// AI-Hints:
//   - Do not profile a second variance in the optimizer or clip invalid reflections.
func evaluate(ctx context.Context, y, u []float64, opts Options) (float64, Parameters, []float64, error) {
	parameters, err := decode(u, opts.Order.AR, opts.Order.MA, opts.IncludeDrift)
	if err != nil {
		return 0, Parameters{}, nil, err
	}
	parameters.Order = opts.Order
	parameters.InnovationVariance = 1 // Only a positive placeholder; profile estimates sigma^2.
	if err := checkParameters(parameters); err != nil {
		return 0, Parameters{}, nil, err
	}

	objective, variance, physicalGradient, err := profile(ctx, y, parameters)
	if err != nil {
		return 0, Parameters{}, nil, err
	}
	parameters.InnovationVariance = variance

	arJacobian, ok := reflectionJac(u[:opts.Order.AR])
	if !ok {
		return 0, Parameters{}, nil, ErrNonStationary
	}
	maJacobian, ok := reflectionJac(u[opts.Order.AR : opts.Order.AR+opts.Order.MA])
	if !ok {
		return 0, Parameters{}, nil, ErrNonInvertible
	}

	gradient := make([]float64, len(u))
	for k := 0; k < opts.Order.AR; k++ {
		for i := 0; i < opts.Order.AR; i++ {
			gradient[k] += physicalGradient[i] * arJacobian[i][k]
		}
	}
	for k := 0; k < opts.Order.MA; k++ {
		for j := 0; j < opts.Order.MA; j++ {
			// theta=-a: the chain rule changes the MA Jacobian sign.
			gradient[opts.Order.AR+k] -= physicalGradient[opts.Order.AR+j] * maJacobian[j][k]
		}
	}
	if opts.IncludeDrift {
		gradient[len(u)-1] = physicalGradient[len(physicalGradient)-1]
	}
	for _, derivative := range gradient {
		if !finite(derivative) {
			return 0, Parameters{}, nil, ErrOverflow
		}
	}
	return objective, parameters, gradient, nil
}

// fitKernel initializes a deterministic conditional model and runs bounded L-BFGS.
// It separates statistical initialization from direction/search mechanics and
// publishes only accepted candidates, never rejected line-search trials.
//
// Implementation:
//   - Stage 1: initialize zero reflections and, if enabled, mean differenced intercept.
//   - Stage 2: score the initial model, then repeatedly search and accept bounded steps.
//   - Stage 3: freeze the latest accepted model on convergence or nonconvergence.
//
// Behavior highlights:
//   - The p=q=0 analytic path does not invoke L-BFGS and uses zero accepted steps.
//   - A failed Armijo search retains the last accepted model with Converged=false.
//
// Inputs:
//   - ctx: active cancellation context; y: differenced training history, read-only.
//   - opts: validated order, drift policy, iteration budget and tolerance.
//
// Returns:
//   - *FitResult: owned accepted objective history and immutable model.
//   - error: nil if converged; ErrNotConverged with last accepted model if budget fails.
//
// Errors:
//   - ErrSingular/ErrOverflow: undefined initial profile or unrepresentable candidate.
//   - ErrNonStationary/ErrNonInvertible: invalid Schur candidate initialization.
//   - ctx.Err(): cancellation, always with nil result.
//
// Determinism:
//   - Accepted iteration number is exact; no random starts or random line steps.
//
// Complexity:
//   - Time O((I+1)*N*(p+q+1)^2), Space O(N*(p+q+1)+I).
//
// Notes:
//   - I counts accepted steps; each step may contain up to twenty candidate scores.
//
// AI-Hints:
//   - Do not publish a rejected trial or declare plateau convergence without a small gradient.
func fitKernel(ctx context.Context, y []float64, opts Options) (*FitResult, error) {
	dimension := opts.Order.AR + opts.Order.MA
	if opts.IncludeDrift {
		dimension++
	}
	coordinates := make([]float64, dimension) // Zero reflection coefficients give AR=MA=0.
	if opts.IncludeDrift {
		coordinates[dimension-1] = mean(y) // c is the mean only at this zero-AR initialization.
		if !finite(coordinates[dimension-1]) {
			return nil, ErrOverflow
		}
	}

	objective, parameters, gradient, err := evaluate(ctx, y, coordinates, opts)
	if err != nil {
		return nil, err
	}
	state := lbfgsState{
		coordinates: coordinates,
		gradient:    gradient,
		parameters:  parameters,
		objective:   objective,
		objectives:  []float64{objective},
		corrections: make([]lbfgsCorrection, 0, lbfgsHistorySize),
	}
	if dimension == 0 || state.converged(opts.Tolerance) {
		return state.publish(len(y), opts.Order, true)
	}

	for state.iterations < opts.MaxIterations {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		trial, searchErr := state.lineSearch(ctx, y, opts)
		if searchErr != nil {
			if !errors.Is(searchErr, ErrNotConverged) {
				return nil, searchErr
			}
			result, publishErr := state.publish(len(y), opts.Order, false)
			if publishErr != nil {
				return nil, publishErr
			}
			return result, ErrNotConverged
		}

		state.accept(trial)
		if state.converged(opts.Tolerance) {
			return state.publish(len(y), opts.Order, true)
		}
	}

	result, err := state.publish(len(y), opts.Order, false)
	if err != nil {
		return nil, err
	}
	return result, ErrNotConverged
}

// dot sums matched vector products in a fixed increasing-index order.
// Time O(k), Space O(1); the caller checks finite outputs before using them.
func dot(a, b []float64) float64 {
	sum := 0.0
	for i, value := range a {
		sum += value * b[i]
	}
	return sum
}

// norm computes Euclidean length without directly squaring large gradient entries.
// Time O(k), Space O(1); a nonfinite entry is never silently discarded.
func norm(g []float64) float64 {
	sum := 0.0
	for _, value := range g {
		sum = math.Hypot(sum, value)
	}
	return sum
}
