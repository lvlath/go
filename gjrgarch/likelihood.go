// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gjrgarch implements the canonical conditional Gaussian QMLE objective.
// Every parameter derivative propagates through the stationary starting variance.
package gjrgarch

import (
	"context"
	"math"
)

const (
	// coordinateMean is the return-location component of the QMLE vector.
	coordinateMean = iota
	// coordinateOmega is the variance intercept in computational squared-return units.
	coordinateOmega
	// coordinateAlpha is the nonnegative positive-shock coefficient.
	coordinateAlpha
	// coordinateGamma is the signed negative-shock asymmetry increment.
	coordinateGamma
	// coordinateBeta is the nonnegative variance persistence coefficient.
	coordinateBeta
	// coordinateCount fixes the optimizer's dimensionality and Hessian shape.
	coordinateCount
	// contextCheckStride bounds cancellation polling overhead within likelihood scans.
	contextCheckStride = 128
)

// vector is the private, fixed-order five-coordinate optimizer carrier.
// Coordinates follow the same order as public Parameters without a backing slice.
type vector [coordinateCount]float64

// parametersFromVector converts a fixed optimizer vector into an independent value.
// No alias or canonicalization occurs; validation belongs to the caller.
func parametersFromVector(v vector) Parameters {
	return Parameters{
		Mean: v[coordinateMean], Omega: v[coordinateOmega], Alpha: v[coordinateAlpha],
		Gamma: v[coordinateGamma], Beta: v[coordinateBeta],
	}
}

// vectorFromParameters returns a copy of the five public scalar coefficients.
// Parameter order remains stable across objective and KKT implementations.
func vectorFromParameters(p Parameters) vector {
	return vector{p.Mean, p.Omega, p.Alpha, p.Gamma, p.Beta}
}

// dot accumulates a fixed-order Euclidean scalar product of parameter vectors.
// Large coordinate products may overflow, which search-direction validation detects.
// Complexity: O(5) time, O(1) space.
func dot(x, y vector) float64 {
	value := 0.0
	for i := range x {
		value += x[i] * y[i]
	}
	return value
}

// norm calculates a scaled Euclidean norm without prematurely squaring inputs.
// math.Hypot performs exponent-aware scaling in each fixed pair of coordinates.
// Complexity: O(5) time, O(1) space.
func norm(x vector) float64 {
	return math.Hypot(
		math.Hypot(x[coordinateMean], x[coordinateOmega]),
		math.Hypot(math.Hypot(x[coordinateAlpha], x[coordinateGamma]), x[coordinateBeta]),
	)
}

// likelihoodState holds the variance and its five forward sensitivities at t.
// Every transition constructs a new value to prevent using next-step sensitivities
// when calculating the previous observation's score.
type likelihoodState struct {
	variance   float64
	derivative vector
}

// initialLikelihoodState builds parameter-dependent stationary h[0] and dh[0].
// It rejects floor-infeasible intercepts instead of clamping the initial variance.
//
// Implementation:
//   - Stage 1: enforce Omega>=floor in candidate parameter units.
//   - Stage 2: compute h0 from the positive stationarity gap.
//   - Stage 3: initialize all forward derivatives, including Gamma/2.
//
// Complexity: Time O(5), Space O(5), no allocation.
func initialLikelihoodState(p Parameters, rho, floor float64) (likelihoodState, error) {
	if p.Omega < floor {
		return likelihoodState{}, ErrDomain
	}
	gap := 1 - rho
	h := p.Omega / gap
	if !finite(h) || h <= 0 || h < floor {
		return likelihoodState{}, ErrDomain
	}
	var derivative vector
	derivative[coordinateOmega] = 1 / gap
	derivative[coordinateAlpha] = h / gap
	derivative[coordinateGamma] = h / (2 * gap)
	derivative[coordinateBeta] = h / gap
	if !allFinite(derivative) {
		return likelihoodState{}, ErrOverflow
	}
	return likelihoodState{variance: h, derivative: derivative}, nil
}

// addGaussianContribution records one conditional Gaussian loss and analytic score.
// The innovation epsilon[t] is evaluated against the PREVIOUSLY established h[t].
//
// Derivative law:
//   - dLoss/dtheta = 0.5*(1-eps^2/h)/h * dh/dtheta - (eps/h)*dMean/dtheta.
//   - dMean/dtheta is 1 only for the Mean coordinate.
//
// Implementation:
//   - Stage 1: check h[t]>=floor and standardized innovation representability.
//   - Stage 2: add the Gaussian contribution using a stable epsilon/sqrt(h) ratio.
//   - Stage 3: accumulate variance sensitivities and the explicit mean derivative.
//
// Complexity: Time O(5), Space O(1), no allocation.
// AI-Hints: Do not plug epsilon[t] into h[t]; the recurrence is strictly lagged.
func addGaussianContribution(
	state likelihoodState,
	residual, floor float64,
	loss *float64,
	gradient *vector,
) error {
	if !finite(state.variance) || state.variance <= 0 || state.variance < floor {
		return ErrDomain
	}
	z := residual / math.Sqrt(state.variance)
	if !finite(z) {
		return ErrOverflow
	}
	z2 := z * z
	if !finite(z2) {
		return ErrOverflow
	}
	*loss += 0.5 * (math.Log(2*math.Pi) + math.Log(state.variance) + z2)
	if !finite(*loss) {
		return ErrOverflow
	}
	score := 0.5 * (1 - z2) / state.variance
	if !finite(score) {
		return ErrOverflow
	}
	for j := range gradient {
		gradient[j] += score * state.derivative[j]
	}
	gradient[coordinateMean] -= residual / state.variance
	if !allFinite(*gradient) {
		return ErrOverflow
	}
	return nil
}

// advanceLikelihoodState differentiates the same known-residual variance transition.
// It reads ONLY the preceding likelihoodState and observed epsilon[t], then
// records the next state for the following observation's contribution.
//
// Implementation:
//   - Stage 1: choose the asymmetric shock coefficient for the previous residual.
//   - Stage 2: propagate every coordinate's derivative from the old state.
//   - Stage 3: evaluate next variance through the canonical nextVariance kernel.
//
// Complexity: Time O(5), Space O(5), no allocation.
// AI-Hints: Compute all next derivatives before changing h, or Beta derivatives drift.
func advanceLikelihoodState(p Parameters, previous likelihoodState, residual float64) (likelihoodState, error) {
	coefficient := p.Alpha
	negative := residual < 0
	if negative {
		coefficient += p.Gamma
	}
	// Coefficient derivatives must account for epsilon^2 even when the shock
	// coefficient itself is zero, so an unrepresentable square is a score error.
	square := residual * residual
	if !finite(square) {
		return likelihoodState{}, ErrOverflow
	}
	var nextDerivative vector
	nextDerivative[coordinateMean] = -2*coefficient*residual + p.Beta*previous.derivative[coordinateMean]
	nextDerivative[coordinateOmega] = 1 + p.Beta*previous.derivative[coordinateOmega]
	nextDerivative[coordinateAlpha] = square + p.Beta*previous.derivative[coordinateAlpha]
	nextDerivative[coordinateGamma] = p.Beta * previous.derivative[coordinateGamma]
	if negative {
		nextDerivative[coordinateGamma] += square
	}
	nextDerivative[coordinateBeta] = previous.variance + p.Beta*previous.derivative[coordinateBeta]
	if !allFinite(nextDerivative) {
		return likelihoodState{}, ErrOverflow
	}
	next, err := nextVariance(p, residual, previous.variance)
	if err != nil {
		return likelihoodState{}, err
	}
	return likelihoodState{variance: next, derivative: nextDerivative}, nil
}

// objective computes negative average conditional Gaussian log-likelihood and its gradient.
// At every candidate it initializes h[0]=Omega/(1-rho), including dh[0]/dtheta;
// no step may clamp h[t] after evaluation to pass the variance floor.
//
// Implementation:
//   - Stage 1: validate stationary parameters and parameter-dependent initial state.
//   - Stage 2: accumulate each observed Gaussian contribution using existing h[t].
//   - Stage 3: propagate h[t+1] and dh[t+1] only if there is another observation.
//   - Stage 4: divide loss and gradient by sample count and check representability.
//
// Behavior highlights:
//   - Returns are read-only; no caller/optimizer data is retained.
//   - Variance floors restrict the feasible domain rather than changing likelihood.
//
// Errors:
//   - ErrNonFinite/ErrDomain for invalid candidates, ErrOverflow for arithmetic.
//   - ctx.Err() on cancellation; all failures discard the candidate score.
//
// Determinism:
//   - Ascending observation indices; derivatives in fixed five-parameter order.
//
// Complexity:
//   - Time O(T*5), Space O(5), excluding the read-only input series.
//
// AI-Hints:
//   - Do not initialize dh0 to zero; h0 depends on Omega, Alpha, Gamma, and Beta.
//   - Do not move epsilon[t] into h[t] by shifting likelihood indexing.
func objective(ctx context.Context, returns []float64, point vector, floor float64) (float64, vector, error) {
	parameters := parametersFromVector(point)
	rho, err := stationary(parameters)
	if err != nil {
		return 0, vector{}, err
	}
	state, err := initialLikelihoodState(parameters, rho, floor)
	if err != nil {
		return 0, vector{}, err
	}

	loss := 0.0
	var gradient vector
	for i, returnValue := range returns {
		if i%contextCheckStride == 0 {
			if err := ctx.Err(); err != nil {
				return 0, vector{}, err
			}
		}
		residual := returnValue - parameters.Mean
		if !finite(residual) {
			return 0, vector{}, ErrOverflow
		}
		if err := addGaussianContribution(state, residual, floor, &loss, &gradient); err != nil {
			return 0, vector{}, err
		}
		// The final observation contributes to the likelihood but has no
		// h[t+1] term inside the training window.
		if i+1 < len(returns) {
			state, err = advanceLikelihoodState(parameters, state, residual)
			if err != nil {
				return 0, vector{}, err
			}
		}
	}
	inv := 1 / float64(len(returns))
	for i := range gradient {
		gradient[i] *= inv
	}
	loss *= inv
	if !finite(loss) || !allFinite(gradient) {
		return 0, vector{}, ErrOverflow
	}
	return loss, gradient, nil
}

// allFinite checks the five coordinates without allocation or reflection.
// The fixed traversal order is shared by optimizer stage and gradient validation.
func allFinite(v vector) bool {
	for _, value := range v {
		if !finite(value) {
			return false
		}
	}
	return true
}
