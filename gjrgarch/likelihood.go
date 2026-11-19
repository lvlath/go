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
//
// Implementation:
//   - Stage 1: Store exactly five float64 values in declared optimizer order.
//
// Behavior highlights:
//   - No slice backing storage or map ordering can alter derivative indices.
//
// Inputs:
//   - Coordinates are Mean, Omega, Alpha, Gamma, Beta.
//
// Returns:
//   - vector is an independently copied fixed array.
//
// Errors:
//   - None; validation occurs at math/kernel boundaries.
//
// Determinism:
//   - Index order is stable across score, Fit and KKT.
//
// Complexity:
//   - Storage O(5), fixed-size stack-compatible value.
//
// Notes:
//   - Physical and normalized vectors share the layout, not units.
//
// AI-Hints:
//   - Use vectorFromParameters/parametersFromVector to preserve coordinate order.
type vector [coordinateCount]float64

// parametersFromVector converts a fixed optimizer vector into an independent value.
// No alias or canonicalization occurs; validation belongs to the caller.
//
// Implementation:
//   - Stage 1: Map each of five fixed array cells to its named public field.
//
// Behavior highlights:
//   - No model validation or numeric change takes place.
//
// Inputs:
//   - v: fixed-order vector in caller-selected unit system.
//
// Returns:
//   - Parameters: independent named value.
//
// Errors:
//   - None, input validation is caller responsibility.
//
// Determinism:
//   - Mean, Omega, Alpha, Gamma, Beta mapped in that exact order.
//
// Complexity:
//   - Time O(5), Space O(1).
//
// Notes:
//   - No backing slice or pointer retained.
//
// AI-Hints:
//   - Do not swap Gamma and Beta in a gradient vector.
func parametersFromVector(v vector) Parameters {
	return Parameters{
		Mean: v[coordinateMean], Omega: v[coordinateOmega], Alpha: v[coordinateAlpha],
		Gamma: v[coordinateGamma], Beta: v[coordinateBeta],
	}
}

// vectorFromParameters returns a copy of the five public scalar coefficients.
// Parameter order remains stable across objective and KKT implementations.
//
// Implementation:
//   - Stage 1: Read five public scalars in the canonical coordinate order.
//
// Behavior highlights:
//   - Copies parameter bits without normalization or validation.
//
// Inputs:
//   - p: Parameters value in current coordinate system.
//
// Returns:
//   - vector: independent fixed array.
//
// Errors:
//   - None, candidate validation follows at its use site.
//
// Determinism:
//   - Mean,Omega,Alpha,Gamma,Beta.
//
// Complexity:
//   - Time O(5), Space O(1).
//
// Notes:
//   - The caller must decide whether coordinates are physical or normalized.
//
// AI-Hints:
//   - No implicit scaling belongs in this conversion helper.
func vectorFromParameters(p Parameters) vector {
	return vector{p.Mean, p.Omega, p.Alpha, p.Gamma, p.Beta}
}

// dot accumulates a fixed-order Euclidean scalar product of parameter vectors.
// Large coordinate products may overflow, which search-direction validation detects.
//
// Implementation:
//   - Stage 1: Multiply corresponding coordinates and sum in ascending index order.
//
// Behavior highlights:
//   - Returns raw Euclidean dot product, not a normalized score.
//
// Inputs:
//   - x,y: compatible five-parameter coordinates.
//
// Returns:
//   - float64 scalar product.
//
// Errors:
//   - None directly; nonfinite results are checked by callers.
//
// Determinism:
//   - Ascending coordinates preserve a fixed accumulation order.
//
// Complexity:
// O(5) time, O(1) space.
//
// Notes:
//   - Representability is handled at the use site, not by silent saturation.
//
// AI-Hints:
//   - Do not use for a scaled residual norm over arbitrary long observations.
func dot(x, y vector) float64 {
	value := 0.0
	for i := range x {
		value += x[i] * y[i]
	}
	return value
}

// norm calculates a scaled Euclidean norm without prematurely squaring inputs.
// math.Hypot performs exponent-aware scaling in each fixed pair of coordinates.
//
// Implementation:
//   - Stage 1: Accumulate stable pairwise hypotenuse norms of fixed coordinates.
//
// Behavior highlights:
//   - math.Hypot avoids premature square overflow for large finite values.
//
// Inputs:
//   - x: five-dimensional vector, typically normalized optimization coordinates.
//
// Returns:
//   - float64 nonnegative Euclidean norm, possibly Inf for an unrepresentable norm.
//
// Errors:
//   - None directly; caller checks finite-domain preconditions.
//
// Determinism:
//   - Fixed nested Hypot association, no concurrency ordering.
//
// Complexity:
// O(5) time, O(1) space.
//
// Notes:
//   - This is not a statistical variance estimator.
//
// AI-Hints:
//   - Use it for projected-gradient convergence, not likelihood residuals.
func norm(x vector) float64 {
	return math.Hypot(
		math.Hypot(x[coordinateMean], x[coordinateOmega]),
		math.Hypot(math.Hypot(x[coordinateAlpha], x[coordinateGamma]), x[coordinateBeta]),
	)
}

// likelihoodState holds the variance and its five forward sensitivities at t.
// Every transition constructs a new value to prevent using next-step sensitivities
// when calculating the previous observation's score.
//
// Implementation:
//   - Stage 1: Carry h[t] and all five forward derivatives until the next observation.
//
// Behavior highlights:
//   - The state is copied before advancing to prevent future derivative reuse.
//
// Inputs:
//   - variance: conditional h[t]; derivative: dh[t]/dtheta in coordinate order.
//
// Returns:
//   - likelihoodState is a private fixed-size value.
//
// Errors:
//   - Validation occurs at initial and transition kernels.
//
// Determinism:
//   - One state per chronological observation.
//
// Complexity:
//   - Storage O(5), no result arrays.
//
// Notes:
//   - Only an internal QMLE carrier; it is not the public streaming State.
//
// AI-Hints:
//   - Never publish this state as a caller-owned checkpoint.
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
// Behavior highlights:
//   - Initial h0 depends on candidate parameters, not just initial solver settings.
//
// Inputs:
//   - p: fixed candidate; rho: its stationary persistence; floor: variance lower bound.
//
// Returns:
//   - likelihoodState at observation zero, or zero/error.
//
// Errors:
//   - ErrDomain for infeasibility; ErrOverflow for nonfinite derivatives.
//
// Determinism:
//   - Derivative entries fixed in Mean, Omega, Alpha, Gamma, Beta order.
//
// Complexity:
// Time O(5), Space O(5), no allocation.
//
// Notes:
//   - Mean derivative of h0 is zero, whereas Omega/weights contribute.
//
// AI-Hints:
//   - Do not freeze h0 across parameter candidates.
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
// Behavior highlights:
//   - The score uses the already-known h[t] and current return residual, never h[t+1].
//
// Inputs:
//   - state: current h and derivatives; residual: observed epsilon; floor: active constraint; loss/gradient: private accumulators.
//
// Returns:
//   - error: nil after adding finite terms, otherwise error and invalid private scratch.
//
// Errors:
//   - ErrDomain for infeasible h; ErrOverflow for z, score, or accumulation arithmetic.
//
// Determinism:
//   - Gradient accumulation follows fixed coordinate order for each observation.
//
// Complexity:
// Time O(5), Space O(1), no allocation.
//
// Notes:
//   - The caller discards partial scratch on error; nothing is published.
//
// AI-Hints:
// Do not plug epsilon[t] into h[t]; the recurrence is strictly lagged.
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
// Behavior highlights:
//   - Differentiates Mean through epsilon[t], including when shock coefficient is zero.
//
// Inputs:
//   - p: candidate; previous: old h and derivative; residual: known epsilon[t].
//
// Returns:
//   - likelihoodState for the *next* observation or zero/error.
//
// Errors:
//   - ErrOverflow from derivative squares or canonical recurrence.
//
// Determinism:
//   - All derivatives derive from the same unchanged previous state.
//
// Complexity:
// Time O(5), Space O(5), no allocation.
//
// Notes:
//   - Requires a following observation; objective omits this transition after the last scored return.
//
// AI-Hints:
// Compute all next derivatives before changing h, or Beta derivatives drift.
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
// Inputs:
//   - ctx: active context; returns: validated nonempty data; point: five candidate coefficients; floor: hard minimum variance.
//
// Returns:
//   - negative average Gaussian QMLE and five-coordinate analytic gradient or zero/error.
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
// Notes:
//   - No retained observation array; every call evaluates its own h0.
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
//
// Implementation:
//   - Stage 1: Scan all five candidate coordinates for IEEE-754 finiteness.
//
// Behavior highlights:
//   - A single invalid derivative invalidates the entire candidate.
//
// Inputs:
//   - v: fixed-size vector of optimization coordinates or score values.
//
// Returns:
//   - bool: true exactly when every cell is finite.
//
// Errors:
//   - None directly; caller chooses ErrOverflow vs ErrNonFinite classification.
//
// Determinism:
//   - Mean to Beta index order.
//
// Complexity:
//   - Time O(5), Space O(1).
//
// Notes:
//   - Does not establish feasibility of coupled halfspaces.
//
// AI-Hints:
//   - Use stationary/project for domain validation after basic finiteness.
func allFinite(v vector) bool {
	for _, value := range v {
		if !finite(value) {
			return false
		}
	}
	return true
}
