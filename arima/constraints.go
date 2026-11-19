// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package arima implements strict Schur coefficient-domain restrictions.
// This file maps unrestricted optimizer coordinates to stationary and invertible polynomials.
package arima

import "math"

// schurStable verifies strict outer-unit-circle roots in AR polynomial coordinates.
// A coefficient vector a represents 1-sum(a_i*z^i). The Schur step-down
// recurrence avoids constructing complex roots or selecting a numerical root tolerance.
//
// Implementation:
//   - Stage 1: detach a small coefficient vector so caller data are untouched.
//   - Stage 2: for descending degree m, take k=a[m], require abs(k)<1,
//     and reconstruct degree-(m-1) coefficients by the mirrored recursion.
//   - Stage 3: reject nonfinite coefficients or return true at degree zero.
//
// Behavior highlights:
//   - Equality abs(k)==1 is rejected, not clipped to an interior value.
//   - A zero-length polynomial is stable by definition.
//
// Inputs:
//   - coefficients: finite physical stable-polynomial coordinates a[1..m].
//
// Returns:
//   - bool: whether every Schur step satisfies the strict stability domain.
//
// Errors:
//   - None directly; checkParameters maps false to ErrNonStationary or
//     ErrNonInvertible according to the AR/MA polynomial being checked.
//
// Determinism:
//   - Degree decreases by one; mirrored indices traverse increasing j.
//
// Complexity:
//   - Time O(m^2), Space O(m) retained plus short-lived degree workspaces.
//
// Notes:
//   - The MA validator supplies negative theta because its polynomial has +theta.
//
// AI-Hints:
//   - Do not weaken >=1 to >1 or introduce a tolerance that accepts unit roots.
//   - This is a binary64 coefficient-domain test, not an arbitrary-precision root proof.
func schurStable(coefficients []float64) bool {
	current := append([]float64(nil), coefficients...)
	for degree := len(current); degree > 0; degree-- {
		reflection := current[degree-1]
		if !finite(reflection) || math.Abs(reflection) >= 1 {
			return false // Exactly on the unit circle is nonstationary/noninvertible.
		}
		denominator := 1 - reflection*reflection
		previous := make([]float64, degree-1)
		for j := range previous {
			mirror := current[degree-2-j]
			previous[j] = (current[j] + reflection*mirror) / denominator
			if !finite(previous[j]) {
				return false
			}
		}
		current = previous
	}
	return true
}

// fromReflection builds a strictly stable polynomial from unrestricted coordinates.
// Each tanh-transformed reflection coefficient is inserted through a Levinson
// step-up, so all valid intermediate polynomials satisfy the Schur domain.
//
// Implementation:
//   - Stage 1: initialize the zero-degree polynomial with no coefficients.
//   - Stage 2: for each u_m, compute k_m=tanh(u_m), reject rounded abs(k_m)>=1,
//     and apply the mirrored coefficient step-up at the new degree.
//   - Stage 3: return the final physical coefficients or reject a nonfinite stage.
//
// Behavior highlights:
//   - The transformation does not clip, rescale or repair unstable candidates.
//
// Inputs:
//   - unrestricted: finite optimization coordinates visited by increasing degree.
//
// Returns:
//   - []float64: owned physical stable-polynomial coefficients;
//     bool: false on numeric boundary or overflow, with no usable coefficients.
//
// Errors:
//   - None directly; decode maps a false result to the proper AR/MA sentinel.
//
// Determinism:
//   - Adds exactly one degree at a time, scanning existing coefficients forward.
//
// Complexity:
//   - Time O(m^2), Space O(m) retained plus temporary degree arrays.
//
// Notes:
//   - For MA the returned stable polynomial is negated to obtain theta.
//
// AI-Hints:
//   - Avoid treating tanh saturation as proof of a finite interior root;
//     at binary64 precision its result can round to exactly +/-1.
func fromReflection(unrestricted []float64) ([]float64, bool) {
	coefficients := make([]float64, 0, len(unrestricted))
	for _, coordinate := range unrestricted {
		reflection := math.Tanh(coordinate)
		if !finite(reflection) || math.Abs(reflection) >= 1 {
			return nil, false
		}
		next := make([]float64, len(coefficients)+1)
		for j := range coefficients {
			next[j] = coefficients[j] - reflection*coefficients[len(coefficients)-1-j]
			if !finite(next[j]) {
				return nil, false
			}
		}
		next[len(coefficients)] = reflection
		coefficients = next
	}
	return coefficients, true
}

// decode converts local optimizer coordinates to signed physical ARMA parameters.
// It applies the same stable reflection map independently to AR and MA,
// preserving the public plus-theta convention in the differenced equation.
//
// Implementation:
//   - Stage 1: decode AR free coordinates by Levinson step-up.
//   - Stage 2: decode MA free coordinates and negate the polynomial to theta.
//   - Stage 3: apply optional c, preserving an exactly zero intercept otherwise.
//
// Behavior highlights:
//   - The order's integration degree is attached by the caller because it does
//     not change the stationary ARMA polynomial map.
//
// Inputs:
//   - unrestricted: [p AR reflections, q MA reflections, optional intercept].
//   - p, q: validated coefficient-vector lengths; includeDrift: fit c when true.
//
// Returns:
//   - Parameters: owned physical AR, MA and intercept candidate.
//
// Errors:
//   - ErrNonStationary: invalid AR reflection candidate.
//   - ErrNonInvertible: invalid MA reflection candidate.
//   - ErrOverflow: candidate intercept cannot be represented as finite float64.
//
// Determinism:
//   - AR degree scan precedes MA, and any intercept is read last.
//
// Complexity:
//   - Time O(p^2+q^2), Space O(p+q).
//
// Notes:
//   - The variance is profiled later; this function does not estimate sigma^2.
//
// AI-Hints:
//   - Reversing the MA sign changes innovation recurrence and forecast impulse
//     response. Do not remove the explicit theta=-stablePolynomial conversion.
func decode(unrestricted []float64, p, q int, includeDrift bool) (Parameters, error) {
	ar, ok := fromReflection(unrestricted[:p])
	if !ok {
		return Parameters{}, ErrNonStationary
	}
	stableMA, ok := fromReflection(unrestricted[p : p+q])
	if !ok {
		return Parameters{}, ErrNonInvertible
	}
	ma := make([]float64, len(stableMA))
	for i, coefficient := range stableMA {
		ma[i] = -coefficient
	}
	intercept := 0.0
	if includeDrift {
		intercept = unrestricted[p+q]
	}
	if !finite(intercept) {
		return Parameters{}, ErrOverflow
	}
	return Parameters{AR: ar, MA: ma, Drift: intercept, Order: Order{AR: p, MA: q}}, nil
}

// reflectionJac differentiates the polynomial step-up map with respect to u.
// The full Jacobian carries both direct and mirrored previous-degree effects
// and multiplies each new reflection derivative by 1-tanh(u_m)^2.
//
// Implementation:
//   - Stage 1: begin with the zero-degree coefficient and sensitivity arrays.
//   - Stage 2: insert each reflection, propagating coefficient derivatives
//     through the same mirrored step-up recurrence used by fromReflection.
//   - Stage 3: return the completed small Jacobian or reject a saturated step.
//
// Behavior highlights:
//   - The temporary matrix is polynomial-order-sized (degree <=8), not the
//     horizon-by-horizon covariance prohibited in forecasting.
//
// Inputs:
//   - unrestricted: finite local polynomial free coordinates.
//
// Returns:
//   - [][]float64: coefficients-by-u Jacobian; bool: false on invalid reflection.
//
// Errors:
//   - None directly; evaluate converts rejection to the AR/MA domain sentinel.
//
// Determinism:
//   - Degrees, Jacobian rows, and coordinate columns increase in a fixed order.
//
// Complexity:
//   - Time O(m^3), Space O(m^2), with bounded m<=8.
//
// Notes:
//   - evaluate negates its MA Jacobian contribution because theta=-a.
//
// AI-Hints:
//   - Do not differentiate the AR transform once and reuse its sign for MA;
//     the MA physical parameter map has an additional negative sign.
func reflectionJac(unrestricted []float64) ([][]float64, bool) {
	coefficients := []float64{}
	jacobian := [][]float64{}
	for degree, coordinate := range unrestricted {
		reflection := math.Tanh(coordinate)
		if !finite(reflection) || math.Abs(reflection) >= 1 {
			return nil, false
		}
		next := make([]float64, degree+1)
		nextJacobian := make([][]float64, degree+1)
		for row := range nextJacobian {
			nextJacobian[row] = make([]float64, len(unrestricted))
		}
		for row := 0; row < degree; row++ {
			mirror := degree - 1 - row
			next[row] = coefficients[row] - reflection*coefficients[mirror]
			for column := 0; column < degree; column++ {
				nextJacobian[row][column] = jacobian[row][column] - reflection*jacobian[mirror][column]
			}
			nextJacobian[row][degree] = -(1 - reflection*reflection) * coefficients[mirror]
		}
		next[degree] = reflection
		nextJacobian[degree][degree] = 1 - reflection*reflection
		coefficients, jacobian = next, nextJacobian
	}
	return jacobian, true
}
