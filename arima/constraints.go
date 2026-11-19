// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package arima implements strict Schur coefficient-domain restrictions.
// This file maps unrestricted optimizer coordinates to stationary and invertible polynomials.
package arima

import "math"

// schurStable checks strict stability of 1-sum(a_i*z^i) without finding complex roots.
// The final coefficient at each step is the reflection coefficient k_m; dividing
// by 1-k_m^2 gives the preceding stable polynomial iff |k_m|<1 at every step.
// A zero-length polynomial is stable by definition; unit roots are rejected.
// Time O(p^2), Space O(p), with p bounded by the public order validator.
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

// fromReflection constructs strictly stable physical coefficients from free u.
// It performs the Levinson step-up recurrence with k_m=tanh(u_m) and rejects
// values rounded to +/-1 instead of clipping coefficients into the stable region.
// Time O(p^2), Space O(p), with increasing polynomial degree as the iteration order.
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

// decode transforms unrestricted AR, MA and optional intercept coordinates.
// AR uses the step-up coefficients directly; MA negates the same stable
// polynomial coefficients to implement 1+sum(theta_j*z^j) correctly.
// Time O(p^2+q^2), Space O(p+q), with errors keeping AR and MA failure classes distinct.
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

// reflectionJac differentiates the Levinson step-up coefficients with respect to u.
// The derivative of tanh(u_m) is 1-k_m^2; older coefficients contribute through
// both direct and mirrored terms, so every step updates the complete Jacobian.
// The returned matrix is tiny (degree at most eight), not a forecast covariance.
// Time O(p^3), Space O(p^2), with deterministic degree/row/column traversal.
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
