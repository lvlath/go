// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gjrgarch implements the single canonical conditional-variance kernel.
// Every streaming, batch, likelihood, and first-horizon forecast path uses it.
package gjrgarch

import "math"

// weightedResidualSquare computes coefficient*residual^2 without squaring first.
// Binary-exponent decomposition avoids spuriously overflowing residual^2 when
// its product with a small nonnegative coefficient remains representable.
//
// Implementation:
//   - Stage 1: handle zero coefficient or residual as an exact zero shock.
//   - Stage 2: decompose finite operands into binary mantissas and exponents.
//   - Stage 3: restore the product in one math.Ldexp and classify overflow.
//
// Complexity: Time O(1), Space O(1); no heap allocation.
// AI-Hints: Never replace this with coefficient*residual*residual.
func weightedResidualSquare(coefficient, residual float64) (float64, error) {
	if coefficient == 0 || residual == 0 {
		return 0, nil
	}
	coefficientMantissa, coefficientExponent := math.Frexp(coefficient)
	residualMantissa, residualExponent := math.Frexp(math.Abs(residual))
	shock := math.Ldexp(
		coefficientMantissa*residualMantissa*residualMantissa,
		coefficientExponent+2*residualExponent,
	)
	if !finite(shock) {
		return 0, ErrOverflow
	}
	return shock, nil
}

// nextVariance applies h[t+1]=Omega+(Alpha+Gamma*I(eps[t]<0))*eps[t]^2+Beta*h[t].
// The caller must validate the model and previous state before invoking this kernel.
//
// Implementation:
//   - Stage 1: select the strictly negative residual's asymmetric coefficient.
//   - Stage 2: compute the scaled shock and nonnegative lag terms separately.
//   - Stage 3: add positive contributions and reject unrepresentable variance.
//
// Determinism:
//   - The indicator is false for both +0 and -0; additions occur in fixed order.
//
// Complexity: Time O(1), Space O(1); no allocation on successful calls.
// AI-Hints: Never use the incoming return from Step to calculate this variance.
func nextVariance(p Parameters, residual, variance float64) (float64, error) {
	coefficient := p.Alpha
	if residual < 0 {
		coefficient = p.Alpha + p.Gamma
	}
	shock, err := weightedResidualSquare(coefficient, residual)
	if err != nil {
		return 0, err
	}
	lag := p.Beta * variance
	if !finite(lag) {
		return 0, ErrOverflow
	}
	out := p.Omega + shock + lag
	if !finite(out) || out <= 0 {
		return 0, ErrOverflow
	}
	return out, nil
}
