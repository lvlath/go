// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package lstm defines numerically stable activation functions for IFGO gates.
// This file keeps saturation safe only after the affine expression has been validated.
package lstm

import "math"

// sigmoid applies the logistic activation to a previously validated finite affine.
// Its sign split avoids exponential overflow when x has very large negative magnitude.
//
// Implementation:
//   - Stage 1: x>=0 uses 1/(1+exp(-x)), whose exponential argument is nonpositive.
//   - Stage 2: x<0 uses exp(x)/(1+exp(x)), also with a nonpositive exponent.
//   - Stage 3: return a value in [0,1] without generating a positive-exponent Inf.
//
// Behavior highlights:
//   - Finite large inputs may legitimately saturate near 0 or 1 in binary64.
//
// Inputs:
//   - x: finite float64 preactivation, already checked by affine.
//
// Returns:
//   - float64: stable logistic value in the closed binary64 interval [0,1].
//
// Errors:
//   - None: affine is responsible for detecting computed overflow first.
//
// Determinism:
//   - Exactly one branch according to x sign, with fixed math.Exp operations.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This function is deliberately private so callers cannot bypass affine validation.
//
// AI-Hints:
//   - Reject an overflowing Wx+Uh+b BEFORE considering gate saturation.
func sigmoid(x float64) float64 {
	if x >= 0 {
		return 1 / (1 + math.Exp(-x))
	}
	exponential := math.Exp(x)
	return exponential / (1 + exponential)
}
