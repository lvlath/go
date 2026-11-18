// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package jumpdiffusion computes the arithmetic mean relative jump multiplier.
// This file separates the Merton drift compensator from the log-jump mean.
package jumpdiffusion

import "math"

// jumpCompensator computes E[exp(Y)-1] for Y distributed as Normal(mean,std^2).
// Wide evaluation of mean+std^2/2 protects valid cancellation and avoids overflow.
//
// Implementation:
//   - Stage 1: square the standard deviation at extended precision.
//   - Stage 2: add the log-jump mean before rounding the combined exponent.
//   - Stage 3: apply Expm1 to retain precision near zero and check representability.
//
// Behavior highlights:
//   - A finite large std is not rejected just because std*std overflows float64.
//   - Compensated callers reject an unrepresentable k before the first RNG draw.
//
// Inputs:
//   - mean: finite dimensionless E[Y]; std: finite dimensionless stddev >=0.
//
// Returns:
//   - float64: finite k=E[exp(Y)-1]; no output is published on overflow.
//
// Errors:
//   - ErrOverflow: the exponent or the resulting arithmetic mean is too large.
//
// Determinism:
//   - One exact-input wide square/add followed by one math.Expm1 call.
//
// Complexity:
//   - Time O(1), Space O(1) at fixed 2304-bit arithmetic precision.
//
// Notes:
//   - Callers validate the input domain; Uncompensated paths do not call this.
//
// AI-Hints:
//   - Never substitute mean for k or use math.Exp(exponent)-1 near zero.
//   - Never reject based solely on std>sqrt(MaxFloat64): half-variance may fit.
func jumpCompensator(mean, std float64) (float64, error) {
	// The squared term has at most 106 significant binary bits, so the fixed
	// wide precision preserves all finite input bits through cancellation.
	halfVariance := newWide().Mul(wide(std), wide(std))
	halfVariance.Mul(halfVariance, wide(0.5))
	exponent := newWide().Add(wide(mean), halfVariance)
	rounded, _ := exponent.Float64()
	if !finite(rounded) {
		return 0, ErrOverflow
	}

	compensator := math.Expm1(rounded)
	if !finite(compensator) {
		return 0, ErrOverflow
	}
	return compensator, nil
}
