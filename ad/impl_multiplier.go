// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ad calculates the close-location multiplier using bounded distances.
// This file owns the floating-point geometry shared by batch and stream transitions.
package ad

// moneyFlowMultiplier returns the Chaikin close-location multiplier for a valid Bar.
// It measures the signed fraction of the candle range favoring the close side.
//
// Implementation:
//   - Stage 1: return +0 for a flat candle, before attempting division.
//   - Stage 2: subtract Low from Close and Close from High to obtain nonnegative distances.
//   - Stage 3: divide their signed difference by the strictly positive candle spread.
//
// Behavior highlights:
//   - For valid positive finite OHLC, left and right lie within [0, spread].
//   - IEEE-754 monotonicity therefore bounds the computed quotient to [-1, 1].
//   - Direct 2*Close-High-Low is forbidden: it can overflow for valid prices.
//
// Inputs:
//   - bar: already validated finite positive OHLC with Low<=Close<=High.
//
// Returns:
//   - float64: finite multiplier in [-1, 1], or +0 for High==Low.
//
// Errors:
//   - None; callers must use validateBar/validateColumns first.
//
// Determinism:
//   - Exactly the same left, right, spread subtraction and division order for every caller.
//
// Complexity:
//   - Time O(1), Space O(1), with no allocations.
//
// Notes:
//   - Result is a float64 approximation of the exact close-location fraction, not rational arithmetic.
//   - There is no epsilon clamp; with validated inputs the calculated bound holds already.
//
// AI-Hints:
//   - Do not replace the difference of distances by 2*Close or an averaged price.
//   - Do not divide when High==Low, even if the candle's Volume is zero.
func moneyFlowMultiplier(bar Bar) float64 {
	// A zero spread makes the analytic fraction undefined; Chaikin A/D assigns
	// a zero contribution to this valid geometric edge case by explicit policy.
	if bar.High == bar.Low {
		return 0
	}

	// Every operand of these subtractions is positive and finite, and the
	// larger endpoint never exceeds MaxFloat64. No subtraction overflows.
	left := bar.Close - bar.Low
	right := bar.High - bar.Close
	spread := bar.High - bar.Low

	// Monotonic rounding preserves left<=spread and right<=spread. Subtracting
	// the nonnegative distances cannot exceed spread in either direction.
	return (left - right) / spread
}
