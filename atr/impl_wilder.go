// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package atr provides exact True Range and Wilder Average True Range computations.
// This file implements the canonical exact Wilder recurrence without rounded alpha.
package atr

// wilder computes a single correctly rounded Wilder affine update.
// Nonnegative weights sum to period, so the exact result lies between finite operands.
//
// Implementation:
//   - Stage 1: Create a zero fixed-width numerator in units of 2^-1074.
//   - Stage 2: Accumulate exact (period-1)*previous and exact tr.
//   - Stage 3: Divide by the exact integer period with one nearest-even rounding.
//
// Behavior highlights:
//   - No rounded alpha, floating numerator, delta update, or heap workspace.
//
// Inputs:
//   - previous, tr: finite nonnegative represented operands from validated state.
//   - period: exact integer count greater than one; period one is handled by advance.
//
// Returns:
//   - float64: finite nonnegative Wilder value with canonical +0 when it rounds to zero.
//
// Errors:
//   - None; the convex-combination and fixed-capacity bounds prove representability.
//
// Determinism:
//   - Integer operations are exact; only the final binary64 publication is rounded.
//
// Complexity:
//   - Time O(W), Space O(W), W=34 fixed limbs independent of period.
//
// Notes:
//   - The prior ATR is already rounded; this is not an unrounded full-history sum.
//
// AI-Hints:
//   - Do not reuse an EMA kernel with weights 2 and P+1.
//   - Do not materialize (P-1)*previous in float64 before dividing.
func wilder(previous, tr float64, period uint64) float64 {
	// Fixed stack storage holds the exact nonnegative numerator.
	var numerator exactSum
	// Retain the exact integer weight, including periods larger than 2^53.
	numerator.addProduct(previous, period-1)
	// Current TR contributes once; it has already been rounded by the TR kernel.
	numerator.addProduct(tr, 1)
	// The only rounding in this affine step occurs after exact integer division.
	return numerator.roundedQuotient(period)
}
