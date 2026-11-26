// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package donchian computes a correctly rounded finite boundary midpoint.
// This file isolates binary64 evaluation order and the proof behind each magnitude branch.
package donchian

import "math"

// midpoint returns RN64((upper+lower)/2) for two admitted finite boundaries.
// Its evaluation order avoids destructive overflow and subnormal underflow.
//
// Implementation:
//   - Stage 1: canonicalize exact cancellation to positive zero.
//   - Stage 2: use sum-then-half when both magnitudes permit finite addition.
//   - Stage 3: otherwise halve large operands exactly, avoiding unnecessary tiny halving.
//
// Behavior highlights:
//   - Only the exact mathematical mean is rounded to nearest, ties to even.
//   - Negative nonzero means that underflow to zero retain negative zero.
//
// Inputs:
//   - upper, lower: finite observed extrema; the caller has established upper >= lower.
//
// Returns:
//   - float64: finite midpoint lying numerically between the two boundaries.
//
// Errors:
//   - None: a convex mean of two finite bounds is always within the finite range.
//
// Determinism:
//   - Only fixed binary64 operations; explicit float64 conversions preserve rounding
//     boundaries against cross-operation fusion.
//
// Complexity:
//   - O(1) time and O(1) space; no arbitrary-precision arithmetic or heap allocation.
//
// Notes:
//   - Below the half-maximum threshold, addition cannot overflow. If halving can
//     underflow, the sum is an exactly representable multiple of the minimum subnormal;
//     otherwise halving the rounded sum is an exact exponent scaling.
//   - In the large branch, operands at least twice MinNormal have exact halves.
//     A smaller operand is far below half an ULP of the huge operand's half and cannot
//     affect its rounding even when left unscaled.
//
// AI-Hints:
//   - Do not replace this by (upper+lower)/2, lower+(upper-lower)/2, or unconditional
//     upper/2+lower/2: they have different overflow or subnormal failure cases.
//   - Do not infer that upper-lower is representable merely because the mean is finite.
func midpoint(upper, lower float64) float64 {
	// Exact cancellation, including both zero signs, has the canonical +0 result.
	if upper == -lower {
		return 0
	}
	// Both operands below this inclusive threshold have a finite sum.
	const halfMax = math.MaxFloat64 / 2
	// Halving a magnitude at least 2*MinNormal is exact and remains normal.
	const twiceMinNormal = 0x1p-1021
	// An exact power-of-two scale avoids introducing a rounded weighting factor.
	const half = 0.5
	// Magnitudes select a numerically safe evaluation order, never a different mean.
	absUpper, absLower := math.Abs(upper), math.Abs(lower)
	// Here addition cannot overflow; any subnormal final rounding occurs only once.
	if absUpper <= halfMax && absLower <= halfMax {
		return float64(upper+lower) * half
	}
	// The other operand is huge: this tiny upper cannot affect its rounded half.
	if absUpper < twiceMinNormal {
		return upper + float64(lower*half)
	}
	// Symmetric case: do not destroy a tiny lower through an unnecessary halving.
	if absLower < twiceMinNormal {
		return float64(upper*half) + lower
	}
	// Both halves are exact and finite; their final addition supplies nearest-even.
	return float64(upper*half) + float64(lower*half)
}
