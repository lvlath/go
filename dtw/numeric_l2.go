// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw supplies stable squared Euclidean local costs for multivariate DTW.
// This file isolates the coordinate-difference kernel from matrix adapters.
package dtw

import "math"

// squaredSumOfDifferences accumulates squared coordinate differences in a
// shared exponent domain, avoiding premature squaring of very large inputs.
// The zero value is a valid empty accumulator.
type squaredSumOfDifferences struct {
	// maxExponent is the largest Frexp exponent of any nonzero difference.
	maxExponent int
	// scaledSum is the ordered sum of squares after common-exponent rescaling.
	scaledSum float64
	// hasValue distinguishes all-zero vector differences from a non-empty scale.
	hasValue bool
	// invalid records non-finite original coordinates (not arithmetic overflow).
	invalid bool
	// overflow records a mathematically unrepresentable difference of finite coordinates.
	overflow bool
}

// addDifference incorporates one pair of finite coordinates into the scaled sum.
// It forms their difference before squaring, avoiding cancellation in the
// norm-identity formula and premature overflow of intermediate squares.
//
// Implementation:
//   - Stage 1: reject a non-finite coordinate and classify an overflowing difference.
//   - Stage 2: decompose each nonzero absolute difference with Frexp.
//   - Stage 3: rescale the existing accumulator or the incoming term to a
//     shared exponent, then accumulate its squared mantissa.
//
// Behavior highlights:
//   - Finite nonzero binary64 differences have Frexp exponents from -1073 to 1024.
//   - Dropping a vastly smaller squared contribution is ordinary binary64 rounding.
//
// Inputs:
//   - a, b: coordinates in the same feature dimension and units.
//
// Returns:
//   - None; modifies only private accumulator fields.
//
// Errors:
//   - None; invalid and overflow conditions are recorded for value().
//
// Determinism:
//   - Fixed coordinate order and IEEE-754 rounding at each arithmetic step.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Inputs are never modified, and the accumulator is caller-owned.
//
// AI-Hints:
//   - Never square each raw coordinate before subtraction.
//   - Preserve subnormal differences; do not apply an absolute-value clamp.
func (sum *squaredSumOfDifferences) addDifference(a, b float64) {
	if math.IsNaN(a) || math.IsNaN(b) || math.IsInf(a, 0) || math.IsInf(b, 0) {
		sum.invalid = true
		return
	}

	absolute := math.Abs(a - b)
	if math.IsInf(absolute, 1) {
		sum.overflow = true
		return
	}
	if absolute == 0 {
		return
	}

	mantissa, exponent := math.Frexp(absolute)
	if !sum.hasValue {
		sum.scaledSum = mantissa * mantissa
		sum.maxExponent = exponent
		sum.hasValue = true
		return
	}

	if exponent > sum.maxExponent {
		// Rescale prior contributions before adopting the larger exponent.
		// Relative exponents can underflow to zero, which is appropriate when
		// those contributions are smaller than the final sum's precision.
		ratio := math.Ldexp(1, sum.maxExponent-exponent)
		sum.scaledSum = sum.scaledSum*ratio*ratio + mantissa*mantissa
		sum.maxExponent = exponent
		return
	}

	term := math.Ldexp(mantissa, exponent-sum.maxExponent)
	sum.scaledSum += term * term
}

// value reconstructs the final squared distance with a single exponent scaling.
// It returns +Inf for an unrepresentably large non-negative squared distance
// and NaN only for invalid original coordinate inputs.
//
// Implementation:
//   - Stage 1: distinguish invalid inputs and overflowing differences.
//   - Stage 2: return zero for an all-zero coordinate difference.
//   - Stage 3: reconstruct the squared norm with Ldexp and detect range overflow.
//
// Behavior highlights:
//   - Positive exact squared values below the smallest subnormal may round to zero.
//
// Inputs:
//   - sum: an accumulator populated by ordered addDifference calls.
//
// Returns:
//   - float64: non-negative squared distance, +Inf on overflow, NaN on invalid input.
//
// Errors:
//   - None; callers classify special float64 results with package sentinels.
//
// Determinism:
//   - One IEEE-754 Ldexp reconstruction after the ordered accumulation.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Rounded float64 distance is not an exact-real arbitrary-precision norm.
//
// AI-Hints:
//   - Do not reconstruct sqrt(sum) and square it again: double rounding can be avoidable.
//   - Do not silently substitute a finite maximum for an unrepresentable cost.
func (sum squaredSumOfDifferences) value() float64 {
	if sum.invalid {
		return math.NaN()
	}
	if sum.overflow {
		return math.Inf(1)
	}
	if !sum.hasValue {
		return 0
	}
	return math.Ldexp(sum.scaledSum, 2*sum.maxExponent)
}

// SquaredL2OfDifferences computes the squared Euclidean distance of two coordinate vectors.
// The local metric is the sum of squared coordinate differences, evaluated
// in float64 without the cancellation-prone identity ||x||²+||y||²-2*x·y.
//
// Implementation:
//   - Stage 1: reject mismatched vector lengths with NaN (the error-free signature
//     cannot convey a typed shape error).
//   - Stage 2: add each coordinate difference to a shared-exponent accumulator.
//   - Stage 3: reconstruct the non-negative squared distance once.
//
// Behavior highlights:
//   - Identical very large vectors have exactly zero local cost.
//   - An unrepresentable squared distance produces +Inf, not a finite clamp.
//   - Non-finite inputs and mismatched lengths produce NaN.
//
// Inputs:
//   - x, y: equally sized, finite coordinate vectors, read-only for this call.
//
// Returns:
//   - float64: squared distance, +Inf for range overflow, NaN for invalid inputs.
//
// Errors:
//   - No Go error result; AlignMatrix maps non-finite outcomes to package sentinels.
//
// Determinism:
//   - Feature coordinates are visited in ascending index order.
//
// Complexity:
//   - Time O(len(x)), Space O(1); no temporary squared-coordinate vectors.
//
// Notes:
//   - A positive squared value below the smallest representable subnormal may round to zero.
//   - Caller mutation of either input during evaluation is unsupported.
//
// AI-Hints:
//   - Use AlignMatrix for validated matrix inputs and sentinel-classified errors.
//   - This helper's +Inf is a numeric range signal, not DTW unreachability.
func SquaredL2OfDifferences(x, y []float64) float64 {
	if len(x) != len(y) {
		return math.NaN()
	}
	var sum squaredSumOfDifferences
	for index := range x {
		sum.addDifference(x[index], y[index])
	}
	return sum.value()
}
