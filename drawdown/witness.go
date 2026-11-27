// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package drawdown orders exact represented-input ratios without floating-point products.
// This file owns normalized binary64 decomposition and bounded product comparison.
package drawdown

import (
	"math"
	"math/bits"
)

const (
	// significandBits includes the implicit leading bit of a normal binary64.
	// Frexp normalizes subnormals to the same 53-bit integer interval.
	significandBits = 53
	// depthGrid is the even integer scale N=2^53 for deep-loss rounding.
	// It is also the exact power-of-two factor used to recover significands.
	depthGrid = 1 << significandBits
	// depthStep is the spacing of binary64 values in [0.5,1), and the gap
	// from the greatest value below 1 to 1 itself.
	depthStep = 0x1p-53
	// productWordBits is the width of each word in Mul64/Div64 arithmetic.
	// It is independent of the host int width, including on 32-bit Go targets.
	productWordBits = 64
)

// binaryValue represents one positive finite binary64 as mantissa*2^exponent.
// It normalizes normal and subnormal values to the same integer interval, so
// product comparisons and deep rounding never depend on floating exponent range.
//
// Behavior highlights:
//   - 2^52 <= mantissa < 2^53; the exponent is adjusted for that normalization.
//
// Inputs:
//   - Construct only with splitBinary after equity validation and zero handling.
//
// Returns:
//   - A value-copy operand with exact binary64 semantics.
//
// Complexity:
//   - Space O(1), no dynamic integer storage.
//
// Notes:
//   - The zero value is not a normalized operand and must not reach Div64.
//
// AI-Hints:
//   - Do not replace this representation with a rounded logarithm or ratio.
type binaryValue struct {
	mantissa uint64 // Integer significand; 53 significant bits even for subnormals.
	exponent int    // Signed power of two; bounded by the binary64 input domain.
}

// splitBinary normalizes a positive finite operand without rounding.
// Frexp and exact power-of-two scaling recover the represented binary significand.
//
// Implementation:
//   - Stage 1: decompose value into fraction in [0.5,1) and a signed exponent.
//   - Stage 2: scale the fraction by 2^53, yielding an exact integer.
//   - Stage 3: compensate the exponent so the represented value is unchanged.
//
// Behavior highlights:
//   - The uint64 conversion is exact and bounded below 2^53, even for subnormals.
//
// Inputs:
//   - value: validated positive finite binary64; never zero or nonfinite.
//
// Returns:
//   - binaryValue: exact normalized pair.
//
// Determinism:
//   - No rounding decisions, global state, or order dependence.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// AI-Hints:
//   - Keep zero handling in the caller; the normalized product proof needs 53 bits.
func splitBinary(value float64) binaryValue {
	fraction, exponent := math.Frexp(value) // Exact binary decomposition.
	// Transfer 53 binary digits into an integer and shift the exponent back.
	return binaryValue{mantissa: uint64(fraction * depthGrid), exponent: exponent - significandBits}
}

// deeper reports whether the candidate drawdown is strictly deeper than the incumbent.
// Comparing residual E/P ratios reverses the depth ordering without rounding.
//
// Implementation:
//   - Stage 1: compare equity directly if both peaks are equal and positive.
//   - Stage 2: resolve zero-equity cases before normalizing positive operands.
//   - Stage 3: compare equity*bestPeak < bestEquity*peak exactly in integer words.
//
// Behavior highlights:
//   - True means strict improvement; equality always preserves the earlier trough.
//   - Positive residuals rounded to depth 1 still lose to actual total loss.
//
// Inputs:
//   - peak, bestPeak: validated positive running maxima.
//   - equity, bestEquity: validated nonnegative values strictly below their peaks.
//
// Returns:
//   - bool: whether the candidate has a strictly greater exact relative depth.
//
// Determinism:
//   - No floating ratio, tolerance, or rounded-depth comparison participates.
//
// Complexity:
//   - Time O(1), Space O(1); positive cross-products fit in two uint64 words.
//
// AI-Hints:
//   - Do not change < to <= or compare published Depth values to resolve a tie.
func deeper(peak, equity, bestPeak, bestEquity float64) bool {
	if peak == bestPeak { // Same denominator: the lower equity has the greater depth.
		return equity < bestEquity
	}
	if equity == 0 || bestEquity == 0 { // A real zero residual is exactly maximal.
		return equity == 0 && bestEquity != 0
	}
	// Cross-multiply exact residual ratios; denominators are strictly positive.
	return productLess(splitBinary(equity), splitBinary(bestPeak), splitBinary(bestEquity), splitBinary(peak))
}

// productLess compares a*b with c*d exactly for normalized positive operands.
// It preserves the full mathematical product even when its float64 value would
// overflow, underflow, or collide with another product after rounding.
//
// Implementation:
//   - Stage 1: sum binary exponents and eliminate differences of two or more.
//   - Stage 2: multiply each pair of 53-bit significands into two 64-bit words.
//   - Stage 3: align to the smaller exponent by at most one bit, then compare
//     unsigned high/low words lexicographically with strict less-than.
//
// Behavior highlights:
//   - Each unshifted product lies in [2^104,2^106), hence a gap >=2 decides.
//   - A shifted product uses at most 107 bits; shifting one word cannot lose data.
//
// Inputs:
//   - a, b, c, d: normalized positive binaryValue operands, never zero values.
//
// Returns:
//   - bool: exact a*b < c*d; false also for exact equality.
//
// Determinism:
//   - Integer exponent and word comparisons are independent of float rounding.
//
// Complexity:
//   - Time O(1), Space O(1), with at most two Mul64 operations and one alignment.
//
// Notes:
//   - Exponent sums fit even a 32-bit int for all finite binary64 inputs.
//
// AI-Hints:
//   - Do not compare only high words or shift without carrying the low word's top bit.
func productLess(a, b, c, d binaryValue) bool {
	// Products share the same normalized significand interval; compare scale first.
	leftExponent := a.exponent + b.exponent
	rightExponent := c.exponent + d.exponent
	if leftExponent > rightExponent+1 {
		return false // The left lower bound already exceeds the right upper bound.
	}
	if rightExponent > leftExponent+1 {
		return true // The symmetric exponent-bound argument settles the order.
	}
	// Retain all significand bits instead of truncating a 106-bit product.
	leftHi, leftLo := bits.Mul64(a.mantissa, b.mantissa)
	rightHi, rightLo := bits.Mul64(c.mantissa, d.mantissa)
	// Align the operand with the larger exponent to the common smaller scale.
	if leftExponent > rightExponent {
		leftHi, leftLo = leftHi<<1|leftLo>>(productWordBits-1), leftLo<<1
	} else if rightExponent > leftExponent {
		rightHi, rightLo = rightHi<<1|rightLo>>(productWordBits-1), rightLo<<1
	}
	// Equal high words require the complete low-word comparison; ties return false.
	return leftHi < rightHi || (leftHi == rightHi && leftLo < rightLo)
}
