// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ema provides exact affine exponential moving average smoothing.
// This file rounds exact convex numerators once, without subtracting float levels.
package ema

import (
	"math"
	"math/bits"
)

const (
	// affineSampleWeight is the exact numerator weight of each new observation.
	// The complementary previous-value weight is period-1; their sum is period+1.
	affineSampleWeight = uint64(2)
)

// affine computes one exactly weighted, nearest-even EMA update.
// Its integer numerator avoids both difference overflow and premature rounded weights.
//
// Implementation:
//   - Stage 1: decompose previous and x into exact signed binary64 units.
//   - Stage 2: multiply by period-1 and 2, then add the signed integers.
//   - Stage 3: divide by period+1 and round exactly once through roundUnits.
//
// Behavior highlights:
//   - Opposite extreme values can cancel to zero without any floating-point overflow.
//   - The result lies in the convex hull of its finite operands.
//
// Inputs:
//   - previous, x: finite represented values; previous is the last rounded output.
//   - period: validated integer in [2, MaxInt-1]; P1 is handled by advance.
//
// Returns:
//   - float64: finite correctly rounded result, with zero canonicalized to +0.
//
// Errors:
//   - None; the convex-result and fixed-capacity bounds exclude numerical overflow.
//
// Determinism:
//   - Exact integer arithmetic followed by a single ties-to-even decision.
//
// Complexity:
//   - Time O(W), Space O(W), W=34; no heap allocation.
//
// Notes:
//   - The numerator is bounded by (period+1)*MaxBinary64Units, below 2^2161.
//   - This does not expand the entire history into an unrounded rational expression.
//
// AI-Hints:
//   - Never implement the weights using float64(period) or pre-rounded alpha.
//   - Never replace this with previous+alpha*(x-previous), even with FMA.
func affine(previous, x float64, period uint64) float64 {
	// Build the old contribution exactly, not as a floating-point product.
	numerator := binaryUnits(previous)
	// This ignored overflow flag is justified by the 2161-bit numerator bound.
	numerator.mag.multiplyAdd(period-1, 0)
	// Build the new contribution at the same exact unit scale.
	next := binaryUnits(x)
	// The new observation has the exact integer coefficient two.
	next.mag.multiplyAdd(affineSampleWeight, 0)
	// Signed cancellation is performed on full magnitudes.
	numerator.add(next)
	// Only the final rational quotient is rounded to binary64.
	return roundUnits(&numerator, period+1)
}

// quotient divides the leading integer window used to choose a binary64 significand.
// Bits below shift are retained in a and are considered separately during rounding.
//
// Implementation:
//   - Stage 1: extract low and high limbs of floor(a/2^shift).
//   - Stage 2: divide this 128-bit window by the positive divisor.
//
// Behavior highlights:
//   - The roundUnits window guarantees a quotient below 2^54.
//
// Inputs:
//   - a: non-nil bounded numerator; shift: selected by roundUnits.
//   - divisor: positive period or period+1, within the validated machine-int domain.
//
// Returns:
//   - uint64, uint64: exact quotient and remainder of the shifted integer.
//
// Errors:
//   - None; the quotient bound implies high<divisor, excluding Div64 overflow.
//
// Determinism:
//   - Pure integer division; no implicit rounding modes are used.
//
// Complexity:
//   - Time O(1), Space O(1); exactly two word extractions and one division.
//
// Notes:
//   - For admitted numerators, the shifted integer fits 128 bits; no high bits are discarded.
//
// AI-Hints:
//   - Do not call with an arbitrary shift: the high<divisor proof is essential.
func quotient(a *magnitude, shift int, divisor uint64) (uint64, uint64) {
	// Select the low word of the bounded shifted numerator.
	lo := a.shiftedWord(shift)
	// The remaining high word is below divisor by the quotient-width proof.
	hi := a.shiftedWord(shift + wordBits)
	// Return both pieces; remainder is essential to exact nearest-even rounding.
	return bits.Div64(hi, lo, divisor)
}

// roundUp decides whether the truncated significand must be incremented.
// It compares the exact rational residual with one half, including every tail bit.
//
// Implementation:
//   - Stage 1: compare remainder with floor(divisor/2).
//   - Stage 2: for an even divisor, resolve equality using the sticky tail and significand parity.
//   - Stage 3: for an odd divisor, require the tail guard bit and then resolve any exact tie.
//
// Behavior highlights:
//   - An exact halfway result increments only an odd significand.
//
// Inputs:
//   - a: original nonnegative numerator; shift: number of discarded low bits.
//   - divisor, remainder: positive denominator and remainder<divisor from quotient.
//   - mantissa: truncated nonnegative significand before rounding.
//
// Returns:
//   - bool: true exactly when nearest-even rounding increases the magnitude.
//
// Errors:
//   - None; all inputs are internal division results.
//
// Determinism:
//   - For N=2^shift*(d*q+r)+L, compare r+L/2^shift with d/2 exactly.
//
// Complexity:
//   - Time O(W), Space O(1), W=34; only sticky-bit scans are linear in W.
//
// Notes:
//   - If d is odd and r=floor(d/2), the low tail must reach 1/2 to tie or exceed.
//   - shift=0 implies L=0, so the odd-denominator case cannot be a halfway tie.
//
// AI-Hints:
//   - Do not drop the sticky bits after finding a guard bit: that changes ties into false ties.
func roundUp(a *magnitude, shift int, divisor, remainder, mantissa uint64) bool {
	// Integer division deliberately floors odd denominators.
	half := divisor / 2
	// A whole-unit difference dominates any fractional low tail, which is below one.
	if remainder != half {
		return remainder > half
	}
	// For an even denominator, equality is a tie only when the entire tail is zero.
	if divisor&1 == 0 {
		return a.lowNonzero(shift) || mantissa&1 != 0
	}
	// With no discarded bits, an odd denominator cannot reach its fractional half.
	if shift == 0 {
		return false
	}
	// The guard bit tests whether the discarded tail reaches one half.
	if a.shiftedWord(shift-1)&1 == 0 {
		return false
	}
	// A nonzero sticky tail exceeds halfway; an exact tie uses significand parity.
	return a.lowNonzero(shift-1) || mantissa&1 != 0
}

// roundUnits converts a bounded exact signed rational in subnormal units to binary64.
// It implements one nearest-even rounding, including subnormal, halfway, and carry cases.
//
// Implementation:
//   - Stage 1: choose shift=max(0, bitLen(N)-bitLen(d)-53) for the leading quotient window.
//   - Stage 2: correct a 54-bit quotient by advancing the window one bit and recomputing.
//   - Stage 3: use exact remainder/guard/sticky information, then scale the rounded integer.
//
// Behavior highlights:
//   - Exact or rounded zero is always +0; no epsilon threshold is used.
//   - At most one window correction is needed: the initial quotient has at most 54 bits.
//
// Inputs:
//   - sum: non-nil exact signed numerator of a convex mean of finite binary64 inputs.
//   - divisor: positive exact integer denominator within the admitted period domain.
//
// Returns:
//   - float64: RN64((sum/divisor)*2^-1074), finite and zero-canonical.
//
// Errors:
//   - None; the caller proves |sum|<=divisor*MaxBinary64Units.
//
// Determinism:
//   - One ties-to-even increment is followed only by exact conversion and power-of-two scaling.
//
// Complexity:
//   - Time O(W), Space O(1) auxiliary, W=34; no heap allocation.
//
// Notes:
//   - After rounding, the integer has at most 53 significant bits or is exactly 2^53.
//   - Its conversion is exact; the scale reaches no lower than the subnormal quantum.
//
// AI-Hints:
//   - Never round the shifted numerator before computing its remainder.
//   - A direct float64 numerator conversion would lose the low bits needed to resolve ties.
func roundUnits(sum *exactSum, divisor uint64) float64 {
	// Preserve enough quotient bits to distinguish all normal and subnormal outputs.
	shift := sum.mag.bitLen() - bits.Len64(divisor) - significandBits
	// Subnormal results retain all integer-unit bits before division.
	if shift < 0 {
		shift = 0
	}
	// Extract the truncated significand and the exact remainder together.
	mantissa, remainder := quotient(&sum.mag, shift, divisor)
	// A 54-bit quotient needs one more discarded bit, not a rounded right shift.
	if mantissa >= hiddenBit<<1 {
		// Recompute division so no bit or remainder is double-rounded.
		shift++
		mantissa, remainder = quotient(&sum.mag, shift, divisor)
	}
	// Apply the only inexact-to-binary64 decision in this kernel.
	if roundUp(&sum.mag, shift, divisor, remainder, mantissa) {
		// A carry to 2^53 is itself exactly representable.
		mantissa++
	}
	// Canonicalize all rounded zeros before considering the original sign.
	if mantissa == 0 {
		return 0
	}
	// The integer has <=53 significant bits (or is exactly 2^53).
	// Its conversion and power-of-two scaling are exact, including subnormals.
	result := math.Ldexp(float64(mantissa), shift-unitExponent)
	// Nonzero sign restoration is exact and cannot change the rounding choice.
	if sum.negative {
		return -result
	}
	return result
}
