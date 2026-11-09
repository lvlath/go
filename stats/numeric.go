// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package stats implements finite-range-independent floating-point arithmetic.
// This file owns the scaled double-double primitives used by both moments kernels.
package stats

import "math"

const (
	// countLowBits is the number of exactly representable bits in each
	// uint64 limb; two limbs preserve every possible accumulator count.
	countLowBits = 32

	// countLowMask extracts the lower limb without a lossy uint64->float64
	// conversion. A direct conversion can round a large count by thousands.
	countLowMask = (uint64(1) << countLowBits) - 1

	// maxExactlyRepresentableCount is the largest integer for which a direct
	// float64 conversion introduces no rounding error.
	maxExactlyRepresentableCount = uint64(1) << 53
)

// scaledNumber stores a signed number as (hi + lo) * 2^exponent.
// The leading significand is normalized to [0.5, 1) in absolute value, and
// the trailing significand preserves a second round of meaningful binary bits.
//
// Invariants:
//   - hi == 0 represents exact zero; lo and exponent are then zero.
//   - All stored significands are finite; the exponent is an ordinary Go int.
//   - No arithmetic operation must materialize an out-of-range float64 value.
//   - Distinct finite inputs separated by one ULP remain distinguishable even
//     when their arithmetic mean lies between adjacent float64 values.
//
// Complexity: O(1) time and space, using a bounded number of IEEE-754 ops.
// AI-Hints: an exponent is NOT a logarithm of a variance; its sign matters for
// subnormal observations, and it must never be clamped to the float64 range.
type scaledNumber struct {
	// hi carries the dominant signed significand and the sign of the value.
	hi float64

	// lo carries the compensated low-order significand after normalization.
	lo float64

	// exponent is the base-two scale, potentially outside float64 exponents.
	exponent int
}

// scaledFromFloat lifts a finite float64 into the extended exponent domain.
// Its representation preserves subnormal inputs exactly and canonicalizes both signs of zero.
// Implementation:
//   - Stage 1: preserve the unique zero representation.
//   - Stage 2: split nonzero input with Frexp, without any arithmetic rounding.
//   - Stage 3: return the normalized, alias-free scalar.
//
// Inputs: value is a finite float64, validated by the public caller.
// Returns: a scaled number equal to the exact binary input.
// Errors: none; callers must reject NaN and infinity before this helper.
// Determinism: IEEE-754 Frexp with no caller-dependent state.
// Complexity: O(1) time and space.
// AI-Hints: do not replace Frexp with log2 or decimal exponent decomposition.
func scaledFromFloat(value float64) scaledNumber {
	if value == 0 {
		return scaledNumber{}
	}

	fraction, exponent := math.Frexp(value)
	return scaledNumber{hi: fraction, exponent: exponent}
}

// scaledFromCount represents every positive uint64 count exactly.
// This avoids count rounding in large partition weights during Chan merging.
// Implementation:
//   - Stage 1: use a direct conversion for counts in the exact float64 range.
//   - Stage 2: decompose larger counts into two 32-bit exact limbs.
//   - Stage 3: normalize their sum as a compensated scaled scalar.
//
// Inputs: count is a uint64 accumulator count, including zero.
// Returns: an exact scaled representation of count.
// Errors: none; overflow of count addition is validated by the caller.
// Determinism: fixed high-limb then low-limb addition order.
// Complexity: O(1) time and space.
// AI-Hints: float64(math.MaxUint64) rounds to 2^64; use this helper instead.
func scaledFromCount(count uint64) scaledNumber {
	if count <= maxExactlyRepresentableCount {
		return scaledFromFloat(float64(count))
	}

	high := scaledFromFloat(float64(count >> countLowBits))
	high.exponent += countLowBits
	low := scaledFromFloat(float64(count & countLowMask))
	return high.add(low)
}

// twoSum separates a floating-point sum into its rounded part and lost bits.
// For finite operands whose sum does not overflow, rounded + residual is equal
// to the exact real sum of the two input float64 values.
// Implementation:
//   - Stage 1: round the ordinary sum.
//   - Stage 2: reconstruct the rounding error using the TwoSum identity.
//   - Stage 3: return both non-aliased scalar components.
//
// Inputs: a and b are finite numbers with a finite sum.
// Returns: rounded sum and exact floating residual.
// Errors: none; the scaled caller keeps operands in a bounded domain.
// Determinism: fixed arithmetic sequence; no parallel reordering.
// Complexity: O(1) time and space.
// AI-Hints: this is the unrestricted TwoSum, not FastTwoSum (which needs |a|>=|b|).
func twoSum(a, b float64) (float64, float64) {
	sum := a + b
	recovered := sum - a
	residual := (a - (sum - recovered)) + (b - recovered)
	return sum, residual
}

// normalizeScaled restores the extended significand invariant after an op.
// A cancellation may leave a tiny nonzero residual; normalizing this residual
// is essential to detect variance in almost-constant training columns.
// Implementation:
//   - Stage 1: fold both significands through error-free TwoSum.
//   - Stage 2: recognize exact zero, including a canceled leading term.
//   - Stage 3: use Frexp on the nonzero lead and scale the trailing bits.
//
// Inputs: hi, lo are finite bounded parts; exponent is their shared scale.
// Returns: normalized signed scaledNumber, with exact-zero canonicalization.
// Errors: none for valid internal arithmetic.
// Determinism: fixed operations, independent of sample grouping.
// Complexity: O(1) time and space.
// AI-Hints: do not test |hi| against an epsilon to declare constant variance.
func normalizeScaled(hi, lo float64, exponent int) scaledNumber {
	lead, tail := twoSum(hi, lo)
	if lead == 0 {
		lead, tail = tail, 0
	}
	if lead == 0 {
		return scaledNumber{}
	}

	fraction, shift := math.Frexp(lead)
	return scaledNumber{
		hi:       fraction,
		lo:       math.Ldexp(tail, -shift),
		exponent: exponent + shift,
	}
}

// neg returns the exact arithmetic negation without changing storage scale.
// Implementation: negate both compensated components and preserve exponent.
// Inputs: number is immutable and may be zero.
// Returns: an independently stored signed number.
// Errors: none.
// Determinism: no branches except zero canonicalization.
// Complexity: O(1) time and space.
// AI-Hints: do not negate only hi; the residual is part of the numeric value.
func (number scaledNumber) neg() scaledNumber {
	return scaledNumber{hi: -number.hi, lo: -number.lo, exponent: number.exponent}
}

// add computes a compensated sum without float64 intermediate overflow.
// Normalized significands are aligned before summation; their magnitudes
// remain bounded even when the mathematical result requires exponent > 1024.
// Implementation:
//   - Stage 1: fast-return if either operand is exact zero.
//   - Stage 2: align both operands to the larger exponent.
//   - Stage 3: combine high and low terms, then renormalize cancellation.
//
// Inputs: number and other are immutable scaled numbers.
// Returns: their sum with roughly double-float significand precision.
// Errors: none for valid internal values.
// Determinism: fixed high-part before low-part evaluation order.
// Complexity: O(1) time and space.
// AI-Hints: never construct number.toFloat()+other.toFloat() here.
func (number scaledNumber) add(other scaledNumber) scaledNumber {
	if number.hi == 0 {
		return other
	}
	if other.hi == 0 {
		return number
	}

	exponent := number.exponent
	if other.exponent > exponent {
		exponent = other.exponent
	}

	aHigh := math.Ldexp(number.hi, number.exponent-exponent)
	aLow := math.Ldexp(number.lo, number.exponent-exponent)
	bHigh := math.Ldexp(other.hi, other.exponent-exponent)
	bLow := math.Ldexp(other.lo, other.exponent-exponent)

	high, highResidual := twoSum(aHigh, bHigh)
	low, lowResidual := twoSum(aLow, bLow)
	combined, combinedResidual := twoSum(high, low)
	return normalizeScaled(combined, highResidual+lowResidual+combinedResidual, exponent)
}

// sub implements a compensated difference through the shared summation law.
// This preserves close float64 observations by aligning their binary exponents
// before subtraction, including when x-mean would overflow in float64.
// Implementation: negate the right operand, then apply the canonical add.
// Inputs: two immutable scaled values.
// Returns: number minus other in extended exponent arithmetic.
// Errors: none.
// Determinism: exactly the add traversal order.
// Complexity: O(1) time and space.
// AI-Hints: do not compute a raw float64 delta before invoking this helper.
func (number scaledNumber) sub(other scaledNumber) scaledNumber {
	return number.add(other.neg())
}

// mul forms a compensated product without materializing huge squares.
// Multiplying two bounded significands and adding their FMA residual keeps
// products of 1e308-sized observations representable internally.
// Implementation:
//   - Stage 1: short-circuit exact zero to avoid meaningless exponents.
//   - Stage 2: multiply leading significands and recover the exact FMA tail.
//   - Stage 3: add first-order low products, then renormalize.
//
// Inputs: immutable operands, each with a normalized significand.
// Returns: compensated product, with exponent equal to the exponent sum.
// Errors: none for valid internal values.
// Determinism: fixed FMA and cross-term evaluation sequence.
// Complexity: O(1) time and space.
// AI-Hints: use this for delta squared; float64(delta*delta) can overflow.
func (number scaledNumber) mul(other scaledNumber) scaledNumber {
	if number.hi == 0 || other.hi == 0 {
		return scaledNumber{}
	}

	leading := number.hi * other.hi
	productResidual := math.FMA(number.hi, other.hi, -leading)
	crossTerms := (number.hi*other.lo + number.lo*other.hi) + number.lo*other.lo
	return normalizeScaled(leading, productResidual+crossTerms, number.exponent+other.exponent)
}

// div performs a compensated ratio while preserving the full exponent range.
// The quotient's low component corrects leading-division rounding using FMA;
// only a zero divisor violates the internal precondition.
// Implementation:
//   - Stage 1: return exact zero if the numerator is zero.
//   - Stage 2: divide normalized leading significands.
//   - Stage 3: correct the remainder, then subtract the exponents.
//
// Inputs: numerator number and nonzero denominator other are immutable.
// Returns: compensated extended-range quotient.
// Errors: none; all public callers enforce a positive count or fitted scale.
// Determinism: fixed FMA remainder and correction order.
// Complexity: O(1) time and space.
// AI-Hints: dividing by float64(count) loses count bits above 2^53.
func (number scaledNumber) div(other scaledNumber) scaledNumber {
	if number.hi == 0 {
		return scaledNumber{}
	}

	leading := number.hi / other.hi
	remainder := math.FMA(-leading, other.hi, number.hi)
	correction := ((remainder + number.lo) - leading*other.lo) / other.hi
	return normalizeScaled(leading, correction, number.exponent-other.exponent)
}

// toFloat rounds an extended-range value at the public float64 boundary.
// This intentionally defers overflow detection until an actual materialized
// result is required, rather than rejecting representable moments early.
// Implementation:
//   - Stage 1: collapse compensated mantissa into nearest float64 fraction.
//   - Stage 2: restore the exponent with Ldexp.
//   - Stage 3: reject non-finite results, preserving the sentinel contract.
//
// Inputs: an internal finite scaled value (possibly larger than MaxFloat64).
// Returns: finite float64, including subnormal values or canonical positive zero.
// Errors: ErrOverflow when the rounded result is infinite.
// Determinism: IEEE nearest-even multiplication by a power of two.
// Complexity: O(1) time and space.
// AI-Hints: a zero result can be an underflowed nonzero variance, not a constant.
func (number scaledNumber) toFloat() (float64, error) {
	if number.hi == 0 {
		return 0, nil
	}

	value := math.Ldexp(number.hi+number.lo, number.exponent)
	if !finite(value) {
		return 0, ErrOverflow
	}
	return value, nil
}

// sqrtFloat materializes the square root of a nonnegative scaled number.
// Even when the variance exponent exceeds float64, the standard deviation
// may still fit; square-rooting the exponent before conversion preserves it.
// Implementation:
//   - Stage 1: reject non-positive values which cannot define a fitted scale.
//   - Stage 2: make the exponent even by shifting one power of two into mantissa.
//   - Stage 3: take sqrt and restore half the exponent once.
//
// Inputs: a positive variance represented in extended-range arithmetic.
// Returns: strictly positive finite float64 standard deviation.
// Errors: ErrOverflow if its magnitude overflows or underflows to zero.
// Determinism: IEEE math.Sqrt with fixed exponent parity adjustment.
// Complexity: O(1) time and space.
// AI-Hints: never compute math.Sqrt(variance.toFloat()); variance may exceed MaxFloat64.
func (number scaledNumber) sqrtFloat() (float64, error) {
	if number.hi <= 0 {
		return 0, ErrOverflow
	}

	mantissa := number.hi + number.lo
	exponent := number.exponent
	if exponent%2 != 0 {
		mantissa *= 2
		exponent--
	}

	root := math.Ldexp(math.Sqrt(mantissa), exponent/2)
	if root == 0 || !finite(root) {
		return 0, ErrOverflow
	}
	return root, nil
}
