// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
//
// Package paa preserves exact signed weighted mass while reducing finite
// binary64 observations. This file implements a fixed-width superaccumulator
// and correctly-rounded publication without floating intermediate overflow.
//
// Every finite float64 is a signed integer multiple of 2^-1074. Instead of
// multiplying a large float64 by an overlap count, this implementation
// accumulates integer mantissa products, cancels signs exactly, and divides
// the final integer by the validated segment length before rounding once.

package paa

import (
	"math"
	"math/bits"
)

// The accumulator stores every value as an integer multiple of 2^-1074.
// The largest finite float64 uses fewer than 2098 such bits. Since the sum of
// nonnegative overlap counts per segment is n<2^63, every per-sign sum is
// strictly less than 2^2161. 36 fixed 64-bit limbs give 2304 bits of space:
// the bound prevents a carry beyond the last limb for every accepted input.
// Fixed width is a correctness proof as well as an O(1) workspace policy.
const accumulatorWords = 36

const (
	// accumulatorWordBits is the number of bits in each uint64 limb.
	// It controls limb offsets and guard/sticky-bit extraction, not Go int size.
	accumulatorWordBits = 64

	// float64FractionBits is the number of explicitly stored fraction bits
	// in an IEEE-754 binary64 encoding (the normal hidden bit is not stored).
	float64FractionBits = 52

	// float64SignificandBits includes the implicit leading bit of a normal
	// float64. Nearest-even rounding retains at most these 53 significant bits.
	float64SignificandBits = float64FractionBits + 1

	// float64ExponentFieldMask isolates the eleven-bit encoded exponent.
	// The all-ones exponent is already rejected by the public finite validator.
	float64ExponentFieldMask = uint64(0x7ff)

	// float64SubnormalShift is -log2(SmallestNonzeroFloat64).
	// Hence every finite binary64 equals an integer times 2^-1074 exactly.
	float64SubnormalShift = 1074
)

// weightedAccumulator retains the exact signed sum of overlap-weighted
// float64 values as two nonnegative fixed-width integers, in 2^-1074 units.
// Each contributor is decoded from its original binary bits, so subtraction
// of very large opposing magnitudes does not discard a subnormal residue.
//
// Implementation:
//   - Stage 1: track the contributor hull and decode each finite float64.
//   - Stage 2: add overlap*mantissa to the appropriate unsigned sign total.
//   - Stage 3: cancel signed totals exactly and round only at finish(n).
//
// Behavior highlights:
//   - positive and negative never share a signed-overflow-prone accumulator.
//   - minimum/maximum are the extrema of samples with positive overlap only.
//   - started distinguishes an empty accumulator from an exact zero total.
//
// Inputs:
//   - add(value,overlap): finite value and strictly positive checked overlap.
//   - finish(n): positive original length in the common integer grid.
//
// Returns:
//   - A finite correctly-rounded segment mean or the classified ErrOverflow.
//
// Determinism:
//   - Contributions arrive in source index order; exact integer addition is
//     order-independent, but the sweep supplies a reproducible traversal.
//
// Complexity:
//   - O(accumulatorWords) worst-case per addition/finalization, a fixed O(1)
//     bound independent of n and w; no per-sample heap allocation.
//
// Notes:
//   - One accumulator is owned by each invocation and reset per output cell.
//   - The fixed bit bound relies on n*w having passed validation.
//
// AI-Hints:
//   - Floating Kahan/Neumaier alone cannot preserve every tiny residual next
//     to arbitrary signed MaxFloat64 contributions; do not downcast the sums.
//   - Never compute value*overlap as float64 before dividing by n.
type weightedAccumulator struct {
	positive [accumulatorWords]uint64
	negative [accumulatorWords]uint64
	minimum  float64
	maximum  float64
	started  bool
}

// add inserts exactly overlap units of a finite source sample into this cell.
// The sample's IEEE-754 bits are decomposed before performing any multiplication.
//
// Implementation:
//   - Stage 1: record contributor min/max for the convex-hull postcondition.
//   - Stage 2: extract sign, fraction and encoded exponent; skip exact zeros.
//   - Stage 3: multiply the integer mantissa by overlap with bits.Mul64.
//   - Stage 4: align the resulting 128-bit product on the 2^-1074 grid and
//     accumulate its up-to-three shifted limbs in the selected sign total.
//
// Inputs:
//   - value: a finite float64 from a validated, read-only source slice.
//   - overlap: strictly positive integer in common 1/w sample-time units.
//
// Returns:
//   - No direct value; updates this per-segment accumulator in place.
//
// Errors:
//   - None from validated inputs: the fixed-width proof precludes carry loss.
//
// Determinism:
//   - Exact IEEE bit decoding and integer arithmetic; no floating reductions.
//
// Complexity:
//   - O(accumulatorWords) worst-case carry propagation, O(1) fixed storage.
//
// Notes:
//   - For normals, exponentField-1 is the mantissa shift on the 2^-1074
//     grid; for subnormals the shift is zero and there is no implicit bit.
//
// AI-Hints:
//   - Multiplying a finite 1e308 float by a raw overlap can overflow even
//     when the final weighted average must remain finite. Only multiply the
//     decoded *integer* mantissa, never the binary64 value itself.
func (a *weightedAccumulator) add(value float64, overlap int64) {
	if !a.started {
		a.started = true
		a.minimum, a.maximum = value, value
	} else {
		if value < a.minimum {
			a.minimum = value
		}
		if value > a.maximum {
			a.maximum = value
		}
	}

	// Decode the stored binary64 value without altering it. Validated finite
	// inputs exclude the all-ones exponent and therefore need no NaN branch.
	bits64 := math.Float64bits(value)
	fraction := bits64 & ((uint64(1) << float64FractionBits) - 1)
	exponent := (bits64 >> float64FractionBits) & float64ExponentFieldMask
	if exponent == 0 && fraction == 0 {
		return // signed zeros are a zero contribution.
	}
	// Subnormals already use 2^-1074 units. Normal values have a hidden
	// leading significand bit and an encoded exponent offset of one.
	shift := uint64(0)
	mantissa := fraction
	if exponent != 0 {
		mantissa |= uint64(1) << float64FractionBits
		shift = exponent - 1
	}

	// The binary64 number is exactly mantissa*2^(shift-1074).
	// Its overlap-weighted numerator stays integral in these units.
	hi, lo := bits.Mul64(mantissa, uint64(overlap))
	// Split the exact power-of-two shift into a full-limb offset and a
	// within-limb shift. At most three limbs hold the aligned product.
	word := int(shift / accumulatorWordBits)
	amount := uint(shift % accumulatorWordBits)
	// The sign bit selects an unsigned magnitude. Positive and negative
	// contributions are not subtracted until the segment is complete.
	var target *[accumulatorWords]uint64
	if bits64>>(accumulatorWordBits-1) == 0 {
		target = &a.positive
	} else {
		target = &a.negative
	}
	// Build the three aligned limbs explicitly. The zero-shift branch
	// avoids an illegal uint64 shift by its full 64-bit width.
	var first, second, third uint64
	if amount == 0 {
		first, second = lo, hi
	} else {
		first = lo << amount
		second = (hi << amount) | (lo >> (accumulatorWordBits - amount))
		third = hi >> (accumulatorWordBits - amount)
	}
	addShifted(target, word, first, second, third)
}

// addShifted adds one unsigned shifted mantissa product to a sign total.
// A 53-bit mantissa times a <2^63 overlap, shifted by at most 63 intra-limb
// bits, occupies at most three consecutive uint64 limbs before propagation.
//
// Implementation:
//   - Stage 1: add the low, middle and high product limbs with carry chaining.
//   - Stage 2: propagate any remaining carry into subsequent high limbs.
//
// Inputs:
//   - dst: the caller-owned nonnegative magnitude of one sign only.
//   - at: validated starting limb offset; x/y/z: consecutive product limbs.
//
// Returns:
//   - The updated magnitude, by pointer; no allocation or arithmetic errors.
//
// Determinism:
//   - Carries propagate strictly from low to high significance.
//
// Complexity:
//   - O(accumulatorWords) worst case, O(1) extra space.
//
// Notes:
//   - n*w<=MaxInt64 implies a bound of <2^2161 on the complete sign total,
//     so no accepted input can leave an unpropagated carry above 2304 bits.
//
// AI-Hints:
//   - The bound applies to the SUM of integer overlaps, not a hypothetical
//     n*w copies of every input value; do not turn it into an O(n*w) loop.
func addShifted(dst *[accumulatorWords]uint64, at int, x, y, z uint64) {
	// The low three additions consume the shifted product itself. A carry
	// can then ripple through higher existing limbs but never beyond the
	// proven 2304-bit fixed bound for a validated complete segment.
	var carry uint64
	dst[at], carry = bits.Add64(dst[at], x, 0)
	dst[at+1], carry = bits.Add64(dst[at+1], y, carry)
	dst[at+2], carry = bits.Add64(dst[at+2], z, carry)
	for k := at + 3; carry != 0 && k < accumulatorWords; k++ {
		dst[k], carry = bits.Add64(dst[k], 0, carry)
	}
}

// compareWords orders two fixed-width unsigned integer magnitudes exactly.
// It determines the final result sign without storing either total as a
// signed integer or subtracting magnitudes in an unsafe order.
//
// Implementation:
//   - Scan most-significant limbs first; return at the first difference.
//   - Return zero only when all accumulatorWords limbs are identical.
//
// Inputs: a and b are initialized unsigned 2304-bit magnitudes.
// Returns: +1 when a>b, -1 when a<b, or 0 on exact equality.
// Determinism: unsigned lexicographic comparison from high to low limbs.
// Complexity: Time O(accumulatorWords), extra space O(1).
// AI-Hints: comparing sums as float64 would erase tiny signed differences.
func compareWords(a, b *[accumulatorWords]uint64) int {
	for k := accumulatorWords - 1; k >= 0; k-- {
		if a[k] > b[k] {
			return 1
		}
		if a[k] < b[k] {
			return -1
		}
	}
	return 0
}

// subtractWords computes the exact nonnegative difference of two magnitudes.
// The caller must establish a>=b using compareWords before subtraction.
//
// Implementation:
//   - Subtract each low-to-high limb with the previous limb's borrow.
//   - Return the full fixed-width difference without floating rounding.
//
// Inputs: a>=b, both initialized fixed-width unsigned magnitudes.
// Returns: detached 2304-bit value of a-b (all-zero when a==b).
// Determinism: low-to-high limb subtraction with explicit borrow propagation.
// Complexity: Time O(accumulatorWords), space O(accumulatorWords)=O(1).
// AI-Hints: reversing the order would wrap unsigned underflow into a huge
// false magnitude; sign selection must precede this helper.
func subtractWords(a, b *[accumulatorWords]uint64) (out [accumulatorWords]uint64) {
	var borrow uint64
	for k := range out {
		out[k], borrow = bits.Sub64(a[k], b[k], borrow)
	}
	return out
}

// divideWords performs exact long division of a nonnegative integer by n.
// It returns both quotient and remainder so final rounding can distinguish
// exact ties from values just above or below a half-ULP.
//
// Implementation:
//   - Traverse high-to-low limbs, carrying the previous remainder.
//   - Divide each 128-bit (remainder,limb) pair with bits.Div64.
//
// Inputs: a is unsigned fixed-point magnitude; n>0 is the checked length.
// Returns: integer quotient and remainder in [0,n), with no information loss.
// Determinism: high-to-low binary long division, one limb at a time.
// Complexity: Time O(accumulatorWords), space O(accumulatorWords)=O(1).
// Notes: bits.Div64 requires remainder<n; this is preserved by every step.
// AI-Hints: never convert the numerator to float64 before division: its
// intermediate magnitude may exceed finite binary64 even for a finite mean.
func divideWords(a *[accumulatorWords]uint64, n uint64) (out [accumulatorWords]uint64, remainder uint64) {
	// At each step, (remainder<<64)+a[k] is the next dividend fragment.
	// The previous step guaranteed remainder<n, satisfying bits.Div64's
	// precondition and preventing a division overflow or division by zero.
	for k := accumulatorWords - 1; k >= 0; k-- {
		out[k], remainder = bits.Div64(remainder, a[k], n)
	}
	return out, remainder
}

// wordBit reports a single guard or significand bit from a fixed integer.
// Callers supply an in-bounds nonnegative bit index obtained from bitLength.
// Inputs: a is an initialized unsigned magnitude; bit identifies one bit.
// Returns: true iff the selected bit is set.
// Determinism: direct uint64 bit mask; no rounding or iteration.
// Complexity: Time O(1), space O(1).
// AI-Hints: bounds are internal preconditions; do not expose this as an
// unchecked public bit-access API for arbitrary caller-provided positions.
func wordBit(a *[accumulatorWords]uint64, bit int) bool {
	return a[bit/accumulatorWordBits]&(uint64(1)<<uint(bit%accumulatorWordBits)) != 0
}

// bitLength identifies the highest occupied binary exponent of a magnitude.
// It returns zero for an all-zero integer, allowing the normal/subnormal
// rounding branch to be selected without floating logarithms.
// Implementation: scan limbs high-to-low and use bits.Len64 on the first nonzero.
// Inputs: a is an initialized unsigned fixed-width magnitude.
// Returns: highest set-bit index plus one, or zero.
// Determinism: most-significant nonzero limb wins, with no tie ambiguity.
// Complexity: Time O(accumulatorWords), space O(1).
// AI-Hints: a float64 log would overflow or underflow before classification.
func bitLength(a *[accumulatorWords]uint64) int {
	for k := accumulatorWords - 1; k >= 0; k-- {
		if a[k] != 0 {
			return k*accumulatorWordBits + bits.Len64(a[k])
		}
	}
	return 0
}

// lowerBitsNonzero extracts the sticky-bit condition for nearest-even rounding.
// It tests whether any integer bit below the discarded half-ULP position is
// set; a true sticky bit turns an apparent exact tie into a round-up case.
// Implementation:
//   - Scan complete low limbs below limit, then mask the partial final limb.
//   - Return false when no bit below limit is set (including limit==0).
//
// Inputs: a is unsigned; limit is an internally checked bit position.
// Returns: true iff at least one bit in [0,limit) is set.
// Determinism: low-to-high scan and an exact uint64 mask.
// Complexity: Time O(accumulatorWords), space O(1).
// AI-Hints: the remainder from divideWords is a SECOND sticky-bit source
// and must be considered alongside discarded quotient bits.
func lowerBitsNonzero(a *[accumulatorWords]uint64, limit int) bool {
	for k := 0; k < limit/accumulatorWordBits; k++ {
		if a[k] != 0 {
			return true
		}
	}
	if rem := uint(limit % accumulatorWordBits); rem != 0 {
		return a[limit/accumulatorWordBits]&((uint64(1)<<rem)-1) != 0
	}
	return false
}

// roundedFloat converts a nonnegative exact weighted mean to IEEE-754 binary64.
// Its input represents (integer/n)*2^-1074, and every discarded fraction is
// accounted for when selecting the nearest float64 (ties to even).
//
// Implementation:
//   - Stage 1: divide integer by n exactly, retaining quotient and remainder.
//   - Stage 2: for subnormal-scale quotients (at most 52 bits), round in units
//     of SmallestNonzeroFloat64 using the remainder and parity of the quotient.
//   - Stage 3: for normal-scale quotients retain 53 leading significand bits;
//     combine quotient guard/sticky bits with nonzero division remainder.
//   - Stage 4: normalize a carried significand and restore the binary exponent.
//   - Stage 5: reject a nonfinite final float instead of publishing +Inf/NaN.
//
// Inputs:
//   - integer: exact nonnegative fixed-point numerator, never a float sum.
//   - n: positive checked segment length, used as the averaging divisor.
//
// Returns:
//   - float64: nonnegative nearest-even rounded magnitude, including +0.
//   - error: ErrOverflow only when the final result cannot be finite.
//
// Determinism:
//   - One integer quotient/remainder, explicit guard/sticky/parity rules,
//     and a single final float construction independent of summation order.
//
// Complexity:
//   - Time O(accumulatorWords), space O(accumulatorWords)=O(1).
//
// Notes:
//   - Even when shift==0 in the *first normal binade*, the quotient may
//     carry a nonzero remainder large enough to round up one normal ULP.
//   - The caller applies the sign only after symmetric magnitude rounding.
//
// AI-Hints:
//   - Never discard divideWords remainder: it distinguishes a true halfway
//     value from one immediately above the tie, including normal/subnormal
//     crossover cases that a quotient-only implementation rounds incorrectly.
//   - Do not use an arbitrary epsilon in place of the IEEE ties-to-even law.
func roundedFloat(integer *[accumulatorWords]uint64, n uint64) (float64, error) {
	// Decompose the exact rational mean into an integer part measured in
	// 2^-1074 units and a remainder strictly smaller than n.
	quotient, remainder := divideWords(integer, n)
	length := bitLength(&quotient)
	if length <= float64FractionBits {
		// The quotient fits within the subnormal significand domain.
		// Any fractional remainder is less than a single 2^-1074 unit.
		m := quotient[0]
		// Decide whether the discarded fraction exceeds one half of the
		// smallest positive subnormal. With even n, an exact half chooses
		// the even low bit of the candidate integer significand. Compare to
		// n/2 rather than calculating 2*remainder, which could overflow.
		half := n / 2
		if remainder > half || (n%2 == 0 && remainder == half && m&1 == 1) {
			m++
		}
		return math.Float64frombits(m), nil
	}

	// Extract the leading 53 significant integer bits from the quotient.
	// All lower quotient bits and the division remainder become rounding
	// evidence: the leading guard bit identifies half an ULP and the rest
	// form the sticky bit.
	shift := length - float64SignificandBits
	index, offset := shift/accumulatorWordBits, uint(shift%accumulatorWordBits)
	mantissa := quotient[index] >> offset
	if offset != 0 && index+1 < accumulatorWords {
		mantissa |= quotient[index+1] << (accumulatorWordBits - offset)
	}
	mantissa &= (uint64(1) << float64SignificandBits) - 1
	if shift == 0 {
		// At the first normal binade, a full 53-bit quotient may still have
		// a fractional remainder. Ignoring it rounds values between adjacent
		// normal floats downward, even when the remainder is above half an ULP.
		// Comparing against n/2 avoids overflowing 2*remainder.
		half := n / 2
		if remainder > half || (n%2 == 0 && remainder == half && mantissa&1 != 0) {
			mantissa++
		}
	} else {
		// Beyond the first normal binade, the quotient's discarded bits
		// include the half-ULP. The division remainder breaks exact ties.
		halfBit := wordBit(&quotient, shift-1)
		belowHalf := lowerBitsNonzero(&quotient, shift-1) || remainder != 0
		if halfBit && (belowHalf || mantissa&1 != 0) {
			mantissa++
		}
	}
	// Rounding may carry a 53-bit all-ones significand into the next
	// binade. Normalize by shifting right and increasing the bit length.
	if mantissa == uint64(1)<<float64SignificandBits {
		mantissa >>= 1
		length++
	}
	// The significand has at most 53 exact bits and is therefore converted
	// to float64 without rounding. Ldexp then restores its binary exponent;
	// the only permitted arithmetic failure is a nonfinite final result.
	out := math.Ldexp(float64(mantissa), length-float64SignificandBits-float64SubnormalShift)
	if math.IsInf(out, 0) || math.IsNaN(out) {
		return 0, ErrOverflow
	}
	return out, nil
}

// finish publishes the one correctly-rounded mean for a completed segment.
// It cancels all positive and negative weighted contributions exactly,
// chooses the signed result, and enforces the contributing convex hull.
//
// Implementation:
//   - Stage 1: reject an unstarted cell (violated overlap invariant).
//   - Stage 2: compare unsigned sign totals; publish exact +0 on cancellation.
//   - Stage 3: subtract smaller from larger and divide/round the magnitude.
//   - Stage 4: apply sign and verify that the result lies in [minimum,maximum].
//
// Inputs:
//   - n: validated positive original source length in common integer units.
//   - receiver: positive overlap contributions to exactly this output cell.
//
// Returns:
//   - float64: finite nearest-even rounded weighted average.
//   - error: ErrOverflow for impossible empty state, nonfinite result or
//     violated numeric hull; an error never publishes a partial Result.
//
// Determinism:
//   - Unsigned high-to-low sign comparison, low-to-high difference, nearest
//     even final rounding and an exact contributor minimum/maximum check.
//
// Complexity:
//   - Time O(accumulatorWords), space O(accumulatorWords)=O(1).
//
// Notes:
//   - A true zero after signed cancellation returns +0; the w=n identity
//     path separately preserves negative zero because it copies input bits.
//
// AI-Hints:
//   - Do not clamp an out-of-range value into the convex hull: that would
//     hide a broken arithmetic invariant instead of reporting ErrOverflow.
//   - Never average the two sign totals as float64 before subtracting them.
func (a *weightedAccumulator) finish(n int64) (float64, error) {
	// A completed segment must have consumed positive overlap mass. Empty
	// completion is an internal geometry violation, not a zero-valued mean.
	if !a.started {
		return 0, ErrOverflow
	}
	// The sign comparison happens before subtraction, avoiding unsigned
	// wrap. Exact cancellation returns +0, independent of input summation
	// order; identity partitions preserve their original zero sign by copy.
	order := compareWords(&a.positive, &a.negative)
	if order == 0 {
		return 0, nil
	}
	var magnitude [accumulatorWords]uint64
	if order > 0 {
		magnitude = subtractWords(&a.positive, &a.negative)
	} else {
		magnitude = subtractWords(&a.negative, &a.positive)
	}
	// Restore the mean from the exact signed mass only after canceling
	// contributions. The validator guarantees the divisor n is positive.
	value, err := roundedFloat(&magnitude, uint64(n))
	if err != nil {
		return 0, err
	}
	if order < 0 {
		value = -value
	}
	// Exact convex weights yield a mean within the contributing hull.
	// Rounding-to-nearest preserves that interval because its endpoints
	// are representable float64 values. Never clamp a failed postcondition.
	if value < a.minimum || value > a.maximum {
		return 0, ErrOverflow
	}
	return value, nil
}
