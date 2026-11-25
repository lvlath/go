// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mfi computes the correctly rounded typical price of three finite positive prices.
// This file implements exact fixed-width binary arithmetic without retaining heap state.
package mfi

import (
	"math"
	"math/bits"
)

const (
	// A binary64 normal has 52 explicit fraction bits and one implicit leading bit.
	typicalFractionBits = 52
	typicalPrecision    = typicalFractionBits + 1
	typicalHiddenBit    = uint64(1) << typicalFractionBits
	typicalFractionMask = typicalHiddenBit - 1
	typicalExponentMask = uint64(0x7ff) << typicalFractionBits

	// One integer unit is the smallest positive subnormal, 2^-1074.
	typicalUnitExponent = -1074
	typicalShiftBias    = 1022
	typicalWordBits     = 64
	typicalDivisor      = 3

	// Three finite binary64 terms fit in at most 2100 integer bits.
	// A 33-word accumulator therefore has spare headroom for every carry.
	typicalAccumulatorWords = 33

	// Three shifted 53-bit significands fit in a single uint64 for span <= 9:
	// 3*(2^53-1)*2^9 < 2^64. No high bits are discarded by this fast path.
	typicalCompactSpan = typicalWordBits - typicalPrecision - 2
)

// typicalTerm decodes a validated positive binary64 as an exact integer term.
// The returned pair represents significand * 2^(shift-1074), including subnormals.
// Implementation:
//   - Stage 1: extract the stored fraction and determine the normal/subnormal class.
//   - Stage 2: restore the implicit leading bit only for normal operands.
//   - Stage 3: derive the nonnegative integer-unit shift from the exact binary exponent.
//
// Behavior highlights:
//   - Subnormal operands keep their original fraction at shift zero; no scaling rounds them.
//
// Inputs:
//   - value: finite, strictly positive binary64, already admitted by validateBar.
//
// Returns:
//   - significand: exact integer of at most 53 bits; shift: integer in [0,2045].
//
// Determinism:
//   - IEEE-754 field decoding; no floating-point addition or division.
//
// Complexity:
//   - Time O(1), Space O(1), no allocation.
//
// Notes:
//   - Ilogb supplies a native signed exponent without narrowing an unsigned wire value.
//
// AI-Hints:
//   - Never normalize a subnormal by a rounded division before extracting its bits.
func typicalTerm(value float64) (significand uint64, shift int) {
	encoded := math.Float64bits(value)
	significand = encoded & typicalFractionMask
	if encoded&typicalExponentMask == 0 {
		return significand, 0
	}
	return significand | typicalHiddenBit, math.Ilogb(value) + typicalShiftBias
}

// canonicalTypical computes RN64((h+l+c)/3) from the original binary64 operands.
// The result is independent of operand order and candle shape for equal exact sums.
// Implementation:
//   - Stage 1: return an identical-price candle directly, or decode three exact terms.
//   - Stage 2: align at the smallest exponent; use a compact integer sum when it fits.
//   - Stage 3: otherwise accumulate all bits in a fixed stack array and round the quotient once.
//
// Behavior highlights:
//   - No floating intermediate sum can overflow, and no low-order input bit is discarded.
//   - The mean of three positive finite prices lies in their closed convex hull.
//
// Inputs:
//   - h,l,c: already validated positive finite High, Low, Close with l<=c<=h.
//
// Returns:
//   - float64: nearest-even rounding of the exact mean, positive and finite.
//
// Errors:
//   - None under the validated OHLC domain; the mean cannot overflow or round to zero.
//
// Determinism:
//   - Both compact and wide paths compute the same integer sum and nearest-even rounding.
//
// Complexity:
//   - Time O(1), Space O(1); at most 33 uint64 limbs on the wide path, no heap allocation.
//
// Notes:
//   - This is exact binary-input averaging, not decimal-price arithmetic.
//
// AI-Hints:
//   - Do not divide operands separately or use an epsilon to classify neighboring prices.
//   - Preserve the final integer remainder and sticky bits through the rounding step.
func canonicalTypical(h, l, c float64) float64 {
	// Geometry guarantees c==h when the two extremes coincide.
	if h == l {
		return h
	}

	// Decode before alignment, so even the smallest subnormal is represented exactly.
	hm, hs := typicalTerm(h)
	lm, ls := typicalTerm(l)
	cm, cs := typicalTerm(c)
	base := min(hs, ls, cs)
	span := max(hs, ls, cs) - base

	// The compact path covers ordinary market candles with a narrow exponent span.
	if span <= typicalCompactSpan {
		sum := (hm << (hs - base)) + (lm << (ls - base)) + (cm << (cs - base))
		return compactTypical(sum, base)
	}

	// The wide path preserves every bit even across the full binary64 exponent range.
	var words [typicalAccumulatorWords]uint64
	addTypicalTerm(&words, hm, hs-base)
	addTypicalTerm(&words, lm, ls-base)
	addTypicalTerm(&words, cm, cs-base)
	return wideTypical(&words, base)
}

// compactTypical divides a proven nonoverflowing integer sum by three.
// Quotient and remainder retain the exact rounding information at the input quantum.
// Implementation:
//   - Stage 1: divide once into quotient and remainder, then determine the binary64 drop count.
//   - Stage 2: round at unit precision or at a guard/sticky/odd-significand boundary.
//   - Stage 3: scale the rounded, exactly representable significand by a power of two.
//
// Inputs:
//   - sum: exact three-term integer sum; base: shared integer-unit exponent offset.
//
// Returns:
//   - float64: correctly rounded mean; no additional information is lost in scaling.
//
// Determinism:
//   - Nearest-even, including an exact halfway remainder after binary truncation.
//
// Complexity:
//   - Time O(1), Space O(1), no allocation.
//
// Notes:
//   - Division by three alone has no half-integer ties; its remainder is 0, 1 or 2.
//
// AI-Hints:
//   - A nonzero division remainder is sticky whenever at least one quotient bit is dropped.
func compactTypical(sum uint64, base int) float64 {
	quotient, remainder := sum/typicalDivisor, sum%typicalDivisor
	drop := max(0, bits.Len64(quotient)-typicalPrecision)
	significand := quotient >> drop
	if drop == 0 {
		if remainder == typicalDivisor-1 {
			significand++
		}
	} else {
		guard := uint64(1) << (drop - 1)
		if quotient&guard != 0 && (quotient&(guard-1) != 0 || remainder != 0 || significand&1 != 0) {
			significand++
		}
	}
	// A carry to 2^53 is itself exactly representable; Ldexp handles renormalization.
	return math.Ldexp(float64(significand), base+drop+typicalUnitExponent)
}

// addTypicalTerm adds a shifted significand to the exact little-endian limb sum.
// Carries are propagated using integer arithmetic, not floating-point expansions.
// Implementation:
//   - Stage 1: identify the two limb positions spanned by the shifted term.
//   - Stage 2: add the low and high portions with carry.
//   - Stage 3: propagate any remaining carry through higher limbs.
//
// Inputs:
//   - words: private 33-word accumulator; significand: <=53 bits; shift: [0,2045].
//
// Returns:
//   - The accumulator is updated in place; no input price or persistent Stream state is touched.
//
// Determinism:
//   - Exact unsigned integer addition; the result is independent of insertion order.
//
// Complexity:
//   - Time O(33) bounded, Space O(1), no allocation.
//
// Notes:
//   - The total of at most three terms is <2^2100, so no carry can leave the array.
//
// AI-Hints:
//   - Do not discard carry propagation even when only one operand crosses a limb boundary.
func addTypicalTerm(words *[typicalAccumulatorWords]uint64, significand uint64, shift int) {
	index, offset := shift/typicalWordBits, shift%typicalWordBits
	low, high := significand<<offset, significand>>(typicalWordBits-offset)
	var carry uint64
	words[index], carry = bits.Add64(words[index], low, 0)
	index++
	words[index], carry = bits.Add64(words[index], high, carry)
	for carry != 0 {
		index++
		words[index], carry = bits.Add64(words[index], 0, carry)
	}
}

// wideTypical performs exact long division followed by one binary64 rounding.
// The same mathematical result as compactTypical is obtained for wide exponent spans.
// Implementation:
//   - Stage 1: find the most significant occupied word and divide the sum by three in place.
//   - Stage 2: identify the target significand and retain guard, sticky and remainder information.
//   - Stage 3: round once and scale to the representable binary64 value.
//
// Inputs:
//   - words: private exact sum of three validated terms; base: shared exponent offset.
//
// Returns:
//   - float64: positive finite nearest-even mean; words is disposable scratch after the call.
//
// Determinism:
//   - Long division runs from most to least significant word; the final tie rule is nearest-even.
//
// Complexity:
//   - Time O(33) bounded, Space O(1) beyond caller scratch, no allocation.
//
// Notes:
//   - Div64's high argument is always the previous remainder, strictly less than three.
//   - This makes its division-overflow condition impossible in the validated domain.
//
// AI-Hints:
//   - Never round the quotient before checking all discarded lower words and the remainder.
func wideTypical(words *[typicalAccumulatorWords]uint64, base int) float64 {
	// Skip leading zero limbs; the sum is nonzero for three positive inputs.
	top := typicalAccumulatorWords - 1
	for words[top] == 0 {
		top--
	}

	// Carry the exact division remainder into each less significant word.
	var remainder uint64
	for i := top; i >= 0; i-- {
		words[i], remainder = bits.Div64(remainder, words[i], typicalDivisor)
	}
	if words[top] == 0 {
		// Dividing by three can remove at most two leading bits, hence at most one limb.
		top--
	}

	// Extract the retained integer significand without rounding or shifting away a sticky bit.
	drop := max(0, top*typicalWordBits+bits.Len64(words[top])-typicalPrecision)
	index, offset := drop/typicalWordBits, drop%typicalWordBits
	significand := words[index] >> offset
	if index < top {
		significand |= words[index+1] << (typicalWordBits - offset)
	}

	// The wide path necessarily has drop>0: a span >9 gives a quotient wider than 53 bits
	// for normal operands. Subnormal operands may only occur at the smallest shift.
	guardIndex, guardOffset := (drop-1)/typicalWordBits, (drop-1)%typicalWordBits
	guard := uint64(1) << guardOffset
	if words[guardIndex]&guard != 0 {
		sticky := remainder != 0 || words[guardIndex]&(guard-1) != 0
		for i := 0; i < guardIndex && !sticky; i++ {
			sticky = words[i] != 0
		}
		if sticky || significand&1 != 0 {
			significand++
		}
	}
	return math.Ldexp(float64(significand), base+drop+typicalUnitExponent)
}
