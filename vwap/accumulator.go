// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package vwap defines fixed-width exact accumulation and correctly rounded division.
// This file explains capacity proofs, signed cancellation, and binary64 lattice selection.
package vwap

import (
	"math"
	"math/bits"
)

const (
	// limbCount bounds every live magnitude and rounding temporary by 4288 bits.
	// Count is uint64: the weighted magnitude needs at most 4260 of those bits.
	limbCount = 67
	// limbBits is the width of the unsigned words used by math/bits arithmetic.
	limbBits = 64
	// fractionBits excludes the implicit leading bit of a normal binary64 value.
	fractionBits = 52
	// significandBits includes that leading bit; all such integers are exact in binary64.
	significandBits = fractionBits + 1
	// subnormalScale converts a binary64 value into integer multiples of 2^-1074.
	subnormalScale = 1074
	// fractionMask selects the explicitly stored significand bits without the sign.
	fractionMask = uint64(1<<fractionBits - 1)
	// implicitBit restores the leading one of a finite normal value, never a subnormal.
	implicitBit = uint64(1 << fractionBits)
	// exponentFieldMask distinguishes normal values from signed zeros and subnormals.
	exponentFieldMask = uint64(0x7ff) << fractionBits
	// signBit is separate from the magnitude, including the sign of an input zero.
	signBit = uint64(1) << (limbBits - 1)
	// maxMantissa is the largest finite binary64 significand, before its unit shift.
	maxMantissa = uint64(1<<significandBits - 1)
	// maxUnitShift places MaxFloat64's significand in the 2^-1074 integer lattice.
	maxUnitShift = 2045
)

// natural stores a normalized nonnegative integer in little-endian fixed words.
// A zero magnitude has used=0; every word at or above used is zero.
//
// Implementation:
//   - Stage 1: decomposition or checked parsing creates a bounded magnitude.
//   - Stage 2: exact carry/borrow operations preserve low-to-high word significance.
//   - Stage 3: normalization removes zero high words without changing the integer.
//
// Behavior highlights:
//   - All fields have value semantics; copying cannot alias another magnitude.
//
// Inputs:
//   - Private arithmetic callers must prove every result fits 4288 bits.
//
// Returns:
//   - The represented integer is sum(words[i]*2^(64*i), i<used).
//
// Errors:
//   - No error carrier; public boundaries establish every arithmetic precondition.
//
// Determinism:
//   - Addition/subtraction run low-to-high; ordering comparisons run high-to-low.
//
// Complexity:
//   - Storage O(L), L=67 fixed words; scalar operations inspect only significant words.
//
// Notes:
//   - Volume needs at most 2162 bits; weighted magnitude needs at most 4260 bits.
//
// AI-Hints:
//   - Do not replace fixed words with history-dependent storage or rounded floating sums.
type natural struct {
	// words stores low significance first; unused high words must remain zero.
	words [limbCount]uint64
	// used is the significant prefix length, not a capacity or a historical count.
	used int
}

// normalize removes zero high limbs without modifying the represented integer.
// The used length becomes zero exactly when the complete magnitude is zero.
//
// Implementation:
//   - Stage 1: inspect the highest currently used word.
//   - Stage 2: discard high zero words until a nonzero word or empty prefix remains.
//   - Stage 3: retain the fixed backing array and all significant words.
//
// Behavior highlights:
//   - Words above used remain zero; the loop performs no allocation.
//
// Inputs:
//   - a: mutable magnitude with 0<=used<=limbCount and zero words above used.
//
// Returns:
//   - No value; a.used is the minimal significant prefix length.
//
// Errors:
//   - No error return; the documented private preconditions are established by the caller.
//
// Determinism:
//   - Input order and integer comparisons determine the result; no shared mutable state is read.
//
// Complexity:
//   - Time O(L), Space O(1), L<=67; only removed high words are inspected.
//
// Notes:
//   - Arithmetic operations, not this method, clear words that become zero.
//
// AI-Hints:
//   - Never decrement used across a nonzero limb or clear a still-significant low word.
func (a *natural) normalize() {
	// Trim only high zero words; lower words may contain the entire surviving precision.
	for a.used > 0 && a.words[a.used-1] == 0 {
		a.used--
	}
}

// bitLen reports the exact binary length of a nonzero magnitude.
// The length selects integer bounds and the binary64 rounding lattice without logarithms.
//
// Implementation:
//   - Stage 1: use the normalized nonzero word count.
//   - Stage 2: count bits in the highest significant word.
//   - Stage 3: add the width of all preceding words.
//
// Behavior highlights:
//   - No floating estimate or rounding is involved.
//
// Inputs:
//   - a: normalized magnitude with used>0.
//
// Returns:
//   - int: floor(log2(a))+1, in [1,4288].
//
// Errors:
//   - No error return; the documented private preconditions are established by the caller.
//
// Determinism:
//   - Input order and integer comparisons determine the result; no shared mutable state is read.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Zero must be classified before calling this private operation.
//
// AI-Hints:
//   - Do not call on zero; volume/weighted readiness and decimal grammar already establish nonzero input.
func (a *natural) bitLen() int {
	// The last word is nonzero by the caller precondition, so its bit count is meaningful.
	return (a.used-1)*limbBits + bits.Len64(a.words[a.used-1])
}

// compare orders two normalized unsigned magnitudes exactly.
// It is used for signed cancellation, magnitude admission, and quotient correction.
//
// Implementation:
//   - Stage 1: compare significant word counts.
//   - Stage 2: when lengths match, inspect words from most to least significant.
//   - Stage 3: return at the first difference, or equality after the scan.
//
// Behavior highlights:
//   - An empty normalized magnitude compares equal to another empty magnitude.
//
// Inputs:
//   - a, b: read-only normalized values; no storage is retained.
//
// Returns:
//   - int: -1, 0, or 1 according to the mathematical order.
//
// Errors:
//   - No error return; the documented private preconditions are established by the caller.
//
// Determinism:
//   - Input order and integer comparisons determine the result; no shared mutable state is read.
//
// Complexity:
//   - Time O(L), Space O(1), L<=67.
//
// Notes:
//   - Leading zero words must not be included in used.
//
// AI-Hints:
//   - Do not compare little-endian storage lexicographically from the low word.
func (a *natural) compare(b *natural) int {
	// A normalized longer magnitude is larger before any word-level comparison.
	if a.used < b.used {
		return -1
	}
	if a.used > b.used {
		return 1
	}
	// The first differing high word determines the order, irrespective of lower bits.
	for i := a.used - 1; i >= 0; i-- {
		if a.words[i] < b.words[i] {
			return -1
		}
		if a.words[i] > b.words[i] {
			return 1
		}
	}
	return 0
}

// add adds a nonnegative magnitude with exact carry propagation.
// The caller proves the sum fits the fixed accumulator before invoking this operation.
//
// Implementation:
//   - Stage 1: select the larger significant word count.
//   - Stage 2: add low-to-high with bits.Add64 and an explicit carry.
//   - Stage 3: append a final carry when it is nonzero.
//
// Behavior highlights:
//   - No low-order bit is discarded, even when the addend is much smaller than the sum.
//
// Inputs:
//   - a: mutable normalized accumulator.
//   - b: read-only normalized addend; a+b must fit limbCount words.
//
// Returns:
//   - No value; a becomes the exact sum.
//
// Errors:
//   - No error return; the documented private preconditions are established by the caller.
//
// Determinism:
//   - Input order and integer comparisons determine the result; no shared mutable state is read.
//
// Complexity:
//   - Time O(L), Space O(1), L<=67.
//
// Notes:
//   - Count-limited live volume and weighted sums cannot overflow the fixed capacity.
//
// AI-Hints:
//   - Do not use this unchecked primitive to parse unbounded input; parsing has its own per-digit bound.
func (a *natural) add(b *natural) {
	// Missing high words are zero under the representation invariant.
	n := max(a.used, b.used)
	var carry uint64
	for i := 0; i < n; i++ {
		// Add the incoming carry together with both complete limbs.
		a.words[i], carry = bits.Add64(a.words[i], b.words[i], carry)
	}
	a.used = n
	// A final carry extends the normalized prefix within the proved fixed capacity.
	if carry != 0 {
		a.words[n] = carry
		a.used++
	}
}

// subtract subtracts a smaller magnitude and normalizes exact cancellation.
// Unsigned subtraction is valid because the caller has already established a>=b.
//
// Implementation:
//   - Stage 1: start with zero borrow.
//   - Stage 2: subtract low-to-high through the significant prefix of a.
//   - Stage 3: remove zero leading words created by the difference.
//
// Behavior highlights:
//   - Exact cancellation produces used=0; sign canonicalization belongs to accumulator.include.
//
// Inputs:
//   - a: mutable normalized minuend.
//   - b: normalized subtrahend with b<=a.
//
// Returns:
//   - No value; a becomes the nonnegative exact difference.
//
// Errors:
//   - No error return; the documented private preconditions are established by the caller.
//
// Determinism:
//   - Input order and integer comparisons determine the result; no shared mutable state is read.
//
// Complexity:
//   - Time O(L), Space O(1), L<=67.
//
// Notes:
//   - Trailing array words are already zero and remain available as implicit zero limbs.
//
// AI-Hints:
//   - Establish the magnitude order first; do not replace subtraction by wrapped unsigned arithmetic.
func (a *natural) subtract(b *natural) {
	// Magnitude ordering guarantees that the final outgoing borrow is zero.
	var borrow uint64
	for i := 0; i < a.used; i++ {
		a.words[i], borrow = bits.Sub64(a.words[i], b.words[i], borrow)
	}
	a.normalize()
}

// multiplyWord multiplies a magnitude by one unsigned word into detached storage.
// Full-width integer products preserve every bit needed by bounds checks and division.
//
// Implementation:
//   - Stage 1: zero-initialize a fixed-size result and carry.
//   - Stage 2: combine both halves of each bits.Mul64 product with the previous carry.
//   - Stage 3: append a final carry and normalize a possible zero product.
//
// Behavior highlights:
//   - The high half is retained instead of allowing multiplication to wrap.
//
// Inputs:
//   - a: read-only normalized magnitude.
//   - word: scalar multiplier; the complete product must fit 4288 bits.
//
// Returns:
//   - natural: owning value storage, not an alias of a.
//
// Errors:
//   - No error return; the documented private preconditions are established by the caller.
//
// Determinism:
//   - Input order and integer comparisons determine the result; no shared mutable state is read.
//
// Complexity:
//   - Time O(L), Space O(L), L<=67.
//
// Notes:
//   - The per-word carry is at most word-1, so hi+carry-bit cannot overflow.
//
// AI-Hints:
//   - Capacity must bound the complete product, not just each individual limb multiplication.
func (a *natural) multiplyWord(word uint64) natural {
	var out natural
	var carry uint64
	for i := 0; i < a.used; i++ {
		// Retain the full product before adding the preceding word's high part.
		hi, lo := bits.Mul64(a.words[i], word)
		var c uint64
		out.words[i], c = bits.Add64(lo, carry, 0)
		// The product/carry bound guarantees this high-word addition is representable.
		carry = hi + c
	}
	out.used = a.used
	if carry != 0 {
		out.words[out.used] = carry
		out.used++
	}
	out.normalize()
	return out
}

// shifted returns an exact nonnegative binary shift in detached fixed storage.
// It aligns lattice units and divisor boundaries without rounding away small terms.
//
// Implementation:
//   - Stage 1: return canonical zero for an empty magnitude.
//   - Stage 2: split the shift into whole-word and intra-word offsets.
//   - Stage 3: place both portions of every source word and derive the normalized length.
//
// Behavior highlights:
//   - A shift by zero is an exact value copy; Go shifts by 64 produce zero high spill.
//
// Inputs:
//   - a: read-only normalized magnitude.
//   - shift: nonnegative; bitLen(a)+shift must fit the fixed array.
//
// Returns:
//   - natural: a*2^shift with independent words.
//
// Errors:
//   - No error return; the documented private preconditions are established by the caller.
//
// Determinism:
//   - Input order and integer comparisons determine the result; no shared mutable state is read.
//
// Complexity:
//   - Time O(L), Space O(L), L<=67.
//
// Notes:
//   - Live and rounding callers keep shifted magnitudes within the proved capacity.
//
// AI-Hints:
//   - Do not mask the intra-word spill shift modulo 64: part=0 must not duplicate a limb.
func (a *natural) shifted(shift int) natural {
	// Zero has no leading word to shift and remains normalized without indexing.
	if a.used == 0 {
		return natural{}
	}
	// Split alignment into an array offset and disjoint within-word bit ranges.
	whole, part := shift/limbBits, shift%limbBits
	var out natural
	for i := 0; i < a.used; i++ {
		out.words[i+whole] |= a.words[i] << part
		// With part=0 the 64-bit right shift yields zero, so no duplicate spill appears.
		hi := a.words[i] >> (limbBits - part)
		if hi != 0 {
			out.words[i+whole+1] |= hi
		}
	}
	// The exact nonzero bit length gives the prefix length without a rescan.
	out.used = (a.bitLen() + shift + limbBits - 1) / limbBits
	return out
}

// term places a two-word product on the exact integer lattice.
// One binary64 significand or a product of two significands occupies at most three shifted words.
//
// Implementation:
//   - Stage 1: split the supplied shift into whole-word and bit components.
//   - Stage 2: place low, middle, and high spill words in a zero-initialized magnitude.
//   - Stage 3: normalize so a zero significand becomes the canonical empty magnitude.
//
// Behavior highlights:
//   - There is no float64 multiplication or temporary unbounded integer.
//
// Inputs:
//   - hi, lo: unsigned 128-bit magnitude hi*2^64+lo.
//   - shift: in [0,4090] for all admitted callers.
//
// Returns:
//   - natural: exact (hi*2^64+lo)*2^shift.
//
// Errors:
//   - No error return; the documented private preconditions are established by the caller.
//
// Determinism:
//   - Input order and integer comparisons determine the result; no shared mutable state is read.
//
// Complexity:
//   - Time O(L) including fixed result initialization, Space O(L); only three words receive data.
//
// Notes:
//   - The largest addressed index is floor(4090/64)+2=65, below limbCount=67.
//
// AI-Hints:
//   - Keep both halves of bits.Mul64; lo alone silently loses high-value products.
func term(hi, lo uint64, shift int) natural {
	// Split alignment into an array offset and disjoint within-word bit ranges.
	whole, part := shift/limbBits, shift%limbBits
	var out natural
	// The two-word product can span three destination words after bit alignment.
	out.words[whole] = lo << part
	out.words[whole+1] = hi<<part | lo>>(limbBits-part)
	out.words[whole+2] = hi >> (limbBits - part)
	out.used = whole + 3
	out.normalize()
	return out
}

// binaryParts decomposes one finite binary64 into exact integer-unit components.
// The result satisfies value = sign * mantissa * 2^shift * 2^-1074.
//
// Implementation:
//   - Stage 1: extract the stored fraction and sign from the binary64 encoding.
//   - Stage 2: return shift zero directly for subnormals and signed zeros.
//   - Stage 3: for normals, use Frexp's int exponent and restore the implicit leading bit.
//
// Behavior highlights:
//   - No potentially narrowing unsigned-to-int conversion is used.
//   - Frexp decomposes an exact power of two; its fractional result is not used in arithmetic.
//
// Inputs:
//   - value: finite float64, admitted by the public domain validators.
//
// Returns:
//   - uint64: significand, at most 53 bits.
//   - int: lattice shift in [0,2045].
//   - bool: original sign bit; zero magnitude still remains zero.
//
// Errors:
//   - No error return; the documented private preconditions are established by the caller.
//
// Determinism:
//   - Input order and integer comparisons determine the result; no shared mutable state is read.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Normal Frexp exponents are in [-1021,1024], giving shift=exponent+1074-53.
//
// AI-Hints:
//   - Subnormals have no implicit leading one; do not apply the normal shift formula to them.
//   - Never replace this exact decomposition with decimal formatting or a float price-volume product.
func binaryParts(value float64) (uint64, int, bool) {
	// Keep the original fraction bits: no decimal conversion or rounding occurs.
	word := math.Float64bits(value)
	mantissa := word & fractionMask
	negative := word&signBit != 0
	// Subnormals and either signed zero already use the 2^-1074 lattice directly.
	if word&exponentFieldMask == 0 {
		return mantissa, 0, negative
	}
	// Frexp returns an int exponent without a narrowing uint64-to-int conversion.
	// For a normal value: value = sign * significand * 2^(exponent-53).
	// Therefore its integer-unit shift is exponent-53+1074, in [0,2045].
	_, exponent := math.Frexp(value)
	return mantissa | implicitBit, exponent + subnormalScale - significandBits, negative
}

// accumulator holds the exact volume and signed weighted-product sums of one segment.
// The denominator uses 2^-1074 units; the signed numerator uses 2^-2148 units.
//
// Implementation:
//   - Stage 1: decompose admitted binary64 inputs into integer components.
//   - Stage 2: add exact volume and full signed product magnitudes.
//   - Stage 3: round the final rational only when a public Value is requested.
//
// Behavior highlights:
//   - Huge finite products and destructive signed cancellation remain exact.
//
// Inputs:
//   - Finite prices, positive volumes, and a segment count bounded by uint64.
//
// Returns:
//   - Fixed exact state; no past observations or borrowed slices are retained.
//
// Errors:
//   - None internally; public validation precedes mutation of a candidate state.
//
// Determinism:
//   - Exact zero numerator is stored with negative=false.
//
// Complexity:
//   - O(L) state, L=67; no dependence on the number of accepted observations.
//
// Notes:
//   - Nonnegative volume gives |weighted|<=MaxFloat64Units*volume.
//
// AI-Hints:
//   - The separate sign belongs only to weighted, never to volume.
type accumulator struct {
	// volume is nonnegative and includes all accepted strictly positive weights.
	volume natural
	// weighted is the absolute exact numerator, with units twice the binary64 scale.
	weighted natural
	// negative applies only to a nonzero weighted magnitude; cancellation clears it.
	negative bool
}

// include adds one admitted positive-volume observation to both exact sums.
// The weighted magnitude and its separate sign support cancellation without losing low bits.
//
// Implementation:
//   - Stage 1: decode and add the volume in 2^-1074 units.
//   - Stage 2: decode price and construct its full integer product with volume.
//   - Stage 3: add same-sign magnitudes or subtract opposite-sign magnitudes, canonicalizing zero.
//
// Behavior highlights:
//   - A zero price still contributes volume but adds nothing to the weighted numerator.
//
// Inputs:
//   - a: mutable exact accumulation state whose next count fits uint64.
//   - price: finite signed price.
//   - volume: finite and strictly positive; zero volume is handled by Stream.accept.
//
// Returns:
//   - No value; exact volume and signed weighted totals are updated together.
//
// Errors:
//   - No error return; the documented private preconditions are established by the caller.
//
// Determinism:
//   - Input order and integer comparisons determine the result; no shared mutable state is read.
//
// Complexity:
//   - Time O(L), Space O(L), L<=67, independent of accepted history length.
//
// Notes:
//   - The caller uses a detached candidate for transactional Push; Compute owns its local stream.
//
// AI-Hints:
//   - When an opposite-sign product is larger, subtract the old magnitude from it and take its sign.
//   - Clear the negative flag on exact cancellation so the exact zero numerator remains canonical.
func (a *accumulator) include(price, volume float64) {
	// Add denominator units first; the caller has already excluded zero volume.
	vm, vs, _ := binaryParts(volume)
	volumeTerm := term(0, vm, vs)
	a.volume.add(&volumeTerm)
	pm, ps, negative := binaryParts(price)
	// A zero-priced execution increases total weight but cannot change the numerator.
	if pm == 0 {
		return
	}
	// Multiplying integer significands preserves the entire 106-bit exact product.
	hi, lo := bits.Mul64(pm, vm)
	product := term(hi, lo, ps+vs)
	// Equal signs add magnitudes directly; no cancellation or sign choice is needed.
	if a.negative == negative {
		a.weighted.add(&product)
		return
	}
	// With opposite signs the larger magnitude owns the sign of the difference.
	if a.weighted.compare(&product) >= 0 {
		a.weighted.subtract(&product)
		// Exact cancellation has one canonical sign, unlike rounded negative underflow.
		// Positive volume with an exactly zero numerator is a valid zero measurement.
		if a.weighted.used == 0 {
			a.negative = false
		}
		return
	}
	// The new term dominates: subtract the old magnitude and replace its sign.
	product.subtract(&a.weighted)
	a.weighted = product
	a.negative = negative
}

// word reads a magnitude word with zero extension outside the significant prefix.
// Virtual normalization uses the same zero-extension rule for absent low and high words.
//
// Implementation:
//   - Stage 1: reject a negative index or one at/above used.
//   - Stage 2: return zero for those virtual positions.
//   - Stage 3: otherwise read the indexed word without mutation.
//
// Behavior highlights:
//   - Negative one is a deliberate query when constructing the lowest normalized word.
//
// Inputs:
//   - a: read-only normalized magnitude.
//   - index: any int; out-of-prefix positions represent zero.
//
// Returns:
//   - uint64: the selected word or an implicit zero.
//
// Errors:
//   - No error return; the documented private preconditions are established by the caller.
//
// Determinism:
//   - Input order and integer comparisons determine the result; no shared mutable state is read.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This is private zero extension, not an unchecked array index.
//
// AI-Hints:
//   - Keep the negative-index check before indexing; the quotient normalizer relies on it.
func (a *natural) word(index int) uint64 {
	// Virtual zero extension must be checked before an array access.
	if index < 0 || index >= a.used {
		return 0
	}
	return a.words[index]
}

// normalizedWord reads one word of a virtually left-normalized magnitude.
// It avoids allocating or populating a complete shifted numerator for quotient estimation.
//
// Implementation:
//   - Stage 1: read the current and previous words using zero extension.
//   - Stage 2: shift the current word up and import spill bits from the previous word.
//   - Stage 3: combine disjoint bit ranges with OR.
//
// Behavior highlights:
//   - For shift zero, the previous-word contribution vanishes.
//
// Inputs:
//   - a: read-only normalized magnitude.
//   - index: desired word position in the virtually shifted value.
//   - shift: normalization distance in [0,63].
//
// Returns:
//   - uint64: the exact word at index in a*2^shift.
//
// Errors:
//   - No error return; the documented private preconditions are established by the caller.
//
// Determinism:
//   - Input order and integer comparisons determine the result; no shared mutable state is read.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - A shift by limbBits is intentional when shift=0.
//
// AI-Hints:
//   - Do not reinterpret absent high words as uninitialized storage.
func (a *natural) normalizedWord(index int, shift int) uint64 {
	// The left and right parts occupy disjoint bit positions, including shift=0.
	return a.word(index)<<shift | a.word(index-1)>>(limbBits-shift)
}

// roundedSmallQuotient rounds a bounded exact positive rational quotient to an integer.
// Its single guard/remainder decision implements nearest-even for the selected binary64 lattice.
//
// Implementation:
//   - Stage 1: normalize the nonzero divisor and obtain a two-word/one-word quotient estimate.
//   - Stage 2: decrease the estimate until its exact product does not exceed the numerator.
//   - Stage 3: compare twice the remainder with the denominator; increment only above half or on an odd tie.
//
// Behavior highlights:
//   - The normalized estimate is at most two too large; correction is bounded, not a history scan.
//   - A true zero quotient is supported; rounding may produce 2^53 after a carry.
//
// Inputs:
//   - numerator: read-only nonnegative magnitude.
//   - denominator: read-only positive magnitude with numerator/denominator<2^53.
//
// Returns:
//   - uint64: round-to-nearest-even(numerator/denominator), at most 2^53.
//
// Errors:
//   - No error return; the documented private preconditions are established by the caller.
//
// Determinism:
//   - Input order and integer comparisons determine the result; no shared mutable state is read.
//
// Complexity:
//   - Time O(L), Space O(L), L<=67; at most two estimate corrections.
//
// Notes:
//   - The quotient bound gives high<divisor for bits.Div64; the divisor cannot be zero.
//   - All product and doubled-remainder temporaries fit the proved fixed capacity.
//
// AI-Hints:
//   - Do not round a floating quotient and then round again to binary64.
//   - A half-way quotient increments only when its current low bit is one.
func roundedSmallQuotient(numerator, denominator *natural) uint64 {
	// Normalize the divisor so its highest word has bit 63 set.
	top := denominator.used - 1
	shift := bits.LeadingZeros64(denominator.words[top])
	divisor := denominator.normalizedWord(top, shift)
	high := numerator.normalizedWord(top+1, shift)
	low := numerator.normalizedWord(top, shift)
	// numerator/denominator<2^53 implies high<divisor, satisfying Div64's contract.
	quotient, _ := bits.Div64(high, low, divisor)
	product := denominator.multiplyWord(quotient)
	// A normalized leading-word estimate never underestimates and overshoots by at most two.
	for product.compare(numerator) > 0 {
		quotient--
		product.subtract(denominator)
	}
	// Recover the exact remainder from the corrected quotient, preserving all low bits.
	remainder := *numerator
	remainder.subtract(&product)
	twice := remainder.shifted(1)
	comparison := twice.compare(denominator)
	// Above half rounds up; exact halves round toward an even significand.
	if comparison > 0 || comparison == 0 && quotient&1 != 0 {
		quotient++
	}
	return quotient
}

// roundingShift selects the binary64 lattice using exact integer comparisons.
// The chosen denominator shift retains 53 significant bits or one least-subnormal unit.
//
// Implementation:
//   - Stage 1: estimate the exponent difference from exact magnitude bit lengths.
//   - Stage 2: select shift zero below the normal significand threshold.
//   - Stage 3: otherwise compare against the candidate power-of-two boundary and correct by one.
//
// Behavior highlights:
//   - No logarithm, approximate exponent, or epsilon participates in lattice selection.
//
// Inputs:
//   - numerator, denominator: positive normalized magnitudes from a structurally valid state.
//
// Returns:
//   - int: nonnegative denominator shift preserving the rounded quotient bound.
//
// Errors:
//   - No error return; the documented private preconditions are established by the caller.
//
// Determinism:
//   - Input order and integer comparisons determine the result; no shared mutable state is read.
//
// Complexity:
//   - Time O(L), Space O(L), L<=67.
//
// Notes:
//   - A bit-length difference can overestimate the true ratio exponent by exactly one.
//
// AI-Hints:
//   - The boundary comparison must use integers; floating division could overflow or underflow first.
func roundingShift(numerator, denominator *natural) int {
	// Estimate the unit exponent needed to retain a 53-bit significand.
	shift := numerator.bitLen() - denominator.bitLen() - fractionBits
	// The subnormal lattice cannot be finer than one 2^-1074 price unit.
	if shift <= 0 {
		return 0
	}
	// Bit lengths alone can overestimate by one, so compare the exact ratio boundary.
	boundary := denominator.shifted(shift + fractionBits)
	if numerator.compare(&boundary) < 0 {
		shift--
	}
	return shift
}

// value converts exact segment sums into the public readiness/value pair.
// Only the final weighted rational is rounded; exact intermediate sums are unchanged.
//
// Implementation:
//   - Stage 1: classify zero volume as undefined and exact zero weighted sum as valid +0.
//   - Stage 2: select the lattice and round its exact quotient once to nearest-even.
//   - Stage 3: scale the exact rounded integer (at most 2^53) and reapply the numerator sign.
//
// Behavior highlights:
//   - Negative nonzero underflow may yield -0; exact cancellation yields +0.
//
// Inputs:
//   - a: read-only sums satisfying V>=0 and |W|<=MaxFloat64Units*V.
//
// Returns:
//   - Value: masked +0 for V=0, otherwise a finite correctly rounded mean.
//
// Errors:
//   - No error return; the documented private preconditions are established by the caller.
//
// Determinism:
//   - Input order and integer comparisons determine the result; no shared mutable state is read.
//
// Complexity:
//   - Time O(L), Space O(L), L<=67.
//
// Notes:
//   - A quotient rounding carry to 2^53 is exactly representable; Ldexp applies an exact lattice scaling.
//
// AI-Hints:
//   - Never turn a valid zero into an undefined result or treat a huge total volume as overflow.
func (a *accumulator) value() Value {
	// No positive volume means undefined 0/0, represented by the masked +0 carrier.
	if a.volume.used == 0 {
		return Value{}
	}
	// Positive volume with an exactly zero numerator is a valid zero measurement.
	if a.weighted.used == 0 {
		return Value{Valid: true}
	}
	shift := roundingShift(&a.weighted, &a.volume)
	// Dividing by the shifted denominator expresses the result in the selected output lattice.
	denominator := a.volume.shifted(shift)
	rounded := roundedSmallQuotient(&a.weighted, &denominator)
	// rounded<=2^53 is exactly representable, and this power-of-two scaling is exact.
	value := math.Ldexp(float64(rounded), shift-subnormalScale)
	// Apply the sign after rounding so a negative nonzero underflow retains -0.
	if a.negative {
		value = -value
	}
	return Value{Value: value, Valid: true}
}
