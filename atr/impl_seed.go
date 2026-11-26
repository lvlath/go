// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package atr provides exact True Range and Wilder Average True Range computations.
// This file implements bounded exact seed arithmetic and one-round conversion.
package atr

import (
	"math"
	"math/bits"
)

const (
	// limbCount reserves 2176 bits, exceeding the 2162-bit maximum weighted binary64 numerator.
	limbCount = 34
	// limbBits states that each exact limb uses all bits of an unsigned machine-independent uint64.
	limbBits = 64
	// decimalBase fixes base ten; the snapshot wire grammar uses base-ten ASCII, independent of locale.
	decimalBase = 10
	// exponentMask selects the exponent; the IEEE binary64 exponent occupies eleven bits above the fraction.
	exponentMask uint64 = 0x7ff
	// fractionBits records that binary64 explicitly stores 52 fraction bits; normal values add the hidden bit.
	fractionBits = 52
	// significandBits records that a normal binary64 significand has 53 effective precision bits.
	significandBits = fractionBits + 1
	// fractionMask excludes sign and exponent while retaining the stored fraction.
	fractionMask uint64 = (1 << fractionBits) - 1
	// hiddenBit is the implicit leading bit of normal values; subnormals do not have it.
	hiddenBit uint64 = 1 << fractionBits
	// maxSeedDigits caps wire input at 660 bytes before any digit is consumed.
	maxSeedDigits = 660
)

// exactSum is an unsigned fixed-width integer in units of 2^-1074.
// Its value-copy semantics let Push calculate a private candidate without shared backing memory.
//
// Implementation:
//   - Stage 1: start with zeroed little-endian uint64 limbs.
//   - Stage 2: add exact binary64 significands times integer weights.
//   - Stage 3: divide and round once, or serialize the unfinished integer seed.
//
// Behavior highlights:
//   - MaxFloat64Units needs 2098 bits; a uint64 multiple needs at most 2162.
//   - The 2176-bit capacity therefore covers every admitted weighted numerator.
//
// Inputs:
//   - Only bounded nonnegative seed/affine magnitudes; parser candidates are checked.
//
// Returns:
//   - A private fixed array, never exposed or aliased through public Snapshot.
//
// Errors:
//   - Digit-capacity failures are rejected by parseSeed and mapped by Restore.
//
// Determinism:
//   - Index zero stores the least significant 64 bits; carry order is explicit.
//
// Complexity:
//   - Fixed O(W) storage and copy cost, W=34; no period-dependent storage.
//
// Notes:
//   - MaxFloat64Units=(2^53-1)<<2045. This is not a rounded float accumulator.
//
// AI-Hints:
//   - Do not replace the array by a slice or heap-growing integer on the Push path.
type exactSum [limbCount]uint64

// addWord adds one word into the fixed integer and propagates carry upward.
// All callers have already proved that the complete sum fits the accumulator capacity.
//
// Implementation:
//   - Stage 1: Add word at the supplied little-endian index.
//   - Stage 2: Propagate each carry to the next higher limb.
//   - Stage 3: Stop as soon as there is no carry left.
//
// Behavior highlights:
//   - A zero word does no work and does not read the indexed limb.
//
// Inputs:
//   - z: an exact nonnegative integer; index: valid starting limb; word: unsigned addend.
//
// Returns:
//   - No value; z contains the exact previous value plus word shifted by index*64.
//
// Errors:
//   - None under the private capacity precondition; public state is bounded before use.
//
// Determinism:
//   - Carry propagates strictly from low to high limbs.
//
// Complexity:
//   - Time O(W) worst case, Space O(1); W=34.
//
// Notes:
//   - The maximum reachable numerator needs 2162 bits, below the allocated 2176.
//
// AI-Hints:
//   - Do not expose arbitrary indexes or call this with an unvalidated external accumulator.
func (z *exactSum) addWord(index uint64, word uint64) {
	// Carry propagation stops before the proven fixed-width limit.
	for word != 0 {
		var carry uint64
		// Add64 preserves the exact carry bit rather than wrapping silently.
		z[index], carry = bits.Add64(z[index], word, 0)
		word = carry
		index++
	}
}

// addProduct adds a finite nonnegative binary64 value times an exact integer weight.
// The product is accumulated in subnormal units without floating multiplication or rounding.
//
// Implementation:
//   - Stage 1: Extract exponent and fraction; restore the hidden bit only for normal operands.
//   - Stage 2: Multiply the significand by weight into two exact machine words.
//   - Stage 3: Distribute those words across aligned limbs, preserving every carry.
//
// Behavior highlights:
//   - Subnormal and zero inputs need no hidden bit; a zero weight contributes nothing.
//
// Inputs:
//   - z: bounded prior sum; x: finite nonnegative float or either zero sign.
//   - weight: unsigned exact coefficient; total weighted magnitude must fit 2176 bits.
//
// Returns:
//   - No value; z receives the exact represented product.
//
// Errors:
//   - None under the validated finite-input and combined-weight bounds.
//
// Determinism:
//   - Bit extraction, integer multiplication, and low-to-high carry propagation only.
//
// Complexity:
//   - Time O(W), Space O(1); at most three word insertions.
//
// Notes:
//   - For normal exponent E, units equal significand << (E-1); subnormals use shift zero.
//
// AI-Hints:
//   - Never multiply x by float64(weight); large periods would lose integer information.
func (z *exactSum) addProduct(x float64, weight uint64) {
	// Decode the represented binary64 value; original decimal text is irrelevant.
	raw := math.Float64bits(x)
	significand := raw & fractionMask
	shift := (raw >> fractionBits) & exponentMask
	// Normal values include an implicit leading one and exponent offset E-1.
	if shift != 0 {
		significand |= hiddenBit
		shift--
	}
	// The full significand-times-weight product occupies at most 117 bits.
	high, low := bits.Mul64(significand, weight)
	// Split the exact binary shift into a limb index and an intra-limb displacement.
	index, offset := shift/limbBits, shift%limbBits
	z.addWord(index, low<<offset)
	// Word alignment eliminates the cross-word fragments; keep the direct path exact.
	if offset == 0 {
		z.addWord(index+1, high)
		return
	}
	// Disjoint fragments join with OR; their bit positions cannot overlap.
	z.addWord(index+1, high<<offset|low>>(limbBits-offset))
	z.addWord(index+2, high>>(limbBits-offset))
}

// divide performs exact long division of the complete fixed-width integer.
// The quotient overwrites z and the returned remainder preserves all rounding evidence.
//
// Implementation:
//   - Stage 1: Initialize a zero remainder above the highest limb.
//   - Stage 2: Divide each combined remainder/word from high limb to low limb.
//   - Stage 3: Return the final remainder after storing all quotient words.
//
// Behavior highlights:
//   - The previous remainder is always smaller than divisor, satisfying bits.Div64.
//
// Inputs:
//   - z: unsigned integer; divisor: nonzero exact denominator.
//
// Returns:
//   - uint64: remainder in [0, divisor); z becomes floor(original/divisor).
//
// Errors:
//   - None; callers establish divisor>=1 before entering this private kernel.
//
// Determinism:
//   - Strict descending limb order; no floating operations.
//
// Complexity:
//   - Time O(W), Space O(1), W=34.
//
// Notes:
//   - Used for both period division and decimal conversion of a private working copy.
//
// AI-Hints:
//   - Do not pass zero or an unvalidated period to bits.Div64.
func (z *exactSum) divide(divisor uint64) uint64 {
	var remainder uint64
	// Descending limbs consume the highest-order information first.
	for i := len(z) - 1; i >= 0; i-- {
		// Prior remainder<divisor prevents quotient overflow in this machine operation.
		z[i], remainder = bits.Div64(remainder, z[i], divisor)
	}
	return remainder
}

// bitLength returns the number of significant bits in a bounded integer.
// Zero has length zero; leading zero limbs do not affect the result.
//
// Implementation:
//   - Stage 1: Scan limbs from the highest position down.
//   - Stage 2: Locate the top bit in the first nonzero word by six binary probes.
//   - Stage 3: Return its one-based absolute bit position, or zero for the empty integer.
//
// Behavior highlights:
//   - No conversion from a possibly signed bit-count helper is needed.
//
// Inputs:
//   - z: any fixed-width unsigned accumulator.
//
// Returns:
//   - uint64: a length in [0,2176].
//
// Errors:
//   - None.
//
// Determinism:
//   - Highest nonzero limb and exact binary probes determine the unique result.
//
// Complexity:
//   - Time O(W+log(64)), Space O(1); both bounds are fixed.
//
// Notes:
//   - The method does not modify the accumulator.
//
// AI-Hints:
//   - Keep zero as a separate case when selecting a rounded significand.
func (z *exactSum) bitLength() uint64 {
	// Unsigned loop index is decremented only while positive.
	for i := uint64(limbCount); i > 0; i-- {
		word := z[i-1]
		// An empty high limb contributes no significant bit.
		if word == 0 {
			continue
		}
		length := (i-1)*limbBits + 1
		// Binary probes 32,16,8,4,2,1 locate the most significant set bit.
		for shift := uint64(limbBits / 2); shift > 0; shift >>= 1 {
			if word>>shift != 0 {
				word >>= shift
				length += shift
			}
		}
		return length
	}
	return 0
}

// lowerBits tests whether any of the least-significant count bits are nonzero.
// It supplies the sticky predicate below the rounding guard bit.
//
// Implementation:
//   - Stage 1: Inspect every complete low limb in the requested bit prefix.
//   - Stage 2: Mask only the remaining low bits of the boundary limb.
//   - Stage 3: Return whether either region contains a set bit.
//
// Behavior highlights:
//   - A zero-bit prefix is false, even when higher bits exist.
//
// Inputs:
//   - z: quotient integer; count: a prefix length below the fixed capacity.
//
// Returns:
//   - bool: exact logical OR over the selected low-bit prefix.
//
// Errors:
//   - None under the private prefix-length bound.
//
// Determinism:
//   - Low-to-high limb scan with an exact final mask.
//
// Complexity:
//   - Time O(W), Space O(1); early exit does not alter the predicate.
//
// Notes:
//   - The division remainder is a separate sticky contribution, added by roundedQuotient.
//
// AI-Hints:
//   - Do not include the guard bit itself in this sticky prefix.
func (z *exactSum) lowerBits(count uint64) bool {
	// Separate full prefix limbs from the final partial prefix.
	index, offset := count/limbBits, count%limbBits
	for i := uint64(0); i < index; i++ {
		if z[i] != 0 {
			return true
		}
	}
	// Offset zero gives an empty mask and therefore no boundary contribution.
	return z[index]&((uint64(1)<<offset)-1) != 0
}

// roundedQuotient rounds the exact rational (z/divisor)*2^-1074 to binary64 once.
// Quotient bits and the integer remainder jointly implement round-to-nearest, ties-to-even.
//
// Implementation:
//   - Stage 1: Divide a private copy and select at most 53 leading significand bits.
//   - Stage 2: Use the exact remainder at unit spacing, or guard/sticky evidence at wider spacing.
//   - Stage 3: Increment on above-half or odd-half ties, handle carry, and encode the float bits.
//
// Behavior highlights:
//   - Handles subnormal ties, positive-zero underflow, and the normal exponent carry.
//
// Inputs:
//   - z: nonnegative numerator whose exact average lies in the finite float64 domain.
//   - divisor: positive exact integer count.
//
// Returns:
//   - float64: nearest-even finite result; z remains unchanged.
//
// Errors:
//   - None; seed means and mature convex combinations establish the finite bound.
//
// Determinism:
//   - No floating intermediate; tie decisions depend on exact discarded bits.
//
// Complexity:
//   - Time O(W), Space O(W) for the quotient copy; W=34.
//
// Notes:
//   - Remainder versus divisor-remainder avoids overflow from doubling the remainder.
//
// AI-Hints:
//   - When shift>0, a nonzero remainder contributes sticky evidence below the quotient bits.
//   - A carry from 53 retained bits changes the exponent, not the rounding rule.
func (z *exactSum) roundedQuotient(divisor uint64) float64 {
	// Division destroys only this local copy, not the exact numerator.
	quotient := *z
	// Keep the remainder: quotient bits alone would misclassify some ties.
	remainder := quotient.divide(divisor)
	length := quotient.bitLength()
	var shift uint64
	// At most 53 significant bits fit in a normal binary64 significand.
	if length > significandBits {
		shift = length - significandBits
	}
	// Split the exact binary shift into a limb index and an intra-limb displacement.
	index, offset := shift/limbBits, shift%limbBits
	// Join the leading retained bits; a Go shift by 64 contributes zero when aligned.
	significand := quotient[index]>>offset | quotient[index+1]<<(limbBits-offset)
	var increment bool
	// Unit spacing includes subnormals; compare residue to its complement without 2*r.
	if shift == 0 {
		increment = remainder > divisor-remainder ||
			(remainder == divisor-remainder && significand&1 != 0)
	} else {
		// The first discarded bit marks the exact half-ULP boundary.
		guard := (quotient[(shift-1)/limbBits] >> ((shift - 1) % limbBits)) & 1
		// Any lower bit or division residue places the exact value strictly above that tie.
		sticky := quotient.lowerBits(shift-1) || remainder != 0
		increment = guard != 0 && (sticky || significand&1 != 0)
	}
	// Halfway values increment only an odd retained significand.
	if increment {
		significand++
	}
	// Rounding can carry out of all 53 retained bits into the next exponent.
	if significand == 1<<significandBits {
		significand >>= 1
		shift++
	}
	// Subnormal encodings have exponent zero and directly store the unit count.
	if significand < hiddenBit {
		return math.Float64frombits(significand)
	}
	// Encode the finite normal exponent and remove its implicit leading bit.
	return math.Float64frombits((shift+1)<<fractionBits | (significand & fractionMask))
}

// decimal serializes the exact seed integer as canonical unsigned ASCII decimal.
// Only the returned nonzero string may allocate; the working integer and digit buffer are fixed.
//
// Implementation:
//   - Stage 1: Return the shared literal for integer zero.
//   - Stage 2: Repeatedly divide a private copy by ten and write remainders backward.
//   - Stage 3: Copy the occupied suffix into the detached immutable result string.
//
// Behavior highlights:
//   - No signs, leading zeros, exponent notation, or locale dependence.
//
// Inputs:
//   - z: fixed unsigned seed integer; the receiver is never mutated.
//
// Returns:
//   - string: canonical decimal with at most 656 digits for the full 2176-bit capacity.
//
// Errors:
//   - None; the maxSeedDigits buffer exceeds the proven decimal capacity.
//
// Determinism:
//   - Repeated exact quotient/remainder operations uniquely determine the text.
//
// Complexity:
//   - Time O(D*W), Space O(D+W) including the returned text; D<=660.
//
// Notes:
//   - Snapshot may allocate; Push never calls decimal.
//
// AI-Hints:
//   - Do not serialize a rounded seed mean in place of this exact integer.
func (z *exactSum) decimal() string {
	// Canonical zero needs no digit buffer or heap allocation.
	if z.bitLength() == 0 {
		return "0"
	}
	// Preserve the live seed while repeatedly dividing a private copy.
	work := *z
	var buffer [maxSeedDigits]byte
	pos := len(buffer)
	// Write least-significant decimal digits backward into bounded storage.
	for work.bitLength() != 0 {
		digit := work.divide(decimalBase)
		pos--
		// The remainder is in [0,9]; indexing avoids a narrowing conversion.
		buffer[pos] = "0123456789"[digit]
	}
	// Return detached immutable text; Snapshot, not Push, pays this allocation.
	return string(buffer[pos:])
}

// appendDigit appends one validated decimal digit to a fixed-width unsigned integer.
// The final carry reports overflow instead of silently accepting a wrapped checkpoint.
//
// Implementation:
//   - Stage 1: Initialize the carry to the incoming digit.
//   - Stage 2: Multiply each low-to-high limb by ten and add the carry.
//   - Stage 3: Return whether the last carry fits inside the fixed capacity.
//
// Behavior highlights:
//   - The partial candidate is discarded by parseSeed after any overflow.
//
// Inputs:
//   - z: private parse candidate; digit: validated integer in [0,9].
//
// Returns:
//   - bool: true when the complete multiplication/addition fit.
//
// Errors:
//   - No sentinel; false denotes capacity overflow to the caller.
//
// Determinism:
//   - Low-to-high multiply/add with exact carry propagation.
//
// Complexity:
//   - Time O(W), Space O(1).
//
// Notes:
//   - A false return may leave z modified; it must never be published.
//
// AI-Hints:
//   - Do not reuse a failed parse candidate or reinterpret wrapped limbs as a seed.
func (z *exactSum) appendDigit(digit uint64) bool {
	// Decimal append equals z*10+digit, carried from the least significant limb.
	carry := digit
	for i := range z {
		high, low := bits.Mul64(z[i], decimalBase)
		var added uint64
		z[i], added = bits.Add64(low, carry, 0)
		carry = high + added
	}
	// Nonzero final carry means the parse candidate overflowed and must be discarded.
	return carry == 0
}

// parseSeed parses a bounded canonical unsigned decimal checkpoint field.
// The entire string must match 0|[1-9][0-9]*; permissive numeric syntax is rejected.
//
// Implementation:
//   - Stage 1: Reject empty/oversized strings before scanning; accept the canonical zero literal.
//   - Stage 2: Require a nonzero first digit and ASCII decimal characters throughout.
//   - Stage 3: Accumulate each digit in fixed storage, rejecting capacity overflow.
//
// Behavior highlights:
//   - No heap allocation even on hostile input; leading signs and whitespace are not accepted.
//
// Inputs:
//   - text: externally supplied seed integer, at most maxSeedDigits bytes.
//
// Returns:
//   - exactSum and bool: exact parsed value on true; ignore the candidate on false.
//
// Errors:
//   - No sentinel; Restore converts any parse failure to ErrStateVersion.
//
// Determinism:
//   - Strict left-to-right ASCII parsing without locale or Unicode normalization.
//
// Complexity:
//   - Time O(D*W), Space O(W), with O(1) length rejection for D>660.
//
// Notes:
//   - The semantic Count*MaxFloat64Units bound is checked separately after parsing.
//
// AI-Hints:
//   - Do not accept +0, -0, 00, decimal points, exponent notation, or Unicode digits.
func parseSeed(text string) (exactSum, bool) {
	var sum exactSum
	// Bound both running time and fixed-buffer capacity before scanning hostile text.
	if len(text) == 0 || len(text) > maxSeedDigits {
		return sum, false
	}
	// Zero has exactly one canonical spelling.
	if text == "0" {
		return sum, true
	}
	// Reject signs, leading zeros, whitespace, and non-ASCII leading bytes.
	if text[0] < '1' || text[0] > '9' {
		return sum, false
	}
	// Validate bytes, not Unicode digits or permissive numeric tokens.
	for i := 0; i < len(text); i++ {
		if text[i] < '0' || text[i] > '9' {
			return exactSum{}, false
		}
		// Conversion is safe because the byte has already been restricted to ASCII 0..9.
		if !sum.appendDigit(uint64(text[i] - '0')) {
			return exactSum{}, false
		}
	}
	return sum, true
}

// atMost compares two complete unsigned exact accumulators.
// The highest unequal limb determines the mathematical integer ordering.
//
// Implementation:
//   - Stage 1: Visit limb pairs from most significant to least significant.
//   - Stage 2: Return the ordering of the first unequal pair.
//   - Stage 3: Accept equality after the entire scan.
//
// Behavior highlights:
//   - Equal sums are admissible at the exact snapshot magnitude bound.
//
// Inputs:
//   - z, other: initialized fixed-width unsigned integer values.
//
// Returns:
//   - bool: true exactly when z <= other.
//
// Errors:
//   - None; private callers supply non-nil operands.
//
// Determinism:
//   - Strict high-to-low unsigned comparison.
//
// Complexity:
//   - Time O(W), Space O(1).
//
// Notes:
//   - Neither operand is modified.
//
// AI-Hints:
//   - Do not convert large seed values to float64 merely to check their bounds.
func (z *exactSum) atMost(other *exactSum) bool {
	// Descending limbs consume the highest-order information first.
	for i := len(z) - 1; i >= 0; i-- {
		// The highest unequal limb determines ordering for the complete unsigned integers.
		if z[i] != other[i] {
			return z[i] < other[i]
		}
	}
	return true
}
