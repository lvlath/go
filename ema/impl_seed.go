// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ema provides exact affine exponential moving average smoothing.
// This file implements bounded signed integer arithmetic in binary64 subnormal units.
package ema

import (
	"math"
	"math/bits"
)

const (
	// limbCount holds 2176 bits: more than 2098 finite-input bits plus 64 count bits.
	// The extra capacity covers every admitted seed and affine numerator without growth.
	limbCount = 34
	// wordBits is the storage width used by math/bits' uint64 carry operations.
	wordBits = 64
	// fractionBits excludes the implicit leading one in a normal binary64 value.
	fractionBits = 52
	// significandBits includes that leading one; rounding retains this many bits.
	significandBits = fractionBits + 1
	// fractionMask extracts the explicitly stored fractional significand.
	fractionMask = uint64(1)<<fractionBits - 1
	// hiddenBit restores the implicit bit for normals, never for subnormals.
	hiddenBit = uint64(1) << fractionBits
	// exponentMask selects the eleven-bit biased exponent field.
	exponentMask = uint64(0x7ff)
	// unitExponent makes one accumulator unit equal to the smallest binary64, 2^-1074.
	unitExponent = 1074
	// decimalRadix is used symmetrically by the canonical seed parser and formatter.
	decimalRadix = uint64(10)
)

// magnitude stores a nonnegative exact integer in fixed little-endian uint64 limbs.
// Its capacity covers every admitted seed and affine numerator without retaining history.
//
// Implementation:
//   - Stage 1: zero-initialize all 34 limbs.
//   - Stage 2: apply bounded carry, borrow, multiply, and divide operations.
//   - Stage 3: extract exact high and low bits for a single rational rounding decision.
//
// Behavior highlights:
//   - Little-endian denotes significance order, not a host-memory serialization format.
//
// Inputs:
//   - Internal arithmetic only; public decimal snapshots are validated before admission.
//
// Returns:
//   - A 2176-bit value type with no slice, pointer-backed buffer, or growth policy.
//
// Errors:
//   - multiplyAdd reports discarded carry for untrusted decimal parsing.
//
// Determinism:
//   - Every primitive has an explicit limb traversal order.
//
// Complexity:
//   - Space O(W), W=34; individual operations document their O(W) or O(1) work.
//
// Notes:
//   - A copied magnitude owns every limb independently.
//
// AI-Hints:
//   - Do not replace the fixed array by a growing big integer in the Push path.
type magnitude [limbCount]uint64

// exactSum associates a canonical sign with an exact fixed-width magnitude.
// Zero is represented only by a zero magnitude and a nonnegative sign.
//
// Implementation:
//   - Stage 1: decompose admitted finite observations into signed subnormal units.
//   - Stage 2: combine signs using exact magnitude addition or ordered subtraction.
//   - Stage 3: round once at seed completion or serialize exact units while still warming up.
//
// Behavior highlights:
//   - The state preserves small terms after large opposite-sign cancellation.
//
// Inputs:
//   - Canonical internal magnitudes or checked snapshot decimal input.
//
// Returns:
//   - A detached sign-magnitude value in units of 2^-1074.
//
// Errors:
//   - Public Restore rejects malformed sign/decimal spellings with ErrStateVersion.
//
// Determinism:
//   - Exact cancellation always clears the sign, independently of input order.
//
// Complexity:
//   - Space O(W), W=34, with no heap-backed storage.
//
// Notes:
//   - The sign is not encoded as two's-complement extension across limbs.
//
// AI-Hints:
//   - Do not admit negative zero or retain an old sign after exact cancellation.
type exactSum struct {
	// mag owns the absolute integer units, least-significant limb first.
	mag magnitude
	// negative is true only for a nonzero negative mathematical integer.
	negative bool
}

// binaryUnits represents a finite binary64 as signed integer multiples of 2^-1074.
// No rounded floating-point arithmetic or loss of subnormal bits occurs.
//
// Implementation:
//   - Stage 1: extract sign, exponent, and fraction from the original bits.
//   - Stage 2: restore the normal hidden bit and derive exponent-1 as the unit shift.
//   - Stage 3: place at most 53 bits in two adjacent limbs and canonicalize zero sign.
//
// Behavior highlights:
//   - Both input zero signs produce the same nonnegative integer zero.
//
// Inputs:
//   - x: finite binary64 admitted by the caller; no NaN or infinity reaches this kernel.
//
// Returns:
//   - exactSum: detached exact signed units of x.
//
// Errors:
//   - None; finite-input validation belongs to the public boundary.
//
// Determinism:
//   - Integer extraction is independent of floating-point expression evaluation.
//
// Complexity:
//   - Time O(W), Space O(W), W=34; zero initialization dominates the two-limb insertion.
//
// Notes:
//   - Maximum shift is 2045: both destination limbs are inside the fixed array.
//   - A Go unsigned shift by 64 yields zero, which handles a limb-aligned value.
//
// AI-Hints:
//   - Do not round x through decimal text or multiply x by 2^1074.
//   - Do not insert the hidden bit for exponent zero.
func binaryUnits(x float64) exactSum {
	// Read the represented value, not its decimal spelling.
	raw := math.Float64bits(x)
	// Extract the stored fraction before restoring the normal leading bit.
	mantissa := raw & fractionMask
	// Select the biased exponent without the sign bit.
	exponent := (raw >> fractionBits) & exponentMask
	// Subnormal integers already use the accumulator unit scale.
	shift := uint64(0)
	// Normal values include an implicit leading one and a larger unit shift.
	if exponent != 0 {
		// Recover all 53 represented significand bits.
		mantissa |= hiddenBit
		// Combining the binary64 bias and 2^-1074 units gives exponent-1.
		shift = exponent - 1
	}
	// Clear all unused limbs so the result owns a canonical representation.
	var result exactSum
	// Split the bit shift into its limb index and offset within that limb.
	word, offset := shift/wordBits, shift%wordBits
	// Place the low part; any crossing bits are handled by the adjacent limb.
	result.mag[word] = mantissa << offset
	// The highest finite exponent still leaves this next limb inside storage.
	result.mag[word+1] = mantissa >> (wordBits - offset)
	// Zero has no negative integer representation, regardless of input sign.
	result.negative = raw>>(wordBits-1) != 0 && mantissa != 0
	return result
}

// compare orders nonnegative magnitudes without modifying either operand.
// The first differing high limb decides the order of the complete integer.
//
// Implementation:
//   - Stage 1: visit limbs from most to least significant.
//   - Stage 2: return at the first inequality, or zero after a complete match.
//
// Behavior highlights:
//   - Leading zero limbs need no normalization pass.
//
// Inputs:
//   - a, b: non-nil fixed-width magnitudes; aliases are safe for this read-only operation.
//
// Returns:
//   - int: -1 for a<b, 0 for a==b, and +1 for a>b.
//
// Errors:
//   - None; callers supply internal fixed arrays.
//
// Determinism:
//   - Most-significant unequal limb wins, with no map or floating-point comparisons.
//
// Complexity:
//   - Time O(W), Space O(1), W=34.
//
// Notes:
//   - Sign comparison, when needed, is performed by exactSum.add.
//
// AI-Hints:
//   - Do not compare low limbs first: a larger low limb cannot override a higher unequal limb.
func (a *magnitude) compare(b *magnitude) int {
	for i := limbCount - 1; i >= 0; i-- {
		// A smaller first unequal high limb decides the entire integer order.
		if a[i] < b[i] {
			return -1
		}
		// Lower limbs cannot reverse this first greater high-limb comparison.
		if a[i] > b[i] {
			return 1
		}
	}
	return 0
}

// add replaces a with the exact bounded unsigned sum a+b.
// The caller proves that the result fits the fixed accumulator before invoking it.
//
// Implementation:
//   - Stage 1: start with a zero carry.
//   - Stage 2: add low-to-high limbs, forwarding each carry once.
//
// Behavior highlights:
//   - No allocation, resizing, or floating-point intermediate is used.
//
// Inputs:
//   - a: mutable magnitude; b: addend; their sum must fit 2176 bits.
//
// Returns:
//   - No value; a contains the sum and b is read only unless it aliases a.
//
// Errors:
//   - None; the final carry is zero under the seed/convex-numerator bounds.
//
// Determinism:
//   - Carry propagation follows ascending limb order.
//
// Complexity:
//   - Time O(W), Space O(1), W=34.
//
// Notes:
//   - Even Count*MaxBinary64Units is below 2^2162.
//
// AI-Hints:
//   - Do not reuse this unchecked operation for parsing an unbounded external integer.
func (a *magnitude) add(b *magnitude) {
	// No lower limb contributes an incoming carry to the first addition.
	var carry uint64
	for i := range a {
		// Forward the exact high carry; the capacity proof excludes a final overflow.
		a[i], carry = bits.Add64(a[i], b[i], carry)
	}
}

// subtract replaces a with the exact nonnegative difference a-b.
// Its ordered-magnitude precondition makes the final borrow zero.
//
// Implementation:
//   - Stage 1: start with a zero borrow.
//   - Stage 2: subtract low-to-high limbs with explicit borrow propagation.
//
// Behavior highlights:
//   - This subtracts exact integers, not floating-point levels in the EMA recurrence.
//
// Inputs:
//   - a, b: non-nil magnitudes with a>=b; a is the destination.
//
// Returns:
//   - No value; a contains the exact difference.
//
// Errors:
//   - None; exactSum.add establishes the magnitude order.
//
// Determinism:
//   - Ascending limbs propagate the unique binary borrow sequence.
//
// Complexity:
//   - Time O(W), Space O(1), W=34.
//
// Notes:
//   - Equal inputs yield all-zero limbs; signed-zero normalization belongs to exactSum.
//
// AI-Hints:
//   - Never call with a<b; that would create a wrapped unsigned magnitude.
func (a *magnitude) subtract(b *magnitude) {
	// The least significant subtraction has no incoming borrow.
	var borrow uint64
	for i := range a {
		// Borrow propagates upward once per limb; a>=b excludes final underflow.
		a[i], borrow = bits.Sub64(a[i], b[i], borrow)
	}
}

// multiplyAdd replaces a with a*factor+digit and reports capacity overflow.
// Its fixed-width carry chain supports both exact weights and bounded decimal parsing.
//
// Implementation:
//   - Stage 1: inject digit as the low carry.
//   - Stage 2: multiply each limb, add the carry, and forward the resulting high word.
//   - Stage 3: report whether a nonzero carry remains beyond the last limb.
//
// Behavior highlights:
//   - Parsing checks overflow before accepting the next decimal digit.
//
// Inputs:
//   - a: non-nil mutable magnitude.
//   - factor, digit: unsigned values; callers use digit<factor, or both zero.
//
// Returns:
//   - bool: true means overflow; a then contains truncated low bits and must be discarded.
//
// Errors:
//   - No error object; parseSeed converts overflow into a rejected snapshot.
//
// Determinism:
//   - Multiply/add traversal is strictly low-to-high.
//
// Complexity:
//   - Time O(W), Space O(1), W=34.
//
// Notes:
//   - Legal numerical callers use digit zero and prove capacity from count/weight bounds.
//   - For digit<factor, the outgoing carry is at most factor-1; factor=digit=0 clears a.
//
// AI-Hints:
//   - Never ignore the overflow flag when consuming caller-controlled decimal text.
func (a *magnitude) multiplyAdd(factor, digit uint64) bool {
	// Adding digit is equivalent to injecting it into the first product limb.
	carry := digit
	for i := range a {
		// Retain both halves of the word product before adding the carry.
		hi, lo := bits.Mul64(a[i], factor)
		// Account for the low-half carry separately from the multiplication high half.
		lo, extra := bits.Add64(lo, carry, 0)
		// Publish only this limb; any final overflow is reported to the caller.
		a[i] = lo
		// With digit<factor, this sum is <=factor-1 and cannot wrap.
		carry = hi + extra
	}
	// Parsing must reject a nonzero word beyond the fixed capacity.
	return carry != 0
}

// add combines two bounded signed integers without dropping small canceled terms.
// Magnitude subtraction preserves every low bit even when enormous terms cancel.
//
// Implementation:
//   - Stage 1: add magnitudes directly when signs agree.
//   - Stage 2: otherwise compare magnitudes and subtract the smaller from the larger.
//   - Stage 3: retain the larger operand sign, or reset exact cancellation to positive zero.
//
// Behavior highlights:
//   - The [large, small, -large] seed retains small exactly.
//
// Inputs:
//   - a: mutable canonical signed accumulator; b: detached canonical signed addend.
//
// Returns:
//   - No value; a becomes the exact bounded signed sum.
//
// Errors:
//   - None; callers establish the fixed-capacity bound.
//
// Determinism:
//   - Equal opposite magnitudes produce a fully cleared canonical zero.
//
// Complexity:
//   - Time O(W), Space O(W) for b passed by value; W=34.
//
// Notes:
//   - Neither operation order nor grouping changes an exact seed sum.
//
// AI-Hints:
//   - Do not replace the sign branch with a floating-point running mean.
func (a *exactSum) add(b exactSum) {
	// Equal signs require addition; canonical inputs cannot create negative zero.
	if a.negative == b.negative {
		// The seed/count and coefficient bounds prove this addition fits.
		a.mag.add(&b.mag)
		return
	}
	// Opposite signs cancel by exact magnitude subtraction, never float subtraction.
	switch a.mag.compare(&b.mag) {
	case 0:
		// Exact cancellation clears both the magnitude and the sign.
		*a = exactSum{}
	case 1:
		// The destination is larger, so its sign survives.
		a.mag.subtract(&b.mag)
	default:
		// The addend is larger; subtract into its detached copy.
		b.mag.subtract(&a.mag)
		// Transfer the winning magnitude and its sign together.
		*a = b
	}
}

// bitLen returns the number of bits needed to represent a nonnegative magnitude.
// It locates the highest occupied limb and then its highest set bit.
//
// Implementation:
//   - Stage 1: scan limbs high-to-low until a nonzero word is found.
//   - Stage 2: combine the word offset and bits.Len64, or return zero for an empty value.
//
// Behavior highlights:
//   - No temporary normalized representation is created.
//
// Inputs:
//   - a: non-nil read-only magnitude.
//
// Returns:
//   - int: zero for zero, otherwise a value in [1,2176].
//
// Errors:
//   - None for internal fixed arrays.
//
// Determinism:
//   - The most significant occupied bit determines the answer uniquely.
//
// Complexity:
//   - Time O(W), Space O(1), W=34.
//
// Notes:
//   - Used for a quotient window and for terminating decimal serialization.
//
// AI-Hints:
//   - Do not infer zero from just the lowest limb.
func (a *magnitude) bitLen() int {
	for i := limbCount - 1; i >= 0; i-- {
		// The first occupied high word determines both magnitude scale and quotient window.
		if a[i] != 0 {
			return wordBits*i + bits.Len64(a[i])
		}
	}
	return 0
}

// lowNonzero detects a sticky bit in the lowest count integer bits.
// It excludes the bit at index count, so a rounding guard bit can be tested separately.
//
// Implementation:
//   - Stage 1: scan all fully included low limbs.
//   - Stage 2: mask the included prefix of the remaining limb.
//
// Behavior highlights:
//   - count=0 yields false, including for nonzero high limbs.
//
// Inputs:
//   - a: non-nil magnitude; count: internal bit count in [0,2175].
//
// Returns:
//   - bool: whether any bit with index strictly below count is set.
//
// Errors:
//   - None; roundUnits derives a bounded count from admitted operands.
//
// Determinism:
//   - Low limbs are scanned ascending; only existence, not count, matters.
//
// Complexity:
//   - Time O(W), Space O(1), W=34.
//
// Notes:
//   - A zero offset produces a zero mask without special indexing.
//
// AI-Hints:
//   - Do not include the guard bit in the sticky region or ties will round incorrectly.
func (a *magnitude) lowNonzero(count int) bool {
	// Separate fully covered limbs from the final partial mask.
	word, offset := count/wordBits, count%wordBits
	for i := 0; i < word; i++ {
		// Any set bit in a fully covered low limb makes the discarded tail sticky.
		if a[i] != 0 {
			return true
		}
	}
	// The mask excludes bit count itself; offset zero deliberately yields zero.
	return a[word]&(uint64(1)<<offset-1) != 0
}

// shiftedWord returns the low uint64 of the exact integer a shifted right.
// Adjacent limbs are merged without rounding or modifying the source.
//
// Implementation:
//   - Stage 1: select the containing limb and intra-limb shift.
//   - Stage 2: merge the next limb when it exists; otherwise extend with zeros.
//
// Behavior highlights:
//   - A shift aligned to a word requires no special branch.
//
// Inputs:
//   - a: non-nil magnitude; shift: internal index in [0,2175].
//
// Returns:
//   - uint64: bits [shift,shift+64) with zero extension above storage.
//
// Errors:
//   - None; callers establish a valid starting limb.
//
// Determinism:
//   - Pure bit extraction, independent of machine endianness.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - A Go left shift by 64 produces zero for uint64, not a modulo-64 shift.
//
// AI-Hints:
//   - Do not replace the fixed bit order with host-endian unsafe loads.
func (a *magnitude) shiftedWord(shift int) uint64 {
	// Split the bit shift into its limb index and offset within that limb.
	word, offset := shift/wordBits, shift%wordBits
	// Extract the portion held by the containing limb.
	result := a[word] >> offset
	// Zero-extend at the high boundary instead of reading beyond the array.
	if word+1 < limbCount {
		// Merge the remaining high portion of this 64-bit extraction window.
		result |= a[word+1] << (wordBits - offset)
	}
	return result
}

// divideWord replaces a by floor(a/divisor) and returns the exact remainder.
// High-to-low long division preserves the remainder needed by the next limb.
//
// Implementation:
//   - Stage 1: start with a zero remainder.
//   - Stage 2: divide each combined remainder/limb pair, most significant first.
//
// Behavior highlights:
//   - Every bits.Div64 call has a high word strictly below divisor.
//
// Inputs:
//   - a: mutable magnitude; divisor: nonzero word, decimalRadix for serialization.
//
// Returns:
//   - uint64: original a modulo divisor; a now contains the quotient.
//
// Errors:
//   - None; positive divisor and remainder<divisor exclude Div64 failure conditions.
//
// Determinism:
//   - Fixed descending limb order performs exact Euclidean division.
//
// Complexity:
//   - Time O(W), Space O(1), W=34.
//
// Notes:
//   - Serialization passes a copy; this mutating primitive never touches the live stream.
//
// AI-Hints:
//   - Never pass a zero divisor or reverse the division traversal.
func (a *magnitude) divideWord(divisor uint64) uint64 {
	// The first high-word dividend is zero, strictly below the positive divisor.
	var remainder uint64
	for i := limbCount - 1; i >= 0; i-- {
		// Euclidean division preserves remainder<divisor for the next lower limb.
		a[i], remainder = bits.Div64(remainder, a[i], divisor)
	}
	// Decimal formatting consumes this exact least-significant digit.
	return remainder
}
