// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package vwap defines the exact canonical decimal continuation format.
// This file bounds parsing and verifies mathematical invariants before replacing live state.
package vwap

import "math/bits"

const (
	// snapshotVersion selects the canonical field and phase rules, not market metadata.
	snapshotVersion = 1
	// algorithm fixes exact binary64-product sums and one nearest-even output rounding.
	algorithm = "vwap/exact-weighted/v1"
	// maxDecimalBytes limits each serialized sum, including an optional minus sign.
	maxDecimalBytes = 1300
	// volumeBits bounds MaxUint64 * MaxFloat64Units before parsing can grow further.
	volumeBits = 2162
	// weightedBits bounds MaxUint64 * MaxFloat64Units^2, excluding the sign.
	weightedBits = 4260
	// decimalBase and decimalDigits define the canonical ASCII integer encoding.
	decimalBase   = 10
	decimalDigits = "0123456789"
)

// Snapshot returns a detached exact continuation record for the active segment.
// It serializes supplied binary64 values exactly rather than reconstructing decimals.
//
// Implementation:
//   - Stage 1: return Snapshot{} for nil or unconstructed receivers.
//   - Stage 2: encode the two integer accumulators as canonical ASCII strings.
//   - Stage 3: copy schema, configuration, count, and phase into the result.
//
// Behavior highlights:
//   - An initialized empty state has Version=1 and both sum strings equal to "0".
//
// Inputs:
//   - s: exclusively owned stream, possibly nil.
//
// Returns:
//   - An independent Snapshot with no mutable alias to stream storage.
//
// Errors:
//   - None; nil/zero receivers return the zero Snapshot.
//
// Determinism:
//   - Identical exact state yields identical decimal strings.
//
// Complexity:
//   - Time O(D*L), Space O(D+L), D<=1300 and L=67; returned strings may allocate.
//
// Notes:
//   - Callers must authenticate persisted records and preserve Count losslessly.
//
// AI-Hints:
//   - Snapshot is not a signed audit log and does not certify a causal anchor.
//   - Never save only the rounded current Value for an exact restart.
func (s *Stream) Snapshot() Snapshot {
	// An absent or unconstructed receiver has no valid numerical checkpoint schema.
	if s == nil || !s.initialized {
		// Strings are independently owned; no mutable accumulator memory escapes.
		return Snapshot{}
	}
	// Strings are independently owned; no mutable accumulator memory escapes.
	return Snapshot{
		Version:     snapshotVersion,
		Algorithm:   algorithm,
		Options:     s.options,
		Count:       s.count,
		Started:     s.started,
		VolumeSum:   decimal(&s.sums.volume, false),
		WeightedSum: decimal(&s.sums.weighted, s.sums.negative),
	}
}

// Restore validates and atomically replaces the exact state of a constructed stream.
// Rejected schema, grammar, phases, or magnitudes never change existing state.
//
// Implementation:
//   - Stage 1: validate receiver, version, algorithm, options, and phase count.
//   - Stage 2: parse bounded ASCII sums and verify both exact magnitude inequalities.
//   - Stage 3: reconstruct the cached output and assign the complete candidate.
//
// Behavior highlights:
//   - No input-dependent heap allocation is needed for parsing or validation.
//
// Inputs:
//   - state: Version=1 checkpoint with the same Options as the destination stream.
//
// Returns:
//   - nil on success; otherwise the original stream remains bitwise equivalent.
//
// Errors:
//   - ErrNilInput for nil receiver; ErrStateVersion for construction or state defects.
//
// Determinism:
//   - Resumed outputs equal uninterrupted Push outputs bit for bit.
//
// Complexity:
//   - Time O(D*L), Space O(L), D<=1300, L=67; fixed-size workspace.
//
// Notes:
//   - Structurally valid state need not be reachable from a genuine trade history.
//   - Count=MaxUint64 is restorable, but the next non-reset Push returns ErrOverflow.
//
// AI-Hints:
//   - Reject leading zeros, plus signs, whitespace, negative zero, and non-ASCII digits.
//   - Do not assign any field until all grammar and mathematical checks succeed.
func (s *Stream) Restore(state Snapshot) error {
	// Stage 1: only an already constructed destination can receive a checkpoint.
	if err := s.validateReceiver(); err != nil {
		return err
	}
	// Algorithm and full Options equality protect continuation semantics, not just field shape.
	if state.Version != snapshotVersion || state.Algorithm != algorithm || state.Options != s.options {
		return ErrStateVersion
	}
	// A successful first anchor contributes one accepted row, even with zero volume.
	if state.Started != (state.Count != 0) {
		return ErrStateVersion
	}
	// Stage 2: bounded parsers reject grammar/size defects without heap-sized integers.
	volume, _, ok := parseDecimal(state.VolumeSum, false, volumeBits)
	if !ok {
		return ErrStateVersion
	}
	weighted, negative, ok := parseDecimal(state.WeightedSum, true, weightedBits)
	if !ok {
		return ErrStateVersion
	}
	// Cross-field inequalities reject impossible magnitude combinations after individual parsing.
	if !validMagnitudes(&volume, &weighted, state.Count) {
		return ErrStateVersion
	}
	// Stage 3: construct all fields locally and derive the cache from the exact stored rational.
	next := Stream{
		initialized: true, options: s.options, count: state.Count, started: state.Started,
		sums: accumulator{volume: volume, weighted: weighted, negative: negative},
	}
	next.current = next.sums.value()
	// No live field changes until every invariant and the derived value are ready.
	*s = next
	return nil
}

// validMagnitudes checks exact snapshot bounds using only fixed-size products.
// The inequalities also enforce zero sums for Count=0 and W=0 whenever V=0.
//
// Implementation:
//   - Stage 1: form Count*MaxFloat64Units exactly.
//   - Stage 2: reject volume above that bound.
//   - Stage 3: form volume*MaxFloat64Units and compare the weighted magnitude.
//
// Behavior highlights:
//   - Negative weighted sums use their magnitude; the sign does not weaken the bound.
//
// Inputs:
//   - volume: parsed nonnegative magnitude of at most 2162 bits.
//   - weighted: parsed absolute weighted sum of at most 4260 bits.
//   - count: accepted row count from the candidate snapshot.
//
// Returns:
//   - bool: true exactly when both magnitude inequalities hold.
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
//   - MaxFloat64Units=(2^53-1)<<2045; intermediate products fit the fixed capacity.
//
// AI-Hints:
//   - These are structural bounds, not proof that any genuine market history produced the sums.
func validMagnitudes(volume, weighted *natural, count uint64) bool {
	// Umax factorization avoids constructing an overflowing floating-point limit.
	hi, lo := bits.Mul64(count, maxMantissa)
	volumeBound := term(hi, lo, maxUnitShift)
	// Reject denominator mass that exceeds even Count observations at MaxFloat64 volume.
	if volume.compare(&volumeBound) > 0 {
		return false
	}
	// Nonnegative weights bound the absolute numerator by maximum price times volume.
	product := volume.multiplyWord(maxMantissa)
	weightedBound := product.shifted(maxUnitShift)
	return weighted.compare(&weightedBound) <= 0
}

// decimal encodes a bounded exact magnitude in canonical ASCII decimal.
// It preserves all integer bits needed for continuation while keeping scratch storage fixed.
//
// Implementation:
//   - Stage 1: return the canonical literal zero for an empty magnitude.
//   - Stage 2: repeatedly divide a detached magnitude by ten, emitting remainders from right to left.
//   - Stage 3: prepend a minus sign only for a nonzero negative magnitude and return an owning string.
//
// Behavior highlights:
//   - The live accumulator is not divided or otherwise mutated.
//
// Inputs:
//   - value: valid checkpoint magnitude bounded by the documented volume/weighted bit limits.
//   - negative: desired sign for nonzero output.
//
// Returns:
//   - string: canonical 0|-?[1-9][0-9]*, at most maxDecimalBytes bytes.
//
// Errors:
//   - No error return; the documented private preconditions are established by the caller.
//
// Determinism:
//   - Input order and integer comparisons determine the result; no shared mutable state is read.
//
// Complexity:
//   - Time O(D*L), Space O(D+L), D<=1300 and L<=67; the returned nonzero string may allocate.
//
// Notes:
//   - Each division carries remainder<10 into bits.Div64, satisfying its precondition.
//
// AI-Hints:
//   - Do not serialize the rounded VWAP instead of these exact sums; that loses continuation information.
func decimal(value *natural, negative bool) string {
	// Zero is always unsigned and has exactly one canonical byte.
	if value.used == 0 {
		return "0"
	}
	// Destructive repeated division is confined to a detached fixed-size copy.
	work := *value
	var buffer [maxDecimalBytes]byte
	position := len(buffer)
	// Extract least-significant decimal digits while filling the buffer backward.
	for work.used != 0 {
		var remainder uint64
		// High-to-low long division carries a remainder below ten into every Div64 call.
		for i := work.used - 1; i >= 0; i-- {
			work.words[i], remainder = bits.Div64(remainder, work.words[i], decimalBase)
		}
		work.normalize()
		position--
		buffer[position] = decimalDigits[remainder]
	}
	// Only a nonzero negative magnitude receives a minus sign.
	if negative {
		position--
		buffer[position] = '-'
	}
	return string(buffer[position:])
}

// parseDecimal admits canonical decimal checkpoint integers into bounded value storage.
// Every byte and incremental magnitude is checked before any candidate can reach live state.
//
// Implementation:
//   - Stage 1: reject empty/oversized strings and illegal sign or zero spelling.
//   - Stage 2: scan ASCII digits left-to-right, appending each with exact integer arithmetic.
//   - Stage 3: reject a bit-length excess immediately; return the detached magnitude and sign on success.
//
// Behavior highlights:
//   - No heap-sized integer is constructed from attacker-controlled text.
//
// Inputs:
//   - text: candidate sum with at most 1300 bytes including sign.
//   - signed: whether a nonzero minus-prefixed integer is admissible.
//   - limit: volumeBits or weightedBits.
//
// Returns:
//   - natural, bool, bool: magnitude, negative flag, and success indicator.
//
// Errors:
//   - No error object; false is translated to ErrStateVersion by Restore.
//
// Determinism:
//   - Input order and integer comparisons determine the result; no shared mutable state is read.
//
// Complexity:
//   - Time O(D*L), Space O(L), D<=1300 and L<=67; no internal heap allocation.
//
// Notes:
//   - The first admitted digit is nonzero; bitLen is never called on an empty magnitude.
//   - A previous valid prefix needs <=4260 bits; multiplying by ten and adding a digit needs <=4264.
//
// AI-Hints:
//   - Reject plus signs, whitespace, Unicode digits, leading zeros, and negative zero instead of repairing them.
func parseDecimal(text string, signed bool, limit int) (natural, bool, bool) {
	var out natural
	// Reject unbounded or absent text before inspecting bytes or doing multiplication.
	if len(text) == 0 || len(text) > maxDecimalBytes {
		return out, false, false
	}
	start := 0
	// Only a minus prefix is meaningful, and only for the weighted numerator.
	negative := text[0] == '-'
	if negative {
		if !signed || len(text) == 1 {
			return out, false, false
		}
		start = 1
	}
	// This rejects negative zero and every longer spelling with a leading zero.
	if text[start] == '0' {
		return out, false, !negative && len(text) == 1
	}
	for i := start; i < len(text); i++ {
		digit := text[i]
		// Work byte-by-byte so Unicode digits and whitespace cannot be normalized silently.
		if digit < '0' || digit > '9' {
			return natural{}, false, false
		}
		out.appendDigit(uint64(digit - '0'))
		// At most four bits were added; the fixed buffer has room before this rejection.
		if out.bitLen() > limit {
			return natural{}, false, false
		}
	}
	return out, negative, true
}

// appendDigit appends one decimal digit to a bounded integer magnitude.
// The parser uses x=10*x+digit with exact carries, then immediately checks the bit limit.
//
// Implementation:
//   - Stage 1: initialize the incoming carry with the admitted ASCII digit value.
//   - Stage 2: multiply each used word by ten and add the carry low-to-high.
//   - Stage 3: append the final nonzero carry to the fixed array.
//
// Behavior highlights:
//   - The carry is at most nine after each limb, so no word multiplication result is truncated.
//
// Inputs:
//   - a: mutable parsed prefix requiring at most 4260 bits.
//   - digit: an integer in [0,9].
//
// Returns:
//   - No value; a becomes 10*a+digit in at most 4264 bits.
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
//   - The parser prevents a zero first digit and performs the post-step bound check.
//
// AI-Hints:
//   - Never expose this operation as an unbounded decimal parser or omit the post-digit limit.
func (a *natural) appendDigit(digit uint64) {
	// The digit starts the low-word carry in x*10+digit.
	carry := digit
	for i := 0; i < a.used; i++ {
		// Retain the high product half and add the incoming decimal carry exactly.
		hi, lo := bits.Mul64(a.words[i], decimalBase)
		var c uint64
		a.words[i], c = bits.Add64(lo, carry, 0)
		carry = hi + c
	}
	if carry != 0 {
		a.words[a.used] = carry
		a.used++
	}
}
