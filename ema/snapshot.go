// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ema provides exact affine exponential moving average smoothing.
// This file implements the bounded v1 continuation ABI and atomic restoration.
package ema

const (
	// stateVersion fixes the scalar/seed field schema; a change requires migration.
	stateVersion uint32 = 1
	// algorithm fixes both seed and recurrence rounding, not just field names.
	algorithm = "ema/affine-rn/v1"
	// zeroSeedText is the sole canonical empty/finished seed spelling.
	zeroSeedText = "0"
	// decimalDigits maps an exact divideWord remainder to a single ASCII byte.
	decimalDigits = "0123456789"
)

// decimalSeed formats exact signed seed units without mutating the live accumulator.
// It uses bounded scratch storage and allocates only the returned nonzero text.
//
// Implementation:
//   - Stage 1: return the shared canonical zero spelling for an empty magnitude.
//   - Stage 2: divide a by-value copy by ten, filling a fixed buffer backwards.
//   - Stage 3: prepend a negative sign when needed and detach the resulting string.
//
// Behavior highlights:
//   - No logarithm or approximate digit-count estimate controls buffer indexing.
//
// Inputs:
//   - sum: canonical bounded seed copied by value.
//
// Returns:
//   - string: immutable ASCII decimal, with no redundant zeros or plus sign.
//
// Errors:
//   - None; admitted sums require at most 651 digits plus an optional sign.
//
// Determinism:
//   - Exact Euclidean remainders define every output digit uniquely.
//
// Complexity:
//   - Time O(D*W), Space O(D+W), D<=660, W=34; nonzero output may allocate.
//
// Notes:
//   - The 660-byte scratch capacity exceeds the mathematical 652-byte bound.
//
// AI-Hints:
//   - Do not divide the stream accumulator in place while serializing it.
//   - Keep serialization outside the zero-allocation Push path.
func decimalSeed(sum exactSum) string {
	// Zero is cheap and never allocates a decimal buffer result.
	if sum.mag.bitLen() == 0 {
		return zeroSeedText
	}
	// Fixed stack scratch covers every admitted value, including its minus sign.
	var buffer [maxSeedText]byte
	// Writing backwards avoids reversing or reallocating the decimal string.
	start := len(buffer)
	// Each iteration consumes exactly one least-significant decimal digit.
	for sum.mag.bitLen() != 0 {
		// Mutate only the detached copy and retain the exact remainder in [0,9].
		digit := sum.mag.divideWord(decimalRadix)
		// The 652-byte mathematical bound leaves room in the 660-byte buffer.
		start--
		// Encode the remainder without locale-sensitive formatting.
		buffer[start] = decimalDigits[digit]
	}
	// A canonical zero was handled above, so a minus sign is meaningful here.
	if sum.negative {
		// The 652-byte mathematical bound leaves room in the 660-byte buffer.
		start--
		// The sign is part of the bounded spelling, not a separate serialized field.
		buffer[start] = '-'
	}
	// Detach the bytes from stack scratch before returning the immutable record.
	return string(buffer[start:])
}

// Snapshot exports the complete, detached continuation state of an initialized Stream.
// During SMA warmup its exact integer SeedSum preserves even canceled tiny addends.
//
// Implementation:
//   - Stage 1: return Snapshot{} for nil or uninitialized receivers.
//   - Stage 2: copy schema, configuration, Count, and Current.
//   - Stage 3: serialize the bounded seed integer, or publish "0" outside warmup.
//
// Behavior highlights:
//   - No observation history, rounded warmup mean, or external timestamp is retained.
//
// Inputs:
//   - s: an exclusively owned stream, including nil.
//
// Returns:
//   - Snapshot: detached scalar values and an immutable canonical SeedSum string.
//
// Errors:
//   - None.
//
// Determinism:
//   - Identical state yields identical fields and decimal text.
//
// Complexity:
//   - Time O(D*W), Space O(D+W), D<=660 and W=34 fixed bounds.
//
// Notes:
//   - Nonzero warmup serialization allocates its returned string; Push does not.
//   - Snapshot is not an authenticated record of historical inputs.
//
// AI-Hints:
//   - Keep Snapshot and serialization out of a zero-allocation Push benchmark.
//   - Persist the source sequence checkpoint with this state outside ema.
func (s *Stream) Snapshot() Snapshot {
	// Invalid receivers publish no plausible initialized checkpoint.
	if s == nil || !s.initialized {
		return Snapshot{}
	}
	// Snapshot fields are value copies; decimalSeed never retains the live limbs.
	return Snapshot{
		Version: stateVersion, Algorithm: algorithm, Options: s.options,
		Count: s.count, Current: s.current, SeedSum: decimalSeed(s.seed),
	}
}

// Restore atomically resumes an authenticated-by-caller snapshot with identical Options.
// It rejects malformed or incompatible state without allocation or partial mutation.
//
// Implementation:
//   - Stage 1: validate receiver, schema, options, canonical Current, and phase.
//   - Stage 2: parse at most 660 ASCII bytes and prove the exact count-based bound.
//   - Stage 3: assign the complete validated candidate in one publication step.
//
// Behavior highlights:
//   - A mature finite Count, including MaxUint64, is a valid continuation record.
//
// Inputs:
//   - state: detached v1 snapshot; its options must equal the receiver's options.
//
// Returns:
//   - nil on success; otherwise the receiver remains bitwise unchanged.
//
// Errors:
//   - ErrNilInput: nil receiver.
//   - ErrStateVersion: zero receiver, schema/configuration mismatch, or invalid fields.
//
// Determinism:
//   - Successful continuation is bit-identical to an uninterrupted accepted history.
//
// Complexity:
//   - Time O(D*W), Space O(W), D<=660 and W=34; no heap allocation.
//
// Notes:
//   - Count bounds are structural, not proof that every accepted state is reachable.
//   - Restore intentionally has no context because work is strictly bounded.
//
// AI-Hints:
//   - Never parse an unbounded decimal string with an allocating big integer.
//   - Never relax the phase invariant to accept nonzero Current during SMA warmup.
func (s *Stream) Restore(state Snapshot) error {
	// Receiver validation precedes reading any configuration field.
	if s == nil {
		return ErrNilInput
	}
	// A persisted state does not initialize a zero-value Stream implicitly.
	if !s.initialized {
		return ErrStateVersion
	}
	// Both arithmetic identity and complete configuration are immutable continuation keys.
	if state.Version != stateVersion || state.Algorithm != algorithm || state.Options != s.options {
		return ErrStateVersion
	}
	// Reject malformed scalar bits instead of silently normalizing persisted data.
	if !validateCurrent(state.Current) {
		return ErrStateVersion
	}
	// The phase is derived from the accepted count and the receiving configuration.
	warmup := s.options.Seed == SeedSMA && state.Count < s.period
	// Start a detached empty candidate; mature streams never retain seed residue.
	var seed exactSum
	// SMA warmup retains exact units but cannot expose a current measurement.
	if warmup {
		// Positive zero was already validated; any nonzero warmup current is invalid.
		if state.Current != 0 {
			return ErrStateVersion
		}
		// Parse at most 660 bytes into fixed storage after the schema/phase gates.
		parsed, ok := parseSeed(state.SeedSum)
		// The bound also proves Count=0 can restore only a zero seed.
		if !ok || !seedWithinBound(&parsed, state.Count) {
			return ErrStateVersion
		}
		// Retain the admitted exact units only in this private candidate.
		seed = parsed
	} else if state.SeedSum != zeroSeedText || (state.Count == 0 && state.Current != 0) {
		// Mature/First phases have no seed; an empty stream has no current value.
		return ErrStateVersion
	}
	// All external fields are now admitted; copy the complete receiver before publication.
	candidate := *s
	// Count includes accepted warmup observations and may equal MaxUint64.
	candidate.count = state.Count
	// The scalar is already finite, zero-canonical, and phase-consistent.
	candidate.current = state.Current
	// Replacing the full accumulator also clears any previous warmup residue.
	candidate.seed = seed
	// Single-owner publication: no failure path follows this assignment.
	*s = candidate
	return nil
}
