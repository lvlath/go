// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package atr provides exact True Range and Wilder Average True Range computations.
// This file implements checkpoint schema v1 and atomic validated restore.
package atr

import "math"

const (
	// snapshotVersion identifies the fixed public field/phase schema.
	snapshotVersion uint32 = 1
	// algorithm pins the one-round Wilder continuation law, not merely an indicator name.
	algorithm = "atr/wilder-rn/v1"
)

// Snapshot exports a detached deterministic continuation state.
// The seed is serialized exactly in subnormal units rather than rounded to a float.
//
// Implementation:
//   - Stage 1: Return Snapshot{} for nil or uninitialized receivers.
//   - Stage 2: Copy metadata, scalars, and counter.
//   - Stage 3: Encode the fixed seed integer as bounded canonical decimal ASCII.
//
// Behavior highlights:
//   - Mature snapshots have SeedSum="0"; PreviousClose retains its stored bits.
//
// Inputs:
//   - s: a single-owner stream or a nil/zero receiver.
//
// Returns:
//   - A detached Snapshot; the caller owns subsequent serialization.
//
// Errors:
//   - None.
//
// Determinism:
//   - Identical state gives identical fields and canonical SeedSum.
//
// Complexity:
//   - Time O(D*W), auxiliary Space O(D+W); D<=660, W=34.
//
// Notes:
//   - A nonzero returned decimal string may allocate; Snapshot is not the Push hot path.
//   - External persistence must preserve all float64 bits and the uint64 count.
//
// AI-Hints:
//   - A snapshot carries continuation data, not integrity protection or historical provenance.
func (s *Stream) Snapshot() Snapshot {
	// A zero snapshot explicitly denotes the absence of a constructed stream.
	if s == nil || s.period == 0 {
		return Snapshot{}
	}
	// Copy scalar bits and encode only the exact seed; no source cursor is inferred.
	return Snapshot{
		Version: snapshotVersion, Algorithm: algorithm, Options: s.options,
		Count: s.count, PreviousClose: s.previousClose, Current: s.current,
		SeedSum: s.seed.decimal(),
	}
}

// Restore replaces state only after complete checkpoint validation.
// The receiver must already have been constructed with exactly the same Options.
//
// Implementation:
//   - Stage 1: Validate receiver, schema, algorithm, configuration, and scalar domains.
//   - Stage 2: Parse bounded canonical SeedSum and verify phase and magnitude.
//   - Stage 3: Assign a fully validated candidate in one publication step.
//
// Behavior highlights:
//   - Invalid input leaves the receiver unchanged; parsing performs no heap allocation.
//
// Inputs:
//   - state: a snapshot with losslessly preserved scalar values and canonical SeedSum.
//
// Returns:
//   - nil on success; the receiver continues the saved accepted prefix.
//
// Errors:
//   - ErrNilInput: nil receiver.
//   - ErrStateVersion: uninitialized receiver, wrong metadata/options, invalid floats, phase, or sum.
//
// Determinism:
//   - No approximate recomputation; restored continuation matches uninterrupted Push.
//
// Complexity:
//   - Time O(D*W), Space O(W), D<=660, W=34; oversized strings rejected before parsing.
//
// Notes:
//   - Count MaxUint64 is restorable; its next Push reports ErrOverflow.
//   - Finite plausible tampering cannot be detected without externally authenticated storage.
//
// AI-Hints:
//   - Do not infer a missing previous close from Current or change Period during restore.
func (s *Stream) Restore(state Snapshot) error {
	// Receiver identity is validated before reading configuration.
	if s == nil {
		return ErrNilInput
	}
	// Restore never turns the zero Stream value into an implicitly configured instance.
	if s.period == 0 {
		return ErrStateVersion
	}
	// Schema, arithmetic law, and every option must match before any state is accepted.
	if state.Version != snapshotVersion || state.Algorithm != algorithm || state.Options != s.options {
		return ErrStateVersion
	}
	// Prior close is signed; mature ATR must be finite and nonnegative.
	if !finite(state.PreviousClose) || !finite(state.Current) || state.Current < 0 {
		return ErrStateVersion
	}
	// Parse and validate in independent fixed storage before touching the receiver.
	seed, ok := restoredSeed(state, s.period)
	if !ok {
		return ErrStateVersion
	}
	// A single assignment publishes a fully checked continuation state.
	*s = Stream{
		options: s.options, period: s.period, count: state.Count,
		previousClose: state.PreviousClose, current: state.Current, seed: seed,
	}
	return nil
}

// restoredSeed validates phase-specific continuation fields and reconstructs the exact seed.
// The magnitude gate checks structural admissibility, not the provenance of historical candles.
//
// Implementation:
//   - Stage 1: Require zero previous close in the empty phase.
//   - Stage 2: In maturity require the canonical empty seed; in warmup require Current=+0 and parse decimal.
//   - Stage 3: Compare the parsed integer with the exact bound Count*MaxFloat64Units.
//
// Behavior highlights:
//   - Count<P permits a nonzero seed; Count>=P requires SeedSum exactly zero.
//
// Inputs:
//   - state: metadata/scalar-validated snapshot; period: positive matching stream period.
//
// Returns:
//   - exactSum and bool: reconstructed seed when valid; callers must ignore it on false.
//
// Errors:
//   - No sentinel locally; Restore maps false to ErrStateVersion without mutation.
//
// Determinism:
//   - Fixed phase checks precede bounded decimal parsing and high-to-low integer comparison.
//
// Complexity:
//   - Time O(D*W), Space O(W), D<=660 and W=34; no heap allocation.
//
// Notes:
//   - Either sign of empty PreviousClose is admissible; warmup Current must be positive zero.
//
// AI-Hints:
//   - Do not try to prove a unique historical candle sequence from a summary snapshot.
func restoredSeed(state Snapshot, period uint64) (exactSum, bool) {
	// Empty history cannot name a nonzero previous close; either zero sign is allowed.
	if state.Count == 0 && state.PreviousClose != 0 {
		return exactSum{}, false
	}
	// A mature continuation needs only rounded Current and PreviousClose, not a seed.
	if state.Count >= period {
		return exactSum{}, state.SeedSum == "0"
	}
	// Warmup payload is specifically positive zero, unlike the mature scalar domain.
	if math.Float64bits(state.Current) != 0 {
		return exactSum{}, false
	}
	// Length and canonical ASCII grammar are checked before exact accumulation.
	seed, ok := parseSeed(state.SeedSum)
	if !ok {
		return exactSum{}, false
	}
	// Count times the largest finite TR bounds every admissible warmup sum.
	var bound exactSum
	bound.addProduct(math.MaxFloat64, state.Count)
	// Equality is valid; a larger seed is structurally impossible for this count.
	return seed, seed.atMost(&bound)
}
