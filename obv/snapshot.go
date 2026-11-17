// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package obv exposes versioned checkpoint transfer for a single cumulative stream.
// This file validates the complete numeric schema before replacing any live state.
package obv

import (
	"crypto/sha256"
	"encoding/hex"
)

const (
	// snapshotAlgorithm fixes the recurrence and FloatState wire layout.
	// Changing compensation semantics requires a new algorithm identifier.
	snapshotAlgorithm = "obv/neumaier/v1"

	// snapshotFloatLength is the exact number of binary64 recurrence coordinates.
	snapshotFloatLength = 3

	// snapshotPreviousClose identifies the last accepted candle's Close.
	snapshotPreviousClose = 0

	// snapshotSum identifies the high component of the compensated volume total.
	snapshotSum = 1

	// snapshotCompensation identifies the accumulated low-order rounding residual.
	snapshotCompensation = 2
)

// configHash returns the lowercase SHA-256 digest of the fixed algorithm identity.
// It detects accidental configuration mismatch; it is not a message authentication code.
//
// Implementation:
//   - Stage 1: hash the ASCII identity of the supported recurrence and snapshot ABI.
//   - Stage 2: encode the 32-byte digest as 64 lowercase hexadecimal characters.
//
// Returns:
//   - string: deterministic, process-independent compatibility fingerprint.
//
// Determinism:
//   - SHA-256 consumes exactly the ASCII bytes of snapshotAlgorithm.
//
// Complexity:
//   - Time O(1), Space O(1) bounded-size scratch and digest string.
//
// AI-Hints:
//   - An attacker can recalculate this public digest; sign or MAC serialized state externally.
func configHash() string {
	digest := sha256.Sum256([]byte(snapshotAlgorithm))
	return hex.EncodeToString(digest[:])
}

// Snapshot exports the exact high and low accumulator components for deterministic failover.
// Unlike the displayed OBV value alone, these coordinates preserve future cancellation.
//
// Implementation:
//   - Stage 1: return a zero Snapshot{} when the receiver is nil.
//   - Stage 2: encode previous close, sum and compensation in the fixed v1 order.
//   - Stage 3: return fresh, detached slices with version/config metadata.
//
// Behavior highlights:
//   - Count zero remains a versioned checkpoint with three zero float coordinates.
//   - IntState is a newly allocated zero-length slice; it carries no hidden cursor.
//
// Inputs:
//   - s: stream owned by the caller; nil is accepted.
//
// Returns:
//   - Snapshot: detached checkpoint, or zero Snapshot{} for nil receiver.
//
// Errors:
//   - None.
//
// Determinism:
//   - Exactly [previousClose, sum, compensation], with no normalization.
//
// Complexity:
//   - Time O(1), Space O(1) newly allocated snapshot storage.
//
// Notes:
//   - The digest identifies compatibility, not payload authenticity or integrity.
//   - Do not snapshot concurrently with mutation of the same stream.
//
// AI-Hints:
//   - Never serialize only sum+compensation; continuation needs both operands.
func (s *Stream) Snapshot() Snapshot {
	if s == nil {
		return Snapshot{}
	}
	// Allocate fresh arrays on every export, including the empty IntState.
	// Reusing internal buffers would make independent failover unsafe.
	return Snapshot{
		Algorithm: snapshotAlgorithm,
		Config:    configHash(),
		Count:     s.count,
		FloatState: []float64{
			s.previousClose, s.sum, s.compensation,
		},
		IntState: make([]uint64, 0),
	}
}

// Restore installs a complete compatible recurrence checkpoint transactionally.
// Invalid metadata or numeric coordinates leave all currently accepted observations intact.
//
// Implementation:
//   - Stage 1: validate receiver, version fingerprint and exact wire slice lengths.
//   - Stage 2: validate finite coordinates, seeded state, and effective representability.
//   - Stage 3: assign a fully validated scalar candidate without retaining input slices.
//
// Behavior highlights:
//   - Count zero requires three zero coordinates; Count one has no accumulated volume.
//   - Count MaxUint64 is valid on restore, but the next Push returns ErrOverflow.
//
// Inputs:
//   - state: caller-owned checkpoint; Algorithm and Config must match v1 exactly.
//
// Returns:
//   - error: nil only after the receiver has been replaced successfully.
//
// Errors:
//   - ErrNilInput: nil stream receiver, checked before inspecting the checkpoint.
//   - ErrStateVersion: mismatched schema/config, invalid count relations or numeric state.
//
// Determinism:
//   - Checks version/length, then numeric state, then commits once.
//
// Complexity:
//   - Time O(1), Space O(1); no caller-owned memory is retained.
//
// Notes:
//   - Compatibility digest is not protection against fabricated valid snapshots.
//   - The caller must authenticate serialized checkpoints before Restore.
//
// AI-Hints:
//   - Do not overwrite the receiver before all checks succeed.
//   - Validate sum and compensation separately, not only their finite total.
func (s *Stream) Restore(state Snapshot) error {
	if s == nil {
		return ErrNilInput
	}
	// Check the public ABI fingerprint and both slice lengths first. No array
	// is indexed or scalar state committed before this structural gate.
	if state.Algorithm != snapshotAlgorithm || state.Config != configHash() ||
		len(state.FloatState) != snapshotFloatLength || len(state.IntState) != 0 {
		return ErrStateVersion
	}

	floats := state.FloatState
	previous := floats[snapshotPreviousClose]
	sum := floats[snapshotSum]
	compensation := floats[snapshotCompensation]
	// A snapshot cannot import nonfinite coordinates or an unrepresentable
	// effective value, even when its metadata and array lengths are plausible.
	if !finite(previous) || !finite(sum) || !finite(compensation) || !finite(sum+compensation) {
		return ErrStateVersion
	}
	// Count 0 has no observed price, while Count 1 has a baseline Close but
	// no signed-volume contribution. Higher counts may have either sum sign.
	if state.Count == 0 && (previous != 0 || sum != 0 || compensation != 0) {
		return ErrStateVersion
	}
	if state.Count > 0 && previous <= 0 {
		return ErrStateVersion
	}
	if state.Count == 1 && (sum != 0 || compensation != 0) {
		return ErrStateVersion
	}

	// Only scalar fields are retained. This single final write preserves the
	// previous receiver byte-for-byte if any earlier validation rejects state.
	*s = Stream{previousClose: previous, sum: sum, compensation: compensation, count: state.Count}
	return nil
}
