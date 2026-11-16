// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package rsi defines the detached v1 checkpoint transport for Wilder RSI streams.
// This file binds normalized configuration to SHA-256 and restores numeric state atomically.
package rsi

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
)

const (
	// algorithmID fixes the recurrence and checkpoint ABI; changing the layout requires a new version.
	algorithmID = "rsi/wilder/v1"

	// encodedPeriodBytes is the number of little-endian bytes encoding uint64(P) in Config.
	encodedPeriodBytes = 8

	// The indices define the public FloatState ABI and must not be reordered.
	snapshotPreviousIndex  = 0
	snapshotGainIndex      = 1
	snapshotLossIndex      = 2
	snapshotFloatStateSize = 3
)

// configHash fingerprints a normalized period together with the exact recurrence version.
// It hashes UTF-8 algorithmID followed by one uint64 period in little-endian byte order.
//
// Implementation:
//   - Stage 1: copy the version identifier into a fixed-size stack buffer.
//   - Stage 2: write the validated unsigned period as little-endian uint64.
//   - Stage 3: SHA-256 the buffer and encode the digest as 64 lowercase hex characters.
//
// Inputs:
//   - period: positive normalized uint64 from normalizeOptions, with representable Period+1.
//
// Returns:
//   - string: stable configuration fingerprint for the algorithm and period.
//
// Determinism:
//   - Exact UTF-8 bytes, little-endian encoding, and SHA-256 digest; no machine endianness leak.
//
// Complexity:
//   - Time O(1), Space O(1), including one output string allocation at construction.
//
// Notes:
//   - This hash identifies configuration; it is not a MAC for snapshot contents.
//
// AI-Hints:
//   - Do not include mutable Count or averages in Config; that changes the published ABI.
//   - Keep byte order fixed on all architectures and never use native-endian casts.
func configHash(period uint64) string {
	var payload [len(algorithmID) + encodedPeriodBytes]byte
	copy(payload[:], algorithmID)
	binary.LittleEndian.PutUint64(payload[len(algorithmID):], period)
	digest := sha256.Sum256(payload[:])
	return hex.EncodeToString(digest[:])
}

// Snapshot exports the complete, detached v1 state for reproducible continuation.
// It does not change the stream and can be safely mutated by the caller after return.
//
// Implementation:
//   - Stage 1: return Snapshot{} when called on nil/uninitialized receivers.
//   - Stage 2: copy Count and the immutable version/configuration metadata.
//   - Stage 3: allocate and populate a fresh three-element FloatState slice.
//
// Behavior highlights:
//   - FloatState=[previous, avgGain, avgLoss] and IntState has zero elements.
//   - A new or reset stream has Count=0 and three positive-zero FloatState values.
//
// Inputs:
//   - s: initialized stream, nil pointer, or uninitialized zero-value receiver.
//
// Returns:
//   - Snapshot: detached caller-owned scalar/slice representation or zero Snapshot{}.
//
// Errors:
//   - None; uninitialized receivers are represented by the zero snapshot.
//
// Determinism:
//   - Fixed field/index order with no sampling of time or external state.
//
// Complexity:
//   - Time O(1), Space O(1), including allocation of exactly three float64 elements.
//
// Notes:
//   - The Stream is not safe for concurrent Snapshot and Push/Restore/Reset.
//   - External encoding must preserve IEEE-754 bits; integrity requires an external MAC.
//
// AI-Hints:
//   - Never return a slice referencing private state; snapshots must be independent.
//   - The Config hash cannot detect tampering with Count or FloatState.
func (s *Stream) Snapshot() Snapshot {
	if s == nil || s.period == 0 {
		return Snapshot{}
	}

	return Snapshot{
		Algorithm: algorithmID,
		Config:    s.config,
		Count:     s.state.count,
		FloatState: []float64{
			s.state.previous,
			s.state.gain,
			s.state.loss,
		},
		IntState: []uint64{},
	}
}

// Restore atomically replaces the recurrence state from a compatible checkpoint.
// It rejects the entire candidate before mutation if its ABI, config, or numeric state is invalid.
//
// Implementation:
//   - Stage 1: reject nil/uninitialized destinations without reading state slices.
//   - Stage 2: validate the whole candidate's metadata, shape, domains, and phase consistency.
//   - Stage 3: copy the verified scalar values into a new state and commit in one assignment.
//
// Behavior highlights:
//   - The normalized period and configuration of the destination never change.
//   - Restoring a midseed Count continues with the same next seed divisor.
//   - Failure does not modify any of the destination's previous/mean/count fields.
//
// Inputs:
//   - state: caller-owned versioned Snapshot; slices are read but never retained.
//
// Returns:
//   - error: nil on a complete state replacement.
//
// Errors:
//   - ErrNilInput: called on a nil *Stream.
//   - ErrStateVersion: zero-value Stream or incompatible/malformed candidate Snapshot.
//
// Determinism:
//   - Restores exact binary64 fields without recomputing past prices or averages.
//
// Complexity:
//   - Time O(1), Space O(1), with no retained heap buffers.
//
// Notes:
//   - A schema-valid snapshot is not proof of an authentic or historically attainable state.
//   - Validate external checkpoint authentication before calling Restore.
//
// AI-Hints:
//   - Do not mutate s.state incrementally while checking slices: rejection must be atomic.
//   - Never auto-adapt a foreign Period or algorithm version into a local stream.
func (s *Stream) Restore(state Snapshot) error {
	if s == nil {
		return ErrNilInput
	}
	if s.period == 0 {
		return ErrStateVersion
	}
	if err := validateSnapshot(state, s.config); err != nil {
		return err
	}

	// Indexing is safe only after strict length validation. No caller-owned
	// backing array is retained, even when the checkpoint is reused elsewhere.
	s.state = wilderState{
		count:    state.Count,
		previous: state.FloatState[snapshotPreviousIndex],
		gain:     state.FloatState[snapshotGainIndex],
		loss:     state.FloatState[snapshotLossIndex],
	}
	return nil
}
