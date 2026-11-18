// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ad implements fixed-layout, transactional stream checkpoint transfer.
// This file validates version metadata and both compensated coordinates before commit.
package ad

const (
	// snapshotAlgorithm identifies the exact Chaikin-Neumaier recurrence and snapshot layout.
	// A future change to the accumulator or FloatState ABI requires a new identifier.
	snapshotAlgorithm = "ad/chaikin-neumaier/v1"

	// snapshotConfig is SHA-256(lowercase hex) of the ASCII snapshotAlgorithm string.
	// This compile-time fingerprint checks compatibility, not authenticity or payload integrity.
	snapshotConfig = "df3132e9b6a35bdbc7d3567d9f8e3c4675dc6736013e4a408c6e150b5441a2df"

	// snapshotFloatLength fixes the exact number of stored binary64 state coordinates.
	snapshotFloatLength = 2

	// snapshotSumIndex is the high Neumaier component within FloatState.
	snapshotSumIndex = 0

	// snapshotCompensationIndex is the low Neumaier correction within FloatState.
	snapshotCompensationIndex = 1
)

// Snapshot exports a detached, versioned checkpoint for the current A/D recurrence.
// Both high and low accumulator components are needed for bit-identical continuation.
//
// Implementation:
//   - Stage 1: return a zero Snapshot{} for a nil receiver without accessing state.
//   - Stage 2: copy Count and the two scalar coordinates into their fixed ABI slots.
//   - Stage 3: attach constant algorithm/config metadata and an empty IntState.
//
// Behavior highlights:
//   - FloatState has length exactly two in [sum, compensation] order.
//   - IntState has length zero; A/D never needs a previous closing price.
//   - This is a structural checkpoint, not a timestamp or provenance record.
//
// Inputs:
//   - s: caller-owned Stream, or nil to request zero Snapshot{}.
//
// Returns:
//   - Snapshot: scalar metadata and detached FloatState, safe to retain or serialize.
//
// Errors:
//   - None; a nil receiver returns the zero value rather than an error.
//
// Determinism:
//   - Fixed algorithm ID, digest, field order, and raw numeric coordinates.
//
// Complexity:
//   - Time O(1), Space O(1), including a fresh two-element float slice.
//
// Notes:
//   - Do not call Snapshot concurrently with Push, Reset, or Restore on this stream.
//   - Authenticate and atomically persist a checkpoint in the application layer.
//
// AI-Hints:
//   - Do not serialize only the rounded displayed total: it discards future corrections.
//   - Hashing Algorithm identifies schema compatibility, not a secure MAC of FloatState.
func (s *Stream) Snapshot() Snapshot {
	if s == nil {
		return Snapshot{}
	}

	// Construct new slice storage; Stream itself owns no backing arrays.
	// Callers may mutate this snapshot without mutating the source engine.
	return Snapshot{
		Algorithm:  snapshotAlgorithm,
		Config:     snapshotConfig,
		Count:      s.count,
		FloatState: []float64{s.sum, s.compensation},
		IntState:   make([]uint64, 0),
	}
}

// Restore installs a compatible, numerically valid A/D checkpoint transactionally.
// It never retains caller-owned slice storage or modifies a receiver on failure.
//
// Implementation:
//   - Stage 1: reject a nil receiver, mismatched identity/config and invalid slice lengths.
//   - Stage 2: check both coordinates and their effective sum for finiteness.
//   - Stage 3: enforce the zero-count invariant and install one complete scalar state.
//
// Behavior highlights:
//   - Count zero requires both state coordinates to be numerically zero.
//   - Count greater than zero allows either sign and any finite representable total.
//   - Count MaxUint64 is restorable but its next Push will return ErrOverflow.
//   - A zero-count candidate is canonicalized to the same +0 state as Stream{}.
//
// Inputs:
//   - state: read-only Snapshot value with Algorithm, Config and exactly two FloatState entries.
//
// Returns:
//   - error: nil only after all stream state has been replaced successfully.
//
// Errors:
//   - ErrNilInput: nil *Stream receiver, checked before reading state.
//   - ErrStateVersion: unsupported ID/config/layout or invalid numeric/count invariants.
//
// Determinism:
//   - Checks fields in fixed order and copies accepted binary64 values without recomputation.
//
// Complexity:
//   - Time O(1), Space O(1) scalar candidate storage; no input slices retained.
//
// Notes:
//   - Config is a public compatibility digest; maliciously forged snapshots are not authenticated.
//   - A successful Restore replaces, rather than merges with, previous history.
//   - Concurrent use of the same Stream with Restore is unsupported.
//
// AI-Hints:
//   - Validate the effective sum, not only each independently finite coordinate.
//   - Never mutate s until every version, shape and numeric check has passed.
func (s *Stream) Restore(state Snapshot) error {
	if s == nil {
		return ErrNilInput
	}

	// Version and lengths form the wire-format boundary. Do not index the
	// float slice until its exact declared size has been checked.
	if state.Algorithm != snapshotAlgorithm || state.Config != snapshotConfig ||
		len(state.FloatState) != snapshotFloatLength || len(state.IntState) != 0 {
		return ErrStateVersion
	}

	sum := state.FloatState[snapshotSumIndex]
	compensation := state.FloatState[snapshotCompensationIndex]
	if !finite(sum) || !finite(compensation) || !finite(sum+compensation) {
		return ErrStateVersion
	}

	// An unseeded stream cannot contain either historical volume or retained
	// correction; reset it to canonical positive zeros on valid zero-count restore.
	if state.Count == 0 {
		if sum != 0 || compensation != 0 {
			return ErrStateVersion
		}
		*s = Stream{}
		return nil
	}

	// Only local scalar values are installed. The Snapshot's backing arrays are
	// never retained; mutation of either side cannot affect the other.
	*s = Stream{sum: sum, compensation: compensation, count: state.Count}
	return nil
}
