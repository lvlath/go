// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package macd defines reproducible state export and transactional recovery.
// This file owns the versioned configuration digest and phase-invariant validation.
package macd

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
)

// configHash pins normalized periods and algorithm version to one lowercase digest.
// Hashing uses ASCII version bytes followed by three LE64 period encodings.
// Time O(1), Space O(1), deterministic across supported Go platforms.
// This is an identity digest, not a MAC authenticating the mutable recurrence fields.
func configHash(opts Options) string {
	var data [len(algorithmID) + snapshotStateSize*encodedPeriodBytes]byte
	copy(data[:], algorithmID)
	binary.LittleEndian.PutUint64(data[len(algorithmID):], periodCount(opts.Fast))
	binary.LittleEndian.PutUint64(data[len(algorithmID)+encodedPeriodBytes:], periodCount(opts.Slow))
	binary.LittleEndian.PutUint64(data[len(algorithmID)+2*encodedPeriodBytes:], periodCount(opts.Signal))
	sum := sha256.Sum256(data[:])
	return hex.EncodeToString(sum[:])
}

// Snapshot exports the complete, detached recurrence state for deterministic replay.
// It preserves EMA values even during warmup, so continuation never needs price history.
//
// Implementation:
//   - Stage 1: check for nil or uninitialized receivers.
//   - Stage 2: bind current normalized periods to the pinned SHA-256 config digest.
//   - Stage 3: allocate independent three-element arrays and copy all EMA states.
//
// Behavior highlights:
//   - Nil or uninitialized receiver yields exactly Snapshot{} with nil slices.
//   - Counts reflect only successfully committed Push calls.
//
// Returns:
//   - Snapshot: new FloatState and IntState slices, detached from the live Stream.
//
// Determinism:
//   - Stable order: fast, slow, signal; fixed algorithm ID and LE64 hash encoding.
//
// Complexity:
//   - Time O(1), Space O(1), including two small allocations.
//
// Notes:
//   - Calling Snapshot concurrently with Stream mutation is unsupported.
//   - Config does not authenticate Count, means or counters; use authenticated storage
//     and a separate record of input history identity for operational failover.
//
// AI-Hints:
//   - Do not persist only the last output Point: phase means and counts are essential.
//   - Restore always checks the target stream's periods against the snapshot hash.
func (s *Stream) Snapshot() Snapshot {
	if s == nil || !s.initialized {
		return Snapshot{}
	}
	return Snapshot{
		Algorithm: algorithmID,
		Config:    configHash(s.opts),
		Count:     s.count,
		FloatState: []float64{
			s.fast.mean,
			s.slow.mean,
			s.signal.mean,
		},
		IntState: []uint64{
			s.fast.seen,
			s.slow.seen,
			s.signal.seen,
		},
	}
}

// smaller computes the saturated count for one fixed-width seed window.
// The input counts are unsigned, and the operation cannot underflow.
// Time O(1), Space O(1).
func smaller(x, y uint64) uint64 {
	if x < y {
		return x
	}
	return y
}

// Restore validates an entire snapshot and only then commits its recurrence fields.
// A rejected version, hash, state shape, count relationship or numeric value changes nothing.
//
// Implementation:
//   - Stage 1: reject nil/uninitialized receivers before dereferencing state.
//   - Stage 2: compare schema, algorithm ID, pinned config digest and array lengths.
//   - Stage 3: derive the three expected warmup counters without uint64 underflow.
//   - Stage 4: require finite means, positive seeded price means and zero inactive means.
//   - Stage 5: assemble a complete candidate and replace the live Stream atomically.
//
// Behavior highlights:
//   - Fast/Slow counters equal min(Count, period).
//   - Signal counter equals min(max(Count-Slow+1, 0), Signal).
//   - Signed negative signal means are valid; the price means remain positive.
//   - Inactive means must use canonical positive zero, including its float64 sign bit.
//   - Only macd/sma-ema/v2 is accepted; v1 must be rebuilt from the original closes.
//
// Inputs:
//   - state: caller-owned Snapshot with fully populated, read-only slices during call.
//
// Returns:
//   - error: nil only after all state fields have committed together.
//
// Errors:
//   - ErrNilInput: nil receiver.
//   - ErrStateVersion: zero Stream or any inconsistent/incompatible snapshot field.
//
// Determinism:
//   - Check version and lengths, then counts, then means in fixed fast/slow/signal order.
//   - The candidate's normalized periods and initialized flag are retained unchanged.
//
// Complexity:
//   - Time O(1), Space O(1); digest computation and three-field validation are bounded.
//
// Notes:
//   - Snapshot state is not cryptographically authenticated. A structurally plausible
//     but fabricated mean can pass Restore; responsibility for trust is external.
//   - Original states and supplied slices remain independently owned after success.
//   - Changing only Algorithm/Config cannot repair old numeric state; replay the
//     exact accepted prefix with New using the same normalized options.
//   - Concurrent Restore and any other Stream access is unsupported.
//
// AI-Hints:
//   - Do not treat SHA-256 config identity as proof of data authenticity or history.
//   - Do not validate signal means with the positive-price domain gate.
//   - Do not publish a partially validated state and then discover a bad late field.
func (s *Stream) Restore(state Snapshot) error {
	if s == nil {
		return ErrNilInput
	}
	if !s.initialized {
		return ErrStateVersion
	}
	if state.Algorithm != algorithmID || state.Config != configHash(s.opts) ||
		len(state.FloatState) != snapshotStateSize || len(state.IntState) != snapshotStateSize {
		return ErrStateVersion
	}

	// A signal observation exists at close number Slow and later, so subtraction
	// is performed only after confirming Count>=Slow.
	count := state.Count
	fastExpected := smaller(count, periodCount(s.opts.Fast))
	slowExpected := smaller(count, periodCount(s.opts.Slow))
	var signalExpected uint64
	if count >= periodCount(s.opts.Slow) {
		signalExpected = smaller(count-periodCount(s.opts.Slow)+1, periodCount(s.opts.Signal))
	}
	if state.IntState[stateFastIndex] != fastExpected ||
		state.IntState[stateSlowIndex] != slowExpected ||
		state.IntState[stateSignalIndex] != signalExpected {
		return ErrStateVersion
	}

	// Reject infinities and NaNs in ALL phases; price means are strictly positive
	// whenever their seed has received a price, whereas signal means are signed.
	for _, mean := range state.FloatState {
		if !finite(mean) {
			return ErrStateVersion
		}
	}
	if (fastExpected > 0 && state.FloatState[stateFastIndex] <= 0) ||
		(slowExpected > 0 && state.FloatState[stateSlowIndex] <= 0) {
		return ErrStateVersion
	}

	// Before the first seed input, canonical state contains positive zero exactly.
	// A negative-zero snapshot is rejected to prevent multiple encodings of the same
	// uninitialized phase from appearing in replicated checkpoint logs.
	if (fastExpected == 0 && math.Float64bits(state.FloatState[stateFastIndex]) != 0) ||
		(slowExpected == 0 && math.Float64bits(state.FloatState[stateSlowIndex]) != 0) ||
		(signalExpected == 0 && math.Float64bits(state.FloatState[stateSignalIndex]) != 0) {
		return ErrStateVersion
	}

	candidate := *s
	candidate.count = count
	candidate.fast = emaState{mean: state.FloatState[stateFastIndex], seen: fastExpected}
	candidate.slow = emaState{mean: state.FloatState[stateSlowIndex], seen: slowExpected}
	candidate.signal = emaState{mean: state.FloatState[stateSignalIndex], seen: signalExpected}
	*s = candidate
	return nil
}
