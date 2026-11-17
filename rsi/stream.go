// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package rsi defines a single-owner, zero-allocation streaming Wilder RSI interface.
// This file applies transactional publication around the canonical scalar transition kernel.
package rsi

// Stream owns one ordered sequence's private Wilder recurrence state.
// It must be initialized with New; a zero-value Stream is deliberately not configured.
//
// Implementation:
//   - Stage 1: New freezes the normalized period and its ABI fingerprint.
//   - Stage 2: Push delegates individual checked updates to stepWilder.
//   - Stage 3: Reset or Restore replaces the complete recurrence state without changing policy.
//
// Behavior highlights:
//   - Exactly one previously accepted close is used to compute the next delta.
//   - A failed Push/Restore leaves previous, averages, Count, and configuration unchanged.
//   - The first P price changes produce the seed; Count=P+1 is first valid.
//
// Inputs:
//   - Created with a value-owned Options configuration through New/NewPeriod/NewDefault.
//
// Returns:
//   - The pointer is caller-owned; internal state is never exported by reference.
//
// Errors:
//   - ErrStateVersion is returned by Push/Restore on uninitialized zero-value receivers.
//
// Determinism:
//   - Successful Push calls are the sole source of price ordering and state progression.
//
// Complexity:
//   - Time O(1), Space O(1) per Push; zero heap allocations after construction.
//
// Notes:
//   - A Stream is not safe for concurrent Push, Reset, Snapshot, or Restore operations.
//   - Detecting gaps, duplicated ticks, or timestamp order belongs to the caller.
//
// AI-Hints:
//   - Never copy a live Stream by value to coordinate goroutines; use Snapshot/Restore.
//   - Do not reset automatically on a rejected price: it must leave recurrence intact.
type Stream struct {
	period uint64      // Frozen unsigned smoothing period; zero means uninitialized.
	config string      // Versioned SHA-256 of algorithm identifier and normalized period.
	state  wilderState // Value-only previous price, means, and accepted-price count.
}

// Push consumes one price and publishes a new RSI observation only after completed warmup.
// Its transaction boundary rejects invalid prices or arithmetic without mutating existing state.
//
// Implementation:
//   - Stage 1: reject nil/uninitialized receivers and validate finite positive close.
//   - Stage 2: compute a complete candidate through the shared, side-effect-free stepWilder.
//   - Stage 3: commit the candidate and return its RSI/validity after all checks succeed.
//
// Behavior highlights:
//   - During warmup returns (0,false,nil); after P changes returns (RSI,true,nil).
//   - On error returns (0,false,error) and leaves the entire stream untouched.
//
// Inputs:
//   - price: strictly positive finite price; the caller controls its chronological order.
//
// Returns:
//   - float64/bool: new oscillator value and whether it is fully seeded.
//   - error: nil on an accepted price; otherwise the specific sentinel classification.
//
// Errors:
//   - ErrNilInput: nil receiver.
//   - ErrStateVersion: receiver was not created with New.
//   - ErrNonFinite, ErrDomain: invalid price before any state transition.
//   - ErrOverflow: an exhausted Count or unrepresentable numeric transition.
//
// Determinism:
//   - Successful Push calls execute stepWilder in exact caller invocation order.
//
// Complexity:
//   - Time O(1), Space O(1), exactly zero heap allocations per call after New.
//
// Notes:
//   - Not thread-safe; synchronize externally or dedicate a stream to one goroutine.
//   - Rejected updates do not alter Count or the baseline for the next valid price.
//
// AI-Hints:
//   - Do not move s.state=next above the kernel's error check.
//   - Do not cache the last returned RSI as state: averages alone determine continuation.
func (s *Stream) Push(price float64) (float64, bool, error) {
	if s == nil {
		return 0, false, ErrNilInput
	}
	if s.period == 0 {
		return 0, false, ErrStateVersion
	}
	if err := validatePrice(price); err != nil {
		return 0, false, err
	}

	// stepWilder works on a value copy; the destination is unchanged even when
	// a late numeric check rejects the candidate transition.
	next, value, valid, err := stepWilder(s.state, price, s.period)
	if err != nil {
		return 0, false, err
	}
	s.state = next
	return value, valid, nil
}

// Reset clears the accepted-price history but preserves the stream's configured period.
// The next successful Push becomes a new baseline and must complete a fresh warmup.
//
// Implementation:
//   - Stage 1: skip a nil receiver safely.
//   - Stage 2: replace the recurrence state with its zero value.
//   - Stage 3: retain period and configuration digest without rehashing or allocating.
//
// Behavior highlights:
//   - Reset(nil) is a no-op; Reset on Stream{} leaves it uninitialized.
//   - A fresh seed is mandatory after reset, even when RSI was previously valid.
//
// Inputs:
//   - s: an initialized stream or a nil/zero-value receiver.
//
// Returns:
//   - No return value; Snapshot reports Count=0 after initialized-stream Reset.
//
// Errors:
//   - None; nil and zero-value receivers are safe no-ops.
//
// Determinism:
//   - The post-reset state matches a newly created stream with the same normalized period.
//
// Complexity:
//   - Time O(1), Space O(1), with no allocation.
//
// Notes:
//   - Call explicitly when upstream detects a true feed gap; the package cannot see timestamps.
//
// AI-Hints:
//   - Never call Reset simply because a price was rejected: rejected Push already rolls back.
func (s *Stream) Reset() {
	if s != nil {
		s.state = wilderState{}
	}
}
