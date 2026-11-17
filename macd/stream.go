// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package macd owns the single-observation transactional streaming facade.
// This file handles price validation, atomic publication and O(1) reset.
package macd

// Push consumes one strictly positive finite close and returns the next MACD point.
// It commits fast, slow, signal and count together only after all arithmetic succeeds.
//
// Implementation:
//   - Stage 1: reject nil/uninitialized receivers before checking the supplied price.
//   - Stage 2: validate the finite and strictly positive input domain.
//   - Stage 3: compute an independent candidate through the sole MACD kernel.
//   - Stage 4: if every component is finite, replace the old state in one assignment.
//
// Behavior highlights:
//   - Before signal readiness, Histogram is +0 with SignalValid=false.
//   - Every failed call returns Point{} and leaves the state bitwise unchanged.
//   - No dynamic history and no heap allocation in the hot path after construction.
//
// Inputs:
//   - price: one finite strictly positive price, in the stream's established cadence.
//
// Returns:
//   - Point: finite values plus phase masks for this exact accepted close.
//   - error: nil on success; otherwise Point{} and no state advancement.
//
// Errors:
//   - ErrNilInput: nil receiver.
//   - ErrStateVersion: receiver was never constructed through New.
//   - ErrNonFinite: close is NaN/Inf; ErrDomain: close is finite nonpositive.
//   - ErrOverflow: total accepted count or required floating arithmetic overflows.
//
// Determinism:
//   - Fixed fast/slow/line/signal/histogram order; batch calls use this same method.
//
// Complexity:
//   - Time O(1), additional Space O(1), strictly 0 heap allocs/op after New.
//
// Notes:
//   - Not safe for concurrent mutation or simultaneous Snapshot with mutation.
//   - A rejected tick does not represent an elapsed bar: upstream gap policy is separate.
//
// AI-Hints:
//   - Do not increment Count before signal/histogram overflow checks complete.
//   - If the feed has missing bars, choose a caller-side resampling policy instead
//     of silently pushing zeros or inventing gap-filling prices.
func (s *Stream) Push(price float64) (Point, error) {
	if s == nil {
		return Point{}, ErrNilInput
	}
	if !s.initialized {
		return Point{}, ErrStateVersion
	}
	if err := validateClose(price); err != nil {
		return Point{}, err
	}

	// nextStep receives a value copy of every recurrence field. No partial mutation
	// reaches this Stream even if its signal or histogram computation fails.
	candidate, point, err := nextStep(*s, price)
	if err != nil {
		return Point{}, err
	}
	*s = candidate
	return point, nil
}

// Reset discards every accepted observation but retains the frozen configuration.
// Its next successful Push acts exactly like the first Push on a fresh matching stream.
//
// Implementation:
//   - Stage 1: safely ignore nil and uninitialized receiver values.
//   - Stage 2: clear total observation count and all three EMA seed states.
//   - Stage 3: keep normalized Options and initialization status unchanged.
//
// Behavior highlights:
//   - Reset never allocates and never changes the stream's configured periods.
//
// Determinism:
//   - Replaying the same prices after Reset yields the same Point bits as a new stream.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Reset(nil) and Reset(&Stream{}) are intentional no-ops.
//   - Not safe for concurrent Push, Restore, Snapshot or other Reset operations.
//
// AI-Hints:
//   - Use Reset for a new independent history, not to represent a gap inside a feed.
func (s *Stream) Reset() {
	if s == nil || !s.initialized {
		return
	}
	s.count = 0
	s.fast = emaState{}
	s.slow = emaState{}
	s.signal = emaState{}
}
