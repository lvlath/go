// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ad exposes the allocation-free, single-owner streaming interface.
// This file validates each new Bar and commits only fully accepted candidate state.
package ad

// Push incorporates one validated candle into the Chaikin A/D line.
// The returned value already includes the bar's signed contribution, including bar zero.
//
// Implementation:
//   - Stage 1: reject a nil receiver, nonfinite candle fields, and invalid geometry.
//   - Stage 2: calculate an immutable candidate through the canonical nextAD kernel.
//   - Stage 3: replace the complete Stream state only on success and publish its total.
//
// Behavior highlights:
//   - A flat or zero-volume candle is valid, yields no contribution, and advances Count.
//   - Count overflow and arithmetic overflow are rejected before state publication.
//   - A failed Push has no usable value and leaves every scalar state coordinate unchanged.
//
// Inputs:
//   - bar: OHLCV by value, with finite positive prices, valid ranges and finite Volume>=0.
//
// Returns:
//   - value: finite compensated cumulative base volume, including this candle on success.
//   - valid: true exactly for an accepted candle; false on every error.
//   - error: nil on success, otherwise a stable sentinel.
//
// Errors:
//   - ErrNilInput: nil *Stream receiver.
//   - ErrNonFinite: invalid IEEE-754 candle field, checked before OHLCV domain predicates.
//   - ErrDomain: invalid price, volume, or geometry after successful finiteness validation.
//   - ErrOverflow: exhausted Count or unrepresentable compensated arithmetic.
//
// Determinism:
//   - Every successful Push applies exactly one nextAD step; no retries or implicit reset.
//   - Values are bitwise equal to the corresponding Compute/ComputeCandles prefix.
//
// Complexity:
//   - Time O(1), Space O(1); zero heap allocations for a warmed or fresh stream.
//
// Notes:
//   - Concurrent mutation or Snapshot during mutation of the same Stream is unsupported.
//   - Caller is responsible for tracking timestamps and calling Reset at grid gaps.
//
// AI-Hints:
//   - Do not skip the first candle or initialize its output with OBV's zero-seed rule.
//   - Treat a non-nil error as transaction failure; never consume its returned zero as data.
func (s *Stream) Push(bar Bar) (value float64, valid bool, err error) {
	// The receiver gate is strictly first: even a malformed candle must not
	// trigger dereference of a nil mutable engine.
	if s == nil {
		return 0, false, ErrNilInput
	}
	if err := validateBar(bar); err != nil {
		return 0, false, err
	}

	// Work on a scalar copy; nextAD may attempt a compensation-first rebase,
	// but no branch can mutate the receiver before full success.
	candidate, total, err := nextAD(*s, bar)
	if err != nil {
		return 0, false, err
	}

	*s = candidate
	return total, true, nil
}

// Reset begins a new A/D cumulative series at zero before the next candle.
// It clears count, high component and compensation together; nil Reset is a no-op.
//
// Implementation:
//   - Stage 1: accept a nil receiver without accessing any state.
//   - Stage 2: replace every scalar coordinate with the Stream zero value.
//   - Stage 3: allow the next Push to add the new first candle's own contribution.
//
// Behavior highlights:
//   - Reset never interpolates missing observations or changes historical Results.
//
// Inputs:
//   - s: stream to reset, or nil for an explicitly safe no-op.
//
// Returns:
//   - None; no error path or heap allocation.
//
// Errors:
//   - None, even when s is nil.
//
// Determinism:
//   - Exactly the same state as Stream{} and *New().
//
// Complexity:
//   - Time O(1), Space O(1), no allocations.
//
// Notes:
//   - Reset changes the accumulation origin; preserve that event in external provenance.
//   - Do not call Reset concurrently with Push, Restore, or Snapshot on one stream.
//
// AI-Hints:
//   - If price data has a known discontinuity, Reset explicitly rather than fabricating bars.
func (s *Stream) Reset() {
	if s != nil {
		*s = Stream{}
	}
}
