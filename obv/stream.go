// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package obv exposes transactional single-observation processing.
// This file owns mutation of the stream, but delegates mathematics to the canonical kernel.
package obv

// Push incorporates one valid OHLCV candle into the cumulative On-Balance Volume.
// The first accepted candle seeds previous close and emits exactly positive zero.
//
// Implementation:
//   - Stage 1: reject nil receiver or invalid OHLCV geometry before state access.
//   - Stage 2: compute the complete compensated candidate on a value copy.
//   - Stage 3: commit the candidate only after finite arithmetic and counter checks.
//
// Behavior highlights:
//   - Equal closes and zero volumes still update previous close and Count.
//   - Rejected inputs leave all recurrence fields, including compensation, unchanged.
//
// Inputs:
//   - bar: finite, positive OHLC prices and finite nonnegative base volume.
//
// Returns:
//   - value: current finite cumulative OBV, or zero on failure.
//   - valid: true on accepted observations, including the first candle.
//   - error: nil on success; otherwise no usable partial observation exists.
//
// Errors:
//   - ErrNilInput: nil stream receiver.
//   - ErrNonFinite, ErrDomain: input OHLCV validation.
//   - ErrOverflow: count exhaustion or nonrepresentable accumulated value.
//
// Determinism:
//   - Exact previous-Close comparison followed by one canonical transition.
//
// Complexity:
//   - Time O(1), Space O(1), zero heap allocations after New.
//
// Notes:
//   - Mutable Stream is goroutine-owned; synchronize concurrent calls externally.
//   - On feed gaps, call Reset explicitly before the next sequence.
//
// AI-Hints:
//   - Never assign previousClose before successful arithmetic: failures must roll back.
//   - Use Snapshot/Restore for failover, rather than copying only the displayed value.
func (s *Stream) Push(bar Bar) (value float64, valid bool, err error) {
	if s == nil {
		return 0, false, ErrNilInput
	}
	if err := validateBar(bar); err != nil {
		return 0, false, err
	}
	// The kernel receives a value copy; all overflow and counter checks happen
	// before this receiver can change any recurrence coordinate.
	candidate, value, err := nextOBV(*s, bar)
	if err != nil {
		return 0, false, err
	}
	// Commit after successful validation and arithmetic, never in stages.
	*s = candidate
	return value, true, nil
}

// Reset starts a new independent OBV epoch, discarding the previous close and sum.
// The next valid Push seeds price and returns positive zero, like a fresh Stream{}.
//
// Implementation:
//   - Stage 1: treat a nil receiver as a harmless no-op.
//   - Stage 2: replace all recurrence coordinates with their zero values.
//   - Stage 3: leave no reference to a prior observation or volume history.
//
// Behavior highlights:
//   - Reset does not emit an observation and does not process a volume.
//
// Inputs:
//   - s: owned stream pointer, optionally nil.
//
// Returns:
//   - Nothing; nil receiver is allowed.
//
// Errors:
//   - None.
//
// Determinism:
//   - Produces exactly the same all-zero state as a new Stream{}.
//
// Complexity:
//   - Time O(1), Space O(1); no allocation.
//
// AI-Hints:
//   - Reset after a discontinuity rather than incorrectly joining unrelated sessions.
func (s *Stream) Reset() {
	if s != nil {
		*s = Stream{}
	}
}
