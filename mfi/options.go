// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mfi defines the MFI window configuration and its allocation ceiling.
// This file normalizes the single mathematical option before any ring allocation.
package mfi

const (
	// DefaultPeriod selects fourteen transitions when Options.Period is zero.
	// This includes zero-volume and unchanged-typical-price transitions.
	DefaultPeriod = 14

	// MaxPeriod limits a stream to 1,048,576 transition slots per sign.
	// Two float64 rings at this ceiling occupy 16 MiB, excluding bookkeeping.
	MaxPeriod = 1 << 20
)

// normalizedPeriod freezes the one effective rolling-window configuration.
// The caller never allocates a ring until this validation succeeds.
//
// Implementation:
//   - Stage 1: reject negative and resource-ceiling-exceeding Period values.
//   - Stage 2: resolve zero to DefaultPeriod without inspecting data.
//   - Stage 3: return the bounded effective period before any ring allocation.
//
// Behavior highlights:
//   - A negative period is never interpreted as an unbounded window.
//
// Inputs:
//   - opts: caller-owned value; zero selects DefaultPeriod.
//
// Returns:
//   - int: normalized period in [1, MaxPeriod].
//   - error: ErrInvalidOption for negative or oversized periods.
//
// Determinism:
//   - Identical Options values always produce the same effective period.
//
// Complexity:
//   - Time O(1), Space O(1); no allocation.
//
// Notes:
//   - Callers must not allocate rings until this function succeeds.
//
// AI-Hints:
//   - Do not change the meaning of zero when introducing more convenience wrappers.
func normalizedPeriod(opts Options) (int, error) {
	if opts.Period < 0 || opts.Period > MaxPeriod {
		return 0, ErrInvalidOption
	}
	if opts.Period == 0 {
		return DefaultPeriod, nil
	}
	return opts.Period, nil
}
