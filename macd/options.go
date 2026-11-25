// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package macd defines the fixed EMA periods and versioned recurrence policy.
// This file centralizes defaults, state layout constants, and option normalization.
package macd

const (
	// algorithmID identifies the initial-window SMA seed and subsequent EMA recurrence.
	// Version 2 preserves the exact current sample at unit weight; v1 states are incompatible.
	// Any change to seed positions or recurrence arithmetic requires a new state version.
	algorithmID = "macd/sma-ema/v2"

	defaultFastPeriod   = 12
	defaultSlowPeriod   = 26
	defaultSignalPeriod = 9

	// snapshotStateSize is the fixed number of persisted independent EMA states.
	snapshotStateSize = 3
	stateFastIndex    = 0
	stateSlowIndex    = 1
	stateSignalIndex  = 2

	// encodedPeriodBytes is the size of each little-endian period in the config hash.
	encodedPeriodBytes = 8
)

// normalizeOptions replaces only zero-valued periods and checks the complete
// period/first-valid-index contract before any price or state is accessed.
//
// Implementation:
//   - Apply independent defaults to Fast, Slow, and Signal.
//   - Reject nonpositive periods and unordered Fast/Slow periods.
//   - Check (Slow-1)+(Signal-1) without constructing an overflowing sum.
//
// The resulting Options value is independent of the caller's original value.
// Time O(1), Space O(1). The caller must not bypass this gate when building a stream.
func normalizeOptions(opts Options) (Options, error) {
	if opts.Fast == 0 {
		opts.Fast = defaultFastPeriod
	}
	if opts.Slow == 0 {
		opts.Slow = defaultSlowPeriod
	}
	if opts.Signal == 0 {
		opts.Signal = defaultSignalPeriod
	}
	// Apply the same positive-domain admission used by the unsigned counter
	// encoder. Invalid signed values must not wrap into large seed limits.
	if periodCount(opts.Fast) == 0 || periodCount(opts.Slow) == 0 ||
		periodCount(opts.Signal) == 0 || opts.Fast >= opts.Slow {
		return Options{}, ErrInvalidOption
	}

	// The first signal index is Slow+Signal-2, expressed as two
	// nonnegative offsets so the bound check is itself overflow-safe.
	maxInt := int(^uint(0) >> 1)
	if opts.Signal-1 > maxInt-(opts.Slow-1) {
		return Options{}, ErrInvalidOption
	}
	return opts, nil
}

// periodCount maps a normalized positive EMA horizon into its unsigned counter domain.
// All public construction paths first enforce Fast>=1, Slow>Fast and Signal>=1;
// therefore a configured Stream can never contain a nonpositive period.
//
// Implementation:
//   - Stage 1: admit only positive signed periods into the conversion branch.
//   - Stage 2: convert without narrowing (int is at most 64 bits on supported Go targets).
//   - Stage 3: return zero as an invalid-period marker for internal misuse.
//
// Behavior highlights:
//   - No valid period is clamped or truncated, including periods above MaxUint32.
//   - A nonpositive internal value never wraps into a huge unsigned counter.
//
// Inputs:
//   - period: normalized, positive number of EMA observations.
//
// Returns:
//   - uint64: identical nonnegative magnitude, or zero for an invalid internal period.
//
// Determinism:
//   - Preserves the exact integer bytes used by the versioned snapshot Config hash.
//
// Complexity:
//   - Time O(1), Space O(1), no heap allocation.
//
// Notes:
//   - This helper is not an options validator. normalizeOptions rejects bad input
//     before constructing the Stream, which never uses the zero marker.
//
// AI-Hints:
//   - Do not convert an unvalidated signed period to uint64 directly: negative
//     values would wrap, corrupting seed limits or checkpoint identity.
//   - The guard is local so static range checkers can see conversion safety.
func periodCount(period int) uint64 {
	if period > 0 {
		return uint64(period)
	}
	return 0
}
