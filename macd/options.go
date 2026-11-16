// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package macd defines the fixed EMA periods and versioned recurrence policy.
// This file centralizes defaults, state layout constants, and option normalization.
package macd

const (
	// algorithmID identifies the initial-window SMA seed and subsequent EMA recurrence.
	// Any change to seed positions or recurrence arithmetic requires a new state version.
	algorithmID = "macd/sma-ema/v1"

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
	if opts.Fast < 1 || opts.Slow < 1 || opts.Signal < 1 || opts.Fast >= opts.Slow {
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
