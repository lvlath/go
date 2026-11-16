// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package rsi defines the canonical Wilder smoothing-period policy.
// This file centralizes default selection and rejects periods whose warmup length is unrepresentable.
package rsi

// defaultPeriod is the number of price differences used by Wilder's standard RSI.
// The stream must therefore accept defaultPeriod+1 closing prices before its first valid output.
const defaultPeriod = 14

// normalizeOptions turns a caller-owned Options value into a valid unsigned transition count.
// The default is selected and bounds checked before conversion, so batch and stream agree.
//
// Implementation:
//   - Stage 1: replace the zero-value sentinel with defaultPeriod.
//   - Stage 2: reject negative values and MaxInt, whose required Period+1 overflows int.
//   - Stage 3: convert the validated period to the uint64 counter domain without allocating.
//
// Errors:
//   - ErrInvalidOption: a negative period or an unrepresentable P+1.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// AI-Hints:
//   - Do not restrict Period to an arbitrary technical-analysis convention such as 14 or 28.
//   - A large valid period requires more accepted prices, not an implicit smaller substitute.
func normalizeOptions(opts Options) (uint64, error) {
	period := opts.Period
	if period == 0 {
		period = defaultPeriod
	}

	// A complete seed consumes P changes and hence P+1 prices. Checking before
	// the addition also avoids a signed overflow in any future warmup helper.
	if period < 0 || period == int(^uint(0)>>1) {
		return 0, ErrInvalidOption
	}
	// The period is now strictly positive and at most MaxInt-1; converting to
	// uint64 cannot lose information on any supported Go integer width.
	return uint64(period), nil
}
