// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gbm declares one stable sentinel per distinct public failure class.
// Use errors.Is; human-readable strings are diagnostics, never branch keys.
package gbm

import "errors"

var (
	// ErrInvalidOption rejects a nil context, unrecognized TimeUnit, or a
	// mismatch between grid and model units before any random draw.
	ErrInvalidOption = errors.New("gbm: invalid option")

	// ErrNilInput rejects an absent grid or RNG. A typed-nil RNG is detected
	// when it supports the optional IsNil() bool protocol, without reflection.
	ErrNilInput = errors.New("gbm: nil input")

	// ErrEmptyInput rejects an allocated-but-empty timestamp grid; it is
	// distinct from a nil grid, which receives ErrNilInput.
	ErrEmptyInput = errors.New("gbm: empty input")

	// ErrOrder rejects a nonzero initial timestamp, duplicates, or decreasing
	// observation times; the simulator never silently sorts or resamples data.
	ErrOrder = errors.New("gbm: invalid time order")

	// ErrDomain rejects nonpositive initial prices or negative volatility.
	// Negative finite drift is allowed because GBM may model depreciation.
	ErrDomain = errors.New("gbm: invalid parameter domain")

	// ErrNonFinite rejects NaN or +/-Inf in supplied parameters, timestamp
	// grids, or a generator-produced Gaussian draw; no bad draw is published.
	ErrNonFinite = errors.New("gbm: non-finite value")

	// ErrOverflow means the newly computed price has no positive finite
	// float64 representation; completed earlier points remain usable as a prefix.
	ErrOverflow = errors.New("gbm: unrepresentable price")
)
