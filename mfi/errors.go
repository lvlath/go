// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mfi defines stable error categories for programmatic classification.
// Sentinels are compared with errors.Is, never by parsing human-readable text.
package mfi

import "errors"

var (
	// ErrNilInput reports a missing OHLCV column, row slice, or nil Stream receiver.
	// Compute tests all five slices after period validation; Push and Restore test receiver first.
	ErrNilInput = errors.New("mfi: nil input")
	// ErrEmptyInput reports a present but zero-length batch or row collection.
	// A short but nonempty history is successful warmup, not this error.
	ErrEmptyInput = errors.New("mfi: empty input")
	// ErrShape reports unequal lengths among five non-nil OHLCV columns.
	// No candle data are read until the shape gate succeeds.
	ErrShape = errors.New("mfi: mismatched input shape")
	// ErrInvalidOption reports a negative window period or one larger than MaxPeriod.
	// This error precedes input access and ring allocation.
	ErrInvalidOption = errors.New("mfi: invalid option")
	// ErrNonFinite reports any NaN or infinite field in the current OHLCV bar.
	// All five fields of the current bar are checked before its domain predicates.
	ErrNonFinite = errors.New("mfi: non-finite input")
	// ErrDomain reports finite OHLCV input outside the accepted physical domain.
	// No clipping, absolute-value repair or missing-data imputation is performed.
	ErrDomain = errors.New("mfi: invalid OHLCV domain")
	// ErrOverflow reports unrepresentable flow, compensated sum, typical price or Count.
	// Failed calculations do not expose a batch prefix or mutate Stream state.
	ErrOverflow = errors.New("mfi: arithmetic or count overflow")
	// ErrStateVersion reports uninitialized Stream state or an incompatible/corrupt Snapshot.
	// Restore rejects the entire candidate without modifying an existing stream.
	ErrStateVersion = errors.New("mfi: incompatible or corrupt state")
)
