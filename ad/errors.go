// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ad declares the stable sentinel-first error protocol.
// This file separates structural, numeric-input, arithmetic, and checkpoint failures.
package ad

import "errors"

var (
	// ErrNilInput reports a missing OHLCV slice, nil candle collection, or nil Stream receiver.
	// It takes priority over length and value checks wherever a receiver or slice can be nil.
	ErrNilInput = errors.New("ad: nil input")

	// ErrEmptyInput reports a present, non-nil observation collection with length zero.
	// The algorithm requires at least one candle, whose contribution is returned at index zero.
	ErrEmptyInput = errors.New("ad: empty input")

	// ErrShape reports unequal lengths of the five non-nil OHLCV columns.
	// Shape is checked before indexing or inspecting any column element.
	ErrShape = errors.New("ad: invalid shape")

	// ErrNonFinite reports NaN or either infinity in a caller-supplied OHLCV field.
	// It is distinct from overflow produced by arithmetic on valid finite inputs.
	ErrNonFinite = errors.New("ad: non-finite input")

	// ErrDomain reports a nonpositive price, negative volume, or invalid candle geometry.
	// A zero spread is valid only when the entire OHLC candle is flat.
	ErrDomain = errors.New("ad: invalid OHLCV domain")

	// ErrOverflow reports an unrepresentable cumulative float64 value or Count exhaustion.
	// A failed transition leaves the caller's Stream state unchanged.
	ErrOverflow = errors.New("ad: arithmetic overflow")

	// ErrStateVersion reports an incompatible or invalid versioned checkpoint.
	// Restore rejects the entire candidate state without modifying its receiver.
	ErrStateVersion = errors.New("ad: invalid snapshot version or state")
)
