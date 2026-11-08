// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 katalvlaran

package series

import "errors"

// Package-owned sentinels are classified with errors.Is, not error text.
var (
	// ErrInvalidOption indicates an invalid cadence, lag, length, width, or stride.
	ErrInvalidOption = errors.New("series: invalid option")
	// ErrNilInput indicates a nil required input slice.
	ErrNilInput = errors.New("series: nil input")
	// ErrEmptyInput indicates a non-nil, zero-length input slice.
	ErrEmptyInput = errors.New("series: empty input")
	// ErrShape indicates unequal timestamp and value lengths.
	ErrShape = errors.New("series: shape mismatch")
	// ErrNonFinite indicates a NaN or infinite input observation.
	ErrNonFinite = errors.New("series: nonfinite observation")
	// ErrOrder indicates duplicate or decreasing timestamps.
	ErrOrder = errors.New("series: timestamps not strictly increasing")
	// ErrGap indicates a forbidden or off-grid interval.
	ErrGap = errors.New("series: invalid grid interval")
	// ErrDomain indicates a zero return denominator or nonpositive log input.
	ErrDomain = errors.New("series: invalid transform domain")
	// ErrOverflow indicates an unrepresentable signed timestamp delta or finite result.
	ErrOverflow = errors.New("series: arithmetic overflow")
)
