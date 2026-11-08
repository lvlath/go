// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package series classifies invalid inputs and unrepresentable arithmetic.
// This file owns the public sentinel identities used by errors.Is callers.

package series

import "errors"

// These error variables are stable package-owned sentinel identities.
// Callers classify failures with errors.Is; error strings are diagnostic only.
var (
	// ErrInvalidOption identifies an invalid cadence, lag, length, width, or stride.
	// Option checks precede inspecting any input observation.
	ErrInvalidOption = errors.New("series: invalid option")

	// ErrNilInput identifies a required timestamp or observation slice that is nil.
	// A nil input is distinct from a present but empty slice.
	ErrNilInput = errors.New("series: nil input")

	// ErrEmptyInput identifies a non-nil, zero-length required input slice.
	// Validate checks equal lengths before declaring equally empty inputs invalid.
	ErrEmptyInput = errors.New("series: empty input")

	// ErrShape identifies different lengths for timestamp and observation slices.
	// Even an empty/nonempty mismatch is classified before the empty-input check.
	ErrShape = errors.New("series: shape mismatch")

	// ErrNonFinite identifies a NaN or infinite observation anywhere in an input.
	// Validation includes positions that produce only invalid warmup output.
	ErrNonFinite = errors.New("series: nonfinite observation")

	// ErrOrder identifies equal or decreasing adjacent signed timestamps.
	// Timestamp order is checked in caller-supplied order without reordering.
	ErrOrder = errors.New("series: timestamps not strictly increasing")

	// ErrGap identifies a cadence interval that is off-grid or contains
	// missing slots when GridOptions.AllowGaps is false.
	ErrGap = errors.New("series: invalid grid interval")

	// ErrDomain identifies a used zero denominator in SimpleReturns or a
	// nonpositive observation in LogReturns; valid negative simple inputs are allowed.
	ErrDomain = errors.New("series: invalid transform domain")

	// ErrOverflow identifies an unrepresentable signed timestamp delta,
	// a nonfinite finite-input arithmetic result, or a slice-size overflow.
	ErrOverflow = errors.New("series: arithmetic overflow")
)
