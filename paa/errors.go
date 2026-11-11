// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
//
// Package paa defines stable, package-owned error identities for its numeric
// and geometric contracts. This file keeps caller-visible classifications
// independent of diagnostic wording and internal implementation choices.

package paa

import "errors"

var (
	// ErrInvalidOption identifies a segment-count or range-bound violation.
	// Transform rejects segments <= 0 before inspecting values and segments > n
	// after nil/empty input checks. TransformRange also uses this sentinel for
	// invalid half-open window bounds and excessive window segment counts.
	// Classify with errors.Is; error message text is not a stable protocol.
	ErrInvalidOption = errors.New("paa: invalid segment count")

	// ErrNilInput identifies an absent []float64 input, not an empty series.
	// It follows the input-independent segment-count check and precedes all
	// length, range-bound, element-finiteness, and arithmetic validation.
	ErrNilInput = errors.New("paa: nil input")

	// ErrEmptyInput identifies a present but zero-length observation slice.
	// PAA requires at least one original sample: it cannot create a meaningful
	// equal-duration partition from an empty domain. This is distinct from nil.
	ErrEmptyInput = errors.New("paa: empty input")

	// ErrNonFinite identifies NaN or positive/negative infinity in a selected
	// source observation. Transform validates the entire series; TransformRange
	// validates its selected window only, after all structural checks succeed.
	// No input element is silently imputed, skipped, or replaced with zero.
	ErrNonFinite = errors.New("paa: non-finite observation")

	// ErrOverflow identifies arithmetic geometry or result sizes that cannot
	// be represented, or an unrepresentable numeric result. Checked n*w and
	// output-buffer byte products are validated before allocation; an internal
	// invariant failure never publishes partially computed result slices.
	// It is not a promise to catch operating-system out-of-memory failures.
	ErrOverflow = errors.New("paa: arithmetic overflow")
)
