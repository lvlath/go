// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gjrgarch defines stable errors.Is-compatible failure categories.
// This file owns all package-level sentinel errors and their public meaning.
package gjrgarch

import "errors"

var (
	// ErrInvalidOption rejects absent contexts, invalid optimization policies, or an infeasible QMLE start.
	ErrInvalidOption = errors.New("gjrgarch: invalid option")
	// ErrNilInput rejects a nil return-series slice before any data-dependent work.
	ErrNilInput = errors.New("gjrgarch: nil input")
	// ErrEmptyInput rejects a present but zero-length return series.
	ErrEmptyInput = errors.New("gjrgarch: empty input")
	// ErrNonFinite rejects caller-supplied NaN and infinity inputs.
	ErrNonFinite = errors.New("gjrgarch: nonfinite input")
	// ErrDomain rejects invalid parameters, initial variances, or infeasible likelihood candidates.
	ErrDomain = errors.New("gjrgarch: invalid model domain")
	// ErrStateVersion rejects nil models, foreign state IDs, or invalid posterior state values.
	ErrStateVersion = errors.New("gjrgarch: invalid or incompatible state")
	// ErrOverflow reports non-representable arithmetic or allocation dimensions.
	ErrOverflow = errors.New("gjrgarch: arithmetic overflow")
	// ErrNotConverged identifies an unfinished fit with a retained valid model candidate.
	ErrNotConverged = errors.New("gjrgarch: optimizer did not converge")
)
