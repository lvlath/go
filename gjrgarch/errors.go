// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gjrgarch defines stable errors.Is-compatible failure categories.
// This file owns all package-level sentinel errors and their public meaning.
package gjrgarch

import "errors"

var (
	// ErrInvalidOption rejects absent contexts, invalid optimization policies, or an infeasible QMLE start.
	// Also used for a nil context or initial parameter point that violates QMLE policy.
	ErrInvalidOption = errors.New("gjrgarch: invalid option")
	// ErrNilInput rejects a nil return-series slice before any data-dependent work.
	// Distinct from ErrEmptyInput so callers can detect an omitted observation buffer.
	ErrNilInput = errors.New("gjrgarch: nil input")
	// ErrEmptyInput rejects a present but zero-length return series.
	// The empty-but-present input is not treated as a valid zero-length likelihood.
	ErrEmptyInput = errors.New("gjrgarch: empty input")
	// ErrNonFinite rejects caller-supplied NaN and infinity inputs.
	// Valid finite inputs that overflow during computation instead return ErrOverflow.
	ErrNonFinite = errors.New("gjrgarch: nonfinite input")
	// ErrDomain rejects invalid parameters, initial variances, or infeasible likelihood candidates.
	// Includes stationarity and coefficient-sign violations; it is not a numerical clamp.
	ErrDomain = errors.New("gjrgarch: invalid model domain")
	// ErrStateVersion rejects nil models, foreign state IDs, or invalid posterior state values.
	// The model digest is a compatibility check, not cryptographic state authentication.
	ErrStateVersion = errors.New("gjrgarch: invalid or incompatible state")
	// ErrOverflow reports non-representable arithmetic or allocation dimensions.
	// This covers unrepresentable final numerical output rather than bad caller float values.
	ErrOverflow = errors.New("gjrgarch: arithmetic overflow")
	// ErrNotConverged identifies an unfinished fit with a retained valid model candidate.
	// The last validated model can accompany this error but must not be treated as a successful fit.
	ErrNotConverged = errors.New("gjrgarch: optimizer did not converge")
)
