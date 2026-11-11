// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ols defines package-owned error sentinels for regression operations.
// Each sentinel identifies a stable failure class for errors.Is, not message parsing.
package ols

import "errors"

var (
	// ErrNilInput classifies missing matrices, response slices, or model receivers.
	// It is raised after context/options checks, before shape or numeric inspection.
	ErrNilInput = errors.New("ols: nil input")

	// ErrEmptyInput classifies zero training rows, zero features, or empty responses.
	// An empty inference batch is valid and does not produce this error.
	ErrEmptyInput = errors.New("ols: empty training input")

	// ErrShape classifies inconsistent n-by-p dimensions, underdetermined designs,
	// invalid shape arithmetic, or prediction matrices with the wrong feature count.
	ErrShape = errors.New("ols: incompatible shape")

	// ErrInvalidOption classifies a nil context or invalid rank tolerance.
	// Context cancellation itself is returned as ctx.Err(), not this sentinel.
	ErrInvalidOption = errors.New("ols: invalid option")

	// ErrNonFinite classifies NaN or +/-Inf in supplied predictors or responses.
	// The input domain is finite float64; missing values are never silently removed.
	ErrNonFinite = errors.New("ols: non-finite input")

	// ErrRankDeficient classifies an insufficient numerical column rank after QR.
	// When supplied by matrix, errors.Is also matches matrix.ErrRankDeficient.
	ErrRankDeficient = errors.New("ols: rank-deficient design")

	// ErrOverflow classifies unrepresentable transformations, fitted coefficients,
	// predictions, or diagnostics; a failed Fit never publishes a partial Model.
	ErrOverflow = errors.New("ols: arithmetic overflow")
)
