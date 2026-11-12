// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ridge defines stable failure classifications for regression operations.
// Error strings are diagnostic only; callers must branch using errors.Is.
package ridge

import "errors"

var (
	// ErrNilInput classifies absent predictor matrices, responses, or model receivers.
	// Matrix typed nils recognized by core.Nilable preserve matrix.ErrNilMatrix as a cause.
	// This classification precedes all shape checks once context/options are valid.
	ErrNilInput = errors.New("ridge: nil input")

	// ErrEmptyInput classifies zero-row or zero-feature training data and empty responses.
	// Unlike Fit, Predict permits an exactly dimensioned zero-row inference matrix.
	// Empty is not a synonym for an absent (nil) training slice.
	ErrEmptyInput = errors.New("ridge: empty input")

	// ErrShape classifies negative dimensions, mismatched training response length,
	// or inference feature widths that differ from the fitted physical predictor width.
	// Wide training matrices are legal; p>n alone is never a shape failure.
	ErrShape = errors.New("ridge: incompatible shape")

	// ErrInvalidOption classifies nil contexts and invalid numeric policy values.
	// Lambda must be finite and strictly positive; RankTolerance must be finite
	// and nonnegative. A canceled non-nil context returns ctx.Err() instead.
	ErrInvalidOption = errors.New("ridge: invalid option")

	// ErrNonFinite classifies supplied predictor or response NaN and infinities.
	// No missing observation is dropped, imputed, or converted into a zero value.
	// Arithmetic overflow after a finite input is instead classified as ErrOverflow.
	ErrNonFinite = errors.New("ridge: non-finite observation")

	// ErrRankDeficient classifies an insufficient numerical rank of augmented QR.
	// A strictly positive Lambda guarantees algebraic uniqueness but cannot
	// guarantee QR's finite-precision tolerance; matrix's rank cause is retained.
	ErrRankDeficient = errors.New("ridge: deficient augmented rank")

	// ErrOverflow classifies unrepresentable allocation sizes, transforms,
	// physical coefficients, inference values, or mandatory fit diagnostics.
	// A failed Fit never publishes a model or partially populated FitResult.
	ErrOverflow = errors.New("ridge: floating-point overflow")
)
