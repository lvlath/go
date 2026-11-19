// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package arima exposes stable sentinel-first failure classifications.
// This file owns errors.Is-compatible errors without embedding caller sample values.
package arima

import "errors"

var (
	// ErrInvalidOption rejects invalid order, solver policy, horizon or absent context/model.
	ErrInvalidOption = errors.New("arima: invalid option")
	// ErrNilInput distinguishes a missing observation slice from an allocated empty slice.
	ErrNilInput = errors.New("arima: nil input")
	// ErrEmptyInput rejects a present observation series with no samples.
	ErrEmptyInput = errors.New("arima: empty input")
	// ErrInsufficientData reports missing conditioning data, anchors or fit degrees of freedom.
	ErrInsufficientData = errors.New("arima: insufficient data")
	// ErrNonFinite rejects NaN or infinity in caller-provided numeric values.
	ErrNonFinite = errors.New("arima: non-finite input")
	// ErrNonStationary rejects an AR polynomial with a root on or inside the unit circle.
	ErrNonStationary = errors.New("arima: non-stationary AR polynomial")
	// ErrNonInvertible rejects an MA polynomial with a root on or inside the unit circle.
	ErrNonInvertible = errors.New("arima: non-invertible MA polynomial")
	// ErrSingular denotes zero/nonpositive innovation variance or zero effective SSE.
	ErrSingular = errors.New("arima: zero innovation variance")
	// ErrOverflow reports nonrepresentable computed values or checked allocation dimensions.
	ErrOverflow = errors.New("arima: numeric overflow")
	// ErrNotConverged reports exhausted L-BFGS iterations or line-search trials.
	// It is the sole fitting error that permits a last accepted, non-nil FitResult.
	ErrNotConverged = errors.New("arima: optimizer did not converge")
)
