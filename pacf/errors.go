// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pacf defines stable sentinel classifications for failed PACF operations.
// Callers must classify errors using errors.Is, never by matching message text.
package pacf

import "errors"

// ErrNilInput means the observation slice itself is nil.
// A non-nil empty slice instead yields ErrEmptyInput after context validation.
var ErrNilInput = errors.New("pacf: nil input")

// ErrEmptyInput means a present observation slice contains no samples.
// A sample window of length one is permitted when MaxLag=0 and energy is positive.
var ErrEmptyInput = errors.New("pacf: empty input")

// ErrInvalidOption means the context is nil or MaxLag lies outside [0,n-1].
// A canceled non-nil context returns ctx.Err() instead of this sentinel.
var ErrInvalidOption = errors.New("pacf: invalid option")

// ErrNonFinite means an observation is NaN, +Inf, or -Inf.
// It is distinguished from overflow caused while processing finite samples.
var ErrNonFinite = errors.New("pacf: non-finite input")

// ErrZeroVariance means the selected lag-zero sample energy is exactly zero.
// It covers centered constants and all-zero uncentered windows, not near singularity.
var ErrZeroVariance = errors.New("pacf: zero variance")

// ErrSingular means the recurrence cannot produce positive error variance.
// In particular, a reflection coefficient with magnitude at least one is rejected.
var ErrSingular = errors.New("pacf: singular prediction system")

// ErrOverflow means a necessary arithmetic result or allocation product is not representable.
// This includes overflow and underflow-to-zero of a required positive physical variance.
var ErrOverflow = errors.New("pacf: arithmetic overflow")
