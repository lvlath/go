// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package montecarlo declares the driver-level sentinel error protocol.
// These identities are stable errors.Is targets, not diagnostic string contracts.
package montecarlo

import "errors"

var (
	// ErrInvalidOption rejects nil context, invalid path or worker counts,
	// nonpositive dimension, or nonascending/nonfinite quantile levels.
	ErrInvalidOption = errors.New("montecarlo: invalid option")

	// ErrNilInput indicates an absent sampler or an optional nilable typed-nil sampler.
	// Custom typed-nil samplers without IsNil cannot be detected without reflection.
	ErrNilInput = errors.New("montecarlo: nil sampler")

	// ErrShape indicates a successful Sample returned a vector with the wrong length.
	// The invalid row is never appended to the reduced prefix.
	ErrShape = errors.New("montecarlo: sample dimension mismatch")

	// ErrNonFinite indicates a Sample returned NaN or an infinite component.
	// It is distinct from representable-input arithmetic overflow.
	ErrNonFinite = errors.New("montecarlo: nonfinite sample")

	// ErrDomain indicates a dependency invariant or storage operation failed.
	// The underlying dependency error is joined when available.
	ErrDomain = errors.New("montecarlo: dependency domain failure")

	// ErrOverflow indicates integer size overflow or unrepresentable output moments.
	// Exact quantile output is also subject to finite result materialization.
	ErrOverflow = errors.New("montecarlo: arithmetic overflow")

	// ErrBudget indicates exact optional sample/sort/quantile storage exceeds the
	// configured MaxSamplesBytes ceiling, before any Sample invocation.
	ErrBudget = errors.New("montecarlo: sample storage budget exceeded")
)
