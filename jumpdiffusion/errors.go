// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package jumpdiffusion declares the stable, sentinel-first failure protocol.
// This file keeps error identity separate from diagnostic text and dependencies.
package jumpdiffusion

import "errors"

var (
	// ErrInvalidOption reports a nil context, invalid drift convention or time unit.
	// A mismatched GBM/grid unit is rejected before inspecting samples or drawing RNG.
	ErrInvalidOption = errors.New("jumpdiffusion: invalid option")

	// ErrNilInput reports a missing RNG or nil grid slice.
	// Recognized typed-nil RNG implementations are also rejected without reflection.
	ErrNilInput = errors.New("jumpdiffusion: nil input")

	// ErrEmptyInput reports an allocated but zero-length time grid.
	// A one-element grid beginning at zero is valid and consumes no RNG.
	ErrEmptyInput = errors.New("jumpdiffusion: empty input")

	// ErrOrder reports an origin other than zero or a nonincreasing timestamp.
	// Simulation does not reorder, merge or interpolate the caller's grid.
	ErrOrder = errors.New("jumpdiffusion: invalid time order")

	// ErrNonFinite reports NaN/Inf among inputs or source-produced random draws.
	// Finite but unrepresentable computed quantities use ErrOverflow instead.
	ErrNonFinite = errors.New("jumpdiffusion: non-finite value")

	// ErrDomain rejects invalid physical parameters or unsupported Poisson means.
	// It includes per-step mean>2^40 or a proposal outside the count domain.
	ErrDomain = errors.New("jumpdiffusion: invalid domain")

	// ErrOverflow reports an unrepresentable compensator, Poisson mean, or price.
	// Positive Poisson means rounded to zero are rejected before any RNG draw;
	// zero or infinite prices are never emitted as completed observations.
	ErrOverflow = errors.New("jumpdiffusion: numeric overflow or underflow")

	// ErrBudget reports exhaustion of the fixed Poisson rejection-attempt limit.
	// Only already-completed grid points are published in the partial Result.
	ErrBudget = errors.New("jumpdiffusion: sampling budget exhausted")
)
