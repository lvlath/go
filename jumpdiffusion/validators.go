// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package jumpdiffusion validates immutable model, grid, and RNG inputs.
// This file enforces error precedence without reflection or entropy consumption.
package jumpdiffusion

import (
	"context"
	"math"

	"github.com/lvlath/go/gbm"
	"github.com/lvlath/go/rng"
)

// finite checks whether a number is neither NaN nor an infinity.
// A finite input remains subject to later checked-arithmetic overflow gates.
//
// Inputs: value is any binary64 value; Returns: true only for finite values.
// Errors: none; Determinism: pure IEEE-754 comparison.
// Complexity: O(1) time/space; AI-Hints: do not confuse finite input with safe math.
func finite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

// validateInputs rejects malformed simulations before any RNG draw or allocation.
// It shares GBM's physical units, shape, numeric domains, and nilable RNG policy.
//
// Implementation:
//   - Stage 1: check context cancellation and closed time-unit/convention enums.
//   - Stage 2: check RNG and grid presence, then scan timestamps in input order.
//   - Stage 3: classify nonfinite parameters separately from negative domains.
//
// Behavior highlights:
//   - Input validation occurs even when Intensity=0 or the grid has one point.
//   - Recognized typed-nil streams support optional IsNil without reflect.
//
// Inputs:
//   - ctx: non-nil context; a pre-canceled context wins over all other checks.
//   - p: fixed GBM and log-jump coefficients, all finite with nonnegative scales.
//   - grid: detached-on-return read-only timestamps starting at zero, strictly rising.
//   - random: non-nil, single-owner RNG; interface nilability is optional.
//
// Returns: nil on success, or the earliest deterministic error; no Result.
// Errors:
//   - ErrInvalidOption: invalid context, enum, or incompatible time units.
//   - ErrNilInput/ErrEmptyInput: missing RNG/grid or present zero-length grid.
//   - ErrNonFinite/ErrOrder: invalid times or coefficients.
//   - ErrDomain: nonpositive initial price or negative volatility/intensity/stddev.
//   - ctx.Err(): canceled before execution, preserved without wrapping.
//
// Determinism: context, units/options, presence, ascending timestamps, coefficients.
// Complexity: Time O(n), Space O(1), n=len(grid.Values).
// Notes: concurrent mutation of grid or RNG is not supported.
// AI-Hints: never infer time units from timestamp magnitude or touch RNG to validate.
func validateInputs(ctx context.Context, p Parameters, grid gbm.TimeGrid, random rng.RNG) error {
	// Context must lead the error precedence: nil cannot be queried, and a
	// pre-canceled call must neither inspect a foreign RNG nor allocate output.
	if ctx == nil {
		return ErrInvalidOption
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	// Enumerations are closed domains. Equality between two unsupported unit
	// values is not enough to make a time grid physically meaningful.
	if p.Diffusion.Unit > gbm.TimeSeconds || grid.Unit > gbm.TimeSeconds || p.Diffusion.Unit != grid.Unit {
		return ErrInvalidOption
	}
	if p.Convention > Uncompensated {
		return ErrInvalidOption
	}

	// The optional IsNil capability recognizes cooperative typed-nil streams
	// without reflect. Do not call entropy-producing methods during validation.
	if random == nil {
		return ErrNilInput
	}
	if nilable, ok := random.(interface{ IsNil() bool }); ok && nilable.IsNil() {
		return ErrNilInput
	}
	// A nil slice is an absent grid; an allocated zero-length slice is a
	// different caller error, while one time-zero observation is valid.
	if grid.Values == nil {
		return ErrNilInput
	}
	if len(grid.Values) == 0 {
		return ErrEmptyInput
	}

	// Numeric rejection precedes comparisons because NaN has no total order.
	// Repeated or descending times are rejected, never sorted or silently merged.
	if !finite(grid.Values[0]) {
		return ErrNonFinite
	}
	if grid.Values[0] != 0 {
		return ErrOrder
	}
	for index := 1; index < len(grid.Values); index++ {
		if !finite(grid.Values[index]) {
			return ErrNonFinite
		}
		if grid.Values[index] <= grid.Values[index-1] {
			return ErrOrder
		}
	}

	// Classify invalid source coefficients before the physical-domain checks:
	// finite negative rates are domain errors, not nonfinite arithmetic errors.
	model := p.Diffusion
	if !finite(model.Initial) || !finite(model.Drift) || !finite(model.Volatility) ||
		!finite(p.Intensity) || !finite(p.LogJumpMean) || !finite(p.LogJumpStdDev) {
		return ErrNonFinite
	}
	if model.Initial <= 0 || model.Volatility < 0 || p.Intensity < 0 || p.LogJumpStdDev < 0 {
		return ErrDomain
	}
	return nil
}
