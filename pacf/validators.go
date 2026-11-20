// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pacf validates input shape, arithmetic domains, and cancellation priority.
// This file prevents invalid data and unrepresentable workspace sizes reaching kernels.
package pacf

import (
	"context"
	"math"
)

const (
	// cancellationPollInterval bounds work between cooperative context checks.
	// The fixed positive interval keeps long covariance and recurrence scans cancelable.
	cancellationPollInterval = 1024

	// float64StorageBytes and liveFloat64Arrays describe worst-case simultaneous storage.
	// One covariance array, two coefficient rows and two public result slices coexist.
	float64StorageBytes = 8
	liveFloat64Arrays   = 5
)

// validate enforces the stable context/presence/shape/finiteness error precedence.
// After checking finite inputs, it also bounds the total slice-allocation product.
//
// Implementation:
//   - Stage 1: reject nil/canceled contexts before touching the input slice.
//   - Stage 2: reject absent/empty samples and invalid inclusive MaxLag.
//   - Stage 3: scan samples for NaN/Inf, then guard the five live float64 arrays.
//
// Behavior highlights:
//   - Validation never allocates an output or changes caller-owned sample storage.
//   - Context is rechecked periodically and at scan completion.
//
// Inputs:
//   - ctx: nil or non-nil context controlling the whole computation.
//   - values: caller-owned chronological observations, not modified or retained.
//   - opts: MaxLag/Center policy; Center does not change shape validation.
//
// Returns:
//   - error: nil only if required public preconditions and byte bounds hold.
//
// Errors:
//   - ErrInvalidOption: nil context or MaxLag outside the admitted sample window.
//   - ErrNilInput, ErrEmptyInput: distinct absent and empty slice conditions.
//   - ErrNonFinite: NaN or infinity, checked only after structural validation.
//   - ErrOverflow: simultaneous float64 slice byte product exceeds integer capacity.
//   - ctx.Err(): cancellation before or during the observation scan.
//
// Determinism:
//   - Precedence is context, nil input, empty input, lag shape, nonfinite cells, capacity.
//   - Cells are inspected by ascending sample index.
//
// Complexity:
//   - Time O(n), Space O(1), n=len(values).
//
// Notes:
//   - Runtime out-of-memory is outside Go's recoverable error model.
//
// AI-Hints:
//   - Do not validate element finiteness before nil/context/shape checks.
//   - Do not convert integer byte overflow into a panic from make.
func validate(ctx context.Context, values []float64, opts Options) error {
	if ctx == nil {
		return ErrInvalidOption
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if values == nil {
		return ErrNilInput
	}
	if len(values) == 0 {
		return ErrEmptyInput
	}
	if opts.MaxLag < 0 || opts.MaxLag >= len(values) {
		return ErrInvalidOption
	}

	// The raw input domain is finite even when scaled arithmetic could handle infinity.
	for i, value := range values {
		if i%cancellationPollInterval == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return ErrNonFinite
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	// Check a simultaneous byte product instead of relying only on len<MaxInt.
	// maxLag+1 is evaluated only after proving it cannot exceed the safe bound.
	if opts.MaxLag >= math.MaxInt/(liveFloat64Arrays*float64StorageBytes) {
		return ErrOverflow
	}
	return nil
}
