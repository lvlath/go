// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package vwap defines ordered validation for contexts, shapes, anchors, and numeric domains.
// This file rejects invalid observations before allocation or mutation can publish state.
package vwap

import (
	"context"
	"errors"
	"math"
)

const (
	// pollInterval bounds the number of rows between context observations.
	// It is a power of two so the hot-loop poll test can use a bit mask.
	pollInterval = 1024
	// pollMask selects row zero and every subsequent pollInterval-th row.
	pollMask = pollInterval - 1
)

// validateContext classifies a missing or already interrupted context before input access.
// It preserves the context cancellation/deadline identity without touching data or stream state.
//
// Implementation:
//   - Stage 1: compare the context interface with nil.
//   - Stage 2: return ErrNilContext for a missing interface.
//   - Stage 3: otherwise return its Err value unchanged.
//
// Behavior highlights:
//   - No reflection or recovery is used to inspect context implementations.
//
// Inputs:
//   - ctx: a context interface; a non-nil implementation must obey the context contract.
//
// Returns:
//   - error: nil, ErrNilContext, or the context's current error.
//
// Errors:
//   - ErrNilContext precedes every receiver, shape, anchor, and numerical input error.
//
// Determinism:
//   - Input order and integer comparisons determine the result; no shared mutable state is read.
//
// Complexity:
//   - Time O(1), Space O(1), excluding caller-defined context behavior.
//
// Notes:
//   - An interface holding an invalid typed-nil context implementation is not a supported context.
//
// AI-Hints:
//   - Pass context.Background when cancellation is not required.
func validateContext(ctx context.Context) error {
	// Interface nil is the explicit missing-context condition; no data is inspected first.
	if ctx == nil {
		return ErrNilContext
	}
	// Preserve the cancellation/deadline sentinel supplied by the context implementation.
	return ctx.Err()
}

// validateShape checks channel alignment and the mandatory explicit first anchor.
// It inspects reset[0] only after all channel lengths are known to agree.
//
// Implementation:
//   - Stage 1: compare price, volume, and reset lengths.
//   - Stage 2: classify a mixed nil/nonempty mismatch with both ErrShape and ErrNilInput.
//   - Stage 3: for equal nonempty channels require reset[0]=true; allow every all-empty combination.
//
// Behavior highlights:
//   - A non-nil empty reset slice is valid only when both numerical channels are empty too.
//
// Inputs:
//   - price, volume, reset: borrowed input slices, read-only during validation.
//
// Returns:
//   - error: nil for aligned anchored or all-empty input.
//
// Errors:
//   - ErrShape for mismatch; joined ErrNilInput for a missing channel in that mismatch; ErrAnchor for a false first reset.
//
// Determinism:
//   - Input order and integer comparisons determine the result; no shared mutable state is read.
//
// Complexity:
//   - Time O(1), Space O(1); joining two error identities may allocate on rejection.
//
// Notes:
//   - Length mismatch precedes anchor and numerical errors.
//
// AI-Hints:
//   - Do not synthesize a first reset or classify all-nil empty input as missing data.
func validateShape(price, volume []float64, reset []bool) error {
	// Establish aligned shape before any reset index or numerical cell can be read.
	if len(price) != len(volume) || len(price) != len(reset) {
		// Preserve both error categories instead of hiding one layer of the input defect.
		if price == nil || volume == nil || reset == nil {
			return errors.Join(ErrShape, ErrNilInput)
		}
		return ErrShape
	}
	// Empty histories have no first bar; all nonempty histories need a caller-supplied anchor.
	if len(price) > 0 && !reset[0] {
		return ErrAnchor
	}
	return nil
}

// validateRow checks one observation without inferring price or volume semantics.
// Finite signed prices are admitted, while volume must be finite and nonnegative.
//
// Implementation:
//   - Stage 1: reject either input when NaN or infinite.
//   - Stage 2: reject a finite volume strictly less than zero.
//   - Stage 3: admit the remaining row without rounding or normalization.
//
// Behavior highlights:
//   - Either zero-volume sign is zero; a zero-volume row still requires a finite price.
//
// Inputs:
//   - price: any finite signed binary64 quote.
//   - volume: finite nonnegative weight in consistent units.
//
// Returns:
//   - error: nil for a valid row.
//
// Errors:
//   - ErrNonFinite for NaN/Inf, otherwise ErrVolume for finite negative volume.
//
// Determinism:
//   - Input order and integer comparisons determine the result; no shared mutable state is read.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Finiteness precedes volume domain within the same row.
//
// AI-Hints:
//   - Do not reject negative prices or silently discard corrupt zero-volume observations.
func validateRow(price, volume float64) error {
	// Finiteness has priority even when volume is zero or a negative infinity.
	if math.IsNaN(price) || math.IsInf(price, 0) ||
		math.IsNaN(volume) || math.IsInf(volume, 0) {
		return ErrNonFinite
	}
	// Negative zero compares equal to zero and is deliberately admitted as zero weight.
	if volume < 0 {
		return ErrVolume
	}
	return nil
}

// validateHistory validates every input row in increasing index order.
// Early rejection never exposes a partial batch result or mutates input observations.
//
// Implementation:
//   - Stage 1: poll context at row zero and every pollInterval rows.
//   - Stage 2: apply per-row finite then volume validation at each index.
//   - Stage 3: poll context once after the complete validation scan.
//
// Behavior highlights:
//   - An earlier finite negative volume precedes a later NaN; precedence is per index.
//
// Inputs:
//   - ctx: validated non-nil context.
//   - price, volume: equal-length read-only channels.
//
// Returns:
//   - error: the first observed validation/cancellation error, or nil.
//
// Errors:
//   - ctx.Err(), ErrNonFinite, or ErrVolume in the documented scan order.
//
// Determinism:
//   - Input order and integer comparisons determine the result; no shared mutable state is read.
//
// Complexity:
//   - Time O(n), Space O(1), excluding caller context behavior.
//
// Notes:
//   - Empty histories still perform the post-validation poll.
//
// AI-Hints:
//   - Do not split this into global finite/domain passes: that would change observable error precedence.
func validateHistory(ctx context.Context, price, volume []float64) error {
	// Scan in input order; no future row can replace an already encountered domain error.
	for i := range price {
		// Power-of-two cadence includes row zero and bounds uninterrupted validation work.
		if i&pollMask == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if err := validateRow(price[i], volume[i]); err != nil {
			return err
		}
	}
	// This also observes cancellation for an empty history before result publication.
	return ctx.Err()
}
