// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package drawdown validates the context and chronological equity domain.
// This file separates shape admission from per-observation checks.
package drawdown

import (
	"context"
	"fmt"
	"math"
)

// cancellationStride bounds the number of observations between context polls.
// It is a work bound, not a latency guarantee: scheduling and allocation are external.
const cancellationStride = 1024

// validateInput admits context and shape before touching the initial observation.
// It prevents indexing an empty slice and rejects an invalid origin before allocation.
//
// Implementation:
//   - Stage 1: distinguish nil context from an observed cancellation/deadline.
//   - Stage 2: reject empty histories (nil and non-nil empty have the same law).
//   - Stage 3: validate index zero, whose strict positivity guarantees every peak >0.
//
// Behavior highlights:
//   - A first nonfinite value is classified before its sign; no other rows are read here.
//
// Inputs:
//   - ctx: caller context interface; equity: read-only observations in index order.
//
// Returns:
//   - nil after admission, otherwise the first detected sentinel or context error.
//
// Errors:
//   - ErrNilContext, ctx.Err(), ErrInsufficient, ErrNonFinite, ErrEquity, in stage order.
//
// Determinism:
//   - Context before shape before first observation; later values belong to scanDepth.
//
// Complexity:
//   - Time O(1), Space O(1); no output allocation.
//
// Notes:
//   - Typed-nil/misbehaving custom contexts are outside the usable-context contract.
//
// AI-Hints:
//   - Do not use reflection to inspect arbitrary contexts or add a third full scan.
func validateInput(ctx context.Context, equity []float64) error {
	if ctx == nil { // No context methods may be called through a nil interface.
		return ErrNilContext
	}
	if err := ctx.Err(); err != nil { // Already canceled work takes precedence.
		return err
	}
	if len(equity) == 0 { // Both nil and empty slices have no positive origin.
		return ErrInsufficient
	}
	// The first observation has a stricter domain than all subsequent observations.
	return validateEquity(0, equity[0])
}

// validateEquity classifies a single observation under the sampled equity domain.
// It enforces a positive initial peak and allows subsequent signed zero as total loss.
//
// Implementation:
//   - Stage 1: reject NaN and either infinity before sign comparisons.
//   - Stage 2: reject negative finite values, including zero at index zero.
//   - Stage 3: admit the value unchanged; wrap failures with its original index.
//
// Behavior highlights:
//   - Later -0 and +0 are equal numerically; only published depths canonicalize zero.
//
// Inputs:
//   - index: original nonnegative position; value: observed equity in caller units.
//
// Returns:
//   - nil for a valid observation, or an errors.Is-compatible indexed error.
//
// Errors:
//   - ErrNonFinite for NaN/Inf; ErrEquity for finite values outside the domain.
//
// Determinism:
//   - Finiteness precedes domain within the same index, not across the entire history.
//
// Complexity:
//   - Time O(1), Space O(1) for fixed-width values; formatting only on error.
//
// Notes:
//   - Caller storage is never modified or retained.
//
// AI-Hints:
//   - Do not allow NaN to bypass validation via comparisons or classify -Inf as insolvency.
func validateEquity(index int, value float64) error {
	if math.IsNaN(value) || math.IsInf(value, 0) { // Reject all nonfinite encodings.
		return fmt.Errorf("equity[%d]: %w", index, ErrNonFinite)
	}
	if value < 0 || (index == 0 && value == 0) { // Only later zero is admissible.
		return fmt.Errorf("equity[%d]: %w", index, ErrEquity)
	}
	return nil // The original binary64 is valid without normalization.
}
