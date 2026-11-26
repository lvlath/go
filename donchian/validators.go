// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package donchian validates call metadata and finite signed candle ranges.
// This file establishes error precedence without normalizing or mutating caller data.
package donchian

import (
	"context"
	"errors"
	"fmt"
	"math"
)

// pollInterval bounds unchecked input rows and unchecked tail removals.
// It is an iteration bound, not a wall-clock latency promise or a sample period.
const pollInterval = 1024

// validateHeader rejects invalid call metadata before dereferencing a price slice.
// It preserves the ordered context, option, and shape failure classes.
//
// Implementation:
//   - Stage 1: reject a nil context, then propagate an already observed context error.
//   - Stage 2: reject nonpositive Period before comparing channel lengths.
//   - Stage 3: classify unequal lengths, preserving both shape and nil/nonempty identity.
//
// Behavior highlights:
//   - Nil and empty slices are equivalent when both lengths are zero.
//   - Every positive int period is admitted, even if no output row can become ready.
//
// Inputs:
//   - ctx: public context interface; typed-nil custom implementations are unsupported.
//   - high, low: caller-owned slices; this function reads only their headers.
//   - opts: caller policy passed by value.
//
// Returns:
//   - error: nil for valid metadata; otherwise a stable category or ctx.Err().
//
// Errors:
//   - ErrNilContext, ctx.Err(), ErrInvalidOption, then ErrShape in that order.
//   - A nil channel with a nonempty peer also matches ErrNilInput through errors.Join.
//
// Determinism:
//   - Fixed check order, with no iteration or input mutation.
//
// Complexity:
//   - O(1) time/space; a joined shape error may allocate a bounded error object.
//
// Notes:
//   - An empty non-nil slice with a nonempty peer is only ErrShape.
//   - Error identity is a set of errors.Is matches, not a required diagnostic string.
//
// AI-Hints:
//   - Never move price inspection or the empty-success path before this check.
//   - Do not replace the joined error by just one sentinel; both matches are observable.
func validateHeader(ctx context.Context, high, low []float64, opts Options) error {
	// A nil interface has no Err method to call and wins over every input defect.
	if ctx == nil {
		return ErrNilContext
	}
	// Preserve cancellation versus deadline identity; do not remap it to a sentinel.
	if err := ctx.Err(); err != nil {
		return err
	}
	// A positive period is explicit policy, including on an empty history.
	if opts.Period < 1 {
		return ErrInvalidOption
	}
	// Equal lengths are the precondition for indexing low during the high scan.
	if len(high) != len(low) {
		// Unequal lengths plus nil imply exactly one nil and one nonempty channel.
		if high == nil || low == nil {
			return errors.Join(ErrShape, ErrNilInput)
		}
		// Preserve the distinction between missing nil data and a non-nil empty slice.
		return ErrShape
	}
	// Header validity permits either a real history or the all-empty success path.
	return nil
}

// validateHistory admits finite signed candle ranges without modifying observations.
// It gives non-finite input global priority over finite high-below-low geometry.
//
// Implementation:
//   - Stage 1: scan rows in ascending order and poll cancellation at bounded intervals.
//   - Stage 2: inspect high then low for finiteness; remember the first geometry defect.
//   - Stage 3: report remembered ErrBar only after the entire scan is known finite.
//
// Behavior highlights:
//   - A later non-finite value outranks an earlier finite geometry violation.
//   - Masked rows and the last excluded bar are validated, not silently discarded.
//
// Inputs:
//   - ctx: active non-nil context admitted by validateHeader.
//   - high, low: equal-length read-only channels admitted by validateHeader.
//
// Returns:
//   - error: nil for finite high >= low everywhere; otherwise no result is published.
//
// Errors:
//   - ctx.Err(): cancellation observed at a polling boundary.
//   - ErrNonFinite: first invalid high/low in row order; ErrBar: first finite geometry defect.
//
// Determinism:
//   - Ascending row index, high before low, first geometry index retained.
//   - Concurrent cancellation and a domain failure are ordered by detection, not scheduling.
//
// Complexity:
//   - O(n) time with one traversal and O(1) auxiliary state.
//
// Notes:
//   - Negative prices and signed zero are valid; comparisons use ordinary numeric order.
//   - The caller performs a context check after successful validation, including empty input.
//
// AI-Hints:
//   - Returning ErrBar immediately would change global non-finite precedence.
//   - Do not skip validation for Period > n or replace non-finite values with zero.
func validateHistory(ctx context.Context, high, low []float64) error {
	// No valid observation index is negative, so -1 means no geometry failure yet.
	firstBadBar := -1
	// A single pass preserves cache locality without weakening global error precedence.
	for i, upper := range high {
		// Poll before a block, so cancellation can stop a long validation pass.
		if i%pollInterval == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		// Reject NaN before comparisons; otherwise NaN would bypass geometry checks.
		if !finite(upper) {
			return fmt.Errorf("%w: high[%d]", ErrNonFinite, i)
		}
		// Equal lengths were checked before this indexed read.
		if !finite(low[i]) {
			return fmt.Errorf("%w: low[%d]", ErrNonFinite, i)
		}
		// Remember only the first finite bad bar; keep searching for non-finite input.
		if upper < low[i] && firstBadBar < 0 {
			firstBadBar = i
		}
	}
	// No non-finite input was found, so a remembered geometry error now wins.
	if firstBadBar >= 0 {
		return fmt.Errorf("%w: index %d", ErrBar, firstBadBar)
	}
	// The empty history also reaches this successful completion path.
	return nil
}

// finite reports membership in the finite binary64 price domain.
// It deliberately admits negative prices, both zero signs, and subnormal values.
//
// Implementation:
//   - Stage 1: reject NaN.
//   - Stage 2: reject either infinity.
//   - Stage 3: accept all remaining bit patterns.
//
// Behavior highlights:
//   - No positivity test, clamping, normalization, or epsilon is applied.
//
// Inputs:
//   - value: one binary64 observation, with no prior domain assumptions.
//
// Returns:
//   - bool: true exactly for finite values.
//
// Errors:
//   - None; the caller attaches ErrNonFinite and channel/index context.
//
// Determinism:
//   - Pure classification of the supplied bit pattern.
//
// Complexity:
//   - O(1) time and O(1) space; no allocation.
//
// Notes:
//   - Domain admission does not itself establish high >= low.
//
// AI-Hints:
//   - Keep finiteness separate from candle geometry and readiness.
func finite(value float64) bool {
	// IsInf with sign zero rejects both infinities without rejecting finite negatives.
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}
