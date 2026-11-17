// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package obv defines independent OHLCV shape and domain guards.
// This file validates every candle field even though OBV itself uses only Close and Volume.
package obv

import "math"

// finite reports whether a binary64 quantity is neither NaN nor infinity.
// It does not imply that a value satisfies the OHLCV domain restrictions.
//
// Implementation:
//   - Stage 1: reject NaN.
//   - Stage 2: reject either infinity.
//   - Stage 3: preserve the original value without normalization.
//
// Inputs:
//   - value: one unmodified floating-point quantity.
//
// Returns:
//   - bool: true exactly when value is finite.
//
// Determinism:
//   - IEEE-754 classification without tolerances or rounding.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// AI-Hints:
//   - Use a separate domain check: finite(-1) is true, but negative volume is invalid.
func finite(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

// validateShape checks the presence and alignment of all five OHLCV columns.
// Structural failure is detected without reading any price or volume element.
//
// Implementation:
//   - Stage 1: reject any nil column, regardless of the other lengths.
//   - Stage 2: reject a present but zero-length Open column.
//   - Stage 3: require all remaining columns to share Open's positive length.
//
// Inputs:
//   - bars: caller-owned, read-only column references; elements are not inspected.
//
// Returns:
//   - int: validated positive observation count, or zero on error.
//   - error: stable structural sentinel on failure.
//
// Errors:
//   - ErrNilInput: at least one column is nil.
//   - ErrEmptyInput: all columns exist, but Open has length zero.
//   - ErrShape: column lengths differ after presence/empty checks.
//
// Determinism:
//   - Precedence is Nil, Empty, Shape; no per-row access occurs here.
//
// Complexity:
//   - Time O(1), Space O(1), no allocations.
//
// AI-Hints:
//   - Never index Close/Volume before the shape gate has completed.
func validateShape(bars Bars) (int, error) {
	if bars.Open == nil || bars.High == nil || bars.Low == nil || bars.Close == nil || bars.Volume == nil {
		return 0, ErrNilInput
	}
	// Open is the length authority only AFTER every column has passed nil
	// validation; a present empty Open has a distinct error from bad shape.
	count := len(bars.Open)
	if count == 0 {
		return 0, ErrEmptyInput
	}
	if len(bars.High) != count || len(bars.Low) != count ||
		len(bars.Close) != count || len(bars.Volume) != count {
		return 0, ErrShape
	}
	return count, nil
}

// validateBar rejects malformed candles without repairing or skipping observations.
// Every OHLCV field is validated although only Close and Volume enter the OBV recurrence.
//
// Implementation:
//   - Stage 1: check all five fields for NaN and infinity.
//   - Stage 2: require positive prices and nonnegative volume.
//   - Stage 3: enforce Low <= Open,Close <= High and High >= Low.
//
// Behavior highlights:
//   - Exact zero volume is valid and must not suppress an observation.
//   - Flat candles and equal consecutive closes are allowed.
//
// Inputs:
//   - bar: a read-only OHLCV value with coherent base-volume units.
//
// Returns:
//   - error: nil for valid geometry and domain.
//
// Errors:
//   - ErrNonFinite: any field is NaN or either infinity.
//   - ErrDomain: positive-price, nonnegative-volume, or geometry condition fails.
//
// Determinism:
//   - Checks finiteness before all domain predicates within each candle.
//
// Complexity:
//   - Time O(1), Space O(1), zero allocations.
//
// Notes:
//   - Batch validation takes the first failing candle in input order.
//
// AI-Hints:
//   - Do not clip High/Low to make invalid bars pass; reject bad exchange data instead.
func validateBar(bar Bar) error {
	if !finite(bar.Open) || !finite(bar.High) || !finite(bar.Low) ||
		!finite(bar.Close) || !finite(bar.Volume) {
		return ErrNonFinite
	}
	// Keep geometry separate from finiteness: an infinite High is a numeric
	// input failure, whereas a finite High below Open is a domain failure.
	if bar.Low <= 0 || bar.High <= 0 || bar.Open <= 0 || bar.Close <= 0 ||
		bar.Volume < 0 || bar.High < bar.Low || bar.Open < bar.Low ||
		bar.Open > bar.High || bar.Close < bar.Low || bar.Close > bar.High {
		return ErrDomain
	}
	return nil
}
