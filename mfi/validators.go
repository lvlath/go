// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mfi defines the structural and numerical validation boundaries for OHLCV.
// No input is mutated, silently clipped, inferred or filtered by these helpers.
package mfi

import "math"

// finite classifies a representable binary64 number without changing its value.
// NaN and either infinity are forbidden in OHLCV inputs and recurrence states.
//
// Implementation:
//   - Stage 1: detect NaN.
//   - Stage 2: detect positive or negative infinity.
//   - Stage 3: report whether the original bits encode a finite value.
//
// Behavior highlights:
//   - Negative finite numbers are still finite; domain checks happen separately.
//
// Inputs:
//   - value: one unmodified float64 observation or intermediate.
//
// Returns:
//   - bool: false exactly for NaN and either infinity.
//
// Determinism:
//   - Uses math.IsNaN and math.IsInf without normalization or tolerance.
//
// Complexity:
//   - Time O(1), Space O(1); no allocation.
//
// Notes:
//   - This helper is not an OHLCV or nonnegative-value validator.
//
// AI-Hints:
//   - Do not use the finite check to accept a negative volume or price.
func finite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

// validateShape enforces batch column presence before indexing any input.
// Nil, non-nil empty and inconsistent column lengths remain separate errors.
//
// Implementation:
//   - Stage 1: reject any nil OHLCV column, regardless of other lengths.
//   - Stage 2: distinguish a present zero-length Open column from a nonempty batch.
//   - Stage 3: compare the remaining four lengths against Open before element access.
//
// Behavior highlights:
//   - A non-nil empty Open column takes priority over differing nonempty column lengths.
//
// Inputs:
//   - bars: caller-owned five-column value; element contents are not inspected here.
//
// Returns:
//   - int: positive shared column length on success, otherwise zero.
//   - error: ErrNilInput, ErrEmptyInput or ErrShape by the stated precedence.
//
// Errors:
//   - ErrNilInput: at least one column is nil.
//   - ErrEmptyInput: all columns are non-nil but Open is empty.
//   - ErrShape: the non-nil column lengths differ after the empty check.
//
// Determinism:
//   - Checks missing fields before Open length, then High, Low, Close, Volume lengths.
//
// Complexity:
//   - Time O(1), Space O(1), zero allocation.
//
// Notes:
//   - The source slices remain caller-owned, read-only and never retained.
//
// AI-Hints:
//   - Do not skip shape validation and index High/Low/Close before this gate.
func validateShape(bars Bars) (int, error) {
	if bars.Open == nil || bars.High == nil || bars.Low == nil || bars.Close == nil || bars.Volume == nil {
		return 0, ErrNilInput
	}
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

// validateBar rejects nonfinite values before checking candle geometry.
// Strictly positive prices and nonnegative volume are required even when
// a given field would not contribute to typical price on that transition.
//
// Implementation:
//   - Stage 1: inspect all five inputs for NaN or infinity.
//   - Stage 2: reject nonpositive prices or negative volume.
//   - Stage 3: enforce Low <= Open,Close <= High without altering the candle.
//
// Behavior highlights:
//   - A zero Volume is valid, including when prices change.
//   - Open is checked even though typical price uses only High, Low and Close.
//
// Inputs:
//   - bar: caller-owned value containing one OHLCV observation in coherent units.
//
// Returns:
//   - error: nil on valid geometry/domain, or a stable input sentinel.
//
// Errors:
//   - ErrNonFinite: any OHLCV field contains NaN or infinity.
//   - ErrDomain: valid finite fields violate positivity, volume or geometry.
//
// Determinism:
//   - Evaluates all five finiteness checks before any OHLCV domain predicate.
//
// Complexity:
//   - Time O(1), Space O(1), zero allocation.
//
// Notes:
//   - The first failing bar wins across a batch; only within one bar is nonfinite globally prioritized.
//
// AI-Hints:
//   - Do not auto-fix invalid OHLC geometry by clipping High, Low or Close.
func validateBar(bar Bar) error {
	if !finite(bar.Open) || !finite(bar.High) || !finite(bar.Low) ||
		!finite(bar.Close) || !finite(bar.Volume) {
		return ErrNonFinite
	}
	if bar.Low <= 0 || bar.High <= 0 || bar.Open <= 0 || bar.Close <= 0 ||
		bar.Volume < 0 || bar.High < bar.Low ||
		bar.Open < bar.Low || bar.Open > bar.High ||
		bar.Close < bar.Low || bar.Close > bar.High {
		return ErrDomain
	}
	return nil
}
