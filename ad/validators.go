// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ad validates OHLCV column shape and individual candle geometry.
// This file fixes error precedence without reflection, clipping, or normalization.
package ad

import "math"

// finite classifies one binary64 quantity without changing its numeric value.
// Negative finite numbers remain finite; domain restrictions are checked separately.
//
// Implementation:
//   - Stage 1: reject NaN.
//   - Stage 2: reject positive or negative infinity.
//   - Stage 3: report whether the original value is finite.
//
// Inputs:
//   - value: any float64 number, including invalid input and computed state.
//
// Returns:
//   - bool: false exactly for NaN or infinity.
//
// Determinism:
//   - Exact IEEE-754 classification, with no thresholds or epsilon comparisons.
//
// Complexity:
//   - Time O(1), Space O(1), zero allocations.
//
// AI-Hints:
//   - Checking finiteness alone cannot validate price positivity or volume sign.
func finite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

// validateShape checks the presence and equality of all five OHLCV column lengths.
// No element is accessed until this structural gate succeeds.
//
// Implementation:
//   - Stage 1: reject any nil column, regardless of the others' lengths.
//   - Stage 2: reject an empty but non-nil Open column.
//   - Stage 3: require every remaining column to match the positive Open length.
//
// Inputs:
//   - bars: borrowed OHLCV slice headers; their elements are not read.
//
// Returns:
//   - int: common positive observation count, or zero on error.
//   - error: nil, ErrNilInput, ErrEmptyInput, or ErrShape.
//
// Errors:
//   - ErrNilInput: any input column is nil.
//   - ErrEmptyInput: Open is non-nil but has length zero.
//   - ErrShape: unequal lengths after the presence and empty checks.
//
// Determinism:
//   - Nil, Empty, Shape precedence is fixed independently of element values.
//
// Complexity:
//   - Time O(1), Space O(1), without allocations.
//
// AI-Hints:
//   - Never index a short High or Volume slice before checking all five lengths.
func validateShape(bars Bars) (int, error) {
	if bars.Open == nil || bars.High == nil || bars.Low == nil ||
		bars.Close == nil || bars.Volume == nil {
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

// barAt copies one aligned column index into a small, detached Bar value.
// The caller must already have validated shape; barAt performs no independent checks.
//
// Inputs:
//   - bars: non-nil equal-length columns; index is a valid element position.
//   - index: zero-based observation index within all five columns.
//
// Returns:
//   - Bar: five scalar values without retaining a reference to input storage.
//
// Determinism:
//   - Reads Open, High, Low, Close and Volume at exactly the supplied index.
//
// Complexity:
//   - Time O(1), Space O(1), zero allocations.
//
// AI-Hints:
//   - This helper must not be called before validateShape has succeeded.
func barAt(bars Bars, index int) Bar {
	return Bar{
		Open: bars.Open[index], High: bars.High[index], Low: bars.Low[index],
		Close: bars.Close[index], Volume: bars.Volume[index],
	}
}

// validateBarFinite checks all five fields for NaN and infinity.
// Finiteness is deliberately separated from domain validation for batch precedence.
//
// Inputs:
//   - bar: five caller-supplied, read-only OHLCV scalar values.
//
// Returns:
//   - error: ErrNonFinite for any nonfinite field, otherwise nil.
//
// Determinism:
//   - Field order is Open, High, Low, Close, Volume; all have the same sentinel.
//
// Complexity:
//   - Time O(1), Space O(1), zero allocations.
//
// AI-Hints:
//   - Do not coerce NaN into a zero-volume candle or infinite High into a flat bar.
func validateBarFinite(bar Bar) error {
	if !finite(bar.Open) || !finite(bar.High) || !finite(bar.Low) ||
		!finite(bar.Close) || !finite(bar.Volume) {
		return ErrNonFinite
	}
	return nil
}

// validateBarDomain checks positive prices, nonnegative volume and OHLC geometry.
// All fields must already be finite; invalid observations are rejected, not repaired.
//
// Inputs:
//   - bar: an OHLCV candle that has already passed validateBarFinite.
//
// Returns:
//   - error: ErrDomain on any violation, or nil for a valid candle.
//
// Determinism:
//   - Domain predicates use exact binary64 comparisons without tolerance.
//
// Complexity:
//   - Time O(1), Space O(1), zero allocations.
//
// AI-Hints:
//   - Open must satisfy range geometry even though it does not enter the multiplier.
func validateBarDomain(bar Bar) error {
	if bar.Low <= 0 || bar.High <= 0 || bar.Open <= 0 || bar.Close <= 0 ||
		bar.Volume < 0 || bar.High < bar.Low ||
		bar.Open < bar.Low || bar.Open > bar.High ||
		bar.Close < bar.Low || bar.Close > bar.High {
		return ErrDomain
	}
	return nil
}

// validateBar applies the public single-candle validation order.
// It is shared by Stream.Push and both batch validators.
//
// Implementation:
//   - Stage 1: reject nonfinite OHLCV values.
//   - Stage 2: reject invalid positive-price, volume and geometry conditions.
//   - Stage 3: leave the input untouched when returning success.
//
// Inputs:
//   - bar: caller-owned OHLCV observation copied by value.
//
// Returns:
//   - error: nil, ErrNonFinite, or ErrDomain.
//
// Errors:
//   - ErrNonFinite: checked before all domain predicates.
//   - ErrDomain: checked only for finite candles.
//
// Determinism:
//   - Identical single-candle error classification in batch and stream paths.
//
// Complexity:
//   - Time O(1), Space O(1), no allocations.
//
// AI-Hints:
//   - A flat candle with nonzero Volume is valid and must produce zero contribution.
func validateBar(bar Bar) error {
	if err := validateBarFinite(bar); err != nil {
		return err
	}
	return validateBarDomain(bar)
}

// validateColumns performs the two full-data numeric passes needed for batch precedence.
// A late NaN must be reported before an earlier domain error, without publishing a prefix.
//
// Implementation:
//   - Stage 1: inspect every element for non-finiteness in ascending input order.
//   - Stage 2: inspect candle geometry and volume sign in the same order.
//   - Stage 3: return nil only after the entire series has passed both passes.
//
// Inputs:
//   - bars: non-nil, nonempty and already shape-validated read-only columns.
//   - count: their shared validated length.
//
// Returns:
//   - error: ErrNonFinite, ErrDomain, or nil.
//
// Determinism:
//   - Global category order is NonFinite before Domain; index order is ascending within a category.
//
// Complexity:
//   - Time O(n), Space O(1) extra; no heap allocation.
//
// AI-Hints:
//   - Combining passes into a per-row check would alter the documented error precedence.
func validateColumns(bars Bars, count int) error {
	for index := 0; index < count; index++ {
		if err := validateBarFinite(barAt(bars, index)); err != nil {
			return err
		}
	}
	for index := 0; index < count; index++ {
		if err := validateBarDomain(barAt(bars, index)); err != nil {
			return err
		}
	}
	return nil
}

// validateCandles validates a non-nil row slice using the same batch numeric order.
// It avoids transposing candles or allocating an intermediate column representation.
//
// Implementation:
//   - Stage 1: reject nil then empty row collections.
//   - Stage 2: scan all rows for non-finiteness, then scan all for domain violations.
//   - Stage 3: return success only when the entire series is valid.
//
// Inputs:
//   - candles: caller-owned, read-only chronological []Bar.
//
// Returns:
//   - error: ErrNilInput, ErrEmptyInput, ErrNonFinite, ErrDomain, or nil.
//
// Determinism:
//   - Uses the same global precedence as Compute and exact input row order.
//
// Complexity:
//   - Time O(n), Space O(1), zero allocations.
//
// AI-Hints:
//   - Do not discard flat, zero-volume or first candles during input adaptation.
func validateCandles(candles []Bar) error {
	if candles == nil {
		return ErrNilInput
	}
	if len(candles) == 0 {
		return ErrEmptyInput
	}
	for _, bar := range candles {
		if err := validateBarFinite(bar); err != nil {
			return err
		}
	}
	for _, bar := range candles {
		if err := validateBarDomain(bar); err != nil {
			return err
		}
	}
	return nil
}
