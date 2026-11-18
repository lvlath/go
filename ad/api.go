// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ad provides canonical batch and stream facades for Chaikin A/D.
// This file adds row-oriented and latest-value conveniences without alternative math.
package ad

import "math"

const (
	// float64StorageBytes bounds the byte product for one Values allocation.
	// It is an allocation guard, not a numeric tolerance or a market-data policy.
	float64StorageBytes = 8
)

// New constructs an independent, mutable Chaikin A/D stream at cumulative origin zero.
// Its state is exactly equivalent to the usable zero value Stream{}.
//
// Implementation:
//   - Stage 1: select the single, fixed Chaikin-Neumaier recurrence.
//   - Stage 2: initialize Count, sum, and compensation to zero.
//   - Stage 3: return a new single-owner stream without any retained input memory.
//
// Behavior highlights:
//   - A first Push immediately includes the first candle's signed volume.
//   - No Options, period, context or previous-close state is needed.
//
// Returns:
//   - *Stream: new independent, initialized accumulator; never nil.
//
// Errors:
//   - None; configuration and ring allocation are absent by contract.
//
// Determinism:
//   - The same all-zero state as Stream{} and Reset.
//
// Complexity:
//   - Time O(1), Space O(1), fixed three-scalar state.
//
// Notes:
//   - Caller must serialize access to one Stream if used by multiple goroutines.
//
// AI-Hints:
//   - Use Stream{} if a stack-owned engine is more convenient than New().
//   - AD has no period; copying MFI's Options would create a meaningless public mode.
func New() *Stream { return &Stream{} }

// checkResultCapacity verifies that the output byte count fits an int address space.
// It is a structural allocation bound, not a promise to recover from runtime OOM.
//
// Inputs:
//   - count: strictly positive, already validated result length.
//
// Returns:
//   - error: ErrOverflow when even one float64 Values slice would overflow int bytes.
//
// Determinism:
//   - Exactly one bounds check before the result buffers are allocated.
//
// Complexity:
//   - Time O(1), Space O(1), zero allocations.
//
// AI-Hints:
//   - Runtime memory exhaustion is outside the recoverable error model.
func checkResultCapacity(count int) error {
	if count > math.MaxInt/float64StorageBytes {
		return ErrOverflow
	}
	return nil
}

// Compute calculates the compensated Chaikin A/D line from five OHLCV columns.
// Every output includes its own candle's money-flow contribution, beginning at index zero.
//
// Implementation:
//   - Stage 1: validate Nil, Empty and Shape, then all NonFinite before all Domain errors.
//   - Stage 2: verify output byte bounds and allocate detached Values and Valid arrays.
//   - Stage 3: apply the same nextAD recurrence as Stream.Push, then publish only full success.
//
// Behavior highlights:
//   - A single bar returns its contribution and Valid[0]=true, not OBV's zero seed.
//   - Flat and zero-volume candles are accepted and counted, with zero contribution.
//   - No partial Result is returned if any bar or accumulated total is invalid.
//
// Inputs:
//   - bars: read-only non-nil, equal-length, nonempty OHLCV columns; finite positive
//     prices with Low<=Open,Close<=High, finite nonnegative base Volume.
//
// Returns:
//   - *Result: independent Values and Valid slices of length n, all masks true.
//   - error: nil on full success; otherwise Result is nil.
//
// Errors:
//   - ErrNilInput, ErrEmptyInput, ErrShape: structural validation before element access.
//   - ErrNonFinite: any nonfinite input, before any geometric error in the batch.
//   - ErrDomain: malformed candle geometry, price or volume sign.
//   - ErrOverflow: unrepresentable result storage size or compensated cumulative value.
//
// Determinism:
//   - Both validation passes and the numerical recurrence scan indices ascending.
//   - Every prefix is bitwise equal to sequential Stream.Push of the same bars.
//
// Complexity:
//   - Time O(n), Space O(n) owned result arrays, O(1) additional recurrence state.
//
// Notes:
//   - No input is retained or mutated; concurrent caller mutation during Compute is unsupported.
//   - Runtime out-of-memory is not converted to a recoverable sentinel.
//
// AI-Hints:
//   - Do not run the arithmetic during the first validation pass; later bad input has priority.
//   - Prefer ComputeCandles when the source feed already contains []Bar.
func Compute(bars Bars) (*Result, error) {
	count, err := validateShape(bars)
	if err != nil {
		return nil, err
	}
	if err := validateColumns(bars, count); err != nil {
		return nil, err
	}
	if err := checkResultCapacity(count); err != nil {
		return nil, err
	}

	// Both slices are owned by this Result. All mathematical work is performed
	// on a private scalar state; allocation does not expose partial progress.
	result := &Result{Values: make([]float64, count), Valid: make([]bool, count)}
	var current Stream
	for index := 0; index < count; index++ {
		// The global validation pass already accepted every candle, so calling
		// Push here would repeat checks and add overhead without changing safety.
		candidate, value, stepErr := nextAD(current, barAt(bars, index))
		if stepErr != nil {
			return nil, stepErr
		}
		current = candidate
		result.Values[index], result.Valid[index] = value, true
	}
	return result, nil
}

// ComputeCandles computes A/D directly from a chronological row-oriented []Bar feed.
// It avoids allocating five temporary columns or changing the canonical recurrence.
//
// Implementation:
//   - Stage 1: validate nil/empty rows and all numeric/domain conditions before work.
//   - Stage 2: allocate detached result arrays after checking their byte bounds.
//   - Stage 3: apply nextAD per row and publish only a complete successful result.
//
// Behavior highlights:
//   - First-bar, flat-candle and compensated-overflow behavior matches Compute exactly.
//   - Rows are neither sorted nor filtered; timestamps are caller-owned metadata.
//
// Inputs:
//   - candles: non-nil, nonempty, read-only chronological OHLCV bar slice.
//
// Returns:
//   - *Result: fully detached Values/Valid of length len(candles), or nil on error.
//
// Errors:
//   - ErrNilInput, ErrEmptyInput, ErrNonFinite, ErrDomain, ErrOverflow.
//
// Determinism:
//   - Identical float64 state transitions to Compute and Stream.Push for the same bars.
//
// Complexity:
//   - Time O(n), Space O(n) result storage plus O(1) scalar recurrence state.
//
// Notes:
//   - No row-copy or column-transposition allocation occurs.
//   - Caller mutation of candles during computation is unsupported.
//
// AI-Hints:
//   - When a decoder yields []Bar, do not transpose to five columns solely to use Compute.
func ComputeCandles(candles []Bar) (*Result, error) {
	if err := validateCandles(candles); err != nil {
		return nil, err
	}
	count := len(candles)
	if err := checkResultCapacity(count); err != nil {
		return nil, err
	}

	result := &Result{Values: make([]float64, count), Valid: make([]bool, count)}
	var current Stream
	for index, bar := range candles {
		candidate, value, stepErr := nextAD(current, bar)
		if stepErr != nil {
			return nil, stepErr
		}
		current = candidate
		result.Values[index], result.Valid[index] = value, true
	}
	return result, nil
}

// Latest returns the final A/D value from the canonical five-column batch result.
// It reduces caller indexing boilerplate but does not introduce another numerical engine.
//
// Implementation:
//   - Stage 1: invoke Compute and preserve all of its validation and error order.
//   - Stage 2: read the final Values and Valid entries from a complete Result.
//   - Stage 3: publish the pair, or zero/false with the exact underlying error.
//
// Behavior highlights:
//   - Valid is true for every nonempty, valid A/D batch; there is no warmup.
//
// Inputs:
//   - bars: same owned read-only column input accepted by Compute.
//
// Returns:
//   - value: final finite cumulative base volume on success, zero on error.
//   - valid: true only on successful computation, false on error.
//   - error: identical to Compute's sentinel or nil.
//
// Errors:
//   - Exactly the same sentinels and precedence as Compute.
//
// Determinism:
//   - Bitwise identical to Compute(bars).Values[n-1] on success.
//
// Complexity:
//   - Time O(n), Space O(n), including the temporary complete Result.
//
// Notes:
//   - Latest deliberately favors facade consistency over memory optimization.
//   - Use Stream.Push for allocation-free online latest values.
//
// AI-Hints:
//   - Do not replace this with a different last-window sum that discards history.
func Latest(bars Bars) (value float64, valid bool, err error) {
	result, err := Compute(bars)
	if err != nil {
		return 0, false, err
	}
	last := len(result.Values) - 1
	return result.Values[last], result.Valid[last], nil
}

// LatestCandles returns the final A/D value from a chronological []Bar batch.
// It projects ComputeCandles' canonical Result without changing the recurrence.
//
// Implementation:
//   - Stage 1: delegate all input checks and arithmetic to ComputeCandles.
//   - Stage 2: select the last computed Values and Valid entries.
//   - Stage 3: return the final pair or propagate the exact batch error.
//
// Behavior highlights:
//   - The final value includes every candle, including the first and any flat candles.
//
// Inputs:
//   - candles: non-nil, nonempty, read-only chronological []Bar.
//
// Returns:
//   - value: final finite A/D total on success, zero on error.
//   - valid: true on full success; otherwise false.
//   - error: nil on success, identical to ComputeCandles on failure.
//
// Errors:
//   - Exactly the same sentinel classes and precedence as ComputeCandles.
//
// Determinism:
//   - Bitwise equal to the last element of ComputeCandles(candles).Values.
//
// Complexity:
//   - Time O(n), Space O(n) for the delegated batch calculation.
//
// Notes:
//   - Prefer Stream.Push when only streaming values are needed with O(1) state.
//
// AI-Hints:
//   - Do not conflate a successful zero A/D balance with an invalid observation.
func LatestCandles(candles []Bar) (value float64, valid bool, err error) {
	result, err := ComputeCandles(candles)
	if err != nil {
		return 0, false, err
	}
	last := len(result.Values) - 1
	return result.Values[last], result.Valid[last], nil
}
