// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package obv provides batch and streaming entry points for the same cumulative engine.
// This file performs input preparation; numerical policy belongs only to impl_obv.go.
package obv

// New constructs an independent, unconfigured OBV stream.
// The zero value Stream{} behaves identically, including first-bar seeding.
//
// Implementation:
//   - Stage 1: select the single fixed Neumaier recurrence, without options.
//   - Stage 2: initialize the zero count, previous close and accumulator.
//   - Stage 3: return a new stream owned by its caller.
//
// Returns:
//   - *Stream: independent mutable stream; never nil.
//
// Errors:
//   - None; there is no allocation-dependent ring, period or configuration.
//
// Determinism:
//   - Every new stream starts with the same all-zero recurrence state.
//
// Complexity:
//   - Time O(1), Space O(1) for the small stream object.
//
// Notes:
//   - No shared state exists; concurrent mutation of one stream requires caller synchronization.
//
// AI-Hints:
//   - Use Stream{} when a stack-owned engine is more convenient than New().
//   - Reset on a known observation gap; this package never inspects timestamps.
func New() *Stream { return &Stream{} }

// Compute calculates compensated OBV over five aligned chronological OHLCV columns.
// The first accepted bar returns +0; all later entries are valid and include the
// signed volume of their own candle relative to the immediately preceding close.
//
// Implementation:
//   - Stage 1: validate column presence and shape before indexing any observation.
//   - Stage 2: validate every candle before allocating a result or computing arithmetic.
//   - Stage 3: run the canonical immutable kernel and publish only a complete result.
//
// Behavior highlights:
//   - Invalid late candles cannot cause a partially visible result.
//   - Equal close and zero volume both advance observation history.
//
// Inputs:
//   - bars: five caller-owned non-nil, equal-length, nonempty OHLCV columns;
//     OHLC must be finite and positive, Volume finite and nonnegative.
//
// Returns:
//   - *Result: detached Values and Valid slices of input length, every Valid true;
//     nil result on any error.
//
// Errors:
//   - ErrNilInput, ErrEmptyInput, ErrShape: structural validation in that order.
//   - ErrNonFinite, ErrDomain: first malformed candle in increasing index order.
//   - ErrOverflow: count or cumulative arithmetic cannot be represented.
//
// Determinism:
//   - OHLCV columns and transitions are visited in increasing index order.
//   - Bitwise equality with successful sequential Stream.Push calls is guaranteed.
//
// Complexity:
//   - Time O(n), Space O(n) result allocation, O(1) additional stream state.
//
// Notes:
//   - Input memory is never retained or modified; concurrent caller mutation is unsupported.
//   - Runtime out-of-memory is not converted into ErrOverflow.
//
// AI-Hints:
//   - Do not substitute typical price, epsilon comparison or quote volume for close/base volume.
//   - A failed batch returns nil, even if all earlier rows were valid.
func Compute(bars Bars) (*Result, error) {
	// Reject missing or misaligned columns before touching any element. A
	// malformed shape must take precedence over a nonfinite cell in that shape.
	count, err := validateShape(bars)
	if err != nil {
		return nil, err
	}

	// Complete OHLCV validation is a separate pass. It prevents allocating
	// and publishing a prefix when a later candle is malformed.
	for i := 0; i < count; i++ {
		bar := Bar{Open: bars.Open[i], High: bars.High[i], Low: bars.Low[i], Close: bars.Close[i], Volume: bars.Volume[i]}
		if err := validateBar(bar); err != nil {
			return nil, err
		}
	}

	// The output is owned independently of input slices. The private scalar
	// Stream follows exactly the same canonical transitions as Push.
	result := &Result{Values: make([]float64, count), Valid: make([]bool, count)}
	var current Stream
	for i := 0; i < count; i++ {
		bar := Bar{Open: bars.Open[i], High: bars.High[i], Low: bars.Low[i], Close: bars.Close[i], Volume: bars.Volume[i]}
		valueState, value, err := nextOBV(current, bar)
		if err != nil {
			return nil, err
		}
		current = valueState
		result.Values[i], result.Valid[i] = value, true
	}
	return result, nil
}

// ComputeCandles evaluates a chronological row-oriented OHLCV feed without transposition.
// It is a convenience for decoders that already expose []Bar rather than five columns.
//
// Implementation:
//   - Stage 1: reject nil or empty slices and validate each input candle.
//   - Stage 2: allocate result slices once and run the exact same canonical kernel as Compute.
//   - Stage 3: return a detached complete Result, or nil on any failure.
//
// Behavior highlights:
//   - Avoids five temporary transposition arrays while preserving every observation.
//
// Inputs:
//   - candles: non-nil, nonempty, chronological caller-owned []Bar, validated as OHLCV.
//
// Returns:
//   - *Result: detached Values and Valid slices, or nil on error.
//
// Errors:
//   - ErrNilInput, ErrEmptyInput: missing or zero-length input.
//   - ErrNonFinite, ErrDomain: malformed candle; ErrOverflow: arithmetic or count.
//
// Determinism:
//   - Preserves exact row order and the same per-bar binary64 arithmetic as Compute.
//
// Complexity:
//   - Time O(n), Space O(n) output plus O(1) stream state.
//
// Notes:
//   - No timestamp sorting or volume-unit conversion is performed.
//
// AI-Hints:
//   - Prefer this wrapper over allocating five new columns from an existing []Bar.
//   - Do not drop flat-price candles while assembling exchange feed histories.
func ComputeCandles(candles []Bar) (*Result, error) {
	if candles == nil {
		return nil, ErrNilInput
	}
	if len(candles) == 0 {
		return nil, ErrEmptyInput
	}
	// Validate every row before any output allocation or numerical work.
	// The row-oriented path does not need an intermediate column transpose.
	for _, bar := range candles {
		if err := validateBar(bar); err != nil {
			return nil, err
		}
	}

	// This facade deliberately shares nextOBV with Compute and Stream.Push;
	// duplicating signed-volume arithmetic would jeopardize bitwise parity.
	result := &Result{Values: make([]float64, len(candles)), Valid: make([]bool, len(candles))}
	var current Stream
	for i, bar := range candles {
		candidate, value, err := nextOBV(current, bar)
		if err != nil {
			return nil, err
		}
		current = candidate
		result.Values[i], result.Valid[i] = value, true
	}
	return result, nil
}
