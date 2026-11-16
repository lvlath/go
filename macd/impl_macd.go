// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package macd composes the three SMA-seeded EMA phases into one canonical step.
// This file owns the complete calculation order shared by batch and stream APIs.
package macd

// nextStep derives the next MACD state and Point without changing its input state.
// The caller commits the candidate only after every required numeric result is finite.
//
// Implementation:
//   - Stage 1: reject a saturated total count before advancing any EMA.
//   - Stage 2: advance fast and slow price EMA candidates in the same order.
//   - Stage 3: once slow is seeded, derive the signed MACD line and advance signal.
//   - Stage 4: once signal is seeded, derive histogram and check representability.
//   - Stage 5: return the whole candidate and Point; no caller-visible mutation occurs here.
//
// Behavior highlights:
//   - Fast and slow accumulate every accepted close from the same initial index.
//   - Signal only consumes valid MACD lines, starting at close index Slow-1.
//   - A failed step publishes an empty Point and returns the unchanged current state.
//
// Inputs:
//   - current: initialized Stream copied by value, with immutable normalized opts.
//   - close: caller-validated finite positive price.
//
// Returns:
//   - Stream: candidate recurrence state, ready for one atomic commit.
//   - Point: finite MACD values and validity flags, with +0 for invalid fields.
//   - error: ErrOverflow if count, line, signal or histogram cannot be represented.
//
// Determinism:
//   - Order is fast -> slow -> line -> signal -> histogram -> commit decision.
//
// Complexity:
//   - Time O(1), Space O(1), no dynamic allocations.
//
// Notes:
//   - This kernel has no public data validation; Push owns that boundary.
//
// AI-Hints:
//   - Never commit fast before signal/histogram can still fail.
//   - Do not start the signal mean at close[0] or feed invalid placeholder zeros.
func nextStep(current Stream, close float64) (Stream, Point, error) {
	if current.count == ^uint64(0) {
		return current, Point{}, ErrOverflow
	}

	// A plain struct copy is the transaction boundary: failures discard all changes.
	candidate := current
	var err error
	candidate.fast, err = advanceEMA(current.fast, close, current.opts.Fast)
	if err != nil {
		return current, Point{}, err
	}
	candidate.slow, err = advanceEMA(current.slow, close, current.opts.Slow)
	if err != nil {
		return current, Point{}, err
	}
	candidate.count++

	var point Point
	if candidate.slow.seen == uint64(current.opts.Slow) {
		// A line exists only after the slow seed has consumed Slow price samples.
		line := candidate.fast.mean - candidate.slow.mean
		if !finite(line) {
			return current, Point{}, ErrOverflow
		}
		point.Line, point.LineValid = line, true

		// The signed line has its own seed counter, independent of close count.
		candidate.signal, err = advanceEMA(current.signal, line, current.opts.Signal)
		if err != nil {
			return current, Point{}, err
		}
		if candidate.signal.seen == uint64(current.opts.Signal) {
			point.Signal, point.SignalValid = candidate.signal.mean, true
			point.Histogram = line - point.Signal
			if !finite(point.Histogram) {
				return current, Point{}, ErrOverflow
			}
		}
	}
	return candidate, point, nil
}
