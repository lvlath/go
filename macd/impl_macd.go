// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package macd composes the three SMA-seeded EMA phases into one canonical step.
// This file owns the complete calculation order shared by batch and stream APIs.
package macd

// nextStep derives the next MACD state and Point without changing its input state.
// The caller commits the candidate only after all means and outputs pass numeric admission.
//
// Implementation:
//   - Stage 1: reject a saturated total count before advancing any EMA.
//   - Stage 2: advance fast and slow price EMA candidates in the same order.
//   - Stage 3: once slow is seeded, derive the signed MACD line and advance signal.
//   - Stage 4: once signal is seeded, derive the signed histogram.
//   - Stage 5: check all candidate means and published values together, then return
//     the complete candidate or the unchanged current state; never mutate the caller.
//
// Behavior highlights:
//   - Fast and slow accumulate every accepted close from the same initial index.
//   - Signal only consumes valid MACD lines, starting at close index Slow-1.
//   - A failed step publishes an empty Point and returns the unchanged current state.
//   - A successful step preserves finite, strictly positive observed price means,
//     the same domain Restore requires, and finite signed signal/output values.
//
// Inputs:
//   - current: initialized Stream copied by value, with immutable normalized opts.
//   - price: caller-validated finite positive observation.
//
// Returns:
//   - Stream: candidate recurrence state, ready for one atomic commit.
//   - Point: finite MACD values and validity flags, with +0 for invalid fields.
//   - error: ErrOverflow for an exhausted count or an invalid numerical candidate/output.
//
// Determinism:
//   - Order is fast -> slow -> line -> signal -> histogram -> commit decision.
//
// Complexity:
//   - Time O(1), Space O(1), no dynamic allocations.
//
// Notes:
//   - Push owns input validation; this kernel owns the numeric commit gate.
//   - Candidate-only arithmetic between the gates is bounded scalar floating-point
//     work; an invalid temporary cannot escape, index storage, or change live state.
//
// AI-Hints:
//   - Never commit fast before signal/histogram can still fail.
//   - Do not start the signal mean at close[0] or feed invalid placeholder zeros.
//   - Do not remove mean checks merely because finite valid convex inputs normally
//     cannot overflow. Every computed state must pass the publication boundary.
func nextStep(current Stream, price float64) (Stream, Point, error) {
	if current.count == ^uint64(0) {
		return current, Point{}, ErrOverflow
	}

	// A plain struct copy is the transaction boundary: failures discard all changes.
	candidate := current
	// Both price EMAs consume the same validated observation in fixed order.
	candidate.fast = advanceEMA(current.fast, price, current.opts.Fast)
	candidate.slow = advanceEMA(current.slow, price, current.opts.Slow)
	candidate.count++

	// A zero-value Point keeps all not-yet-ready observations at canonical zero.
	var point Point
	if candidate.slow.seen == periodCount(current.opts.Slow) {
		// A line exists only after the slow seed has consumed Slow price samples.
		line := candidate.fast.mean - candidate.slow.mean
		point.Line, point.LineValid = line, true

		// The signed line has its own seed counter, independent of price count.
		candidate.signal = advanceEMA(current.signal, line, current.opts.Signal)
		if candidate.signal.seen == periodCount(current.opts.Signal) {
			point.Signal, point.SignalValid = candidate.signal.mean, true
			// Do not force a zero histogram: Signal=1 already preserves Line exactly.
			point.Histogram = line - point.Signal
		}
	}

	// One publication gate checks every calculated mean, even during warmup.
	// Signal, if published, is exactly candidate.signal.mean; inactive fields
	// remain zero. A signed Histogram may overflow even when its operands do not.
	// Positive observed price means additionally preserve Restore's domain law.
	if !finite(candidate.fast.mean) || candidate.fast.mean <= 0 ||
		!finite(candidate.slow.mean) || candidate.slow.mean <= 0 ||
		!finite(candidate.signal.mean) || !finite(point.Line) || !finite(point.Histogram) {
		return current, Point{}, ErrOverflow
	}

	// Only a complete, numerically admitted candidate may reach Push's commit.
	return candidate, point, nil
}
