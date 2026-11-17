// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package obv implements the sole signed-volume recurrence and its compensated arithmetic.
// This file owns the numerical transition shared by batch and streaming callers.
package obv

import "math"

// neumaierPair adds two finite operands and recovers the low-order rounding residual.
// The residual is preserved separately so a later opposite-signed volume can recover it.
//
// Implementation:
//   - Stage 1: compute the correctly rounded binary64 high component.
//   - Stage 2: subtract the rounded high component from the larger-magnitude operand.
//   - Stage 3: add the smaller operand to obtain Neumaier's residual.
//
// Inputs:
//   - left, right: finite binary64 addends in the same volume unit.
//
// Returns:
//   - high, low: two-component representation of the addition when high is finite.
//   - ok: false when even the high component cannot be represented.
//
// Determinism:
//   - Operand order and the >= magnitude tie rule are fixed.
//
// Complexity:
//   - Time O(1), Space O(1), no heap allocation.
//
// AI-Hints:
//   - Do not replace the compensation expression with (left+right)-high;
//     that subtraction cannot recover bits already discarded by rounding.
func neumaierPair(left, right float64) (high, low float64, ok bool) {
	high = left + right
	if !finite(high) {
		return 0, 0, false
	}
	if math.Abs(left) >= math.Abs(right) {
		low = (left - high) + right
	} else {
		low = (right - high) + left
	}
	return high, low, true
}

// rebaseOverflow evaluates a three-term compensated total after an overflowing
// intermediate addition. Cancellation is performed before any same-sign addition.
// This is needed because an overflowing sum+delta does not prove that
// sum+compensation+delta is unrepresentable.
//
// Implementation:
//   - Stage 1: cancel the opposing compensation against the prior high component.
//   - Stage 2: add the contribution to that reduced high component.
//   - Stage 3: retain both rounding residuals and check the effective total.
//
// Inputs:
//   - current: validated, already representable stream accumulator.
//   - delta: finite signed volume for the new close transition.
//
// Returns:
//   - sum, compensation, value: finite, rebalanced state or ErrOverflow.
//
// Errors:
//   - ErrOverflow: no supported finite compensated representation can be formed.
//
// Determinism:
//   - Performs cancellation in the fixed order (sum, compensation), then delta.
//
// Complexity:
//   - Time O(1), Space O(1), zero allocations.
//
// Notes:
//   - A rebase is exceptional and preserves explicit residuals; it is not a
//     guarantee of arbitrary precision or correctly rounded infinite-precision OBV.
//
// AI-Hints:
//   - Do not scale every term by 1/2 and erase compensation: that introduces
//     avoidable underflow and can discard the very low bits being protected.
func rebaseOverflow(current Stream, delta float64) (sum, compensation, value float64, err error) {
	// First apply the previously retained correction against the high sum.
	// This can make room for a contribution that overflowed the raw high sum.
	reduced, residual, ok := neumaierPair(current.sum, current.compensation)
	if !ok {
		return 0, 0, 0, ErrOverflow
	}
	// Reintroduce the current candle's signed contribution only after the
	// cancellation. Preserve its own rounding residual as another low term.
	high, low, ok := neumaierPair(reduced, delta)
	if !ok {
		return 0, 0, 0, ErrOverflow
	}
	// Both low terms are genuine recurrence data. If this chosen two-component
	// representation cannot remain finite, reject rather than erase a residual.
	correction := residual + low
	value = high + correction
	if !finite(correction) || !finite(value) {
		return 0, 0, 0, ErrOverflow
	}
	return high, correction, value, nil
}

// nextOBV computes one immutable candidate state from a validated candle.
// It never modifies current; only the calling facade commits a successful candidate.
//
// Implementation:
//   - Stage 1: reject counter exhaustion and seed only the first previous close.
//   - Stage 2: classify the exact close comparison and sign the current volume.
//   - Stage 3: update Neumaier's pair, check the effective total, and publish a candidate.
//
// Behavior highlights:
//   - Equal-close and zero-volume bars advance the observation count, not the sum.
//   - The first bar's volume is never passed to the accumulator.
//   - When the high addend overflows, a cancellation-first rebase is attempted.
//
// Inputs:
//   - current: an unaliased value copy of the previously accepted stream state.
//   - bar: a previously validated OHLCV candle; its close and volume are read-only.
//
// Returns:
//   - Stream: next candidate, or the original state on failure.
//   - float64: finite effective OBV (positive zero for the first bar).
//   - error: ErrOverflow on counter or unrepresentable arithmetic.
//
// Determinism:
//   - Each input bar is classified once, with strict binary64 > and < comparisons.
//
// Complexity:
//   - Time O(1), Space O(1), without heap allocation.
//
// AI-Hints:
//   - Never commit previousClose or count before the compensated total is valid.
//   - A zero signed contribution is still a real observation and must not be skipped.
func nextOBV(current Stream, bar Bar) (Stream, float64, error) {
	// Reject count exhaustion before making even a tentative price update.
	// A failed Push must be observationally identical to no Push at all.
	if current.count == math.MaxUint64 {
		return current, 0, ErrOverflow
	}

	// Modify only the value-copy candidate: the caller commits it after all
	// checks succeed, including the exceptional compensated rebase.
	candidate := current
	candidate.count++
	candidate.previousClose = bar.Close
	// The first bar establishes the reference Close. In particular, it never
	// adds even a finite, nonzero Volume to the accumulator.
	if current.count == 0 {
		return candidate, 0, nil
	}

	// Classify by the exact comparison with the LAST accepted close. Equal
	// closes and zero volumes still count as distinct observations.
	var delta float64
	switch {
	case bar.Close > current.previousClose:
		delta = bar.Volume
	case bar.Close < current.previousClose:
		delta = -bar.Volume
	}

	// A zero signed contribution changes only the new Count/previousClose.
	// It cannot overflow through the bar's possibly enormous raw Volume.
	if delta == 0 {
		value := current.sum + current.compensation
		if !finite(value) {
			return current, 0, ErrOverflow
		}
		return candidate, value, nil
	}

	// Neumaier's rounding residual protects low-order signed contributions
	// that an ordinary sum += delta operation would permanently discard.
	high, residual, ok := neumaierPair(current.sum, delta)
	if !ok {
		newSum, newCorrection, value, err := rebaseOverflow(current, delta)
		if err != nil {
			return current, 0, err
		}
		candidate.sum, candidate.compensation = newSum, newCorrection
		return candidate, value, nil
	}

	correction := current.compensation + residual
	value := high + correction
	if !finite(correction) || !finite(value) {
		// A valid finite total can sometimes be hidden by overflow in a
		// low-component addition. Rebalance before classifying the failure.
		newSum, newCorrection, rebasedValue, err := rebaseOverflow(current, delta)
		if err != nil {
			return current, 0, err
		}
		candidate.sum, candidate.compensation = newSum, newCorrection
		return candidate, rebasedValue, nil
	}
	// Publish the high and low components together. The caller will commit
	// the candidate only after the effective total has passed finiteness checks.
	candidate.sum, candidate.compensation = high, correction
	return candidate, value, nil
}
