// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ad implements the single compensated cumulative arithmetic kernel.
// This file keeps all intermediate state private so batch and stream cannot diverge.
package ad

import "math"

// neumaierPair computes a finite binary64 sum and a low-order rounding residual.
// The residual recovers significant digits that the high component alone would drop.
//
// Implementation:
//   - Stage 1: form the rounded high component high=left+right and check finiteness.
//   - Stage 2: select the larger-magnitude operand with a fixed >= tie rule.
//   - Stage 3: recover the low component by Neumaier's cancellation-safe formula.
//
// Behavior highlights:
//   - It does not silently saturate overflowing sums or discard correction terms.
//
// Inputs:
//   - left, right: finite binary64 operands representing compatible volume units.
//
// Returns:
//   - high, low: rounded addition and its residual if high is finite.
//   - ok: false if forming the rounded high component overflows.
//
// Determinism:
//   - No dependency on iteration order beyond argument order and >= magnitude ties.
//
// Complexity:
//   - Time O(1), Space O(1), zero allocations.
//
// Notes:
//   - This is compensated binary64 addition, not a general arbitrary-precision accumulator.
//
// AI-Hints:
//   - Do not replace (left-high)+right with (left+right)-high: that loses the residual.
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

// rebaseAccumulator cancels stored correction before adding a large contribution.
// It handles cases where raw sum+delta overflows although the effective total is finite.
//
// Implementation:
//   - Stage 1: fold the prior high and compensation into a reduced pair.
//   - Stage 2: add the current signed contribution to the reduced high component.
//   - Stage 3: combine both retained residuals and verify the effective total.
//
// Behavior highlights:
//   - This exceptional path is used only if the ordinary Neumaier representation fails.
//   - Every intermediate candidate is local; no Stream mutation occurs here.
//
// Inputs:
//   - sum, compensation: finite operands of a previously representable accumulator.
//   - delta: one finite signed-volume contribution.
//
// Returns:
//   - sum, compensation: new finite two-component state if representable.
//   - value: the finite effective total of that state.
//   - error: ErrOverflow if no finite representation is found by this recurrence.
//
// Errors:
//   - ErrOverflow: a rebalanced high, residual, or effective total is nonfinite.
//
// Determinism:
//   - Cancellation is applied in the exact order (sum, compensation), then delta.
//
// Complexity:
//   - Time O(1), Space O(1), no allocations.
//
// Notes:
//   - A failure does not establish that an arbitrary-precision mathematical total is infinite;
//     it establishes that this documented binary64 recurrence cannot represent the state.
//
// AI-Hints:
//   - Never scale every component by 1/2 to hide overflow: it may erase genuine low bits.
//   - A stored correction can be large after Restore and must not be ignored.
func rebaseAccumulator(sum, compensation, delta float64) (float64, float64, float64, error) {
	// Cancel the previous components first, before the large new addition.
	reduced, reducedResidual, ok := neumaierPair(sum, compensation)
	if !ok {
		return 0, 0, 0, ErrOverflow
	}

	// Add the new delta only after potentially opposing historical correction
	// has been applied; an otherwise overflowing high may now be representable.
	high, low, ok := neumaierPair(reduced, delta)
	if !ok {
		return 0, 0, 0, ErrOverflow
	}

	// Both residuals contain distinct rounding information. Neither can be
	// discarded without changing the next accepted candle's behavior.
	correction := reducedResidual + low
	value := high + correction
	if !finite(correction) || !finite(value) {
		return 0, 0, 0, ErrOverflow
	}
	return high, correction, value, nil
}

// nextAccumulator produces one candidate Neumaier state for a signed contribution.
// A failed candidate is discarded by its caller without changing the live Stream.
//
// Implementation:
//   - Stage 1: add contribution to the high component with a recovered residual.
//   - Stage 2: add the residual into historical compensation and check effective total.
//   - Stage 3: return the candidate, or attempt one cancellation-first rebase on overflow.
//
// Behavior highlights:
//   - Compensation recovers the +1 term in +1e16,+1,-1e16.
//   - A finite high alone is insufficient: high+compensation must also be finite.
//
// Inputs:
//   - sum, compensation: finite, representable prior state.
//   - contribution: finite signed money-flow volume of the next accepted candle.
//
// Returns:
//   - sum, compensation, value: complete next state and effective total on success.
//   - error: ErrOverflow when the normal and cancellation-first paths both fail.
//
// Errors:
//   - ErrOverflow: unrepresentable candidate high, correction, or effective total.
//
// Determinism:
//   - Fixed Neumaier order, with the rebase attempted only after a failed ordinary path.
//
// Complexity:
//   - Time O(1), Space O(1), zero allocations.
//
// Notes:
//   - Neither NaN nor Inf is allowed into a successful Stream checkpoint.
//
// AI-Hints:
//   - Do not use sum += contribution, including for supposedly insignificant bars.
//   - Do not treat a finite high component as sufficient without checking compensation.
func nextAccumulator(sum, compensation, contribution float64) (float64, float64, float64, error) {
	high, residual, ok := neumaierPair(sum, contribution)
	if !ok {
		// The historical correction may cancel the large high component, so
		// a failed direct sum is not yet a definitive overflow diagnosis.
		return rebaseAccumulator(sum, compensation, contribution)
	}

	correction := compensation + residual
	value := high + correction
	if !finite(correction) || !finite(value) {
		// A compensation overflow or an overflowing effective total can hide
		// cancellation; try a deterministic rebalance before rejecting it.
		return rebaseAccumulator(sum, compensation, contribution)
	}
	return high, correction, value, nil
}

// nextAD computes one immutable candidate Stream from an already validated Bar.
// It owns the sole transition used by Compute, ComputeCandles, and Stream.Push.
//
// Implementation:
//   - Stage 1: reject Count exhaustion before deriving a new contribution.
//   - Stage 2: form one distance-based multiplier and multiply by base volume.
//   - Stage 3: run the Neumaier recurrence and publish a complete scalar candidate.
//
// Behavior highlights:
//   - The first bar is not a seed-only observation; it adds its own contribution.
//   - Flat candles have zero contribution but still advance Count once.
//
// Inputs:
//   - current: a validated, value-copied Stream state, never mutated here.
//   - bar: finite positive OHLC with valid geometry and nonnegative finite Volume.
//
// Returns:
//   - Stream: complete candidate state, or the original on error.
//   - float64: finite effective A/D balance after this bar, or zero on error.
//   - error: ErrOverflow on Count exhaustion or unrepresentable arithmetic.
//
// Errors:
//   - ErrOverflow: Count is MaxUint64 or compensated accumulation is unrepresentable.
//
// Determinism:
//   - One multiplier and one compensated step per candle in exactly supplied order.
//
// Complexity:
//   - Time O(1), Space O(1), no allocations.
//
// Notes:
//   - Input validity is a precondition, enforced by both public facades.
//
// AI-Hints:
//   - Do not add a previousClose field: the Chaikin multiplier uses the current candle only.
//   - Do not commit Count until the effective total has passed all overflow checks.
func nextAD(current Stream, bar Bar) (Stream, float64, error) {
	if current.count == math.MaxUint64 {
		return current, 0, ErrOverflow
	}

	// Geometric validity bounds the multiplier by [-1,1]. Thus finite Volume
	// cannot overflow solely through this multiplication on a valid bar.
	contribution := moneyFlowMultiplier(bar) * bar.Volume
	if !finite(contribution) {
		// Defensive arithmetic-domain guard: never publish a nonfinite state
		// even if a future multiplier implementation violates its contract.
		return current, 0, ErrOverflow
	}

	high, correction, value, err := nextAccumulator(current.sum, current.compensation, contribution)
	if err != nil {
		return current, 0, err
	}

	// Only the candidate is modified. The public entry point owns the
	// final commit, keeping a failed Push observationally unchanged.
	candidate := current
	candidate.sum, candidate.compensation = high, correction
	candidate.count++
	return candidate, value, nil
}
