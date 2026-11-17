// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package macd implements the single EMA seed/recurrence used by every MACD phase.
// This file keeps signed arithmetic separate from positive-price validation.
package macd

// emaState stores the arithmetic mean while seeding and the EMA after seeding.
// The count saturates at period, so storage stays O(1) regardless of stream length.
type emaState struct {
	mean float64 // Current online SMA seed or, after seeding, the live EMA value.
	seen uint64  // Accepted samples in the seed phase, capped at its period.
}

// weightedMean computes old*(1-weight)+sample*weight with magnitude-aware grouping.
// A direct sample-old difference is unsafe when signed operands have opposite signs.
//
// Implementation:
//   - For same-sign values, use old+weight*(sample-old) to avoid a raw weighted sum.
//   - For opposite signs, form the two weighted terms separately; their sum is finite
//     when the convex combination is representable even if sample-old overflows.
//
// Numeric policy:
//   - Both operands must be finite, and weight must lie in (0,1] by the caller contract.
//   - The final result is checked by advanceEMA; this helper does not validate prices.
//
// Complexity:
//   - Time O(1), Space O(1), no allocations.
//
// AI-Hints:
//   - Do not replace the signed branch with FMA(sample-old,weight,old):
//     the subtracted difference may already be infinite.
func weightedMean(old, sample, weight float64) float64 {
	if (old >= 0 && sample >= 0) || (old <= 0 && sample <= 0) {
		return old + weight*(sample-old)
	}
	return (1-weight)*old + weight*sample
}

// advanceEMA processes exactly one accepted sample against one private EMA state.
// Its first period observations build an online SMA; subsequent ones use alpha=2/(P+1).
//
// Implementation:
//   - For the first sample, seed the mean without summing unscaled values.
//   - For subsequent warmup samples, incorporate sample with weight 1/nextCount.
//   - After seen==period, apply the exponential coefficient 2/(period+1).
//   - Check finiteness before returning the newly derived value.
//
// Inputs:
//   - state: by-value candidate; the caller's existing state is never mutated.
//   - sample: finite price for price EMAs, or finite signed MACD line for signal EMA.
//   - period: validated integer >=1 and within snapshot count ranges.
//
// Returns:
//   - emaState: candidate online seed or EMA, with saturated count.
//   - error: ErrOverflow if the newly computed mean is unrepresentable.
//
// Determinism:
//   - Every call uses exactly one candidate and a fixed, branch-defined operation order.
//
// Complexity:
//   - Time O(1), Space O(1) with no heap allocation.
//
// Notes:
//   - The arithmetic order is stable for batch/stream parity; it is not correctly
//     rounded arbitrary-precision summation for every possible price history.
//
// AI-Hints:
//   - Converting period to float64 before +1 avoids integer overflow at MaxInt.
//   - The signed signal EMA must use this path, never validateClose.
func advanceEMA(state emaState, sample float64, period int) (emaState, error) {
	if state.seen == 0 {
		return emaState{mean: sample, seen: 1}, nil
	}

	if state.seen < periodCount(period) {
		// Accumulate a bounded mean, not a potentially overflowing sum of closes.
		nextCount := state.seen + 1
		state.mean = weightedMean(state.mean, sample, 1/float64(nextCount))
		if !finite(state.mean) {
			return emaState{}, ErrOverflow
		}
		state.seen = nextCount
		return state, nil
	}

	// Once the seed is complete, use the same recurrence at every later index.
	alpha := 2 / (float64(period) + 1)
	state.mean = weightedMean(state.mean, sample, alpha)
	if !finite(state.mean) {
		return emaState{}, ErrOverflow
	}
	return state, nil
}
