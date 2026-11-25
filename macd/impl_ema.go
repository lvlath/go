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
//   - Stage 1: return sample directly for the exact unit-weight identity.
//   - Stage 2: for same-sign values, use old+weight*(sample-old) to avoid a raw weighted sum.
//   - Stage 3: for opposite signs, form the two weighted terms separately; their sum is finite
//     when the convex combination is representable even if sample-old overflows.
//
// Numeric policy:
//   - Both operands must be finite, and weight must lie in (0,1] by the caller contract.
//   - The candidate is checked at the nextStep publication boundary; signed samples
//     do not pass through the positive-price validator.
//
// Complexity:
//   - Time O(1), Space O(1), no allocations.
//
// AI-Hints:
//   - Do not evaluate sample-old at weight=1: cancellation can erase sample entirely.
//   - Do not replace the signed branch with FMA(sample-old,weight,old):
//     the subtracted difference may already be infinite.
func weightedMean(old, sample, weight float64) float64 {
	// Unit weight selects the current binary64 sample, including its sign bit.
	if weight == 1.0 {
		return sample
	}
	// Same-sign subtraction avoids forming an unscaled sum of large operands.
	if (old >= 0 && sample >= 0) || (old <= 0 && sample <= 0) {
		return old + weight*(sample-old)
	}
	// Opposite signs require the weighted terms to be formed before addition.
	return (1-weight)*old + weight*sample
}

// advanceEMA processes exactly one accepted sample against one private EMA state.
// Its first period observations build an online SMA; subsequent ones use alpha=2/(P+1).
//
// Implementation:
//   - Stage 1: for the first sample, seed the mean without summing unscaled values.
//   - Stage 2: for subsequent warmup samples, incorporate sample with weight 1/nextCount.
//   - Stage 3: after seen==period, apply the exponential coefficient 2/(period+1).
//   - Stage 4: return only the candidate; nextStep checks every mean and output together
//     before the complete Stream can be committed.
//
// Inputs:
//   - state: by-value candidate; the caller's existing state is never mutated.
//   - sample: finite price for price EMAs, or finite signed MACD line for signal EMA.
//   - period: validated integer >=1 and within snapshot count ranges.
//
// Returns:
//   - emaState: candidate online seed or EMA, with saturated count.
//   - No error is published by this arithmetic-only helper. The transaction's
//     numeric admission in nextStep classifies invalid candidates as ErrOverflow.
//
// Determinism:
//   - Every call uses exactly one candidate and a fixed, branch-defined operation order.
//
// Complexity:
//   - Time O(1), Space O(1) with no heap allocation.
//
// Notes:
//   - The counter increments only while seen < period <= MaxInt, so it cannot wrap.
//   - The arithmetic order is stable for batch/stream parity; it is not correctly
//     rounded arbitrary-precision summation for every possible price history.
//
// AI-Hints:
//   - Converting period to float64 before +1 avoids integer overflow at MaxInt.
//   - The signed signal EMA must use this path, never validateClose.
func advanceEMA(state emaState, sample float64, period int) emaState {
	// The first observation is the exact one-sample arithmetic mean.
	if state.seen == 0 {
		return emaState{mean: sample, seen: 1}
	}

	if state.seen < periodCount(period) {
		// Accumulate a bounded mean, not a potentially overflowing sum of closes.
		nextCount := state.seen + 1
		state.mean = weightedMean(state.mean, sample, 1/float64(nextCount))
		state.seen = nextCount
		return state
	}

	// Once the seed is complete, use the same recurrence at every later index.
	alpha := 2 / (float64(period) + 1)
	state.mean = weightedMean(state.mean, sample, alpha)
	return state
}
