// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package rsi implements the sole numerical state transition for Wilder RSI.
// This file owns seed averaging, the recurrence, checked output conversion, and candidate-state rollback.
package rsi

import "math"

const (
	// rsiMinimum and rsiMaximum bound a mathematically valid oscillator output.
	// An invalid calculated value is an error, not a candidate for forced clamping.
	rsiMinimum = 0.0
	rsiMaximum = 100.0

	// rsiNeutral represents a completely flat history with zero average gain and loss.
	rsiNeutral = 50.0
)

// wilderState is the complete private scalar recurrence state after Count accepted prices.
// Keeping the fields as values permits transactional candidate calculation without allocations.
type wilderState struct {
	count uint64 // Successfully accepted closes, including the initial baseline.

	previous float64 // Most recently accepted strictly positive finite close.
	gain     float64 // Running mean of positive price differences, never negative.
	loss     float64 // Running mean of absolute negative differences, never negative.
}

// updateMean performs one nonnegative incremental averaging transition.
// With divisor=k during seed it computes the arithmetic mean of k gains/losses;
// with divisor=P afterward it performs one Wilder exponential smoothing update.
//
// Implementation:
//   - Stage 1: subtract the old mean from the new nonnegative observation.
//   - Stage 2: scale the difference by the positive divisor, then update the mean.
//   - Stage 3: reject any unrepresentable or negative result before publication.
//
// Behavior highlights:
//   - The algorithm never forms a sum of P potentially near-MaxFloat64 observations.
//   - Subtracting two nonnegative finite inputs cannot overflow because their
//     difference has magnitude at most the greater operand.
//
// Inputs:
//   - mean: current finite nonnegative mean.
//   - observation: current finite nonnegative gain or loss.
//   - divisor: finite positive seed count or configured smoothing period.
//
// Returns:
//   - float64: candidate nonnegative finite mean, or zero on error.
//
// Errors:
//   - ErrOverflow: nonfinite or negative computed mean; no caller state is mutated.
//
// Determinism:
//   - Fixed binary64 subtraction, division, addition order on a single scalar path.
//
// Complexity:
//   - Time O(1), Space O(1); no allocation or hidden buffering.
//
// Notes:
//   - Floating-point rounding can erase a sub-ULP contribution; exact real arithmetic
//     is not claimed, and extra compensation would require a different snapshot ABI.
//
// AI-Hints:
//   - Do not replace this recurrence by summing P gains: huge finite values may overflow.
//   - Preserve the arithmetic operation order across batch, stream, and restore.
func updateMean(mean, observation, divisor float64) (float64, error) {
	candidate := mean + (observation-mean)/divisor
	if !finite(candidate) || candidate < 0 {
		return 0, ErrOverflow
	}
	return candidate, nil
}

// rsiFromAverages converts the nonnegative gain/loss means into a bounded RSI.
// It chooses the smaller-to-larger ratio to avoid both an overflowing sum and an overflowing ratio.
//
// Implementation:
//   - Stage 1: verify the mean domain and classify exact zero combinations.
//   - Stage 2: divide smaller by larger, then form the appropriately scaled branch.
//   - Stage 3: reject an invalid calculated output instead of forcing it into [0,100].
//
// Behavior highlights:
//   - Both means zero yields 50; loss only yields 0; gain only yields 100.
//   - Every nonzero denominator is strictly positive, and every computed ratio is <=1.
//
// Inputs:
//   - gain, loss: finite nonnegative Wilder-smoothed magnitudes in price units.
//
// Returns:
//   - float64: finite RSI in [0,100], or zero on ErrOverflow.
//
// Errors:
//   - ErrOverflow: nonfinite/negative means or invalid numeric result.
//
// Determinism:
//   - Exact zeros are classified first; gain>=loss selects the upper branch on ties.
//
// Complexity:
//   - Time O(1), Space O(1), with no allocation.
//
// Notes:
//   - An extremely small positive ratio may underflow to zero in binary64.
//   - The function does not introduce policy thresholds such as 30/70.
//
// AI-Hints:
//   - Never form gain+loss or a potentially huge gain/loss before choosing a branch.
//   - Do not clip a failed computation into the valid range; return ErrOverflow.
func rsiFromAverages(gain, loss float64) (float64, error) {
	if !finite(gain) || !finite(loss) || gain < 0 || loss < 0 {
		return 0, ErrOverflow
	}

	// No movement is a genuinely neutral oscillator; it is not a signal failure.
	if gain == 0 && loss == 0 {
		return rsiNeutral, nil
	}
	if loss == 0 {
		return rsiMaximum, nil
	}
	if gain == 0 {
		return rsiMinimum, nil
	}

	var value float64
	if gain >= loss {
		// loss/gain lies in (0,1], so 1+loss/gain stays within [1,2].
		value = rsiMaximum / (1 + loss/gain)
	} else {
		// gain/loss lies in (0,1), avoiding the unbounded conventional RS.
		ratio := gain / loss
		value = rsiMaximum * (ratio / (1 + ratio))
	}

	// Numeric corruption must remain observable to the caller. No artificial
	// correction (clamp or abs) is mathematically justified here.
	if !finite(value) || value < rsiMinimum || value > rsiMaximum {
		return 0, ErrOverflow
	}
	return value, nil
}

// stepWilder computes one candidate transition for either batch or streaming RSI.
// It never mutates the input state and returns the unchanged original state on error.
//
// Implementation:
//   - Stage 1: reject exhausted Count, or accept the first price as baseline only.
//   - Stage 2: classify the signed delta and compute seed means or Wilder recurrence.
//   - Stage 3: advance Count/previous, derive RSI when warmed, and return a candidate.
//
// Behavior highlights:
//   - The first P differences seed arithmetic means; subsequent differences use divisor P.
//   - Warmup returns (0,false,nil), with first valid output at Count=P+1.
//   - Invalid candidate calculations never replace any part of the original state.
//
// Inputs:
//   - current: private, value-copied state after zero or more accepted prices.
//   - price: previously validated finite positive price, in the caller's price units.
//   - period: strictly positive uint64 smoothing period, normalized from Options.
//
// Returns:
//   - wilderState: complete next state on success, original state on error.
//   - float64/bool: computed RSI and readiness flag, or zero/false before warmup/error.
//
// Errors:
//   - ErrOverflow: exhausted Count or an unrepresentable numerical transition.
//
// Determinism:
//   - One fixed delta classification and arithmetic sequence per accepted close.
//
// Complexity:
//   - Time O(1), Space O(1), including the value-copy state; zero allocations.
//
// Notes:
//   - The public facades own receiver/input validation; the kernel assumes both are valid.
//   - Fixed operation order makes batch, streaming, and restored continuation bitwise identical
//     within the same floating-point execution environment.
//
// AI-Hints:
//   - Never update Count/previous on the destination Stream before this function succeeds.
//   - Do not add a second seed or recurrence implementation inside the batch facade.
func stepWilder(current wilderState, price float64, period uint64) (wilderState, float64, bool, error) {
	// Count counts prices, not changes. Protect the increment before doing any
	// numerical work so the exhausted checkpoint remains byte-for-byte intact.
	if current.count == math.MaxUint64 {
		return current, 0, false, ErrOverflow
	}

	candidate := current
	if current.count == 0 {
		// No delta exists until two prices have been accepted. This transition is
		// intentionally an unobserved baseline, irrespective of the period.
		candidate.previous = price
		candidate.count = 1
		return candidate, 0, false, nil
	}

	// Both prices are finite and strictly positive, so their difference cannot
	// overflow in binary64. A checked guard still catches a broken private state.
	delta := price - current.previous
	if !finite(delta) {
		return current, 0, false, ErrOverflow
	}
	gain, loss := 0.0, 0.0
	if delta > 0 {
		gain = delta
	} else if delta < 0 {
		loss = -delta
	}

	// At Count<=P the pending difference is among the first P changes: use its
	// exact integer seed position as divisor. Later differences use constant P.
	divisor := float64(period)
	if current.count <= period {
		divisor = float64(current.count)
	}

	// Calculate both means against the same original state. A failure in either
	// branch abandons the candidate, including any mean that computed successfully.
	nextGain, err := updateMean(current.gain, gain, divisor)
	if err != nil {
		return current, 0, false, err
	}
	nextLoss, err := updateMean(current.loss, loss, divisor)
	if err != nil {
		return current, 0, false, err
	}
	candidate.gain, candidate.loss = nextGain, nextLoss
	candidate.previous = price
	candidate.count++

	// After exactly P differences the Count becomes P+1 and RSI is published;
	// earlier observations remain masked even though their means have evolved.
	if candidate.count <= period {
		return candidate, 0, false, nil
	}

	value, err := rsiFromAverages(candidate.gain, candidate.loss)
	if err != nil {
		return current, 0, false, err
	}
	return candidate, value, true, nil
}
