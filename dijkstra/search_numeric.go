// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dijkstra checks target-search arithmetic before mutating shortest-path state.
// This file certifies binary64 sums, exact cutoffs and balanced half-differences.
package dijkstra

import "math"

// splitSearchSum separates a finite two-operand sum into a rounded value and residual.
// With finite intermediate results the represented binary values satisfy a+b=hi+lo exactly.
// Implementation:
//   - Stage 1: Put the larger-magnitude operand first for FastTwoSum's precondition.
//   - Stage 2: Round the sum and compensated subtraction explicitly to float64.
//   - Stage 3: Return the exact residual, or signal arithmetic overflow.
//
// Behavior highlights:
//   - This is a short-lived arithmetic certificate, not an approximate path accumulator.
//   - No tolerance or hidden rounding-policy switch is used.
//
// Inputs:
//   - left, right: finite binary64 numbers whose exact represented-value sum is considered.
//
// Returns:
//   - hi: rounded sum; lo: exact residual; overflow: true if hi is infinite.
//
// Errors:
//   - None; callers classify overflow and inexact storage by their domains.
//
// Determinism:
//   - Uses a fixed operand magnitude rule and explicit binary64 rounding points.
//
// Complexity:
//   - Time O(1), Space O(1); a fixed number of floating-point operations.
//
// Notes:
//   - The exactness identity requires IEEE binary64 round-to-nearest operations
//     and finite operands/intermediates; successful residuals are representable here.
//
// AI-Hints:
//   - Do not replace with Kahan applied only after heap decisions.
//   - Do not omit the magnitude swap, which FastTwoSum requires.
func splitSearchSum(left, right float64) (hi, lo float64, overflow bool) {
	if math.Abs(left) < math.Abs(right) {
		left, right = right, left
	}
	hi = float64(left + right)
	if math.IsInf(hi, 0) {
		return hi, 0, true
	}

	// FastTwoSum: hi is rounded once; z and lo recover the missing low part.
	// The conversions make the intended intermediate rounding explicit in Go.
	z := float64(hi - left)
	lo = float64(right - z)
	return hi, lo, false
}

// searchSumAboveBound compares an exact represented-binary sum with one bound.
// It is used for cutoff decisions and does not require the sum to fit one float64.
// Implementation:
//   - Stage 1: Split the sum or recognize positive overflow.
//   - Stage 2: Compare the rounded high part against the bound.
//   - Stage 3: Use the residual's sign only when the high part equals the bound.
//
// Behavior highlights:
//   - An overflowing sum of non-negative operands is above every finite bound.
//
// Inputs:
//   - left, right: finite non-negative addends in original edge-cost units.
//   - bound: finite non-negative inclusive limit.
//
// Returns:
//   - bool: true exactly when the mathematical binary sum is greater than bound.
//
// Errors:
//   - None; bound and operands are validated by the calling search stages.
//
// Determinism:
//   - Strict exact comparison without a non-transitive epsilon tie-break.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - A value above the bound may be discarded before classifying storage precision.
//
// AI-Hints:
//   - A rounded hi equal to bound is not sufficient: inspect lo before admitting.
func searchSumAboveBound(left, right, bound float64) bool {
	hi, lo, overflow := splitSearchSum(left, right)
	if overflow {
		return true
	}
	return hi > bound || (hi == bound && lo > 0)
}

// addSearchCost extends one original-cost path prefix by one finite edge weight.
// It enforces a single exact represented-binary objective for target Search.
// Implementation:
//   - Stage 1: Decide finite-cutoff admission using an exact sum predicate.
//   - Stage 2: Reject unbounded distance overflow separately from inexact storage.
//   - Stage 3: Return a finite, exactly representable candidate for strict relaxation.
//
// Behavior highlights:
//   - Candidates above a finite cutoff are outside the search policy, including overflow.
//   - Admitted candidate prefixes must be exactly representable in one float64.
//
// Inputs:
//   - prefix: finite non-negative, already certified original path cost.
//   - weight: finite non-negative original edge weight, validated by classifyWeight.
//   - limit: finite non-negative inclusive cutoff, or +Inf for no cutoff.
//
// Returns:
//   - cost: exact represented-binary sum when allowed; allowed: cutoff membership.
//   - error: non-nil only for admissible candidates without a finite exact storage value.
//
// Errors:
//   - ErrDistanceOverflow for a positive infinite rounded sum with unbounded cutoff.
//   - ErrInexactArithmetic when the admitted exact sum cannot be stored in binary64.
//
// Determinism:
//   - Fixed FastTwoSum operands, strict cutoff and no epsilon comparisons.
//
// Complexity:
//   - Time O(1), Space O(1) per attempted relaxation.
//
// Notes:
//   - A dominated but admitted inexact candidate still reports its numeric failure.
//   - The legacy SSSP runner retains its distinct rounded-candidate cutoff policy.
//
// AI-Hints:
//   - Do not compare weight with limit-prefix as a substitute for the exact predicate.
//   - Do not convert arithmetic overflow into an unreachable target.
func addSearchCost(prefix, weight, limit float64) (cost float64, allowed bool, err error) {
	if !math.IsInf(limit, 1) && searchSumAboveBound(prefix, weight, limit) {
		return 0, false, nil
	}

	sum, residual, overflow := splitSearchSum(prefix, weight)
	if overflow {
		return 0, false, ErrDistanceOverflow
	}
	if residual != 0 {
		return 0, false, ErrInexactArithmetic
	}
	return sum, true, nil
}

// searchSumAtLeastBound compares a sum of two non-negative frontier minima to mu.
// It provides the exact lower-bound predicate for bidirectional termination.
// Implementation:
//   - Stage 1: Obtain a rounded high sum and its error-free residual.
//   - Stage 2: Recognize positive overflow as above every finite incumbent.
//   - Stage 3: Compare the high part and use the residual for exact equality.
//
// Behavior highlights:
//   - Equality permits stopping; a rounded equality with negative residual does not.
//   - The lower-bound sum is never stored in a distance label or heap entry.
//
// Inputs:
//   - left, right: finite non-negative clean heap minima, in original-cost units.
//   - bound: finite incumbent original cost from a certified complete witness.
//
// Returns:
//   - bool: true exactly when the represented-binary sum is at least bound.
//
// Errors:
//   - None; caller guarantees the finite non-negative input domain.
//
// Determinism:
//   - Fixed FastTwoSum rounding/correction, without an epsilon tolerance.
//
// Complexity:
//   - Time O(1), Space O(1) for a fixed number of binary64 operations.
//
// Notes:
//   - Overflow here is a comparison fact, not an overflowing route label.
//
// AI-Hints:
//   - Do not replace this with left+right>=bound: rounding can stop too early.
func searchSumAtLeastBound(left, right, bound float64) bool {
	hi, lo, overflow := splitSearchSum(left, right)
	return overflow || hi > bound || (hi == bound && lo >= 0)
}

// searchThreeSumAboveBound checks whether three finite non-negative costs
// exceed an inclusive finite cutoff before materializing an intermediate join.
// Implementation:
//   - Stage 1: Express a+b+c>bound as the equivalent a+b>bound-c.
//   - Stage 2: Split both two-operand expressions into exact high/low components.
//   - Stage 3: Compare rounded highs, then residuals when the highs are equal.
//
// Behavior highlights:
//   - A full bridge outside the cutoff is pruned even if a+b is not exactly storable.
//   - Both sides are exact represented-binary quantities, not rounded path costs.
//
// Inputs:
//   - a, b, c: finite non-negative exact original costs from certified segments.
//   - bound: finite non-negative original-cost cutoff.
//
// Returns:
//   - bool: true exactly when the mathematical three-term sum exceeds bound.
//
// Errors:
//   - None; operands and bound are validated by the caller.
//
// Determinism:
//   - The high part orders distinct rounded sums; the residual orders equal highs.
//
// Complexity:
//   - Time O(1), Space O(1); no path-length accumulator or heap allocation.
//
// Notes:
//   - Since bound and c are non-negative finite values, bound-c cannot overflow.
//   - Correctly rounded binary64 is monotone: different high parts preserve the
//     order of their underlying exact sums; equal highs require the low parts.
//
// AI-Hints:
//   - Do not check a+b first with addSearchCost: inexact intermediate storage
//     must not prevent rejecting a complete crossing already above the cutoff.
func searchThreeSumAboveBound(a, b, c, bound float64) bool {
	sumHi, sumLo, overflow := splitSearchSum(a, b)
	if overflow {
		return true
	}
	remainingHi, remainingLo, _ := splitSearchSum(bound, -c)
	return sumHi > remainingHi || (sumHi == remainingHi && sumLo > remainingLo)
}

// certifySearchKeySum materializes one exactly representable signed key operation.
// It is shared by normalized heuristic offsets and distance-plus-offset priorities.
// Implementation:
//   - Stage 1: Split the exact represented-binary sum into high and residual parts.
//   - Stage 2: Separate a non-finite materialized key from finite inexact storage.
//   - Stage 3: Return the high part only when its residual is exactly zero.
//
// Behavior highlights:
//   - Both positive and negative operands are permitted; no priority is clamped.
//   - Every successful key is finite and exact, making heap ordering strict.
//
// Inputs:
//   - left, right: validated finite float64 operands; their units are edge costs.
//
// Returns:
//   - float64: exact materialized sum, or zero with a non-nil error.
//
// Errors:
//   - ErrSearchKeyOverflow when the rounded sum is infinite.
//   - ErrInexactArithmetic when finite rounding loses a non-zero residual.
//
// Determinism:
//   - Fixed FastTwoSum magnitude ordering and exact zero-residual requirement.
//
// Complexity:
//   - Time O(1), Space O(1) per key materialization.
//
// Notes:
//   - An original-cost overflow is separately classified by addSearchCost.
//
// AI-Hints:
//   - Do not use a final-summation correction after heap pops: a wrong priority
//     comparison cannot be repaired by checking the returned path afterward.
func certifySearchKeySum(left, right float64) (float64, error) {
	hi, lo, overflow := splitSearchSum(left, right)
	if overflow {
		return 0, ErrSearchKeyOverflow
	}
	if lo != 0 {
		return 0, ErrInexactArithmetic
	}
	return hi, nil
}

// balancedPotentialHalf is the exact power-of-two scale in p=(hT-hS)/2.
// A named mathematical factor prevents accidental substitutions in signed arithmetic.
const balancedPotentialHalf = 0.5

// certifyBalancedPotential computes the exact half-difference of two estimates.
// It avoids overflowing a finite balanced potential when hT-hS exceeds MaxFloat64.
// Implementation:
//   - Stage 1: Certify the signed difference using the shared error-free two-sum.
//   - Stage 2: For finite differences, halve exactly and check subnormal rounding.
//   - Stage 3: For overflowing differences, halve each large input before summing.
//
// Behavior highlights:
//   - The mathematical result (hT-hS)/2 is retained only when exact in float64.
//   - Opposite large estimates can yield a legal potential even if subtraction overflows.
//
// Inputs:
//   - toTarget: finite frozen target estimate, in original edge-cost units.
//   - fromSource: finite frozen source estimate, in the same units.
//
// Returns:
//   - float64: exactly representable balanced potential p(v), or zero on error.
//
// Errors:
//   - ErrInexactArithmetic when the half-difference needs unavailable precision.
//   - ErrSearchKeyOverflow if a required potential materialization overflows.
//
// Determinism:
//   - Fixed signed operations and explicit exactness checks, with no epsilon.
//
// Complexity:
//   - Time O(1), Space O(1), independent of graph size.
//
// Notes:
//   - A half of an odd subnormal is not representable, even when the input is finite.
//
// AI-Hints:
//   - Never calculate (toTarget-fromSource)/2 with unchecked float64 subtraction.
//   - Do not silently round a reduced potential: ordering and stopping proofs need exact keys.
func certifyBalancedPotential(toTarget, fromSource float64) (float64, error) {
	difference, residual, overflow := splitSearchSum(toTarget, -fromSource)
	if !overflow {
		if residual != 0 {
			return 0, ErrInexactArithmetic
		}
		half := difference * balancedPotentialHalf
		if half*2 != difference {
			return 0, ErrInexactArithmetic
		}
		return half, nil
	}

	// Divide BEFORE adding only when the full signed difference exceeds the
	// finite float64 range. Both halves must retain their exact input values.
	left := toTarget * balancedPotentialHalf
	right := -fromSource * balancedPotentialHalf
	if left*2 != toTarget || right*2 != -fromSource {
		return 0, ErrInexactArithmetic
	}
	return certifySearchKeySum(left, right)
}
