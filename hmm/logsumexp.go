// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package hmm provides stable log-domain addition and vector reduction.
// This boundary preserves the all-impossible -Inf case without producing NaN.
package hmm

import "math"

// LogSumExp computes a numerically stable log(exp(a)+exp(b)) reduction.
// It preserves -Inf for two impossible terms and +Inf for an infinite positive term.
// Implementation:
//   - Stage 1: Handle log-domain infinity sentinels explicitly.
//   - Stage 2: Choose the larger finite argument as a common scale.
//   - Stage 3: Add the nonpositive exponential difference using Log1p.
//
// Behavior highlights:
//   - No direct evaluation of exp(max(a,b)) is performed.
//
// Inputs:
//   - a,b: real log scores; -Inf denotes zero mass; NaN is not a valid log score.
//
// Returns:
//   - float64: stable log-sum, potentially infinite for infinite inputs.
//
// Errors:
//   - None; this scalar function has no error channel and propagates invalid NaN arithmetic.
//
// Determinism:
//   - Fixed pairwise comparison and evaluation order.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Gaussian emission overflow is classified by callers, not repaired by this helper.
//
// AI-Hints:
//   - Test the all--Inf case; naive maximum-shift code otherwise yields NaN.
func LogSumExp(a, b float64) float64 {
	if math.IsInf(a, 1) || math.IsInf(b, 1) {
		return math.Inf(1)
	}
	if math.IsInf(a, -1) {
		return b
	}
	if math.IsInf(b, -1) {
		return a
	}
	if a < b {
		a, b = b, a
	}
	return a + math.Log1p(math.Exp(b-a))
}

// LogSumExpSlice reduces a vector of log scores by a stable maximum-shift identity.
// It never converts small probabilities into raw products or mutates the input slice.
// Implementation:
//   - Stage 1: Find the maximum log score using a fixed left-to-right scan.
//   - Stage 2: Return -Inf/+Inf directly when the maximum has infinite support.
//   - Stage 3: Sum exp(x-max) and add log(sum) back to max.
//
// Behavior highlights:
//   - Empty vectors and all-impossible vectors both yield -Inf.
//
// Inputs:
//   - v: read-only log values, typically finite or -Inf; NaN is outside the valid domain.
//
// Returns:
//   - float64: stable aggregate log score.
//
// Errors:
//   - None; invalid numeric inputs have no separate error channel.
//
// Determinism:
//   - Ascending slice order for maxima and exponent sums.
//
// Complexity:
//   - Time O(len(v)), Space O(1).
//
// Notes:
//   - +Inf input yields +Inf; an all--Inf vector never yields NaN.
//
// AI-Hints:
//   - Normalize a log posterior by subtracting this value only when it is finite.
func LogSumExpSlice(v []float64) float64 {
	m := math.Inf(-1)
	for _, x := range v {
		if x > m {
			m = x
		}
	}
	if math.IsInf(m, -1) {
		return m
	}
	if math.IsInf(m, 1) {
		return m
	}
	var sum float64
	for _, x := range v {
		sum += math.Exp(x - m)
	}
	return m + math.Log(sum)
}

// logProbability converts one prevalidated nonnegative probability to log mass.
// Exact zero becomes -Inf, preserving impossible transitions without any epsilon.
//
// Implementation:
//   - Stage 1: Detect an exact zero cell from a validated simplex.
//   - Stage 2: Publish negative infinity for zero; otherwise compute math.Log(x).
//   - Stage 3: Return the scalar log score unchanged for caller composition.
//
// Behavior highlights:
//   - No positivity threshold: an arbitrarily small valid positive mass stays possible.
//
// Inputs:
//   - x: finite probability in [0,1] validated before the kernel.
//
// Returns:
//   - float64: log mass in [-Inf,0].
//
// Errors:
//   - None directly; invalid probability data is rejected by constructors.
//
// Determinism:
//   - Exact zero check, followed by one math.Log call for positive input.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Gaussian emission log densities are computed separately, not by this helper.
//
// AI-Hints:
//   - Distinguish a zero probability's -Inf from ErrOverflow caused by arithmetic.
func logProbability(x float64) float64 {
	if x == 0 {
		return math.Inf(-1)
	}
	return math.Log(x)
}
