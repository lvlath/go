// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package hmm implements penalized EM objectives and constrained simplex updates.
// This boundary preserves zero-pseudocount semantics and explicit convergence failures.
package hmm

import (
	"context"
	"math"
)

// simplexPenalty evaluates the log Dirichlet MAP regularizer on stochastic parameter arrays.
// Lambda=0 disables the entire term, including formally indeterminate 0*log(0).
//
// Implementation:
//   - Stage 1: Return exact zero when lambda is zero.
//   - Stage 2: For positive lambda, scan all active probability cells and reject any exact zero with -Inf score.
//   - Stage 3: Accumulate lambda*log(p) for every strictly positive cell.
//
// Behavior highlights:
//   - Only prior/transition and categorical emission simplexes are penalized; Gaussians are not.
//
// Inputs:
//   - lambda: nonnegative Dirichlet alpha-1; arrays: copied normalized parameter simplexes.
//
// Returns:
//   - float64: total log prior contribution, or -Inf if a regularized cell is zero.
//
// Errors:
//   - No direct error; fitting objective validation classifies unrepresentable results.
//
// Determinism:
//   - Arrays then elements are visited in the caller-specified order.
//
// Complexity:
//   - Time O(sum of simplex cells), Space O(1).
//
// Notes:
//   - Zero initial probabilities are allowed but yield initial objective -Inf when lambda>0.
//
// AI-Hints:
//   - Do not regularize Gaussian means/variances with a Dirichlet prior.
func simplexPenalty(lambda float64, arrays ...[]float64) float64 {
	if lambda == 0 {
		return 0
	}
	var score float64
	for _, a := range arrays {
		for _, v := range a {
			if v == 0 {
				return math.Inf(-1)
			}
			score += lambda * math.Log(v)
		}
	}
	return score
}

// updateSimplex applies the closed-form Dirichlet-penalized M-step to each probability row.
// Rows with zero expected mass retain previous parameters only when lambda=0.
//
// Implementation:
//   - Stage 1: Allocate a detached row-major output buffer.
//   - Stage 2: For each row compute expected count total and add symmetric lambda to every cell.
//   - Stage 3: Normalize updated rows or copy unchanged dead rows when lambda is zero.
//
// Behavior highlights:
//   - No random reset and no epsilon floor are used to revive dead states.
//
// Inputs:
//   - counts: expected counts; old: previous simplex rows; n: rows; cols: symbols/states; lambda>=0.
//
// Returns:
//   - []float64: detached M-step probabilities, preserving number of rows and columns.
//
// Errors:
//   - No direct error; callers validate the resulting candidate with NewCategorical/NewGaussian.
//
// Determinism:
//   - Row-major ascending index accumulation; no maps or stochastic initialization.
//
// Complexity:
//   - Time O(n*cols), Space O(n*cols).
//
// Notes:
//   - Counts represent expectations over separate sequences; no cross-boundary xi is supplied.
//
// AI-Hints:
//   - For lambda>0, a zero-count row becomes uniform; for lambda=0 it retains old values.
func updateSimplex(counts, old []float64, n, cols int, lambda float64) []float64 {
	result := make([]float64, len(counts))
	for i := 0; i < n; i++ {
		var total float64
		for j := 0; j < cols; j++ {
			total += counts[i*cols+j]
		}
		// A zero-count row has no evidence-driven replacement when lambda=0;
		// preserve the previous stochastic row instead of inventing uniformity.
		if total == 0 && lambda == 0 {
			copy(result[i*cols:(i+1)*cols], old[i*cols:(i+1)*cols])
			continue
		}
		den := total + float64(cols)*lambda
		for j := 0; j < cols; j++ {
			result[i*cols+j] = (counts[i*cols+j] + lambda) / den
		}
	}
	return result
}

// monotonic checks whether a candidate increases the penalized EM objective and has converged.
// It compares the MAP objective, not raw log-likelihood or individual parameter movement.
//
// Implementation:
//   - Stage 1: Allow a finite candidate after an initial negative-infinite Dirichlet penalty.
//   - Stage 2: Reject nonfinite new scores and any strictly negative objective change.
//   - Stage 3: Test nonnegative relative improvement against tolerance*max(1,abs(previous)).
//
// Behavior highlights:
//   - The implementation deliberately rejects even a tiny negative change rather than silently accepting it.
//
// Inputs:
//   - previous, next: consecutive MAP scores; tolerance: nonnegative relative threshold.
//
// Returns:
//   - bool, error: converged flag on nondecreasing score, or a typed numerical failure.
//
// Errors:
//   - ErrNumerical for negative motion or nonfinite candidate score.
//
// Determinism:
//   - The same scalar comparisons are applied to every EM iteration.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Callers keep the last verified model when this check rejects a candidate.
//
// AI-Hints:
//   - A small likelihood decrease may still be a valid *penalized* improvement; compare MAP objectives.
func monotonic(previous, next float64, tolerance float64) (bool, error) {
	if math.IsInf(previous, -1) && finite(next) {
		return false, nil
	}
	if !finite(next) {
		return false, ErrNumerical
	}
	delta := next - previous
	// A declining MAP objective is not convergence; the caller must retain
	// its last verified model rather than publishing this candidate.
	if delta < 0 {
		return false, ErrNumerical
	}
	return delta <= tolerance*math.Max(1, math.Abs(previous)), nil
}

// checkFitInput establishes context and option precedence for both training entry points.
// It delegates all option defaulting to normalizeFit and does not read observation data.
//
// Implementation:
//   - Stage 1: Reject missing or already canceled context.
//   - Stage 2: Validate negative/nonfinite option fields.
//   - Stage 3: Return finalized training options for subsequent model and data validation.
//
// Behavior highlights:
//   - No hidden random seed or regularization is introduced.
//
// Inputs:
//   - ctx: required training context; o: value-copy user training policy.
//
// Returns:
//   - FitOptions and error: effective policy, or an input error before any E-step.
//
// Errors:
//   - ErrInvalidOption for nil context/invalid settings; ctx.Err() for cancellation.
//
// Determinism:
//   - Context is checked before options in the specified order.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Callers receive no partial model after failed input/context validation.
//
// AI-Hints:
//   - Use this from both FitCategorical and FitGaussian to prevent policy divergence.
func checkFitInput(ctx context.Context, o FitOptions) (FitOptions, error) {
	if err := checkContext(ctx); err != nil {
		return o, err
	}
	return normalizeFit(o)
}
