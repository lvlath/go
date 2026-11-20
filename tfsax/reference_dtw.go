// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tfsax adapts the accepted exact scalar DTW implementation.
// This file delegates to dtw.Align instead of maintaining a second production recurrence.
package tfsax

import (
	"errors"
	"math"

	"github.com/lvlath/go/dtw"
)

// ExactDTW returns the accepted scalar absolute-cost DTW distance and reachability.
// It is the shared mathematical oracle for exact and heuristic-refinement branches.
//
// Implementation:
//   - Stage 1: assemble the explicit Sakoe-Chiba window and an overflow-aware
//     adapter for dtw.AbsoluteCost; upstream validates sample finiteness.
//   - Stage 2: delegate all dynamic programming to dtw.Align, including band
//     admission, path selection and accumulated-cost overflow classification.
//   - Stage 3: retain upstream sentinel identities and add the research-level
//     error classification; return no usable distance on failure.
//
// Behavior highlights:
//   - Local cost is exactly |a-b| for all finite, representable differences.
//   - Finite operands whose subtraction overflows are classified as arithmetic
//     overflow, not as invalid NaN/Inf input. Off-band pairs are not evaluated.
//   - Endpoint policy is fixed-start/fixed-end with the accepted dtw.Align recurrence.
//
// Inputs:
//   - a, b: non-nil nonempty finite read-only scalar sample vectors.
//   - window: -1 unrestricted, otherwise integer radius >=0.
//
// Returns:
//   - distance: finite for reachable success; +Inf for a structurally
//     unreachable endpoint; zero when an error prevents publication.
//   - reachable: true only when the accepted alignment reaches the endpoint.
//   - error: nil on success, otherwise an errors.Is-compatible local classification
//     joined with all applicable upstream DTW sentinel identities.
//
// Errors:
//   - ErrInvalidInput joined with dtw.ErrNilInput or dtw.ErrEmptyInput for
//     absent or empty sequences.
//   - ErrInvalidOption joined with dtw.ErrInvalidWindow for window < -1.
//   - ErrNonFinite joined with dtw.ErrNaNInf for NaN/Inf input samples.
//   - ErrOverflow joined with dtw.ErrOverflow and dtw.ErrNaNInf for an
//     unrepresentable local absolute cost or terminal accumulated optimum.
//   - Other dtw errors are propagated without replacing their root identity.
//
// Determinism:
//   - Delegates row-major DTW DP and tie rules to the single upstream oracle.
//
// Complexity:
//   - Time O(len(a)*len(b)), including constant-time local-cost checks per
//     admitted cell; Space O(len(b)) for the upstream distance-only rolling rows.
//
// Notes:
//   - This adapter does not mutate or retain caller-owned input slices.
//   - Concurrent input mutation is unsupported; independent calls are safe.
//   - dtw.ErrNaNInf also accompanies dtw.ErrOverflow for numerical compatibility;
//     that does not imply the caller provided an invalid sample.
//
// AI-Hints:
//   - Keep cost and window identical across benchmark branches; do not substitute
//     squared L2 or compute a complete pairwise cost matrix before band admission.
//   - Preserve the overflow-aware callback until the upstream DTW local-cost
//     contract independently distinguishes finite-input arithmetic overflow.
func ExactDTW(a, b []float64, window int) (float64, bool, error) {
	result, err := dtw.Align(a, b, dtw.WithWindow(window), dtw.WithCostFunc(absoluteCostChecked))
	if err != nil {
		// The underlying error determines classification. An overflow must take
		// precedence over its accompanying nonfinite-computation diagnostic.
		switch {
		case errors.Is(err, dtw.ErrOverflow):
			return 0, false, errors.Join(ErrOverflow, err)
		case errors.Is(err, dtw.ErrNaNInf):
			return 0, false, errors.Join(ErrNonFinite, err)
		case errors.Is(err, dtw.ErrNilInput), errors.Is(err, dtw.ErrEmptyInput):
			return 0, false, errors.Join(ErrInvalidInput, err)
		case errors.Is(err, dtw.ErrInvalidWindow):
			return 0, false, errors.Join(ErrInvalidOption, err)
		default:
			return 0, false, err
		}
	}
	if result == nil {
		return 0, false, ErrInvalidInput
	}
	return result.Distance, result.Reachable, nil
}

// absoluteCostChecked adapts the canonical DTW absolute cost to the research
// harness's distinction between invalid input and arithmetic overflow.
//
// Implementation:
//   - Stage 1: invoke dtw.AbsoluteCost without changing its numerical formula.
//   - Stage 2: recognize an infinite cost caused by finite operands; an
//     infinite result from an invalid operand is left to DTW input validation.
//   - Stage 3: attach both upstream numerical sentinels for errors.Is callers.
//
// Behavior highlights:
//   - No pre-scan of the Cartesian product: only cells admitted by dtw.Align
//     are evaluated, so an off-band overflowing pair cannot reject the result.
//   - A finite local cost is passed through unchanged, including signed-zero
//     behavior and subnormal differences.
//
// Inputs:
//   - ai, bj: scalar operands from the read-only, upstream-validated DTW inputs.
//
// Returns:
//   - float64: canonical finite local cost on success, zero on failure.
//   - error: dtw.ErrOverflow joined with dtw.ErrNaNInf when finite operands
//     produce an unrepresentable local cost, or the upstream cost-function error.
//
// Errors:
//   - dtw.ErrOverflow: finite operands produce +Inf in |ai-bj|.
//   - dtw.ErrNaNInf: compatibility classification accompanying overflow.
//
// Determinism:
//   - Pure arithmetic evaluated only in the upstream DP cell traversal order.
//
// Complexity:
//   - Time O(1), Space O(1) per admitted cell; no allocations on success.
//
// Notes:
//   - dtw.Align validates raw sample finiteness before invoking this callback.
//   - The function does not change the classification of NaN/Inf input samples.
//
// AI-Hints:
//   - Do not classify every +Inf local cost as a nonfinite *input*: the original
//     finite operands are required to distinguish arithmetic overflow.
//   - Keep the mathematical cost delegated to dtw.AbsoluteCost; do not round,
//     clip or substitute a finite sentinel for a truly overflowing subtraction.
func absoluteCostChecked(ai, bj float64) (float64, error) {
	cost, err := dtw.AbsoluteCost(ai, bj)
	if err != nil {
		return 0, err
	}

	// Ordinary finite values require no extra arithmetic or tolerance policy.
	// Inspecting the operands only on overflow keeps the successful hot path small.
	if math.IsInf(cost, 1) && finite(ai) && finite(bj) {
		return 0, errors.Join(dtw.ErrOverflow, dtw.ErrNaNInf)
	}
	return cost, nil
}
