// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package match implements the deterministic legacy greedy matching kernel.
// It computes detached pairs and never mutates Christofides adjacency lists.
package match

import "math"

// greedySelect matches each last remaining vertex with its cheapest partner.
// It preserves the historic strict-less/tolerance/index rule and swap-delete.
// Implementation:
//   - Stage 1: Copy the ordered subset into a local remaining buffer.
//   - Stage 2: Pop the last vertex, scan partners in remaining order.
//   - Stage 3: Save selected pair and directional weight, then swap-delete.
//
// Behavior highlights:
//   - This is not global minimum-edge greedy and is not an optimum solver.
//   - All weights are read from the already validated full input copy.
//
// Inputs:
//   - vertices: valid original subset; weights: detached source costs.
//
// Returns:
//   - []Pair: oriented selection-order pairs, float64: raw checked cost.
//
// Errors:
//   - ErrInvalidMatching, ErrNoMatching, ErrNumericOverflow as applicable.
//
// Determinism:
//   - Last-element pivot; left-to-right candidate scan; original-index tie-break.
//
// Complexity:
//   - Time O(k^2), Space O(k) beyond the input matrix copy.
//
// Notes:
//   - The public facade later normalizes/sorts pairs; it does not recompute cost.
//
// AI-Hints:
//   - Never replace swap-delete with a stable delete or global edge sort.
func greedySelect(vertices []int, weights completeWeights) ([]Pair, float64, error) {
	// Keep the caller's subset order intact: pivot selection consumes its final
	// position and swap-delete subsequently changes only this private working copy.
	remaining := append([]int(nil), vertices...)
	pairs := make([]Pair, 0, len(remaining)/2)
	total := 0.0

	for len(remaining) > 1 {
		last := len(remaining) - 1
		from := remaining[last]
		remaining = remaining[:last]
		if from < 0 || from >= weights.n {
			return nil, 0, ErrInvalidMatching
		}

		// Preserve the legacy comparison exactly. A candidate with a slightly
		// larger weight may win within symTol when its original index is lower;
		// this is a deterministic policy, not a global minimum-edge search.
		bestIndex := matchingUnmatched
		bestWeight := math.Inf(1)
		for candidateIndex := 0; candidateIndex < len(remaining); candidateIndex++ {
			to := remaining[candidateIndex]
			if to < 0 || to >= weights.n {
				return nil, 0, ErrInvalidMatching
			}
			weight := weights.at(from, to)
			if bestIndex == matchingUnmatched ||
				weight < bestWeight ||
				(math.Abs(weight-bestWeight) <= symTol && to < remaining[bestIndex]) {
				bestIndex = candidateIndex
				bestWeight = weight
			}
		}
		if bestIndex == matchingUnmatched || math.IsInf(bestWeight, 0) {
			return nil, 0, ErrNoMatching
		}

		// Swap-delete is intentionally not a stable removal: the index moved
		// into bestIndex determines the candidate order of subsequent pivots.
		last = len(remaining) - 1
		to := remaining[bestIndex]
		remaining[bestIndex] = remaining[last]
		remaining = remaining[:last]
		// Retain the selection direction until the public facade canonicalizes
		// endpoints. Sum the original selected cost, not a rounded display cost.
		pairs = append(pairs, Pair{U: from, V: to})
		total += bestWeight
		if math.IsNaN(total) || math.IsInf(total, 0) {
			return nil, 0, ErrNumericOverflow
		}
	}
	return pairs, total, nil
}
