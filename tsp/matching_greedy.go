// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp implements deterministic greedy matching for explicit weak Christofides mode.
// Greedy matching is a selected heuristic policy and never proves the formal
// Christofides approximation ratio.
package tsp

import (
	"errors"
	"math"

	"github.com/lvlath/go/matrix"
)

// greedyMatch appends a deterministic greedy perfect matching over odd vertices.
//
// Implementation:
//   - Stage 1: Validate odd cardinality and trivial empty input.
//   - Stage 2: Repeatedly choose the unmatched pair with the cheapest finite edge.
//   - Stage 3: Append each selected pair as two undirected adjacency half-edges.
//
// Behavior highlights:
//   - Explicit heuristic mode.
//   - Does not prove the Christofides 1.5 approximation ratio.
//   - Deterministic tie-breaks follow local odd-index scan order.
//
// Inputs:
//   - odd: odd-degree MST vertices.
//   - dist: final symmetric complete matrix.
//   - adj: mutable Christofides multigraph adjacency.
//
// Returns:
//   - error: nil when every odd vertex is paired.
//
// Errors:
//   - ErrInvalidMatching for odd cardinality or corrupt local state.
//   - ErrIncompleteGraph when no finite pair can be selected.
//   - Matrix/tour sentinels from edge cost reads.
//
// Determinism:
//   - Fixed i,j scan order and first strict best pair selection.
//
// Complexity:
//   - Time O(k^3) in the simple repeated-scan implementation, Space O(k).
//
// Notes:
//   - This function mutates adj on success.
//
// AI-Hints:
//   - Do not publish ChristofidesApproximationRatio after this mode.
//   - Do not use this function when exact MWPM was requested.
func greedyMatch(odd []int, dist matrix.Matrix, adj [][]int) error {
	oddCount := len(odd)
	// oddCount==0 is a valid (degenerate) case - nothing to do.
	if oddCount == 0 {
		return nil
	}
	if (oddCount & 1) == 1 {
		return ErrInvalidMatching
	}

	remaining := append([]int(nil), odd...)

	var (
		fromVertex     int
		toVertex       int
		lastIndex      int
		bestIndex      int
		candidateIndex int
		weight         float64
		bestWeight     float64
		err            error
	)

	// Main matching loop: execute O(k) iterations, matching exactly two vertices per cycle.
	// Pops the last element from the 'remaining' buffer to serve as the baseline endpoint.
	for len(remaining) > 1 {
		lastIndex = len(remaining) - 1
		fromVertex = remaining[lastIndex]
		remaining = remaining[:lastIndex]

		// Structural assertion: verify that the popped vertex fits within the global multigraph scope.
		if fromVertex < 0 || fromVertex >= len(adj) {
			return ErrInvalidMatching
		}

		// Initialize local state to track the cheapest candidate partner available in the remaining pool.
		bestIndex = matchingUnmatched
		bestWeight = math.Inf(1)

		// Linear scan over the remaining pool to discover the absolute closest available partner vertex.
		for candidateIndex = 0; candidateIndex < len(remaining); candidateIndex++ {
			// Bounds check: validate that the candidate vertex is safe for global adjacency mapping.
			toVertex = remaining[candidateIndex]
			if toVertex < 0 || toVertex >= len(adj) {
				return ErrInvalidMatching
			}

			// Retrieve the exact cost of the edge from the underlying TSP distance matrix.
			weight, err = edgeCost(dist, fromVertex, toVertex)
			if err != nil {
				// If the edge is missing (+Inf sentinel), skip it and let final feasibility checks decide.
				if errors.Is(err, ErrIncompleteGraph) {
					continue
				}

				return err
			}

			// Deterministic tie-breaking selection: pick the cheapest weight.
			// If costs match within tolerance, resolve ties by selecting the smaller original vertex index.
			if bestIndex == matchingUnmatched ||
				weight < bestWeight ||
				(math.Abs(weight-bestWeight) <= symTol && toVertex < remaining[bestIndex]) {
				bestIndex = candidateIndex
				bestWeight = weight
			}
		}

		// Graph integrity check: fail if the current vertex is completely isolated from all remaining candidates.
		if bestIndex == matchingUnmatched || math.IsInf(bestWeight, 1) {
			return ErrIncompleteGraph
		}

		// O(1) buffer contraction: replace the selected partner at 'bestIndex' with the element at the
		// end of the slice, then truncate. This preserves efficiency without triggering slice shifts.
		lastIndex = len(remaining) - 1
		toVertex = remaining[bestIndex]
		remaining[bestIndex] = remaining[lastIndex]
		remaining = remaining[:lastIndex]

		// Mutate the multigraph by appending the matched pair as an undirected edge.
		adj[fromVertex] = append(adj[fromVertex], toVertex)
		adj[toVertex] = append(adj[toVertex], fromVertex)
	}

	return nil
}

// greedyMatchAtomic runs greedyMatch and rolls back adjacency on failure.
// It is the Christofides-safe wrapper for deterministic greedy matching.
//
// Implementation:
//   - Stage 1: Snapshot adjacency lengths.
//   - Stage 2: Run greedyMatch.
//   - Stage 3: Roll back all appended edges if greedyMatch returns an error.
//
// Behavior highlights:
//   - Successful execution preserves greedyMatch edge order.
//   - Failed execution leaves adj length-equivalent to its input state.
//   - Does not hide greedyMatch errors.
//
// Inputs:
//   - odd: odd-degree MST vertices.
//   - dist: final complete distance matrix.
//   - adj: mutable Christofides multigraph adjacency.
//
// Returns:
//   - error: nil on success or greedyMatch sentinel failure.
//
// Errors:
//   - Same as greedyMatch.
//
// Determinism:
//   - Same matching order as greedyMatch.
//
// Complexity:
//   - Time O(k^2+n), Space O(k+n).
//
// Notes:
//   - Use this wrapper in Christofides, not raw greedyMatch.
//
// AI-Hints:
//   - Do not call greedyMatch directly from Christofides.
func greedyMatchAtomic(odd []int, dist matrix.Matrix, adj [][]int) (err error) {
	lengths := snapshotAdjLengths(adj)
	defer func() {
		if err != nil {
			rollbackAdj(adj, lengths)
		}
	}()

	return greedyMatch(odd, dist, adj)
}
