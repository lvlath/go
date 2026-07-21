// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp implements Minimum Spanning Tree helpers for the Christofides pipeline.
//
// MinimumSpanningTree builds an MST over a non-negative symmetric distance matrix.
// This is the first constructive stage of Christofides. The current implementation
// uses Prim in O(n²), which is appropriate for dense matrix-backed TSP instances and
// avoids heap allocations in the common dense regime.
//
// Contracts:
//   - dist is a square n×n matrix.
//   - diagonal is approximately zero under symTol.
//   - NaN, -Inf, and negative weights are rejected.
//   - +Inf off-diagonal values make vertices unreachable and lead to ErrIncompleteGraph.
//   - symmetric validation is required because Christofides consumes undirected metric input.
//
// Behavior:
//   - n==1 returns an empty tree with weight 0.
//   - The returned adjacency is an undirected simple MST adjacency list.
//   - Ties are resolved by vertex-index scan order.
//   - Dense and generic paths must preserve the same sentinel classification.
//
// Return values:
//   - totalW: total MST weight, stabilized with round1e9.
//   - adj: undirected adjacency lists of the MST.
//
// Complexity:
//   - Time  : O(n²).
//   - Memory: O(n) state plus O(n) adjacency entries for the tree.
//
// AI-Hints:
//   - Do not use map iteration as a traversal source.
//   - Do not let dense and generic paths drift in sentinel behavior.
//   - Do not replace the dense O(n²) scan with a heap unless sparse matrix semantics are introduced.
package tsp

import (
	"errors"
	"math"

	"github.com/lvlath/go/matrix"
)

// MinimumSpanningTree computes an undirected minimum spanning tree with Prim's algorithm.
// It is the MST stage used by Christofides and a public helper for symmetric,
// non-negative matrix-backed complete graphs.
//
// Implementation:
//   - Stage 1: Validate and handle the one-vertex tree edge case.
//   - Stage 2: Copy the matrix through the shared TSP weight firewall.
//   - Stage 3: Run the single Prim kernel over weightBuffer.
//   - Stage 4: Return rounded total weight and deterministic undirected adjacency.
//
// Behavior highlights:
//   - Uses one semantic path for dense and generic matrices.
//   - Preserves weightBuffer sentinel classification.
//   - Does not use map iteration.
//   - Resolves ties by stable vertex-index scan order.
//
// Inputs:
//   - dist: symmetric matrix distance model.
//
// Returns:
//   - float64: total MST weight rounded with round1e9.
//   - [][]int: undirected MST adjacency lists.
//   - error: nil on success or sentinel-classified failure.
//
// Errors:
//   - ErrNilDistanceMatrix, ErrNonSquare, ErrDimensionMismatch.
//   - ErrNaNInf, ErrNonZeroDiagonal, ErrNegativeWeight, ErrIncompleteGraph, ErrAsymmetry.
//
// Determinism:
//   - Prim starts from vertex 0.
//   - Vertex selection scans from 0 to n-1.
//   - Parent relaxation uses strict `<`, preserving earlier ties.
//
// Complexity:
//   - Time O(n^2), Space O(n^2) for weightBuffer plus O(n) MST state.
//
// Notes:
//   - The O(n^2) weightBuffer copy intentionally removes dense/generic sentinel drift.
//
// AI-Hints:
//   - Do not restore separate Dense and generic Prim paths unless tests prove identical sentinel behavior.
//   - Do not use heap-based Prim until sparse matrix semantics are explicit.
func MinimumSpanningTree(dist matrix.Matrix) (float64, [][]int, error) {
	if err := matrix.ValidateSquare(dist); err != nil {
		if errors.Is(err, matrix.ErrNilMatrix) {
			return 0, nil, errors.Join(ErrNilDistanceMatrix, err)
		}
		if errors.Is(err, matrix.ErrNonSquare) {
			return 0, nil, errors.Join(ErrNonSquare, err)
		}

		return 0, nil, errors.Join(ErrDimensionMismatch, err)
	}

	if dist.Rows() == 1 {
		if err := validateSingletonMSTMatrix(dist); err != nil {
			return 0, nil, err
		}

		return 0, make([][]int, 1), nil
	}

	weights, err := copyCompleteWeights(dist, true)
	if err != nil {
		return 0, nil, err
	}

	return primMST(weights)
}

// validateSingletonMSTMatrix validates the only cell of a one-vertex MST matrix.
// The one-vertex MST has no edges, but its diagonal must still satisfy the same
// structural zero-distance law as every TSP matrix.
//
// Implementation:
//   - Stage 1: Read dist[0][0].
//   - Stage 2: Reject NaN and -Inf as numeric invalidity.
//   - Stage 3: Reject +Inf or non-zero diagonal as ErrNonZeroDiagonal.
//
// Behavior highlights:
//   - Does not allocate.
//   - Preserves matrix read errors through ErrDimensionMismatch.
//
// Inputs:
//   - dist: already square one-by-one matrix.
//
// Returns:
//   - error: nil when the singleton diagonal is valid.
//
// Errors:
//   - ErrDimensionMismatch joined with matrix read errors.
//   - ErrNaNInf joined with matrix.ErrNaNInf.
//   - ErrNonZeroDiagonal.
//
// Determinism:
//   - Single fixed cell read.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This helper exists only to keep public MST singleton behavior well-defined.
//
// AI-Hints:
//   - Do not bypass diagonal validation for singleton matrices.
func validateSingletonMSTMatrix(dist matrix.Matrix) error {
	value, err := dist.At(0, 0)
	if err != nil {
		return errors.Join(ErrDimensionMismatch, err)
	}
	if math.IsNaN(value) || math.IsInf(value, -1) {
		return errors.Join(ErrNaNInf, matrix.ErrNaNInf)
	}
	if math.IsInf(value, 1) || math.Abs(value) > symTol {
		return ErrNonZeroDiagonal
	}

	return nil
}

// primMST runs deterministic O(n^2) Prim over a validated weightBuffer.
// It is the only MST construction kernel, preventing matrix implementation
// differences from changing sentinel behavior or tie-breaking.
//
// Implementation:
//   - Stage 1: Initialize best incoming edge cost for every vertex.
//   - Stage 2: Repeatedly select the non-tree vertex with minimal connection cost.
//   - Stage 3: Append the selected parent edge to undirected adjacency.
//   - Stage 4: Relax all remaining vertices through the selected vertex.
//
// Behavior highlights:
//   - Starts from vertex 0.
//   - Uses strict `<` relaxation for stable tie behavior.
//   - Consumes only complete symmetric weights produced by copyCompleteWeights.
//
// Inputs:
//   - weights: validated complete symmetric row-major distance buffer.
//
// Returns:
//   - float64: rounded total MST weight.
//   - [][]int: undirected tree adjacency.
//
// Errors:
//   - ErrDimensionMismatch for malformed internal buffers.
//   - ErrIncompleteGraph if no reachable non-tree vertex can be selected.
//
// Determinism:
//   - Fixed vertex scan order for selection and relaxation.
//
// Complexity:
//   - Time O(n^2), Space O(n).
//
// Notes:
//   - Missing edges should not exist after copyCompleteWeights, but +Inf is still checked defensively.
//
// AI-Hints:
//   - Do not read matrix.Matrix here; all matrix semantics belong to weights.go.
func primMST(weights weightBuffer) (float64, [][]int, error) {
	if weights.n <= 0 || len(weights.w) != weights.n*weights.n {
		return 0, nil, ErrDimensionMismatch
	}
	if weights.n == 1 {
		return 0, make([][]int, 1), nil
	}

	n := weights.n
	inTree := make([]bool, n)
	bestCost := make([]float64, n)
	parent := make([]int, n)
	adjacency := make([][]int, n)

	for vertex := 0; vertex < n; vertex++ {
		bestCost[vertex] = math.Inf(1)
		parent[vertex] = -1
	}
	bestCost[0] = 0

	total := 0.0

	for iteration := 0; iteration < n; iteration++ {
		selected := -1
		selectedCost := math.Inf(1)

		for vertex := 0; vertex < n; vertex++ {
			if !inTree[vertex] && bestCost[vertex] < selectedCost {
				selected = vertex
				selectedCost = bestCost[vertex]
			}
		}

		if selected == -1 || math.IsInf(selectedCost, 1) {
			return 0, nil, ErrIncompleteGraph
		}

		inTree[selected] = true

		if parent[selected] != -1 {
			parentVertex := parent[selected]
			adjacency[selected] = append(adjacency[selected], parentVertex)
			adjacency[parentVertex] = append(adjacency[parentVertex], selected)
			total += bestCost[selected]
		}

		for vertex := 0; vertex < n; vertex++ {
			if inTree[vertex] || vertex == selected {
				continue
			}

			candidateCost := weights.at(selected, vertex)
			if math.IsInf(candidateCost, 1) {
				continue
			}

			if candidateCost < bestCost[vertex] {
				bestCost[vertex] = candidateCost
				parent[vertex] = selected
			}
		}
	}

	return round1e9(total), adjacency, nil
}
