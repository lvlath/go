// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp exposes the exact MWPM facade selected by BlossomMatch.
// The facade builds a detached local matching problem, delegates to the dense
// Blossom engine, verifies the result, and appends matching edges atomically.
package tsp

import (
	"github.com/lvlath/go/matrix"
)

// blossomMatch computes exact minimum-weight perfect matching for Christofides.
// It is the only production path for MatchingAlgo==BlossomMatch.
//
// Implementation:
//   - Stage 1: Accept the empty odd set as a no-op.
//   - Stage 2: Snapshot adjacency lengths for atomic rollback.
//   - Stage 3: Build a detached local matching problem over odd vertices.
//   - Stage 4: Solve exact MWPM through solveMinimumWeightPerfectMatching.
//   - Stage 5: Verify and append matching pairs to the Christofides multigraph.
//
// Behavior highlights:
//   - Exact-or-error.
//   - Does not call greedy matching.
//   - Does not return size-based unavailability for large odd sets.
//   - Leaves adj unchanged on any error.
//
// Inputs:
//   - odd: odd-degree MST vertices in deterministic scan order.
//   - dist: final symmetric complete finite TSP matrix.
//   - adj: mutable Christofides multigraph adjacency.
//   - opts: solver policy; opts.Eps controls Blossom float tolerance.
//
// Returns:
//   - error: nil when matching edges were appended successfully.
//
// Errors:
//   - ErrInvalidOptions for invalid tolerance.
//   - ErrInvalidMatching for malformed odd set, corrupt match, or adjacency shape.
//   - ErrIncompleteGraph when no perfect matching exists.
//   - ErrNaNInf, ErrNegativeWeight, ErrAsymmetry propagated through the matrix firewall.
//
// Determinism:
//   - Odd-order, edge construction, matching verification, and append order are deterministic.
//
// Complexity:
//   - Dense Blossom target: polynomial in k=len(odd), with O(k^2) dense edge storage.
//   - Append stage: O(k).
//
// Notes:
//   - GreedyMatch is a separate explicit weaker mode.
//
// AI-Hints:
//   - Do not call exactSmallPerfectMatching as the general production solver here.
//   - Do not convert Blossom errors into GreedyMatch output.
func blossomMatch(odd []int, dist matrix.Matrix, adj [][]int, opts Options) (err error) {
	if len(odd) == 0 {
		return nil
	}
	if (len(odd) & 1) == 1 {
		return ErrInvalidMatching
	}

	// Capture a transactional checkpoint of slice lengths before attempting graph mutation.
	// The deferred closure guarantees a clean rollback of all appended edges if subsequent phases fail.
	lengths := snapshotAdjLengths(adj)
	defer func() {
		if err != nil {
			// Trigger state recovery: discard intermediate mutations and restore original graph layout.
			rollbackAdj(adj, lengths)
		}
	}()

	// Construct the independent, localized matching problem sub-graph induced by the odd vertices.
	problem, err := buildMatchingProblem(odd, dist)
	if err != nil {
		return err
	}

	// Invoke the dense exact MWPM engine and keep all solver state detached from adjacency mutation.
	match, _, _, err := solveMinimumWeightPerfectMatching(problem, blossomOptions{Eps: opts.Eps})
	if err != nil {
		return err
	}

	return appendPerfectMatching(problem, match, adj)
}
