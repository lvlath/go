// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp implements the Christofides approximation pipeline.
//
// ChristofidesSolve computes a Hamiltonian cycle for symmetric metric TSP using
// the Christofides pipeline:
//
//  1. Minimum Spanning Tree (MST) on the complete metric graph.
//  2. Minimum-weight perfect matching on odd-degree vertices of the MST.
//  3. Eulerian circuit on the resulting multigraph.
//  4. Shortcutting the Eulerian walk to a Hamiltonian cycle (skip revisits).
//
// Mathematical guarantee:
//   - The returned tour has the Christofides 1.5 bound only when the matching
//     stage computes exact minimum-weight perfect matching.
//
// Contracts:
//   - Public facades and christofides validate matrix shape, symmetry, completeness,
//     start vertex, and matching policy before mutating Christofides multigraph state.
//   - dist is square n×n, n ≥ 2,
//   - diagonal ≈ 0, no negative weights, no NaN,
//   - symmetric (opts.Symmetric==true / mustEnforceSymmetry(opts) == true),
//   - if opts.RunMetricClosure==false: no +Inf edges allowed.
//
// Options notes:
//   - opts.StartVertex fixes the start/closure of the cycle.
//   - opts.MatchingAlgo selects exact matching or explicit greedy matching.
//   - No RNG is used here; determinism is intrinsic.
//   - Local-search post-passes are orchestrated by the matrix facade:
//   - if EnableLocalSearch && !BestImprovement → fast 2-opt
//   - if EnableLocalSearch &&  BestImprovement → hybrid [2-opt → 3-opt(best) → 2-opt polish]
//     This keeps Christofides pure and predictable; tuning lives in the dispatcher.
//
// Complexity (dense representation):
//   - MST (Prim O(n^2)) + odd collection O(n) +
//     matching (implementation-dependent; greedy O(k^2), blossom polytime) +
//     Eulerian (O(E)), shortcut O(n)  ⇒ typically O(n^2) for metric instances.
//
// Returned value:
//   - Result with a detached closed tour and proof metadata.
//   - Tour invariants: len==n+1, Tour[0]==Tour[n]==opts.StartVertex, each vertex appears once.
//
// Errors:
//   - Strict package sentinels such as ErrStartOutOfRange and ErrIncompleteGraph.
//
// Guarantee note:
//   - Exact MWPM is required before the formal 1.5 metadata can be published.
//   - Explicit greedy matching produces a valid heuristic tour with no formal ratio.
package tsp

import (
	"github.com/lvlath/go/matrix"
)

// christofides runs the Christofides pipeline and publishes a canonical Result.
//
// Implementation:
//   - Stage 1: Validate options, symmetric complete matrix shape, and start vertex.
//   - Stage 2: Build MST and collect odd-degree vertices.
//   - Stage 3: Apply selected odd-vertex matching and record proof metadata.
//   - Stage 4: Build an Eulerian circuit and shortcut it to a Hamiltonian cycle.
//   - Stage 5: Canonicalize orientation, compute cost, validate the final tour, and publish Result.
//
// Behavior highlights:
//   - Requires symmetric complete final matrix input.
//   - GreedyMatch is explicit weaker behavior with no formal ratio.
//   - BlossomMatch publishes the formal 1.5 ratio only after exact matching success.
//   - Local-search post-passes are owned by solver.go, not by this kernel.
//
// Inputs:
//   - dist: symmetric complete distance matrix.
//   - opts: Christofides policy.
//
// Returns:
//   - *Result: detached Christofides result.
//   - error: nil on success or sentinel-classified failure.
//
// Errors:
//   - ErrInvalidOptions.
//   - ErrATSPNotSupportedByAlgo.
//   - ErrStartOutOfRange.
//   - ErrIncompleteGraph from MST, matching, Eulerian shortcut, or cost checks.
//   - ErrNonEulerian.
//   - ErrInvalidTour.
//   - Matrix numeric sentinels from validation and cost helpers.
//
// Determinism:
//   - MST, odd collection, matching, Eulerian traversal, shortcutting, and orientation are deterministic.
//
// Complexity:
//   - Time O(n^2) plus selected matching complexity.
//   - Space O(n+E) plus matching-local buffers.
//
// Notes:
//   - This function does not attach IDs or metric-closure metadata.
//
// AI-Hints:
//   - Do not apply 2-opt or 3-opt here; solver.go owns post-pass policy.
//   - Do not publish a 1.5 ratio from GreedyMatch.
func christofides(dist matrix.Matrix, opts Options) (*Result, error) {
	if err := validateOptionsStandalone(opts); err != nil {
		return nil, err
	}

	n, err := validateSolverDistanceMatrix(dist, true, true, symTol)
	if err != nil {
		return nil, err
	}
	if err = validateStartVertex(n, opts.StartVertex); err != nil {
		return nil, err
	}

	_, mstAdjacency, err := MinimumSpanningTree(dist)
	if err != nil {
		return nil, err
	}

	odd := collectOddDegreeVertices(mstAdjacency)

	approximationRatio, err := applyChristofidesMatching(odd, dist, mstAdjacency, opts)
	if err != nil {
		return nil, err
	}

	eulerianAdjacency, err := canonicalizeUndirectedMultigraph(mstAdjacency)
	if err != nil {
		return nil, err
	}

	eulerianWalk, err := EulerianCircuit(eulerianAdjacency, opts.StartVertex)
	if err != nil {
		return nil, err
	}

	tour, err := ShortcutEulerianToHamiltonian(eulerianWalk, n, opts.StartVertex)
	if err != nil {
		return nil, err
	}

	if err = CanonicalizeOrientationInPlace(tour); err != nil {
		return nil, err
	}

	cost, err := TourCost(dist, tour)
	if err != nil {
		return nil, err
	}

	if err = ValidateTour(tour, n, opts.StartVertex); err != nil {
		return nil, err
	}

	return &Result{
		Tour:               append([]int(nil), tour...),
		Cost:               cost,
		Algorithm:          Christofides,
		Exact:              false,
		Optimal:            false,
		TimedOut:           false,
		Symmetric:          true,
		ApproximationRatio: approximationRatio,
	}, nil
}

// applyChristofidesMatching appends the selected odd-vertex matching to the multigraph.
// It returns the formal approximation ratio proven by the selected matching mode.
//
// Implementation:
//   - Stage 1: Dispatch by explicit MatchingAlgo.
//   - Stage 2: Run exact MWPM or explicit greedy matching.
//   - Stage 3: Return ChristofidesApproximationRatio only after exact MWPM succeeds.
//
// Behavior highlights:
//   - BlossomMatch is exact-or-error.
//   - GreedyMatch is explicit weaker behavior and returns NoApproximationRatio.
//   - No hidden matching substitution is performed.
//
// Inputs:
//   - odd: odd-degree MST vertices.
//   - dist: final symmetric complete distance matrix.
//   - adj: mutable Christofides multigraph adjacency.
//   - opts: Christofides policy.
//
// Returns:
//   - float64: proven approximation ratio or NoApproximationRatio.
//   - error: nil on successful matching.
//
// Errors:
//   - ErrInvalidOptions for unknown MatchingAlgo.
//   - ErrIncompleteGraph when exact matching proves that no perfect matching exists.
//   - ErrInvalidMatching, ErrIncompleteGraph, and matrix sentinels from matching engines.
//
// Determinism:
//   - Matching engines preserve deterministic odd-set order.
//
// Complexity:
//   - BlossomMatch: dense exact MWPM engine with O(k^2) dense edge storage.
//   - GreedyMatch: O(k^2) in the repeated-scan implementation.
//
// Notes:
//   - The returned ratio is proof metadata, not a performance estimate.
//
// AI-Hints:
//   - Do not claim ChristofidesApproximationRatio after GreedyMatch.
//   - Do not convert exact matching errors into a successful greedy result.
func applyChristofidesMatching(odd []int, dist matrix.Matrix, adj [][]int, opts Options) (float64, error) {
	switch opts.MatchingAlgo {
	case BlossomMatch:
		if err := blossomMatch(odd, dist, adj, opts); err != nil {
			return NoApproximationRatio, err
		}

		return ChristofidesApproximationRatio, nil

	case GreedyMatch:
		if err := greedyMatchAtomic(odd, dist, adj); err != nil {
			return NoApproximationRatio, err
		}

		return NoApproximationRatio, nil

	default:
		return NoApproximationRatio, ErrInvalidOptions
	}
}

// collectOddDegreeVertices returns vertices with odd degree in deterministic vertex order.
//
// Implementation:
//   - Stage 1: Allocate a compact result slice.
//   - Stage 2: Scan adjacency from vertex 0 to vertex n-1.
//   - Stage 3: Append every vertex whose degree has odd parity.
//
// Behavior highlights:
//   - Does not mutate adjacency.
//   - Does not sort because scan order is already canonical.
//   - Does not validate multigraph symmetry.
//
// Inputs:
//   - adj: undirected multigraph adjacency.
//
// Returns:
//   - []int: odd-degree vertices in ascending index order.
//
// Errors:
//   - None.
//
// Determinism:
//   - Fixed increasing vertex-index scan.
//
// Complexity:
//   - Time O(n), Space O(k), where k is the number of odd-degree vertices.
//
// Notes:
//   - Perfect matching validation checks that k is even.
//
// AI-Hints:
//   - Do not collect through map iteration.
//   - Do not reorder this slice without updating matching tie-break tests.
func collectOddDegreeVertices(adj [][]int) []int {
	// Collect odd-degree vertices of the MST.
	// V has odd degree iff degree(v) mod 2 == 1. Fast parity check via bit-test.
	// len(adj[v])&1 == 1  ⇔ degree(v) is odd (LSB set).
	odd := make([]int, 0, len(adj)/2+1) // conservative capacity avoids reslices
	var vertex int
	for vertex = 0; vertex < len(adj); vertex++ {
		if len(adj[vertex])&1 == 1 {
			odd = append(odd, vertex)
		}
	}

	return odd
}
