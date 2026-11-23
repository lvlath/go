// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp adapts independent perfect matching to Christofides topology.
// It preserves legacy error classes and exact per-row multigraph topology.
package tsp

import (
	"errors"
	"math"

	"github.com/lvlath/go/match"
	"github.com/lvlath/go/matrix"
)

// blossomMatch runs only the exact matching kernel and publishes complete topology.
// It retains the legacy empty/odd/Eps error precedence for TSP private callers.
// Implementation:
//   - Stage 1: Apply the historic empty and odd subset fast paths.
//   - Stage 2: Translate TSP Eps and call the match facade without fallback.
//   - Stage 3: Validate all returned endpoints, then append reciprocal half-edges.
//
// Behavior highlights:
//   - Does not compute any alternative matching itself.
//
// Inputs:
//   - odd: ordered odd-degree vertices; dist: stable distance matrix.
//   - adj: mutable TSP multigraph; opts: historic TSP solver options.
//
// Returns:
//   - error: nil only after the complete matching has been published.
//
// Errors:
//   - Existing TSP sentinel classes and original match/matrix causes via errors.Is.
//
// Determinism:
//   - Original subset order affects the solver, not the one-edge-per-row publication.
//
// Complexity:
//   - Matching cost dominates; adapter O(k+n) preflight/commit.
//
// Notes:
//   - Visible adj remains unchanged on returned errors; backing capacities are not transactional.
//
// AI-Hints:
//   - Never call Greedy when Blossom returns an error.
func blossomMatch(odd []int, dist matrix.Matrix, adj [][]int, opts Options) error {
	if len(odd) == 0 {
		return nil
	}
	if len(odd)&1 != 0 {
		return ErrInvalidMatching
	}
	if opts.Eps <= 0 || math.IsNaN(opts.Eps) || math.IsInf(opts.Eps, 0) {
		// Historically the matrix and subset were validated before engine Eps.
		weights, err := copyCompleteWeights(dist, true)
		if err != nil {
			return err
		}
		seen := make([]bool, weights.n)
		for _, vertex := range odd {
			if vertex < 0 || vertex >= weights.n || seen[vertex] {
				return ErrInvalidMatching
			}
			seen[vertex] = true
		}
		return ErrInvalidOptions
	}
	result, err := match.PerfectWithOptions(dist, odd, match.Options{Algorithm: match.Blossom, Eps: opts.Eps})
	if err != nil {
		return translateMatchingError(err)
	}
	return appendMatchingResult(odd, result, adj, match.Blossom)
}

// greedyMatch runs the existing greedy selection policy without partial topology.
// It delegates to the independent match package and then publishes complete pairs.
// Implementation:
//   - Stage 1: Preserve the historic empty and odd subset fast paths.
//   - Stage 2: Run explicit Greedy, never invoking Blossom.
//   - Stage 3: Append only after full validation; disjoint pair order commutes.
//
// Behavior highlights:
//   - Selected partners follow the original last-pivot/swap-delete kernel.
//
// Inputs:
//   - odd: ordered original indices; dist: source matrix; adj: TSP adjacency.
//
// Returns:
//   - error: nil only for a complete published matching.
//
// Errors:
//   - Translated matching errors retaining the original cause.
//
// Determinism:
//   - Exact historical partner relation, with one new edge per affected row.
//
// Complexity:
//   - Source admission O(n^2); greedy O(k^2); publication O(k+n).
//
// Notes:
//   - This private name remains for legacy internal callers.
//
// AI-Hints:
//   - Do not select another partner in the adapter; consume the existing result.
func greedyMatch(odd []int, dist matrix.Matrix, adj [][]int) error {
	if len(odd) == 0 {
		return nil
	}
	if len(odd)&1 != 0 {
		return ErrInvalidMatching
	}
	result, err := match.GreedyPerfect(dist, odd)
	if err != nil {
		return translateMatchingError(err)
	}
	return appendMatchingResult(odd, result, adj, match.Greedy)
}

// greedyMatchAtomic is the Christofides-safe alias of completed greedy selection.
// The delegated implementation validates all pairs before mutating adjacency.
// Implementation:
//   - Stage 1: Invoke the same private greedy adapter.
//
// Behavior highlights:
//   - No distinct selection kernel or hidden fallback exists here.
//
// Inputs:
//   - odd, dist, adj: same validated domains as greedyMatch.
//
// Returns:
//   - error: result of greedyMatch.
//
// Errors:
//   - Same TSP and original lower-level classes as greedyMatch.
//
// Determinism:
//   - No additional ordering decisions.
//
// Complexity:
//   - Same as greedyMatch.
//
// Notes:
//   - Old callers need not know about result ownership changes.
//
// AI-Hints:
//   - Do not restore the old append-during-selection behavior.
func greedyMatchAtomic(odd []int, dist matrix.Matrix, adj [][]int) error {
	return greedyMatch(odd, dist, adj)
}

// translateMatchingError retains the original cause and one applicable TSP class.
// Unknown errors retain their original identity without invented sentinel unions.
// Implementation:
//   - Stage 1: Classify the matching failure with errors.Is.
//   - Stage 2: Join only its semantically corresponding public TSP sentinel.
//
// Behavior highlights:
//   - Never converts matching failures to a different algorithm's success.
//
// Inputs:
//   - cause: matching or underlying matrix error.
//
// Returns:
//   - error: classified TSP error with original cause preserved.
//
// Errors:
//   - Existing TSP sentinels plus match/matrix sentinels.
//
// Determinism:
//   - Fixed precedence switch.
//
// Complexity:
//   - Time O(1) for finite-depth error chains, Space O(1).
//
// Notes:
//   - A cancellation/unknown cause is not flattened to a generic matching error.
//
// AI-Hints:
//   - Do not join unrelated sentinels simply for compatibility.
func translateMatchingError(cause error) error {
	if cause == nil {
		return nil
	}
	var category error
	switch {
	case errors.Is(cause, match.ErrInvalidOptions):
		category = ErrInvalidOptions
	case errors.Is(cause, match.ErrOddVertices), errors.Is(cause, match.ErrInvalidVertices), errors.Is(cause, match.ErrInvalidMatching):
		category = ErrInvalidMatching
	case errors.Is(cause, match.ErrNilMatrix):
		category = ErrNilDistanceMatrix
	case errors.Is(cause, match.ErrNonSquare):
		category = ErrNonSquare
	case errors.Is(cause, match.ErrInvalidDimensions), errors.Is(cause, match.ErrDimensionMismatch):
		category = ErrDimensionMismatch
	case errors.Is(cause, match.ErrNonZeroDiagonal):
		category = ErrNonZeroDiagonal
	case errors.Is(cause, match.ErrNaNInf):
		category = ErrNaNInf
	case errors.Is(cause, match.ErrNegativeWeight):
		category = ErrNegativeWeight
	case errors.Is(cause, match.ErrAsymmetry):
		category = ErrAsymmetry
	case errors.Is(cause, match.ErrIncompleteGraph), errors.Is(cause, match.ErrNoMatching):
		category = ErrIncompleteGraph
	case errors.Is(cause, match.ErrNumericOverflow):
		category = ErrNumericOverflow
	case errors.Is(cause, match.ErrSizeOverflow):
		category = ErrSizeOverflow
	default:
		return cause
	}
	return errors.Join(category, cause)
}

// appendMatchingResult validates complete matching before any TSP topology change.
// Disjoint matching edge appends preserve every adjacency row under canonical pair order.
// Implementation:
//   - Stage 1: Check algorithm, vertex domain, pair structure and adjacency bounds.
//   - Stage 2: Prove each endpoint receives one new neighbor only.
//   - Stage 3: Append reciprocal edges only after preflight succeeds.
//
// Behavior highlights:
//   - Input result Pairs is neither sorted nor changed by this adapter.
//
// Inputs:
//   - odd: original ordered subset; result: detached matching; adj: mutable multigraph.
//
// Returns:
//   - error: nil after all reciprocal edges have been added.
//
// Errors:
//   - ErrInvalidMatching on structural or destination-shape failure.
//
// Determinism:
//   - Pair order is canonical; row-level topology equals legacy disjoint appends.
//
// Complexity:
//   - Time O(k+n), Space O(k+n) for preflight and membership tracking.
//
// Notes:
//   - Visible adjacency stays unchanged on returned errors; panic recovery is not promised.
//
// AI-Hints:
//   - Do not mutate adj inside pair validation or recompute partner choices.
func appendMatchingResult(odd []int, result *match.Result, adj [][]int, algorithm match.Algorithm) error {
	if (algorithm != match.Blossom && algorithm != match.Greedy) ||
		result == nil || result.Algorithm != algorithm || len(odd)&1 != 0 || len(result.Pairs) != len(odd)/2 {
		return ErrInvalidMatching
	}
	if math.IsNaN(result.Cost) || math.IsInf(result.Cost, 0) || result.Cost < 0 {
		return ErrInvalidMatching
	}
	selected := make(map[int]bool, len(odd))
	for _, v := range odd {
		if v < 0 || v >= len(adj) || selected[v] {
			return ErrInvalidMatching
		}
		selected[v] = true
	}
	partner := make(map[int]int, len(odd))
	previousU := -1
	for _, pair := range result.Pairs {
		if pair.U >= pair.V || pair.U <= previousU || !selected[pair.U] || !selected[pair.V] {
			return ErrInvalidMatching
		}
		if _, exists := partner[pair.U]; exists {
			return ErrInvalidMatching
		}
		if _, exists := partner[pair.V]; exists {
			return ErrInvalidMatching
		}
		partner[pair.U] = pair.V
		partner[pair.V] = pair.U
		previousU = pair.U
	}
	if len(partner) != len(odd) {
		return ErrInvalidMatching
	}

	// Each selected vertex occurs in exactly one matching edge. Consequently,
	// changing the order/orientation of disjoint pair appends cannot change any
	// adjacency row: each receives exactly one new neighbor.
	// Preflight all capacity arithmetic before the first append. The normal
	// append operation itself may still fail under real runtime OOM.
	counts := make([]int, len(adj))
	for _, pair := range result.Pairs {
		counts[pair.U]++
		counts[pair.V]++
	}
	maxInt := math.MaxInt
	for vertex := range counts {
		if counts[vertex] > maxInt-len(adj[vertex]) {
			return ErrInvalidMatching
		}
	}
	for _, pair := range result.Pairs {
		adj[pair.U] = append(adj[pair.U], pair.V)
		adj[pair.V] = append(adj[pair.V], pair.U)
	}
	return nil
}
