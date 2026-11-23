// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package match exposes one complete perfect-matching facade for two kernels.
// The API never accepts TSP adjacency or publishes partial matching outcomes.
package match

import (
	"errors"
	"math"
	"sort"

	"github.com/lvlath/go/matrix"
)

// Perfect computes a perfect matching over ordered original matrix indices.
// Blossom solves minimum-weight perfect matching or returns an error; explicit
// Greedy returns a deterministic feasible heuristic without an optimum claim.
// Implementation:
//   - Stage 1: Apply and validate every option, then handle empty/odd subsets.
//   - Stage 2: Validate the whole finite complete matrix and prepare detached data.
//   - Stage 3: Execute only the requested kernel; verify and normalize output.
//
// Behavior highlights:
//   - Empty subset never reads dist; it does not mean all vertices.
//   - Local engine order follows vertices and is not sorted before solving.
//   - Pairs are normalized to U<V and sorted by ascending U after solving.
//   - Cost is checked raw directional selected-weight sum, without rounding.
//
// Inputs:
//   - dist: caller-owned matrix stable for the whole call.
//   - vertices: even-size ordered original indices; not mutated or retained.
//   - options: ordered callbacks, all checked before matrix access.
//
// Returns:
//   - *Result: non-nil complete detached result, or nil on any error.
//
// Errors:
//   - ErrInvalidOptions, ErrOddVertices, matrix-domain and solver errors,
//     ErrInvalidMatching, ErrNumericOverflow, ErrSizeOverflow.
//
// Determinism:
//   - Input subset order for kernel ties, canonical strict integer output sort.
//
// Complexity:
//   - Whole-matrix admission O(n^2); Blossom preparation O(k^2) plus engine;
//     Greedy matching O(k^2), and result normalization O(k log k).
//
// Notes:
//   - Concurrent mutation of dist is unsupported; no general snapshot contract.
//   - The legacy engine includes a potentially exponential nested-lifting path.
//
// AI-Hints:
//   - Never substitute Greedy after a failed Blossom solve.
//   - Do not convert local mates to public endpoint positions.
func Perfect(dist matrix.Matrix, vertices []int, options ...Option) (*Result, error) {
	policy, err := applyOptions(options...)
	if err != nil {
		return nil, err
	}
	if len(vertices) == 0 {
		return &Result{Pairs: make([]Pair, 0), Cost: 0, Algorithm: policy.Algorithm, Optimal: policy.Algorithm == Blossom}, nil
	}
	if len(vertices)&1 != 0 {
		return nil, ErrOddVertices
	}

	var pairs []Pair
	var total float64

	switch policy.Algorithm {
	case Blossom:
		problem, err := buildMatchingProblem(vertices, dist)
		if err != nil {
			return nil, err
		}
		if err = checkBlossomSizes(problem.n); err != nil {
			return nil, err
		}
		mates, cost, _, err := solveMinimumWeightPerfectMatching(problem, blossomOptions{Eps: policy.Eps})
		if err != nil {
			if errors.Is(err, ErrIncompleteGraph) {
				return nil, errors.Join(ErrNoMatching, err)
			}
			return nil, err
		}
		if err = verifyPerfectMatching(mates); err != nil {
			return nil, err
		}
		pairs = make([]Pair, 0, problem.n/2)
		for local, partner := range mates {
			if local < partner {
				pairs = append(pairs, Pair{U: problem.odd[local], V: problem.odd[partner]})
			}
		}
		total = cost

	case Greedy:
		weights, err := copyCompleteWeights(dist, true)
		if err != nil {
			return nil, err
		}
		ordered, err := validateSubset(vertices, weights.n)
		if err != nil {
			return nil, err
		}
		pairs, total, err = greedySelect(ordered, weights)
		if err != nil {
			return nil, err
		}
	default:
		return nil, ErrInvalidOptions
	}

	if err := canonicalizeMatchingPairs(pairs, vertices); err != nil {
		return nil, err
	}
	if math.IsNaN(total) || math.IsInf(total, 0) || total < 0 {
		return nil, ErrNumericOverflow
	}
	return &Result{Pairs: pairs, Cost: total, Algorithm: policy.Algorithm, Optimal: policy.Algorithm == Blossom}, nil
}

// canonicalizeMatchingPairs verifies original-index coverage before output sort.
// It normalizes each endpoint pair but never changes algorithmic subset order.
// Implementation:
//   - Stage 1: Build a selected-index map and verify pair count.
//   - Stage 2: Reject foreign, repeated or self endpoints; orient U<V.
//   - Stage 3: Sort by ascending integer U (then V) for public publication.
//
// Behavior highlights:
//   - This is a post-solve presentation transform, not a kernel tie-break.
//
// Inputs:
//   - pairs: caller-owned mutable output; vertices: original chosen indices.
//
// Returns:
//   - error: nil only for exactly one coverage of every selected vertex.
//
// Errors:
//   - ErrInvalidMatching for malformed pairing or subset coverage.
//
// Determinism:
//   - Original validation scan followed by strict integer ascending sort.
//
// Complexity:
//   - Time O(k log k), Space O(k) for membership bookkeeping.
//
// Notes:
//   - The TSP adapter receives canonical pairs; every selected vertex acquires
//     exactly one new neighbor, so pair append order does not change adjacency rows.
//
// AI-Hints:
//   - Sorting vertices before solving would change ties; only sort final pairs.
func canonicalizeMatchingPairs(pairs []Pair, vertices []int) error {
	if len(vertices)&1 != 0 || len(pairs) != len(vertices)/2 {
		return ErrInvalidMatching
	}
	selected := make(map[int]bool, len(vertices))
	for _, v := range vertices {
		if selected[v] {
			return ErrInvalidMatching
		}
		selected[v] = true
	}
	seen := make(map[int]bool, len(vertices))
	for i := range pairs {
		u, v := pairs[i].U, pairs[i].V
		if u == v || !selected[u] || !selected[v] || seen[u] || seen[v] {
			return ErrInvalidMatching
		}
		seen[u], seen[v] = true, true
		if u > v {
			pairs[i].U, pairs[i].V = v, u
		}
	}
	if len(seen) != len(vertices) {
		return ErrInvalidMatching
	}
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i].U != pairs[j].U {
			return pairs[i].U < pairs[j].U
		}
		return pairs[i].V < pairs[j].V
	})
	return nil
}

// PerfectWithOptions solves a selected matching problem using a complete Options value.
// It is a value-based facade for callers that store or reuse an explicit policy.
// Implementation:
//   - Stage 1: Copy the caller's policy through one normal option callback.
//   - Stage 2: Delegate to Perfect, which validates the complete scalar policy.
//   - Stage 3: Return its detached pairing result unchanged.
//
// Behavior highlights:
//   - A zero Options value is not silently defaulted: Eps must be finite and positive.
//   - Passing Options does not select a second solver or alter numerical tie-breaking.
//
// Inputs:
//   - dist: stable, caller-owned complete matrix, never mutated.
//   - vertices: ordered original indices, neither sorted nor retained.
//   - policy: caller-owned value whose Algorithm and Eps are checked by Perfect.
//
// Returns:
//   - *Result: detached complete matching; nil on error.
//   - error: the exact Perfect validation/solver error with its underlying causes.
//
// Errors:
//   - ErrInvalidOptions for malformed policy; matching-domain errors from Perfect.
//
// Determinism:
//   - Same matrix, subset order, numeric policy and kernel order as Perfect.
//
// Complexity:
//   - O(1) delegation plus the preparation and selected-kernel complexity of Perfect.
//
// Notes:
//   - This wrapper exists for a real TSP-adapter consumer of a pre-built policy.
//
// AI-Hints:
//   - Do not treat this facade as a separate path with weaker validation.
func PerfectWithOptions(dist matrix.Matrix, vertices []int, policy Options) (*Result, error) {
	return Perfect(dist, vertices, func(dst *Options) error {
		*dst = policy
		return nil
	})
}

// GreedyPerfect finds a complete matching with the explicit legacy greedy strategy.
// It is a convenient selection facade, not a minimum-cost or exactness guarantee.
// Implementation:
//   - Stage 1: Request the Greedy algorithm through the canonical WithAlgorithm option.
//   - Stage 2: Delegate all input admission, pair selection and publication to Perfect.
//
// Behavior highlights:
//   - Keeps the last-remaining pivot, partner scan and swap-delete selection policy.
//   - Uses the default valid configuration; it does not bypass whole-matrix admission.
//
// Inputs:
//   - dist: complete finite symmetric caller-owned cost matrix.
//   - vertices: even ordered subset of original matrix indices.
//
// Returns:
//   - *Result: detached perfect pairing with Algorithm==Greedy and Optimal==false.
//   - error: inherited from Perfect; result is nil on error.
//
// Errors:
//   - All matching-domain errors from Perfect, including ErrOddVertices and ErrNumericOverflow.
//
// Determinism:
//   - Exactly the same selected pairs and raw cost as Perfect with WithAlgorithm(Greedy).
//
// Complexity:
//   - Time O(n^2+k^2), Space O(n^2+k) in the current admission/copy model.
//
// Notes:
//   - This is useful when a fast feasible matching is required and optimality is not.
//
// AI-Hints:
//   - Do not assume the minimum-cost pairing or Christofides 1.5 ratio for this result.
func GreedyPerfect(dist matrix.Matrix, vertices []int) (*Result, error) {
	return Perfect(dist, vertices, WithAlgorithm(Greedy))
}
