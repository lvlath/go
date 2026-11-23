// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp defines solver policy options for Traveling Salesman Problem
// algorithms. Options are explicit contract switches: they select algorithms,
// matching policy, local-search limits, metric-closure behavior, deterministic
// randomness, and exact-solver guards.
package tsp

import "time"

const (
	// DefaultEps is the minimal strictly-better improvement accepted by local search.
	// A candidate move is accepted only when delta < -DefaultEps under default options.
	DefaultEps = 1e-12

	// DefaultTwoOptMaxIters caps accepted 2-opt moves across one local-search run.
	// Zero in Options means unlimited; DefaultOptions uses this finite production guard.
	DefaultTwoOptMaxIters = 10_000

	// DefaultThreeOptMaxMoves caps accepted 3-opt moves across one local-search run.
	// ThreeOpt has a larger neighborhood than 2-opt, so it receives a distinct policy knob.
	DefaultThreeOptMaxMoves = 10_000
)

const (
	// DefaultMaxExactN is the opt-in Auto-policy cap for exact Held-Karp selection.
	DefaultMaxExactN = MaxExactN

	// NoApproximationRatio marks solvers or weaker explicit modes with no proven ratio.
	NoApproximationRatio = 0.0

	// ChristofidesApproximationRatio is the formal ratio only when true MWPM is used.
	ChristofidesApproximationRatio = 1.5
)

// Options defines explicit solver policy for TSP and ATSP algorithms.
//
// Implementation:
//   - Stage 1: DefaultOptions builds a complete deterministic policy.
//   - Stage 2: validateOptionsStandalone rejects invalid numeric knobs and enum values.
//   - Stage 3: finalizeSolverOptions resolves Auto after matrix order is known.
//
// Behavior highlights:
//   - No hidden exact-to-heuristic downgrade exists.
//   - Algo==Auto is an explicit dispatcher policy, not an implicit algorithm substitution.
//   - BlossomMatch and GreedyMatch are distinct caller-selected matching policies.
//   - MatchingAlgo==GreedyMatch is an explicit weaker Christofides mode.
//   - TimeLimit==0 means unlimited.
//   - Seed==0 is interpreted by randomized kernels as a fixed deterministic stream.
//
// Inputs:
//   - Passed to SolveMatrix, SolveGraph, and direct solver wrappers.
//
// Returns:
//   - Options is consumed by solvers and does not own external memory.
//
// Errors:
//   - ErrInvalidOptions for invalid enum values, negative caps, negative TimeLimit, or invalid Eps.
//   - ErrATSPNotSupportedByAlgo for symmetric-only algorithms used with asymmetric policy.
//
// Determinism:
//   - Same options and same matrix order produce the same selected algorithm and route policy.
//   - ShuffleNeighborhood is deterministic under Seed.
//
// Complexity:
//   - Validation and finalization are O(1), except Auto selection depends on matrix order only.
//
// Notes:
//   - BlossomMatch is the default exact MWPM policy.
//   - GreedyMatch remains available only as an explicit weaker heuristic mode.
//
// AI-Hints:
//   - Do not silently clamp invalid options.
//   - Do not add a hidden matching-substitution policy.
//   - Do not claim ChristofidesApproximationRatio for GreedyMatch.
type Options struct {
	// StartVertex selects the start/end vertex index [0..n-1].
	StartVertex int

	// Algo selects the top-level solver path.
	Algo Algorithm

	// Symmetric controls TSP/ATSP matrix validation.
	Symmetric bool

	// MatchingAlgo chooses the odd-vertex matching engine for Christofides.
	MatchingAlgo MatchingAlgo

	// BoundAlgo controls Branch-and-Bound lower-bound strategy.
	BoundAlgo BoundAlgo

	// RunMetricClosure runs Floyd-Warshall before solving direct matrix input.
	RunMetricClosure bool

	// EnableLocalSearch enables local-search post-passes where the dispatcher supports them.
	EnableLocalSearch bool

	// TwoOptMaxIters bounds accepted 2-opt moves. Zero means unlimited.
	TwoOptMaxIters int

	// ThreeOptMaxMoves bounds accepted 3-opt moves. Zero means unlimited.
	ThreeOptMaxMoves int

	// BestImprovement selects best-improvement policy where a local-search kernel supports it.
	BestImprovement bool

	// ShuffleNeighborhood enables deterministic seed-controlled neighborhood shuffling.
	ShuffleNeighborhood bool

	// Eps is the minimal accepted improvement threshold.
	Eps float64

	// TimeLimit optionally bounds wall-clock time. Zero means unlimited.
	TimeLimit time.Duration

	// Seed controls deterministic randomized components.
	Seed int64

	// MaxExactN bounds exact Held-Karp selection when Algo==Auto.
	MaxExactN int
}

// DefaultOptions returns a complete deterministic production policy.
//
// Implementation:
//   - Stage 1: Select the symmetric Christofides pipeline.
//   - Stage 2: Select BlossomMatch as the exact default matching policy.
//   - Stage 3: Enable local search with bounded move counts and deterministic RNG policy.
//
// Behavior highlights:
//   - Returns a fully populated Options value.
//   - Does not enable metric closure by default.
//   - Does not enable Auto by default.
//   - Does not contain hidden Blossom-to-greedy substitution policy.
//
// Inputs:
//   - None.
//
// Returns:
//   - Options: detached value safe for caller mutation.
//
// Errors:
//   - None.
//
// Determinism:
//   - Seed==0 is interpreted by RNG consumers as a fixed deterministic stream.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - GreedyMatch is a selected algorithmic mode, not a side-channel downgrade.
//   - The default already selects BlossomMatch; explicitly selecting it is useful
//     when callers make the approximation contract visible at the call site.
//
// AI-Hints:
//   - Do not add hidden heuristic substitution here.
//   - Do not set Algo=Auto by default in a patch release.
//   - A 1.5 ratio also requires metric input; this constructor does not check
//     the triangle inequality, so the caller must establish that hypothesis.
func DefaultOptions() Options {
	return Options{
		StartVertex:         0,
		Algo:                Christofides,
		Symmetric:           true,
		MatchingAlgo:        BlossomMatch,
		BoundAlgo:           NoBound,
		RunMetricClosure:    false,
		EnableLocalSearch:   true,
		TwoOptMaxIters:      DefaultTwoOptMaxIters,
		ThreeOptMaxMoves:    DefaultThreeOptMaxMoves,
		BestImprovement:     false,
		ShuffleNeighborhood: false,
		Eps:                 DefaultEps,
		TimeLimit:           0,
		Seed:                0,
		MaxExactN:           DefaultMaxExactN,
	}
}
