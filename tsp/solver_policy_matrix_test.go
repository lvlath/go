// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp_test checks real dispatcher policies and boundary failures.
// All tests use public APIs and independent tour validity checks.
package tsp_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/tsp"
)

// TestTSPDispatcher_ChristofidesLocalSearchAndExactBounds checks solver composition.
// Local search must retain a feasible route while keeping correct ratio metadata.
func TestTSPDispatcher_ChristofidesLocalSearchAndExactBounds(t *testing.T) {
	dist := coverageLineMatrix(t, 7)
	configs := []struct {
		name   string
		algo   tsp.Algorithm
		bound  tsp.BoundAlgo
		search bool
		best   bool
	}{
		{"christofides_local_2opt", tsp.Christofides, tsp.NoBound, true, false},
		{"christofides_local_3opt", tsp.Christofides, tsp.NoBound, true, true},
		{"threeopt_local_2opt", tsp.ThreeOptOnly, tsp.NoBound, true, false},
		{"threeopt_local_best", tsp.ThreeOptOnly, tsp.NoBound, true, true},
		{"branch_simple", tsp.BranchAndBound, tsp.SimpleBound, false, false},
		{"branch_onetree", tsp.BranchAndBound, tsp.OneTreeBound, false, false},
		{"branch_none", tsp.BranchAndBound, tsp.NoBound, false, false},
	}
	for _, tc := range configs {
		t.Run(tc.name, func(t *testing.T) {
			options := tsp.DefaultOptions()
			options.Algo = tc.algo
			options.BoundAlgo = tc.bound
			options.EnableLocalSearch = tc.search
			options.BestImprovement = tc.best
			result, err := tsp.SolveMatrix(dist, nil, options)
			if err != nil || result == nil {
				t.Fatalf("solver: %+v %v", result, err)
			}
			if err := tsp.ValidateTour(result.Tour, 7, 0); err != nil {
				t.Fatalf("invalid tour: %v", err)
			}
			if result.Cost != 12 || result.Algorithm != tc.algo || result.TimedOut {
				t.Fatalf("incorrect result: %+v", result)
			}
			// Only the Christofides exact-matching path publishes the metric ratio.
			if tc.algo == tsp.Christofides && result.ApproximationRatio != tsp.ChristofidesApproximationRatio {
				t.Fatalf("lost approximation ratio: %+v", result)
			}
			if tc.algo == tsp.ThreeOptOnly && (result.Exact || result.Optimal) {
				t.Fatalf("local search claimed an optimum: %+v", result)
			}
		})
	}
}

// TestTSPDispatcher_OptionErrorClasses checks rejected policies before kernel work.
// Changing unsupported algorithms, bounds or numerical knobs must not cause fallback.
func TestTSPDispatcher_OptionErrorClasses(t *testing.T) {
	distance := coverageLineMatrix(t, 4)
	base := tsp.DefaultOptions()
	for _, tc := range []struct {
		name   string
		mutate func(*tsp.Options)
		want   error
	}{
		{"unknown_algorithm", func(o *tsp.Options) { o.Algo = tsp.Algorithm(99) }, tsp.ErrUnsupportedAlgorithm},
		{"unknown_matching", func(o *tsp.Options) { o.MatchingAlgo = tsp.MatchingAlgo(99) }, tsp.ErrInvalidOptions},
		{"unknown_bound", func(o *tsp.Options) { o.BoundAlgo = tsp.BoundAlgo(99) }, tsp.ErrInvalidOptions},
		{"negative_twoopt_iters", func(o *tsp.Options) { o.TwoOptMaxIters = -1 }, tsp.ErrInvalidOptions},
		{"negative_threeopt_moves", func(o *tsp.Options) { o.ThreeOptMaxMoves = -1 }, tsp.ErrInvalidOptions},
		{"negative_max_exact", func(o *tsp.Options) { o.MaxExactN = -1 }, tsp.ErrInvalidOptions},
		{"nan_epsilon", func(o *tsp.Options) { o.Eps = math.NaN() }, tsp.ErrInvalidOptions},
		{"inf_epsilon", func(o *tsp.Options) { o.Eps = math.Inf(1) }, tsp.ErrInvalidOptions},
		{"negative_epsilon", func(o *tsp.Options) { o.Eps = -1 }, tsp.ErrInvalidOptions},
		{"asymmetric_christofides", func(o *tsp.Options) { o.Symmetric = false }, tsp.ErrATSPNotSupportedByAlgo},
		{"asymmetric_onetree", func(o *tsp.Options) { o.Symmetric = false; o.BoundAlgo = tsp.OneTreeBound; o.Algo = tsp.BranchAndBound }, tsp.ErrATSPNotSupportedByAlgo},
	} {
		t.Run(tc.name, func(t *testing.T) {
			opts := base
			tc.mutate(&opts)
			result, err := tsp.SolveMatrix(distance, nil, opts)
			if result != nil || !errors.Is(err, tc.want) {
				t.Fatalf("invalid policy result=%+v err=%v want=%v", result, err, tc.want)
			}
		})
	}
	// Existing TSP option admission allows Eps==0, but nonempty weighted
	// Blossom rejects it only after the historical matrix/subset preflight.
	zero := base
	zero.Eps = 0
	zero.EnableLocalSearch = false
	result, err := tsp.ChristofidesSolve(distance, zero)
	if result != nil || !errors.Is(err, tsp.ErrInvalidOptions) {
		t.Fatalf("historical Blossom epsilon policy: %+v %v", result, err)
	}
}
