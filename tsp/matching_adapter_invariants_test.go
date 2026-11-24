// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp_test verifies matching error identity and transactional topology publication.
// The tests exercise the real adapter through a test-build-only bridge.
package tsp_test

import (
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/lvlath/go/match"
	"github.com/lvlath/go/tsp"
)

// TestMatchingAdapterErrorTranslation checks the complete cross-package error contract.
// Every known cause retains its original identity and gains the corresponding TSP class.
// Unknown backend failures are not relabeled as matching errors.
func TestMatchingAdapterErrorTranslation(t *testing.T) {
	cases := []struct{ cause, category error }{
		{match.ErrInvalidOptions, tsp.ErrInvalidOptions},
		{match.ErrOddVertices, tsp.ErrInvalidMatching},
		{match.ErrInvalidVertices, tsp.ErrInvalidMatching},
		{match.ErrInvalidMatching, tsp.ErrInvalidMatching},
		{match.ErrNilMatrix, tsp.ErrNilDistanceMatrix},
		{match.ErrNonSquare, tsp.ErrNonSquare},
		{match.ErrInvalidDimensions, tsp.ErrDimensionMismatch},
		{match.ErrDimensionMismatch, tsp.ErrDimensionMismatch},
		{match.ErrNonZeroDiagonal, tsp.ErrNonZeroDiagonal},
		{match.ErrNaNInf, tsp.ErrNaNInf},
		{match.ErrNegativeWeight, tsp.ErrNegativeWeight},
		{match.ErrAsymmetry, tsp.ErrAsymmetry},
		{match.ErrIncompleteGraph, tsp.ErrIncompleteGraph},
		{match.ErrNoMatching, tsp.ErrIncompleteGraph},
		{match.ErrNumericOverflow, tsp.ErrNumericOverflow},
		{match.ErrSizeOverflow, tsp.ErrSizeOverflow},
	}
	backend := errors.New("test matrix backend failure")
	for _, tc := range cases {
		t.Run(tc.cause.Error(), func(t *testing.T) {
			// Preserve both identities, including a wrapped backend failure.
			err := tsp.TranslateMatchingErrorForTest(errors.Join(tc.cause, backend))
			mustErrIs(t, err, tc.category)
			mustErrIs(t, err, tc.cause)
			mustErrIs(t, err, backend)
		})
	}
	if got := tsp.TranslateMatchingErrorForTest(nil); got != nil {
		t.Fatalf("nil cause translated to %v", got)
	}
	if got := tsp.TranslateMatchingErrorForTest(backend); !errors.Is(got, backend) || errors.Is(got, tsp.ErrInvalidMatching) || errors.Is(got, tsp.ErrIncompleteGraph) {
		t.Fatalf("unknown cause replaced: %v", got)
	}
}

// TestMatchingAdapterRejectsBeforeCommit verifies all reachable structural guards.
// Each malformed result must leave a pre-existing multigraph and its inputs unchanged.
// The failure fixtures deliberately include valid prefixes followed by invalid pairs.
func TestMatchingAdapterRejectsBeforeCommit(t *testing.T) {
	odd := []int{3, 1, 0, 2}
	pairs := []match.Pair{{U: 0, V: 1}, {U: 2, V: 3}}
	valid := func() *match.Result {
		return &match.Result{Pairs: slices.Clone(pairs), Cost: 2, Algorithm: match.Blossom, Optimal: true}
	}
	cases := []struct {
		name      string
		algorithm match.Algorithm
		edit      func(*match.Result) *match.Result
		vertices  []int
	}{
		{"unknown-algorithm", match.Algorithm(255), nil, odd},
		{"nil-result", match.Blossom, func(*match.Result) *match.Result { return nil }, odd},
		{"wrong-algorithm", match.Greedy, nil, odd},
		{"odd-cardinality", match.Blossom, nil, odd[:3]},
		{"wrong-pair-count", match.Blossom, func(r *match.Result) *match.Result { r.Pairs = r.Pairs[:1]; return r }, odd},
		{"NaN-cost", match.Blossom, func(r *match.Result) *match.Result { r.Cost = math.NaN(); return r }, odd},
		{"infinite-cost", match.Blossom, func(r *match.Result) *match.Result { r.Cost = math.Inf(1); return r }, odd},
		{"negative-cost", match.Blossom, func(r *match.Result) *match.Result { r.Cost = -1; return r }, odd},
		{"negative-index", match.Blossom, nil, []int{-1, 1, 2, 3}},
		{"out-of-range-index", match.Blossom, nil, []int{0, 1, 2, 4}},
		{"duplicate-index", match.Blossom, nil, []int{0, 1, 2, 2}},
		{"self-pair", match.Blossom, func(r *match.Result) *match.Result { r.Pairs[1] = match.Pair{U: 2, V: 2}; return r }, odd},
		{"reversed-pair", match.Blossom, func(r *match.Result) *match.Result { r.Pairs[1] = match.Pair{U: 3, V: 2}; return r }, odd},
		{"unsorted-pairs", match.Blossom, func(r *match.Result) *match.Result { r.Pairs[0], r.Pairs[1] = r.Pairs[1], r.Pairs[0]; return r }, odd},
		{"foreign-endpoint", match.Blossom, func(r *match.Result) *match.Result { r.Pairs[1].V = 4; return r }, odd},
		{"reused-first-endpoint", match.Blossom, func(r *match.Result) *match.Result { r.Pairs = []match.Pair{{U: 0, V: 2}, {U: 2, V: 3}}; return r }, odd},
		{"reused-second-endpoint", match.Blossom, func(r *match.Result) *match.Result { r.Pairs = []match.Pair{{U: 0, V: 3}, {U: 1, V: 3}}; return r }, odd},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Existing parallel edges must survive any rejected matching unchanged.
			adj := [][]int{{1, 1}, {0, 0}, nil, {}}
			before := coverageCloneAdj(adj)
			vertices := slices.Clone(tc.vertices)
			result := valid()
			if tc.edit != nil {
				result = tc.edit(result)
			}
			var beforePairs []match.Pair
			if result != nil {
				beforePairs = slices.Clone(result.Pairs)
			}
			err := tsp.AppendMatchingResultForTest(vertices, result, adj, tc.algorithm)
			mustErrIs(t, err, tsp.ErrInvalidMatching)
			requireCoverageAdjEqual(t, adj, before)
			mustEqualInts(t, vertices, tc.vertices)
			if adj[2] != nil || adj[3] == nil {
				t.Fatal("nil/empty adjacency row changed")
			}
			if result != nil && !slices.Equal(result.Pairs, beforePairs) {
				t.Fatal("result pairs changed")
			}
		})
	}
	for _, algo := range []match.Algorithm{match.Blossom, match.Greedy} {
		adj := [][]int{{1, 1}, {0, 0}, nil, {}}
		result := valid()
		result.Algorithm = algo
		mustNoError(t, tsp.AppendMatchingResultForTest(odd, result, adj, algo))
		requireCoverageAdjEqual(t, adj, [][]int{{1, 1, 1}, {0, 0, 0}, {3}, {2}})
		if !slices.Equal(result.Pairs, pairs) {
			t.Fatal("adapter changed canonical pairs")
		}
	}
}

// TestMatchingAdapterPreflight preserves legacy empty/odd/epsilon precedence.
// It also verifies that real matching failures do not mutate the destination.
func TestMatchingAdapterPreflight(t *testing.T) {
	opts := tsp.DefaultOptions()
	opts.Eps = 0
	valid := coverageMatrixFromRows(coverageUnitRows(4))
	adj := [][]int{{1}, {0}, {3}, {2}}
	before := coverageCloneAdj(adj)
	mustNoError(t, tsp.BlossomMatchForTest(nil, nil, adj, opts))
	mustNoError(t, tsp.GreedyMatchForTest(nil, nil, adj))
	mustErrIs(t, tsp.BlossomMatchForTest([]int{0}, nil, adj, opts), tsp.ErrInvalidMatching)
	mustErrIs(t, tsp.GreedyMatchForTest([]int{0}, nil, adj), tsp.ErrInvalidMatching)
	mustErrIs(t, tsp.BlossomMatchForTest([]int{0, 1}, nil, adj, opts), tsp.ErrNilDistanceMatrix)
	mustErrIs(t, tsp.BlossomMatchForTest([]int{0, 0}, valid, adj, opts), tsp.ErrInvalidMatching)
	mustErrIs(t, tsp.BlossomMatchForTest([]int{0, 1}, valid, adj, opts), tsp.ErrInvalidOptions)
	opts = tsp.DefaultOptions()
	err := tsp.BlossomMatchForTest([]int{0, 1}, coverageMatrixWithSymmetricEdge(4, 0, 1, math.NaN()), adj, opts)
	mustErrIs(t, err, tsp.ErrNaNInf)
	mustErrIs(t, err, match.ErrNaNInf)
	err = tsp.GreedyMatchForTest([]int{0, 1}, coverageMatrixWithSymmetricEdge(4, 0, 1, math.Inf(1)), adj)
	mustErrIs(t, err, tsp.ErrIncompleteGraph)
	mustErrIs(t, err, match.ErrIncompleteGraph)
	requireCoverageAdjEqual(t, adj, before)
	// Matching policy errors never publish a positive approximation ratio.
	for _, algo := range []tsp.MatchingAlgo{tsp.BlossomMatch, tsp.GreedyMatch, tsp.MatchingAlgo(255)} {
		opts.MatchingAlgo = algo
		ratio, err := tsp.ApplyChristofidesMatchingForTest([]int{0}, valid, adj, opts)
		if err == nil || ratio != tsp.NoApproximationRatio {
			t.Fatalf("policy %v: ratio=%g err=%v", algo, ratio, err)
		}
		requireCoverageAdjEqual(t, adj, before)
	}
}
