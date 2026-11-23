// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp_test protects public Eulerian handling of parallel edges and matching.
// Tests build multigraphs through real public APIs instead of private topology hooks.
package tsp_test

import (
	"errors"
	"testing"

	"github.com/lvlath/go/match"
	"github.com/lvlath/go/tsp"
)

// TestEulerianParallelEdgesPreserveMultiplicity checks a reciprocal raw multigraph.
// Its six undirected edges must be traversed once each without deduplication.
func TestEulerianParallelEdgesPreserveMultiplicity(t *testing.T) {
	adjacency := [][]int{
		{2, 1, 2, 1},
		{2, 0, 2, 0},
		{1, 0, 1, 0},
	}
	walk, err := tsp.EulerianCircuit(adjacency, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := verifyEulerianWalkUsesEveryEdgeForTest(adjacency, walk, 0); err != nil {
		t.Fatalf("multigraph edge multiplicities were lost: %v", err)
	}
}

// TestEulerianRejectsNonReciprocalEvenDegrees rejects malformed half-edge topology.
// Every vertex has even degree, but opposite directed multiplicities disagree.
func TestEulerianRejectsNonReciprocalEvenDegrees(t *testing.T) {
	adjacency := [][]int{
		{1, 1, 2, 2},
		{2, 2},
		{0, 0},
	}
	walk, err := tsp.EulerianCircuit(adjacency, 0)
	if walk != nil || !errors.Is(err, tsp.ErrNonEulerian) {
		t.Fatalf("nonreciprocal multigraph: walk=%v err=%v", walk, err)
	}
}

// TestChristofidesMatchingProducesEulerianMultigraphSeedN16Seed2 protects parity.
// It composes public Prim MST and match.Perfect, then checks every edge of the Euler tour.
func TestChristofidesMatchingProducesEulerianMultigraphSeedN16Seed2(t *testing.T) {
	const n, seed = 16, int64(2)
	distance := seededMetricCompleteForTest(t, n, seed)
	_, adjacency, err := tsp.MinimumSpanningTree(distance)
	if err != nil {
		t.Fatalf("MST: %v", err)
	}
	odd := make([]int, 0, n)
	for vertex, neighbors := range adjacency {
		if len(neighbors)&1 != 0 {
			odd = append(odd, vertex)
		}
	}
	if len(odd)&1 != 0 {
		t.Fatalf("odd-degree set has odd size: %v", odd)
	}
	result, err := match.Perfect(distance, odd)
	if err != nil {
		t.Fatalf("Blossom on MST odd vertices: %v", err)
	}
	if result == nil {
		t.Fatal("solver returned a nil result without an error")
	}
	if len(result.Pairs) != len(odd)/2 {
		t.Fatalf("incomplete matching: %+v", result)
	}
	// Each matched endpoint's parity changes once; the complete graph becomes Eulerian.
	for _, edge := range result.Pairs {
		adjacency[edge.U] = append(adjacency[edge.U], edge.V)
		adjacency[edge.V] = append(adjacency[edge.V], edge.U)
	}
	for vertex, neighbors := range adjacency {
		if len(neighbors)&1 != 0 {
			t.Fatalf("odd degree after matching: v=%d neighbors=%v", vertex, neighbors)
		}
	}
	walk, err := tsp.EulerianCircuit(adjacency, int(seed%n))
	if err != nil {
		t.Fatalf("postmatching Eulerian circuit: %v", err)
	}
	if err := verifyEulerianWalkUsesEveryEdgeForTest(adjacency, walk, int(seed%n)); err != nil {
		t.Fatalf("Eulerian graph traversal: %v", err)
	}
}

// TestChristofidesMetricRegressionSeedN16Seed2 checks full public TSP integration.
// It independently validates route cardinality, closure and exact starting position.
func TestChristofidesMetricRegressionSeedN16Seed2(t *testing.T) {
	const n, seed = 16, int64(2)
	distance := seededMetricCompleteForTest(t, n, seed)
	opts := tsp.DefaultOptions()
	opts.Algo = tsp.Christofides
	opts.Symmetric = true
	opts.MatchingAlgo = tsp.BlossomMatch
	opts.EnableLocalSearch = false
	opts.StartVertex = int(seed % n)
	result, err := tsp.ChristofidesSolve(distance, opts)
	if err != nil {
		t.Fatalf("Christofides: %v", err)
	}
	if err := validateResultForTest(result, n, opts.StartVertex, tsp.Christofides); err != nil {
		t.Fatalf("invalid route: %v", err)
	}
}

// TestEulerianParallelEdgePairingAtArbitraryStart guards a larger multigraph.
// Every pair of opposite half-edges must be consumed even with parallel edges.
func TestEulerianParallelEdgePairingAtArbitraryStart(t *testing.T) {
	adjacency := [][]int{
		{1, 4},
		{0, 4},
		{7, 9},
		{4, 6, 10, 10},
		{0, 1, 3, 8},
		{7, 11},
		{3, 12},
		{2, 5},
		{4, 10},
		{2, 12},
		{3, 3, 8, 14},
		{5, 14},
		{6, 9},
		{14, 15},
		{10, 11, 13, 15},
		{13, 14},
	}
	walk, err := tsp.EulerianCircuit(adjacency, 2)
	if err != nil {
		t.Fatal(err)
	}
	if err := verifyEulerianWalkUsesEveryEdgeForTest(adjacency, walk, 2); err != nil {
		t.Fatalf("parallel-edge pairing: %v", err)
	}
}
