// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp_test verifies Prim's MST (O(n^2)) over dense metric matrices.
// Focus:
//  1. Correct total weight and tree structure on a small instance.
//  2. Deterministic result under uniform weights (tie-breaking on indices).
//  3. Proper sentinel on disconnection when +Inf edges isolate a vertex.
package tsp_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/tsp"
)

// degFromAdj computes degrees for an undirected simple graph encoded by adjacency.
// Time: O(n + E), where E is the number of undirected edges.
// Memory: O(n).
func degFromAdj(adj [][]int) []int {
	n := len(adj)            // number of vertices
	var deg = make([]int, n) // degree accumulator
	var u int                // row index
	for u = 0; u < n; u++ {  // iterate all rows
		deg[u] = len(adj[u]) // degree equals row length in simple graphs
	}
	return deg // return degree vector
}

// -----------------------------------------------------------------------------
// 1) Small graph (n=4): known MST total = 3 and a path structure (0-1-2-3).
// -----------------------------------------------------------------------------

func TestMST_SmallGraph_TotalWeight_And_Structure(t *testing.T) {
	// Build a 4×4 symmetric metric where the unique MST is a path 0-1-2-3 of total weight 3.
	// Distances:
	//  - Edges on the path have weight 1 (0-1, 1-2, 2-3),
	//  - Cross edges have weight 2 (heavier to avoid ambiguity).
	var a = [][]float64{
		{0, 1, 2, 2},
		{1, 0, 1, 2},
		{2, 1, 0, 1},
		{2, 2, 1, 0},
	}
	var m = testDense{a: a} // minimal matrix type from test utilities

	// Run Prim O(n^2) via the public entry.
	var total float64
	var adj [][]int
	var err error
	total, adj, err = tsp.MinimumSpanningTree(m) // build MST
	if err != nil {                              // strict: no error expected
		t.Fatalf("MinimumSpanningTree failed: %v", err)
	}

	// The total weight must equal 3 (rounded by implementation to 1e-9).
	// Use strict float comparison helper to avoid CI flakes.
	mustFloatClose(t, total, 3.0, 0, 1e-12)

	// Structure: a tree on 4 vertices has |E|=3 edges.
	if edgesCount(adj) != 3 {
		t.Fatalf("unexpected number of edges: got=%d want=3; adj=%+v", edgesCount(adj), adj)
	}

	// Degrees on a path of length 3 are {1,2,2,1} in some order;
	// here we know it's 0-1-2-3, so degrees are exactly [1,2,2,1].
	var deg = degFromAdj(adj)
	var want = []int{1, 2, 2, 1}
	mustEqualInts(t, deg, want) // exact structural match
}

// -----------------------------------------------------------------------------
// 2) Tie-breaking determinism on uniform weights: star centered at 0.
//    With all non-diagonal weights identical and start fixed at 0,
//    Prim picks vertices in increasing index order => parent[v]=0 for v>0.
// -----------------------------------------------------------------------------

func TestMST_TieBreak_UniformWeights_StarAtZero(t *testing.T) {
	const n = 6 // graph size
	// Build a complete symmetric matrix with 0 on the diagonal and 1 off-diagonal.
	var a = make([][]float64, n)
	var i, j int
	for i = 0; i < n; i++ { // allocate and fill each row
		a[i] = make([]float64, n)
		for j = 0; j < n; j++ {
			if i == j { // exact zeros on the diagonal
				a[i][j] = 0
			} else { // uniform cost for all pairs
				a[i][j] = 1
			}
		}
	}
	var m = testDense{a: a} // matrix under test

	// Run MST; on ties the implementation picks the smallest-index vertex.
	var total float64
	var adj [][]int
	var err error
	total, adj, err = tsp.MinimumSpanningTree(m)
	if err != nil {
		t.Fatalf("MinimumSpanningTree failed: %v", err)
	}

	// Weight sanity: star with n-1 edges of weight 1 ⇒ total = n-1.
	mustFloatClose(t, total, float64(n-1), 0, 1e-12)

	// Shape: star centered at 0 ⇒ deg(0)=n-1, deg(i)=1 for all i>0.
	var deg = degFromAdj(adj)
	if deg[0] != n-1 {
		t.Fatalf("deg(0)=%d; want %d; adj=%+v", deg[0], n-1, adj)
	}
	var v int
	for v = 1; v < n; v++ {
		if deg[v] != 1 {
			t.Fatalf("deg(%d)=%d; want 1; adj=%+v", v, deg[v], adj)
		}
	}
}

// -----------------------------------------------------------------------------
// 3) Disconnection sentinel: +Inf isolates a vertex ⇒ ErrIncompleteGraph.
// -----------------------------------------------------------------------------

func TestMST_Disconnected_ByInf_ErrIncompleteGraph(t *testing.T) {
	// 4×4 symmetric matrix where vertex 3 is isolated via +Inf edges.
	var inf = math.Inf(1)
	var a = [][]float64{
		{0, 1, 1, inf},
		{1, 0, 1, inf},
		{1, 1, 0, inf},
		{inf, inf, inf, 0},
	}
	var m = testDense{a: a}

	// MST must fail with ErrIncompleteGraph; no panics, no partial results.
	var _, _, err = tsp.MinimumSpanningTree(m)
	if !errors.Is(err, tsp.ErrIncompleteGraph) {
		t.Fatalf("want ErrIncompleteGraph, got %v", err)
	}
}
