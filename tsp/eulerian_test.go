// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp_test validates the Eulerian-circuit construction (Hierholzer)
// and the shortcut-to-Hamiltonian step used by the Christofides pipeline.
// Scope (focused):
//  1. Build an Eulerian multigraph by *doubling* the MST edges (tree-doubling).
//     - Check that EulerianCircuit returns a closed walk of length |E|+1,
//     starts/ends at startV, and induces even degrees for every vertex.
//  2. Shortcut that Eulerian walk to a Hamiltonian cycle and validate:
//     - tsp.ValidateTour passes,
//     - tsp.TourCost is positive/finite,
//     - tour cost ≤ 2 · MST  (classical bound of the doubled-tree method).
//  3. Determinism and a degenerate defensive case (no edges).
package tsp_test

import (
	"math"
	"slices"
	"testing"

	"github.com/lvlath/go/tsp"
)

// degFromCircuit computes vertex degrees induced by a walk of edges.
// A closed Eulerian circuit encodes E edges as consecutive pairs walk[i]-walk[i+1].
// Complexity: O(E).
func degFromCircuit(walk []int, n int) []int {
	deg := make([]int, n)
	// Walk length is E+1; iterate edges between consecutive vertices.
	var i int
	for i = 0; i+1 < len(walk); i++ {
		u := walk[i]
		v := walk[i+1]
		if 0 <= u && u < n { // ??
			deg[u]++ // ??
		}
		if 0 <= v && v < n { // ??
			deg[v]++ // ??
		}
	}

	return deg
}

// -----------------------------------------------------------------------------
// 1) Eulerian circuit on doubled MST: even degrees and expected length.
// -----------------------------------------------------------------------------

func TestEulerian_DoubleMST_EvenDegrees_And_Length(t *testing.T) {
	// Regular pentagon (symmetric Euclidean metric, zero diagonal).
	pts := [][2]float64{
		{1, 0},
		{math.Cos(2 * math.Pi / 5), math.Sin(2 * math.Pi / 5)},
		{math.Cos(4 * math.Pi / 5), math.Sin(4 * math.Pi / 5)},
		{math.Cos(6 * math.Pi / 5), math.Sin(6 * math.Pi / 5)},
		{math.Cos(8 * math.Pi / 5), math.Sin(8 * math.Pi / 5)},
	}
	m := euclid(pts) // helper from testutil_test.go

	// Build a *tree-doubled* Eulerian multigraph:
	//  1) compute MST adjacency (simple graph),
	//  2) duplicate every edge to make all degrees even.
	_, mstAdj, err := tsp.MinimumSpanningTree(m)
	if err != nil {
		t.Fatalf("MinimumSpanningTree failed: %v", err)
	}
	multi := doubleAdj(mstAdj) // Eulerian by construction

	// Expected Eulerian circuit length is |E|+1 with |E| = 2·(n-1) after doubling.
	const n = 5
	wantEdges := 2 * (n - 1)
	if gotEdges := edgesCount(multi); gotEdges != wantEdges {
		t.Fatalf("unexpected multigraph size: got |E|=%d want=%d", gotEdges, wantEdges)
	}

	// Run Hierholzer starting at canonical startV.
	walk, _ := tsp.EulerianCircuit(multi, startV)

	// Basic structure checks: closed walk, exact length |E|+1.
	if len(walk) != wantEdges+1 {
		t.Fatalf("walk length mismatch: got=%d want=%d", len(walk), wantEdges+1)
	}
	if walk[0] != startV || walk[len(walk)-1] != startV {
		t.Fatalf("walk must start/end at %d: first=%d last=%d", startV, walk[0], walk[len(walk)-1])
	}

	// Degree parity induced by the circuit must be even for every vertex.
	deg := degFromCircuit(walk, n)
	var v int
	for v = 0; v < n; v++ {
		if (deg[v] & 1) != 0 {
			t.Fatalf("degree parity must be even: deg[%d]=%d", v, deg[v])
		}
	}
}

// -----------------------------------------------------------------------------
// 2) Shortcut to Hamiltonian: validity + finite cost + cost ≤ 2×MST.
// -----------------------------------------------------------------------------

func TestEulerian_ShortcutToHamiltonian_Valid_And_CostBound(t *testing.T) {
	// Slightly rippled hexagon to avoid accidental identical distances.
	const n = 6
	pts := make([][2]float64, n)
	var i int
	var th float64
	for i = 0; i < n; i++ {
		th = 2 * math.Pi * float64(i) / float64(n)
		r := 1.0 + 0.03*math.Cos(3*th)
		pts[i] = [2]float64{r * math.Cos(th), r * math.Sin(th)}
	}
	m := euclid(pts)

	// Compute MST and build a doubled-edge Eulerian multigraph.
	mstW, mstAdj, err := tsp.MinimumSpanningTree(m)
	if err != nil {
		t.Fatalf("MinimumSpanningTree failed: %v", err)
	}
	multi := doubleAdj(mstAdj)

	// Eulerian walk (Hierholzer) and shortcut to a Hamiltonian cycle.
	walk, _ := tsp.EulerianCircuit(multi, startV)
	tour, err := tsp.ShortcutEulerianToHamiltonian(walk, n, startV)
	if err != nil {
		t.Fatalf("ShortcutEulerianToHamiltonian failed: %v", err)
	}

	// Tour invariants and finite positive cost.
	if err = tsp.ValidateTour(tour, n, startV); err != nil {
		t.Fatalf("Hamiltonian tour invalid: %v", err)
	}
	cost, err := tsp.TourCost(m, tour)
	if err != nil {
		t.Fatalf("TourCost failed: %v", err)
	}
	if !(cost > 0) || math.IsInf(cost, 0) || math.IsNaN(cost) {
		t.Fatalf("unexpected tour cost: %.12f", cost)
	}

	// Classical doubled-tree bound: cost ≤ 2 · MST (compare with stabilized rounding).
	limit := 2.0 * mstW
	if round1e9(cost) > round1e9(limit) {
		t.Fatalf("shortcut cost exceeds 2×MST: cost=%.12f mst=%.12f limit=%.12f", cost, mstW, limit)
	}
}

// -----------------------------------------------------------------------------
// 3) Determinism and a zero-edge defensive case.
// -----------------------------------------------------------------------------

func TestEulerian_Determinism_Repeat3(t *testing.T) {
	// Nontrivial metric - small ripple to avoid ties; MST is deterministic in our impl.
	const n = 9
	pts := make([][2]float64, n)
	var i int
	var th float64
	for i = 0; i < n; i++ {
		th = 2 * math.Pi * float64(i) / float64(n)
		r := 1.0 + 0.02*math.Sin(5*th)
		pts[i] = [2]float64{r * math.Cos(th), r * math.Sin(th)}
	}
	m := euclid(pts)

	_, adj, err := tsp.MinimumSpanningTree(m)
	if err != nil {
		t.Fatalf("MinimumSpanningTree failed: %v", err)
	}
	multi := doubleAdj(adj)

	var base []int
	Repeat(t, 3, func(t *testing.T) {
		w, _ := tsp.EulerianCircuit(multi, startV)
		if base == nil {
			base = append([]int(nil), w...) // capture first outcome
			return
		}
		if !slices.Equal(w, base) {
			t.Fatalf("nondeterministic Eulerian circuit.\nfirst: %v\nthis:  %v", base, w)
		}
	})
}

func TestEulerian_Defensive_NoEdges_ReturnsStartOnly(t *testing.T) {
	// Build an "empty" undirected graph with 4 vertices and 0 edges.
	adj := make([][]int, 4)
	// Call EulerianCircuit; contract says no panics and returns [start].
	walk, _ := tsp.EulerianCircuit(adj, startV)
	if len(walk) != 1 || walk[0] != startV {
		t.Fatalf("want single-vertex walk [%d], got %v", startV, walk)
	}
}
