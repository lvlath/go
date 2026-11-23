// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp_test extends contract coverage for the unaffected routing kernels.
// These tests check independent route mathematics, not matching implementation details.
package tsp_test

import (
	"errors"
	"testing"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/tsp"
)

// coverageLineMatrix constructs a complete symmetric metric line instance.
// Its unique edge lengths make analytic path and tree costs easy to verify.
func coverageLineMatrix(t *testing.T, n int) *matrix.Dense {
	t.Helper()
	m, err := matrix.NewDense(n, n)
	if err != nil {
		t.Fatalf("NewDense: %v", err)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			delta := i - j
			if delta < 0 {
				delta = -delta
			}
			if err := m.Set(i, j, float64(delta)); err != nil {
				t.Fatalf("Set: %v", err)
			}
		}
	}
	return m
}

// TestTSPCoverage_HeldKarpExactAndSizeLimit checks exact tour cost and resource refusal.
// The line metric has minimum closed Hamiltonian tour length 2*(n-1).
func TestTSPCoverage_HeldKarpExactAndSizeLimit(t *testing.T) {
	m := coverageLineMatrix(t, 5)
	opts := tsp.DefaultOptions()
	opts.EnableLocalSearch = false
	result, err := tsp.HeldKarp(m, opts)
	if err != nil {
		t.Fatalf("HeldKarp: %v", err)
	}
	if result == nil || !result.Exact || !result.Optimal || result.Cost != 8 {
		t.Fatalf("invalid exact result: %+v", result)
	}
	if err := tsp.ValidateTour(result.Tour, 5, opts.StartVertex); err != nil {
		t.Fatalf("ValidateTour: %v", err)
	}
	opts.MaxExactN = 4
	refused, err := tsp.HeldKarp(m, opts)
	if refused != nil || !errors.Is(err, tsp.ErrSizeTooLarge) {
		t.Fatalf("size-limit result=%+v err=%v", refused, err)
	}
}

// TestTSPCoverage_MSTAnalyticAndSingleton checks tree degree and cost invariants.
// A six-node line has a minimum spanning tree of five unit-length edges.
func TestTSPCoverage_MSTAnalyticAndSingleton(t *testing.T) {
	m := coverageLineMatrix(t, 6)
	cost, adj, err := tsp.MinimumSpanningTree(m)
	if err != nil {
		t.Fatalf("MinimumSpanningTree: %v", err)
	}
	if cost != 5 || len(adj) != 6 {
		t.Fatalf("unexpected MST cost=%v adj=%v", cost, adj)
	}
	halfEdges := 0
	for u, neighbors := range adj {
		halfEdges += len(neighbors)
		for _, v := range neighbors {
			if v < 0 || v >= len(adj) {
				t.Fatalf("bad endpoint %d->%d", u, v)
			}
			found := false
			for _, reverse := range adj[v] {
				if reverse == u {
					found = true
					break
				}
			}
			if !found {
				t.Fatalf("missing reverse edge %d->%d", u, v)
			}
		}
	}
	if halfEdges != 10 {
		t.Fatalf("MST must have 5 undirected edges, got %d half edges", halfEdges)
	}
	singleton := coverageLineMatrix(t, 1)
	cost, adj, err = tsp.MinimumSpanningTree(singleton)
	if err != nil || cost != 0 || len(adj) != 1 || len(adj[0]) != 0 {
		t.Fatalf("singleton MST cost=%v adj=%v err=%v", cost, adj, err)
	}
}

// TestTSPCoverage_LocalSearchNeverWorsens verifies separate 2-opt and 3-opt facades.
// Both start from the same legal suboptimal line-metric tour.
func TestTSPCoverage_LocalSearchNeverWorsens(t *testing.T) {
	m := coverageLineMatrix(t, 5)
	initial := []int{0, 2, 1, 3, 4, 0}
	before, err := tsp.TourCost(m, initial)
	if err != nil {
		t.Fatalf("initial TourCost: %v", err)
	}
	if before != 10 {
		t.Fatalf("incorrect analytic baseline: %v", before)
	}
	opts := tsp.DefaultOptions()
	opts.ShuffleNeighborhood = false
	for _, tc := range []struct {
		name  string
		solve func(matrix.Matrix, []int, tsp.Options) (*tsp.Result, error)
	}{
		{"2opt", tsp.TwoOptSearch},
		{"3opt", tsp.ThreeOptSearch},
	} {
		t.Run(tc.name, func(t *testing.T) {
			result, err := tc.solve(m, initial, opts)
			if err != nil {
				t.Fatalf("%s: %v", tc.name, err)
			}
			if result == nil {
				t.Fatal("nil successful search")
			}
			if err := tsp.ValidateTour(result.Tour, 5, 0); err != nil {
				t.Fatalf("invalid result tour: %v", err)
			}
			measured, err := tsp.TourCost(m, result.Tour)
			if err != nil {
				t.Fatalf("result TourCost: %v", err)
			}
			if result.Cost != measured || measured > before {
				t.Fatalf("search worsened cost: initial=%v result=%+v measured=%v", before, result, measured)
			}
			if result.Optimal || result.Exact {
				t.Fatalf("local search must not claim an exact optimum: %+v", result)
			}
		})
	}
	for i, want := range []int{0, 2, 1, 3, 4, 0} {
		if initial[i] != want {
			t.Fatalf("caller tour mutated: %v", initial)
		}
	}
}

// TestTSPCoverage_TourCostOverflowAndPermutation validates arithmetic and shape.
// Two individually finite directed weights can exceed the float64 sum domain.
func TestTSPCoverage_TourCostOverflowAndPermutation(t *testing.T) {
	m, err := matrix.NewDense(2, 2)
	if err != nil {
		t.Fatalf("NewDense: %v", err)
	}
	if err = m.Set(0, 1, 1e308); err != nil {
		t.Fatalf("Set(0,1): %v", err)
	}
	if err = m.Set(1, 0, 1e308); err != nil {
		t.Fatalf("Set(1,0): %v", err)
	}
	cost, err := tsp.TourCost(m, []int{0, 1, 0})
	if cost != 0 || !errors.Is(err, tsp.ErrNumericOverflow) {
		t.Fatalf("overflow cost=%v err=%v", cost, err)
	}
	if err = tsp.ValidatePermutation([]int{1, 1}, 2); err == nil {
		t.Fatal("duplicate permutation unexpectedly accepted")
	}
	if err = tsp.ValidateTour([]int{0, 0, 0}, 2, 0); err == nil {
		t.Fatal("invalid repeated tour unexpectedly accepted")
	}
}
