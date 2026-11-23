// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp_test checks asymmetric route publication against directed arc costs.
// An ATSP cycle cannot be reversed solely to produce a lexicographically smaller tour.
package tsp_test

import (
	"slices"
	"testing"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/tsp"
)

// directedCycleFixture constructs a small directed instance with a unique optimum.
// Every selected low-cost arc lies on 0->3->2->1->0; reversing that cycle
// increases the total cost from 4 to 200.
//
// Implementation:
//   - Stage 1: Allocate a four-vertex matrix.
//   - Stage 2: Set non-optimal directed arcs to fifty and optimal arcs to one.
//   - Stage 3: Return the initialized matrix without an unverified write.
//
// Behavior highlights:
//   - Matrix asymmetry makes reversal of an otherwise valid tour costly.
//
// Inputs:
//   - t: test owner for allocation and matrix-construction failures.
//
// Returns:
//   - *matrix.Dense: a complete directed cost matrix with zero diagonal.
//
// Errors:
//   - Constructor or matrix write errors fail the owning test.
//
// Determinism:
//   - Fixed row-major writes with four explicitly selected directed arcs.
//
// Complexity:
//   - Time O(n^2), Space O(n^2), for fixed n=4.
//
// Notes:
//   - The matrix intentionally violates symmetry; it is not a Christofides input.
//
// AI-Hints:
//   - Do not treat a reversed ATSP tour as an equivalent cycle witness.
func directedCycleFixture(t *testing.T) *matrix.Dense {
	t.Helper()
	const n = 4
	m, err := matrix.NewDense(n, n)
	if err != nil {
		t.Fatal(err)
	}
	for u := 0; u < n; u++ {
		for v := 0; v < n; v++ {
			weight := 0.0
			if u != v {
				weight = 50
			}
			if (u == 0 && v == 3) || (u == 3 && v == 2) ||
				(u == 2 && v == 1) || (u == 1 && v == 0) {
				weight = 1
			}
			if err := m.Set(u, v, weight); err != nil {
				t.Fatal(err)
			}
		}
	}
	return m
}

// TestATSPReverseOrientationChangesObjective establishes the asymmetric cost law.
// A reversed directed Hamiltonian cycle is structurally valid but need not have
// the same objective as the forward cycle.
//
// Implementation:
//   - Stage 1: Construct the directed four-vertex counterexample.
//   - Stage 2: Price forward and reversed arc sequences independently.
//   - Stage 3: Assert their exact integer-representable totals.
//
// Behavior highlights:
//   - Valid tour geometry does not authorize cost-preserving reversal.
//
// Inputs:
//   - t: test owner for source and objective errors.
//
// Returns:
//   - None; two independent TourCost evaluations prove cost 4 versus 200.
//
// Errors:
//   - Unexpected matrix or tour-cost failures fail the test.
//
// Determinism:
//   - The test uses only literal arcs and deterministic arithmetic.
//
// Complexity:
//   - Time O(n), Space O(n) including fixture ownership, n=4.
//
// Notes:
//   - It does not assert anything about symmetric orientation selection.
//
// AI-Hints:
//   - Never publish a reversed ATSP witness with the original tour cost.
func TestATSPReverseOrientationChangesObjective(t *testing.T) {
	m := directedCycleFixture(t)
	forward := []int{0, 3, 2, 1, 0}
	reverse := []int{0, 1, 2, 3, 0}
	best, err := tsp.TourCost(m, forward)
	if err != nil {
		t.Fatal(err)
	}
	bad, err := tsp.TourCost(m, reverse)
	if err != nil {
		t.Fatal(err)
	}
	if best != 4 || bad != 200 {
		t.Fatalf("directed costs were not distinct: best=%g reverse=%g", best, bad)
	}
}

// TestATSPExactSolversPreserveDirectedOrientation protects exact route publication.
// Both Held-Karp and Branch-and-Bound must return the cost-bearing directed
// orientation instead of the lexicographically smaller but costlier reverse.
//
// Implementation:
//   - Stage 1: Select asymmetric mode for both exact TSP kernels.
//   - Stage 2: Solve the unique-optimum complete directed instance.
//   - Stage 3: Compare tour order and recompute cost through public TourCost.
//
// Behavior highlights:
//   - Checks tour witness and public objective jointly, not cost alone.
//
// Inputs:
//   - t: test owner; same immutable input is shared across both solvers.
//
// Returns:
//   - None; assertions demand the unique cost-four directed cycle.
//
// Errors:
//   - Any unexpected exact-solver or TourCost error fails the test.
//
// Determinism:
//   - Fixed matrix, initial policy and deterministic exact solver scans.
//
// Complexity:
//   - Held-Karp O(n^2*2^n); Branch-and-Bound worst-case factorial, n=4.
//
// Notes:
//   - Disable unrelated local-search refinement to isolate exact publication.
//
// AI-Hints:
//   - Do not canonicalize a directed tour by reversal after pricing it.
func TestATSPExactSolversPreserveDirectedOrientation(t *testing.T) {
	m := directedCycleFixture(t)
	opts := tsp.DefaultOptions()
	opts.Symmetric = false
	opts.EnableLocalSearch = false
	opts.BoundAlgo = tsp.SimpleBound

	for _, tc := range []struct {
		name  string
		solve func(matrix.Matrix, tsp.Options) (*tsp.Result, error)
	}{
		{"held-karp", tsp.HeldKarp},
		{"branch-and-bound", tsp.BranchAndBoundSolve},
	} {
		t.Run(tc.name, func(t *testing.T) {
			result, err := tc.solve(m, opts)
			if err != nil || result == nil {
				t.Fatalf("solve: result=%+v err=%v", result, err)
			}
			if !slices.Equal(result.Tour, []int{0, 3, 2, 1, 0}) {
				t.Fatalf("asymmetric optimal tour reversed: %v", result.Tour)
			}
			actual, err := tsp.TourCost(m, result.Tour)
			if err != nil || result.Cost != actual || result.Cost != 4 {
				t.Fatalf("cost=%g actual=%g err=%v", result.Cost, actual, err)
			}
		})
	}
}

// TestATSPLocalSearchPreservesDirectedOrientation checks stable local-optimum output.
// With no improving move available, 2-opt and 3-opt must retain the caller's
// directed cycle orientation and its actual directed objective.
//
// Implementation:
//   - Stage 1: Seed the unique optimum and disable symmetric policy.
//   - Stage 2: Invoke each local-search kernel without a wall-clock budget.
//   - Stage 3: Assert identical witness and independently priced cost.
//
// Behavior highlights:
//   - The zero-move finalization path cannot reverse an ATSP incumbent.
//
// Inputs:
//   - t: owner of the deterministic local-search regression.
//
// Returns:
//   - None; both policies must publish an unchanged cost-four directed tour.
//
// Errors:
//   - Any local-search or TourCost failure fails the test.
//
// Determinism:
//   - Fixed start and strict deterministic neighborhood traversal.
//
// Complexity:
//   - Constant-size test: 2-opt candidate work O(n^3), 3-opt O(n^3).
//
// Notes:
//   - The large-scale improvement test exercises a different finalization path.
//
// AI-Hints:
//   - Do not confuse stable tour orientation with absence of search execution.
func TestATSPLocalSearchPreservesDirectedOrientation(t *testing.T) {
	m := directedCycleFixture(t)
	start := []int{0, 3, 2, 1, 0}
	opts := tsp.DefaultOptions()
	opts.Symmetric = false
	for _, tc := range []struct {
		name  string
		solve func(matrix.Matrix, []int, tsp.Options) (*tsp.Result, error)
	}{
		{"2-opt", tsp.TwoOptSearch},
		{"3-opt", tsp.ThreeOptSearch},
	} {
		t.Run(tc.name, func(t *testing.T) {
			result, err := tc.solve(m, start, opts)
			if err != nil || result == nil {
				t.Fatalf("solve: result=%+v err=%v", result, err)
			}
			if !slices.Equal(result.Tour, start) {
				t.Fatalf("directed tour changed: got=%v want=%v", result.Tour, start)
			}
			actual, err := tsp.TourCost(m, result.Tour)
			if err != nil || result.Cost != actual || result.Cost != 4 {
				t.Fatalf("cost=%g actual=%g err=%v", result.Cost, actual, err)
			}
		})
	}
}

// TestATSPLocalSearchLargeScaleCostAgreesWithPublishedArcs protects cost publication.
// Incremental delta summation at a large common weight scale can lose low-order
// bits, even when every accepted directed reversal is a valid improving move.
//
// Implementation:
//   - Stage 1: Build a complete asymmetric matrix with a large common offset.
//   - Stage 2: Require one accepted improvement from both local-search kernels.
//   - Stage 3: Independently price the returned directed tour through TourCost.
//
// Behavior highlights:
//   - The test compares exact float64 publication at the same summation order.
//   - An iteration cap forces the early-result path without a wall-clock timeout.
//
// Inputs:
//   - t: testing handle; the input matrix and initial tour are deterministic.
//
// Returns:
//   - None; failures report the affected kernel and measured cost discrepancy.
//
// Errors:
//   - All unexpected constructor, solver, and TourCost errors fail the test.
//
// Determinism:
//   - Explicit data, fixed start vertex, deterministic first-improvement order.
//
// Complexity:
//   - Constant-size matrix; at most one accepted move per kernel.
//
// Notes:
//   - The test does not change any solver delta or tolerance to force its answer.
//
// AI-Hints:
//   - A reported iteration cost must price the published tour, not a previous tour.
//   - Do not use percentage tolerance to conceal final cost drift.
func TestATSPLocalSearchLargeScaleCostAgreesWithPublishedArcs(t *testing.T) {
	const n = 4
	weights := [...]float64{
		0, 1000000000195.7, 1000000000137.6, 1000000000176.1,
		1000000000193.5, 0, 1000000000193.3, 1000000000146.4,
		1000000000075.2, 1000000000019.2, 0, 1000000000081.5,
		1000000000130.7, 1000000000036.3, 1000000000144.8, 0,
	}
	m, err := matrix.NewDense(n, n)
	if err != nil {
		t.Fatal(err)
	}
	if err = m.Fill(weights[:]); err != nil {
		t.Fatal(err)
	}
	start := []int{0, 1, 2, 3, 0}
	original, err := tsp.TourCost(m, start)
	if err != nil {
		t.Fatal(err)
	}
	policy := tsp.DefaultOptions()
	policy.Symmetric = false
	policy.TwoOptMaxIters = 1
	policy.ThreeOptMaxMoves = 1

	for _, tc := range []struct {
		name  string
		solve func(matrix.Matrix, []int, tsp.Options) (*tsp.Result, error)
	}{
		{name: "2-opt", solve: tsp.TwoOptSearch},
		{name: "3-opt", solve: tsp.ThreeOptSearch},
	} {
		t.Run(tc.name, func(t *testing.T) {
			result, err := tc.solve(m, start, policy)
			if err != nil || result == nil {
				t.Fatalf("local search result=%+v: %v", result, err)
			}
			if result.Iterations != 1 {
				t.Fatalf("expected a capped accepted move: iterations=%d", result.Iterations)
			}
			if result.Cost >= original {
				t.Fatalf("non-improving published solution: before=%g after=%g", original, result.Cost)
			}
			actual, err := tsp.TourCost(m, result.Tour)
			if err != nil {
				t.Fatal(err)
			}
			if result.Cost != actual {
				t.Fatalf("published cost does not price the directed arcs: reported=%.16g actual=%.16g diff=%.16g tour=%v", result.Cost, actual, result.Cost-actual, result.Tour)
			}
			if !slices.Equal(start, []int{0, 1, 2, 3, 0}) {
				t.Fatalf("local search mutated caller input: %v", start)
			}
		})
	}
}
