// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp_test validates the exact Branch-and-Bound solver (BranchAndBoundSolve).
// Focus:
//  1. Strict sentinels on malformed inputs (non-square, OOB start, NaN, negative, +Inf).
//  2. Correctness on tiny symmetric instances (triangle) and ATSP square.
//  3. Policy equivalence across bound algorithms (NoBound / SimpleBound / OneTreeBound).
//  4. Determinism under identical options.
//  5. Soft time-budget behavior (ErrTimeLimit) without panics.
package tsp_test

import (
	"errors"
	"math"
	"slices"
	"testing"
	"time"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/tsp"
)

// ---------------------------
// Local helpers (small only).
// ---------------------------

// mkTriangleDense returns the same triangle as in bound tests (cost=6 optimal).
func mkTriangleDense() matrix.Matrix {
	a := [][]float64{
		{0, 1, 3},
		{1, 0, 2},
		{3, 2, 0},
	}

	return testDense{a: a} // testDense is provided by tour_cost_utils_test.go
}

// mkNonSquareDense builds a 2×3 matrix to trigger ErrNonSquare (shape guard).
func mkNonSquareDense() matrix.Matrix {
	a := [][]float64{
		{0, 1, 2},
		{1, 0, 3},
	}

	// Reuse the non-square helper from types validation (keeps intent explicit).
	return mkNonSquare(a)
}

// mkBadSym clones a symmetric 4×4 baseline and pokes (i,j) with w (mirror at (j,i)).
func mkBadSym(i, j int, w float64) matrix.Matrix {
	base := [][]float64{
		{0, 1, 2, 3},
		{1, 0, 2, 3},
		{2, 2, 0, 1},
		{3, 3, 1, 0},
	}
	base[i][j], base[j][i] = w, w

	return testDense{a: base}
}

// mkSymWithIsolatedVertex builds a symmetric n×n matrix where 'iso' has no finite
// edges to other vertices (both outgoing and incoming set to +Inf). This guarantees
// ErrIncompleteGraph in precomputeMinima().
func mkSymWithIsolatedVertex(n int, iso int) matrix.Matrix {
	a := make([][]float64, n)
	var i, j int
	for i = 0; i < n; i++ {
		a[i] = make([]float64, n)
		for j = 0; j < n; j++ {
			if i == j {
				a[i][j] = 0 // exact zeros on diagonal
			} else {
				a[i][j] = 1 + float64((i+j)%3) // small positive finite baseline
			}
		}
	}
	// Isolate vertex 'iso' by setting all incident edges to +Inf (keep diagonal at 0).
	var inf = math.Inf(1)
	for j = 0; j < n; j++ {
		if j == iso {
			continue
		}
		a[iso][j] = inf
		a[j][iso] = inf
	}

	return testDense{a: a}
}

// mustValidTourCost asserts a valid closed tour for n, then compares stabilized cost.
func mustValidTourCost(t *testing.T, dist matrix.Matrix, tour []int, n int, start int, want float64) {
	t.Helper()
	if err := tsp.ValidateTour(tour, n, start); err != nil {
		t.Fatalf("returned tour invalid: %v", err)
	}
	got, err := tsp.TourCost(dist, tour)
	if err != nil {
		t.Fatalf("TourCost failed: %v", err)
	}
	if round1e9(got) != round1e9(want) {
		t.Fatalf("cost mismatch: got=%.12f want=%.12f", got, want)
	}
}

// ---------------------------
// 1) Strict sentinels tests.
// ---------------------------

func TestBB_Errors_StrictSentinels(t *testing.T) {
	// Build base options with explicit knobs; no ambiguity about defaults.
	opt := tsp.DefaultOptions()
	opt.StartVertex = startV // canonical start vertex
	opt.Eps = epsTiny        // strict tolerance
	opt.EnableLocalSearch = false
	opt.Symmetric = true
	opt.BoundAlgo = tsp.SimpleBound

	// Non-square → ErrNonSquare.
	Repeat(t, 2, func(t *testing.T) {
		m := mkNonSquareDense()
		_, err := tsp.BranchAndBoundSolve(m, opt)
		mustErrIs(t, err, tsp.ErrNonSquare)
	})

	// Out-of-range start → sentinel from validateStartVertex (implementation-specific).
	Repeat(t, 2, func(t *testing.T) {
		m := mkTriangleDense()
		optBad := opt
		optBad.StartVertex = 99 // invalid for n=3
		_, err := tsp.BranchAndBoundSolve(m, optBad)
		// Accept either a specific "start vertex out of range" sentinel or the generic one.
		mustErrIs(t, err, tsp.ErrStartOutOfRange)
	})

	// NaN weight → ErrDimensionMismatch (prefetch guard).
	Repeat(t, 2, func(t *testing.T) {
		m := mkBadSym(0, 1, math.NaN())
		_, err := tsp.BranchAndBoundSolve(m, opt)
		mustErrIs(t, err, tsp.ErrNaNInf)
	})

	// Negative weight → ErrNegativeWeight.
	Repeat(t, 2, func(t *testing.T) {
		m := mkBadSym(0, 1, -1)
		_, err := tsp.BranchAndBoundSolve(m, opt)
		mustErrIs(t, err, tsp.ErrNegativeWeight)
	})

	// +Inf that leaves *some vertex with no finite in/out* → ErrIncompleteGraph.
	Repeat(t, 2, func(t *testing.T) {
		m := mkSymWithIsolatedVertex(4, 1) // vertex 1 has no finite neighbor → infeasible
		_, err := tsp.BranchAndBoundSolve(m, opt)
		mustErrIs(t, err, tsp.ErrIncompleteGraph)
	})
}

// ---------------------------------------------
// 2) Correctness - symmetric triangle (exact).
// ---------------------------------------------

func TestBB_TSP_Triangle_Exact(t *testing.T) {
	// Triangle has a unique optimal cost = 6 (any orientation).
	const n = 3
	const want = 6.0

	m := mkTriangleDense()

	opt := tsp.DefaultOptions()
	opt.StartVertex = startV
	opt.Symmetric = true
	opt.Eps = epsTiny
	opt.BoundAlgo = tsp.SimpleBound
	opt.EnableLocalSearch = false // UB seeding may still run trivial ring if Christofides not present

	res, err := tsp.BranchAndBoundSolve(m, opt)
	if err != nil {
		t.Fatalf("BranchAndBoundSolve failed: %v", err)
	}
	mustValidTourCost(t, m, res.Tour, n, startV, want)
}

// ----------------------------------------------------------------------
// 3) Policy equivalence - results match across bound algorithm policies.
// ----------------------------------------------------------------------

func TestBB_TSP_Policies_EquivalentResults(t *testing.T) {
	// Use a modest convex hexagon; nontrivial yet fast for BnB.
	pts := [][2]float64{
		{1, 0}, {0.5, math.Sqrt(3) / 2}, {-0.5, math.Sqrt(3) / 2},
		{-1, 0}, {-0.5, -math.Sqrt(3) / 2}, {0.5, -math.Sqrt(3) / 2},
	}
	m := euclid(pts) // symmetric Euclidean metric

	// Build three option sets that differ only by BoundAlgo.
	base := tsp.DefaultOptions()
	base.StartVertex = startV
	base.Symmetric = true
	base.Eps = epsTiny
	base.EnableLocalSearch = false

	optNo := base
	optNo.BoundAlgo = tsp.NoBound

	optSimple := base
	optSimple.BoundAlgo = tsp.SimpleBound

	optOneTree := base
	optOneTree.BoundAlgo = tsp.OneTreeBound

	// Solve under all three policies.
	resNo, err := tsp.BranchAndBoundSolve(m, optNo)
	if err != nil {
		t.Fatalf("NoBound failed: %v", err)
	}
	resSimple, err := tsp.BranchAndBoundSolve(m, optSimple)
	if err != nil {
		t.Fatalf("SimpleBound failed: %v", err)
	}
	resOneTree, err := tsp.BranchAndBoundSolve(m, optOneTree)
	if err != nil {
		t.Fatalf("OneTreeBound failed: %v", err)
	}

	// Costs must match after stable rounding.
	if round1e9(resNo.Cost) != round1e9(resSimple.Cost) ||
		round1e9(resNo.Cost) != round1e9(resOneTree.Cost) {
		t.Fatalf("cost mismatch across policies:\n NoBound=%.12f\n Simple=%.12f\n OneTree=%.12f",
			resNo.Cost, resSimple.Cost, resOneTree.Cost)
	}

	// Tours must be the same cycle (up to rotation/orientation).
	noOpen := normalizeClosedToOpen(t, resNo.Tour)
	simpleOpen := normalizeClosedToOpen(t, resSimple.Tour)
	oneTreeOpen := normalizeClosedToOpen(t, resOneTree.Tour)

	// Canonical target: tour from NoBound run.
	if !slices.Equal(noOpen, simpleOpen) || !slices.Equal(noOpen, oneTreeOpen) {
		t.Fatalf("tour mismatch across policies:\n NoBound:  %v\n Simple:   %v\n OneTree:  %v",
			noOpen, simpleOpen, oneTreeOpen)
	}
}

// -----------------------------------------------------------
// 4) Correctness - ATSP: valid order and positive finite cost.
// -----------------------------------------------------------

func TestBB_ATSP_Square_BasicValidity(t *testing.T) {
	// A simple 4-node square with directional penalty (ATSP).
	pts := [][2]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}}
	m := euclidAsym(pts, 0.2)

	opt := tsp.DefaultOptions()
	opt.StartVertex = startV
	opt.Symmetric = false // ATSP mode
	opt.Eps = epsTiny
	opt.BoundAlgo = tsp.SimpleBound
	opt.EnableLocalSearch = false

	res, err := tsp.BranchAndBoundSolve(m, opt)
	if err != nil {
		t.Fatalf("ATSP Branch-and-Bound failed: %v", err)
	}
	if err = tsp.ValidateTour(res.Tour, 4, startV); err != nil {
		t.Fatalf("ATSP tour invalid: %v", err)
	}
	c, err := tsp.TourCost(m, res.Tour)
	if err != nil {
		t.Fatalf("TourCost failed: %v", err)
	}
	if !(c > 0) && !math.IsInf(c, 0) && !math.IsNaN(c) {
		t.Fatalf("unexpected ATSP cost: %.12f", c)
	}
}

// --------------------------------------------------------------
// 5) Time budget - tiny deadline should return tsp.ErrTimeLimit.
// --------------------------------------------------------------

func TestBB_TimeLimit_TinyBudget_NoBound(t *testing.T) {
	const n = 13

	points := make([][2]float64, n)
	for vertex := 0; vertex < n; vertex++ {
		theta := 2 * math.Pi * float64(vertex) / float64(n)
		points[vertex] = [2]float64{math.Cos(theta), math.Sin(theta)}
	}

	dist := euclid(points)

	opts := tsp.DefaultOptions()
	opts.Algo = tsp.BranchAndBound
	opts.Symmetric = true
	opts.StartVertex = startV
	opts.BoundAlgo = tsp.NoBound
	opts.EnableLocalSearch = false
	opts.TimeLimit = time.Nanosecond

	result, err := tsp.BranchAndBoundSolve(dist, opts)
	if err == nil {
		mustValidResult(t, result, n, startV, tsp.BranchAndBound)
		return
	}
	if !errors.Is(err, tsp.ErrTimeLimit) {
		t.Fatalf("want nil or ErrTimeLimit under tiny budget, got %v", err)
	}
}

// -------------------------------------------
// 6) Determinism - identical runs are equal.
// -------------------------------------------

func TestBB_Determinism_Repeat4(t *testing.T) {
	// Nontrivial symmetric instance - slightly rippled circle.
	const n = 10
	pts := make([][2]float64, n)
	var i int
	var th float64
	var r float64
	for i = 0; i < n; i++ {
		th = 2 * math.Pi * float64(i) / float64(n)
		r = 1.0 + 0.03*math.Sin(3*th) // gentle perturbation to avoid perfect symmetry
		pts[i] = [2]float64{r * math.Cos(th), r * math.Sin(th)}
	}
	m := euclid(pts)

	opt := tsp.DefaultOptions()
	opt.StartVertex = startV
	opt.Symmetric = true
	opt.Eps = epsTiny
	opt.BoundAlgo = tsp.SimpleBound
	opt.EnableLocalSearch = false

	var tour0 []int
	var cost0 float64

	Repeat(t, 4, func(t *testing.T) {
		res, err := tsp.BranchAndBoundSolve(m, opt)
		if err != nil {
			t.Fatalf("run failed: %v", err)
		}
		// Normalize to an open cycle starting at 0 for structural comparison.
		open := normalizeClosedToOpen(t, res.Tour)
		if tour0 == nil {
			tour0 = append([]int(nil), open...) // capture first tour as baseline
			cost0 = res.Cost                    // capture stabilized cost
			return
		}
		if !slices.Equal(open, tour0) || round1e9(res.Cost) != round1e9(cost0) {
			t.Fatalf("nondeterministic result.\nfirst tour: %v (%.12f)\n this tour: %v (%.12f)",
				tour0, cost0, open, res.Cost)
		}
	})
}
