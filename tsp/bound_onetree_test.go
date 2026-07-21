// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp_test validates the Held–Karp 1-tree lower bound routine.
// Focus areas:
//  1. Strict sentinels on invalid inputs (non-square, OOB root, NaN, negative, +Inf).
//  2. Structural invariants of the resulting 1-tree degrees.
//  3. Tightness on a triangle (bound == optimal tour).
//  4. Sanity on a Euclidean pentagon (bound ≤ a trivial feasible tour).
//  5. Simple root-selection sanity: min across roots is consistent.
package tsp_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/tsp"
)

// sumInts returns Σ a[i] (small helper to make assertions readable).
func sumInts(a []int) int {
	var s int // running sum
	var i int // loop iterator
	for i = 0; i < len(a); i++ {
		s += a[i] // accumulate degrees
	}

	return s
}

// mkTriangle builds a 3×3 symmetric metric with edges d01=1, d12=2, d20=3.
// The optimal Hamiltonian cycle cost is 1+2+3=6; the 1-tree matches the cycle.
func mkTriangle() matrix.Matrix {
	a := [][]float64{
		{0, 1, 3},
		{1, 0, 2},
		{3, 2, 0},
	}

	return testDense{a: a} // testDense is defined in tour_cost_utils_test.go
}

// mkBad clones a symmetric 3×3 baseline and replaces (i,j) with w (and symmetrically (j,i)).
func mkBad(i, j int, w float64) matrix.Matrix {
	base := [][]float64{
		{0, 1, 2},
		{1, 0, 3},
		{2, 3, 0},
	}
	base[i][j], base[j][i] = w, w // enforce symmetry for these probes

	return testDense{a: base}
}

// -----------------------------------------------------------------------------
// 1) Validation - strict sentinels on invalid inputs.
//    Covered cases: non-square, out-of-range root, NaN, negative, +Inf (disconnected).
//    Note: nil matrix is intentionally NOT tested here because calling
//    dist.Rows() on a nil interface would panic at this layer; nil is covered
//    by higher-level validation tests that guard earlier in the pipeline.
// -----------------------------------------------------------------------------

func TestOneTree_Errors_StrictSentinels(t *testing.T) {
	cfg := tsp.DefaultOneTreeConfig() // default deterministic subgradient knobs
	var err error                     // shared error variable

	// non-square → ErrNonSquare (reuse helper from types_validate_test.go)
	Repeat(t, 2, func(t *testing.T) {
		// Build a 2×3 matrix to trigger the shape error.
		m := mkNonSquare([][]float64{
			{0, 1, 2},
			{1, 0, 3},
		})
		_, _, err = tsp.OneTreeLowerBound(m, 0, true, cfg)
		if !errors.Is(err, tsp.ErrNonSquare) {
			t.Fatalf("want ErrNonSquare, got %v", err)
		}
	})

	// out-of-range root → sentinel from validateStartVertex
	// Some modules may surface a specific "start vertex out of range" sentinel,
	// others reuse ErrDimensionMismatch. Accept either strictly.
	Repeat(t, 2, func(t *testing.T) {
		m := mkTriangle() // n = 3
		_, _, err = tsp.OneTreeLowerBound(m, 9, true, cfg)
		if !errors.Is(err, tsp.ErrStartOutOfRange) {
			t.Fatalf("want ErrStartOutOfRange, got %v", err)
		}
	})

	// NaN entry → ErrDimensionMismatch (caught during dense prefetch)
	Repeat(t, 2, func(t *testing.T) {
		m := mkBad(0, 1, math.NaN())
		_, _, err = tsp.OneTreeLowerBound(m, 0, true, cfg)
		if !errors.Is(err, tsp.ErrNaNInf) {
			t.Fatalf("want ErrNaNInf, got %v", err)
		}
	})

	// negative entry → ErrNegativeWeight
	Repeat(t, 2, func(t *testing.T) {
		m := mkBad(0, 1, -1)
		_, _, err = tsp.OneTreeLowerBound(m, 0, true, cfg)
		if !errors.Is(err, tsp.ErrNegativeWeight) {
			t.Fatalf("want ErrNegativeWeight, got %v", err)
		}
	})

	// +Inf entry on (1,2) disconnects V\{root} in the MST stage → ErrIncompleteGraph
	Repeat(t, 2, func(t *testing.T) {
		m := mkBad(1, 2, math.Inf(1))
		_, _, err = tsp.OneTreeLowerBound(m, 0, true, cfg)
		if !errors.Is(err, tsp.ErrIncompleteGraph) {
			t.Fatalf("want ErrIncompleteGraph, got %v", err)
		}
	})
}

// -----------------------------------------------------------------------------
// 2) Medium - Triangle tightness and degree invariants.
//    For n=3, the 1-tree equals the tour for ANY multipliers π, therefore
//    L(π) == optimal tour cost (6). Also check degree structure.
// -----------------------------------------------------------------------------

func TestOneTree_Triangle_TightAndDegrees(t *testing.T) {
	const n = 3                       // number of vertices
	const root = 0                    // distinguished root
	cfg := tsp.DefaultOneTreeConfig() // subgradient defaults

	m := mkTriangle() // symmetric 3-node metric

	lb, deg, err := tsp.OneTreeLowerBound(m, root, true, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Tightness: bound equals the optimal triangle tour cost (1+2+3=6).
	const want = 6.0
	if round1e9(lb) != round1e9(want) {
		t.Fatalf("triangle bound mismatch: got=%.12f want=%.12f", lb, want)
	}

	// Degree invariants.
	if len(deg) != n {
		t.Fatalf("degree vector length mismatch: got=%d want=%d", len(deg), n)
	}
	if deg[root] != 2 {
		t.Fatalf("root degree mismatch: got=%d want=2", deg[root])
	}
	var i int // loop iterator
	for i = 0; i < n; i++ {
		if i == root {
			continue // skip the root for the ≥1 check
		}
		if deg[i] < 1 {
			t.Fatalf("non-root degree must be ≥1: deg[%d]=%d", i, deg[i])
		}
	}
	if sumInts(deg) != 2*n {
		t.Fatalf("sum of degrees mismatch: got=%d want=%d", sumInts(deg), 2*n)
	}
}

// -----------------------------------------------------------------------------
// 3) Medium - Euclidean pentagon sanity:
//    The lower bound must be positive and ≤ a trivial perimeter tour cost,
//    and degree invariants must hold.
// -----------------------------------------------------------------------------

func TestOneTree_Pentagon_SaneBoundAndDegrees(t *testing.T) {
	const n = 5                       // number of vertices
	cfg := tsp.DefaultOneTreeConfig() // subgradient defaults

	// Build a regular pentagon on the unit circle.
	pts := make([][2]float64, n)
	var i int // loop iterator
	var theta float64
	for i = 0; i < n; i++ {
		theta = 2 * math.Pi * float64(i) / float64(n) // uniform angle
		pts[i] = [2]float64{math.Cos(theta), math.Sin(theta)}
	}
	m := euclid(pts) // symmetric Euclidean metric (helper from two_opt_test.go)

	// Compute the perimeter tour cost as an easy feasible upper bound.
	perim := []int{0, 1, 2, 3, 4, 0} // closed cycle around the polygon
	perimCost, err := tsp.TourCost(m, perim)
	if err != nil {
		t.Fatalf("TourCost failed on perimeter: %v", err)
	}

	// 1-tree lower bound for root=0.
	lb, deg, err := tsp.OneTreeLowerBound(m, 0, true, cfg)
	if err != nil {
		t.Fatalf("OneTreeLowerBound failed: %v", err)
	}

	// Bound sanity: positive and ≤ a feasible tour.
	if !(lb > 0) {
		t.Fatalf("lower bound must be positive: %.12f", lb)
	}
	if round1e9(lb) > round1e9(perimCost) {
		t.Fatalf("lower bound exceeds a feasible tour: lb=%.12f perim=%.12f", lb, perimCost)
	}

	// Degree invariants as above.
	if len(deg) != n {
		t.Fatalf("degree vector length mismatch: got=%d want=%d", len(deg), n)
	}
	if deg[0] != 2 {
		t.Fatalf("root degree mismatch: got=%d want=2", deg[0])
	}
	var j int    // iterator
	var dsum int // Σ degrees
	for j = 0; j < n; j++ {
		if j != 0 && deg[j] < 1 {
			t.Fatalf("non-root degree must be ≥1: deg[%d]=%d", j, deg[j])
		}
		dsum += deg[j]
	}
	if dsum != 2*n {
		t.Fatalf("sum of degrees mismatch: got=%d want=%d", dsum, 2*n)
	}
}

// -----------------------------------------------------------------------------
// 4) Special - Root scan sanity:
//    Take the rounded min lower bound across all roots; no single-root recompute
//    should produce a strictly smaller rounded value.
// -----------------------------------------------------------------------------

func TestOneTree_MinAcrossRoots_NonIncreasing(t *testing.T) {
	const n = 8                       // instance size
	cfg := tsp.DefaultOneTreeConfig() // subgradient defaults

	// Slightly rippled circle to avoid perfect symmetry while staying metric.
	pts := make([][2]float64, n)
	var i int // iterator
	var th float64
	var r float64
	for i = 0; i < n; i++ {
		th = 2 * math.Pi * float64(i) / float64(n) // angle
		r = 1.0 + 0.04*math.Cos(3*th)              // gentle radial ripple
		pts[i] = [2]float64{r * math.Cos(th), r * math.Sin(th)}
	}
	m := euclid(pts)

	// Compute rounded min across all possible roots.
	var root int         // current root
	var haveMin bool     // whether minRounded has been initialized
	var minRounded int64 // min of round1e9(lb) across roots
	var rnd int64        // per-root rounded bound

	for root = 0; root < n; root++ {
		lb, deg, err := tsp.OneTreeLowerBound(m, root, true, cfg)
		if err != nil {
			t.Fatalf("OneTreeLowerBound failed for root=%d: %v", root, err)
		}
		// Degree invariants for each root.
		if len(deg) != n || deg[root] != 2 || sumInts(deg) != 2*n {
			t.Fatalf("degree invariants broken for root=%d: deg=%v", root, deg)
		}
		rnd = round1e9(lb) // stabilize per root
		if !haveMin || rnd < minRounded {
			minRounded, haveMin = rnd, true // record the minimum rounded bound
		}
	}

	// Verify no recomputed bound falls strictly below the previously recorded rounded min.
	for root = 0; root < n; root++ {
		lb, _, err := tsp.OneTreeLowerBound(m, root, true, cfg)
		if err != nil {
			t.Fatalf("repeat OneTreeLowerBound failed for root=%d: %v", root, err)
		}
		if round1e9(lb) < minRounded {
			t.Fatalf("found a rounded bound below the recorded min: root=%d lb=%.12f min=%.12f",
				root, lb, float64(minRounded)/1e9)
		}
	}
}
