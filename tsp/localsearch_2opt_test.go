// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp_test exercises the 2-opt local search via the public API.
// Focus: determinism, epsilon semantics, correctness on symmetric/ATSP cases,
// and safe handling of +Inf candidates without touching internals.
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

// -----------------------------------------------------------------------------
// Helpers (minimal, stdlib-only)
// -----------------------------------------------------------------------------

// run2opt configures Options for TwoOptOnly and executes SolveWithMatrix.
// Returns the final solver result (TSResult).
func run2opt(
	dist matrix.Matrix,
	eps float64,
	symmetric bool,
	seed int64,
	start int,
	timeLimit time.Duration,
) (*tsp.Result, error) {
	opts := tsp.DefaultOptions()
	opts.Algo = tsp.TwoOptOnly
	opts.Symmetric = symmetric
	opts.Eps = eps
	opts.Seed = seed
	opts.StartVertex = start
	opts.TimeLimit = timeLimit

	return tsp.SolveMatrix(dist, nil, opts)
}

// sameCycleEitherDir checks whether two tours represent the same cycle when both
// start at 0; reversal of orientation is allowed. Accepts open or closed input.
func sameCycleEitherDir(a, b []int) bool {
	a = normalizeOpenCycle(a)
	b = normalizeOpenCycle(b)

	if len(a) == 0 || len(a) != len(b) || a[0] != 0 || b[0] != 0 {
		return false
	}
	if slices.Equal(a, b) {
		return true
	}
	n := len(a)
	rev := make([]int, n)
	rev[0] = 0
	var i int
	for i = 1; i < n; i++ {
		rev[i] = a[n-i]
	}

	return slices.Equal(rev, b)
}

// -----------------------------------------------------------------------------
// 1) Medium - TSP: 2-opt removes crossings on a convex hexagon.
// -----------------------------------------------------------------------------

func TestTwoOpt_TSP_ImprovesConvexHexagon(t *testing.T) {
	const n = 6
	pts := [][2]float64{
		{1, 0}, {0.5, math.Sqrt(3) / 2}, {-0.5, math.Sqrt(3) / 2},
		{-1, 0}, {-0.5, -math.Sqrt(3) / 2}, {0.5, -math.Sqrt(3) / 2},
	}
	m := euclid(pts)
	want := []int{0, 1, 2, 3, 4, 5} // polygon boundary (either orientation)

	Repeat(t, 3, func(t *testing.T) {
		res, err := run2opt(m, epsTiny, true, seedDet, startV, 0)
		if err != nil {
			t.Fatalf("SolveWithMatrix(2-opt) error: %v", err)
		}
		if err = tsp.ValidateTour(res.Tour, n, 0); err != nil {
			t.Fatalf("returned tour invalid: %v", err)
		}
		rot := rotateToStart0(t, res.Tour) // normalize to open then check
		if !sameCycleEitherDir(rot, want) {
			t.Fatalf("unexpected tour:\n got:  %v\n want: %v (either direction, start=0)", rot, want)
		}
		mustValidResult(t, res, n, startV, tsp.TwoOptOnly)
	})
}

// -----------------------------------------------------------------------------
// 2) Validation - EPS monotonicity: high-eps cannot beat low-eps.
// -----------------------------------------------------------------------------

func TestTwoOpt_EpsMonotonicity(t *testing.T) {
	pts := [][2]float64{
		{0, 0}, {1, 0}, {2, 0.05}, {3, 0}, {4, 0}, // slight non-collinearity
	}
	m := euclid(pts)

	lo, err := run2opt(m, epsTiny, true, seedDet, startV, 0)
	if err != nil {
		t.Fatalf("low-eps run failed: %v", err)
	}
	hi, err := run2opt(m, 1e-1, true, seedDet, startV, 0) // large eps blocks tiny deltas
	if err != nil {
		t.Fatalf("high-eps run failed: %v", err)
	}

	if round1e9(hi.Cost) < round1e9(lo.Cost) {
		t.Fatalf("eps monotonicity violated: high-eps cost %.12f < low-eps cost %.12f", hi.Cost, lo.Cost)
	}
	if err = tsp.ValidateTour(lo.Tour, len(pts), 0); err != nil {
		t.Fatalf("low-eps tour invalid: %v", err)
	}
	if err = tsp.ValidateTour(hi.Tour, len(pts), 0); err != nil {
		t.Fatalf("high-eps tour invalid: %v", err)
	}
	_ = rotateToStart0(t, lo.Tour)
	_ = rotateToStart0(t, hi.Tour)
}

// -----------------------------------------------------------------------------
// 3) Validation - ATSP: 2-opt must return a valid order under asymmetry.
// -----------------------------------------------------------------------------

func TestTwoOpt_ATSP_BasicSuccessorOrder(t *testing.T) {
	pts := [][2]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}}
	m := euclidAsym(pts, 0.2)

	res, err := run2opt(m, epsTiny, false, seedDet, startV, 0)
	if err != nil {
		t.Fatalf("ATSP 2-opt failed: %v", err)
	}
	if err = tsp.ValidateTour(res.Tour, 4, 0); err != nil {
		t.Fatalf("ATSP tour invalid: %v", err)
	}
}

// -----------------------------------------------------------------------------
// 4) Validation - +Inf candidate edges must be rejected (no panics).
// If global validation rejects such a matrix up-front, that’s acceptable too.
// -----------------------------------------------------------------------------

func TestTwoOpt_RejectsInfCandidates_NoError(t *testing.T) {
	var I = math.Inf(1)

	a := [][]float64{
		{0, 1, 1.04, 9, 1},
		{1, 0, 1, 1.0, 9},
		{1.04, 1, 0, 1.05, 9},
		{9, 1.0, 1.05, 0, 1},
		{1, 9, 9, 1, 0},
	}
	// Block an improving move by making one of the new chords +Inf.
	a[0][2], a[2][0] = I, I
	m := testDense{a: a}

	res, err := run2opt(m, epsTiny, true, seedDet, startV, 0)
	if err != nil {
		// Some validators reject +Inf globally - also a correct outcome.
		if !errors.Is(err, tsp.ErrIncompleteGraph) && !errors.Is(err, tsp.ErrDimensionMismatch) {
			t.Fatalf("unexpected error: %v", err)
		}

		return
	}

	// If the instance passed validation, ensure no “improvement” happened via +Inf.
	after, err := tsp.TourCost(m, res.Tour)
	if err != nil {
		t.Fatalf("TourCost failed: %v", err)
	}
	if round1e9(after) != round1e9(res.Cost) {
		t.Fatalf("cost changed unexpectedly in presence of +Inf candidate: base=%.12f after=%.12f",
			res.Cost, after)
	}
}

// -----------------------------------------------------------------------------
// 5) Special - Determinism: 5 identical runs must produce identical tour/cost.
// -----------------------------------------------------------------------------

func TestTwoOpt_Determinism_Repeat5(t *testing.T) {
	pts := [][2]float64{
		{0, 0}, {1, 0}, {2, 0.05}, {3, 0}, {4, 0}, {5, 0.02},
	}
	m := euclid(pts)

	var tour0 []int
	var cost0 float64

	Repeat(t, 5, func(t *testing.T) {
		res, err := run2opt(m, epsTiny, true, seedDet, startV, 0)
		if err != nil {
			t.Fatalf("run failed: %v", err)
		}
		if tour0 == nil {
			// Capture the first successful result as the determinism baseline.
			tour0 = append([]int(nil), normalizeOpenCycle(res.Tour)...)
			cost0 = res.Cost

			return
		}
		if !slices.Equal(normalizeOpenCycle(res.Tour), tour0) || round1e9(res.Cost) != round1e9(cost0) {
			t.Fatalf("nondeterministic result.\nfirst tour: %v (%.12f)\n this tour: %v (%.12f)",
				tour0, cost0, res.Tour, res.Cost)
		}
	})
}

// -----------------------------------------------------------------------------
// 6) Special - Soft time budget: nil or ErrTimeLimit are both acceptable outcomes.
// No panics, no instability.
// -----------------------------------------------------------------------------

func TestTwoOpt_TimeLimit_SoftBudget(t *testing.T) {
	// 120 points on a unit circle - decent workload for neighborhood scanning.
	pts := make([][2]float64, radiusN120)
	var i int
	var theta float64
	for i = 0; i < radiusN120; i++ {
		theta = 2 * math.Pi * float64(i) / float64(radiusN120) // uniform angle on the circle
		pts[i] = [2]float64{math.Cos(theta), math.Sin(theta)}
	}
	m := euclid(pts)

	_, err := run2opt(m, epsTiny, true, seedDet, startV, timeTiny)
	if err != nil && !errors.Is(err, tsp.ErrTimeLimit) {
		t.Fatalf("unexpected error under tiny time budget: %v", err)
	}
}
