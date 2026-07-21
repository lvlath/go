// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp_test exercises the 3-opt local search via the public API.
// Focus: policy correctness (first vs best), improvement over 2-opt on TSP,
// validity on ATSP, rejection of +Inf candidates, shuffle determinism,
// and time-budget behavior - all with strict sentinel errors and stable rounding.
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
// Local helpers for this file (reusing shared helpers from other *_test.go)
// -----------------------------------------------------------------------------

// run3opt configures Options for ThreeOptOnly and executes SolveWithMatrix.
// It allows selecting policy (first vs best), shuffling, symmetry, eps, seed,
// start vertex and time budget; returns the final TSResult from the solver.
func run3opt(
	dist matrix.Matrix,
	bestImprovement bool,
	shuffleNeighborhood bool,
	symmetric bool,
	eps float64,
	seed int64,
	start int,
	timeLimit time.Duration,
) (*tsp.Result, error) {
	opts := tsp.DefaultOptions()
	opts.Algo = tsp.ThreeOptOnly
	opts.BestImprovement = bestImprovement
	opts.ShuffleNeighborhood = shuffleNeighborhood
	opts.Symmetric = symmetric
	opts.Eps = eps
	opts.Seed = seed
	opts.StartVertex = start
	opts.TimeLimit = timeLimit

	return tsp.SolveMatrix(dist, nil, opts)
}

// bestLEqFirst asserts that the "best-improvement" policy is never worse
// than "first-improvement" on the same instance (monotonic policy strength).
func bestLEqFirst(t *testing.T, m matrix.Matrix) {
	t.Helper()

	// Run First-Improvement policy.
	first, err := run3opt(m, false, false, true, epsTiny, seedDet, startV, 0)
	if err != nil {
		t.Fatalf("3-opt first-improvement run failed: %v", err)
	}

	// Run Best-Improvement policy.
	best, err := run3opt(m, true, false, true, epsTiny, seedDet, startV, 0)
	if err != nil {
		t.Fatalf("3-opt best-improvement run failed: %v", err)
	}

	// Best policy may match but must not be worse after stable rounding.
	if round1e9(best.Cost) > round1e9(first.Cost) {
		t.Fatalf("best-improvement produced worse cost: best=%.12f first=%.12f", best.Cost, first.Cost)
	}
}

// -----------------------------------------------------------------------------
// 1) Medium - TSP: 3-opt should be at least as good as 2-opt on a nontrivial set.
//    We generate 10 points on a slightly perturbed circle to create enough
//    opportunities where 3-opt can improve upon a 2-opt local optimum.
// -----------------------------------------------------------------------------

func TestThreeOpt_TSP_ImprovesOverTwoOpt(t *testing.T) {
	// Build a near-circular instance with small radial ripples (to create crossings).
	const n = 10
	pts := make([][2]float64, n)
	var i int
	var theta, r float64
	for i = 0; i < n; i++ {
		theta = 2 * math.Pi * float64(i) / float64(n)
		r = 1.0 + 0.03*math.Sin(3*theta) // small perturbation to avoid perfect symmetry
		pts[i] = [2]float64{r * math.Cos(theta), r * math.Sin(theta)}
	}
	m := euclid(pts) // reuse helper from two_opt_test.go

	// Baseline: 2-opt result.
	two, err := run2opt(m, epsTiny, true, seedDet, startV, 0)
	if err != nil {
		t.Fatalf("2-opt baseline failed: %v", err)
	}
	if err = tsp.ValidateTour(two.Tour, n, startV); err != nil {
		t.Fatalf("2-opt returned invalid tour: %v", err)
	}

	// ThreeOptOnly with best-improvement policy should not be worse than 2-opt.
	thr, err := run3opt(m, true, false, true, epsTiny, seedDet, startV, 0)
	if err != nil {
		t.Fatalf("3-opt run failed: %v", err)
	}
	if err = tsp.ValidateTour(thr.Tour, n, startV); err != nil {
		t.Fatalf("3-opt returned invalid tour: %v", err)
	}

	// Compare after stable rounding to avoid cross-platform noise.
	if round1e9(thr.Cost) > round1e9(two.Cost) {
		t.Fatalf("3-opt failed to improve or match 2-opt: 3-opt=%.12f  2-opt=%.12f", thr.Cost, two.Cost)
	}
}

// -----------------------------------------------------------------------------
// 2) Validation - Policy: best-improvement must not perform worse than first.
//    This is a direct policy check on the same TSP instance.
// -----------------------------------------------------------------------------

func TestThreeOpt_Policy_BestVsFirst(t *testing.T) {
	// Mildly irregular octagon - enough structure for multiple 3-opt choices.
	const n = 8
	pts := make([][2]float64, n)
	var i int
	var theta, r float64
	for i = 0; i < n; i++ {
		theta = 2 * math.Pi * float64(i) / float64(n)
		r = 1.0 + 0.05*math.Cos(2*theta) // alternating “squash” to create multiple Δ options
		pts[i] = [2]float64{r * math.Cos(theta), r * math.Sin(theta)}
	}
	m := euclid(pts)

	// Assert “best ≤ first” under stable rounding.
	bestLEqFirst(t, m)
}

// -----------------------------------------------------------------------------
// 3) Validation - ATSP: 3-opt must return a valid permutation on asymmetric
//    distances; we also check it is not worse than 2-opt under the same config.
// -----------------------------------------------------------------------------

func TestThreeOpt_ATSP_Basic(t *testing.T) {
	// A square with directional penalty - simple ATSP fixture.
	pts := [][2]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}}
	m := euclidAsym(pts, 0.2) // reuse helper from two_opt_test.go

	two, twoErr := run2opt(m, epsTiny, false, seedDet, startV, 0)

	thr, err := run3opt(m, true, false, false, epsTiny, seedDet, startV, 0)
	mustNoError(t, err)
	mustValidResult(t, thr, 4, startV, tsp.ThreeOptOnly)

	if twoErr == nil {
		mustValidResult(t, two, 4, startV, tsp.TwoOptOnly)
		if thr.Cost > two.Cost+epsTiny {
			t.Fatalf("ATSP: 3-opt worse than 2-opt: three=%.12f two=%.12f", thr.Cost, two.Cost)
		}
	} else if !errors.Is(twoErr, tsp.ErrATSPNotSupportedByAlgo) {
		t.Fatalf("unexpected 2-opt ATSP error: %v", twoErr)
	}
}

// -----------------------------------------------------------------------------
// 4) Validation - +Inf candidates must be rejected without panics.
//    Construct a 5-node instance where a tempting 3-edge reconnection would
//    require a +Inf chord; solver must avoid that path or reject early.
// -----------------------------------------------------------------------------

func TestThreeOpt_RejectsInfCandidates_NoError(t *testing.T) {
	// Symmetric weights; make one of the potential new chords +Inf.
	var I = math.Inf(1)
	a := [][]float64{
		{0, 1, 1.04, 9, 1},
		{1, 0, 1, 1.0, 9},
		{1.04, 1, 0, 1.05, 9},
		{9, 1.0, 1.05, 0, 1},
		{1, 9, 9, 1, 0},
	}
	// Block a candidate chord that a 3-opt reconnection might want to use.
	a[0][2], a[2][0] = I, I
	m := testDense{a: a}

	res, err := run3opt(m, true, false, true, epsTiny, seedDet, startV, 0)
	if err != nil {
		// If global validation bans +Inf upfront - that is acceptable; assert sentinel clarity.
		if !errors.Is(err, tsp.ErrIncompleteGraph) && !errors.Is(err, tsp.ErrDimensionMismatch) {
			t.Fatalf("unexpected error for +Inf candidate: %v", err)
		}

		return
	}

	// If it passed validation, ensure no “improvement via +Inf” slipped through.
	after, err := tsp.TourCost(m, res.Tour)
	if err != nil {
		t.Fatalf("TourCost failed: %v", err)
	}
	if round1e9(after) != round1e9(res.Cost) {
		t.Fatalf("cost changed unexpectedly with +Inf candidate present: before=%.12f after=%.12f",
			res.Cost, after)
	}
}

// -----------------------------------------------------------------------------
// 5) Special - Shuffle determinism: with Seed=0, enabling neighborhood shuffle
//    must not change the final tour/cost (order of scanning differs, result same).
// -----------------------------------------------------------------------------

func TestThreeOpt_ShuffleNeighborhood_Determinism(t *testing.T) {
	// Use a modest-size circle to give the neighborhood some breadth.
	const n = 16
	pts := make([][2]float64, n)
	var i int
	var theta float64
	for i = 0; i < n; i++ {
		theta = 2 * math.Pi * float64(i) / float64(n)
		pts[i] = [2]float64{math.Cos(theta), math.Sin(theta)}
	}
	m := euclid(pts)

	// No shuffle.
	noShuffle, err := run3opt(m, true, false, true, epsTiny, seedDet, startV, 0)
	if err != nil {
		t.Fatalf("3-opt(no-shuffle) failed: %v", err)
	}

	// Shuffle enabled with the same seed.
	shuf, err := run3opt(m, true, true, true, epsTiny, seedDet, startV, 0)
	if err != nil {
		t.Fatalf("3-opt(shuffle) failed: %v", err)
	}

	// Compare normalized open tours and rounded costs.
	if !slices.Equal(normalizeOpenCycle(noShuffle.Tour), normalizeOpenCycle(shuf.Tour)) ||
		round1e9(noShuffle.Cost) != round1e9(shuf.Cost) {
		t.Fatalf("shuffle changed the final result.\nno-shuffle: %v (%.12f)\n shuffle:  %v (%.12f)",
			noShuffle.Tour, noShuffle.Cost, shuf.Tour, shuf.Cost)
	}
}

// -----------------------------------------------------------------------------
// 6) Special - Time budget: with best-improvement policy on a medium instance
//    and a tiny budget, either ErrTimeLimit is returned or the run exits
//    cleanly (soft budget). In both cases: no panics, no instability.
// -----------------------------------------------------------------------------

func TestThreeOpt_TimeLimit_TinyBudget(t *testing.T) {
	// Circle with many vertices → sizable 3-opt neighborhood to scan.
	pts := make([][2]float64, radiusN120)
	var i int
	var theta float64
	for i = 0; i < radiusN120; i++ {
		theta = 2 * math.Pi * float64(i) / float64(radiusN120)
		pts[i] = [2]float64{math.Cos(theta), math.Sin(theta)}
	}
	m := euclid(pts)

	_, err := run3opt(m, true, false, true, epsTiny, seedDet, startV, timeTiny)
	if err != nil && !errors.Is(err, tsp.ErrTimeLimit) {
		t.Fatalf("unexpected error under tiny time budget: %v", err)
	}
}

// -----------------------------------------------------------------------------
// 7) Validation - Direct API guardrail: invalid base tour should be rejected.
//    Here we call tsp.ThreeOpt directly with an out-of-range index in base.
//    The function must surface ErrDimensionMismatch (strict sentinel).
// -----------------------------------------------------------------------------

func TestThreeOpt_InvalidBaseTour_StrictSentinel(t *testing.T) {
	// Small symmetric matrix (4x4) with valid weights.
	a := [][]float64{
		{0, 1, 2, 3},
		{1, 0, 2, 3},
		{2, 2, 0, 1},
		{3, 3, 1, 0},
	}
	m := testDense{a: a}

	// Base tour contains an out-of-range vertex 99 (n=4 → valid indices 0..3).
	base := []int{0, 1, 2, 99, 0}

	// Options for a direct ThreeOpt call (symmetric TSP).
	opt := tsp.DefaultOptions()
	opt.Algo = tsp.ThreeOptOnly
	opt.Symmetric = true
	opt.Eps = epsTiny
	opt.Seed = seedDet
	opt.StartVertex = startV

	// Call the algorithm directly; must error with a strict sentinel.
	_, err := tsp.ThreeOptSearch(m, base, opt)
	if !errors.Is(err, tsp.ErrInvalidTour) {
		t.Fatalf("want ErrInvalidTour on invalid base, got %v", err)
	}
}
