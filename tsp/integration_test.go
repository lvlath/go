// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp_test provides end-to-end (integration) checks for the public API.
// Goals:
//  1. SolveWithMatrix (Auto path) returns a valid Hamiltonian cycle with sane cost.
//  2. The exact BnB solution is never worse than the Auto pipeline on symmetric TSP.
//  3. On ATSP, the Auto pipeline returns a valid tour with a positive finite cost.
package tsp_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/tsp"
)

// TestIntegration_AutoVsBnB_Symmetric validates that the high-level pipeline
// produces a valid solution and that the exact BnB solver is never worse.
// We also compare both solutions to a trivial perimeter upper bound for sanity.
func TestIntegration_AutoVsBnB_Symmetric(t *testing.T) {
	// Use a modest convex hexagon: small, deterministic, and non-trivial.
	const n = 6
	pts := [][2]float64{
		{1, 0}, {0.5, math.Sqrt(3) / 2}, {-0.5, math.Sqrt(3) / 2},
		{-1, 0}, {-0.5, -math.Sqrt(3) / 2}, {0.5, -math.Sqrt(3) / 2},
	}
	m := euclid(pts) // symmetric Euclidean metric from shared test utils

	// Build a trivial perimeter (closed) tour to compute an easy upper bound.
	perim := []int{0, 1, 2, 3, 4, 5, 0}
	perimCost, err := tsp.TourCost(m, perim)
	if err != nil {
		t.Fatalf("TourCost(perimeter) failed: %v", err)
	}

	// ---- Auto pipeline via SolveWithMatrix (integration target).
	optAuto := tsp.DefaultOptions()
	optAuto.Symmetric = true     // symmetric TSP
	optAuto.StartVertex = startV // canonical start
	optAuto.Eps = epsTiny        // strict acceptance
	optAuto.EnableLocalSearch = true
	// Intentionally leave BoundAlgo/Algo as defaults to exercise the dispatcher.

	resAuto, err := tsp.SolveMatrix(m, nil, optAuto)
	if err != nil {
		t.Fatalf("SolveWithMatrix (Auto) failed: %v", err)
	}
	if err = tsp.ValidateTour(resAuto.Tour, n, startV); err != nil {
		t.Fatalf("Auto: returned tour invalid: %v", err)
	}
	autoCost, err := tsp.TourCost(m, resAuto.Tour)
	if err != nil {
		t.Fatalf("Auto: TourCost failed: %v", err)
	}

	// ---- Exact solution via Branch-and-Bound.
	optBB := tsp.DefaultOptions()
	optBB.Symmetric = true
	optBB.StartVertex = startV
	optBB.Eps = epsTiny
	optBB.BoundAlgo = tsp.SimpleBound // deterministic admissible LB
	optBB.EnableLocalSearch = false   // avoid extra work; correctness unaffected

	resBB, err := tsp.BranchAndBoundSolve(m, optBB)
	if err != nil {
		t.Fatalf("BranchAndBoundSolve failed: %v", err)
	}
	if err = tsp.ValidateTour(resBB.Tour, n, startV); err != nil {
		t.Fatalf("BnB: returned tour invalid: %v", err)
	}
	// Costs sanity versus perimeter (stabilized).
	if round1e9(resBB.Cost) > round1e9(perimCost) {
		t.Fatalf("BnB cost above perimeter: bnb=%.12f perim=%.12f", resBB.Cost, perimCost)
	}
	if round1e9(autoCost) > round1e9(perimCost) {
		t.Fatalf("Auto cost above perimeter: auto=%.12f perim=%.12f", autoCost, perimCost)
	}
	// Optimality relation: BnB must be <= Auto (after stable rounding).
	if round1e9(resBB.Cost) > round1e9(autoCost) {
		t.Fatalf("BnB cost worse than Auto: bnb=%.12f auto=%.12f", resBB.Cost, autoCost)
	}
}

// TestIntegration_Auto_ATSP validates that the Auto pipeline returns a valid
// Hamiltonian cycle on ATSP with a positive finite cost. We do not compare to
// BnB (ATSP exact is not exposed here); this is a pipeline sanity check.
func TestIntegration_Auto_ATSP(t *testing.T) {
	// Seven points on a circle; add a directional bias to break symmetry.
	const n = 7
	pts := make([][2]float64, n)
	var i int
	var th float64
	for i = 0; i < n; i++ {
		th = 2 * math.Pi * float64(i) / float64(n)
		pts[i] = [2]float64{math.Cos(th), math.Sin(th)}
	}
	m := euclidAsym(pts, 0.15) // asymmetric Euclidean-like metric

	opt := tsp.DefaultOptions()
	opt.Symmetric = false // ATSP
	opt.StartVertex = startV
	opt.Eps = epsTiny
	opt.EnableLocalSearch = true
	opt.Algo = tsp.TwoOptOnly

	res, err := tsp.SolveMatrix(m, nil, opt)
	if err != nil {
		t.Fatalf("SolveWithMatrix (ATSP) failed: %v", err)
	}
	if err = tsp.ValidateTour(res.Tour, n, startV); err != nil {
		t.Fatalf("ATSP: returned tour invalid: %v", err)
	}
	c, err := tsp.TourCost(m, res.Tour)
	if err != nil {
		t.Fatalf("ATSP: TourCost failed: %v", err)
	}
	if !(c > 0) || math.IsInf(c, 0) || math.IsNaN(c) {
		t.Fatalf("ATSP: unexpected cost: %.12f", c)
	}
}

func TestIntegration_BranchAndBound_OneTree_NotWorse_Than_Simple(t *testing.T) {
	// Modest instance where BnB finishes quickly but pruning differs meaningfully.
	const n = 8
	pts := make([][2]float64, n)
	var i int
	var th float64
	for i = 0; i < n; i++ {
		th = 2 * math.Pi * float64(i) / float64(n)
		r := 1.0 + 0.04*math.Cos(3*th)
		pts[i] = [2]float64{r * math.Cos(th), r * math.Sin(th)}
	}
	m := euclid(pts)

	base := tsp.DefaultOptions()
	base.Algo = tsp.BranchAndBound
	base.Symmetric = true
	base.StartVertex = startV
	base.Eps = epsTiny
	base.EnableLocalSearch = false

	// Simple bound.
	simple := base
	simple.BoundAlgo = tsp.SimpleBound
	rS, err := tsp.SolveMatrix(m, nil, simple)
	if err != nil {
		t.Fatalf("BnB SimpleBound failed: %v", err)
	}

	// OneTree bound.
	one := base
	one.BoundAlgo = tsp.OneTreeBound
	rO, err := tsp.SolveMatrix(m, nil, one)
	if err != nil {
		t.Fatalf("BnB OneTreeBound failed: %v", err)
	}

	// OneTree is a stronger LB; cost must be identical (both optimal),
	// though runtime/pruning differs. We assert non-worsening (≤) to be robust.
	if round1e9(rO.Cost) > round1e9(rS.Cost) {
		t.Fatalf("OneTreeBound produced worse cost: one=%.12f simple=%.12f", rO.Cost, rS.Cost)
	}
}
