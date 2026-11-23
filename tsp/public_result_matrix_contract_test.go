// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp_test verifies public result ownership, routing dispatch and numeric guards.
// No private kernel is exported for the sake of testing implementation details.
package tsp_test

import (
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/tsp"
)

// TestTSPResult_NilCloneAndIDProjection checks nilable and detached result APIs.
// VertexTour must preserve the exact tour order and report absent/bad mappings.
func TestTSPResult_NilCloneAndIDProjection(t *testing.T) {
	var missing *tsp.Result
	if !missing.IsNil() || missing.Clone() != nil {
		t.Fatal("nil result must satisfy Nilable and clone to nil")
	}
	if _, err := missing.VertexTour(); !errors.Is(err, tsp.ErrNilResult) {
		t.Fatalf("nil VertexTour: %v", err)
	}
	original := &tsp.Result{Tour: []int{1, 2, 0, 1}, IDs: []string{"hub", "lab", "ward"}, Cost: 17}
	if original.IsNil() {
		t.Fatal("non-nil result rejected")
	}
	labels, err := original.VertexTour()
	if err != nil || !slices.Equal(labels, []string{"lab", "ward", "hub", "lab"}) {
		t.Fatalf("projection=%v err=%v", labels, err)
	}
	clone := original.Clone()
	if clone == nil || clone == original || !slices.Equal(clone.Tour, original.Tour) || !slices.Equal(clone.IDs, original.IDs) {
		t.Fatalf("bad clone: %+v", clone)
	}
	clone.Tour[0], clone.IDs[0] = 100, "changed"
	if original.Tour[0] != 1 || original.IDs[0] != "hub" {
		t.Fatal("result retained aliases to clone")
	}
	labels[0] = "changed"
	if original.IDs[1] != "lab" {
		t.Fatal("VertexTour retained alias to IDs")
	}
	for _, tc := range []struct {
		result *tsp.Result
		want   error
	}{
		{&tsp.Result{Tour: []int{0}}, tsp.ErrMissingVertexIDs},
		{&tsp.Result{IDs: []string{"a"}}, tsp.ErrInvalidTour},
		{&tsp.Result{Tour: []int{-1}, IDs: []string{"a"}}, tsp.ErrInvalidTour},
		{&tsp.Result{Tour: []int{1}, IDs: []string{"a"}}, tsp.ErrInvalidTour},
	} {
		got, err := tc.result.VertexTour()
		if got != nil || !errors.Is(err, tc.want) {
			t.Fatalf("VertexTour(%+v)=%v, %v; want %v", tc.result, got, err, tc.want)
		}
	}
}

// TestTSPPublic_DispatchAndIDValidation exercises actual public consumer paths.
// Auto selection must be explicit and metadata must reflect the selected kernel.
func TestTSPPublic_DispatchAndIDValidation(t *testing.T) {
	distance := coverageLineMatrix(t, 5)
	ids := []string{"dispatch", "clinic", "pharmacy", "warehouse", "station"}
	for _, tc := range []struct {
		name  string
		algo  tsp.Algorithm
		exact bool
	}{
		{"christofides", tsp.Christofides, false},
		{"held_karp", tsp.ExactHeldKarp, true},
		{"two_opt", tsp.TwoOptOnly, false},
		{"three_opt", tsp.ThreeOptOnly, false},
		{"branch_and_bound", tsp.BranchAndBound, true},
		{"auto_exact", tsp.Auto, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			opts := tsp.DefaultOptions()
			opts.Algo = tc.algo
			opts.EnableLocalSearch = false
			result, err := tsp.SolveMatrix(distance, ids, opts)
			if err != nil || result == nil {
				t.Fatalf("SolveMatrix %v: result=%+v err=%v", tc.algo, result, err)
			}
			if err := tsp.ValidateTour(result.Tour, 5, 0); err != nil {
				t.Fatalf("invalid tour: %v", err)
			}
			if result.Algorithm == tsp.Auto || result.Cost != 8 {
				t.Fatalf("incorrect dispatch or cost: %+v", result)
			}
			if tc.exact && (!result.Exact || !result.Optimal) {
				t.Fatalf("expected exact result: %+v", result)
			}
			projected, err := result.VertexTour()
			if err != nil || len(projected) != 6 || projected[0] != "dispatch" {
				t.Fatalf("bad IDs projection: %v, %v", projected, err)
			}
		})
	}
	// Explicit Auto must select Christofides instead of a hidden Held-Karp fallback.
	opts := tsp.DefaultOptions()
	opts.Algo = tsp.Auto
	opts.MaxExactN = 3
	opts.EnableLocalSearch = false
	result, err := tsp.SolveMatrix(distance, nil, opts)
	if err != nil || result == nil || result.Algorithm != tsp.Christofides {
		t.Fatalf("Auto large symmetric: %+v %v", result, err)
	}
	for _, tc := range []struct {
		ids  []string
		want error
	}{
		{[]string{"only_one"}, tsp.ErrDimensionMismatch},
		{[]string{"a", "b", "", "d", "e"}, tsp.ErrInvalidIDs},
		{[]string{"a", "b", "a", "d", "e"}, tsp.ErrInvalidIDs},
	} {
		out, err := tsp.SolveMatrix(distance, tc.ids, tsp.DefaultOptions())
		if out != nil || !errors.Is(err, tc.want) {
			t.Fatalf("bad IDs %v: result=%+v err=%v want=%v", tc.ids, out, err, tc.want)
		}
	}
}

// TestTSPPublic_DegenerateAndPreflightErrors covers the one-vertex specialization.
// Empty, nil, invalid diagonal and invalid starts must never become successful tours.
func TestTSPPublic_DegenerateAndPreflightErrors(t *testing.T) {
	opts := tsp.DefaultOptions()
	for _, tc := range []struct {
		name string
		mat  matrix.Matrix
		want error
	}{
		{"nil", nil, tsp.ErrNilDistanceMatrix},
		{"empty", testDense{a: [][]float64{}}, tsp.ErrDimensionMismatch},
		{"non_square", testDense{a: [][]float64{{0, 1}}}, tsp.ErrNonSquare},
		{"nonzero_diag", testDense{a: [][]float64{{5}}}, tsp.ErrNonZeroDiagonal},
		{"nan_diag", testDense{a: [][]float64{{math.NaN()}}}, tsp.ErrNaNInf},
		{"inf_diag", testDense{a: [][]float64{{math.Inf(1)}}}, tsp.ErrNaNInf},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tsp.SolveMatrix(tc.mat, nil, opts)
			if got != nil || !errors.Is(err, tc.want) {
				t.Fatalf("SolveMatrix(%s)=%+v, %v want %v", tc.name, got, err, tc.want)
			}
		})
	}
	single := coverageLineMatrix(t, 1)
	for _, algo := range []tsp.Algorithm{tsp.Auto, tsp.Christofides, tsp.ExactHeldKarp, tsp.TwoOptOnly} {
		opts.Algo = algo
		got, err := tsp.SolveMatrix(single, []string{"solo"}, opts)
		if err != nil || got == nil || !slices.Equal(got.Tour, []int{0, 0}) || !got.Exact || !got.Optimal {
			t.Fatalf("one-vertex %v: result=%+v err=%v", algo, got, err)
		}
	}
	opts.StartVertex = 1
	out, err := tsp.SolveMatrix(single, nil, opts)
	if out != nil || !errors.Is(err, tsp.ErrStartOutOfRange) {
		t.Fatalf("single invalid start: %+v %v", out, err)
	}
	opts.StartVertex = 0
	out, err = tsp.SolveMatrix(single, []string{""}, opts)
	if out != nil || !errors.Is(err, tsp.ErrInvalidIDs) {
		t.Fatalf("single invalid ID: %+v %v", out, err)
	}
	out, err = tsp.SolveGraph(nil, opts)
	if out != nil || !errors.Is(err, tsp.ErrNilGraph) {
		t.Fatalf("nil graph: %+v %v", out, err)
	}
}

// TestTSPPublic_PermutationAndTourCostBoundary checks exact cycle construction.
// Large representable costs must survive display rounding without creating infinity.
func TestTSPPublic_PermutationAndTourCostBoundary(t *testing.T) {
	original := []int{2, 0, 3, 1}
	tour, err := tsp.MakeTourFromPermutation(original, 4, 3)
	if err != nil || !slices.Equal(tour, []int{3, 1, 2, 0, 3}) {
		t.Fatalf("MakeTourFromPermutation: %v %v", tour, err)
	}
	if !slices.Equal(original, []int{2, 0, 3, 1}) {
		t.Fatalf("input permutation was mutated: %v", original)
	}
	for _, tc := range []struct {
		perm  []int
		n     int
		start int
		want  error
	}{
		{nil, 4, 0, tsp.ErrDimensionMismatch},
		{[]int{0, 0, 1}, 3, 0, tsp.ErrDimensionMismatch},
		{[]int{0, 1, 2}, 3, 3, tsp.ErrStartOutOfRange},
	} {
		out, err := tsp.MakeTourFromPermutation(tc.perm, tc.n, tc.start)
		if out != nil || !errors.Is(err, tc.want) {
			t.Fatalf("bad perm %+v, %v: %+v %v", tc.perm, tc.start, out, err)
		}
	}
	distance := newTSPFractionalMatrix(t, 1e300)
	cost, err := tsp.TourCost(distance, []int{0, 1, 0})
	if err != nil || math.IsInf(cost, 0) || cost != 2e300 {
		t.Fatalf("finite large tour cost=%v err=%v", cost, err)
	}
	small := newTSPFractionalMatrix(t, 4e-10)
	cost, err = tsp.TourCost(small, []int{0, 1, 0})
	if err != nil || cost != 1e-9 { // legacy round1e9 of 8e-10
		t.Fatalf("legacy rounded cost=%v err=%v", cost, err)
	}
}

// newTSPFractionalMatrix constructs an exact symmetric two-vertex matrix.
// This helper uses the public Dense API and never accepts an ignored setup error.
func newTSPFractionalMatrix(t *testing.T, weight float64) *matrix.Dense {
	t.Helper()
	m, err := matrix.NewDense(2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if err = m.Set(0, 1, weight); err != nil {
		t.Fatal(err)
	}
	if err = m.Set(1, 0, weight); err != nil {
		t.Fatal(err)
	}
	return m
}
