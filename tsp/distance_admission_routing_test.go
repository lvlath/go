// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp_test protects graph-distance preparation and asymmetric local search.
// Every assertion uses public facades and real original-distance matrices.
package tsp_test

import (
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/tsp"
)

// TestTSPMetricClosure_MissingLinkRepair checks explicit shortest-path completion.
// Closing a length-4 chain yields a line metric and a closed tour of length 18.
func TestTSPMetricClosure_MissingLinkRepair(t *testing.T) {
	inf := math.Inf(1)
	rows := [][]float64{
		{0, 2, inf, inf},
		{2, 0, 3, inf},
		{inf, 3, 0, 4},
		{inf, inf, 4, 0},
	}
	source := testDense{a: rows}
	opts := tsp.DefaultOptions()
	opts.EnableLocalSearch = false
	direct, err := tsp.SolveMatrix(source, nil, opts)
	if direct != nil || !errors.Is(err, tsp.ErrIncompleteGraph) {
		t.Fatalf("unclosed incomplete matrix: %+v, %v", direct, err)
	}
	opts.RunMetricClosure = true
	ids := []string{"hub", "clinic", "relay", "depot"}
	result, err := tsp.SolveMatrix(source, ids, opts)
	if err != nil || result == nil {
		t.Fatalf("closed SolveMatrix: %+v, %v", result, err)
	}
	if result.Cost != 18 || !result.MetricClosureApplied || len(result.Tour) != 5 ||
		!slices.Equal(result.IDs, ids) {
		t.Fatalf("unexpected closed distance tour: %+v", result)
	}
	if !math.IsInf(rows[0][3], 1) || !math.IsInf(rows[3][0], 1) {
		t.Fatalf("closure unexpectedly mutated caller matrix: %+v", rows)
	}
}

// TestTSPATSP_SelectionAndLocalImprovements checks valid directed local improvements.
// Directed 2-opt accounts for reversed internal arcs; restricted 3-opt uses a legal tail swap.
func TestTSPATSP_SelectionAndLocalImprovements(t *testing.T) {
	// Starting cycle is 0-1-2-3-4-5-0. Swapping [1,2] and [3,4]
	// creates the cost-6 directed cycle 0-3-4-1-2-5-0.
	const n = 6
	rows := make([][]float64, n)
	for i := range rows {
		rows[i] = make([]float64, n)
		for j := range rows[i] {
			if i != j {
				rows[i][j] = 100
			}
		}
	}
	for _, arc := range [][2]int{{0, 3}, {3, 4}, {4, 1}, {1, 2}, {2, 5}, {5, 0}} {
		rows[arc[0]][arc[1]] = 1
	}
	source := testDense{a: rows}
	base := []int{0, 1, 2, 3, 4, 5, 0}
	costBefore, err := tsp.TourCost(source, base)
	if err != nil || costBefore <= 6 {
		t.Fatalf("invalid bad baseline: %v, %v", costBefore, err)
	}
	opts := tsp.DefaultOptions()
	opts.Symmetric = false
	opts.EnableLocalSearch = false
	opts.ShuffleNeighborhood = false
	for _, tc := range []struct {
		name string
		run  func(matrix.Matrix, []int, tsp.Options) (*tsp.Result, error)
	}{
		{"two_opt_directed", tsp.TwoOptSearch},
		{"three_opt_tail_swap", tsp.ThreeOptSearch},
	} {
		t.Run(tc.name, func(t *testing.T) {
			result, err := tc.run(source, base, opts)
			if err != nil || result == nil {
				t.Fatalf("ATSP improvement: %+v, %v", result, err)
			}
			if err := tsp.ValidateTour(result.Tour, n, 0); err != nil {
				t.Fatalf("invalid improved ATSP tour: %v", err)
			}
			measured, err := tsp.TourCost(source, result.Tour)
			if err != nil || measured > costBefore || measured != result.Cost {
				t.Fatalf("invalid ATSP improvement: before=%v result=%+v measured=%v err=%v", costBefore, result, measured, err)
			}
			if result.Exact || result.Optimal {
				t.Fatal("ATSP local search must not assert global optimality")
			}
		})
	}
	// Auto on asymmetric n=6 with explicit MaxExactN=4 selects 2-opt,
	// never Christofides and never an implicit exact fallback.
	opts.Algo = tsp.Auto
	opts.MaxExactN = 4
	result, err := tsp.SolveMatrix(source, nil, opts)
	if err != nil || result == nil || result.Algorithm != tsp.TwoOptOnly || result.Symmetric {
		t.Fatalf("ATSP Auto: %+v, %v", result, err)
	}
	if !slices.Equal(base, []int{0, 1, 2, 3, 4, 5, 0}) {
		t.Fatalf("local search modified caller tour: %v", base)
	}
}

// TestTSPValidation_WeightDomainAndMSTErrors checks exact domain classification.
// MST and solver admission must reject shape, missing edges and bad singleton data.
func TestTSPValidation_WeightDomainAndMSTErrors(t *testing.T) {
	for _, tc := range []struct {
		name string
		data matrix.Matrix
		want error
	}{
		{"nil", nil, tsp.ErrNilDistanceMatrix},
		{"non_square", testDense{a: [][]float64{{0, 1}}}, tsp.ErrNonSquare},
		{"empty", testDense{a: [][]float64{}}, tsp.ErrDimensionMismatch},
		{"singleton_nonzero", testDense{a: [][]float64{{0.5}}}, tsp.ErrNonZeroDiagonal},
		{"singleton_nan", testDense{a: [][]float64{{math.NaN()}}}, tsp.ErrNaNInf},
		{"singleton_inf", testDense{a: [][]float64{{math.Inf(1)}}}, tsp.ErrNonZeroDiagonal},
		{"negative", testDense{a: [][]float64{{0, -1}, {-1, 0}}}, tsp.ErrNegativeWeight},
		{"asymmetric", testDense{a: [][]float64{{0, 1}, {2, 0}}}, tsp.ErrAsymmetry},
		{"missing_edge", testDense{a: [][]float64{{0, math.Inf(1)}, {math.Inf(1), 0}}}, tsp.ErrIncompleteGraph},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cost, adjacency, err := tsp.MinimumSpanningTree(tc.data)
			if err == nil || !errors.Is(err, tc.want) || cost != 0 || adjacency != nil {
				t.Fatalf("MST invalid input %s: cost=%v adj=%v err=%v want=%v", tc.name, cost, adjacency, err, tc.want)
			}
		})
	}
}

// TestTSPATSP_TwoOptReversalAccountsForInternalArcs prevents a disconnected-tail bug.
// Reversing [1,2] converts three expensive arcs to a cost-six directed cycle;
// a two-cut successor swap would instead create two disjoint subtours.
func TestTSPATSP_TwoOptReversalAccountsForInternalArcs(t *testing.T) {
	const n = 6
	rows := make([][]float64, n)
	for i := range rows {
		rows[i] = make([]float64, n)
		for j := range rows[i] {
			if i != j {
				rows[i][j] = 100
			}
		}
	}
	for _, edge := range [][2]int{{0, 2}, {2, 1}, {1, 3}, {3, 4}, {4, 5}, {5, 0}} {
		rows[edge[0]][edge[1]] = 1
	}
	source := testDense{a: rows}
	original := []int{0, 1, 2, 3, 4, 5, 0}
	options := tsp.DefaultOptions()
	options.Symmetric = false
	result, err := tsp.TwoOptSearch(source, original, options)
	if err != nil || result == nil {
		t.Fatalf("directed 2-opt: %+v %v", result, err)
	}
	if !slices.Equal(result.Tour, []int{0, 2, 1, 3, 4, 5, 0}) || result.Cost != 6 {
		t.Fatalf("wrong directed reversal or cost: %+v", result)
	}
	if err := tsp.ValidateTour(result.Tour, n, 0); err != nil {
		t.Fatal(err)
	}
	actualCost, err := tsp.TourCost(source, result.Tour)
	if err != nil || actualCost != result.Cost {
		t.Fatalf("directed cost mismatch: %v %v", actualCost, err)
	}
	if !slices.Equal(original, []int{0, 1, 2, 3, 4, 5, 0}) {
		t.Fatalf("source tour changed: %v", original)
	}
}
