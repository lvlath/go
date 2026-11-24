// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp_test checks local-search admission, move algebra and partial publication.
// Independent tour sums and a deterministic clock make the assertions reproducible.
package tsp_test

import (
	"fmt"
	"math"
	"slices"
	"testing"
	"time"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/tsp"
)

// TestLocalSearchAdmissionAndFinalization verifies that neither direct wrapper
// publishes a result on invalid input or an unrepresentable initial objective.
func TestLocalSearchAdmissionAndFinalization(t *testing.T) {
	valid := coverageMatrixFromRows(coverageUnitRows(4))
	base := []int{0, 1, 2, 3, 0}
	for _, solver := range []struct {
		name string
		run  func(matrix.Matrix, []int, tsp.Options) (*tsp.Result, error)
	}{{"two-opt", tsp.TwoOptSearch}, {"three-opt", tsp.ThreeOptSearch}} {
		t.Run(solver.name, func(t *testing.T) {
			for _, tc := range []struct {
				name       string
				dist       matrix.Matrix
				tour       []int
				badOptions bool
				want       error
			}{
				{"options", valid, base, true, tsp.ErrInvalidOptions},
				{"nil-tour", valid, nil, false, tsp.ErrNilTour},
				{"empty-tour", valid, []int{}, false, tsp.ErrInvalidTour},
				{"singleton-tour", valid, []int{0, 0}, false, tsp.ErrInvalidTour},
				{"nil-matrix", nil, base, false, tsp.ErrNilDistanceMatrix},
				{"incomplete", coverageMatrixWithSymmetricEdge(4, 0, 1, math.Inf(1)), base, false, tsp.ErrIncompleteGraph},
				{"dimension", valid, []int{0, 1, 2, 0}, false, tsp.ErrDimensionMismatch},
				{"duplicate", valid, []int{0, 1, 1, 3, 0}, false, tsp.ErrInvalidTour},
				{"overflow", coverageMatrixFromRows([][]float64{{0, math.MaxFloat64}, {math.MaxFloat64, 0}}), []int{0, 1, 0}, false, tsp.ErrNumericOverflow},
			} {
				t.Run(tc.name, func(t *testing.T) {
					opts := tsp.DefaultOptions()
					if tc.badOptions {
						opts.Eps = math.NaN()
					}
					before := slices.Clone(tc.tour)
					result, err := solver.run(tc.dist, tc.tour, opts)
					mustErrIs(t, err, tc.want)
					if result != nil {
						t.Fatalf("invalid input published %+v", result)
					}
					mustEqualInts(t, tc.tour, before)
				})
			}
		})
	}
	// Inject invalid scalars only at the internal snapshot boundary; public
	// admission rejects them earlier. A failed finalizer must publish no tour.
	for _, tc := range []struct {
		value float64
		want  error
	}{
		{math.NaN(), tsp.ErrNaNInf}, {math.Inf(-1), tsp.ErrNaNInf},
		{math.Inf(1), tsp.ErrIncompleteGraph}, {-1, tsp.ErrNegativeWeight},
		{math.MaxFloat64, tsp.ErrNumericOverflow},
	} {
		weights := tsp.NewWeightBufferForTest(2, []float64{0, tc.value, tc.value, 0})
		_, err := tsp.CostClosedTourFromWeightsForTest(weights, []int{0, 1, 0})
		mustErrIs(t, err, tc.want)
		local, err := tsp.FinishLocalSearchCurrentForTest([]int{0, 1, 0}, weights, 0, true, 2, 0, false)
		mustErrIs(t, err, tc.want)
		tour, _, _, _ := tsp.InspectLocalSearchResultForTest(local)
		if len(tour) != 0 {
			t.Fatal("failed finalization published a tour")
		}
	}
	weights := tsp.NewWeightBufferForTest(3, []float64{0, 7, 1, 1, 0, 7, 7, 1, 0})
	for _, tc := range []struct {
		tour      []int
		n         int
		symmetric bool
		want      error
	}{
		{[]int{0, 1, 0}, 2, true, tsp.ErrDimensionMismatch},
		{[]int{0, 1, 2, 1}, 3, true, tsp.ErrDimensionMismatch},
		{[]int{0, 1, 1, 0}, 3, false, tsp.ErrInvalidTour},
	} {
		_, err := tsp.FinishLocalSearchCurrentForTest(tc.tour, weights, 0, false, tc.n, 0, tc.symmetric)
		mustErrIs(t, err, tc.want)
	}
	// The reverse direction costs 21, so directed finalization must retain 3.
	working := []int{0, 2, 1, 0}
	local, err := tsp.FinishLocalSearchCurrentForTest(working, weights, 2, true, 3, 0, false)
	mustNoError(t, err)
	working[1] = 1
	tour, cost, moves, timedOut := tsp.InspectLocalSearchResultForTest(local)
	mustEqualInts(t, tour, []int{0, 2, 1, 0})
	if cost != 3 || moves != 2 || !timedOut {
		t.Fatalf("progress: %g %d %v", cost, moves, timedOut)
	}
	result := &tsp.Result{Tour: []int{0, 1, 2, 0}, Cost: 21, Optimal: true, Iterations: 4}
	tsp.AttachLocalSearchProgressForTest(nil, local)
	tsp.AttachLocalSearchProgressForTest(result, tsp.NewLocalSearchResultForTest(nil, 0, 0, false))
	if result.Cost != 21 || result.Iterations != 4 || !result.Optimal {
		t.Fatal("empty progress changed incumbent")
	}
	tsp.AttachLocalSearchProgressForTest(result, local)
	if result.Cost != 3 || result.Iterations != 6 || !result.TimedOut || result.Optimal {
		t.Fatalf("metadata: %+v", result)
	}
	result.Tour[1] = 1
	detached, _, _, _ := tsp.InspectLocalSearchResultForTest(local)
	mustEqualInts(t, detached, tour)
}

// TestLocalSearchDeterministicTimeouts interrupts real neighborhood sweeps.
// Equal weights ensure no accepted move changes the candidate-count cadence.
// This is a serial test because the clock source is package-scoped.
func TestLocalSearchDeterministicTimeouts(t *testing.T) {
	const n = 70
	const budget = time.Second
	dist := coverageMatrixFromRows(coverageUnitRows(n))
	base := make([]int, n+1)
	for i := 0; i < n; i++ {
		base[i] = i
	}
	// Each phase reads the clock at deadline creation and each periodic check.
	twoCalls := 1 + (n-1)*(n-2)/2/(tsp.TwoOptDeadlineMaskForTest+1)
	threeCalls := 1 + 7*(n-1)*(n-2)*(n-3)/6/(tsp.ThreeOptDeadlineMaskForTest+1)
	for _, tc := range []struct {
		name      string
		algo      tsp.Algorithm
		symmetric bool
		direct    bool
		expireAt  int
	}{
		{"direct-two", tsp.TwoOptOnly, true, true, 2},
		{"direct-three", tsp.ThreeOptOnly, true, true, 2},
		{"direct-three-directed", tsp.ThreeOptOnly, false, true, 2},
		{"christofides-pre-two", tsp.Christofides, true, false, 2},
		{"christofides-three", tsp.Christofides, true, false, twoCalls + 2},
		{"christofides-post-two", tsp.Christofides, true, false, twoCalls + threeCalls + 2},
		{"three-pre-two", tsp.ThreeOptOnly, true, false, 2},
		{"three-main", tsp.ThreeOptOnly, true, false, twoCalls + 2},
		{"three-post-two", tsp.ThreeOptOnly, true, false, twoCalls + threeCalls + 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			epoch := time.Unix(0, 0)
			t.Cleanup(tsp.SetLocalSearchClockForTest(func() time.Time {
				calls++
				if calls >= tc.expireAt {
					return epoch.Add(2 * budget)
				}
				return epoch
			}))
			opts := tsp.DefaultOptions()
			opts.Algo, opts.Symmetric, opts.TimeLimit = tc.algo, tc.symmetric, budget
			opts.EnableLocalSearch, opts.BestImprovement, opts.ShuffleNeighborhood = true, true, false
			opts.BoundAlgo = tsp.NoBound
			var result *tsp.Result
			var err error
			initial := slices.Clone(base)
			if !tc.direct {
				result, err = tsp.SolveMatrix(dist, nil, opts)
			} else if tc.algo == tsp.TwoOptOnly {
				result, err = tsp.TwoOptSearch(dist, initial, opts)
			} else {
				result, err = tsp.ThreeOptSearch(dist, initial, opts)
			}
			mustErrIs(t, err, tsp.ErrTimeLimit)
			if result == nil {
				t.Fatal("deadline lost feasible incumbent")
			}
			mustValidResult(t, result, n, 0, tc.algo)
			if !result.TimedOut || result.Optimal || result.Exact || result.Cost != n || result.Iterations != 0 || calls != tc.expireAt {
				t.Fatalf("wrong timeout publication: %+v; clock reads %d want %d", result, calls, tc.expireAt)
			}
			if tc.algo == tsp.Christofides && result.ApproximationRatio != tsp.ChristofidesApproximationRatio {
				t.Fatal("local timeout lost the construction guarantee")
			}
			result.Tour[1] = -1
			mustEqualInts(t, initial, base)
		})
	}
}

// TestThreeOptMoveAlgebra compares each of seven symmetric reconnections with
// an independently enumerated closed tour and a full objective sum.
// Integer weights make these algebraic comparisons exact in binary64.
func TestThreeOptMoveAlgebra(t *testing.T) {
	const n = 7
	const i, j, k = 1, 3, 5
	base := []int{0, 1, 2, 3, 4, 5, 6, 0}
	expected := [][]int{
		{0, 2, 1, 3, 4, 5, 6, 0}, {0, 1, 2, 4, 3, 5, 6, 0},
		{0, 2, 1, 4, 3, 5, 6, 0}, {0, 3, 4, 1, 2, 5, 6, 0},
		{0, 3, 4, 2, 1, 5, 6, 0}, {0, 4, 3, 1, 2, 5, 6, 0},
		{0, 4, 3, 2, 1, 5, 6, 0},
	}
	rows := coverageUnitRows(n)
	for u := 0; u < n; u++ {
		for v := u + 1; v < n; v++ {
			rows[u][v] = float64(1 + (u+3)*(v+2)%19)
			rows[v][u] = rows[u][v]
		}
	}
	at := func(u, v int) float64 { return rows[u][v] }
	sum := func(tour []int) float64 {
		total := 0.0
		for p := 1; p < len(tour); p++ {
			total += at(tour[p-1], tour[p])
		}
		return total
	}
	moves := tsp.SymmetricThreeOptMovesForTest()
	if len(moves) != len(expected) {
		t.Fatalf("unexpected move count %d", len(moves))
	}
	for index, move := range moves {
		t.Run(fmt.Sprint(index), func(t *testing.T) {
			mustNoError(t, tsp.ValidateThreeOptMoveForTest(move))
			next := tsp.ApplyThreeOptSymmetricForTest(base, i, j, k, move)
			mustEqualInts(t, next, expected[index])
			mustNoError(t, tsp.ValidateTour(next, n, 0))
			delta, feasible := tsp.ThreeOptDeltaSymmetricForTest(at, base, i, j, k, move)
			if !feasible || delta != sum(expected[index])-sum(base) {
				t.Fatalf("wrong symmetric delta %g", delta)
			}
		})
	}
	// Directed costs are deliberately asymmetric. Tail swap must preserve
	// internal orientations and change exactly three connecting arcs.
	rows[0][3] += 11
	next := tsp.ApplyThreeOptATSPTailSwapForTest(base, i, j, k)
	mustEqualInts(t, next, expected[3])
	delta, feasible := tsp.ThreeOptDeltaATSPTailSwapForTest(at, base, i, j, k)
	if !feasible || delta != sum(expected[3])-sum(base) {
		t.Fatalf("wrong directed delta %g", delta)
	}
	rows[0][3] = math.Inf(1)
	if _, ok := tsp.ThreeOptDeltaATSPTailSwapForTest(at, base, i, j, k); ok {
		t.Fatal("missing directed arc accepted")
	}
	rows[0][2] = math.Inf(1)
	if _, ok := tsp.ThreeOptDeltaSymmetricForTest(at, base, i, j, k, moves[0]); ok {
		t.Fatal("missing symmetric arc accepted")
	}
	s1 := tsp.NewOrientedSegmentForTest(tsp.SegmentS1ForTest, false)
	bad := tsp.NewOrientedSegmentForTest(tsp.SegmentIDForTest(255), false)
	mustErrIs(t, tsp.ValidateThreeOptMoveForTest(tsp.NewThreeOptMoveForTest(s1, s1)), tsp.ErrInvalidTour)
	mustErrIs(t, tsp.ValidateThreeOptMoveForTest(tsp.NewThreeOptMoveForTest(s1, bad)), tsp.ErrInvalidOptions)
	first, last := tsp.SegmentEndpointsForTest(bad, 1, 2, 3, 4)
	if first != -1 || last != -1 {
		t.Fatal("invalid segment acquired endpoints")
	}
	out := []int{0}
	tsp.AppendOrientedSegmentForTest(&out, []int{1, 2}, []int{3, 4}, bad)
	mustEqualInts(t, out, []int{0})
	mustEqualInts(t, base, []int{0, 1, 2, 3, 4, 5, 6, 0})
}

// TestDirectedThreeOptBestImprovement uses a unique cost-six Hamiltonian cycle.
// All other arcs cost 100, so the selected one-move improvement is provably optimal
// for the fixture, without claiming that the heuristic is generally exact.
func TestDirectedThreeOptBestImprovement(t *testing.T) {
	const n = 6
	rows := coverageUnitRows(n)
	for i := range rows {
		for j := range rows[i] {
			if i != j {
				rows[i][j] = 100
			}
		}
	}
	want := []int{0, 3, 4, 1, 2, 5, 0}
	for i := 1; i < len(want); i++ {
		rows[want[i-1]][want[i]] = 1
	}
	opts := tsp.DefaultOptions()
	opts.Symmetric, opts.BestImprovement, opts.ShuffleNeighborhood = false, true, true
	opts.ThreeOptMaxMoves, opts.Seed = 1, 41
	opts.BoundAlgo = tsp.NoBound
	base := []int{0, 1, 2, 3, 4, 5, 0}
	for run := 0; run < 2; run++ {
		result, err := tsp.ThreeOptSearch(coverageMatrixFromRows(rows), base, opts)
		if err != nil || result == nil {
			t.Fatalf("directed best improvement: %v, %v", result, err)
		}
		mustEqualInts(t, result.Tour, want)
		if result.Cost != n || result.Iterations != 1 || result.Optimal || result.Exact {
			t.Fatalf("result: %+v", result)
		}
	}
	mustEqualInts(t, base, []int{0, 1, 2, 3, 4, 5, 0})
}
