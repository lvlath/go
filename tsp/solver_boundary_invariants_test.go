// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp_test verifies matrix admission, solver publication and tour utilities.
// Fault-injecting matrix implementations target second-pass safety without data races.
package tsp_test

import (
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/tsp"
)

// TestWeightCopySecondPassGuards exercises the real snapshot boundary.
// The two-vertex source is valid during four scalar reads and two symmetry reads;
// only the subsequent copy is faulty. No partial weight buffer may escape.
func TestWeightCopySecondPassGuards(t *testing.T) {
	const firstCopy = 7
	const offDiagonal = firstCopy + 1
	backend := errors.New("test backend read failure")
	for _, tc := range []struct {
		name        string
		failAt      int
		value       float64
		cause, want error
	}{
		{"read-error", firstCopy, 0, backend, tsp.ErrDimensionMismatch},
		{"nan", offDiagonal, math.NaN(), nil, tsp.ErrNaNInf},
		{"negative-infinity", offDiagonal, math.Inf(-1), nil, tsp.ErrNaNInf},
		{"infinite-diagonal", firstCopy, math.Inf(1), nil, tsp.ErrNonZeroDiagonal},
		{"nonzero-diagonal", firstCopy, 1, nil, tsp.ErrNonZeroDiagonal},
		{"negative-weight", offDiagonal, -1, nil, tsp.ErrNegativeWeight},
		{"missing-edge", offDiagonal, math.Inf(1), nil, tsp.ErrIncompleteGraph},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dist := &lateTSPRead{testDense: testDense{a: coverageUnitRows(2)}, failAt: tc.failAt, value: tc.value, cause: tc.cause}
			b, err := tsp.CopyWeightsWithModeForTest(dist, true, true)
			mustErrIs(t, err, tc.want)
			if tc.cause != nil {
				mustErrIs(t, err, tc.cause)
			}
			n, weights := tsp.InspectWeightBufferForTest(b)
			if n != 0 || len(weights) != 0 {
				t.Fatal("failed copy published a partial weight buffer")
			}
		})
	}
	// A missing edge is allowed only in the explicitly closure-ready snapshot.
	dist := testDense{a: [][]float64{{0, math.Inf(1)}, {math.Inf(1), 0}}}
	b, err := tsp.CopyWeightsWithModeForTest(dist, true, false)
	mustNoError(t, err)
	n, weights := tsp.InspectWeightBufferForTest(b)
	if n != 2 || len(weights) != 4 || !math.IsInf(weights[1], 1) {
		t.Fatal("closure-ready buffer lost absence sentinel")
	}
	mustNoError(t, dist.Set(0, 1, 9))
	_, weights = tsp.InspectWeightBufferForTest(b)
	if !math.IsInf(weights[1], 1) {
		t.Fatal("weight snapshot aliases source")
	}
}

// TestSolverPreparationAndPublication tests two independent phases of SolveMatrix.
// A failed closure copy preserves the original cause; a published result owns its
// route and IDs independently of both the kernel result and the caller.
func TestSolverPreparationAndPublication(t *testing.T) {
	opts := tsp.DefaultOptions()
	opts.RunMetricClosure = true
	backend := errors.New("test closure-copy fault")
	// The first admission of a symmetric 2x2 matrix consumes exactly six reads.
	dist := &lateTSPRead{testDense: testDense{a: coverageUnitRows(2)}, failAt: 7, cause: backend}
	_, err := tsp.PrepareSolverDistanceMatrixForTest(dist, opts)
	mustErrIs(t, err, tsp.ErrDimensionMismatch)
	mustErrIs(t, err, backend)
	// The next fixture passes admission, but violates the Dense write policy.
	dist = &lateTSPRead{testDense: testDense{a: coverageUnitRows(2)}, failAt: 8, value: math.NaN()}
	_, err = tsp.PrepareSolverDistanceMatrixForTest(dist, opts)
	mustErrIs(t, err, tsp.ErrNaNInf)
	mustErrIs(t, err, matrix.ErrNaNInf)
	// Disconnected components cannot acquire finite paths from metric closure.
	_, err = tsp.PrepareSolverDistanceMatrixForTest(testDense{a: [][]float64{{0, math.Inf(1)}, {math.Inf(1), 0}}}, opts)
	mustErrIs(t, err, tsp.ErrIncompleteGraph)
	// Finite distances do not guarantee representable APSP arithmetic.
	_, err = tsp.PrepareSolverDistanceMatrixForTest(testDense{a: [][]float64{{0, math.MaxFloat64}, {math.MaxFloat64, 0}}}, opts)
	mustErrIs(t, err, tsp.ErrNaNInf)
	mustErrIs(t, err, matrix.ErrNaNInf)
	if tsp.PublishKernelResultForTest(nil, nil, opts, true) != nil {
		t.Fatal("nil kernel result became nonnil")
	}
	kernel := &tsp.Result{Tour: []int{0, 1, 0}, Cost: 2, Algorithm: tsp.TwoOptOnly, IDs: []string{"stale", "ids"}}
	ids := []string{"A", "B"}
	published := tsp.PublishKernelResultForTest(kernel, ids, opts, true)
	if published == nil {
		t.Fatal("lost kernel result")
	}
	if published.Algorithm != opts.Algo || !published.MetricClosureApplied || published.Cost != 2 {
		t.Fatalf("publication: %+v", published)
	}
	published.Tour[1] = 0
	published.IDs[0] = "changed"
	mustEqualInts(t, kernel.Tour, []int{0, 1, 0})
	requireCoverageStringsEqual(t, ids, []string{"A", "B"}, "input IDs")
	requireCoverageStringsEqual(t, kernel.IDs, []string{"stale", "ids"}, "kernel IDs")
	published = tsp.PublishKernelResultForTest(kernel, nil, opts, false)
	if published == nil || published.IDs != nil || published.MetricClosureApplied {
		t.Fatalf("stale metadata: %+v", published)
	}
}

// TestPreparedDispatchRefusesInvalidKernelInputs checks the no-fallback contract.
// Direct dispatcher fixtures bypass facade admission only to exercise the
// propagation of actual kernel failures; no solver or matching result is mocked.
func TestPreparedDispatchRefusesInvalidKernelInputs(t *testing.T) {
	for _, algo := range []tsp.Algorithm{tsp.Christofides, tsp.ExactHeldKarp, tsp.BranchAndBound, tsp.TwoOptOnly, tsp.ThreeOptOnly} {
		opts := tsp.DefaultOptions()
		opts.Algo = algo
		result, err := tsp.SolvePreparedMatrixForTest(tsp.NewPreparedMatrixForTest(nil, 4, false), nil, opts)
		mustErrIs(t, err, tsp.ErrNilDistanceMatrix)
		if result != nil {
			t.Fatalf("algorithm %v published invalid result", algo)
		}
	}
	for _, algo := range []tsp.Algorithm{tsp.TwoOptOnly, tsp.ThreeOptOnly} {
		opts := tsp.DefaultOptions()
		opts.Algo = algo
		result, err := tsp.SolvePreparedMatrixForTest(tsp.NewPreparedMatrixForTest(nil, 1, false), nil, opts)
		mustErrIs(t, err, tsp.ErrDimensionMismatch)
		if result != nil {
			t.Fatal("invalid ring published a result")
		}
	}
	opts := tsp.DefaultOptions()
	opts.Algo = tsp.Algorithm(255)
	_, err := tsp.SolvePreparedMatrixForTest(tsp.NewPreparedMatrixForTest(nil, 4, false), nil, opts)
	mustErrIs(t, err, tsp.ErrUnsupportedAlgorithm)
	opts = tsp.DefaultOptions()
	opts.MaxExactN = -1
	_, err = tsp.FinalizeSolverOptionsForTest(4, opts)
	mustErrIs(t, err, tsp.ErrInvalidOptions)
	opts.MaxExactN = 0
	opts.Algo = tsp.Auto
	final, err := tsp.FinalizeSolverOptionsForTest(4, opts)
	mustNoError(t, err)
	if final.MaxExactN != tsp.DefaultMaxExactN || final.Algo != tsp.ExactHeldKarp {
		t.Fatalf("normalization: %+v", final)
	}
	opts.Eps = math.NaN()
	_, err = tsp.FinalizeSolverOptionsForTest(4, opts)
	mustErrIs(t, err, tsp.ErrInvalidOptions)
	opts = tsp.DefaultOptions()
	// The singleton shortcut must not publish a route if the backend read fails.
	backend := errors.New("test singleton read failure")
	source := &lateTSPRead{testDense: testDense{a: [][]float64{{0}}}, failAt: 1, cause: backend}
	result, handled, err := tsp.SolveDegenerateIfAnyForTest(source, nil, opts)
	mustErrIs(t, err, tsp.ErrDimensionMismatch)
	mustErrIs(t, err, backend)
	if !handled || result != nil {
		t.Fatal("invalid singleton publication")
	}
	opts.Eps = math.NaN()
	result, handled, err = tsp.SolveDegenerateIfAnyForTest(testDense{a: [][]float64{{0}}}, nil, opts)
	mustErrIs(t, err, tsp.ErrInvalidOptions)
	if !handled || result != nil {
		t.Fatal("invalid singleton policy published")
	}
	_, err = tsp.TrivialRingForTest(4, -1)
	mustErrIs(t, err, tsp.ErrStartOutOfRange)
	ring, err := tsp.TrivialRingForTest(4, 2)
	mustNoError(t, err)
	mustEqualInts(t, ring, []int{2, 3, 0, 1, 2})
}

// TestTourUtilityGuardBoundaries verifies that rejected structural operations
// leave caller slices unchanged. TourCost is deliberately tested separately:
// it evaluates a sequence of arcs and does not promise Hamiltonian validation.
func TestTourUtilityGuardBoundaries(t *testing.T) {
	mustErrIs(t, tsp.ValidatePermutation([]int{}, 0), tsp.ErrDimensionMismatch)
	mustErrIs(t, tsp.ValidatePermutation([]int{0, 2}, 2), tsp.ErrDimensionMismatch)
	mustErrIs(t, tsp.ValidateTour(nil, 3, 0), tsp.ErrNilTour)
	mustErrIs(t, tsp.ValidateTour([]int{0, 1, 2, 0}, 3, 3), tsp.ErrStartOutOfRange)
	mustErrIs(t, tsp.ValidateTour([]int{0, 1, 0}, 3, 0), tsp.ErrInvalidTour)
	_, err := tsp.RotateTourToStart(nil, 0)
	mustErrIs(t, err, tsp.ErrDimensionMismatch)
	_, err = tsp.RotateTourToStart([]int{0, 1, 0}, 2)
	mustErrIs(t, err, tsp.ErrStartOutOfRange)
	mustErrIs(t, tsp.CanonicalizeOrientationInPlace([]int{0, 0}), tsp.ErrDimensionMismatch)
	mustErrIs(t, tsp.CanonicalizeOrientationInPlace([]int{0, 1, 2}), tsp.ErrDimensionMismatch)
	for _, tc := range []struct {
		tour []int
		i, k int
	}{
		{[]int{0, 0}, 1, 1}, {[]int{0, 1, 2}, 1, 2}, {[]int{0, 1, 2, 0}, 0, 2}, {[]int{0, 1, 2, 0}, 1, 3}, {[]int{0, 1, 2, 0}, 2, 1},
	} {
		before := append([]int(nil), tc.tour...)
		mustErrIs(t, tsp.ReverseArcInPlaceForTest(tc.tour, tc.i, tc.k), tsp.ErrDimensionMismatch)
		mustEqualInts(t, tc.tour, before)
	}
	route := []int{0, 1, 2, 3, 0}
	mustNoError(t, tsp.ReverseArcInPlaceForTest(route, 1, 3))
	mustEqualInts(t, route, []int{0, 3, 2, 1, 0})
}

// TestTourCostDenseAndGenericGuardParity checks actual numeric classifications
// for both backends, including a Dense created with explicit dirty-data policy.
// These values are intentionally inadmissible for TSP, not suggested route costs.
func TestTourCostDenseAndGenericGuardParity(t *testing.T) {
	tour := []int{0, 1, 0}
	var typedNil *matrix.Dense
	_, err := tsp.TourCost(typedNil, tour)
	mustErrIs(t, err, tsp.ErrNilDistanceMatrix)
	mustErrIs(t, err, matrix.ErrNilMatrix)
	dist := newTSPFractionalMatrix(t, 1)
	_, err = tsp.TourCost(dist, nil)
	mustErrIs(t, err, tsp.ErrNilTour)
	_, err = tsp.TourCost(dist, []int{0})
	mustErrIs(t, err, tsp.ErrInvalidTour)
	rect, err := matrix.NewDense(1, 2)
	mustNoError(t, err)
	_, err = tsp.TourCost(rect, tour)
	mustErrIs(t, err, tsp.ErrNonSquare)
	_, err = tsp.TourCost(testDense{a: [][]float64{{0, 1}}}, tour)
	mustErrIs(t, err, tsp.ErrNonSquare)
	_, err = tsp.TourCost(dist, []int{0, 2, 0})
	mustErrIs(t, err, tsp.ErrInvalidTour)
	for _, tc := range []struct {
		weight float64
		want   error
	}{
		{math.NaN(), tsp.ErrNaNInf}, {math.Inf(-1), tsp.ErrIncompleteGraph},
		{math.Inf(1), tsp.ErrIncompleteGraph}, {-1, tsp.ErrNegativeWeight}, {math.MaxFloat64, tsp.ErrNumericOverflow},
	} {
		dense, err := matrix.NewPreparedDense(2, 2, matrix.WithNoValidateNaNInf())
		mustNoError(t, err)
		if dense == nil {
			t.Fatal("nil prepared matrix")
		}
		mustNoError(t, dense.Set(0, 1, tc.weight))
		mustNoError(t, dense.Set(1, 0, tc.weight))
		for _, m := range []matrix.Matrix{dense, testDense{a: [][]float64{{0, tc.weight}, {tc.weight, 0}}}} {
			cost, err := tsp.TourCost(m, tour)
			mustErrIs(t, err, tc.want)
			if cost != 0 {
				t.Fatalf("failed cost publication: %g", cost)
			}
		}
	}
	faulty := &lateTSPRead{testDense: testDense{a: coverageUnitRows(2)}, failAt: 1, cause: errors.New("test edge read failure")}
	_, err = tsp.TourCost(faulty, tour)
	mustErrIs(t, err, tsp.ErrDimensionMismatch)
}

// TestEulerianRejectsInvalidTopology checks that an incomplete circuit is never
// published for malformed half-edge ownership, odd degrees or disconnected edges.
// Each fixture is small enough to verify by inspection; no timing assumptions apply.
func TestEulerianRejectsInvalidTopology(t *testing.T) {
	for _, tc := range []struct {
		name  string
		adj   [][]int
		start int
		want  error
	}{
		{"empty", nil, 0, tsp.ErrDimensionMismatch},
		{"start", [][]int{{1, 1}, {0, 0}}, 2, tsp.ErrStartOutOfRange},
		{"odd-degree", [][]int{{1}, {0}}, 0, tsp.ErrNonEulerian},
		{"out-of-range", [][]int{{2, 2}, {}}, 0, tsp.ErrDimensionMismatch},
		{"self-loop", [][]int{{0, 0}}, 0, tsp.ErrNonEulerian},
		{"missing-twin", [][]int{{1, 1}, {}}, 0, tsp.ErrNonEulerian},
		{"disconnected-even", [][]int{{1, 1}, {0, 0}, {3, 3}, {2, 2}}, 0, tsp.ErrNonEulerian},
	} {
		t.Run(tc.name, func(t *testing.T) {
			before := coverageCloneAdj(tc.adj)
			circuit, err := tsp.EulerianCircuit(tc.adj, tc.start)
			mustErrIs(t, err, tc.want)
			if circuit != nil {
				t.Fatalf("invalid topology published %v", circuit)
			}
			requireCoverageAdjEqual(t, tc.adj, before)
		})
	}
	for _, tc := range []struct {
		walk     []int
		n, start int
		want     error
	}{
		{nil, 0, 0, tsp.ErrDimensionMismatch},
		{[]int{0, 1, 0}, 2, 2, tsp.ErrStartOutOfRange},
		{[]int{0, 2, 0}, 2, 0, tsp.ErrDimensionMismatch},
		{[]int{0, 1, 0}, 3, 0, tsp.ErrDimensionMismatch},
	} {
		tour, err := tsp.ShortcutEulerianToHamiltonian(tc.walk, tc.n, tc.start)
		mustErrIs(t, err, tc.want)
		if tour != nil {
			t.Fatalf("invalid shortcut published %v", tour)
		}
	}
}

// exactPermutationObjective computes the minimum complete directed cycle cost.
// It enumerates every permutation of the non-start vertices and accumulates each
// route independently of Held-Karp, Branch-and-Bound, matching and TourCost.
// Implementation:
//   - Fix the start and recursively enumerate unused vertices.
//   - Close every complete path with its original directed return arc.
//   - Retain the smallest sum without pruning or a numerical tolerance.
//
// Inputs: square, finite small integer costs; valid fixed start.
// Returns: exact binary64 minimum for the bounded integer fixtures below.
// Errors: none; this test oracle requires a complete small matrix.
// Determinism: ascending candidate vertex order.
// Complexity: O((n-1)!) time, O(n) stack and visited storage.
// Notes: this exhaustive oracle is intentionally limited to small fixtures.
// AI-Hints: never share the production bound, recurrence or cost evaluator here.
func exactPermutationObjective(rows [][]float64, start int) float64 {
	n := len(rows)
	used := make([]bool, n)
	used[start] = true
	best := math.Inf(1)
	var visit func(int, int, float64)
	visit = func(last, depth int, cost float64) {
		if depth == n {
			best = math.Min(best, cost+rows[last][start])
			return
		}
		for next := 0; next < n; next++ {
			if used[next] {
				continue
			}
			used[next] = true
			visit(next, depth+1, cost+rows[last][next])
			used[next] = false
		}
	}
	visit(start, 1, 0)
	return best
}

// TestExactFacadesAgainstIndependentPermutationOracle compares the two exact
// solvers across every supported bound policy on symmetric and directed costs.
// A nonzero start catches index normalization and direction-publication defects.
func TestExactFacadesAgainstIndependentPermutationOracle(t *testing.T) {
	const n, start = 6, 3
	for _, symmetric := range []bool{true, false} {
		rows := coverageUnitRows(n)
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				if i == j {
					continue
				}
				if symmetric && i > j {
					rows[i][j] = rows[j][i]
					continue
				}
				rows[i][j] = float64(1 + ((i+2)*(j+5)+3*i)%23)
			}
		}
		want := exactPermutationObjective(rows, start)
		dist := coverageMatrixFromRows(rows)
		opts := tsp.DefaultOptions()
		opts.StartVertex, opts.Symmetric, opts.EnableLocalSearch = start, symmetric, false
		opts.MaxExactN = 0 // Explicitly exercise the direct Held-Karp default size guard.
		opts.BoundAlgo = tsp.NoBound
		hk, err := tsp.HeldKarp(dist, opts)
		mustNoError(t, err)
		mustValidResult(t, hk, n, start, tsp.ExactHeldKarp)
		if hk == nil || !hk.Exact || !hk.Optimal || hk.Cost != want {
			t.Fatalf("Held-Karp optimum %g: %+v", want, hk)
		}
		// Check the witness itself rather than trusting its published objective.
		hkTotal := 0.0
		for i := 1; i < len(hk.Tour); i++ {
			hkTotal += rows[hk.Tour[i-1]][hk.Tour[i]]
		}
		if hkTotal != want {
			t.Fatalf("Held-Karp route cost %g != %g", hkTotal, want)
		}
		for _, bound := range []tsp.BoundAlgo{tsp.NoBound, tsp.SimpleBound, tsp.OneTreeBound} {
			if !symmetric && bound == tsp.OneTreeBound {
				continue
			}
			opts.BoundAlgo = bound
			result, err := tsp.BranchAndBoundSolve(dist, opts)
			mustNoError(t, err)
			mustValidResult(t, result, n, start, tsp.BranchAndBound)
			if result == nil || !result.Exact || !result.Optimal || result.Cost != want {
				t.Fatalf("bound %v optimum %g: %+v", bound, want, result)
			}
			total := 0.0
			for i := 1; i < len(result.Tour); i++ {
				total += rows[result.Tour[i-1]][result.Tour[i]]
			}
			if total != want {
				t.Fatalf("published route cost %g != %g", total, want)
			}
		}
	}
	dist := coverageMatrixFromRows(coverageUnitRows(n))
	opts := tsp.DefaultOptions()
	opts.Eps = math.NaN()
	result, err := tsp.HeldKarp(dist, opts)
	mustErrIs(t, err, tsp.ErrInvalidOptions)
	if result != nil {
		t.Fatal("invalid exact policy published a result")
	}
	result, err = tsp.BranchAndBoundSolve(dist, opts)
	mustErrIs(t, err, tsp.ErrInvalidOptions)
	if result != nil {
		t.Fatal("invalid bound policy published a result")
	}
	opts = tsp.DefaultOptions()
	opts.StartVertex = n
	result, err = tsp.HeldKarp(dist, opts)
	mustErrIs(t, err, tsp.ErrStartOutOfRange)
	if result != nil {
		t.Fatal("invalid start published a result")
	}
	result, err = tsp.ChristofidesSolve(dist, opts)
	mustErrIs(t, err, tsp.ErrStartOutOfRange)
	if result != nil {
		t.Fatal("invalid Christofides start published a result")
	}
}

// coverageUnitRows builds a complete unit-cost symmetric matrix with zero diagonal.
// Its exact cycle objective equals the vertex count for every Hamiltonian route.
// Complexity: O(n*n) time and space; each invocation owns its rows.
func coverageUnitRows(n int) [][]float64 {
	rows := make([][]float64, n)
	for u := range rows {
		rows[u] = make([]float64, n)
		for v := range rows[u] {
			if u != v {
				rows[u][v] = 1
			}
		}
	}
	return rows
}

// coverageMatrixFromRows detaches a generic matrix fixture from oracle weights.
// The existing testDense backend deliberately retains invalid scalar inputs.
// Complexity: O(n*n) time and space for square rows.
func coverageMatrixFromRows(rows [][]float64) matrix.Matrix {
	copied := make([][]float64, len(rows))
	for i := range rows {
		copied[i] = slices.Clone(rows[i])
	}
	return testDense{a: copied}
}

// coverageMatrixWithSymmetricEdge modifies one undirected edge in a unit graph.
// A symmetric malformed scalar isolates numeric admission from asymmetry errors.
// Complexity: O(n*n) time and space, including the independent fixture.
func coverageMatrixWithSymmetricEdge(n, u, v int, value float64) matrix.Matrix {
	rows := coverageUnitRows(n)
	rows[u][v], rows[v][u] = value, value
	return testDense{a: rows}
}

// coverageCloneAdj snapshots all adjacency rows, preserving nil versus empty.
// Complexity: O(V+E) time and space for the multigraph's half-edge storage.
func coverageCloneAdj(adj [][]int) [][]int {
	if adj == nil {
		return nil
	}
	out := make([][]int, len(adj))
	for i := range adj {
		out[i] = slices.Clone(adj[i])
	}
	return out
}

// requireCoverageAdjEqual checks exact row order, multiplicity and nilness.
// It does not sort or collapse parallel edges, which would conceal mutation.
// Complexity: O(V+E) time and O(1) additional comparison space.
func requireCoverageAdjEqual(t *testing.T, got, want [][]int) {
	t.Helper()
	if (got == nil) != (want == nil) || len(got) != len(want) {
		t.Fatalf("adjacency shape: got %v want %v", got, want)
	}
	for i := range got {
		if (got[i] == nil) != (want[i] == nil) || !slices.Equal(got[i], want[i]) {
			t.Fatalf("adjacency row %d: got %v want %v", i, got[i], want[i])
		}
	}
}

// requireCoverageStringsEqual checks stable original ID order without reflection.
// Complexity: O(len(got)) time and O(1) additional space.
func requireCoverageStringsEqual(t *testing.T, got, want []string, operation string) {
	t.Helper()
	if !slices.Equal(got, want) {
		t.Fatalf("%s: got %v want %v", operation, got, want)
	}
}
