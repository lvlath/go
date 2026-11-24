// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix_test verifies finite-distance admission, in-place relaxation and metric closure.
// Failures are classified by public sentinels and must not be confused with unreachable paths.
package matrix_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/matrix"
)

// TestCoverageInitDistancesAdmission verifies diagonal and absence conversion at the public boundary.
// It distinguishes invalid numeric input from an incompatible Dense write policy.
// Implementation: Exercise nil, rectangular, dirty, and strict finite-only matrices; then check a
// valid +Inf-enabled matrix, including preservation of a negative self-loop for cycle detection.
// Behavior highlights: A positive loop is reduced to zero, a negative loop is retained, and
// an absent off-diagonal zero is replaced with positive infinity.
// Inputs: Small fixed Dense fixtures; none of the failing cases depend on internal buffer corruption.
// Returns: Test assertions only.
// Errors: ErrNilMatrix, ErrDimensionMismatch, ErrNaNInf and ErrBadShape.
// Determinism: Fixed row-major admission order.
// Complexity: O(n^2) time and storage per fixture.
// Notes: Initialization is in-place; do not require rollback of prior valid writes on error.
// AI-Hints: Allocate distance storage with WithAllowInfDistances before converting absent edges.
func TestCoverageInitDistancesAdmission(t *testing.T) {
	AssertErrorIs(t, matrix.InitDistancesInPlace(nil), matrix.ErrNilMatrix)
	AssertErrorIs(t, matrix.InitDistancesInPlace(MustDense(t, 2, 3)), matrix.ErrDimensionMismatch)
	for _, tc := range []struct {
		name string
		data []float64
		want error
	}{
		{"nan-diagonal", []float64{math.NaN(), 1, 1, 0}, matrix.ErrNaNInf},
		{"negative-inf-diagonal", []float64{math.Inf(-1), 1, 1, 0}, matrix.ErrNaNInf},
		{"positive-inf-diagonal", []float64{math.Inf(1), 1, 1, 0}, matrix.ErrBadShape},
		{"nan-edge", []float64{0, math.NaN(), 1, 0}, matrix.ErrNaNInf},
		{"negative-inf-edge", []float64{0, math.Inf(-1), 1, 0}, matrix.ErrNaNInf},
	} {
		t.Run(tc.name, func(t *testing.T) {
			AssertErrorIs(t, matrix.InitDistancesInPlace(coverageRawDense(t, 2, 2, tc.data)), tc.want)
		})
	}
	// A default finite-only Dense cannot store the absence sentinel.
	AssertErrorIs(t, matrix.InitDistancesInPlace(MustDense(t, 2, 2)), matrix.ErrNaNInf)
	inf := math.Inf(1)
	d := coverageDistanceDense(t, 2, []float64{-2, 0, 3, 4})
	if err := matrix.InitDistancesInPlace(d); err != nil {
		t.Fatal(err)
	}
	CompareExact(t, [][]float64{{-2, inf}, {3, 0}}, d)
	AssertErrorIs(t, matrix.ValidateDistanceMatrix(d), matrix.ErrNegativeCycle)
}

// TestCoverageFloydWarshallBackendFailures checks both validation and each relaxation read boundary.
// The backend is a legitimate generic Matrix with deterministic per-coordinate access refusal.
// Implementation: Admit a 1x1 matrix, refuse its diagonal at successive reads, and separately
// refuse an actual shortest-path write in a three-node distance matrix.
// Behavior highlights: Errors after admission remain backend errors rather than shape errors.
// Inputs: A zero diagonal and a directed three-node chain with an initially expensive direct edge.
// Returns: Test assertions only.
// Errors: The same injected sentinel under errors.Is at every access boundary.
// Determinism: Coordinate-specific access counts; no time-based faults.
// Complexity: O(n^3) relaxation per case, O(n^2) fixture storage.
// Notes: The exact read sequence here is intentional for distinguishing the published generic stages.
// AI-Hints: FloydWarshall is in-place; do not claim transactional rollback after a backend write error.
func TestCoverageFloydWarshallBackendFailures(t *testing.T) {
	for _, tc := range []struct {
		name string
		skip int
	}{
		{"admission", 0}, {"source-distance", 1}, {"target-distance", 2},
		{"current-distance", 3}, {"final-diagonal", 4},
	} {
		t.Run(tc.name, func(t *testing.T) {
			backend := &coverageMatrix{Matrix: MustDense(t, 1, 1), read: coverageFaultAt(0, 0, tc.skip)}
			AssertErrorIs(t, matrix.APSPInPlace(backend), errCoverageBackend)
		})
	}
	inf := math.Inf(1)
	input := coverageDistanceDense(t, 3, []float64{0, 2, 10, inf, 0, 3, inf, inf, 0})
	backend := &coverageMatrix{Matrix: input, write: coverageFaultAt(0, 2, 0)}
	AssertErrorIs(t, matrix.FloydWarshall(backend), errCoverageBackend)
	if got := MustAt(t, input, 0, 2); got != 10 {
		t.Fatalf("rejected write changed cell: %g", got)
	}
	// A fresh backend with no refusal must complete the same path reduction.
	if err := matrix.APSPInPlace(&coverageMatrix{Matrix: input}); err != nil {
		t.Fatal(err)
	}
	CompareExact(t, [][]float64{{0, 2, 5}, {inf, 0, 3}, {inf, inf, 0}}, input)
}

// TestCoverageFloydWarshallNumericFailure distinguishes negative cycles from arithmetic overflow.
// The same finite inputs must be rejected in both Dense and generic backends.
// Implementation: Run a negative cycle and an overflowing path candidate under both dispatch modes.
// Behavior highlights: Neither failure may be encoded as an ordinary unreachable +Inf path.
// Inputs: Two finite square matrices with valid zero diagonals.
// Returns: Test assertions only.
// Errors: ErrNegativeCycle and the historical FloydWarshall ErrNaNInf arithmetic classification.
// Determinism: Fresh storage and fixed k/i/j ordering for each invocation.
// Complexity: O(n^3) time, O(n^2) fixture space.
// Notes: The arithmetic sentinel is specific to this legacy kernel, not the factorization API.
// AI-Hints: Do not assert the mutated matrix as a valid completed distance artifact after an error.
func TestCoverageFloydWarshallNumericFailure(t *testing.T) {
	for _, tc := range []struct {
		name string
		data []float64
		want error
	}{
		{"negative-cycle", []float64{0, -2, 1, 0}, matrix.ErrNegativeCycle},
		{"overflowing-candidate", []float64{0, math.MaxFloat64, math.MaxFloat64, 0}, matrix.ErrNaNInf},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, generic := range []bool{false, true} {
				var input matrix.Matrix = NewFilledDense(t, 2, 2, tc.data)
				if generic {
					input = &coverageMatrix{Matrix: input}
				}
				AssertErrorIs(t, matrix.APSPInPlace(input), tc.want)
			}
		})
	}
}

// TestCoverageMetricClosureZeroWeightAndReplay verifies the complete zero-preserving closure workflow.
// It confirms that the closure preserves a real zero-cost link and marks derived topology nonexportable.
// Implementation: Build one directed weighted graph, run in-place closure twice, and compare the
// direct closure builder with an independent expected distance table.
// Behavior highlights: The finite zero edge remains reachable; closure is idempotent on this fixture.
// Inputs: A three-node graph with zero and positive weights, and a missing reverse path.
// Returns: Test assertions only.
// Errors: ErrMatrixNotImplemented for graph export and generic storage; invalid inputs are distinct.
// Determinism: Stable vertex order and strict distance improvement.
// Complexity: O(V^3) time, O(V^2) storage per closure.
// Notes: The in-place facade currently requires Dense storage.
// AI-Hints: Do not convert derived reachability into original graph edges.
func TestCoverageMetricClosureZeroWeightAndReplay(t *testing.T) {
	g, err := core.NewGraph(core.WithDirected(true), core.WithWeighted())
	if err != nil {
		t.Fatal(err)
	}
	for _, edge := range []struct {
		from, to string
		weight   float64
	}{{"a", "b", 0}, {"b", "c", 3}, {"a", "c", 8}} {
		if _, err = g.AddEdge(edge.from, edge.to, edge.weight); err != nil {
			t.Fatal(err)
		}
	}
	options := matrixMustOptions(t, matrix.WithDirected(), matrix.WithWeighted())
	am, err := matrix.BuildAdjacency(g, options)
	if err != nil {
		t.Fatal(err)
	}
	inf := math.Inf(1)
	want := [][]float64{{0, 0, 3}, {inf, 0, 3}, {inf, inf, 0}}
	for run := 0; run < 2; run++ {
		if err = matrix.MetricClosure(am); err != nil {
			t.Fatal(err)
		}
		CompareExact(t, want, am.Mat)
	}
	// Closure-specific absence policy also applies when querying finite reachability.
	neighbors, err := am.Neighbors("a")
	if err != nil || len(neighbors) != 2 || neighbors[0] != "b" || neighbors[1] != "c" {
		t.Fatalf("closure neighbors=%v err=%v", neighbors, err)
	}
	_, err = matrix.AdjacencyToGraph(am)
	AssertErrorIs(t, err, matrix.ErrMatrixNotImplemented)
	built, err := matrix.BuildMetricClosure(g, options)
	if err != nil {
		t.Fatal(err)
	}
	CompareExact(t, want, built.Mat)
	built.Mat = &coverageMatrix{Matrix: built.Mat}
	AssertErrorIs(t, matrix.MetricClosure(built), matrix.ErrMatrixNotImplemented)
	AssertErrorIs(t, matrix.MetricClosure(nil), matrix.ErrNilMatrix)
}

// TestCoverageMetricClosureClassicAndFailure verifies classic initialization and cycle propagation.
// It uses a complete classic graph so no invalid +Inf write policy is assumed on default storage.
// Implementation: Close a finite two-node classic adjacency, then exercise both closure facades
// with a real negative loop and a raw invalid diagonal on an Inf-enabled adjacency.
// Behavior highlights: A negative cycle is not hidden by diagonal initialization.
// Inputs: Complete two-node graph, negative self-loop graph, and dirty public adjacency storage.
// Returns: Test assertions only.
// Errors: ErrNegativeCycle and ErrNaNInf; failed allocating builders publish nil.
// Determinism: Fixed topology and finite inputs except the explicit dirty-data case.
// Complexity: O(V^3) time and O(V^2) fixture space.
// Notes: Classic sparse adjacency requires an explicit distance-capable storage policy.
// AI-Hints: Use BuildMetricClosure when creating a new distance artifact from topology.
func TestCoverageMetricClosureClassicAndFailure(t *testing.T) {
	g, err := core.NewGraph(core.WithDirected(true), core.WithWeighted(), core.WithLoops())
	if err != nil {
		t.Fatal(err)
	}
	if _, err = g.AddEdge("a", "b", 2); err != nil {
		t.Fatal(err)
	}
	if _, err = g.AddEdge("b", "a", 3); err != nil {
		t.Fatal(err)
	}
	opts := matrixMustOptions(t, matrix.WithDirected(), matrix.WithWeighted(), matrix.WithAllowLoops())
	am, err := matrix.NewAdjacencyMatrix(g, opts)
	if err != nil {
		t.Fatal(err)
	}
	if err = matrix.MetricClosure(am); err != nil {
		t.Fatal(err)
	}
	CompareExact(t, [][]float64{{0, 2}, {3, 0}}, am.Mat)
	// A fresh classic adjacency must reject an invalid diagonal during initialization.
	am, err = matrix.NewAdjacencyMatrix(g, opts)
	if err != nil {
		t.Fatal(err)
	}
	am.Mat = coverageRawDense(t, 2, 2, []float64{math.NaN(), 2, 3, 0})
	AssertErrorIs(t, matrix.MetricClosure(am), matrix.ErrNaNInf)
	if _, err = g.AddEdge("a", "a", -1); err != nil {
		t.Fatal(err)
	}
	am, err = matrix.NewAdjacencyMatrix(g, opts)
	if err != nil {
		t.Fatal(err)
	}
	AssertErrorIs(t, matrix.MetricClosure(am), matrix.ErrNegativeCycle)
	closed, err := matrix.BuildMetricClosure(g, opts)
	AssertErrorIs(t, err, matrix.ErrNegativeCycle)
	if closed != nil {
		t.Fatal("failed closure builder published adjacency")
	}
	// Build an actual zero-preserving snapshot to exercise the Inf-diagonal normalization path.
	zeroGraph, err := core.NewGraph(core.WithDirected(true), core.WithWeighted())
	if err != nil {
		t.Fatal(err)
	}
	if _, err = zeroGraph.AddEdge("a", "b", 0); err != nil {
		t.Fatal(err)
	}
	am, err = matrix.BuildAdjacency(zeroGraph, matrixMustOptions(t, matrix.WithDirected(), matrix.WithPreserveZeroWeights()))
	if err != nil {
		t.Fatal(err)
	}
	// Preserve the +Inf policy while explicitly admitting a NaN for the validation probe.
	dirty, err := matrix.NewPreparedDense(2, 2, matrix.WithAllowInfDistances(), matrix.WithNoValidateNaNInf())
	if err != nil {
		t.Fatal(err)
	}
	MustFillRowMajor(t, dirty, []float64{math.NaN(), 0, math.Inf(1), math.Inf(1)})
	am.Mat = dirty
	AssertErrorIs(t, matrix.MetricClosure(am), matrix.ErrNaNInf)
}
