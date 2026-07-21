// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix_test contains unit tests for BuildDenseAdjacency and BuildDenseIncidence
// functions in the matrix package, ensuring compliance with expected behavior
// under various Options configurations.
package matrix_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/matrix"
)

// --- Adjacency tests ---

// TestBuildDenseAdjacency_EmptyVertices validates that empty vertices are a valid degenerate case.
func TestBuildDenseAdjacency_EmptyVertices(t *testing.T) {
	mOpts, _ := matrix.NewMatrixOptions()
	idx, mat, err := matrix.BuildDenseAdjacency([]string{}, nil, mOpts)
	if err != nil {
		t.Fatalf("BuildDenseAdjacency(empty): %v", err)
	}
	if len(idx) != 0 {
		t.Fatalf("idx size: got %d, want 0", len(idx))
	}
	if mat.Rows() != 0 || mat.Cols() != 0 {
		t.Fatalf("shape: got %dx%d, want 0x0", mat.Rows(), mat.Cols())
	}
}

// TestBuildDenseAdjacency_NilEdges ensures nil edges slice is treated as no edges, producing zero matrix.
func TestBuildDenseAdjacency_NilEdges(t *testing.T) {
	vertices := []string{"A", "B"}
	mOpts, _ := matrix.NewMatrixOptions()
	idx, mat, err := matrix.BuildDenseAdjacency(vertices, nil, mOpts)
	if err != nil {
		t.Fatalf("BuildDenseAdjacency: %v", err)
	}
	if len(idx) != 2 {
		t.Fatalf("idx size: got %d, want 2", len(idx))
	}
	iA, iB := idx["A"], idx["B"]
	if got := MustAt(t, mat, iA, iB); got != 0.0 {
		t.Fatalf("A->B: got %v, want 0", got)
	}
	if got := MustAt(t, mat, iB, iA); got != 0.0 {
		t.Fatalf("B->A: got %v, want 0", got)
	}
	// Diagonal forced to 0
	if got := MustAt(t, mat, iA, iA); got != 0.0 {
		t.Fatalf("A->A: got %v, want 0", got)
	}
	if got := MustAt(t, mat, iB, iB); got != 0.0 {
		t.Fatalf("B->B: got %v, want 0", got)
	}
}

// TestBuildDenseAdjacency_DirectedVsUndirected tests correct placement of edge weights.
func TestBuildDenseAdjacency_DirectedVsUndirected(t *testing.T) {
	vertices := []string{"A", "B"}
	edges := []*core.Edge{{From: "A", To: "B", Weight: 5}}

	// Directed, unweighted (default weight=1)

	opts, _ := matrix.NewMatrixOptions(matrix.WithDirected())
	idx, mat, err := matrix.BuildDenseAdjacency(vertices, edges, opts)
	if err != nil {
		t.Fatalf("BuildDenseAdjacency directed: %v", err)
	}
	iA, iB := idx["A"], idx["B"]
	if got := MustAt(t, mat, iA, iB); got != 1.0 {
		t.Fatalf("directed A->B: got %v, want 1", got)
	}
	if got := MustAt(t, mat, iB, iA); got != 0.0 {
		t.Fatalf("directed B->A: got %v, want 0", got)
	}

	// Undirected, weighted
	opts, _ = matrix.NewMatrixOptions(matrix.WithWeighted())
	idx2, mat2, err := matrix.BuildDenseAdjacency(vertices, edges, opts)
	if err != nil {
		t.Fatalf("BuildDenseAdjacency undirected weighted: %v", err)
	}
	iA2, iB2 := idx2["A"], idx2["B"]
	if got := MustAt(t, mat2, iA2, iB2); got != 5.0 {
		t.Fatalf("undirected A-B: got %v, want 5", got)
	}
	if got := MustAt(t, mat2, iB2, iA2); got != 5.0 {
		t.Fatalf("undirected B-A: got %v, want 5", got)
	}
}

// TestBuildDenseAdjacency_MultiEdgeCollapse tests AllowMulti option handling.
func TestBuildDenseAdjacency_MultiEdgeCollapse(t *testing.T) {
	vertices := []string{"A", "B"}
	edges := []*core.Edge{
		{From: "A", To: "B", Weight: 2},
		{From: "A", To: "B", Weight: 3},
	}
	iA, iB := 0, 1 // consistent with vertices order

	// AllowMulti=true (default), weighted: second overwrites first
	opts, _ := matrix.NewMatrixOptions(matrix.WithWeighted(), matrix.WithAllowMulti())
	_, mat, err := matrix.BuildDenseAdjacency(vertices, edges, opts)
	if err != nil {
		t.Fatalf("BuildDenseAdjacency allow multi: %v", err)
	}
	if got := MustAt(t, mat, iA, iB); got != 3.0 {
		t.Fatalf("allow multi A->B: got %v, want 3", got)
	}

	// AllowMulti=false, weighted: first weight only
	opts, _ = matrix.NewMatrixOptions(matrix.WithWeighted(), matrix.WithDisallowMulti())
	_, mat2, err := matrix.BuildDenseAdjacency(vertices, edges, opts)
	if err != nil {
		t.Fatalf("BuildDenseAdjacency disallow multi: %v", err)
	}
	if got := MustAt(t, mat2, iA, iB); got != 2.0 {
		t.Fatalf("disallow multi A->B: got %v, want 2", got)
	}
}

// TestBuildDenseAdjacency_Loops tests AllowLoops option.
func TestBuildDenseAdjacency_Loops(t *testing.T) {
	vertices := []string{"A"}
	edges := []*core.Edge{{From: "A", To: "A", Weight: 7}}

	// AllowLoops=false (default)
	opts, _ := matrix.NewMatrixOptions()
	idx, mat, err := matrix.BuildDenseAdjacency(vertices, edges, opts)
	if err != nil {
		t.Fatalf("BuildDenseAdjacency no loops: %v", err)
	}
	iA := idx["A"]
	if got := MustAt(t, mat, iA, iA); got != 0.0 {
		t.Fatalf("no-loops A->A: got %v, want 0", got)
	}

	// AllowLoops=true, weighted
	opts, _ = matrix.NewMatrixOptions(matrix.WithAllowLoops(), matrix.WithWeighted())
	idx2, mat2, err := matrix.BuildDenseAdjacency(vertices, edges, opts)
	if err != nil {
		t.Fatalf("BuildDenseAdjacency loops weighted: %v", err)
	}
	iA2 := idx2["A"]
	if got := MustAt(t, mat2, iA2, iA2); got != 7.0 {
		t.Fatalf("loops A->A: got %v, want 7", got)
	}
}

// TestBuildDenseAdjacency_MetricClosure verifies that metric-closure/APSP
// builds a proper distance matrix using +Inf as "no path" and finite
// distances for reachable pairs, without requiring callers to disable
// NaN/Inf validation manually.
func TestBuildDenseAdjacency_MetricClosure(t *testing.T) {
	t.Parallel()

	vertices := []string{"A", "B", "C", "D"}
	edges := []*core.Edge{
		{From: "A", To: "B", Weight: 1},
		{From: "B", To: "C", Weight: 1},
		// "D" is intentionally unreachable from "A", "B", "C".
	}

	// Weighted + MetricClosure: APSP is computed over edge weights,
	// +Inf is used for unreachable pairs, diag is forced to 0.
	opts, _ := matrix.NewMatrixOptions(
		matrix.WithWeighted(),
		matrix.WithMetricClosure(),
	)

	idx, mat, err := matrix.BuildDenseAdjacency(vertices, edges, opts)
	if err != nil {
		t.Fatalf("BuildDenseAdjacency metric: %v", err)
	}

	iA := idx["A"]
	iC := idx["C"]
	iD := idx["D"]

	// A->C must have finite shortest-path distance 2.0 (A→B→C).
	if got := MustAt(t, mat, iA, iC); got != 2.0 {
		t.Fatalf("distance A->C: got %v, want 2", got)
	}

	// All diagonals must be 0 (self-distance).
	for v, i := range idx {
		if got := MustAt(t, mat, i, i); got != 0.0 {
			t.Fatalf("diag %s->%s: got %v, want 0", v, v, got)
		}
	}

	// Unreachable vertices must have +Inf distance from A.
	if got := MustAt(t, mat, iA, iD); !math.IsInf(got, +1) {
		t.Fatalf("distance A->D: got %v, want +Inf (unreachable)", got)
	}
}

func TestBuildDenseAdjacency_MetricClosureRejectsNaNEdgeWeight(t *testing.T) {
	t.Parallel()

	vertices := []string{"A", "B"}
	edges := []*core.Edge{
		{ID: "e1", From: "A", To: "B", Weight: math.NaN(), Directed: true},
	}

	opts, err := matrix.NewMatrixOptions(
		matrix.WithDirected(),
		matrix.WithWeighted(),
		matrix.WithMetricClosure(),
		matrix.WithNoValidateNaNInf(),
	)
	if err != nil {
		t.Fatalf("NewMatrixOptions: %v", err)
	}

	_, _, err = matrix.BuildDenseAdjacency(vertices, edges, opts)
	if !errors.Is(err, matrix.ErrInvalidWeight) {
		t.Fatalf("BuildDenseAdjacency(metric closure NaN): got %v, want ErrInvalidWeight", err)
	}
}

// TestBuildDenseAdjacency_PreserveAllZeroWeights verifies the explicit P3 escape hatch:
// WithPreserveZeroWeights forces +Inf absence encoding even when every real edge has weight 0.
func TestBuildDenseAdjacency_PreserveAllZeroWeights(t *testing.T) {
	t.Parallel()

	vertices := []string{"A", "B", "C"}
	edges := []*core.Edge{
		{ID: "e1", From: "A", To: "B", Weight: 0, Directed: true},
	}

	opts, err := matrix.NewMatrixOptions(
		matrix.WithDirected(),
		matrix.WithWeighted(),
		matrix.WithPreserveZeroWeights(),
	)
	if err != nil {
		t.Fatalf("NewMatrixOptions: %v", err)
	}

	idx, mat, err := matrix.BuildDenseAdjacency(vertices, edges, opts)
	if err != nil {
		t.Fatalf("BuildDenseAdjacency: %v", err)
	}

	got, err := mat.At(idx["A"], idx["B"])
	if err != nil {
		t.Fatalf("At(A,B): %v", err)
	}
	if got != 0 {
		t.Fatalf("preserved zero edge A->B: got %v, want 0", got)
	}

	absent, err := mat.At(idx["A"], idx["C"])
	if err != nil {
		t.Fatalf("At(A,C): %v", err)
	}
	if !math.IsInf(absent, +1) {
		t.Fatalf("missing edge A->C: got %v, want +Inf", absent)
	}
}

// TestBuildDenseAdjacency_InvalidWeight_NaNOrInf ensures that any attempt
// to use NaN or ±Inf as an edge weight in weighted mode is rejected with
// ErrInvalidWeight before the value ever reaches the Dense matrix.
func TestBuildDenseAdjacency_InvalidWeight_NaNOrInf(t *testing.T) {
	t.Parallel()

	vertices := []string{"A", "B"}
	opts, _ := matrix.NewMatrixOptions(matrix.WithWeighted())

	cases := []struct {
		name  string
		edges []*core.Edge
	}{
		{
			name: "NaN",
			edges: []*core.Edge{
				{From: "A", To: "B", Weight: math.NaN()},
			},
		},
		{
			name: "InfPos",
			edges: []*core.Edge{
				{From: "A", To: "B", Weight: math.Inf(+1)},
			},
		},
		{
			name: "InfNeg",
			edges: []*core.Edge{
				{From: "A", To: "B", Weight: math.Inf(-1)},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			_, _, err := matrix.BuildDenseAdjacency(vertices, tc.edges, opts)
			if !errors.Is(err, matrix.ErrInvalidWeight) {
				t.Fatalf("%s: want ErrInvalidWeight, got %v", tc.name, err)
			}
		})
	}
}

// Unknown vertex must surface ErrUnknownVertex.
func TestBuildDenseAdjacency_UnknownVertex(t *testing.T) {
	vertices := []string{"A"}
	edges := []*core.Edge{{From: "A", To: "B", Weight: 1}} // "B" not in vertices
	mOpts, _ := matrix.NewMatrixOptions(matrix.WithWeighted())
	_, _, err := matrix.BuildDenseAdjacency(vertices, edges, mOpts)
	if !errors.Is(err, matrix.ErrUnknownVertex) {
		t.Fatalf("want ErrUnknownVertex, got %v", err)
	}
}

// --- Incidence tests ---

// TestBuildDenseIncidence_EmptyVertices validates that empty vertices are a valid degenerate case.
func TestBuildDenseIncidence_EmptyVertices(t *testing.T) {
	mOpts, _ := matrix.NewMatrixOptions()
	idx, cols, mat, err := matrix.BuildDenseIncidence([]string{}, nil, mOpts)
	if err != nil {
		t.Fatalf("BuildDenseIncidence(empty): %v", err)
	}
	if len(idx) != 0 {
		t.Fatalf("idx size: got %d, want 0", len(idx))
	}
	if len(cols) != 0 {
		t.Fatalf("cols size: got %d, want 0", len(cols))
	}
	if mat.Rows() != 0 || mat.Cols() != 0 {
		t.Fatalf("shape: got %dx%d, want 0x0", mat.Rows(), mat.Cols())
	}
}

// TestBuildDenseIncidence_NilEdges ensures nil edges yields zero-column matrix.
func TestBuildDenseIncidence_NilEdges(t *testing.T) {
	vertices := []string{"A", "B"}
	mOpts, _ := matrix.NewMatrixOptions()
	idx, cols, mat, err := matrix.BuildDenseIncidence(vertices, nil, mOpts)
	if err != nil {
		t.Fatalf("BuildDenseIncidence: %v", err)
	}
	if len(idx) != 2 {
		t.Fatalf("idx size: got %d, want 2", len(idx))
	}
	if len(cols) != 0 {
		t.Fatalf("cols size: got %d, want 0", len(cols))
	}
	if mat.Cols() != 0 {
		t.Fatalf("mat.Cols: got %d, want 0", mat.Cols())
	}
}

// TestBuildDenseIncidence_DirectedVsUndirected tests incidence entries for directed and undirected.
func TestBuildDenseIncidence_DirectedVsUndirected(t *testing.T) {
	vertices := []string{"A", "B"}
	edges := []*core.Edge{{From: "A", To: "B", Weight: 0}}

	// Directed
	opts, _ := matrix.NewMatrixOptions(matrix.WithDirected())
	idxD, colsD, matD, err := matrix.BuildDenseIncidence(vertices, edges, opts)
	if err != nil {
		t.Fatalf("BuildDenseIncidence directed: %v", err)
	}
	if len(colsD) != 1 {
		t.Fatalf("directed cols: got %d, want 1", len(colsD))
	}
	iA, iB := idxD["A"], idxD["B"]
	if got := MustAt(t, matD, iA, 0); got != -1.0 {
		t.Fatalf("directed A row, col0: got %v, want -1", got)
	}
	if got := MustAt(t, matD, iB, 0); got != +1.0 {
		t.Fatalf("directed B row, col0: got %v, want +1", got)
	}

	// Undirected
	opts, _ = matrix.NewMatrixOptions()
	idxU, colsU, matU, err := matrix.BuildDenseIncidence(vertices, edges, opts)
	if err != nil {
		t.Fatalf("BuildDenseIncidence undirected: %v", err)
	}
	if len(colsU) != 1 {
		t.Fatalf("undirected cols: got %d, want 1", len(colsU))
	}
	if got := MustAt(t, matU, idxU["A"], 0); got != 1.0 {
		t.Fatalf("undirected A row: got %v, want +1", got)
	}
	if got := MustAt(t, matU, idxU["B"], 0); got != 1.0 {
		t.Fatalf("undirected B row: got %v, want +1", got)
	}
}

// TestBuildDenseIncidence_MultiEdgeCollapse tests collapse behavior for incidence.
func TestBuildDenseIncidence_MultiEdgeCollapse(t *testing.T) {
	vertices := []string{"A", "B"}
	edges := []*core.Edge{
		{From: "A", To: "B", Weight: 0},
		{From: "A", To: "B", Weight: 0},
	}

	// AllowMulti=true (default)
	opts, _ := matrix.NewMatrixOptions(matrix.WithAllowMulti())
	_, cols, _, err := matrix.BuildDenseIncidence(vertices, edges, opts)
	if err != nil {
		t.Fatalf("BuildDenseIncidence allow multi: %v", err)
	}
	if len(cols) != 2 {
		t.Fatalf("allow multi cols: got %d, want 2", len(cols))
	}

	// AllowMulti=false
	opts, _ = matrix.NewMatrixOptions(matrix.WithDisallowMulti())
	_, cols2, _, err := matrix.BuildDenseIncidence(vertices, edges, opts)
	if err != nil {
		t.Fatalf("BuildDenseIncidence disallow multi: %v", err)
	}
	if len(cols2) != 1 {
		t.Fatalf("disallow multi cols: got %d, want 1", len(cols2))
	}
}

// TestBuildDenseIncidence_Loops tests AllowLoops policy, including directed loop skip and undirected +2.
func TestBuildDenseIncidence_Loops(t *testing.T) {
	// No loops allowed
	vertices := []string{"A"}
	edges := []*core.Edge{{From: "A", To: "A", Weight: 0}}

	opts, _ := matrix.NewMatrixOptions() // AllowLoops=false
	_, cols0, _, err := matrix.BuildDenseIncidence(vertices, edges, opts)
	if err != nil {
		t.Fatalf("BuildDenseIncidence no loops: %v", err)
	}
	if len(cols0) != 0 {
		t.Fatalf("no loops cols: got %d, want 0", len(cols0))
	}

	// Undirected + AllowLoops=true ⇒ +2 in the single row
	opts, _ = matrix.NewMatrixOptions(matrix.WithAllowLoops())
	_, colsU, matU, err := matrix.BuildDenseIncidence(vertices, edges, opts)
	if err != nil {
		t.Fatalf("BuildDenseIncidence undirected loop: %v", err)
	}
	if len(colsU) != 1 || matU.Cols() != 1 {
		t.Fatalf("undirected loop shape: cols=%d, want 1", matU.Cols())
	}
	if got := MustAt(t, matU, 0, 0); got != 2.0 {
		t.Fatalf("undirected loop value: got %v, want 2", got)
	}

	// Directed + AllowLoops=true ⇒ column is skipped
	opts, _ = matrix.NewMatrixOptions(matrix.WithDirected(), matrix.WithAllowLoops())
	_, colsD, matD, err := matrix.BuildDenseIncidence(vertices, edges, opts)
	if err != nil {
		t.Fatalf("BuildDenseIncidence directed loop: %v", err)
	}
	if len(colsD) != 0 || matD.Cols() != 0 {
		t.Fatalf("directed loop should be skipped: got cols=%d", len(colsD))
	}
}

// Unknown vertex must surface ErrUnknownVertex for incidence.
func TestBuildDenseIncidence_UnknownVertex(t *testing.T) {
	vertices := []string{"A"}
	edges := []*core.Edge{{From: "A", To: "B", Weight: 0}} // "B" not known
	mOpts, _ := matrix.NewMatrixOptions()
	_, _, _, err := matrix.BuildDenseIncidence(vertices, edges, mOpts)
	if !errors.Is(err, matrix.ErrUnknownVertex) {
		t.Fatalf("want ErrUnknownVertex, got %v", err)
	}
}
