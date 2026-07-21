// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix_test provides comprehensive unit tests for incidence-matrix wrappers,
// using stdlib only. All tests are deterministic and table/parallel where applicable.
package matrix_test

import (
	"errors"
	"fmt"
	"testing"

	"github.com/lvlath/go/builder"
	"github.com/lvlath/go/core"
	"github.com/lvlath/go/matrix"
)

// --- helpers ---

// countRowSigns returns counts of non-zero, negative and positive entries in row.
func countRowSigns(row []float64) (nonZero, neg, pos int) {
	for _, v := range row {
		if v != 0 {
			nonZero++
		}
		if v < 0 {
			neg++
		}
		if v > 0 {
			pos++
		}
	}

	return
}

// --- tests ---

// TestIncidence_Blueprint validates constructor guards and basic shape.
func TestIncidence_Blueprint(t *testing.T) {
	t.Parallel()

	// nil graph ⇒ ErrGraphNil
	mOpts, _ := matrix.NewMatrixOptions()
	if im, err := matrix.NewIncidenceMatrix(nil, mOpts); !errors.Is(err, matrix.ErrGraphNil) || im != nil {
		t.Fatalf("nil graph: want ErrGraphNil, got im=%v err=%v", im, err)
	}

	// complete undirected graph of V vertices
	g, err := builder.BuildGraph(
		[]core.GraphOption{core.WithWeighted()}, // weights ignored by incidence, but allowed
		[]builder.Option{builder.WithSymbNumb("v")},
		builder.Complete(V),
	)
	if err != nil {
		t.Fatalf("BuildGraph: %v", err)
	}

	im, err := matrix.NewIncidenceMatrix(g, mOpts)
	if err != nil {
		t.Fatalf("NewIncidenceMatrix: %v", err)
	}
	if im == nil {
		t.Fatalf("NewIncidenceMatrix returned nil")
	}

	if got, err := im.VertexCount(); err != nil || got != V {
		t.Fatalf("VertexCount: got (%d,%v), want (%d,nil)", got, err, V)
	}
	if got, err := im.EdgeCount(); err != nil || got != EComplete {
		t.Fatalf("EdgeCount: got (%d,%v), want (%d,nil)", got, err, EComplete)
	}
}

// TestIncidence_EmptyGraph_Degenerate validates that empty graphs are a valid degenerate case.
func TestIncidence_EmptyGraph_Degenerate(t *testing.T) {
	t.Parallel()

	g, _ := core.NewGraph()
	mOpts, _ := matrix.NewMatrixOptions()
	im, err := matrix.NewIncidenceMatrix(g, mOpts)
	if err != nil {
		t.Fatalf("NewIncidenceMatrix(empty): %v", err)
	}
	if im == nil {
		t.Fatalf("NewIncidenceMatrix(empty) returned nil")
	}

	// Counts must be 0 without errors.
	if n, err := im.VertexCount(); err != nil || n != 0 {
		t.Fatalf("VertexCount(empty): got (%d,%v), want (0,nil)", n, err)
	}
	if m, err := im.EdgeCount(); err != nil || m != 0 {
		t.Fatalf("EdgeCount(empty): got (%d,%v), want (0,nil)", m, err)
	}

	// Per-vertex query must fail with ErrUnknownVertex (no vertices exist).
	if _, err := im.VertexIncidence("X"); !errors.Is(err, matrix.ErrUnknownVertex) {
		t.Fatalf("VertexIncidence(empty): want ErrUnknownVertex, got %v", err)
	}
}

// Table-driven coverage for per-vertex incidence rows on a path graph.
func TestVertexIncidence_TableDriven(t *testing.T) {
	t.Parallel()

	type scenario struct {
		name       string
		coreOpts   []core.GraphOption
		matrixOpts []matrix.Option
		wantDeg    []int
		wantNeg    []int // for directed
		wantPos    []int // for directed
	}

	tests := []scenario{
		{
			name:       "Undirected_Path",
			coreOpts:   nil, // default undirected
			matrixOpts: nil,
			wantDeg:    []int{1, 2, 2, 2, 2, 2, 2, 1},
		},
		{
			name:       "Directed_Path",
			coreOpts:   []core.GraphOption{core.WithDirected(true)},
			matrixOpts: []matrix.Option{matrix.WithDirected()},
			wantDeg:    []int{1, 2, 2, 2, 2, 2, 2, 1},
			wantNeg:    []int{1, 1, 1, 1, 1, 1, 1, 0}, // outgoing
			wantPos:    []int{0, 1, 1, 1, 1, 1, 1, 1}, // incoming
		},
	}

	for _, sc := range tests {
		t.Run(sc.name, func(t *testing.T) {
			t.Parallel()

			// Build path v0-v1-...-v7
			g, err := builder.BuildGraph(sc.coreOpts, []builder.Option{builder.WithSymbNumb("v")}, builder.Path(V))
			if err != nil {
				t.Fatalf("BuildGraph: %v", err)
			}

			mOpts, _ := matrix.NewMatrixOptions(sc.matrixOpts...)
			im, err := matrix.NewIncidenceMatrix(g, mOpts)
			if err != nil {
				t.Fatalf("NewIncidenceMatrix: %v", err)
			}

			for i := 0; i < V; i++ {
				id := fmt.Sprintf("v%d", i)
				row, err := im.VertexIncidence(id)
				if err != nil {
					t.Fatalf("VertexIncidence(%q): %v", id, err)
				}

				nz, neg, pos := countRowSigns(row)
				if nz != sc.wantDeg[i] {
					t.Fatalf("non-zero count for %q: got %d, want %d", id, nz, sc.wantDeg[i])
				}
				if len(sc.wantNeg) > 0 {
					if neg != sc.wantNeg[i] {
						t.Fatalf("neg count for %q: got %d, want %d", id, neg, sc.wantNeg[i])
					}
					if pos != sc.wantPos[i] {
						t.Fatalf("pos count for %q: got %d, want %d", id, pos, sc.wantPos[i])
					}
				}
			}
		})
	}
}

// Validate EdgeEndpoints: invalid indices error out; valid ones match im.Edges[j].
func TestEdgeEndpoints_Cases(t *testing.T) {
	t.Parallel()

	mOpts, _ := matrix.NewMatrixOptions()
	g, err := builder.BuildGraph([]core.GraphOption{core.WithWeighted()}, []builder.Option{builder.WithSymbNumb("v")}, builder.Path(V))
	if err != nil {
		t.Fatalf("BuildGraph: %v", err)
	}
	im, err := matrix.NewIncidenceMatrix(g, mOpts)
	if err != nil {
		t.Fatalf("NewIncidenceMatrix: %v", err)
	}

	if _, _, err = im.EdgeEndpoints(-1); !errors.Is(err, matrix.ErrDimensionMismatch) {
		t.Fatalf("EdgeEndpoints(-1): want ErrDimensionMismatch, got %v", err)
	}
	eCount, err := im.EdgeCount()
	if err != nil {
		t.Fatalf("EdgeCount: %v", err)
	}
	if _, _, err = im.EdgeEndpoints(eCount); !errors.Is(err, matrix.ErrDimensionMismatch) {
		t.Fatalf("EdgeEndpoints(Cols): want ErrDimensionMismatch, got %v", err)
	}

	for j := 0; j < eCount; j++ {
		from, to, err := im.EdgeEndpoints(j)
		if err != nil {
			t.Fatalf("EdgeEndpoints(%d): %v", j, err)
		}
		e := im.Edges[j]
		if e.From != from || e.To != to {
			t.Fatalf("edge endpoints mismatch at col %d: got (%s,%s), want (%s,%s)", j, from, to, e.From, e.To)
		}
	}
}

// Idempotency: repeated construction yields identical indices, edges and cells.
func TestIncidence_Idempotency(t *testing.T) {
	t.Parallel()

	g, err := builder.BuildGraph([]core.GraphOption{core.WithWeighted()}, []builder.Option{builder.WithSymbNumb("v")}, builder.Path(V))
	if err != nil {
		t.Fatalf("BuildGraph: %v", err)
	}
	mOpts, _ := matrix.NewMatrixOptions()
	im1, err1 := matrix.NewIncidenceMatrix(g, mOpts)
	im2, err2 := matrix.NewIncidenceMatrix(g, mOpts)
	if err1 != nil || err2 != nil {
		t.Fatalf("NewIncidenceMatrix errs: %v %v", err1, err2)
	}

	// VertexIndex maps must be identical
	if len(im1.VertexIndex) != len(im2.VertexIndex) {
		t.Fatalf("VertexIndex size mismatch: %d vs %d", len(im1.VertexIndex), len(im2.VertexIndex))
	}
	for id, i := range im1.VertexIndex {
		j, ok := im2.VertexIndex[id]
		if !ok || j != i {
			t.Fatalf("VertexIndex entry mismatch for %q: im1=%d im2=%d ok=%v", id, i, j, ok)
		}
	}

	// Edges slices equal (same order, same endpoints)
	if len(im1.Edges) != len(im2.Edges) {
		t.Fatalf("Edges slice size mismatch: %d vs %d", len(im1.Edges), len(im2.Edges))
	}
	for k := range im1.Edges {
		e1, e2 := im1.Edges[k], im2.Edges[k]
		if e1.From != e2.From || e1.To != e2.To {
			t.Fatalf("Edges[%d] mismatch: (%s,%s) vs (%s,%s)", k, e1.From, e1.To, e2.From, e2.To)
		}
	}

	// Cell-by-cell equality
	rows, err := im1.VertexCount()
	if err != nil {
		t.Fatalf("VertexCount im1: %v", err)
	}
	cols, err := im1.EdgeCount()
	if err != nil {
		t.Fatalf("EdgeCount im1: %v", err)
	}
	if r2, _ := im2.VertexCount(); r2 != rows {
		t.Fatalf("row mismatch: im1=%d im2=%d", rows, r2)
	}
	if c2, _ := im2.EdgeCount(); c2 != cols {
		t.Fatalf("col mismatch: im1=%d im2=%d", cols, c2)
	}
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			v1, err := im1.Mat.At(i, j)
			if err != nil {
				t.Fatalf("im1.At(%d,%d): %v", i, j, err)
			}
			v2, err := im2.Mat.At(i, j)
			if err != nil {
				t.Fatalf("im2.At(%d,%d): %v", i, j, err)
			}
			if v1 != v2 {
				t.Fatalf("cell mismatch at (%d,%d): %v vs %v", i, j, v1, v2)
			}
		}
	}
}

// Multi-edges: first-edge-wins when DisallowMulti; otherwise both columns are present.
func TestIncidence_MultiEdges_FirstEdgeWins(t *testing.T) {
	t.Parallel()

	// Prepare a directed graph allowing multi-edges, two identical parallel edges v0->v1
	g, _ := core.NewGraph(core.WithDirected(true), core.WithWeighted(), core.WithMultiEdges())
	_ = g.AddVertex("v0")
	_ = g.AddVertex("v1")
	if _, err := g.AddEdge("v0", "v1", 10); err != nil {
		t.Fatalf("AddEdge 10: %v", err)
	}
	if _, err := g.AddEdge("v0", "v1", 99); err != nil {
		t.Fatalf("AddEdge 99: %v", err)
	}

	// DisallowMulti ⇒ only one column
	mOpts, _ := matrix.NewMatrixOptions(matrix.WithDirected(), matrix.WithDisallowMulti())
	imDis, err := matrix.NewIncidenceMatrix(g, mOpts)
	if err != nil {
		t.Fatalf("NewIncidenceMatrix disallow: %v", err)
	}
	if got, err := imDis.EdgeCount(); err != nil || got != 1 {
		t.Fatalf("EdgeCount (disallow): got %d (err=%v), want 1", got, err)
	}

	// First-edge-wins: the surviving column must correspond to the first inserted edge (weight=10).
	if len(imDis.Edges) != 1 {
		t.Fatalf("Edges (disallow): got len=%d, want 1", len(imDis.Edges))
	}
	if imDis.Edges[0].Weight != 10 {
		t.Fatalf("first-edge-wins: got weight=%v, want 10", imDis.Edges[0].Weight)
	}

	// AllowMulti ⇒ two columns
	mOpts, _ = matrix.NewMatrixOptions(matrix.WithDirected(), matrix.WithAllowMulti())
	imAllow, err := matrix.NewIncidenceMatrix(g, mOpts)
	if err != nil {
		t.Fatalf("NewIncidenceMatrix allow: %v", err)
	}
	if got, err := imAllow.EdgeCount(); err != nil || got != 2 {
		t.Fatalf("EdgeCount (allow): got %d (err=%v), want 2", got, err)
	}
	// First-edge-wins: the surviving column must correspond to the first inserted edge (weight=10).
	if len(imDis.Edges) != 1 {
		t.Fatalf("Edges (disallow): got len=%d, want 1", len(imDis.Edges))
	}
	if imDis.Edges[0].Weight != 10 {
		t.Fatalf("first-edge-wins: got weight=%v, want 10", imDis.Edges[0].Weight)
	}
}

// Undirected self-loop must be represented as +2 in the incident row.
func TestIncidence_UndirectedLoop_Plus2(t *testing.T) {
	t.Parallel()

	g, _ := core.NewGraph(core.WithLoops(), core.WithWeighted())
	_ = g.AddVertex("x")
	if _, err := g.AddEdge("x", "x", 1); err != nil {
		t.Fatalf("AddEdge loop: %v", err)
	}

	mOpts, _ := matrix.NewMatrixOptions(matrix.WithAllowLoops())
	im, err := matrix.NewIncidenceMatrix(g, mOpts)
	if err != nil {
		t.Fatalf("NewIncidenceMatrix: %v", err)
	}
	rc, err := im.VertexCount()
	if err != nil {
		t.Fatalf("VertexCount: %v", err)
	}
	cc, err := im.EdgeCount()
	if err != nil {
		t.Fatalf("EdgeCount: %v", err)
	}
	if rc != 1 || cc != 1 {
		t.Fatalf("shape: got %dx%d, want 1x1", rc, cc)
	}
	row, err := im.VertexIncidence("x")
	if err != nil {
		t.Fatalf("VertexIncidence: %v", err)
	}
	if len(row) != 1 || row[0] != 2.0 {
		t.Fatalf("undirected loop row: got %v, want [2.0]", row)
	}
}

// Directed self-loop must be skipped (no zero column materialized).
func TestIncidence_DirectedLoop_SkippedColumn(t *testing.T) {
	t.Parallel()

	g, _ := core.NewGraph(core.WithDirected(true), core.WithLoops(), core.WithWeighted())
	_ = g.AddVertex("x")
	if _, err := g.AddEdge("x", "x", 1); err != nil {
		t.Fatalf("AddEdge loop: %v", err)
	}

	mOpts, _ := matrix.NewMatrixOptions(matrix.WithDirected())
	im, err := matrix.NewIncidenceMatrix(g, mOpts)
	if err != nil {
		t.Fatalf("NewIncidenceMatrix: %v", err)
	}
	if ec, err := im.EdgeCount(); err != nil || ec != 0 {
		t.Fatalf("EdgeCount: got %d (err=%v), want 0 (directed loop skipped)", ec, err)
	}
	row, err := im.VertexIncidence("x")
	if err != nil {
		t.Fatalf("VertexIncidence: %v", err)
	}
	if len(row) != 0 {
		t.Fatalf("row length: got %d, want 0 (no columns)", len(row))
	}
}

// Weights are ignored: entries must be -1 and +1, not the numeric weight.
func TestIncidence_WeightsIgnored(t *testing.T) {
	t.Parallel()

	g, _ := core.NewGraph(core.WithDirected(true), core.WithWeighted())
	_ = g.AddVertex("a")
	_ = g.AddVertex("b")
	if _, err := g.AddEdge("a", "b", 7); err != nil {
		t.Fatalf("AddEdge: %v", err)
	}

	mOpts, _ := matrix.NewMatrixOptions(matrix.WithDirected(), matrix.WithWeighted())
	im, err := matrix.NewIncidenceMatrix(g, mOpts)
	if err != nil {
		t.Fatalf("NewIncidenceMatrix: %v", err)
	}
	if ec, err := im.EdgeCount(); err != nil || ec != 1 {
		t.Fatalf("EdgeCount: got %d (err=%v), want 1", ec, err)
	}

	rowA, err := im.VertexIncidence("a")
	if err != nil {
		t.Fatalf("VertexIncidence(a): %v", err)
	}
	rowB, err := im.VertexIncidence("b")
	if err != nil {
		t.Fatalf("VertexIncidence(b): %v", err)
	}
	if len(rowA) != 1 || len(rowB) != 1 {
		t.Fatalf("row lengths: a=%d b=%d, want 1/1", len(rowA), len(rowB))
	}
	if rowA[0] != -1 || rowB[0] != +1 {
		t.Fatalf("entries: a=%v b=%v, want a=[-1] b=[+1]", rowA, rowB)
	}
}

// Nil receiver behavior: methods must surface ErrNilMatrix.
func TestIncidence_NilReceiver_Errors(t *testing.T) {
	t.Parallel()

	var im *matrix.IncidenceMatrix

	// VertexIncidence on nil ⇒ ErrNilMatrix
	if _, err := im.VertexIncidence("x"); !errors.Is(err, matrix.ErrNilMatrix) {
		t.Fatalf("VertexIncidence on nil: want ErrNilMatrix, got %v", err)
	}
	// EdgeEndpoints on nil ⇒ ErrNilMatrix
	if _, _, err := im.EdgeEndpoints(0); !errors.Is(err, matrix.ErrNilMatrix) {
		t.Fatalf("EdgeEndpoints on nil: want ErrNilMatrix, got %v", err)
	}
	// VertexCount/EdgeCount on nil ⇒ (0, ErrNilMatrix)
	if n, err := im.VertexCount(); n != 0 || !errors.Is(err, matrix.ErrNilMatrix) {
		t.Fatalf("VertexCount on nil: got (%d,%v), want (0,ErrNilMatrix)", n, err)
	}
	if m, err := im.EdgeCount(); m != 0 || !errors.Is(err, matrix.ErrNilMatrix) {
		t.Fatalf("EdgeCount on nil: got (%d,%v), want (0,ErrNilMatrix)", m, err)
	}
}

// Counts must report ErrDimensionMismatch when metadata diverges from matrix shape.
func TestIncidence_Counts_DimensionMismatch(t *testing.T) {
	t.Parallel()

	g, _ := core.NewGraph(core.WithWeighted())
	for _, id := range []string{"v0", "v1", "v2"} {
		_ = g.AddVertex(id)
	}
	_, _ = g.AddEdge("v0", "v1", 1)

	mOpts, _ := matrix.NewMatrixOptions()
	im, err := matrix.NewIncidenceMatrix(g, mOpts)
	if err != nil {
		t.Fatalf("NewIncidenceMatrix: %v", err)
	}

	// Break VertexIndex invariant: append bogus entry; Rows stays the same
	im.VertexIndex["zz"] = len(im.VertexIndex)
	if _, err = im.VertexCount(); !errors.Is(err, matrix.ErrDimensionMismatch) {
		t.Fatalf("VertexCount: want ErrDimensionMismatch after tamper, got %v", err)
	}

	// Restore and break Edges invariant: shrink Edges to be less than Cols
	delete(im.VertexIndex, "zz")
	cols := im.Mat.Cols()
	if cols == 0 {
		t.Fatalf("unexpected zero columns")
	}
	im.Edges = im.Edges[:cols-1]
	if _, err = im.EdgeCount(); !errors.Is(err, matrix.ErrDimensionMismatch) {
		t.Fatalf("EdgeCount: want ErrDimensionMismatch after tamper, got %v", err)
	}
}
