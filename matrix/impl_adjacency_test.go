// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix_test provides comprehensive unit tests for adjacency-matrix wrappers,
// using stdlib only. All tests are deterministic and table/parallel where applicable.
package matrix_test

import (
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/lvlath/go/builder"
	"github.com/lvlath/go/core"
	"github.com/lvlath/go/matrix"
)

// --- helpers ---

// edgesMap collapses edges to a unique (from,to)->w map (last write wins).
func edgesMap(g *core.Graph) map[[2]string]float64 {
	m := map[[2]string]float64{}
	for _, e := range g.Edges() {
		m[[2]string{e.From, e.To}] = e.Weight
	}

	return m
}

// --- tests ---

// TestAdjacency_Blueprint validates constructor guards and basic shape.
func TestAdjacency_Blueprint(t *testing.T) {
	t.Parallel()
	mOpts, _ := matrix.NewMatrixOptions()
	// nil graph ⇒ ErrGraphNil
	if am, err := matrix.NewAdjacencyMatrix(nil, mOpts); !errors.Is(err, matrix.ErrGraphNil) || am != nil {
		t.Fatalf("nil graph: want ErrGraphNil, got am=%v err=%v", am, err)
	}

	// full graph via builder
	g, err := builder.BuildGraph(
		[]core.GraphOption{core.WithWeighted(), core.WithMultiEdges(), core.WithLoops()},
		[]builder.Option{builder.WithSymbNumb("v")},
		builder.Complete(V),
	)
	if err != nil {
		t.Fatalf("BuildGraph: %v", err)
	}

	mOpts, _ = matrix.NewMatrixOptions(
		matrix.WithWeighted(),
		matrix.WithAllowMulti(),
		matrix.WithAllowLoops(),
	)
	am, err := matrix.NewAdjacencyMatrix(g, mOpts)
	if err != nil {
		t.Fatalf("NewAdjacencyMatrix: %v", err)
	}
	if am == nil {
		t.Fatalf("NewAdjacencyMatrix returned nil")
	}
	if n, err := am.VertexCount(); err != nil || n != V {
		t.Fatalf("VertexCount: got (%d,%v), want (%d,nil)", n, err, V)
	}
}

// Table-driven coverage for neighbor enumeration in key modes.
func TestNeighbors_TableDriven(t *testing.T) {
	t.Parallel()

	type scenario struct {
		name       string
		graphOpts  []core.GraphOption
		matrixOpts []matrix.Option
		wantCount  int
	}

	tests := []scenario{
		{
			name:       "Undirected_Unweighted",
			graphOpts:  []core.GraphOption{},
			matrixOpts: []matrix.Option{},
			wantCount:  V - 1,
		},
		{
			name:      "Directed_Weighted",
			graphOpts: []core.GraphOption{core.WithDirected(true), core.WithWeighted()},
			matrixOpts: []matrix.Option{
				matrix.WithDirected(),
				matrix.WithWeighted(),
			},
			wantCount: V - 1,
		},
		{
			name:      "WithLoops",
			graphOpts: []core.GraphOption{core.WithLoops()},
			matrixOpts: []matrix.Option{
				matrix.WithAllowLoops(),
			},
			wantCount: V - 1, // neighbors exclude self even if loop exists
		},
	}

	for _, sc := range tests {
		t.Run(sc.name, func(t *testing.T) {
			t.Parallel()

			g, err := builder.BuildGraph(sc.graphOpts, []builder.Option{builder.WithSymbNumb("v")}, builder.Complete(V))
			if err != nil {
				t.Fatalf("BuildGraph: %v", err)
			}
			mOpts, _ := matrix.NewMatrixOptions(sc.matrixOpts...)
			am, err := matrix.NewAdjacencyMatrix(g, mOpts)
			if err != nil {
				t.Fatalf("NewAdjacencyMatrix: %v", err)
			}

			u := "v0"
			neighbors, err := am.Neighbors(u)
			if err != nil {
				t.Fatalf("Neighbors(%q): %v", u, err)
			}
			if len(neighbors) != sc.wantCount {
				t.Fatalf("neighbors count: got %d, want %d", len(neighbors), sc.wantCount)
			}
			for _, v := range neighbors {
				if _, ok := am.VertexIndex[v]; !ok {
					t.Fatalf("neighbor %q not found in VertexIndex", v)
				}
			}
		})
	}
}

// First-edge-wins for directed graphs when DisallowMulti.
func TestFirstEdgeWins_DisallowMulti_Directed(t *testing.T) {
	t.Parallel()

	g, _ := core.NewGraph(core.WithDirected(true), core.WithWeighted(), core.WithMultiEdges())
	if err := g.AddVertex("v0"); err != nil {
		t.Fatalf("AddVertex: %v", err)
	}
	if err := g.AddVertex("v1"); err != nil {
		t.Fatalf("AddVertex: %v", err)
	}

	// two parallel edges same (u,v): first should win in adjacency (when DisallowMulti)
	if _, err := g.AddEdge("v0", "v1", 10); err != nil {
		t.Fatalf("AddEdge 10: %v", err)
	}
	if _, err := g.AddEdge("v0", "v1", 99); err != nil {
		t.Fatalf("AddEdge 99: %v", err)
	}

	mOpts, _ := matrix.NewMatrixOptions(
		matrix.WithDirected(),
		matrix.WithWeighted(),
		matrix.WithDisallowMulti(),
	)
	am, err := matrix.NewAdjacencyMatrix(g, mOpts)
	if err != nil {
		t.Fatalf("NewAdjacencyMatrix: %v", err)
	}
	i0 := am.VertexIndex["v0"]
	i1 := am.VertexIndex["v1"]
	v01, err := am.Mat.At(i0, i1)
	if err != nil {
		t.Fatalf("At: %v", err)
	}
	if v01 != 10 {
		t.Fatalf("first-edge-wins failed: got %v, want 10", v01)
	}

	// export: keepWeights ⇒ single edge (v0,v1,10)
	g2, err := am.ToGraph(matrix.WithKeepWeights())
	if err != nil {
		t.Fatalf("ToGraph: %v", err)
	}
	em := edgesMap(g2)
	if w, ok := em[[2]string{"v0", "v1"}]; !ok || w != 10 {
		t.Fatalf("export mismatch: presence=%v weight=%g, want weight=10", ok, w)
	}
	if len(g2.Edges()) != 1 {
		t.Fatalf("export edges count: got %d, want 1", len(g2.Edges()))
	}
}

// First-edge-wins for undirected graphs when DisallowMulti.
func TestFirstEdgeWins_DisallowMulti_Undirected(t *testing.T) {
	t.Parallel()

	g, _ := core.NewGraph(core.WithDirected(false), core.WithWeighted(), core.WithMultiEdges())
	_ = g.AddVertex("v0")
	_ = g.AddVertex("v1")

	// simulate conflicting insertion order for pair {v0,v1}
	if _, err := g.AddEdge("v0", "v1", 5); err != nil {
		t.Fatalf("AddEdge 5: %v", err)
	}
	if _, err := g.AddEdge("v1", "v0", 99); err != nil {
		t.Fatalf("AddEdge 99: %v", err)
	}

	mOpts, _ := matrix.NewMatrixOptions(
		matrix.WithUndirected(),
		matrix.WithWeighted(),
		matrix.WithDisallowMulti(),
	)
	am, err := matrix.NewAdjacencyMatrix(g, mOpts)
	if err != nil {
		t.Fatalf("NewAdjacencyMatrix: %v", err)
	}
	i0 := am.VertexIndex["v0"]
	i1 := am.VertexIndex["v1"]
	v01, _ := am.Mat.At(i0, i1)
	v10, _ := am.Mat.At(i1, i0)
	// builder mirrors; both should reflect the first weight (5)
	if v01 != 5 || v10 != 5 {
		t.Fatalf("undirected first-edge-wins failed: A01=%v A10=%v, want 5/5", v01, v10)
	}

	// export undirected ⇒ single unordered edge
	g2, err := am.ToGraph(matrix.WithKeepWeights())
	if err != nil {
		t.Fatalf("ToGraph: %v", err)
	}
	if len(g2.Edges()) != 1 {
		t.Fatalf("export edges count: got %d, want 1", len(g2.Edges()))
	}
	if w := g2.Edges()[0].Weight; w != 5 {
		t.Fatalf("export weight: got %g, want 5", w)
	}
}

// Loops are dropped when DisallowLoops during adjacency build.
func TestLoops_DisallowLoops(t *testing.T) {
	t.Parallel()

	g, _ := core.NewGraph(core.WithLoops(), core.WithWeighted())
	_ = g.AddVertex("v0")
	if _, err := g.AddEdge("v0", "v0", 7); err != nil {
		t.Fatalf("AddEdge loop: %v", err)
	}

	mOpts, _ := matrix.NewMatrixOptions(matrix.WithDisallowLoops())
	am, err := matrix.NewAdjacencyMatrix(g, mOpts)
	if err != nil {
		t.Fatalf("NewAdjacencyMatrix: %v", err)
	}
	i := am.VertexIndex["v0"]
	v, err := am.Mat.At(i, i)
	if err != nil {
		t.Fatalf("At: %v", err)
	}
	if v != 0 {
		t.Fatalf("loop should be dropped: Aii=%v, want 0", v)
	}

	// export should not produce the loop
	g2, err := am.ToGraph()
	if err != nil {
		t.Fatalf("ToGraph: %v", err)
	}
	if len(g2.Edges()) != 0 {
		t.Fatalf("export should have 0 edges, got %d", len(g2.Edges()))
	}
}

// Round-trip preserves IDs and weights (no multi, no loops).
func TestToGraph_RoundTrip_PreserveIDsAndWeights(t *testing.T) {
	t.Parallel()

	g, _ := core.NewGraph(core.WithDirected(true), core.WithWeighted())
	ids := []string{"v0", "v1", "v2", "v3"}
	for _, id := range ids {
		if err := g.AddVertex(id); err != nil {
			t.Fatalf("AddVertex %s: %v", id, err)
		}
	}
	add := func(u, v string, w float64) {
		if _, err := g.AddEdge(u, v, w); err != nil {
			t.Fatalf("AddEdge %s->%s: %v", u, v, err)
		}
	}
	add("v0", "v1", 3)
	add("v1", "v2", 5)
	add("v2", "v0", 7)
	add("v3", "v1", 11)
	add("v0", "v3", 13)

	mOpts, _ := matrix.NewMatrixOptions(matrix.WithDirected(), matrix.WithWeighted())
	am, err := matrix.NewAdjacencyMatrix(g, mOpts)
	if err != nil {
		t.Fatalf("NewAdjacencyMatrix: %v", err)
	}
	g2, err := am.ToGraph() // defaults: keepWeights=true, threshold=0.5
	if err != nil {
		t.Fatalf("ToGraph: %v", err)
	}

	// compare vertex sets
	if len(g.Vertices()) != len(g2.Vertices()) {
		t.Fatalf("vertex count mismatch: %d vs %d", len(g.Vertices()), len(g2.Vertices()))
	}
	// compare edge sets (no multi => unique pairs)
	m1 := edgesMap(g)
	m2 := edgesMap(g2)
	if len(m1) != len(m2) {
		t.Fatalf("edge map size mismatch: %d vs %d", len(m1), len(m2))
	}
	for k, w := range m1 {
		if w2, ok := m2[k]; !ok || w2 != w {
			t.Fatalf("edge mismatch on %v: got %f, want %g (ok=%v)", k, w2, w, ok)
		}
	}
}

// Idempotent build: same graph+opts ⇒ identical indices and cells.
func TestAdjacency_Idempotency(t *testing.T) {
	t.Parallel()

	g, err := builder.BuildGraph([]core.GraphOption{core.WithWeighted()}, []builder.Option{builder.WithSymbNumb("v")}, builder.Complete(V))
	if err != nil {
		t.Fatalf("BuildGraph: %v", err)
	}
	opts, _ := matrix.NewMatrixOptions(matrix.WithWeighted())

	am1, err1 := matrix.NewAdjacencyMatrix(g, opts)
	am2, err2 := matrix.NewAdjacencyMatrix(g, opts)
	if err1 != nil || err2 != nil {
		t.Fatalf("NewAdjacencyMatrix errs: %v %v", err1, err2)
	}

	// indices map equality
	if len(am1.VertexIndex) != len(am2.VertexIndex) {
		t.Fatalf("VertexIndex size mismatch: %d vs %d", len(am1.VertexIndex), len(am2.VertexIndex))
	}
	for id, i := range am1.VertexIndex {
		if j, ok := am2.VertexIndex[id]; !ok || j != i {
			t.Fatalf("VertexIndex entry mismatch for %q: am1=%d am2=%d ok=%v", id, i, j, ok)
		}
	}

	// cell-by-cell equality
	n1, _ := am1.VertexCount()
	n2, _ := am2.VertexCount()
	if n1 != n2 {
		t.Fatalf("VertexCount mismatch: %d vs %d", n1, n2)
	}
	for i := 0; i < n1; i++ {
		for j := 0; j < n1; j++ {
			v1, _ := am1.Mat.At(i, j)
			v2, _ := am2.Mat.At(i, j)
			if v1 != v2 {
				t.Fatalf("A1[%d,%d]=%v A2[%d,%d]=%v mismatch", i, j, v1, i, j, v2)
			}
		}
	}
}

// Error surfaces: unknown vertex and nil receiver for Neighbors.
func TestNeighbors_ErrorCases(t *testing.T) {
	t.Parallel()

	g, err := builder.BuildGraph([]core.GraphOption{}, []builder.Option{builder.WithSymbNumb("v")}, builder.Complete(4))
	if err != nil {
		t.Fatalf("BuildGraph: %v", err)
	}
	mOpts, _ := matrix.NewMatrixOptions()
	am, err := matrix.NewAdjacencyMatrix(g, mOpts)
	if err != nil {
		t.Fatalf("NewAdjacencyMatrix: %v", err)
	}

	_, err = am.Neighbors("unknown")
	if !errors.Is(err, matrix.ErrUnknownVertex) {
		t.Fatalf("Neighbors unknown: want ErrUnknownVertex, got %v", err)
	}

	var nilAM *matrix.AdjacencyMatrix
	_, err = nilAM.Neighbors("v0")
	if !errors.Is(err, matrix.ErrNilMatrix) {
		t.Fatalf("Neighbors on nil receiver: want ErrNilMatrix, got %v", err)
	}
}

// Metric-closure adjacency cannot be exported.
func TestToGraph_MetricClosure_Unsupported(t *testing.T) {
	t.Parallel()

	g, _ := core.NewGraph()
	_ = g.AddVertex("v0")
	_ = g.AddVertex("v1")
	_, _ = g.AddEdge("v0", "v1", 1)

	mOpts, _ := matrix.NewMatrixOptions(matrix.WithMetricClosure())
	am, err := matrix.NewAdjacencyMatrix(g, mOpts)
	if err != nil {
		t.Fatalf("NewAdjacencyMatrix: %v", err)
	}
	if _, err := am.ToGraph(); !errors.Is(err, matrix.ErrMatrixNotImplemented) {
		t.Fatalf("ToGraph metric-closure: want ErrMatrixNotImplemented, got %v", err)
	}
}

// Threshold and binary/keep weights policies during export.
func TestToGraph_ThresholdAndBinary(t *testing.T) {
	t.Parallel()

	g, _ := core.NewGraph(core.WithDirected(true), core.WithWeighted())
	for _, id := range []string{"v0", "v1", "v2"} {
		_ = g.AddVertex(id)
	}
	_, _ = g.AddEdge("v0", "v1", 1) // low
	_, _ = g.AddEdge("v0", "v2", 3) // high

	mOpts, _ := matrix.NewMatrixOptions(matrix.WithDirected(), matrix.WithWeighted())
	am, err := matrix.NewAdjacencyMatrix(g, mOpts)
	if err != nil {
		t.Fatalf("NewAdjacencyMatrix: %v", err)
	}

	// threshold=2, binary ⇒ only v0->v2 with weight 1
	gBin, err := am.ToGraph(matrix.WithEdgeThreshold(2), matrix.WithBinaryWeights())
	//gBin, err := am.ToGraph(matrix.WithEdgeThreshold(2), matrix.WithKeepWeights())
	if err != nil {
		t.Fatalf("ToGraph binary: %v", err)
	}
	em := edgesMap(gBin)
	if len(em) != 1 || em[[2]string{"v0", "v2"}] != 1 {
		t.Fatalf("binary export mismatch: edges=%v", em)
	}

	// threshold=0, keep ⇒ both edges with original weights
	gKeep, err := am.ToGraph(matrix.WithEdgeThreshold(0), matrix.WithKeepWeights())
	if err != nil {
		t.Fatalf("ToGraph keep: %v", err)
	}
	em2 := edgesMap(gKeep)
	if len(em2) != 2 || em2[[2]string{"v0", "v1"}] != 1 || em2[[2]string{"v0", "v2"}] != 3 {
		t.Fatalf("keep export mismatch: edges=%v", em2)
	}
}

// No-edges and single-vertex edge cases.
func TestAdjacency_NoEdges(t *testing.T) {
	t.Parallel()

	g, _ := core.NewGraph()
	for i := 0; i < 5; i++ {
		_ = g.AddVertex(fmt.Sprintf("v%d", i))
	}
	mOpts, _ := matrix.NewMatrixOptions()
	am, err := matrix.NewAdjacencyMatrix(g, mOpts)
	if err != nil {
		t.Fatalf("NewAdjacencyMatrix: %v", err)
	}
	n, err := am.VertexCount()
	if err != nil || n != 5 {
		t.Fatalf("VertexCount: (%d,%v), want (5,nil)", n, err)
	}
	// check zeros
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			v, _ := am.Mat.At(i, j)
			if v != 0 {
				t.Fatalf("A[%d,%d]=%v, want 0", i, j, v)
			}
		}
	}
	// export ⇒ no edges
	g2, err := am.ToGraph()
	if err != nil {
		t.Fatalf("ToGraph: %v", err)
	}
	if len(g2.Edges()) != 0 {
		t.Fatalf("export edges: got %d, want 0", len(g2.Edges()))
	}
}

// TestAdjacency_MixedZeroWeightEdge_PreservedAsFiniteEdge verifies the P3 contract:
// in mixed zero/non-zero weighted adjacency, finite 0 is a real edge weight,
// while absence is represented by +Inf. The test also verifies that Neighbors()
// and ToGraph() do not accidentally drop the zero-weight edge.
func TestAdjacency_MixedZeroWeightEdge_PreservedAsFiniteEdge(t *testing.T) {
	t.Parallel()

	g, _ := core.NewGraph(core.WithDirected(true), core.WithWeighted())

	_, _ = g.AddEdge("A", "B", 0) // real zero-weight edge
	_, _ = g.AddEdge("A", "C", 2) // non-zero edge in the same outgoing row

	mOpts, _ := matrix.NewMatrixOptions(matrix.WithDirected(), matrix.WithWeighted())

	am, err := matrix.NewAdjacencyMatrix(g, mOpts)
	if err != nil {
		t.Fatalf("NewAdjacencyMatrix: %v", err)
	}

	gotNeighbors, err := am.Neighbors("A")
	if err != nil {
		t.Fatalf("Neighbors(A): %v", err)
	}
	wantNeighbors := []string{"B", "C"}
	if len(gotNeighbors) != len(wantNeighbors) {
		t.Fatalf("Neighbors(A): got %v, want %v", gotNeighbors, wantNeighbors)
	}
	for i := range wantNeighbors {
		if gotNeighbors[i] != wantNeighbors[i] {
			t.Fatalf("Neighbors(A)[%d]: got %q, want %q; full=%v",
				i, gotNeighbors[i], wantNeighbors[i], gotNeighbors)
		}
	}

	a := am.VertexIndex["A"]
	b := am.VertexIndex["B"]
	c := am.VertexIndex["C"]

	zeroEdge, err := am.Mat.At(a, b)
	if err != nil {
		t.Fatalf("At(A,B): %v", err)
	}
	if zeroEdge != 0 {
		t.Fatalf("A->B weight: got %v, want 0", zeroEdge)
	}

	nonZeroEdge, err := am.Mat.At(a, c)
	if err != nil {
		t.Fatalf("At(A,C): %v", err)
	}
	if nonZeroEdge != 2 {
		t.Fatalf("A->C weight: got %v, want 2", nonZeroEdge)
	}

	absent, err := am.Mat.At(b, a)
	if err != nil {
		t.Fatalf("At(B,A): %v", err)
	}
	if !math.IsInf(absent, +1) {
		t.Fatalf("B->A absent marker: got %v, want +Inf", absent)
	}

	g2, err := am.ToGraph()
	if err != nil {
		t.Fatalf("ToGraph: %v", err)
	}

	em := edgesMap(g2)
	if len(em) != 2 {
		t.Fatalf("export edge count: got %d edges=%v, want 2", len(em), em)
	}
	if w, ok := em[[2]string{"A", "B"}]; !ok || w != 0 {
		t.Fatalf("export must include A->B with weight=0; ok=%v w=%v", ok, w)
	}
	if w, ok := em[[2]string{"A", "C"}]; !ok || w != 2 {
		t.Fatalf("export must include A->C with weight=2; ok=%v w=%v", ok, w)
	}
}

func TestAdjacency_AllZeroWeightedAutoMode_DegradesToBinary(t *testing.T) {
	t.Parallel()

	g, _ := core.NewGraph(core.WithDirected(true), core.WithWeighted())
	_, _ = g.AddEdge("A", "B", 0)

	mOpts, _ := matrix.NewMatrixOptions(matrix.WithDirected(), matrix.WithWeighted())

	am, err := matrix.NewAdjacencyMatrix(g, mOpts)
	if err != nil {
		t.Fatalf("NewAdjacencyMatrix: %v", err)
	}

	a := am.VertexIndex["A"]
	b := am.VertexIndex["B"]

	got, err := am.Mat.At(a, b)
	if err != nil {
		t.Fatalf("At(A,B): %v", err)
	}
	if got != 1 {
		t.Fatalf("all-zero weighted auto mode: got %v, want binary 1", got)
	}
}

func TestAdjacency_SingleVertex(t *testing.T) {
	t.Parallel()

	g, _ := core.NewGraph()
	_ = g.AddVertex("v0")

	mOpts, _ := matrix.NewMatrixOptions()
	am, err := matrix.NewAdjacencyMatrix(g, mOpts)
	if err != nil {
		t.Fatalf("NewAdjacencyMatrix: %v", err)
	}
	if n, err := am.VertexCount(); err != nil || n != 1 {
		t.Fatalf("VertexCount: (%d,%v), want (1,nil)", n, err)
	}
	v, _ := am.Mat.At(0, 0)
	if v != 0 {
		t.Fatalf("A[0,0]=%v, want 0", v)
	}
	if g2, err := am.ToGraph(); err != nil || len(g2.Edges()) != 0 {
		t.Fatalf("ToGraph: err=%v edges=%d, want 0", err, len(g2.Edges()))
	}
}

// Degree vector tests (directed/undirected/loop semantics).
func TestDegreeVector_Directed_Unweighted(t *testing.T) {
	// A->B, A->C, B->C, C->C(loop), D isolated
	// core forbids non-zero weights in unweighted graphs, so we must enable weighted mode.
	// Matrix options still build an unweighted adjacency (presence==1), but core accepts edge insertion.
	g, _ := core.NewGraph(core.WithDirected(true), core.WithWeighted(), core.WithLoops())
	for _, id := range []string{"A", "B", "C", "D"} {
		_ = g.AddVertex(id)
	}
	_, _ = g.AddEdge("A", "B", 1)
	_, _ = g.AddEdge("A", "C", 1)
	_, _ = g.AddEdge("B", "C", 1)
	_, _ = g.AddEdge("C", "C", 1) // loop

	mOpts, _ := matrix.NewMatrixOptions(matrix.WithDirected(), matrix.WithAllowLoops())
	am, err := matrix.NewAdjacencyMatrix(g, mOpts)
	if err != nil {
		t.Fatalf("NewAdjacencyMatrix: %v", err)
	}
	vec, err := am.DegreeVector()
	if err != nil {
		t.Fatalf("DegreeVector: %v", err)
	}
	// Expected: A=2, B=1, C=1(loop counts 1), D=0
	want := []float64{2, 1, 1, 0}
	if !AlmostEqualSlice(vec, want, 1e-12) {
		t.Fatalf("degree mismatch: got %v, want %v", vec, want)
	}
}

// TestAdjacency_EmptyGraph_Degenerate validates that empty graphs are a valid degenerate case.
func TestAdjacency_EmptyGraph_Degenerate(t *testing.T) {
	t.Parallel()

	g, _ := core.NewGraph()
	mOpts, _ := matrix.NewMatrixOptions()
	am, err := matrix.NewAdjacencyMatrix(g, mOpts)
	if err != nil {
		t.Fatalf("NewAdjacencyMatrix(empty): %v", err)
	}
	if am == nil {
		t.Fatalf("NewAdjacencyMatrix(empty) returned nil")
	}

	// VertexCount must be 0 without errors.
	if n, err := am.VertexCount(); err != nil || n != 0 {
		t.Fatalf("VertexCount(empty): got (%d,%v), want (0,nil)", n, err)
	}

	// Neighbors on any ID must be ErrUnknownVertex (no vertices exist).
	if _, err := am.Neighbors("X"); !errors.Is(err, matrix.ErrUnknownVertex) {
		t.Fatalf("Neighbors(empty): want ErrUnknownVertex, got %v", err)
	}

	// Export must produce an empty graph (no vertices, no edges).
	g2, err := am.ToGraph()
	if err != nil {
		t.Fatalf("ToGraph(empty): %v", err)
	}
	if len(g2.Vertices()) != 0 {
		t.Fatalf("export empty vertices: got %d, want 0", len(g2.Vertices()))
	}
	if len(g2.Edges()) != 0 {
		t.Fatalf("export empty edges: got %d, want 0", len(g2.Edges()))
	}
}

func TestDegreeVector_Undirected_Unweighted(t *testing.T) {
	// Undirected edges: A-B, B-C
	g, _ := core.NewGraph(core.WithDirected(false), core.WithWeighted())
	for _, id := range []string{"A", "B", "C", "D"} {
		_ = g.AddVertex(id)
	}
	_, _ = g.AddEdge("A", "B", 1)
	_, _ = g.AddEdge("B", "C", 1)

	mOpts, _ := matrix.NewMatrixOptions(matrix.WithUndirected())
	am, err := matrix.NewAdjacencyMatrix(g, mOpts)
	if err != nil {
		t.Fatalf("NewAdjacencyMatrix: %v", err)
	}
	vec, err := am.DegreeVector()
	if err != nil {
		t.Fatalf("DegreeVector: %v", err)
	}
	want := []float64{1, 2, 1, 0}
	if !AlmostEqualSlice(vec, want, 1e-12) {
		t.Fatalf("degree mismatch: got %v, want %v", vec, want)
	}
}

func TestDegreeVector_LoopWeightedCountsAsOne(t *testing.T) {
	g, _ := core.NewGraph(core.WithDirected(false), core.WithWeighted(), core.WithLoops())
	_ = g.AddVertex("X")
	_, _ = g.AddEdge("X", "X", 7)

	mOpts, _ := matrix.NewMatrixOptions(matrix.WithAllowLoops(), matrix.WithWeighted())
	am, err := matrix.NewAdjacencyMatrix(g, mOpts)
	if err != nil {
		t.Fatalf("NewAdjacencyMatrix: %v", err)
	}
	vec, err := am.DegreeVector()
	if err != nil {
		t.Fatalf("DegreeVector: %v", err)
	}
	want := []float64{1}
	if !AlmostEqualSlice(vec, want, 1e-12) {
		t.Fatalf("degree mismatch: got %v, want %v", vec, want)
	}
}
