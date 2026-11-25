// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package flow_test

import (
	"context"
	"math"
	"slices"
	"testing"
	"time"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/flow"
	"github.com/lvlath/go/matrix"
)

func TestUndirectedBackboneAdaptedAsTwoDirectedCapacityArcs(t *testing.T) {
	for _, tt := range allAlgorithms() {
		t.Run(tt.name, func(t *testing.T) {
			g := buildRegionalUndirectedBackbone(t)

			result, err := flow.MaxFlow(
				g,
				"Tbilisi",
				"Frankfurt",
				flow.WithAlgorithm(tt.algorithm),
				flow.WithMaxAugmentations(128),
			)

			mustNoError(t, err, "MaxFlow undirected regional backbone")
			mustSuccessfulCertificate(t, g, result, "Tbilisi", "Frankfurt", 100, 1e-9, tt.name)
			mustResidualDirectedWeighted(t, result)
		})
	}
}

func TestCapacityMatrixUsesSameAdapterAsMaxFlowOnRegionalBackbone(t *testing.T) {
	g := buildRegionalUndirectedBackbone(t)

	capacityMatrix, order, err := flow.CapacityMatrix(g)
	mustNoError(t, err, "CapacityMatrix regional backbone")

	if len(order) != 6 {
		t.Fatalf("vertex order len=%d want=6 order=%v", len(order), order)
	}

	index := make(map[string]int, len(order))
	for i, vertexID := range order {
		index[vertexID] = i
	}

	assertMatrixCapacity := func(from, to string, want float64) {
		t.Helper()

		got, err := capacityMatrix.At(index[from], index[to])
		mustNoError(t, err, "CapacityMatrix.At("+from+","+to+")")
		mustEqualFloat(t, got, want, "capacity "+from+"->"+to)
	}

	assertMatrixCapacity("Frankfurt", "Warsaw", 160)
	assertMatrixCapacity("Warsaw", "Frankfurt", 160)
	assertMatrixCapacity("Tbilisi", "Bucharest", 45)
	assertMatrixCapacity("Bucharest", "Tbilisi", 45)
	assertMatrixCapacity("Kyiv", "Frankfurt", 0)
	assertMatrixCapacity("Frankfurt", "Tbilisi", 0)
}

func TestResidualPublicationIsDirectedWeightedForUndirectedBackbone(t *testing.T) {
	g := buildRegionalUndirectedBackbone(t)

	result, err := flow.MaxFlow(
		g,
		"Tbilisi",
		"Frankfurt",
		flow.WithAlgorithm(flow.AlgorithmDinic),
	)

	mustNoError(t, err, "MaxFlow residual publication")
	mustResidualDirectedWeighted(t, result)

	if result.Residual == nil {
		t.Fatalf("published residual graph is nil")
	}

	reverseCapacity := residualCapacity(t, result.Residual, "Frankfurt", "Warsaw")
	if reverseCapacity <= 0 {
		t.Fatalf("expected positive residual capacity on a published directed residual arc")
	}
}

func TestCapacityMatrixUndirectedMatchesFlowAdapter(t *testing.T) {
	g := mustGraph(t, core.WithDirected(false), core.WithWeighted())
	mustAddEdge(t, g, "A", "B", 4)
	mustAddEdge(t, g, "B", "C", 3)

	capacityMatrix, order, err := flow.CapacityMatrix(g)
	mustNoError(t, err, "CapacityMatrix")

	index := make(map[string]int, len(order))
	for i, vertexID := range order {
		index[vertexID] = i
	}

	ab, err := capacityMatrix.At(index["A"], index["B"])
	mustNoError(t, err, "CapacityMatrix.At(A,B)")

	ba, err := capacityMatrix.At(index["B"], index["A"])
	mustNoError(t, err, "CapacityMatrix.At(B,A)")

	bc, err := capacityMatrix.At(index["B"], index["C"])
	mustNoError(t, err, "CapacityMatrix.At(B,C)")

	cb, err := capacityMatrix.At(index["C"], index["B"])
	mustNoError(t, err, "CapacityMatrix.At(C,B)")

	mustEqualFloat(t, ab, 4, "A->B capacity")
	mustEqualFloat(t, ba, 4, "B->A capacity")
	mustEqualFloat(t, bc, 3, "B->C capacity")
	mustEqualFloat(t, cb, 3, "C->B capacity")
}

func TestMaxFlowPublishesDirectedWeightedResidual(t *testing.T) {
	g := mustGraph(t, core.WithDirected(false), core.WithWeighted())
	mustAddEdge(t, g, "A", "B", 4)

	result, err := flow.MaxFlow(g, "A", "B")

	mustNoError(t, err, "MaxFlow undirected")
	mustResidualDirectedWeighted(t, result)
}

// TestCapacityMatrixRejectsInvalidPublicInputs verifies every reachable validation boundary.
// Failure must publish neither a matrix nor a vertex-order slice.
// Implementation:
//   - Stage 1: Construct valid, unweighted, and negative-capacity graph fixtures.
//   - Stage 2: Prepare nil, canceled, expired, and invalid option cases.
//   - Stage 3: Check sentinel identities and transactional output clearing.
//
// Behavior highlights: Preserves flow and core error classifications across the adapter.
// Inputs: t is the test handle.
// Returns: None.
// Errors: Missing sentinel or partially published matrix fails the test.
// Determinism: The deadline is in the past; no sleeping or scheduler assumptions.
// Complexity: O(K) fail-fast adapter invocations for K validation cases.
// Notes: A negative finite weight is accepted by core but rejected as flow capacity.
// AI-Hints: Do not try to insert NaN into core.Graph to test flow's numeric boundary.
func TestCapacityMatrixRejectsInvalidPublicInputs(t *testing.T) {
	valid := buildFlowObservationNetwork(t)
	unweighted := mustGraph(t, core.WithDirected(true))
	mustAddVertex(t, unweighted, flowTestSource)
	negative := mustGraph(t, core.WithDirected(true), core.WithWeighted())
	mustAddEdge(t, negative, flowTestSource, flowTestSink, -1)
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	expired, release := context.WithDeadline(context.Background(), time.Unix(0, 0))
	defer release()
	cases := []struct {
		name  string
		graph *core.Graph
		opts  []flow.Option
		want  []error
	}{
		{name: "nil_option", graph: valid, opts: []flow.Option{nil}, want: []error{flow.ErrInvalidOptions}},
		{name: "invalid_epsilon", graph: valid, opts: []flow.Option{flow.WithEpsilon(math.NaN())}, want: []error{flow.ErrInvalidEpsilon}},
		{name: "nil_graph", want: []error{flow.ErrNilGraph}},
		{name: "canceled", graph: valid, opts: []flow.Option{flow.WithContext(canceled)}, want: []error{context.Canceled}},
		{name: "deadline", graph: valid, opts: []flow.Option{flow.WithContext(expired)}, want: []error{context.DeadlineExceeded}},
		{name: "unweighted", graph: unweighted, want: []error{flow.ErrUnweightedGraph, core.ErrBadWeight}},
		{name: "negative_capacity", graph: negative, want: []error{flow.ErrInvalidCapacity, flow.ErrNegativeCapacity}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result, order, err := flow.CapacityMatrix(tc.graph, tc.opts...)
			for _, target := range tc.want {
				mustErrorIs(t, err, target, "capacity adapter validation")
			}
			if result != nil || order != nil {
				t.Fatalf("invalid adapter result: matrix=%p order=%v", result, order)
			}
		})
	}
}

// TestCapacityMatrixPreservesEmptyAndIsolatedVertexShapes verifies zero-edge networks.
// Matrix axes describe vertices even when there are no capacity arcs.
// Implementation:
//   - Stage 1: Export an empty weighted graph.
//   - Stage 2: Add isolated vertices and export again.
//   - Stage 3: Check exact shape, lexical axis order, and all zero cells.
//
// Behavior highlights: A 0x0 matrix is a valid empty result, not an allocation failure.
// Inputs: t is the test handle.
// Returns: None.
// Errors: Shape, ordering, or nonzero-cell mismatches fail the test.
// Determinism: Vertices are added out of order to verify canonical ordering.
// Complexity: O(V^2) time and space for the dense output.
// AI-Hints: Do not invent an error for a zero-sized matrix accepted by matrix.NewPreparedDense.
func TestCapacityMatrixPreservesEmptyAndIsolatedVertexShapes(t *testing.T) {
	g := mustGraph(t, core.WithDirected(true), core.WithWeighted())
	empty, order, err := flow.CapacityMatrix(g)
	mustNoError(t, err, "empty CapacityMatrix")
	if empty == nil || empty.Rows() != 0 || empty.Cols() != 0 || len(order) != 0 {
		t.Fatalf("unexpected empty matrix: matrix=%p order=%v", empty, order)
	}
	for _, vertex := range []string{"Z", "B", "A"} {
		mustAddVertex(t, g, vertex)
	}
	result, order, err := flow.CapacityMatrix(g)
	mustNoError(t, err, "isolated CapacityMatrix")
	wantOrder := []string{"A", "B", "Z"}
	if result == nil || result.Rows() != len(wantOrder) || result.Cols() != len(wantOrder) || !slices.Equal(order, wantOrder) {
		t.Fatalf("isolated matrix axis mismatch: order=%v want=%v", order, wantOrder)
	}
	for row := range order {
		for col := range order {
			value, err := result.At(row, col)
			mustNoError(t, err, "isolated matrix cell")
			if value != 0 {
				t.Fatalf("isolated matrix[%d,%d]=%g, want 0", row, col, value)
			}
		}
	}
}

// TestCapacityMatrixDirectedParallelAndEpsilonInventory verifies every cell of a capacity snapshot.
// The independent dense oracle distinguishes reverse placeholders, directed antiparallel arcs, and filtered edges.
// Implementation:
//   - Stage 1: Build a directed multigraph with a loop, zero arcs, threshold arcs, and isolated vertices.
//   - Stage 2: Export the effective capacity network using epsilon=0.5.
//   - Stage 3: Compare the full matrix against independently enumerated cells.
//   - Stage 4: Mutate the returned matrix and vertex-order slice, then repeat the export.
//
// Behavior highlights: Threshold filtering occurs per original edge before parallel aggregation.
// Inputs: t is the test handle.
// Returns: None.
// Errors: Missing cells, wrong shape, incorrect filtering, or aliasing fails the test.
// Determinism: Expected rows/columns are explicitly ordered A,B,C,D,I,S,T.
// Complexity: O(V^2+E) output verification, excluding core enumeration sorting.
// Notes: Dyadic capacities allow exact comparison without a numerical tolerance.
// AI-Hints: Do not construct the expected matrix using CapacityMatrix itself.
func TestCapacityMatrixDirectedParallelAndEpsilonInventory(t *testing.T) {
	const epsilon = 0.5
	g := mustGraph(t, core.WithDirected(true), core.WithWeighted(), core.WithMultiEdges(), core.WithLoops())
	// Directed parallel arcs aggregate to 4; the antiparallel original remains distinct.
	mustAddEdge(t, g, "S", "A", 2.25)
	mustAddEdge(t, g, "S", "A", 1.75)
	mustAddEdge(t, g, "A", "S", 1.25)
	mustAddEdge(t, g, "A", "B", 3.5)
	mustAddEdge(t, g, "B", "T", 3.5)
	mustAddEdge(t, g, "S", "C", 2)
	mustAddEdge(t, g, "C", "T", 2)
	// All these entries are excluded: a threshold tie, a tiny edge, zero, and a loop.
	mustAddEdge(t, g, "C", "A", epsilon)
	mustAddEdge(t, g, "B", "C", 0.25)
	mustAddEdge(t, g, "D", "T", 0)
	mustAddEdge(t, g, "B", "B", 99)
	// Two individually filtered parallel edges must not be revived by summing first.
	mustAddEdge(t, g, "D", "A", 0.375)
	mustAddEdge(t, g, "D", "A", 0.375)
	mustAddVertex(t, g, "I")
	before := g.Clone()
	wantOrder := []string{"A", "B", "C", "D", "I", "S", "T"}
	want := [][]float64{
		{0, 3.5, 0, 0, 0, 1.25, 0}, // A
		{0, 0, 0, 0, 0, 0, 3.5},    // B
		{0, 0, 0, 0, 0, 0, 2},      // C
		{0, 0, 0, 0, 0, 0, 0},      // D
		{0, 0, 0, 0, 0, 0, 0},      // I
		{4, 0, 2, 0, 0, 0, 0},      // S
		{0, 0, 0, 0, 0, 0, 0},      // T
	}
	result, order, err := flow.CapacityMatrix(g, flow.WithEpsilon(epsilon))
	mustNoError(t, err, "directed capacity matrix")
	mustCapacityCells(t, result, order, want, wantOrder)
	// Exported artifacts are caller-owned; editing them must not affect the input or a later export.
	mustNoError(t, result.Set(0, 1, 0), "mutate exported matrix")
	order[0] = "caller-owned-axis"
	repeated, repeatedOrder, err := flow.CapacityMatrix(g, flow.WithEpsilon(epsilon))
	mustNoError(t, err, "re-export capacity matrix")
	mustCapacityCells(t, repeated, repeatedOrder, want, wantOrder)
	mustEqualFlowGraphs(t, g, before, "capacity export does not mutate input")
}

// TestCapacityMatrixPreservesNumericErrorOnParallelOverflow verifies a real lower-layer failure.
// Individually valid finite capacities may aggregate to a non-representable float64 value.
// Implementation:
//   - Stage 1: Add two MaxFloat64 parallel arcs to a weighted multigraph.
//   - Stage 2: Export the capacity matrix using the actual adapter.
//   - Stage 3: Require both flow and matrix sentinels with no published artifacts.
//
// Behavior highlights: Reaches matrix.Set's strict numeric validation naturally.
// Inputs: t is the test handle.
// Returns: None.
// Errors: ErrInvalidCapacity and matrix.ErrNaNInf must both be preserved by the current adapter.
// Determinism: Summation of two positive MaxFloat64 values overflows on every run.
// Complexity: O(1) on the fixed two-vertex fixture.
// Notes: This guards current late overflow rejection, not a preferred overflow detection design.
// AI-Hints: Do not disable matrix numeric validation to make overflow tests pass.
func TestCapacityMatrixPreservesNumericErrorOnParallelOverflow(t *testing.T) {
	g := mustGraph(t, core.WithDirected(true), core.WithWeighted(), core.WithMultiEdges())
	mustAddEdge(t, g, flowTestSource, flowTestSink, math.MaxFloat64)
	mustAddEdge(t, g, flowTestSource, flowTestSink, math.MaxFloat64)
	result, order, err := flow.CapacityMatrix(g)
	mustErrorIs(t, err, flow.ErrInvalidCapacity, "parallel overflow classification")
	mustErrorIs(t, err, matrix.ErrNaNInf, "matrix numeric root cause")
	if result != nil || order != nil {
		t.Fatalf("overflow published a matrix: result=%p order=%v", result, order)
	}
}
