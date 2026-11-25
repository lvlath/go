// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package flow_test

import (
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/flow"
)

func TestMaxFlowResultResidualCloneNilReceiver(t *testing.T) {
	var result *flow.MaxFlowResult

	clone, err := result.ResidualClone()

	mustErrorIs(t, err, flow.ErrNilResult, "nil result ResidualClone")
	if clone != nil {
		t.Fatalf("nil result ResidualClone: got clone %#v want nil", clone)
	}
}

func TestMaxFlowResultResidualCloneRequiresResidual(t *testing.T) {
	result := &flow.MaxFlowResult{}

	clone, err := result.ResidualClone()

	mustErrorIs(t, err, flow.ErrNoResidual, "missing residual ResidualClone")
	if clone != nil {
		t.Fatalf("missing residual clone: got %#v want nil", clone)
	}
}

func TestMaxFlowPublishesDirectedWeightedResidualForUndirectedInput(t *testing.T) {
	g := mustGraph(t, core.WithDirected(false), core.WithWeighted())
	mustAddEdge(t, g, "A", "B", 4)

	result, err := flow.MaxFlow(g, "A", "B")

	mustNoError(t, err, "MaxFlow undirected")
	mustResidualDirectedWeighted(t, result)
}

func TestCapacityMatrixUsesSameUndirectedAdapterAsMaxFlow(t *testing.T) {
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

// TestMaxFlowResultIsNilSupportsCoreNilable verifies typed-nil semantics.
// An allocated result is non-nil even when no residual has been published.
// Implementation:
//   - Stage 1: Convert a nil *MaxFlowResult to core.Nilable.
//   - Stage 2: Check allocated zero-state and completed solver results.
//
// Behavior highlights: Nil detection is independent of the flow value and residual field.
// Inputs: t is the test handle.
// Returns: None.
// Errors: Incorrect nil classification fails the test.
// Determinism: Pointer state and a fixed solver fixture.
// Complexity: O(1) per predicate, plus one fixed-fixture solver run.
// AI-Hints: A typed-nil interface is not itself a nil interface.
func TestMaxFlowResultIsNilSupportsCoreNilable(t *testing.T) {
	var absent *flow.MaxFlowResult
	var nilable core.Nilable = absent
	if nilable == nil || !nilable.IsNil() || !absent.IsNil() {
		t.Fatal("typed-nil MaxFlowResult must satisfy the core.Nilable contract")
	}
	allocated := &flow.MaxFlowResult{}
	nilable = allocated
	if allocated.IsNil() || nilable.IsNil() {
		t.Fatal("allocated zero-state result must not be classified as nil")
	}
	g := buildFlowObservationNetwork(t)
	result, err := flow.MaxFlow(g, flowTestSource, flowTestSink)
	mustNoError(t, err, "completed result")
	mustSuccessfulCertificate(t, g, result, flowTestSource, flowTestSink, flowTestTotalValue, flowTestDefaultEpsilon, "non-nil completed result")
	if result.IsNil() {
		t.Fatal("completed MaxFlowResult reported nil")
	}
}

// TestMaxFlowResultResidualCloneOwnsIndependentTopology verifies the successful clone path.
// Renaming an edge in the clone must not mutate the original Edge object or its catalog.
// Implementation:
//   - Stage 1: Solve the observation network and clone the published residual.
//   - Stage 2: Compare all vertices, directedness, weights, and edge identities.
//   - Stage 3: Rename a cloned edge and add a cloned-only vertex.
//   - Stage 4: Verify the original result and input network remain unchanged.
//
// Behavior highlights: Tests structural ownership, not just pointer inequality.
// Inputs: t is the test handle.
// Returns: None.
// Errors: Clone, mutation, or ownership failures fail the test.
// Determinism: The selected edge is the first edge in core's sorted ID order.
// Complexity: O(V log V + E log E) verification plus one solver run.
// Notes: Vertex metadata follows core.Clone's shallow-copy contract and is not mutated here.
// AI-Hints: Use core mutators, not direct writes to published read-only Edge pointers.
func TestMaxFlowResultResidualCloneOwnsIndependentTopology(t *testing.T) {
	const renamedID = "detached-residual-edge"
	const newVertex = "clone-only-vertex"
	g := buildFlowObservationNetwork(t)
	inputBefore := g.Clone()
	result, err := flow.MaxFlow(g, flowTestSource, flowTestSink)
	mustNoError(t, err, "MaxFlow for residual clone")
	mustSuccessfulCertificate(t, g, result, flowTestSource, flowTestSink, flowTestTotalValue, flowTestDefaultEpsilon, "residual clone source")
	clone, err := result.ResidualClone()
	mustNoError(t, err, "ResidualClone")
	if clone == nil || clone == result.Residual || result.Residual == g {
		t.Fatal("residual publication and clone must own separate graph objects")
	}
	mustEqualFlowGraphs(t, clone, result.Residual, "cloned residual topology")
	// Snapshot the edge value before renaming; retaining only a pointer would hide aliasing.
	edges := result.Residual.Edges()
	if len(edges) == 0 || edges[0] == nil {
		t.Fatal("fixture did not produce a residual edge")
	}
	before := *edges[0]
	mustNoError(t, clone.SetEdgeID(before.ID, renamedID), "rename cloned edge")
	mustAddVertex(t, clone, newVertex)
	_, err = clone.GetEdge(before.ID)
	mustErrorIs(t, err, core.ErrEdgeNotFound, "old cloned edge ID")
	changed, err := clone.GetEdge(renamedID)
	mustNoError(t, err, "renamed cloned edge")
	want := before
	want.ID = renamedID
	if changed == nil || *changed != want {
		t.Fatalf("renamed edge: got %+v want %+v", changed, want)
	}
	original, err := result.Residual.GetEdge(before.ID)
	mustNoError(t, err, "original residual edge")
	if original == nil || *original != before || result.Residual.HasVertex(newVertex) {
		t.Fatal("clone mutation escaped into the original residual")
	}
	_, err = result.Residual.GetEdge(renamedID)
	mustErrorIs(t, err, core.ErrEdgeNotFound, "clone-only edge ID")
	mustEqualFlowGraphs(t, g, inputBefore, "input graph remains unchanged")
	mustSuccessfulCertificate(t, g, result, flowTestSource, flowTestSink, flowTestTotalValue, flowTestDefaultEpsilon, "original certificate survives clone mutation")
}
