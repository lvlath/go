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
