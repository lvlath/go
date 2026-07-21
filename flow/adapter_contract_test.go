// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package flow_test

import (
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/flow"
)

func buildRegionalUndirectedBackbone(t *testing.T) *core.Graph {
	t.Helper()

	g := mustGraph(t, core.WithDirected(false), core.WithWeighted())

	mustAddEdge(t, g, "Frankfurt", "Warsaw", 160)
	mustAddEdge(t, g, "Frankfurt", "Prague", 120)
	mustAddEdge(t, g, "Warsaw", "Kyiv", 90)
	mustAddEdge(t, g, "Prague", "Kyiv", 70)
	mustAddEdge(t, g, "Kyiv", "Tbilisi", 55)
	mustAddEdge(t, g, "Tbilisi", "Bucharest", 45)
	mustAddEdge(t, g, "Bucharest", "Prague", 40)
	mustAddEdge(t, g, "Bucharest", "Warsaw", 20)

	return g
}

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
