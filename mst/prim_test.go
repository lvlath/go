// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package mst_test

import (
	"testing"

	"github.com/lvlath/go/mst"
)

func TestPrim_TriangleStrictMST(t *testing.T) {
	graph := mustWeightedGraph(t)
	_, _ = graph.AddEdge("A", "B", 1)
	_, _ = graph.AddEdge("B", "C", 2)
	_, _ = graph.AddEdge("A", "C", 3)

	result, err := mst.Prim(graph, "A")

	mustNoError(t, err, "Prim triangle")
	mustValidStrictMST(t, graph, result, 3)
	mustEqualString(t, string(result.Algorithm), string(mst.AlgorithmPrim), "Prim algorithm")
	mustEqualString(t, result.Root, "A", "Prim root")
}

func TestPrim_UndirectedEdgeRootAtStoredToEndpoint(t *testing.T) {
	graph := mustWeightedGraph(t)
	_, _ = graph.AddEdge("A", "B", 7)

	result, err := mst.Prim(graph, "B")

	mustNoError(t, err, "Prim root at stored To endpoint")
	mustValidStrictMST(t, graph, result, 7)
	mustEqualString(t, result.Root, "B", "Prim root metadata")
}

func TestPrim_SingleVertexGraph(t *testing.T) {
	graph := mustWeightedGraph(t)
	_ = graph.AddVertex("X")

	result, err := mst.Prim(graph, "X")

	mustNoError(t, err, "Prim single vertex")
	mustValidStrictMST(t, graph, result, 0)
	mustEqualString(t, result.Root, "X", "single vertex root")
}

func TestPrim_NegativeFiniteWeightsAccepted(t *testing.T) {
	graph := mustWeightedGraph(t)
	_, _ = graph.AddEdge("A", "B", -2)
	_, _ = graph.AddEdge("B", "C", 1)
	_, _ = graph.AddEdge("A", "C", 5)

	result, err := mst.Prim(graph, "A")

	mustNoError(t, err, "Prim negative finite weights")
	mustValidStrictMST(t, graph, result, -1)
}
