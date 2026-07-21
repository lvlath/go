// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package mst_test

import (
	"testing"

	"github.com/lvlath/go/mst"
)

func TestKruskal_TriangleStrictMST(t *testing.T) {
	graph := mustWeightedGraph(t)
	_, _ = graph.AddEdge("A", "B", 1)
	_, _ = graph.AddEdge("B", "C", 2)
	_, _ = graph.AddEdge("A", "C", 3)

	result, err := mst.Kruskal(graph)

	mustNoError(t, err, "Kruskal triangle")
	mustValidStrictMST(t, graph, result, 3)
	mustEqualString(t, string(result.Algorithm), string(mst.AlgorithmKruskal), "Kruskal algorithm")
}

func TestKruskal_SingleVertexGraph(t *testing.T) {
	graph := mustWeightedGraph(t)
	_ = graph.AddVertex("X")

	result, err := mst.Kruskal(graph)

	mustNoError(t, err, "Kruskal single vertex")
	mustValidStrictMST(t, graph, result, 0)
}

func TestKruskal_ParallelEdgesSelectsLightest(t *testing.T) {
	graph := mustWeightedMultiGraph(t)
	_, _ = graph.AddEdge("A", "B", 5)
	lightEdgeID, _ := graph.AddEdge("A", "B", 1)

	result, err := mst.Kruskal(graph)

	mustNoError(t, err, "Kruskal parallel edges")
	mustValidStrictMST(t, graph, result, 1)
	mustEqualEdgeIDs(t, result.Edges, []string{lightEdgeID}, "Kruskal selected lighter parallel edge")
}

func TestKruskal_NegativeFiniteWeightsAccepted(t *testing.T) {
	graph := mustWeightedGraph(t)
	_, _ = graph.AddEdge("A", "B", -2)
	_, _ = graph.AddEdge("B", "C", 1)
	_, _ = graph.AddEdge("A", "C", 5)

	result, err := mst.Kruskal(graph)

	mustNoError(t, err, "Kruskal negative finite weights")
	mustValidStrictMST(t, graph, result, -1)
}
