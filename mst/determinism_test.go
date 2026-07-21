// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package mst_test

import (
	"testing"

	"github.com/lvlath/go/mst"
)

func TestForest_KruskalDisconnectedGraph(t *testing.T) {
	graph := mustWeightedGraph(t)
	_, _ = graph.AddEdge("A", "B", 1)
	_, _ = graph.AddEdge("C", "D", 2)

	result, err := mst.MinimumSpanningTree(graph, mst.WithForest())

	mustNoError(t, err, "Kruskal forest")
	mustEqualString(t, string(result.Algorithm), string(mst.AlgorithmKruskal), "forest algorithm")
	mustEqualString(t, string(result.Mode), string(mst.ModeForest), "forest mode")
	mustValidForest(t, graph, result, 2, 3)
	mustEqualString(t, result.ComponentRoots[0], "A", "forest first root")
	mustEqualString(t, result.ComponentRoots[1], "C", "forest second root")
}

func TestForest_PrimDisconnectedGraphWithExplicitRoot(t *testing.T) {
	graph := mustWeightedGraph(t)
	_, _ = graph.AddEdge("A", "B", 1)
	_, _ = graph.AddEdge("C", "D", 2)

	result, err := mst.MinimumSpanningTree(
		graph,
		mst.WithAlgorithm(mst.AlgorithmPrim),
		mst.WithRoot("C"),
		mst.WithForest(),
	)

	mustNoError(t, err, "Prim forest with explicit root")
	mustEqualString(t, string(result.Algorithm), string(mst.AlgorithmPrim), "forest algorithm")
	mustEqualString(t, result.ComponentRoots[0], "C", "explicit component root first")
	mustValidForest(t, graph, result, 2, 3)
}

func TestDeterminism_EqualWeightCompleteGraphStableRepresentative(t *testing.T) {
	graph := mustWeightedGraph(t)

	edgeAB, _ := graph.AddEdge("A", "B", 1)
	edgeAC, _ := graph.AddEdge("A", "C", 1)
	edgeAD, _ := graph.AddEdge("A", "D", 1)
	_, _ = graph.AddEdge("B", "C", 1)
	_, _ = graph.AddEdge("B", "D", 1)
	_, _ = graph.AddEdge("C", "D", 1)

	wantIDs := []string{edgeAB, edgeAC, edgeAD}

	kruskalResult, err := mst.Kruskal(graph)
	mustNoError(t, err, "Kruskal equal-weight deterministic representative")
	mustValidStrictMST(t, graph, kruskalResult, 3)
	mustEqualEdgeIDs(t, kruskalResult.Edges, wantIDs, "Kruskal equal-weight edge order")

	primResult, err := mst.Prim(graph, "A")
	mustNoError(t, err, "Prim equal-weight deterministic representative")
	mustValidStrictMST(t, graph, primResult, 3)
	mustEqualEdgeIDs(t, primResult.Edges, wantIDs, "Prim equal-weight edge order")
}

func TestDeterminism_RepeatedRunsReturnSameEdgeIDs(t *testing.T) {
	graph := mustWeightedGraph(t)
	_, _ = graph.AddEdge("A", "B", 4)
	_, _ = graph.AddEdge("A", "C", 1)
	_, _ = graph.AddEdge("B", "C", 2)
	_, _ = graph.AddEdge("B", "D", 3)
	_, _ = graph.AddEdge("C", "D", 5)

	first, err := mst.MinimumSpanningTree(graph)
	mustNoError(t, err, "first MST run")

	firstIDs := edgeIDs(first.Edges)
	for i := 0; i < 10; i++ {
		next, err := mst.MinimumSpanningTree(graph)
		mustNoError(t, err, "repeated MST run")
		mustEqualEdgeIDs(t, next.Edges, firstIDs, "repeated MST edge IDs")
	}
}
