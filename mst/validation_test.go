// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package mst_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/mst"
)

func TestValidation_EmptyGraphReturnsDisconnectedAndEmptyGraph(t *testing.T) {
	graph := mustWeightedGraph(t)

	_, err := mst.Kruskal(graph)

	mustErrorIs(t, err, mst.ErrDisconnected, "Kruskal empty graph disconnected")
	mustErrorIs(t, err, mst.ErrEmptyGraph, "Kruskal empty graph precise")
}

func TestValidation_NilGraphReturnsPreciseSentinels(t *testing.T) {
	_, err := mst.MinimumSpanningTree(nil)

	mustErrorIs(t, err, mst.ErrInvalidGraph, "nil graph umbrella")
	mustErrorIs(t, err, mst.ErrNilGraph, "nil graph precise")
}

func TestValidation_UnweightedGraphRejected(t *testing.T) {
	graph, err := core.NewGraph()
	mustNoError(t, err, "core.NewGraph")

	_, err = mst.Kruskal(graph)

	mustErrorIs(t, err, mst.ErrInvalidGraph, "unweighted graph umbrella")
	mustErrorIs(t, err, mst.ErrUnweightedGraph, "unweighted graph precise")
}

func TestValidation_DirectedGraphRejected(t *testing.T) {
	graph, err := core.NewGraph(core.WithWeighted(), core.WithDirected(true))
	mustNoError(t, err, "core.NewGraph directed weighted")

	_, err = mst.Kruskal(graph)

	mustErrorIs(t, err, mst.ErrInvalidGraph, "directed graph umbrella")
	mustErrorIs(t, err, mst.ErrDirectedGraph, "directed graph precise")
}

func TestValidation_DirectedEdgeRejected(t *testing.T) {
	graph, err := core.NewGraph(core.WithWeighted(), core.WithMixedEdges())
	mustNoError(t, err, "core.NewGraph mixed weighted")

	_, err = graph.AddEdge("A", "B", 1, core.WithEdgeDirected(true))
	mustNoError(t, err, "Graph.AddEdge directed override")

	_, err = mst.Prim(graph, "A")

	mustErrorIs(t, err, mst.ErrInvalidGraph, "directed edge umbrella")
	mustErrorIs(t, err, mst.ErrDirectedEdge, "directed edge precise")
}

func TestValidation_EmptyPrimRootRejectedBeforeTraversal(t *testing.T) {
	graph := mustWeightedGraph(t)
	_ = graph.AddVertex("X")

	_, err := mst.Prim(graph, "")

	mustErrorIs(t, err, mst.ErrEmptyRoot, "empty Prim root")
}

func TestValidation_NonFiniteWeightRejectedAfterAliasMutation(t *testing.T) {
	tests := []struct {
		name   string
		weight float64
	}{
		{name: "NaN", weight: math.NaN()},
		{name: "PositiveInf", weight: math.Inf(1)},
		{name: "NegativeInf", weight: math.Inf(-1)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			graph := mustWeightedGraph(t)
			edgeID, _ := graph.AddEdge("A", "B", 1)
			mustCorruptEdgeWeight(t, graph, edgeID, tt.weight)

			_, err := mst.Kruskal(graph)
			mustErrorIs(t, err, mst.ErrNaNInfWeight, "Kruskal non-finite weight")

			_, err = mst.Prim(graph, "A")
			mustErrorIs(t, err, mst.ErrNaNInfWeight, "Prim non-finite weight")
		})
	}
}

func TestValidation_NilOptionRejected(t *testing.T) {
	graph := mustWeightedGraph(t)
	_, _ = graph.AddEdge("A", "B", 1)

	_, err := mst.MinimumSpanningTree(graph, nil)

	mustErrorIs(t, err, mst.ErrNilOption, "nil option")
}

func TestValidation_UnsupportedAlgorithmRejected(t *testing.T) {
	graph := mustWeightedGraph(t)
	_, _ = graph.AddEdge("A", "B", 1)

	_, err := mst.MinimumSpanningTree(
		graph,
		mst.WithAlgorithm(mst.Algorithm("unknown")),
	)

	mustErrorIs(t, err, mst.ErrUnsupportedAlgorithm, "unsupported algorithm")
}

func TestValidation_StrictModeRejectsDisconnectedGraph(t *testing.T) {
	graph := mustWeightedGraph(t)
	_, _ = graph.AddEdge("A", "B", 1)
	_, _ = graph.AddEdge("C", "D", 2)

	_, err := mst.MinimumSpanningTree(graph)

	mustErrorIs(t, err, mst.ErrDisconnected, "strict disconnected graph")
}

func TestValidation_NoErrorStringMatching(t *testing.T) {
	err := errors.Join(mst.ErrInvalidGraph, mst.ErrNilGraph)

	mustErrorIs(t, err, mst.ErrInvalidGraph, "joined invalid graph")
	mustErrorIs(t, err, mst.ErrNilGraph, "joined nil graph")
}
