// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package dfs_test

import (
	"context"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/dfs"
)

func TestTopologicalSort_NilGraph(t *testing.T) {
	order, err := dfs.TopologicalSort(nil)

	mustNilState(t, order, true, "TopologicalSort(nil) order")
	mustErrorIs(t, err, dfs.ErrGraphNil)
}

func TestTopologicalSort_UndirectedGraphUsesSentinel(t *testing.T) {
	g, _ := core.NewGraph()

	order, err := dfs.TopologicalSort(g)

	mustNilState(t, order, true, "TopologicalSort undirected order")
	mustErrorIs(t, err, dfs.ErrGraphNotDirected)
}

func TestTopologicalSortContext_NilContext(t *testing.T) {
	g, _ := core.NewGraph(core.WithDirected(true))

	order, err := dfs.TopologicalSortContext(nil, g)

	mustNilState(t, order, true, "TopologicalSortContext nil-context order")
	mustErrorIs(t, err, dfs.ErrOptionViolation)
}

func TestTopologicalSort_EmptyGraph(t *testing.T) {
	g, _ := core.NewGraph(core.WithDirected(true))

	order, err := dfs.TopologicalSort(g)
	mustNoError(t, err)

	mustEqualInt(t, len(order), 0, "")
}

func TestTopologicalSort_NoEdgesDeterministicOrder(t *testing.T) {
	g, _ := core.NewGraph(core.WithDirected(true))
	mustNoError(t, g.AddVertex("A"))
	mustNoError(t, g.AddVertex("B"))
	mustNoError(t, g.AddVertex("C"))

	order, err := dfs.TopologicalSort(g)
	mustNoError(t, err)

	// Independent roots are visited in graph vertex order and then reversed from post-order.
	mustEqualSlice(t, order, []string{"C", "B", "A"})
}

func TestTopologicalSort_SimpleChain(t *testing.T) {
	g, _ := core.NewGraph(core.WithDirected(true))
	_, _ = g.AddEdge("A", "B", 0)
	_, _ = g.AddEdge("B", "C", 0)

	order, err := dfs.TopologicalSort(g)
	mustNoError(t, err)

	mustEqualSlice(t, order, []string{"A", "B", "C"})
}

func TestTopologicalSort_BranchingDAGDeterministicOrder(t *testing.T) {
	g, _ := core.NewGraph(core.WithDirected(true))
	_, _ = g.AddEdge("A", "B", 0)
	_, _ = g.AddEdge("A", "C", 0)

	order, err := dfs.TopologicalSort(g)
	mustNoError(t, err)

	// DFS visits B before C, so post-order is [B,C,A] and the final topological order is [A,C,B].
	mustEqualSlice(t, order, []string{"A", "C", "B"})
}

func TestTopologicalSort_DisconnectedGraphRespectsDependencies(t *testing.T) {
	g, _ := core.NewGraph(core.WithDirected(true))
	_, _ = g.AddEdge("X", "Y", 0)
	_, _ = g.AddEdge("A", "B", 0)

	order, err := dfs.TopologicalSort(g)
	mustNoError(t, err)

	mustEqualStringSet(t, order, []string{"X", "Y", "A", "B"})
	mustTopoOrderRespectsEdges(t, order, [][2]string{
		{"X", "Y"},
		{"A", "B"},
	})
}

func TestTopologicalSort_Cycle(t *testing.T) {
	g, _ := core.NewGraph(core.WithDirected(true))
	_, _ = g.AddEdge("A", "B", 0)
	_, _ = g.AddEdge("B", "C", 0)
	_, _ = g.AddEdge("C", "A", 0)

	order, err := dfs.TopologicalSort(g)

	mustNilState(t, order, true, "TopologicalSort cycle order")
	mustErrorIs(t, err, dfs.ErrCycleDetected)
}

func TestTopologicalSort_LargeLinearChain(t *testing.T) {
	g, _ := core.NewGraph(core.WithDirected(true))
	vertices := []string{"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"}

	for index := 0; index < len(vertices)-1; index++ {
		_, err := g.AddEdge(vertices[index], vertices[index+1], 0)
		mustNoError(t, err)
	}

	order, err := dfs.TopologicalSort(g)
	mustNoError(t, err)

	mustEqualSlice(t, order, vertices)
}

func TestTopologicalSort_DisconnectedLargeRespectsDependencies(t *testing.T) {
	g, _ := core.NewGraph(core.WithDirected(true))

	chainOne := []string{"1", "2", "3", "4", "5"}
	for index := 0; index < len(chainOne)-1; index++ {
		_, err := g.AddEdge(chainOne[index], chainOne[index+1], 0)
		mustNoError(t, err)
	}

	chainTwo := []string{"A", "B", "C", "D", "E", "F"}
	for index := 0; index < len(chainTwo)-1; index++ {
		_, err := g.AddEdge(chainTwo[index], chainTwo[index+1], 0)
		mustNoError(t, err)
	}

	order, err := dfs.TopologicalSort(g)
	mustNoError(t, err)

	mustEqualStringSet(t, order, []string{"1", "2", "3", "4", "5", "A", "B", "C", "D", "E", "F"})
	mustTopoOrderRespectsEdges(t, order, [][2]string{
		{"1", "2"},
		{"2", "3"},
		{"3", "4"},
		{"4", "5"},
		{"A", "B"},
		{"B", "C"},
		{"C", "D"},
		{"D", "E"},
		{"E", "F"},
	})
}

func TestTopologicalSort_ComplexDAGRespectsDependencies(t *testing.T) {
	g, _ := core.NewGraph(core.WithDirected(true))

	vertices := []string{
		"V1", "V2", "V3", "V4", "V5", "V6",
		"V7", "V8", "V9", "V10", "V11", "V12",
	}
	for _, vertexID := range vertices {
		mustNoError(t, g.AddVertex(vertexID))
	}

	edges := [][2]string{
		{"V1", "V3"},
		{"V1", "V2"},
		{"V2", "V5"},
		{"V3", "V5"},
		{"V2", "V4"},
		{"V4", "V6"},
		{"V5", "V7"},
		{"V6", "V8"},
		{"V7", "V9"},
		{"V8", "V10"},
		{"V3", "V11"},
		{"V11", "V12"},
	}
	for _, edge := range edges {
		_, err := g.AddEdge(edge[0], edge[1], 0)
		mustNoError(t, err)
	}

	order, err := dfs.TopologicalSort(g)
	mustNoError(t, err)

	mustEqualStringSet(t, order, vertices)
	mustTopoOrderRespectsEdges(t, order, edges)
}

func TestTopologicalSort_ContextCanceled(t *testing.T) {
	g, _ := core.NewGraph(core.WithDirected(true))
	_, _ = g.AddEdge("A", "B", 0)
	_, _ = g.AddEdge("B", "C", 0)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	order, err := dfs.TopologicalSortContext(ctx, g)

	mustNilState(t, order, true, "TopologicalSortContext canceled order")
	mustErrorIs(t, err, context.Canceled)
}

func TestTopologicalSort_MixedGraphIgnoresUndirectedEdges(t *testing.T) {
	g, _ := core.NewMixedGraph(core.WithDirected(true))

	_, _ = g.AddEdge("A", "B", 0, core.WithEdgeDirected(true))
	_, _ = g.AddEdge("B", "D", 0, core.WithEdgeDirected(true))

	// This edge is intentionally undirected and must be ignored by topological traversal logic.
	_, _ = g.AddEdge("A", "C", 0, core.WithEdgeDirected(false))

	order, err := dfs.TopologicalSort(g)
	mustNoError(t, err)

	mustEqualStringSet(t, order, []string{"A", "B", "C", "D"})
	mustTopoOrderRespectsEdges(t, order, [][2]string{
		{"A", "B"},
		{"B", "D"},
	})
}
