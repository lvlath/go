// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package dfs_test

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/dfs"
)

// buildChain creates a directed chain graph of length n: N0->N1->...->N(n-1).
func buildChain(n int) *core.Graph {
	g, _ := core.NewGraph(core.WithDirected(true))

	for index := 0; index < n-1; index++ {
		fromID := fmt.Sprintf("N%d", index)
		toID := fmt.Sprintf("N%d", index+1)

		_ = g.AddVertex(fromID)
		_ = g.AddVertex(toID)
		_, _ = g.AddEdge(fromID, toID, 0)
	}

	return g
}

// buildBinaryTree creates a complete directed binary tree of the requested depth.
// Vertex IDs are T-1, T-2, ..., T-(2^depth-1).
func buildBinaryTree(depth int) *core.Graph {
	g, _ := core.NewGraph(core.WithDirected(true))
	vertexCount := (1 << depth) - 1

	for index := 1; index <= vertexCount; index++ {
		vertexID := fmt.Sprintf("T-%d", index)
		_ = g.AddVertex(vertexID)

		if index == 1 {
			continue
		}

		parentID := fmt.Sprintf("T-%d", index/2)
		_, _ = g.AddEdge(parentID, vertexID, 0)
	}

	return g
}

func TestDFS_NilGraph(t *testing.T) {
	result, err := dfs.DFS(nil, "A")

	mustNilState(t, result, true, "DFS(nil) result")
	mustErrorIs(t, err, dfs.ErrGraphNil)
}

func TestDFS_StartVertexNotFound(t *testing.T) {
	g, _ := core.NewGraph(core.WithDirected(true))

	result, err := dfs.DFS(g, "missing")

	mustNilState(t, result, true, "DFS missing-start result")
	mustErrorIs(t, err, dfs.ErrStartVertexNotFound)
}

func TestDFS_InvalidOptionNilContext(t *testing.T) {
	g, _ := core.NewGraph(core.WithDirected(true))
	_ = g.AddVertex("A")

	result, err := dfs.DFS(g, "A", dfs.WithContext(nil))

	mustNilState(t, result, true, "DFS invalid nil-context result")
	mustErrorIs(t, err, dfs.ErrOptionViolation)
}

func TestDFS_InvalidOptionNegativeMaxDepth(t *testing.T) {
	g, _ := core.NewGraph(core.WithDirected(true))
	_ = g.AddVertex("A")

	result, err := dfs.DFS(g, "A", dfs.WithMaxDepth(dfs.NoDepthLimit-1))

	mustNilState(t, result, true, "DFS invalid negative-depth result")
	mustErrorIs(t, err, dfs.ErrOptionViolation)
}

func TestDFS_SingleVertex_NoEdges(t *testing.T) {
	g, _ := core.NewGraph(core.WithDirected(true))
	mustNoError(t, g.AddVertex("X"))

	result, err := dfs.DFS(g, "X")
	mustNoError(t, err)

	mustNilState(t, result, false, "DFS single-vertex result")
	mustEqualSlice(t, result.Order, []string{"X"})
	mustEqualBoolMap(t, result.Visited, map[string]bool{
		"X": true,
	})
	mustEqualIntMap(t, result.Depth, map[string]int{
		"X": 0,
	})
	mustEqualStringMap(t, result.Parent, map[string]string{})
	mustEqualInt(t, result.SkippedNeighbors, 0, "")
}

func TestDFS_SelfLoop(t *testing.T) {
	g, _ := core.NewGraph(core.WithDirected(true), core.WithLoops())
	mustNoError(t, g.AddVertex("A"))

	edgeID, err := g.AddEdge("A", "A", 0)
	mustNoError(t, err)
	mustEqualBool(t, edgeID != "", true, "expected a non-empty edge ID for the self-loop")

	result, err := dfs.DFS(g, "A")
	mustNoError(t, err)

	mustNilState(t, result, false, "DFS self-loop result")
	mustEqualSlice(t, result.Order, []string{"A"})
	mustEqualBoolMap(t, result.Visited, map[string]bool{
		"A": true,
	})
	mustEqualIntMap(t, result.Depth, map[string]int{
		"A": 0,
	})
	mustEqualStringMap(t, result.Parent, map[string]string{})
}

func TestDFS_ChainDepthParentExactOrder(t *testing.T) {
	g, _ := core.NewGraph(core.WithDirected(true))
	_, _ = g.AddEdge("A", "B", 0)
	_, _ = g.AddEdge("B", "C", 0)

	result, err := dfs.DFS(g, "A")
	mustNoError(t, err)

	mustNilState(t, result, false, "DFS chain result")
	mustEqualSlice(t, result.Order, []string{"C", "B", "A"})
	mustEqualBoolMap(t, result.Visited, map[string]bool{
		"A": true,
		"B": true,
		"C": true,
	})
	mustEqualIntMap(t, result.Depth, map[string]int{
		"A": 0,
		"B": 1,
		"C": 2,
	})
	mustEqualStringMap(t, result.Parent, map[string]string{
		"B": "A",
		"C": "B",
	})
}

func TestDFS_DisconnectedSingleSource(t *testing.T) {
	g, _ := core.NewGraph(core.WithDirected(true))
	_, _ = g.AddEdge("A", "B", 0)
	mustNoError(t, g.AddVertex("C"))

	result, err := dfs.DFS(g, "A")
	mustNoError(t, err)

	mustNilState(t, result, false, "DFS disconnected single-source result")
	mustEqualSlice(t, result.Order, []string{"B", "A"})
	mustEqualBoolMap(t, result.Visited, map[string]bool{
		"A": true,
		"B": true,
	})
	mustEqualIntMap(t, result.Depth, map[string]int{
		"A": 0,
		"B": 1,
	})
	mustEqualStringMap(t, result.Parent, map[string]string{
		"B": "A",
	})
}

func TestDFS_FullTraversal_IgnoresMissingStartVertex(t *testing.T) {
	g, _ := core.NewGraph(core.WithDirected(true))
	_, _ = g.AddEdge("A", "B", 0)
	mustNoError(t, g.AddVertex("C"))

	result, err := dfs.DFS(g, "missing-start", dfs.WithFullTraversal())
	mustNoError(t, err)

	mustNilState(t, result, false, "DFS full-traversal missing-start result")
	mustEqualSlice(t, result.Order, []string{"B", "A", "C"})
	mustEqualBoolMap(t, result.Visited, map[string]bool{
		"A": true,
		"B": true,
		"C": true,
	})
	mustEqualIntMap(t, result.Depth, map[string]int{
		"A": 0,
		"B": 1,
		"C": 0,
	})
	mustEqualStringMap(t, result.Parent, map[string]string{
		"B": "A",
	})
	mustHaveNoKey(t, result.Parent, "A")
	mustHaveNoKey(t, result.Parent, "C")
}

func TestDFS_MaxDepthZero(t *testing.T) {
	g, _ := core.NewGraph(core.WithDirected(true))
	_, _ = g.AddEdge("A", "B", 0)
	_, _ = g.AddEdge("B", "C", 0)

	result, err := dfs.DFS(g, "A", dfs.WithMaxDepth(0))
	mustNoError(t, err)

	mustNilState(t, result, false, "DFS max-depth-zero result")
	mustEqualSlice(t, result.Order, []string{"A"})
	mustEqualBoolMap(t, result.Visited, map[string]bool{
		"A": true,
	})
	mustEqualIntMap(t, result.Depth, map[string]int{
		"A": 0,
	})
	mustEqualStringMap(t, result.Parent, map[string]string{})
}

func TestDFS_FilterNeighbor(t *testing.T) {
	g, _ := core.NewGraph(core.WithDirected(true))
	_, _ = g.AddEdge("A", "B", 0)
	_, _ = g.AddEdge("A", "C", 0)

	result, err := dfs.DFS(g, "A", dfs.WithFilterNeighbor(func(vertexID string) bool {
		return vertexID != "C"
	}))
	mustNoError(t, err)

	mustNilState(t, result, false, "DFS filter-neighbor result")
	mustEqualSlice(t, result.Order, []string{"B", "A"})
	mustEqualBoolMap(t, result.Visited, map[string]bool{
		"A": true,
		"B": true,
	})
	mustEqualIntMap(t, result.Depth, map[string]int{
		"A": 0,
		"B": 1,
	})
	mustEqualStringMap(t, result.Parent, map[string]string{
		"B": "A",
	})
	mustEqualInt(t, result.SkippedNeighbors, 1, "")
}

func TestDFS_OnExitErrorPreservesCauseAndClearsOrder(t *testing.T) {
	g, _ := core.NewGraph(core.WithDirected(true))
	_, _ = g.AddEdge("A", "B", 0)

	hookErr := errors.New("stop on B exit")

	result, err := dfs.DFS(g, "A", dfs.WithOnExit(func(vertexID string) error {
		if vertexID == "B" {
			return hookErr
		}

		return nil
	}))

	mustNilState(t, result, false, "DFS on-exit-error result")
	mustErrorIs(t, err, hookErr)
	mustEqualSlice(t, result.Order, nil)
	mustEqualBoolMap(t, result.Visited, map[string]bool{
		"A": true,
		"B": true,
	})
	mustEqualIntMap(t, result.Depth, map[string]int{
		"A": 0,
		"B": 1,
	})
	mustEqualStringMap(t, result.Parent, map[string]string{
		"B": "A",
	})
}

func TestDFS_OnVisitErrorPreservesCauseAndClearsOrder(t *testing.T) {
	g := buildBinaryTree(3)

	var preOrder []string
	var postOrder []string

	hookErr := errors.New("stop on T-4 visit")

	result, err := dfs.DFS(
		g,
		"T-1",
		dfs.WithOnVisit(func(vertexID string) error {
			preOrder = append(preOrder, vertexID)

			if vertexID == "T-4" {
				return hookErr
			}

			return nil
		}),
		dfs.WithOnExit(func(vertexID string) error {
			postOrder = append(postOrder, vertexID)
			return nil
		}),
	)

	mustNilState(t, result, false, "DFS on-visit-error result")
	mustErrorIs(t, err, hookErr)

	mustEqualSlice(t, preOrder, []string{"T-1", "T-2", "T-4"})
	mustEqualSlice(t, postOrder, nil)
	mustEqualSlice(t, result.Order, nil)

	mustEqualBoolMap(t, result.Visited, map[string]bool{
		"T-1": true,
		"T-2": true,
		"T-4": true,
	})
	mustEqualIntMap(t, result.Depth, map[string]int{
		"T-1": 0,
		"T-2": 1,
		"T-4": 2,
	})
	mustEqualStringMap(t, result.Parent, map[string]string{
		"T-2": "T-1",
		"T-4": "T-2",
	})
}

func TestDFS_HooksExactPreAndPostOrder(t *testing.T) {
	g := buildBinaryTree(3)

	var preOrder []string
	var postOrder []string

	result, err := dfs.DFS(
		g,
		"T-1",
		dfs.WithOnVisit(func(vertexID string) error {
			preOrder = append(preOrder, vertexID)
			return nil
		}),
		dfs.WithOnExit(func(vertexID string) error {
			postOrder = append(postOrder, vertexID)
			return nil
		}),
	)
	mustNoError(t, err)

	mustNilState(t, result, false, "DFS hook-success result")
	mustEqualSlice(t, preOrder, []string{
		"T-1",
		"T-2",
		"T-4",
		"T-5",
		"T-3",
		"T-6",
		"T-7",
	})
	mustEqualSlice(t, postOrder, []string{
		"T-4",
		"T-5",
		"T-2",
		"T-6",
		"T-7",
		"T-3",
		"T-1",
	})
	mustEqualSlice(t, result.Order, postOrder)
}

func TestDFS_CancellationImmediate(t *testing.T) {
	g := buildChain(100)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	result, err := dfs.DFS(g, "N0", dfs.WithContext(ctx))

	mustNilState(t, result, false, "DFS cancellation result")
	mustErrorIs(t, err, context.Canceled)
	mustEqualSlice(t, result.Order, nil)
	mustEqualBoolMap(t, result.Visited, map[string]bool{})
	mustEqualIntMap(t, result.Depth, map[string]int{})
	mustEqualStringMap(t, result.Parent, map[string]string{})
	mustEqualInt(t, result.SkippedNeighbors, 0, "")
}

func TestDFS_FullTraversalForestSemantics(t *testing.T) {
	g, _ := core.NewGraph(core.WithDirected(true))
	_, _ = g.AddEdge("A", "B", 0)
	_, _ = g.AddEdge("C", "D", 0)
	mustNoError(t, g.AddVertex("E"))

	result, err := dfs.DFS(g, "", dfs.WithFullTraversal())
	mustNoError(t, err)

	mustNilState(t, result, false, "DFS forest result")
	mustEqualSlice(t, result.Order, []string{"B", "A", "D", "C", "E"})
	mustEqualBoolMap(t, result.Visited, map[string]bool{
		"A": true,
		"B": true,
		"C": true,
		"D": true,
		"E": true,
	})
	mustEqualIntMap(t, result.Depth, map[string]int{
		"A": 0,
		"B": 1,
		"C": 0,
		"D": 1,
		"E": 0,
	})
	mustEqualStringMap(t, result.Parent, map[string]string{
		"B": "A",
		"D": "C",
	})
	mustHaveNoKey(t, result.Parent, "A")
	mustHaveNoKey(t, result.Parent, "C")
	mustHaveNoKey(t, result.Parent, "E")
}

func TestDFS_UndirectedRegression_StartFromToEndpointVisitsOppositeEndpoint(t *testing.T) {
	g, _ := core.NewGraph()
	_, _ = g.AddEdge("A", "B", 0)

	result, err := dfs.DFS(g, "B")
	mustNoError(t, err)

	mustNilState(t, result, false, "DFS undirected regression result")
	mustEqualSlice(t, result.Order, []string{"A", "B"})
	mustEqualBoolMap(t, result.Visited, map[string]bool{
		"A": true,
		"B": true,
	})
	mustEqualIntMap(t, result.Depth, map[string]int{
		"B": 0,
		"A": 1,
	})
	mustEqualStringMap(t, result.Parent, map[string]string{
		"A": "B",
	})
}

func TestDFS_MixedGraph_PerEdgeDirectionIsRespected(t *testing.T) {
	g, _ := core.NewMixedGraph(core.WithDirected(true))

	// Directed edge B->A.
	_, _ = g.AddEdge("B", "A", 0, core.WithEdgeDirected(true))

	// Undirected edge B--C.
	_, _ = g.AddEdge("B", "C", 0, core.WithEdgeDirected(false))

	result, err := dfs.DFS(g, "C")
	mustNoError(t, err)

	mustNilState(t, result, false, "DFS mixed-edge result")
	mustEqualSlice(t, result.Order, []string{"A", "B", "C"})
	mustEqualBoolMap(t, result.Visited, map[string]bool{
		"A": true,
		"B": true,
		"C": true,
	})
	mustEqualIntMap(t, result.Depth, map[string]int{
		"C": 0,
		"B": 1,
		"A": 2,
	})
	mustEqualStringMap(t, result.Parent, map[string]string{
		"B": "C",
		"A": "B",
	})
}

func TestDFS_LargeChain_PostOrderDepthParent(t *testing.T) {
	const vertexCount = 10

	g := buildChain(vertexCount)
	result, err := dfs.DFS(g, "N0")
	mustNoError(t, err)

	mustNilState(t, result, false, "DFS large-chain result")

	expectedOrder := make([]string, vertexCount)
	for index := vertexCount - 1; index >= 0; index-- {
		expectedOrder[vertexCount-1-index] = fmt.Sprintf("N%d", index)
	}

	mustEqualSlice(t, result.Order, expectedOrder)
	mustEqualInt(t, result.Depth[mustFmt(t, "N%d", vertexCount-1)], vertexCount-1, "")
	mustEqualString(t, result.Parent[mustFmt(t, "N%d", vertexCount-1)], mustFmt(t, "N%d", vertexCount-2), "")
}

func TestDFS_BinaryTree_VisitsAllVerticesAndFinishesRootLast(t *testing.T) {
	const depth = 5

	g := buildBinaryTree(depth)
	result, err := dfs.DFS(g, "T-1")
	mustNoError(t, err)

	mustNilState(t, result, false, "DFS binary-tree result")

	wantVertexCount := (1 << depth) - 1

	mustEqualInt(t, len(result.Visited), wantVertexCount, "")
	mustEqualInt(t, len(result.Order), wantVertexCount, "")
	mustEqualString(t, result.Order[len(result.Order)-1], "T-1", "")

	for index := 1; index <= wantVertexCount; index++ {
		vertexID := mustFmt(t, "T-%d", index)

		if !result.Visited[vertexID] {
			t.Fatalf("expected vertex %q to be visited, got visited=%v", vertexID, result.Visited)
		}
	}

	mustEqualInt(t, result.Depth["T-16"], 4, "")
	mustEqualString(t, result.Parent["T-16"], "T-8", "")
}
