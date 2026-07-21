// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package bfs_test

import (
	"context"
	"errors"
	"testing"

	"github.com/lvlath/go/bfs"
	"github.com/lvlath/go/core"
)

// AI-HINTS (file):
//   - Tests are the arbiter of the math/contract: do not change correct behavior to satisfy a wrong test.
//   - Error protocol MUST use errors.Is only; never compare error strings.
//   - Determinism anchors rely on core's published ordering:
//       - Vertices() are lex-sorted,
//       - NeighborIDs() are unique and lex-sorted.
//   - Forbidden pattern: calling t.Fatal/t.FailNow inside a goroutine. Use channels and fail in the main goroutine.
//   - Directed reachability anchors MUST use core.WithDirected(true); core defaults to undirected graphs.

// --- Validation -------------------------------------------------------------

func TestBFS_Validation(t *testing.T) {
	t.Run("nil graph", func(t *testing.T) {
		_, err := bfs.BFS(nil, "A")
		mustErrorIs(t, err, bfs.ErrGraphNil)
	})

	t.Run("unknown start", func(t *testing.T) {
		g, _ := core.NewGraph()
		_, err := bfs.BFS(g, "missing")
		mustErrorIs(t, err, bfs.ErrStartVertexNotFound)
	})

	t.Run("weighted graph unsupported", func(t *testing.T) {
		g, _ := core.NewGraph(core.WithWeighted())
		mustNoError(t, g.AddVertex("A"))

		_, err := bfs.BFS(g, "A")
		mustErrorIs(t, err, bfs.ErrWeightedGraph)
	})

	t.Run("invalid option - nil option", func(t *testing.T) {
		g, _ := core.NewGraph()
		mustNoError(t, g.AddVertex("A"))

		var opt bfs.Option // nil option value
		_, err := bfs.BFS(g, "A", opt)
		mustErrorIs(t, err, bfs.ErrOptionViolation)
	})

	t.Run("invalid option - depth below MaxDepthUnlimited", func(t *testing.T) {
		g, _ := core.NewGraph()
		mustNoError(t, g.AddVertex("A"))

		const invalidDepth = bfs.MaxDepthUnlimited - 1
		_, err := bfs.BFS(g, "A", bfs.WithMaxDepth(invalidDepth))
		mustErrorIs(t, err, bfs.ErrOptionViolation)
	})
}

// --- Medium ----------------------------------------------------------------

func TestBFS_Medium_ShortestPathAnchor(t *testing.T) {
	// Directed graph to make reachability and parent choices explicit.
	g, _ := core.NewGraph(core.WithDirected(true))

	_, err := g.AddEdge("A", "B", 0)
	mustNoError(t, err)
	_, err = g.AddEdge("A", "C", 0)
	mustNoError(t, err)
	_, err = g.AddEdge("B", "D", 0)
	mustNoError(t, err)
	_, err = g.AddEdge("C", "D", 0)
	mustNoError(t, err)

	res, err := bfs.BFS(g, "A")
	mustNoError(t, err)

	mustEqualBool(t, res.StartID == "A", true, "StartID=%q want %q", res.StartID, "A")
	mustEqualSlice(t, res.Order, []string{"A", "B", "C", "D"})

	mustEqualIntMap(t, res.Depth, map[string]int{
		"A": 0,
		"B": 1,
		"C": 1,
		"D": 2,
	})

	// Parent selection is deterministic: D is discovered from B first.
	mustEqualStringMap(t, res.Parent, map[string]string{
		"B": "A",
		"C": "A",
		"D": "B",
	})

	path, err := res.PathTo("D")
	mustNoError(t, err)
	mustEqualSlice(t, path, []string{"A", "B", "D"})
}

func TestBFS_Medium_DeterminismAnchor_NeighborIDsLexOrder(t *testing.T) {
	// This test anchors the tie-break rule:
	// core.NeighborIDs() returns unique neighbor IDs sorted lex asc.
	// Even if edges are added out-of-order, BFS must follow NeighborIDs order.

	g, _ := core.NewGraph(core.WithDirected(true))

	_, err := g.AddEdge("A", "C", 0)
	mustNoError(t, err)
	_, err = g.AddEdge("A", "B", 0)
	mustNoError(t, err)

	res, err := bfs.BFS(g, "A")
	mustNoError(t, err)

	// Neighbors are B then C (lex order), so BFS order is A, B, C.
	mustEqualSlice(t, res.Order, []string{"A", "B", "C"})
}

// --- Special ---------------------------------------------------------------

func TestBFS_Special_MaxDepthInclusive(t *testing.T) {
	// A -> B -> C directed chain.
	g, _ := core.NewGraph(core.WithDirected(true))
	_, err := g.AddEdge("A", "B", 0)
	mustNoError(t, err)
	_, err = g.AddEdge("B", "C", 0)
	mustNoError(t, err)

	// MaxDepth == 0 means "root only".
	res0, err := bfs.BFS(g, "A", bfs.WithMaxDepth(0))
	mustNoError(t, err)
	mustEqualSlice(t, res0.Order, []string{"A"})
	mustEqualBool(t, res0.Visited["B"], false, "B must not be visited at MaxDepth=0")

	// MaxDepth == 1 means we visit depth 0 and depth 1, but do not expand depth 1.
	res1, err := bfs.BFS(g, "A", bfs.WithMaxDepth(1))
	mustNoError(t, err)
	mustEqualSlice(t, res1.Order, []string{"A", "B"})
	mustEqualBool(t, res1.Visited["C"], false, "C must not be visited at MaxDepth=1")

	// MaxDepthUnlimited means no limit.
	resU, err := bfs.BFS(g, "A", bfs.WithMaxDepth(bfs.MaxDepthUnlimited))
	mustNoError(t, err)
	mustEqualSlice(t, resU.Order, []string{"A", "B", "C"})
}

func TestBFS_Special_DirectedReachabilityAnchor(t *testing.T) {
	// A -> B directed. Start from B, A is unreachable.
	g, _ := core.NewGraph(core.WithDirected(true))
	_, err := g.AddEdge("A", "B", 0)
	mustNoError(t, err)

	res, err := bfs.BFS(g, "B")
	mustNoError(t, err)

	mustEqualSlice(t, res.Order, []string{"B"})
	_, ok := res.Depth["A"]
	mustEqualBool(t, ok, false, "Depth must not contain unreachable vertex A")

	_, err = res.PathTo("A")
	mustErrorIs(t, err, bfs.ErrNoPath)
}

func TestBFS_Special_PartialResult_OnVisitError(t *testing.T) {
	// A -> B and A -> C (directed).
	// NeighborIDs are lex-sorted: B then C.
	// BFS will enqueue both B and C when processing A.
	// If OnVisit errors at B, Order must be [A, B], while Visited may include C (enqueued but not visited).
	g, _ := core.NewGraph(core.WithDirected(true))
	_, err := g.AddEdge("A", "B", 0)
	mustNoError(t, err)
	_, err = g.AddEdge("A", "C", 0)
	mustNoError(t, err)

	stopErr := errors.New("test: stop")

	res, err := bfs.BFS(g, "A",
		bfs.WithOnVisit(func(id string) error {
			if id == "B" {
				return stopErr
			}
			return nil
		}),
	)
	mustErrorIs(t, err, stopErr)

	// "visit" is dequeue-time, and Order is appended before OnVisit runs,
	// so the vertex that triggers the error is included in Order.
	mustEqualSlice(t, res.Order, []string{"A", "B"})

	// Visited is set at enqueue time, so C may already be marked visited.
	mustEqualBool(t, res.Visited["C"], true, "C must be visited=true (enqueued) on partial result")
	mustEqualBool(t, res.Depth["C"] == 1, true, "Depth[C]=%d want 1", res.Depth["C"])
}

func TestBFS_Special_FilterNeighbor_Skipped(t *testing.T) {
	// A -> B and A -> C (directed). Filter out A->C.
	g, _ := core.NewGraph(core.WithDirected(true))
	_, err := g.AddEdge("A", "B", 0)
	mustNoError(t, err)
	_, err = g.AddEdge("A", "C", 0)
	mustNoError(t, err)

	res, err := bfs.BFS(g, "A",
		bfs.WithFilterNeighbor(func(currID, nbrID string) bool {
			return !(currID == "A" && nbrID == "C")
		}),
	)
	mustNoError(t, err)

	mustEqualSlice(t, res.Order, []string{"A", "B"})
	mustEqualBool(t, res.Skipped == 1, true, "Skipped=%d want 1", res.Skipped)

	_, ok := res.Depth["C"]
	mustEqualBool(t, ok, false, "C must not be in Depth when filtered out")
	mustEqualBool(t, res.Visited["C"], false, "C must not be visited when filtered out")
}

func TestBFS_Special_LoopsAndMultiEdges_NoDoubleEnqueue(t *testing.T) {
	// This test anchors: loops and parallel edges do not cause multiple enqueues of the same vertex.
	// Graph is unweighted (BFS rejects weighted graphs).
	g, _ := core.NewGraph(core.WithLoops(), core.WithMultiEdges())

	_, err := g.AddEdge("A", "A", 0) // loop
	mustNoError(t, err)
	_, err = g.AddEdge("A", "B", 0)
	mustNoError(t, err)
	_, err = g.AddEdge("A", "B", 0) // parallel edge
	mustNoError(t, err)

	res, err := bfs.BFS(g, "A")
	mustNoError(t, err)

	// NeighborIDs for A includes A (loop) and B; visited-check prevents re-enqueue of A.
	mustEqualSlice(t, res.Order, []string{"A", "B"})
}

func TestBFS_Special_Cancellation(t *testing.T) {
	// Cancellation is checked in the kernel loop; result may be partial.
	g, _ := core.NewGraph(core.WithDirected(true))
	for i := 0; i < 100; i++ {
		u := mustFmt(t, "v%d", i)
		v := mustFmt(t, "v%d", i+1)
		_, err := g.AddEdge(u, v, 0)
		mustNoError(t, err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	res, err := bfs.BFS(g, "v0", bfs.WithContext(ctx))
	mustErrorIs(t, err, context.Canceled)
	mustEqualBool(t, res != nil, true, "expected non-nil partial result on cancellation")
	mustEqualBool(t, res.StartID == "v0", true, "StartID=%q want %q", res.StartID, "v0")
}

func TestBFS_Special_ConcurrentRuns(t *testing.T) {
	// This test asserts that concurrent BFS calls do not interfere.
	// NOTE: Do not call t.Fatal inside goroutines; report errors via channels.
	g, _ := core.NewGraph(core.WithDirected(true))
	_, err := g.AddEdge("A", "B", 0)
	mustNoError(t, err)

	errCh := make(chan error, 2)
	for i := 0; i < 2; i++ {
		go func() {
			_, e := bfs.BFS(g, "A")
			errCh <- e
		}()
	}

	for i := 0; i < 2; i++ {
		if e := <-errCh; e != nil {
			t.Fatalf("concurrent run #%d: unexpected error %v", i, e)
		}
	}
}

// --- Components ------------------------------------------------------------

func TestComponents_WeakConnectivity_DirectedChain(t *testing.T) {
	// Directed chain A->B->C must still be a single weakly-connected component.
	g, _ := core.NewGraph(core.WithDirected(true))
	_, err := g.AddEdge("A", "B", 0)
	mustNoError(t, err)
	_, err = g.AddEdge("B", "C", 0)
	mustNoError(t, err)

	res, err := bfs.Components(context.Background(), g)
	mustNoError(t, err)

	mustEqualBool(t, res.UndirectedView, true, "UndirectedView must be true")
	mustEqualBool(t, res.Count == 1, true, "Count=%d want 1", res.Count)
	mustEqualBool(t, len(res.Components) == 1, true, "len(Components)=%d want 1", len(res.Components))

	mustEqualSlice(t, res.Components[0], []string{"A", "B", "C"})
}

func TestComponents_Cancellation_PartialResult(t *testing.T) {
	g, _ := core.NewGraph(core.WithDirected(true))
	_, err := g.AddEdge("A", "B", 0)
	mustNoError(t, err)
	_, err = g.AddEdge("B", "C", 0)
	mustNoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	res, err := bfs.Components(ctx, g)
	mustErrorIs(t, err, context.Canceled)
	mustEqualBool(t, res != nil, true, "expected non-nil partial result on cancellation")
	mustEqualBool(t, res.UndirectedView, true, "UndirectedView must be true")
}
