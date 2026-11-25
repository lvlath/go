// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package mst_test

import (
	"context"
	"errors"
	"math"
	"sync/atomic"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/mst"
)

const mstWeightTolerance = 1e-12

// testDisjointSet is a minimal test-only union-find used by structural assertions and exact oracles.
// It intentionally stays independent from the production disjointSet to avoid testing the kernel with itself.
//
// Implementation:
//   - Stage 1: newTestDisjointSet creates one singleton set per vertex.
//   - Stage 2: find compresses paths.
//   - Stage 3: union merges lexicographically larger roots under smaller roots for stable test behavior.
//
// Behavior highlights:
//   - Used only in tests.
//   - Does not expose production package internals.
//   - Panics are avoided; invalid test fixtures fail through callers.
//
// Complexity:
//   - find/union are amortized near O(1), storage O(V).
//
// AI-Hints:
//   - Do not import private production DSU into external tests; oracle checks must stay independent.
type testDisjointSet struct {
	parent map[string]string
}

func newTestDisjointSet(vertices []string) *testDisjointSet {
	set := &testDisjointSet{
		parent: make(map[string]string, len(vertices)),
	}

	for _, vertexID := range vertices {
		set.parent[vertexID] = vertexID
	}

	return set
}

func (set *testDisjointSet) find(vertexID string) string {
	root := vertexID
	for set.parent[root] != root {
		root = set.parent[root]
	}

	for vertexID != root {
		next := set.parent[vertexID]
		set.parent[vertexID] = root
		vertexID = next
	}

	return root
}

func (set *testDisjointSet) union(left string, right string) bool {
	leftRoot := set.find(left)
	rightRoot := set.find(right)
	if leftRoot == rightRoot {
		return false
	}

	if leftRoot > rightRoot {
		leftRoot, rightRoot = rightRoot, leftRoot
	}

	set.parent[rightRoot] = leftRoot
	return true
}

func mustNoError(t *testing.T, err error, op string) {
	t.Helper()
	if err != nil {
		t.Fatalf("%s: unexpected error: %v", op, err)
	}
}

func mustErrorIs(t *testing.T, err error, target error, op string) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Fatalf("%s: want errors.Is(err, %v); got %v", op, target, err)
	}
}

func mustEqualInt(t *testing.T, got int, want int, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %d, want %d", op, got, want)
	}
}

func mustFloatClose(t *testing.T, got float64, want float64, op string) {
	t.Helper()
	if math.IsNaN(got) || math.IsNaN(want) {
		t.Fatalf("%s: NaN comparison got=%v want=%v", op, got, want)
	}
	if math.IsInf(got, 0) || math.IsInf(want, 0) {
		if got != want {
			t.Fatalf("%s: got %v, want %v", op, got, want)
		}
		return
	}
	if math.Abs(got-want) > mstWeightTolerance {
		t.Fatalf("%s: got %.12g, want %.12g, tolerance %.12g", op, got, want, mstWeightTolerance)
	}
}

func mustWeightedGraph(t *testing.T) *core.Graph {
	t.Helper()
	graph, err := core.NewGraph(core.WithWeighted())
	mustNoError(t, err, "core.NewGraph(core.WithWeighted())")
	return graph
}

func mustWeightedMultiGraph(t *testing.T) *core.Graph {
	t.Helper()

	graph, err := core.NewGraph(core.WithWeighted(), core.WithMultiEdges())
	mustNoError(t, err, "core.NewGraph(core.WithWeighted(), core.WithMultiEdges())")

	return graph
}

func mustCorruptEdgeWeight(t *testing.T, graph *core.Graph, edgeID string, weight float64) {
	t.Helper()
	edge, err := graph.GetEdge(edgeID)
	mustNoError(t, err, "Graph.GetEdge")
	edge.Weight = weight
}

func mustEqualString(t *testing.T, got string, want string, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %q, want %q", op, got, want)
	}
}

func mustValidStrictMST(t *testing.T, graph *core.Graph, result *mst.Result, wantWeight float64) {
	t.Helper()

	if result == nil {
		t.Fatalf("strict MST: nil result")
	}

	vertices := graph.Vertices()
	mustEqualInt(t, result.VertexCount, len(vertices), "strict MST vertex count")
	mustEqualInt(t, result.ComponentCount, 1, "strict MST component count")
	mustEqualInt(t, len(result.Edges), maxTreeEdges(len(vertices)), "strict MST edge count")
	mustFloatClose(t, result.TotalWeight, wantWeight, "strict MST total weight")

	mustAcyclicSelectedEdges(t, graph, result.Edges, "strict MST")
	mustConnectedSelectedEdges(t, vertices, result.Edges, "strict MST")
	mustResultWeightMatchesEdges(t, result, "strict MST")
}

func mustValidForest(t *testing.T, graph *core.Graph, result *mst.Result, wantComponents int, wantWeight float64) {
	t.Helper()

	if result == nil {
		t.Fatalf("forest: nil result")
	}

	vertices := graph.Vertices()
	mustEqualInt(t, result.VertexCount, len(vertices), "forest vertex count")
	mustEqualInt(t, result.ComponentCount, wantComponents, "forest component count")
	mustEqualInt(t, len(result.ComponentRoots), wantComponents, "forest root count")
	mustEqualInt(t, len(result.Edges), len(vertices)-wantComponents, "forest edge count")
	mustFloatClose(t, result.TotalWeight, wantWeight, "forest total weight")

	mustAcyclicSelectedEdges(t, graph, result.Edges, "forest")
	mustResultWeightMatchesEdges(t, result, "forest")
}

func mustAcyclicSelectedEdges(t *testing.T, graph *core.Graph, edges []core.Edge, op string) {
	t.Helper()

	vertices := graph.Vertices()
	set := newTestDisjointSet(vertices)

	knownEdges := make(map[string]struct{}, graph.EdgeCount())
	for _, edge := range graph.Edges() {
		knownEdges[edge.ID] = struct{}{}
	}

	for _, edge := range edges {
		if _, ok := knownEdges[edge.ID]; !ok {
			t.Fatalf("%s: selected edge %q does not belong to graph", op, edge.ID)
		}
		if !graph.HasVertex(edge.From) || !graph.HasVertex(edge.To) {
			t.Fatalf("%s: selected edge %q has unknown endpoint %q -> %q", op, edge.ID, edge.From, edge.To)
		}
		if edge.From == edge.To {
			t.Fatalf("%s: selected self-loop edge %q", op, edge.ID)
		}
		if !set.union(edge.From, edge.To) {
			t.Fatalf("%s: selected edge %q creates a cycle", op, edge.ID)
		}
	}
}

func mustConnectedSelectedEdges(t *testing.T, vertices []string, edges []core.Edge, op string) {
	t.Helper()

	if len(vertices) == 0 {
		t.Fatalf("%s: strict MST cannot be validated over empty vertex set", op)
	}
	if len(vertices) == 1 {
		if len(edges) != 0 {
			t.Fatalf("%s: single-vertex MST got %d edges", op, len(edges))
		}
		return
	}

	set := newTestDisjointSet(vertices)
	for _, edge := range edges {
		set.union(edge.From, edge.To)
	}

	root := set.find(vertices[0])
	for _, vertexID := range vertices[1:] {
		if set.find(vertexID) != root {
			t.Fatalf("%s: vertex %q is not connected to root %q", op, vertexID, vertices[0])
		}
	}
}

func mustResultWeightMatchesEdges(t *testing.T, result *mst.Result, op string) {
	t.Helper()

	var total float64
	for _, edge := range result.Edges {
		total += edge.Weight
	}

	mustFloatClose(t, total, result.TotalWeight, op+" edge sum")
}

func mustEqualEdgeIDs(t *testing.T, edges []core.Edge, want []string, op string) {
	t.Helper()

	if len(edges) != len(want) {
		t.Fatalf("%s: got %d edges, want %d", op, len(edges), len(want))
	}

	for i, edge := range edges {
		if edge.ID != want[i] {
			t.Fatalf("%s: edge[%d] got ID %q, want %q", op, i, edge.ID, want[i])
		}
	}
}

func maxTreeEdges(vertexCount int) int {
	if vertexCount <= 1 {
		return 0
	}
	return vertexCount - 1
}

// steinerCheckpointContext cancels a real context at a chosen Err observation.
// It makes cooperative cancellation reproducible without sleep, racing graph writes, or wall-clock deadlines.
// The counter is atomic and the embedded context retains normal Done/Err semantics after cancellation.
type steinerCheckpointContext struct {
	context.Context
	cancel   context.CancelFunc
	cancelAt int64
	polls    atomic.Int64
}

// newSteinerCheckpointContext creates a fresh checkpoint-driven cancellation context.
// A zero checkpoint records observations without canceling; positive values cancel exactly once.
// Implementation:
//   - Stage 1: Derive a real cancellable context from Background.
//   - Stage 2: Record the desired observation index in test-owned state.
//   - Stage 3: Return the context and its cleanup callback to the caller.
//
// Behavior highlights:
//   - Done is closed by context.CancelFunc before Err reports context.Canceled.
//
// Inputs:
//   - cancelAt: one-based Err observation index, or zero to record the full successful run.
//
// Returns:
//   - *steinerCheckpointContext and context.CancelFunc; both belong to the calling test.
//
// Errors:
//   - None during construction.
//
// Determinism:
//   - Single synchronous algorithm calls produce an observation sequence independent of the clock.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This is a test context, not a production cancellation policy.
//
// AI-Hints:
//   - Derive the number of checkpoints from a successful run instead of hardcoding private line counts.
func newSteinerCheckpointContext(cancelAt int64) (*steinerCheckpointContext, context.CancelFunc) {
	// Use the standard cancellation primitive rather than fabricating inconsistent context errors.
	ctx, cancel := context.WithCancel(context.Background())
	return &steinerCheckpointContext{Context: ctx, cancel: cancel, cancelAt: cancelAt}, cancel
}

// Err observes one cooperative cancellation checkpoint and delegates to the real context.
// The configured observation closes Done through CancelFunc before returning the first cancellation error.
// Implementation:
//   - Stage 1: Increment the observation counter atomically.
//   - Stage 2: Trigger cancellation at the configured positive checkpoint.
//   - Stage 3: Return the standard context error unchanged.
//
// Behavior highlights:
//   - Once canceled, every later Err call continues to report context.Canceled.
//
// Inputs:
//   - Receiver: a test-owned checkpoint context; do not copy it after first use.
//
// Returns:
//   - error: nil before cancellation, context.Canceled afterwards.
//
// Errors:
//   - context.Canceled is returned by the embedded standard context.
//
// Determinism:
//   - Cancellation is driven by observation count, not scheduling delays.
//
// Complexity:
//   - Time O(1), Space O(1) per observation for this context without children.
//
// Notes:
//   - Atomic counting avoids a race if a future consumer observes Err concurrently.
//
// AI-Hints:
//   - Do not use a fabricated Err implementation that reports cancellation while Done remains open.
func (ctx *steinerCheckpointContext) Err() error {
	// Advance the counter even in recording mode so the test can discover all checkpoints.
	observation := ctx.polls.Add(1)
	if ctx.cancelAt > 0 && observation == ctx.cancelAt {
		ctx.cancel()
	}
	return ctx.Context.Err()
}

// mustSteinerBoundaryError checks a failed stage's sentinel identity and absence of a public result.
// It intentionally accepts only a SteinerResult pointer, avoiding reflection or typed-nil ambiguity.
// Implementation:
//   - Stage 1: Reject any published result on failure.
//   - Stage 2: Check the precise error identity using the shared sentinel assertion.
//
// Behavior highlights:
//   - Joined and wrapped errors are accepted through errors.Is.
//
// Inputs:
//   - t: active test; result: returned result; err: actual error; want: required sentinel; op: diagnostic label.
//
// Returns:
//   - None; the helper terminates the current test on a mismatch.
//
// Errors:
//   - Test failure for an unexpected non-nil result or missing error identity.
//
// Determinism:
//   - Fixed assertion order; no message parsing.
//
// Complexity:
//   - O(error-chain size) time and errors.Is traversal space.
//
// Notes:
//   - A non-nil partial result is a failure even when its accompanying error is correct.
//
// AI-Hints:
//   - Do not replace the nil-result check with IsNil on an unrelated result type.
func mustSteinerBoundaryError(t *testing.T, result *mst.SteinerResult, err, want error, op string) {
	t.Helper()
	if result != nil {
		t.Fatalf("%s: published a partial Steiner result: %+v", op, result)
	}
	mustErrorIs(t, err, want, op)
}
