// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
// Package core_test verifies core.Graph configuration, identity contracts, and cloning semantics.
//
// Purpose:
//   - Lock in option flags, vertex lifecycle rules, ID uniqueness under concurrency.
//   - Demonstrate read-only map snapshots and deep-copy behavior (no pointer aliasing).

package core_test

import (
	"testing"

	"github.com/lvlath/go/core"
)

// TestGraph_Options ASSERTS GraphOption flags are applied correctly.
//
// Implementation:
//   - Stage 1: Build a feature-rich graph via NewGraphFull().
//   - Stage 2: Assert Directed defaults to false.
//   - Stage 3: Assert Weighted is enabled.
//   - Stage 4: Assert empty vertex ID is absent.
//   - Stage 5: Assert WithDirected(true) overrides.
//   - Stage 6: Assert multi-edge policy rejects duplicates when disabled.
//
// Behavior highlights:
//   - Documents option semantics explicitly.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal on any contract mismatch.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Multi-edge rejection is a sentinel contract (ErrMultiEdgeNotAllowed).
//
// AI-Hints:
//   - Prefer option tests to stay minimal: assert flags and one representative behavior per flag.
func TestGraph_Options(t *testing.T) {
	g := NewGraphFull(t)

	MustEqualBool(t, g.Directed(), false, "Directed() default must be false (undirected)")
	MustEqualBool(t, g.Weighted(), true, "Weighted() must be true on NewGraphFull")
	MustEqualBool(t, g.HasVertex(VertexEmpty), false, "HasVertex(empty) must be false")

	dg := MustNewGraph(t, core.WithDirected(true))
	MustEqualBool(t, dg.Directed(), true, "WithDirected(true) must set Directed()==true")

	sg := MustNewGraph(t)
	_, err := sg.AddEdge(VertexX, VertexY, Weight0)
	MustErrorNil(t, err, "AddEdge(X,Y,0) first on default graph")

	_, err = sg.AddEdge(VertexX, VertexY, Weight0)
	MustErrorIs(t, err, core.ErrMultiEdgeNotAllowed, "AddEdge(X,Y,0) second on default graph")
}

// TestGraph_NewGraphRejectsNilOption ASSERTS constructor-level nil-option validation.
//
// Implementation:
// - Stage 1: Call NewGraph with a nil GraphOption.
// - Stage 2: Assert ErrNilGraphOption and nil graph result.
// - Stage 3: Call NewGraph with a valid option prefix followed by a nil GraphOption.
// - Stage 4: Assert the same sentinel and nil graph result.
//
// Behavior highlights:
// - Locks in fail-fast nil-option validation for constructor inputs.
// - Prevents panic-based option handling from re-entering the public contract.
//
// Inputs:
// - None.
//
// Returns:
// - None.
//
// Errors:
// - Fatal on mismatch.
//
// Determinism:
// - Deterministic.
//
// Complexity:
// - Time O(len(opts)), Space O(1).
//
// Notes:
// - This is a regression anchor for the public no-panic constructor contract.
//
// AI-Hints:
// - Nil option slots are invalid inputs, not no-ops.
func TestGraph_NewGraphRejectsNilOption(t *testing.T) {
	var nilOpt core.GraphOption

	g, err := core.NewGraph(nilOpt)
	MustErrorIs(t, err, core.ErrNilGraphOption, "NewGraph(nil GraphOption)")
	if g != nil {
		t.Fatalf("NewGraph(nil GraphOption): got non-nil graph on error")
	}

	g, err = core.NewGraph(core.WithWeighted(), nilOpt)
	MustErrorIs(t, err, core.ErrNilGraphOption, "NewGraph(WithWeighted(), nil GraphOption)")
	if g != nil {
		t.Fatalf("NewGraph(WithWeighted(), nil GraphOption): got non-nil graph on error")
	}
}

// TestGraph_StatsSnapshot VERIFIES GraphStats matches graph counts, flags, and directed/undirected tallies.
// Implementation:
//   - Stage 1: Create a weighted mixed graph with undirected default.
//   - Stage 2: Add vertices explicitly for deterministic VertexCount.
//   - Stage 3: Add one undirected edge and one directed override edge.
//   - Stage 4: Call Stats() and assert counts/flags/tallies.
//
// Behavior highlights:
//   - Locks in Stats() as a coherent diagnostic snapshot (O(V+E)).
//
// Inputs:
//   - None (uses package constants).
//
// Returns:
//   - None.
//
// Errors:
//   - Propagates sentinels from AddVertex/AddEdge.
//
// Determinism:
//   - Deterministic for a fixed single-threaded graph state.
//
// Complexity:
//   - Time O(V+E) if Stats walks catalogs, Space O(1) extra.
//
// Notes:
//   - DirectedDefault describes the graph default, not the presence of directed edges.
//
// AI-Hints:
//   - Treat Stats() as best-effort for metrics; avoid correctness-critical dependence under concurrent mutation.
func TestGraph_StatsSnapshot(t *testing.T) {
	// Stage 1: Create a weighted mixed graph with an explicit undirected default.
	g, _ := core.NewGraph(core.WithDirected(false), core.WithWeighted(), core.WithMixedEdges())

	// Stage 2: Add vertices explicitly so VertexCount is deterministic.
	MustErrorNil(t, g.AddVertex(VertexA), "AddVertex(A) setup for Stats()")
	MustErrorNil(t, g.AddVertex(VertexB), "AddVertex(B) setup for Stats()")
	MustErrorNil(t, g.AddVertex(VertexC), "AddVertex(C) setup for Stats()")

	// Stage 3: Add one undirected edge (default) and one directed override edge.
	_, err := g.AddEdge(VertexA, VertexB, Weight1)
	MustErrorNil(t, err, "AddEdge(A,B,1) undirected default on mixed graph")
	_, err = g.AddEdge(VertexB, VertexC, Weight2, core.WithEdgeDirected(true))
	MustErrorNil(t, err, "AddEdge(B,C,2,WithEdgeDirected(true)) on mixed graph")

	// Stage 4: Read stats snapshot.
	s := g.Stats()

	// Stage 4: Counts must match public counters.
	MustEqualInt(t, s.VertexCount, g.VertexCount(), "Stats.VertexCount must match VertexCount()")
	MustEqualInt(t, s.EdgeCount, g.EdgeCount(), "Stats.EdgeCount must match EdgeCount()")

	// Stage 4: Flags must reflect constructor options.
	MustEqualBool(t, s.DirectedDefault, false, "Stats.DirectedDefault must be false for WithDirected(false)")
	MustEqualBool(t, s.Weighted, true, "Stats.Weighted must be true for WithWeighted()")
	MustEqualBool(t, s.MixedMode, true, "Stats.MixedMode must be true for WithMixedEdges()")
	MustEqualBool(t, s.AllowsMulti, false, "Stats.AllowsMulti must be false when WithMultiEdges() is not set")
	MustEqualBool(t, s.AllowsLoops, false, "Stats.AllowsLoops must be false when WithLoops() is not set")

	// Stage 4: Directed/undirected edge tallies must match this construction.
	MustEqualInt(t, s.DirectedEdgeCount, Count1, "Stats.DirectedEdgeCount must be 1 (one override-directed edge)")
	MustEqualInt(t, s.UndirectedEdgeCount, Count1, "Stats.UndirectedEdgeCount must be 1 (one default-undirected edge)")
	MustEqualInt(t, s.EdgeCount, Count2, "Stats.EdgeCount must be 2 in this setup")
}

// TestGraph_LoopedAndMixedEdgesAccessors verifies immutable policy accessors.
// Looped() and MixedEdges() report immutable construction-time policy flags.
func TestGraph_LoopedAndMixedEdgesAccessors(t *testing.T) {
	plain := MustNewGraph(t)
	MustEqualBool(t, plain.Looped(), false, "plain.Looped()")
	MustEqualBool(t, plain.MixedEdges(), false, "plain.MixedEdges()")

	configured := MustNewMixedGraph(t, core.WithLoops())
	MustEqualBool(t, configured.Looped(), true, "configured.Looped()")
	MustEqualBool(t, configured.MixedEdges(), true, "configured.MixedEdges()")
}

// TestVertex_IsNilTypedNil verifies reflect-free typed-nil classification through core.Nilable.
// Typed nil *Vertex implements core.Nilable safely without reflection-heavy callers.
func TestVertex_IsNilTypedNil(t *testing.T) {
	var v *core.Vertex
	var n core.Nilable = v

	MustEqualBool(t, v.IsNil(), true, "typed nil *Vertex IsNil")
	MustEqualBool(t, n.IsNil(), true, "typed nil *Vertex through Nilable")
	MustEqualBool(t, (&core.Vertex{ID: VertexA}).IsNil(), false, "non-nil *Vertex IsNil")
}

// TestGraph_ClearPreservesFlagsAndResetsState VERIFIES Clear() empties the graph but preserves flags.
// Implementation:
//   - Stage 1: Create a configured graph and add at least one edge.
//   - Stage 2: Call Clear().
//   - Stage 3: Assert counts are zero and configuration flags are unchanged.
//   - Stage 4: Assert edge ID counter resets ("e1" is returned for the first new edge).
//
// Behavior highlights:
//   - Clear() is a topology reset, not a configuration reset.
//
// Inputs:
//   - None (uses package constants).
//
// Returns:
//   - None.
//
// Errors:
//   - Propagates any sentinels from AddEdge.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1) expected, Space O(1) extra.
//
// Notes:
//   - "e1" reset is a documented ID contract (types.go + doc.go).
//
// AI-Hints:
//   - Use Clear() to reuse configured graphs without reallocating options repeatedly.
func TestGraph_ClearPreservesFlagsAndResetsState(t *testing.T) {
	// Stage 1: Create a configured graph and add an edge to advance the internal ID counter.
	g, _ := core.NewGraph(core.WithDirected(true), core.WithWeighted(), core.WithMultiEdges())

	// Add one edge to ensure the graph is non-empty before Clear().
	_, err := g.AddEdge(VertexA, VertexB, Weight5)
	MustErrorNil(t, err, "AddEdge(A,B,5) setup for Clear()")

	// Stage 2: Clear the graph.
	g.Clear()

	// Stage 3: Counts must be zero.
	MustEqualInt(t, g.VertexCount(), Count0, "VertexCount() after Clear()")
	MustEqualInt(t, g.EdgeCount(), Count0, "EdgeCount() after Clear()")

	// Stage 3: Flags must be preserved.
	MustEqualBool(t, g.Directed(), true, "Directed() must be preserved after Clear()")
	MustEqualBool(t, g.Weighted(), true, "Weighted() must be preserved after Clear()")
	MustEqualBool(t, g.Multigraph(), true, "Multigraph() must be preserved after Clear()")

	// Stage 4: First edge ID after Clear must reset to "e1".
	eid, err := g.AddEdge(VertexA, VertexB, Weight5)
	MustErrorNil(t, err, "AddEdge(A,B,5) after Clear()")
	MustEqualString(t, eid, EdgeIDFirst, "first edge ID after Clear() must be EdgeIDFirst")
}
