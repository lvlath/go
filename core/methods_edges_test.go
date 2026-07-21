package core_test

import (
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/lvlath/go/core"
)

// TestGraph_AddEdgeConstraints VERIFIES AddEdge constraint enforcement for weights, loops, multi-edges.
// Implementation:
//   - Stage 1: Assert unweighted graph rejects non-zero weight (ErrBadWeight).
//   - Stage 2: Assert weighted graph accepts non-zero weight.
//   - Stage 3: Assert loop-disabled graph rejects self-loop (ErrLoopNotAllowed).
//   - Stage 4: Assert loop-enabled graph accepts self-loop and returns non-empty edge ID.
//   - Stage 5: Assert multi-edge-disabled graph rejects parallel edge (ErrMultiEdgeNotAllowed).
//   - Stage 6: Assert multi-edge-enabled graph accepts parallel edges with distinct IDs.
//
// Behavior highlights:
//   - Fixes sentinel error mapping for invalid edge insertions.
//
// Inputs:
//   - None (uses package constants).
//
// Returns:
//   - None.
//
// Errors:
//   - core.ErrBadWeight when Weighted()==false and weight != 0.
//   - core.ErrLoopNotAllowed when Loops()==false and from==to.
//   - core.ErrMultiEdgeNotAllowed when MultiEdges()==false and endpoints duplicate an existing edge.
//
// Determinism:
//   - Deterministic (no randomness).
//
// Complexity:
//   - Time O(1) per AddEdge membership/constraint check (implementation-dependent), Space O(1) incremental per edge.
//
// Notes:
//   - This test does not assert edge ID format, only non-emptiness and uniqueness when required.
//
// AI-Hints:
//   - Keep constraint tests isolated by building a fresh graph per stage.
//   - Use constants for weights to avoid magic numerics in failure output.
func TestGraph_AddEdgeConstraints(t *testing.T) {
	// Stage 1: Unweighted graph rejects non-zero weight.
	// Create unweighted default graph.
	g := MustNewGraph(t)
	// Attempt to add a weighted edge on an unweighted graph.
	_, err := g.AddEdge(VertexA, VertexB, Weight5)
	// Enforce sentinel error contract.
	MustErrorIs(t, err, core.ErrBadWeight, "AddEdge(A,B,5) on unweighted graph")

	// Stage 2: Weighted graph accepts non-zero weight and creates the edge.
	// Create weighted graph.
	g, _ = core.NewGraph(core.WithWeighted())
	// Add weighted edge.
	_, err = g.AddEdge(VertexA, VertexB, Weight7)
	// Must succeed.
	MustErrorNil(t, err, "AddEdge(A,B,7) on weighted graph")
	// Membership query must succeed via adjacency.
	MustEqualBool(t, g.HasEdge(VertexA, VertexB), true, "HasEdge(A,B) after AddEdge(A,B,7)")

	// Stage 3: Default graph disallows self-loops.
	// Create default graph (loops disabled).
	g = MustNewGraph(t)
	// Attempt to add self-loop.
	_, err = g.AddEdge(VertexX, VertexX, Weight0)
	// Enforce sentinel error contract.
	MustErrorIs(t, err, core.ErrLoopNotAllowed, "AddEdge(X,X,0) when loops disabled")

	// Stage 4: Loop-enabled graph accepts self-loops.
	// Create loop-enabled graph.
	g, _ = core.NewGraph(core.WithLoops())
	// Add self-loop.
	loopID, err := g.AddEdge(VertexX, VertexX, Weight0)
	// Must succeed.
	MustErrorNil(t, err, "AddEdge(X,X,0) when loops enabled")
	// ID must be non-empty (format is not a contract here).
	MustNotEqualString(t, loopID, "", "AddEdge(X,X,0) must return non-empty edge ID")
	// Membership query must be true.
	MustEqualBool(t, g.HasEdge(VertexX, VertexX), true, "HasEdge(X,X) after adding self-loop")

	// Stage 5: Multi-edge disallowed by default (second edge with same endpoints must error).
	// Create default graph (multi-edges disabled).
	g = MustNewGraph(t)
	// Add first edge (must succeed).
	_, err = g.AddEdge(VertexA, VertexB, Weight0)
	MustErrorNil(t, err, "first AddEdge(A,B,0) on default graph")
	// Add second parallel edge (must fail).
	_, err = g.AddEdge(VertexA, VertexB, Weight0)
	MustErrorIs(t, err, core.ErrMultiEdgeNotAllowed, "second AddEdge(A,B,0) on default graph")

	// Stage 6: Multi-edge enabled graph allows parallel edges with distinct IDs.
	// Create graph with multi-edges, weights, and loops enabled to maximize surface.
	g, _ = core.NewGraph(core.WithMultiEdges(), core.WithWeighted(), core.WithLoops())
	// Add first edge.
	e1, err := g.AddEdge(VertexA, VertexB, Weight1)
	MustErrorNil(t, err, "first AddEdge(A,B,1) on multigraph")
	// Add second parallel edge.
	e2, err := g.AddEdge(VertexA, VertexB, Weight2)
	MustErrorNil(t, err, "second AddEdge(A,B,2) on multigraph")
	// Parallel edges must produce distinct IDs.
	MustNotEqualString(t, e1, e2, "parallel AddEdge(A,B,*) must return distinct IDs when multi-edges enabled")
}

// TestGraph_AddEdgeRejectsNaNAndInfWeights verifies finite-weight numeric policy.
//
// Contract anchors:
//   - NaN and +/-Inf are invalid weights even when weighted graphs are enabled.
//   - Current core policy classifies non-finite weights as ErrBadWeight.
//   - Invalid numeric input must not publish an edge or auto-create endpoints.
//
// Notes:
//   - If core later introduces ErrNaNInf as a distinct sentinel, update only the expected target.
func TestGraph_AddEdgeRejectsNaNAndInfWeights(t *testing.T) {
	cases := []struct {
		name   string
		weight float64
	}{
		{name: "NaN", weight: math.NaN()},
		{name: "+Inf", weight: math.Inf(1)},
		{name: "-Inf", weight: math.Inf(-1)},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g := MustNewGraph(t, core.WithWeighted())

			id, err := g.AddEdge(VertexA, VertexB, tc.weight)

			MustErrorIs(t, err, core.ErrNaNInf, "AddEdge rejects non-finite weight")
			MustEqualString(t, id, "", "AddEdge non-finite weight returned edge ID")
			MustEqualInt(t, g.EdgeCount(), Count0, "non-finite weight must not publish edge")
			MustEqualBool(t, g.HasVertex(VertexA), false, "non-finite weight must not auto-create source")
			MustEqualBool(t, g.HasVertex(VertexB), false, "non-finite weight must not auto-create target")
		})
	}
}

// TestGraph_MixedEdgesDirectedOverride VERIFIES per-edge directedness override gating and behavior.
// Implementation:
//   - Stage 1: Non-mixed graph must reject WithEdgeDirected override (ErrMixedEdgesNotAllowed).
//   - Stage 2: Mixed graph must accept override and set Edge.Directed accordingly.
//
// Behavior highlights:
//   - Prevents silent “mixed behavior” on graphs that did not opt into mixed mode.
//
// Inputs:
//   - None (uses package constants).
//
// Returns:
//   - None.
//
// Errors:
//   - core.ErrMixedEdgesNotAllowed from AddEdge when mixed mode is disabled and an EdgeOption is passed.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1) expected, Space O(1) incremental per edge.
//
// Notes:
//   - This test does not assert edge ID format; only the policy gate + Directed flag semantics.
//
// AI-Hints:
//   - Always create mixed graphs explicitly (NewMixedGraph or WithMixedEdges) before using WithEdgeDirected.
func TestGraph_MixedEdgesDirectedOverride(t *testing.T) {
	// Stage 1: Non-mixed graph rejects per-edge override.
	{
		// Create a default (non-mixed) graph.
		g := MustNewGraph(t)
		// Attempt to override per-edge directedness without mixed mode.
		_, err := g.AddEdge(VertexX, VertexY, Weight0, core.WithEdgeDirected(true))
		// Enforce sentinel gate contract.
		MustErrorIs(t, err, core.ErrMixedEdgesNotAllowed, "AddEdge(X,Y,0,WithEdgeDirected) on non-mixed graph")
	}

	// Stage 2: Mixed graph accepts per-edge override and sets Edge.Directed=true.
	{
		// Create a mixed graph (per-edge directedness overrides allowed).
		g, _ := core.NewMixedGraph()
		// Add an edge overriding directedness to true.
		eid, err := g.AddEdge(VertexX, VertexY, Weight0, core.WithEdgeDirected(true))
		MustErrorNil(t, err, "AddEdge(X,Y,0,WithEdgeDirected(true)) on mixed graph")
		// Read back the edge by ID.
		e, err := g.GetEdge(eid)
		MustErrorNil(t, err, "GetEdge(eid) on mixed graph")
		// Validate per-edge directedness override effect.
		MustEqualBool(t, e.Directed, true, "mixed edge must have Directed=true after WithEdgeDirected(true)")
	}
}

// TestGraph_RemoveEdge VERIFIES RemoveEdge sentinel behavior and adjacency cleanup.
// Implementation:
//   - Stage 1: Create a weighted graph and add two edges.
//   - Stage 2: Assert RemoveEdge(missing) returns ErrEdgeNotFound.
//   - Stage 3: Remove an existing edge and assert adjacency is cleaned.
//
// Behavior highlights:
//   - Locks in ErrEdgeNotFound sentinel for unknown IDs.
//   - Locks in undirected mirror cleanup behavior.
//
// Inputs:
//   - None (uses package constants).
//
// Returns:
//   - None.
//
// Errors:
//   - core.ErrEdgeNotFound when removing an unknown edge ID.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1) expected for remove by ID (implementation-dependent), Space O(1) extra.
//
// Notes:
//   - This test assumes HasEdge is safe even if vertices exist but edge is removed.
//
// AI-Hints:
//   - Prefer verifying cleanup via HasEdge(from,to) and HasEdge(to,from) in undirected graphs.
func TestGraph_RemoveEdge(t *testing.T) {
	// Stage 1: Create weighted graph and add two edges.
	g, _ := core.NewGraph(core.WithWeighted())

	// Add edge A-B to later remove.
	eidAB, err := g.AddEdge(VertexA, VertexB, Weight1)
	MustErrorNil(t, err, "AddEdge(A,B,1) setup")

	// Add edge B-C to ensure unrelated edges remain.
	_, err = g.AddEdge(VertexB, VertexC, Weight2)
	MustErrorNil(t, err, "AddEdge(B,C,2) setup")

	// Stage 2: Removing a non-existent edge must yield ErrEdgeNotFound.
	err = g.RemoveEdge(EdgeIDMissing)
	MustErrorIs(t, err, core.ErrEdgeNotFound, "RemoveEdge(missing)")
	// Stage 3: Remove existing A-B and verify undirected adjacency cleanup.
	MustErrorNil(t, g.RemoveEdge(eidAB), "RemoveEdge(eidAB)")

	// Verify forward adjacency removed.
	MustEqualBool(t, g.HasEdge(VertexA, VertexB), false, "HasEdge(A,B) after RemoveEdge(eidAB)")
	// Verify mirror adjacency removed in undirected graph.
	MustEqualBool(t, g.HasEdge(VertexB, VertexA), false, "HasEdge(B,A) after RemoveEdge(eidAB)")
	// Verify unrelated edge remains.
	MustEqualBool(t, g.HasEdge(VertexB, VertexC), true, "HasEdge(B,C) after RemoveEdge(eidAB)")
}

// TestGraph_GetEdgeAndRemoveEdgeRejectEmptyID verifies edge-ID validation at lookup/removal boundaries.
//
// Contract anchors:
//   - Empty edge IDs are malformed input, not missing catalog entries.
//   - GetEdge("") and RemoveEdge("") must classify with ErrEmptyEdgeID.
//   - Failed validation must not mutate graph state.
func TestGraph_GetEdgeAndRemoveEdgeRejectEmptyID(t *testing.T) {
	g := MustNewGraph(t)
	id, err := g.AddEdge(VertexA, VertexB, Weight0)
	MustErrorNil(t, err, "AddEdge(A,B) setup")

	edge, err := g.GetEdge("")
	MustErrorIs(t, err, core.ErrEmptyEdgeID, "GetEdge(empty)")
	MustEqualBool(t, edge == nil, true, "GetEdge(empty) must return nil edge")

	err = g.RemoveEdge("")
	MustErrorIs(t, err, core.ErrEmptyEdgeID, "RemoveEdge(empty)")
	MustEqualInt(t, g.EdgeCount(), Count1, "RemoveEdge(empty) must not mutate edge count")
	MustEqualBool(t, g.HasEdge(VertexA, VertexB), true, "RemoveEdge(empty) must not unlink existing edge")

	_, err = g.GetEdge(id)
	MustErrorNil(t, err, "GetEdge(existing) after empty-ID validation failures")
}

// TestGraph_MultiEdges VERIFIES parallel-edge semantics and weight preservation when enabled.
// Implementation:
//   - Stage 1: Create a multi-edge, weighted graph.
//   - Stage 2: Add two parallel edges A-B with weights 1 and 2.
//   - Stage 3: Assert IDs differ.
//   - Stage 4: Read edges by ID and assert weights match.
//
// Behavior highlights:
//   - Locks in ID uniqueness under multi-edge policy.
//
// Inputs:
//   - None (uses package constants).
//
// Returns:
//   - None.
//
// Errors:
//   - Propagates errors from AddEdge/GetEdge.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1) per AddEdge/GetEdge (implementation-dependent), Space O(1) extra.
//
// Notes:
//   - Weight equality is exact here because the inputs are integer-like constants.
//
// AI-Hints:
//   - Prefer validating edge attributes via GetEdge(id) instead of scanning Edges().
func TestGraph_MultiEdges(t *testing.T) {
	// Stage 1: Enable multi-edges and weights.
	g, _ := core.NewGraph(core.WithMultiEdges(), core.WithWeighted())

	// Stage 2: Add parallel edges A-B with different weights.
	e1, err := g.AddEdge(VertexA, VertexB, Weight1)
	MustErrorNil(t, err, "AddEdge(A,B,1)")
	e2, err := g.AddEdge(VertexA, VertexB, Weight2)
	MustErrorNil(t, err, "AddEdge(A,B,2)")

	// Stage 3: IDs must differ.
	MustNotEqualString(t, e1, e2, "parallel edges must produce distinct IDs")

	// Stage 4: Validate stored weights by reading edges back by ID.
	edge1, err := g.GetEdge(e1)
	MustErrorNil(t, err, "GetEdge(e1)")
	edge2, err := g.GetEdge(e2)
	MustErrorNil(t, err, "GetEdge(e2)")

	// Compare weights exactly (integer-like float64 constants).
	MustEqualBool(t, edge1.Weight == float64(Weight1), true, "edge1 weight must equal 1")
	MustEqualBool(t, edge2.Weight == float64(Weight2), true, "edge2 weight must equal 2")
}

// TestGraph_EdgesAreSorted ANCHORS the contract: Edges() must be sorted by Edge.ID ascending.
// Implementation:
//   - Stage 1: Create a multi-edge weighted graph and add multiple parallel edges.
//   - Stage 2: Extract IDs from Edges() and assert sortedness.
//
// Behavior highlights:
//   - Deterministic ordering simplifies downstream algorithms and stable tests.
//
// Inputs:
//   - None (uses package constants).
//
// Returns:
//   - None.
//
// Errors:
//   - Propagates errors from AddEdge.
//
// Determinism:
//   - Deterministic: Edges() order is stable and sorted by ID.
//
// Complexity:
//   - Time O(E log E) if Edges() sorts internally, Space O(E) for returned slice.
//
// Notes:
//   - This test checks sortedness only; uniqueness is covered by AddEdge/multi-edge tests and ID uniqueness tests.
//
// AI-Hints:
//   - If you change edge-ID representation, keep ordering deterministic (lexicographic over IDs is simplest).
func TestGraph_EdgesAreSorted(t *testing.T) {
	// Stage 1: Create a multigraph and add multiple edges so sorting is observable.
	g, _ := core.NewGraph(core.WithMultiEdges(), core.WithWeighted())

	// Add three parallel edges.
	_, err := g.AddEdge(VertexA, VertexB, Weight1)
	MustErrorNil(t, err, "AddEdge(A,B,1)")
	_, err = g.AddEdge(VertexA, VertexB, Weight2)
	MustErrorNil(t, err, "AddEdge(A,B,2)")
	_, err = g.AddEdge(VertexA, VertexB, Weight3)
	MustErrorNil(t, err, "AddEdge(A,B,3)")

	// Stage 2: Validate stable sorted order by Edge.ID (lexicographic).
	ees := g.Edges()
	ids := ExtractEdgeIDs(ees)
	MustSortedStrings(t, ids, "Edges() IDs must be sorted asc")
}

// TestGraph_AddEdgeWithIDUsesExplicitID verifies that AddEdge honors WithID for a unique, non-empty edge identifier and that
// the created edge is addressable through the edge catalog under that exact ID.
//
// Implementation:
//   - Stage 1: Construct a default Graph (unweighted, undirected).
//   - Stage 2: AddEdge(A,B,0,WithID(customID)) and assert it succeeds.
//   - Stage 3: Assert the returned edge ID equals customID (no auto-ID fallback).
//   - Stage 4: Lookup via GetEdge(customID) and assert Edge.ID == customID.
//   - Stage 5: Assert HasEdge(A,B) reflects the insertion.
//
// Behavior highlights:
//   - Confirms that WithID bypasses auto-generated IDs for that edge.
//   - Confirms catalog key consistency: returned ID == GetEdge lookup key == Edge.ID field.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - The test fails if AddEdge returns an error, if the returned ID mismatches,
//     if GetEdge cannot retrieve the edge, or if HasEdge does not observe the insertion.
//
// Determinism:
//   - Deterministic: explicit ID and deterministic catalog operations; no iteration-order dependencies.
//
// Complexity:
//   - Time O(1) average per operation, Space O(1) (excluding graph allocations).
//
// Notes:
//   - Uses Weight0 to satisfy the default unweighted graph constraint.
//
// AI-Hints:
//   - Use WithID to create stable external references (golden tests, trace correlation, interop).
func TestGraph_AddEdgeWithIDUsesExplicitID(t *testing.T) {
	g := MustNewGraph(t)
	eid, err := g.AddEdge(VertexA, VertexB, Weight0, core.WithID("customID123"))
	MustErrorNil(t, err, "AddEdge(A,B,0,WithID) should succeed with unique ID")
	MustEqualString(t, eid, "customID123", "returned edge ID should match provided ID")
	// Check the edge is retrievable and has the correct ID
	e, err := g.GetEdge("customID123")
	MustErrorNil(t, err, "GetEdgeByID(customID123) after AddEdge")
	MustNotNil(t, e, "GetEdgeByID should return an Edge")
	MustEqualString(t, e.ID, "customID123", "Edge.ID should be the custom ID")
	MustEqualBool(t, g.HasEdge(VertexA, VertexB), true, "HasEdge(A,B) should reflect the new edge")
}

// TestGraph_AddEdgeNilOptionDoesNotPublishVerticesOrEdges verifies fail-fast nil EdgeOption validation.
//
// Contract anchors:
//   - Nil EdgeOption values are invalid public input and return ErrNilEdgeOption.
//   - Nil option validation runs before lock acquisition, endpoint auto-creation, ID assignment, or adjacency mutation.
//   - Failed validation leaves the graph empty.
func TestGraph_AddEdgeNilOptionDoesNotPublishVerticesOrEdges(t *testing.T) {
	g := MustNewGraph(t)
	var nilOpt core.EdgeOption

	id, err := g.AddEdge(VertexA, VertexB, Weight0, nilOpt)

	MustErrorIs(t, err, core.ErrNilEdgeOption, "AddEdge nil EdgeOption")
	MustEqualString(t, id, "", "AddEdge nil EdgeOption returned edge ID")
	MustEqualInt(t, g.EdgeCount(), Count0, "nil EdgeOption must not publish edge")
	MustEqualInt(t, g.VertexCount(), Count0, "nil EdgeOption must not auto-create endpoints")
	MustEqualBool(t, g.HasVertex(VertexA), false, "nil EdgeOption must not create source")
	MustEqualBool(t, g.HasVertex(VertexB), false, "nil EdgeOption must not create target")
}

// TestGraph_AddEdgeRejectsEdgeOptionEndpointMutation verifies that custom EdgeOption
// values cannot rewrite topology-owned endpoint fields.
//
// Contract anchor:
//   - EdgeOption may not mutate Edge.From or Edge.To.
//   - AddEdge must reject such mutation with ErrInvalidEdgeOption before edge publication.
//   - The attempted replacement endpoints must not be auto-created or linked.
func TestGraph_AddEdgeRejectsEdgeOptionEndpointMutation(t *testing.T) {
	g := MustNewGraph(t)

	mutateEndpoints := func(_ *core.Graph, e *core.Edge) error {
		e.From = VertexX
		e.To = VertexY
		return nil
	}

	id, err := g.AddEdge(VertexA, VertexB, Weight0, mutateEndpoints)

	MustErrorIs(t, err, core.ErrInvalidEdgeOption, "AddEdge rejects endpoint-mutating EdgeOption")
	MustEqualString(t, id, "", "AddEdge invalid option returned edge ID")
	MustEqualInt(t, g.EdgeCount(), Count0, "invalid endpoint option must not publish edge")
	MustEqualBool(t, g.HasEdge(VertexA, VertexB), false, "invalid endpoint option must not publish A-B adjacency")
	MustEqualBool(t, g.HasVertex(VertexX), false, "invalid endpoint option must not auto-create mutated source")
	MustEqualBool(t, g.HasVertex(VertexY), false, "invalid endpoint option must not auto-create mutated target")
}

// TestGraph_AddEdgePropagatesEdgeOptionErrorWithoutPublishingEdge verifies option-stage error handling.
//
// Contract anchors:
//   - Non-nil EdgeOption errors are propagated through errors.Is.
//   - Option-stage failure must not publish an edge or adjacency entry.
//   - Endpoint auto-creation has already occurred before option application and remains valid catalog state.
//
// Notes:
//   - This differs from nil EdgeOption validation, which fails before endpoint auto-creation.
func TestGraph_AddEdgePropagatesEdgeOptionErrorWithoutPublishingEdge(t *testing.T) {
	g := MustNewGraph(t)
	errOptionFailure := errors.New("core test: option failure")

	failOption := func(_ *core.Graph, _ *core.Edge) error {
		return errOptionFailure
	}

	id, err := g.AddEdge(VertexA, VertexB, Weight0, failOption)

	MustErrorIs(t, err, errOptionFailure, "AddEdge propagates EdgeOption error")
	MustEqualString(t, id, "", "AddEdge option failure returned edge ID")
	MustEqualInt(t, g.EdgeCount(), Count0, "option failure must not publish edge")
	MustEqualBool(t, g.HasEdge(VertexA, VertexB), false, "option failure must not publish adjacency")
	MustEqualBool(t, g.HasVertex(VertexA), true, "option-stage failure keeps auto-created source")
	MustEqualBool(t, g.HasVertex(VertexB), true, "option-stage failure keeps auto-created target")
}

// TestGraph_AddEdgeRejectsEdgeOptionDirectednessMutationWithoutMixedMode verifies
// that a custom EdgeOption cannot bypass the mixed-edge policy by mutating Directed directly.
//
// Contract anchor:
//   - Directedness override is allowed only in mixed mode.
//   - Direct Directed mutation by a custom option must be classified as ErrMixedEdgesNotAllowed.
func TestGraph_AddEdgeRejectsEdgeOptionDirectednessMutationWithoutMixedMode(t *testing.T) {
	g := MustNewGraph(t)

	forceDirected := func(_ *core.Graph, e *core.Edge) error {
		e.Directed = true
		return nil
	}

	id, err := g.AddEdge(VertexA, VertexB, Weight0, forceDirected)

	MustErrorIs(t, err, core.ErrMixedEdgesNotAllowed, "AddEdge rejects directedness mutation without mixed mode")
	MustEqualString(t, id, "", "AddEdge directedness policy failure returned edge ID")
	MustEqualInt(t, g.EdgeCount(), Count0, "directedness policy failure must not publish edge")
	MustEqualBool(t, g.HasEdge(VertexA, VertexB), false, "directedness policy failure must not publish adjacency")
}

// TestGraph_SetEdgeIDRenamesCatalogAndAdjacency verifies that SetEdgeID rewrites
// the edge catalog and both adjacency directions for an undirected edge.
func TestGraph_SetEdgeIDRenamesCatalogAndAdjacency(t *testing.T) {
	g := MustNewGraph(t)
	oldID, err := g.AddEdge(VertexA, VertexB, Weight0)
	MustErrorNil(t, err, "AddEdge(A,B)")

	const newID = "friend_A_B"
	MustErrorNil(t, g.SetEdgeID(oldID, newID), "SetEdgeID(e1,friend_A_B)")

	_, err = g.GetEdge(oldID)
	MustErrorIs(t, err, core.ErrEdgeNotFound, "GetEdge(oldID after rename)")

	e, err := g.GetEdge(newID)
	MustErrorNil(t, err, "GetEdge(newID after rename)")
	MustEqualString(t, e.ID, newID, "renamed edge ID")
	MustEqualString(t, e.From, VertexA, "renamed edge From")
	MustEqualString(t, e.To, VertexB, "renamed edge To")

	adj := g.AdjacencyList()
	MustSameStringSet(t, adj[VertexA], []string{newID}, "A adjacency after SetEdgeID")
	MustSameStringSet(t, adj[VertexB], []string{newID}, "B mirror adjacency after SetEdgeID")
}

// TestGraph_SetEdgeIDBumpsAutoCounter verifies that renaming to a canonical auto-ID
// advances the monotonic generator and prevents a future collision.
func TestGraph_SetEdgeIDBumpsAutoCounter(t *testing.T) {
	g := MustNewGraph(t, core.WithMultiEdges())
	firstID, err := g.AddEdge(VertexA, VertexB, Weight0)
	MustErrorNil(t, err, "AddEdge(A,B)")

	MustErrorNil(t, g.SetEdgeID(firstID, "e100"), "SetEdgeID(e1,e100)")

	nextID, err := g.AddEdge(VertexA, VertexB, Weight0)
	MustErrorNil(t, err, "AddEdge(A,B) after e100 rename")
	MustEqualString(t, nextID, "e101", "auto-ID counter after SetEdgeID(e100)")
}

// TestGraph_SetEdgeIDValidationSentinels verifies SetEdgeID validation and no-op behavior.
//
// Contract anchors:
//   - Empty old or new IDs are malformed input and return ErrEmptyEdgeID.
//   - Missing old ID returns ErrEdgeNotFound.
//   - Renaming to an existing different ID returns ErrEdgeIDConflict.
//   - Renaming an edge to the same ID is a no-op success.
//   - Validation failures must not mutate catalog or adjacency state.
func TestGraph_SetEdgeIDValidationSentinels(t *testing.T) {
	g := MustNewGraph(t, core.WithMultiEdges())

	id1, err := g.AddEdge(VertexA, VertexB, Weight0)
	MustErrorNil(t, err, "AddEdge(A,B) id1")
	id2, err := g.AddEdge(VertexA, VertexB, Weight0)
	MustErrorNil(t, err, "AddEdge(A,B) id2")

	MustErrorIs(t, g.SetEdgeID("", "new_id"), core.ErrEmptyEdgeID, "SetEdgeID(empty,new_id)")
	MustErrorIs(t, g.SetEdgeID(id1, ""), core.ErrEmptyEdgeID, "SetEdgeID(id1,empty)")
	MustErrorIs(t, g.SetEdgeID(EdgeIDMissing, "new_id"), core.ErrEdgeNotFound, "SetEdgeID(missing,new_id)")
	MustErrorIs(t, g.SetEdgeID(id1, id2), core.ErrEdgeIDConflict, "SetEdgeID(id1,id2 conflict)")

	MustErrorNil(t, g.SetEdgeID(id1, id1), "SetEdgeID(id1,id1 no-op)")

	_, err = g.GetEdge(id1)
	MustErrorNil(t, err, "GetEdge(id1) after failed/no-op SetEdgeID calls")
	_, err = g.GetEdge(id2)
	MustErrorNil(t, err, "GetEdge(id2) after failed/no-op SetEdgeID calls")
	MustEqualBool(t, g.HasEdge(VertexA, VertexB), true, "SetEdgeID validation failures must preserve adjacency")
	MustEqualInt(t, g.EdgeCount(), Count2, "SetEdgeID validation failures must preserve edge count")
}

// TestGraph_GetNamedEdgesSkipsCanonicalAutoIDs verifies that GetNamedEdges returns
// non-auto-shaped IDs only, sorted by Edge.ID ascending.
func TestGraph_GetNamedEdgesSkipsCanonicalAutoIDs(t *testing.T) {
	g := NewGraphFull(t)

	_, err := g.AddEdge(VertexA, VertexB, Weight0)
	MustErrorNil(t, err, "AddEdge auto e1")
	_, err = g.AddEdge(VertexB, VertexC, Weight0)
	MustErrorNil(t, err, "AddEdge auto e2")
	_, err = g.AddEdge(VertexC, VertexD, Weight0, core.WithID("friend_C_D"))
	MustErrorNil(t, err, "AddEdge named friend_C_D")
	_, err = g.AddEdge(VertexD, VertexA, Weight0, core.WithID("edge-007"))
	MustErrorNil(t, err, "AddEdge named edge-007")
	_, err = g.AddEdge(VertexA, VertexC, Weight0, core.WithID("e01"))
	MustErrorNil(t, err, "AddEdge noncanonical e01")

	ids := ExtractEdgeIDs(g.GetNamedEdges())

	MustSortedStrings(t, ids, "GetNamedEdges order")
	MustSameStringSet(t, ids, []string{"e01", "edge-007", "friend_C_D"}, "GetNamedEdges named IDs")
}

// TestGraph_HasDirectedEdgesMixedAndUniform verifies directed-edge existence across
// undirected, directed-default, and mixed-mode graphs.
func TestGraph_HasDirectedEdgesMixedAndUniform(t *testing.T) {
	undirected := MustNewGraph(t)
	_, err := undirected.AddEdge(VertexA, VertexB, Weight0)
	MustErrorNil(t, err, "undirected AddEdge(A,B)")
	MustEqualBool(t, undirected.HasDirectedEdges(), false, "undirected.HasDirectedEdges()")

	directed := MustNewGraph(t, core.WithDirected(true))
	_, err = directed.AddEdge(VertexA, VertexB, Weight0)
	MustErrorNil(t, err, "directed AddEdge(A,B)")
	MustEqualBool(t, directed.HasDirectedEdges(), true, "directed.HasDirectedEdges()")

	mixed := MustNewMixedGraph(t)
	_, err = mixed.AddEdge(VertexA, VertexB, Weight0)
	MustErrorNil(t, err, "mixed AddEdge undirected default")
	MustEqualBool(t, mixed.HasDirectedEdges(), false, "mixed.HasDirectedEdges before directed override")
	_, err = mixed.AddEdge(VertexB, VertexC, Weight0, core.WithEdgeDirected(true))
	MustErrorNil(t, err, "mixed AddEdge directed override")
	MustEqualBool(t, mixed.HasDirectedEdges(), true, "mixed.HasDirectedEdges after directed override")
}

// TestGraph_FilterEdgesNilPredicate verifies sentinel-first validation for nil read-only filters.
//
// Contract anchor:
//   - FilterEdges is a read-only query surface.
//   - A nil predicate is invalid input and must be classified through ErrNilEdgePredicate.
//   - Nil predicate failure must not mutate graph state.
func TestGraph_FilterEdgesNilPredicate(t *testing.T) {
	g := NewGraphFull(t)
	matches, err := g.FilterEdges(nil)

	MustErrorIs(t, err, core.ErrNilEdgePredicate, "FilterEdges(nil)")
	MustEqualInt(t, len(matches), Count0, "FilterEdges(nil) matches count")
	MustEqualInt(t, g.EdgeCount(), Count0, "FilterEdges(nil) must not mutate graph")
}

// TestGraph_FilterEdgesReturnsSortedDetachedCopies verifies the read-only filtering contract.
//
// Scenario:
//   - John and Mary are isolated vertices.
//   - Alice, Bob, Charlie form a triangle.
//   - David, Emma, Frank, Grace, Henry, Ivy form an 11-edge group.
//
// Contract anchors:
//   - FilterEdges returns matching edges and does not remove anything.
//   - Returned edges are detached value copies, not live catalog pointers.
//   - Output is sorted by Edge.ID ascending.
//   - Public AdjacencyList still includes isolated vertices through the vertex catalog.
func TestGraph_FilterEdgesReturnsSortedDetachedCopies(t *testing.T) {
	g := newSocialComponentsGraph(t)
	MustGraphCounts(t, g, socialVertexCount, socialTotalEdgeCount, "initial social graph")

	matches, err := g.FilterEdges(func(e core.Edge) bool {
		return strings.HasPrefix(e.ID, "group_")
	})

	MustErrorNil(t, err, "FilterEdges(group_*)")
	MustEqualInt(t, len(matches), socialGroupEdgeCount, "FilterEdges group match count")
	MustEqualInt(t, g.EdgeCount(), socialTotalEdgeCount, "FilterEdges must not remove edges")
	MustEqualInt(t, g.VertexCount(), socialVertexCount, "FilterEdges must not remove vertices")

	ids := make([]string, 0, len(matches))
	for _, e := range matches {
		ids = append(ids, e.ID)
	}
	MustSortedStrings(t, ids, "FilterEdges output order")
	MustSameStringSet(t, ids, []string{
		"group_david_emma",
		"group_david_frank",
		"group_david_grace",
		"group_david_henry",
		"group_emma_grace",
		"group_emma_henry",
		"group_emma_ivy",
		"group_frank_henry",
		"group_grace_henry",
		"group_grace_ivy",
		"group_henry_ivy",
	}, "FilterEdges group edge IDs")

	first := matches[0]

	matches[0].ID = "corrupted_copy_id"
	matches[0].From = vertexJohn
	matches[0].To = vertexMary
	matches[0].Weight = 999
	matches[0].Directed = !matches[0].Directed

	edge, err := g.GetEdge(first.ID)
	MustErrorNil(t, err, "GetEdge(first FilterEdges match after caller mutation)")
	MustEqualString(t, edge.ID, first.ID, "catalog edge ID after caller mutates result copy")
	MustEqualString(t, edge.From, first.From, "catalog edge From after caller mutates result copy")
	MustEqualString(t, edge.To, first.To, "catalog edge To after caller mutates result copy")
	MustEqualBool(t, edge.Weight == first.Weight, true, "catalog edge Weight after caller mutates result copy")
	MustEqualBool(t, edge.Directed == first.Directed, true, "catalog edge Directed after caller mutates result copy")

	adj := g.AdjacencyList()
	MustEqualInt(t, len(adj), socialVertexCount, "AdjacencyList key count after read-only FilterEdges")
	MustEqualInt(t, len(adj[vertexJohn]), Count0, "John remains isolated after read-only FilterEdges")
	MustEqualInt(t, len(adj[vertexMary]), Count0, "Mary remains isolated after read-only FilterEdges")
	MustEqualBool(t, g.HasEdge(vertexAlice, vertexBob), true, "FilterEdges must keep triangle Alice-Bob")
	MustEqualBool(t, g.HasEdge(vertexDavid, vertexEmma), true, "FilterEdges must keep group David-Emma")
}

// TestGraph_RemoveEdgesWhereNilPredicate verifies sentinel-first validation for nil mutating predicates.
//
// Contract anchor:
//   - RemoveEdgesWhere is the mutating bulk-deletion surface.
//   - Nil predicate failure must not mutate graph state.
func TestGraph_RemoveEdgesWhereNilPredicate(t *testing.T) {
	g := newSocialComponentsGraph(t)
	removed, err := g.RemoveEdgesWhere(nil)

	MustErrorIs(t, err, core.ErrNilEdgePredicate, "RemoveEdgesWhere(nil)")
	MustEqualInt(t, removed, Count0, "RemoveEdgesWhere(nil) removed count")
	MustGraphCounts(t, g, socialVertexCount, socialTotalEdgeCount, "RemoveEdgesWhere(nil) must not mutate graph")
}

// TestGraph_RemoveEdgesWhereRemovesCatalogAndAdjacency verifies bulk edge deletion on a social graph
// with 11 vertices and 4 connected components before deletion.
//
// Scenario:
//   - John, Mary are isolated.
//   - Alice, Bob, Charlie form a triangle.
//   - David, Emma, Frank, Grace, Henry, Ivy form an 11-edge group.
//
// Contract anchors:
//   - RemoveEdgesWhere removes matching edges from both edge catalog and sparse adjacency index.
//   - Endpoint vertices remain present even when they become isolated.
//   - Public AdjacencyList reconstructs isolated vertex keys from the vertex catalog.
//   - Degree remains mathematically correct after sparse-index cleanup.
//   - RemoveVertex can cleanup a high-degree vertex without deadlocking or dangling adjacency.
func TestGraph_RemoveEdgesWhereRemovesCatalogAndAdjacency(t *testing.T) {
	g := newSocialComponentsGraph(t)

	MustGraphCounts(t, g, socialVertexCount, socialTotalEdgeCount, "initial social graph")
	MustUndirectedDegree(t, g, vertexJohn, Count0)
	MustUndirectedDegree(t, g, vertexMary, Count0)
	MustUndirectedDegree(t, g, vertexAlice, Count2)
	MustUndirectedDegree(t, g, vertexBob, Count2)
	MustUndirectedDegree(t, g, vertexCharlie, Count2)
	MustUndirectedDegree(t, g, vertexDavid, 4)
	MustUndirectedDegree(t, g, vertexEmma, 4)
	MustUndirectedDegree(t, g, vertexFrank, Count2)
	MustUndirectedDegree(t, g, vertexGrace, 4)
	MustUndirectedDegree(t, g, vertexHenry, 5)
	MustUndirectedDegree(t, g, vertexIvy, Count3)

	removed, err := g.RemoveEdgesWhere(func(e core.Edge) bool {
		return strings.HasPrefix(e.ID, "triangle_")
	})
	MustErrorNil(t, err, "RemoveEdgesWhere(triangle_*)")
	MustEqualInt(t, removed, socialTriangleEdgeCount, "RemoveEdgesWhere removed triangle count")
	MustGraphCounts(t, g, socialVertexCount, socialGroupEdgeCount, "after removing triangle edges")

	for _, id := range []string{vertexJohn, vertexMary, vertexAlice, vertexBob, vertexCharlie} {
		MustUndirectedDegree(t, g, id, Count0)
	}
	MustUndirectedDegree(t, g, vertexDavid, 4)
	MustUndirectedDegree(t, g, vertexEmma, 4)
	MustUndirectedDegree(t, g, vertexFrank, Count2)
	MustUndirectedDegree(t, g, vertexGrace, 4)
	MustUndirectedDegree(t, g, vertexHenry, 5)
	MustUndirectedDegree(t, g, vertexIvy, Count3)

	adj := g.AdjacencyList()
	MustEqualInt(t, len(adj), socialVertexCount, "AdjacencyList key count after RemoveEdgesWhere")
	MustEqualInt(t, len(adj[vertexJohn]), Count0, "John adjacency after RemoveEdgesWhere")
	MustEqualInt(t, len(adj[vertexMary]), Count0, "Mary adjacency after RemoveEdgesWhere")
	MustEqualInt(t, len(adj[vertexAlice]), Count0, "Alice adjacency after RemoveEdgesWhere")
	MustEqualInt(t, len(adj[vertexBob]), Count0, "Bob adjacency after RemoveEdgesWhere")
	MustEqualInt(t, len(adj[vertexCharlie]), Count0, "Charlie adjacency after RemoveEdgesWhere")
	MustEqualBool(t, g.HasEdge(vertexAlice, vertexBob), false, "triangle adjacency removed Alice-Bob")
	MustEqualBool(t, g.HasEdge(vertexBob, vertexCharlie), false, "triangle adjacency removed Bob-Charlie")
	MustEqualBool(t, g.HasEdge(vertexCharlie, vertexAlice), false, "triangle adjacency removed Charlie-Alice")
	MustEqualBool(t, g.HasEdge(vertexDavid, vertexEmma), true, "group adjacency kept David-Emma")

	MustErrorNil(t, g.RemoveVertex(vertexHenry), "RemoveVertex(Henry)")
	MustGraphCounts(t, g, socialVertexCount-1, socialAfterHenryEdges, "after removing Henry")
	MustEqualBool(t, g.HasVertex(vertexHenry), false, "Henry removed from vertex catalog")
	MustEqualBool(t, g.HasEdge(vertexDavid, vertexHenry), false, "David-Henry adjacency removed")
	MustEqualBool(t, g.HasEdge(vertexHenry, vertexDavid), false, "Henry-David adjacency removed")
	MustEqualBool(t, g.HasEdge(vertexEmma, vertexHenry), false, "Emma-Henry adjacency removed")
	MustEqualBool(t, g.HasEdge(vertexIvy, vertexHenry), false, "Ivy-Henry adjacency removed")

	adj = g.AdjacencyList()
	_, hasHenry := adj[vertexHenry]
	MustEqualBool(t, hasHenry, false, "AdjacencyList must not include removed vertex")
}

// TestGraph_RemoveEdgesWhereRemovesMatchingEdges verifies the value-copy bulk deletion API
// on a compact contract path.
func TestGraph_RemoveEdgesWhereRemovesMatchingEdges(t *testing.T) {
	g := newSocialComponentsGraph(t)

	removed, err := g.RemoveEdgesWhere(func(e core.Edge) bool {
		return strings.HasPrefix(e.ID, "triangle_")
	})

	MustErrorNil(t, err, "RemoveEdgesWhere(triangle_*)")
	MustEqualInt(t, removed, socialTriangleEdgeCount, "RemoveEdgesWhere removed triangle count")
	MustEqualInt(t, g.EdgeCount(), socialGroupEdgeCount, "RemoveEdgesWhere remaining group edges")
	MustEqualBool(t, g.HasEdge(vertexAlice, vertexBob), false, "RemoveEdgesWhere removed Alice-Bob")
	MustEqualBool(t, g.HasEdge(vertexDavid, vertexEmma), true, "RemoveEdgesWhere kept David-Emma")
}

// TestGraph_ValidateEdgeIDEmptyConflictOK verifies edge-ID validation sentinel classes.
func TestGraph_ValidateEdgeIDEmptyConflictOK(t *testing.T) {
	g := MustNewGraph(t)

	MustErrorIs(t, g.ValidateEdgeID(""), core.ErrEmptyEdgeID, "ValidateEdgeID(empty)")
	MustErrorNil(t, g.ValidateEdgeID("custom_edge"), "ValidateEdgeID(custom_edge before use)")

	_, err := g.AddEdge(VertexA, VertexB, Weight0, core.WithID("custom_edge"))
	MustErrorNil(t, err, "AddEdge(custom_edge)")

	MustErrorIs(t, g.ValidateEdgeID("custom_edge"), core.ErrEdgeIDConflict, "ValidateEdgeID(custom_edge after use)")
	MustErrorNil(t, g.ValidateEdgeID("another_custom_edge"), "ValidateEdgeID(another_custom_edge)")
}
