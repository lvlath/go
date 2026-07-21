package core_test

import (
	"testing"

	"github.com/lvlath/go/core"
)

// TestGraph_CloneAndCloneEmptyContracts verifies clone ownership and inventory rules.
//
// Contract anchors:
//   - CloneEmpty preserves vertex IDs and drops all edges.
//   - Clone preserves vertex IDs and edge IDs.
//   - Clone deep-copies edge objects; returned edge pointers must not alias original pointers.
//   - Scalar edge fields are preserved exactly.
func TestGraph_CloneAndCloneEmptyContracts(t *testing.T) {
	g := NewGraphFull(t)

	eid1, err := g.AddEdge(VertexA, VertexB, Weight1)
	MustErrorNil(t, err, "AddEdge(A,B,1)")
	_, err = g.AddEdge(VertexA, VertexB, Weight2)
	MustErrorNil(t, err, "AddEdge(A,B,2)")

	ce := g.CloneEmpty()
	MustSameStringSet(t, g.Vertices(), ce.Vertices(), "CloneEmpty preserves vertex IDs")
	MustEqualInt(t, ce.EdgeCount(), Count0, "CloneEmpty has zero edges")

	c := g.Clone()
	MustSameStringSet(t, g.Vertices(), c.Vertices(), "Clone preserves vertex IDs")
	MustSameStringSet(t, ExtractEdgeIDs(g.Edges()), ExtractEdgeIDs(c.Edges()), "Clone preserves edge IDs")

	origEdge, err := g.GetEdge(eid1)
	MustErrorNil(t, err, "GetEdge(eid1) on original graph")
	cloneEdge, err := c.GetEdge(eid1)
	MustErrorNil(t, err, "GetEdge(eid1) on cloned graph")

	MustEqualBool(t, origEdge != cloneEdge, true, "Clone deep-copy: edge pointers must not alias")
	MustEqualString(t, cloneEdge.From, origEdge.From, "Clone preserves Edge.From")
	MustEqualString(t, cloneEdge.To, origEdge.To, "Clone preserves Edge.To")
	MustEqualBool(t, cloneEdge.Weight == origEdge.Weight, true, "Clone preserves Edge.Weight")
	MustEqualBool(t, cloneEdge.Directed == origEdge.Directed, true, "Clone preserves Edge.Directed")
}

// TestGraph_UnweightedViewCarriesNextEdgeID VERIFIES UnweightedView preserves edge-ID counter to avoid collisions.
// Implementation:
//   - Stage 1: Create a weighted source graph and add edges to advance ID counter.
//   - Stage 2: Build UnweightedView and assert it is unweighted.
//   - Stage 3: Assert copied edge weight is forced to zero in the view.
//   - Stage 4: AddEdge on the view must increase edge count and must not reuse an existing copied ID.
//   - Stage 5: Assert previously copied edge is still retrievable and unchanged in endpoints.
//
// Behavior highlights:
//   - Prevents ID collision bugs where new edges overwrite copied edges in the view.
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
//   - Time O(E) for view construction (copy), plus O(1) AddEdge, Space O(E) for copied edges.
//
// Notes:
//   - The contract is about counter carry-over, not ID formatting.
//
// AI-Hints:
//   - When debugging view/subgraph behaviors, always add a new edge after construction to probe ID collision risk.
func TestGraph_UnweightedViewCarriesNextEdgeID(t *testing.T) {
	// Stage 1: Build a weighted source graph and add edges to advance edge-ID counter.
	src, _ := core.NewGraph(core.WithWeighted())

	// Add first weighted edge and retain its ID.
	eid1, err := src.AddEdge(VertexA, VertexB, Weight1)
	MustErrorNil(t, err, "src.AddEdge(A,B,1)")
	// Add second weighted edge to further advance the counter.
	_, err = src.AddEdge(VertexB, VertexC, Weight2)
	MustErrorNil(t, err, "src.AddEdge(B,C,2)")

	// Stage 2: Build the unweighted view and validate Weighted()==false.
	view := core.UnweightedView(src)
	MustEqualBool(t, view.Weighted(), false, "UnweightedView(src) must return an unweighted graph")

	// Stage 3: Validate forced weight=0 for copied edges.
	e1, err := view.GetEdge(eid1)
	MustErrorNil(t, err, "view.GetEdge(eid1)")
	MustEqualBool(t, e1.Weight == float64(Weight0), true, "UnweightedView must force copied edge weights to 0")

	// Stage 4: AddEdge must not reuse an existing edge ID in the view.
	before := view.EdgeCount()
	newID, err := view.AddEdge(VertexX, VertexY, Weight0)
	MustErrorNil(t, err, "view.AddEdge(X,Y,0)")
	MustEqualInt(t, view.EdgeCount(), before+Count1, "AddEdge on view must increase edge count by 1")
	MustNotEqualString(t, newID, eid1, "AddEdge on view must not collide with copied edge IDs")

	// Stage 5: Previously copied edge must still exist and keep endpoints.
	e1After, err := view.GetEdge(eid1)
	MustErrorNil(t, err, "view.GetEdge(eid1) after adding new edge")
	MustEqualString(t, e1After.From, e1.From, "copied edge From must be preserved after AddEdge on view")
	MustEqualString(t, e1After.To, e1.To, "copied edge To must be preserved after AddEdge on view")
}

// TestGraph_UnweightedViewFunctionalSnapshot VERIFIES UnweightedView preserves topology and forces weights to zero.
// Implementation:
//   - Stage 1: Build a weighted directed source graph with two edges.
//   - Stage 2: Build UnweightedView(src).
//   - Stage 3: Assert Weighted()==false and inventories (Vertices, Edge IDs) match.
//   - Stage 4: For each edge ID, assert From/To/Directed preserved and Weight==0 in the view.
//   - Stage 5: Assert mutating the view does not mutate the source graph.
//
// Behavior highlights:
//   - Non-destructive, deterministic transform suitable for unweighted algorithms.
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
//   - Deterministic for a fixed input graph.
//
// Complexity:
//   - Time O(V+E) to build + O(E) to verify, Space O(V+E) for the view.
//
// Notes:
//   - Edge.ID preservation is a key debugging and stability feature.
//
// AI-Hints:
//   - Use UnweightedView when you need BFS/DFS-like behavior on a weighted input.
func TestGraph_UnweightedViewFunctionalSnapshot(t *testing.T) {
	// Stage 1: Build weighted directed source graph.
	src, _ := core.NewGraph(core.WithDirected(true), core.WithWeighted())

	// Add two edges with non-zero weights.
	id1, err := src.AddEdge(VertexA, VertexB, Weight1)
	MustErrorNil(t, err, "src.AddEdge(A,B,1)")
	id2, err := src.AddEdge(VertexB, VertexC, Weight7)
	MustErrorNil(t, err, "src.AddEdge(B,C,7)")

	// Stage 2: Build view.
	view := core.UnweightedView(src)

	// Stage 3: Inventories must match; view must be unweighted.
	MustEqualBool(t, view.Weighted(), false, "UnweightedView must return Weighted()==false")
	MustSameStringSet(t, view.Vertices(), src.Vertices(), "UnweightedView must preserve vertex ID set")
	MustSameStringSet(t, ExtractEdgeIDs(view.Edges()), ExtractEdgeIDs(src.Edges()), "UnweightedView must preserve edge ID set")

	// Stage 4: Per-edge topology and directedness must be preserved; weight forced to zero.
	ids := []string{id1, id2}
	for _, eid := range ids {
		orig, err := src.GetEdge(eid)
		MustErrorNil(t, err, "src.GetEdge(eid)")
		cpy, err := view.GetEdge(eid)
		MustErrorNil(t, err, "view.GetEdge(eid)")

		MustEqualString(t, cpy.From, orig.From, "UnweightedView must preserve Edge.From")
		MustEqualString(t, cpy.To, orig.To, "UnweightedView must preserve Edge.To")
		MustEqualBool(t, cpy.Directed == orig.Directed, true, "UnweightedView must preserve Edge.Directed")
		MustEqualBool(t, cpy.Weight == float64(Weight0), true, "UnweightedView must force Edge.Weight==0")
	}

	// Stage 5: Mutating view must not mutate src.
	before := src.EdgeCount()
	_, err = view.AddEdge(VertexX, VertexY, Weight0)
	MustErrorNil(t, err, "view.AddEdge(X,Y,0)")
	MustEqualInt(t, src.EdgeCount(), before, "mutating view must not change src.EdgeCount()")
}

// TestGraph_InducedSubgraphCarriesNextEdgeID VERIFIES InducedSubgraph preserves edge-ID counter to avoid collisions.
// Implementation:
//   - Stage 1: Create a weighted source graph and add edges A-B and B-C.
//   - Stage 2: Induce subgraph keeping only {A,B}; assert it keeps exactly one edge (A-B).
//   - Stage 3: AddEdge on subgraph must increase edge count and must not reuse an existing kept ID.
//   - Stage 4: Assert previously kept edge is still retrievable and unchanged in endpoints.
//
// Behavior highlights:
//   - Prevents ID collision bugs in induced subgraphs similar to view-collision class.
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
//   - Deterministic for fixed input graph.
//
// Complexity:
//   - Time O(E) to build induced subgraph (copy/filter), Space O(E_sub).
//
// Notes:
//   - This test asserts the kept-edge count for keep={A,B} based on the constructed source graph.
//
// AI-Hints:
//   - When changing subgraph policies, keep the “collision probe” pattern: build → add edge → re-read kept edge.
func TestGraph_InducedSubgraphCarriesNextEdgeID(t *testing.T) {
	// Stage 1: Create a weighted source graph with two edges.
	src, _ := core.NewGraph(core.WithWeighted())

	// Add edge A-B and retain its ID (it must be included in keep={A,B}).
	eidAB, err := src.AddEdge(VertexA, VertexB, Weight1)
	MustErrorNil(t, err, "src.AddEdge(A,B,1)")
	// Add edge B-C (will be excluded by keep={A,B}).
	_, err = src.AddEdge(VertexB, VertexC, Weight2)
	MustErrorNil(t, err, "src.AddEdge(B,C,2)")

	// Stage 2: Keep only {A,B}; induced subgraph must keep exactly the A-B edge.
	keep := map[string]bool{VertexA: true, VertexB: true}
	sub := core.InducedSubgraph(src, keep)

	MustEqualInt(t, sub.EdgeCount(), Count1, "InducedSubgraph keep={A,B} must keep exactly 1 edge")

	eAB, err := sub.GetEdge(eidAB)
	MustErrorNil(t, err, "sub.GetEdge(eidAB)")

	// Stage 3: AddEdge must not reuse an existing kept edge ID.
	before := sub.EdgeCount()
	newID, err := sub.AddEdge(VertexA, VertexD, Weight3)
	MustErrorNil(t, err, "sub.AddEdge(A,D,3)")
	MustEqualInt(t, sub.EdgeCount(), before+Count1, "AddEdge on subgraph must increase edge count by 1")
	MustNotEqualString(t, newID, eidAB, "new subgraph edge ID must not collide with kept eidAB")

	// Stage 4: Previously kept edge must still exist and keep endpoints.
	eABAfter, err := sub.GetEdge(eidAB)
	MustErrorNil(t, err, "sub.GetEdge(eidAB) after adding new edge")
	MustEqualString(t, eABAfter.From, eAB.From, "kept edge From must be preserved after AddEdge on subgraph")
	MustEqualString(t, eABAfter.To, eAB.To, "kept edge To must be preserved after AddEdge on subgraph")
}

// TestGraph_InducedSubgraphFunctionalCorrectness VERIFIES InducedSubgraph keeps exactly requested vertices and internal edges.
// Implementation:
//   - Stage 1: Create a weighted graph and add edges A-B, B-C, and A-C.
//   - Stage 2: Induce subgraph with keep={A,C}.
//   - Stage 3: Assert vertices are exactly {A,C}.
//   - Stage 4: Assert only the A-C edge remains and preserves weight/directness.
//
// Behavior highlights:
//   - InducedSubgraph is a deterministic filter that does not mutate the source graph.
//
// Inputs:
//   - None (uses package constants).
//
// Returns:
//   - None.
//
// Errors:
//   - Propagates sentinel errors from AddEdge/GetEdge.
//
// Determinism:
//   - Deterministic for fixed input and keep-set.
//
// Complexity:
//   - Time O(V+E) to filter/copy, Space O(V_sub+E_sub) for subgraph storage.
//
// Notes:
//   - This test intentionally uses a keep-set that excludes a “bridge” vertex to ensure edges are filtered correctly.
//
// AI-Hints:
//   - Prefer InducedSubgraph when you need an isolated working set of vertices for an algorithm stage.
func TestGraph_InducedSubgraphFunctionalCorrectness(t *testing.T) {
	// Stage 1: Build a weighted graph with a triangle A-B-C.
	src, _ := core.NewGraph(core.WithWeighted())

	_, err := src.AddEdge(VertexA, VertexB, Weight1)
	MustErrorNil(t, err, "src.AddEdge(A,B,1)")
	_, err = src.AddEdge(VertexB, VertexC, Weight2)
	MustErrorNil(t, err, "src.AddEdge(B,C,2)")
	idAC, err := src.AddEdge(VertexA, VertexC, Weight3)
	MustErrorNil(t, err, "src.AddEdge(A,C,3)")

	// Stage 2: Induce keep={A,C}.
	keep := map[string]bool{VertexA: true, VertexC: true}
	sub := core.InducedSubgraph(src, keep)

	// Stage 3: Vertices must be exactly {A,C}.
	MustSameStringSet(t, sub.Vertices(), []string{VertexA, VertexC}, "InducedSubgraph must keep exactly {A,C}")

	// Stage 4: Only A-C edge remains and preserves weight.
	MustEqualInt(t, sub.EdgeCount(), Count1, "InducedSubgraph keep={A,C} must keep exactly 1 edge")
	e, err := sub.GetEdge(idAC)
	MustErrorNil(t, err, "sub.GetEdge(idAC)")
	MustEqualString(t, e.From, VertexA, "kept edge must have From==A")
	MustEqualString(t, e.To, VertexC, "kept edge must have To==C")
	MustEqualBool(t, e.Weight == float64(Weight3), true, "kept edge must preserve Weight==3")

	// Edges incident to removed vertex must not exist (and predicate must be safe).
	MustEqualBool(t, sub.HasEdge(VertexA, VertexB), false, "sub.HasEdge(A,B) must be false when B is not kept")
	MustEqualBool(t, sub.HasEdge(VertexB, VertexC), false, "sub.HasEdge(B,C) must be false when B is not kept")
}
