package core_test

import (
	"testing"

	"github.com/lvlath/go/core"
)

// TestGraph_Queries VERIFIES HasEdge mirror behavior, Neighbors ordering, Vertices ordering, and Edges inventory count.
// Implementation:
//   - Stage 1: Create a weighted, loop-enabled graph.
//   - Stage 2: Add one undirected edge V1-V2 and one self-loop V1-V1.
//   - Stage 3: Assert HasEdge mirrors undirected adjacency.
//   - Stage 4: Assert Neighbors(V1) returns edges sorted by Edge.ID and includes exactly two edges.
//   - Stage 5: Assert Vertices() returns sorted vertex IDs.
//   - Stage 6: Assert Edges() returns exactly two edges in this setup.
//
// Behavior highlights:
//   - Locks in deterministic ordering contracts for Vertices() and Neighbors().
//
// Inputs:
//   - None (uses package constants).
//
// Returns:
//   - None.
//
// Errors:
//   - Propagates any sentinel errors from AddVertex/AddEdge/Neighbors.
//
// Determinism:
//   - Vertices() order is deterministic (sorted).
//   - Neighbors() order is deterministic (sorted by Edge.ID).
//
// Complexity:
//   - Time O(k log k) for sorting in Vertices/Neighbors, Space O(k) for returned slices.
//
// Notes:
//   - This test uses minimal topology to keep ordering expectations unambiguous.
//
// AI-Hints:
//   - To validate determinism, always check sortedness of returned IDs rather than relying on insertion order.
func TestGraph_Queries(t *testing.T) {
	// Stage 1: Use a weighted, loop-enabled graph.
	g, _ := core.NewGraph(core.WithWeighted(), core.WithLoops())

	// Stage 2: Add one undirected edge V1–V2 and one self-loop V1–V1.
	MustErrorNil(t, g.AddVertex(VertexV1), "AddVertex(V1)")
	_, err := g.AddEdge(VertexV1, VertexV2, Weight0)
	MustErrorNil(t, err, "AddEdge(V1,V2,0)")
	_, err = g.AddEdge(VertexV1, VertexV1, Weight1)
	MustErrorNil(t, err, "AddEdge(V1,V1,1)")

	// Stage 3: Undirected edge must be mirrored for membership queries.
	MustEqualBool(t, g.HasEdge(VertexV2, VertexV1), true, "HasEdge(V2,V1) mirror for undirected edge")

	// Stage 4: Neighbors must return edges sorted by Edge.ID.
	nbs, err := g.Neighbors(VertexV1)
	MustErrorNil(t, err, "Neighbors(V1)")

	// Extract neighbor IDs in returned order.
	ids := make([]string, 0, len(nbs))
	for _, e := range nbs {
		ids = append(ids, e.ID)
	}

	// Validate sorted-by-ID contract.
	MustSortedStrings(t, ids, "Neighbors(V1) IDs must be sorted asc")
	// Validate neighbor count contract.
	MustEqualInt(t, len(ids), Count2, "Neighbors(V1) must contain exactly 2 edges (V1-V2 and V1-V1)")

	// Stage 5: Vertices() must return sorted IDs.
	vs := g.Vertices()
	MustSortedStrings(t, vs, "Vertices() must be sorted asc")

	// Stage 6: Edges inventory must include exactly two edges.
	ees := g.Edges()
	MustEqualInt(t, len(ees), Count2, "Edges() must contain exactly 2 edges in this setup")
}

// TestGraph_LoopsAndDirection VERIFIES self-loop behavior in undirected vs directed graphs.
// Implementation:
//   - Stage 1: Undirected + loops enabled: AddEdge(X,X,0) yields exactly one neighbor and one edge.
//   - Stage 2: Directed + loops enabled: AddEdge(Y,Y,0) yields exactly one neighbor with Directed==true.
//
// Behavior highlights:
//   - Ensures self-loop inventory is not duplicated in undirected mirror logic.
//
// Inputs:
//   - None (uses package constants).
//
// Returns:
//   - None.
//
// Errors:
//   - Propagates errors from AddEdge/Neighbors.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1) setup + O(deg(v)) for Neighbors, Space O(deg(v)) for returned slice.
//
// Notes:
//   - This test assumes Edges() is de-duplicated by Edge.ID in undirected mode.
//
// AI-Hints:
//   - Self-loop handling is a common source of double-counting bugs; keep this anchor test stable.
func TestGraph_LoopsAndDirection(t *testing.T) {
	// Stage 1: Undirected loop-enabled graph.
	{
		// Create undirected graph with loops enabled.
		g, _ := core.NewGraph(core.WithLoops())

		// Add self-loop on X.
		eid, err := g.AddEdge(VertexX, VertexX, Weight0)
		MustErrorNil(t, err, "AddEdge(X,X,0) undirected loops-enabled")

		// Neighbors(X) must return the loop exactly once.
		nbs, err := g.Neighbors(VertexX)
		MustErrorNil(t, err, "Neighbors(X) undirected loop")
		MustEqualInt(t, len(nbs), Count1, "Neighbors(X) undirected self-loop appears once")

		// Edges() must yield exactly one edge for a self-loop.
		ees := g.Edges()
		MustEqualInt(t, len(ees), Count1, "Edges() undirected self-loop yields one edge")
		MustEqualString(t, ees[0].ID, eid, "Edges()[0].ID equals AddEdge returned ID (undirected loop)")
	}

	// Stage 2: Directed loop-enabled graph.
	{
		// Create directed graph with loops enabled.
		g, _ := core.NewGraph(core.WithLoops(), core.WithDirected(true))

		// Add self-loop on Y.
		eid, err := g.AddEdge(VertexY, VertexY, Weight0)
		MustErrorNil(t, err, "AddEdge(Y,Y,0) directed loops-enabled")

		// Neighbors(Y) must return the loop once.
		nbs, err := g.Neighbors(VertexY)
		MustErrorNil(t, err, "Neighbors(Y) directed loop")
		MustEqualInt(t, len(nbs), Count1, "Neighbors(Y) directed self-loop appears once")

		// Directed flag must be true for directed self-loop edge.
		MustEqualBool(t, nbs[0].Directed, true, "Neighbors(Y)[0].Directed must be true in directed graph")
		// ID must match AddEdge return.
		MustEqualString(t, nbs[0].ID, eid, "Neighbors(Y)[0].ID equals AddEdge returned ID (directed loop)")
	}
}

// TestGraph_HasEdgeUnknownVertices ANCHORS the contract: HasEdge must be safe for unknown vertex IDs.
// Implementation:
//   - Stage 1: Create an empty graph.
//   - Stage 2: Call HasEdge(U,V) and assert false (and no panic).
//
// Behavior highlights:
//   - Keeps HasEdge usable as a fast-path membership predicate.
//
// Inputs:
//   - None (uses package constants).
//
// Returns:
//   - None.
//
// Errors:
//   - None (pure predicate).
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1) expected, Space O(1).
//
// Notes:
//   - This test is intentionally minimal; any panic fails the test implicitly.
//
// AI-Hints:
//   - Keep HasEdge safe even when vertices are not created (avoid forced AddVertex in callers).
func TestGraph_HasEdgeUnknownVertices(t *testing.T) {
	// Stage 1: Querying an empty graph with unknown vertices must be safe and return false.
	g := MustNewGraph(t)
	// Stage 2: Validate predicate result.
	MustEqualBool(t, g.HasEdge(VertexU, VertexV), false, "HasEdge(U,V) on unknown vertices must be false")
}

func TestGraph_RemoveVertexRemovesPublicAdjacencyKey(t *testing.T) {
	g := MustNewGraph(t)
	_, err := g.AddEdge("center", "up", 0)
	MustErrorNil(t, err, "AddEdge(center,up)")
	MustErrorNil(t, g.RemoveVertex("up"), "RemoveVertex(up)")

	adj := g.AdjacencyList()
	_, ok := adj["up"]
	MustEqualBool(t, ok, false, "removed vertex absent from public AdjacencyList")
}

func TestGraph_AdjacencyListIncludesNewIsolatedVertex(t *testing.T) {
	g := MustNewGraph(t)
	MustErrorNil(t, g.AddVertex("up"), "AddVertex(up)")

	adj := g.AdjacencyList()
	MustEqualInt(t, len(adj["up"]), 0, "public AdjacencyList includes isolated up")
}

func TestGraph_RemoveEdgeLeavesIsolatedEndpointInPublicAdjacencyList(t *testing.T) {
	g := MustNewGraph(t)
	id, err := g.AddEdge("center", "up", 0)
	MustErrorNil(t, err, "AddEdge(center,up)")
	MustErrorNil(t, g.RemoveEdge(id), "RemoveEdge(center-up)")

	MustEqualBool(t, g.HasVertex("up"), true, "up remains a vertex")
	adj := g.AdjacencyList()
	MustEqualInt(t, len(adj["up"]), 0, "up has empty public adjacency")
}

// TestGraph_NeighborIDsUniqueSorted verifies NeighborIDs projection semantics.
//
// Contract anchors:
//   - Parallel edges do not duplicate neighbor IDs.
//   - Returned IDs are sorted lexicographically ascending.
//   - NeighborIDs consumes the public adjacency relation, not raw edge count.
func TestGraph_NeighborIDsUniqueSorted(t *testing.T) {
	g := MustNewGraph(t, core.WithMultiEdges())

	_, err := g.AddEdge(VertexA, VertexC, Weight0)
	MustErrorNil(t, err, "AddEdge(A,C)")
	_, err = g.AddEdge(VertexA, VertexB, Weight0)
	MustErrorNil(t, err, "AddEdge(A,B) first")
	_, err = g.AddEdge(VertexA, VertexB, Weight0)
	MustErrorNil(t, err, "AddEdge(A,B) parallel")

	ids, err := g.NeighborIDs(VertexA)
	MustErrorNil(t, err, "NeighborIDs(A)")
	MustEqualInt(t, len(ids), Count2, "NeighborIDs(A) unique count")
	MustSortedStrings(t, ids, "NeighborIDs(A) sorted order")
	MustSameStringSet(t, ids, []string{VertexB, VertexC}, "NeighborIDs(A) unique neighbor set")
}

// TestGraph_NeighborsMissingVertex verifies sentinel-first validation for neighborhood lookup.
//
// Contract anchors:
//   - Neighbors(missing) returns ErrVertexNotFound.
//   - Missing vertex lookup returns no neighbor slice.
//   - Callers can classify failure through errors.Is, not error strings.
func TestGraph_NeighborsMissingVertex(t *testing.T) {
	g := MustNewGraph(t)

	nbs, err := g.Neighbors(VertexMissing)

	MustErrorIs(t, err, core.ErrVertexNotFound, "Neighbors(missing)")
	MustEqualInt(t, len(nbs), Count0, "Neighbors(missing) result length")
}

// TestGraph_AdjacencyListIncludesIsolatedVertices verifies that AdjacencyList is a
// vertex-catalog snapshot, not merely a non-empty adjacency-bucket dump.
func TestGraph_AdjacencyListIncludesIsolatedVertices(t *testing.T) {
	g := MustNewGraph(t)
	MustErrorNil(t, g.AddVertex(vertexJohn), "AddVertex(John)")
	MustErrorNil(t, g.AddVertex(vertexMary), "AddVertex(Mary)")
	_, err := g.AddEdge(vertexAlice, vertexBob, Weight0, core.WithID("friend_alice_bob"))
	MustErrorNil(t, err, "AddEdge(Alice,Bob)")

	adj := g.AdjacencyList()

	MustEqualInt(t, len(adj), Count4, "AdjacencyList vertex-key count")
	MustEqualInt(t, len(adj[vertexJohn]), Count0, "John isolated adjacency")
	MustEqualInt(t, len(adj[vertexMary]), Count0, "Mary isolated adjacency")
	MustSameStringSet(t, adj[vertexAlice], []string{"friend_alice_bob"}, "Alice adjacency IDs")
	MustSameStringSet(t, adj[vertexBob], []string{"friend_alice_bob"}, "Bob mirrored adjacency IDs")

	adj[vertexJohn] = append(adj[vertexJohn], "fake")
	MustEqualBool(t, g.HasEdge(vertexJohn, vertexMary), false, "mutating returned adjacency snapshot must not mutate graph")
}

// TestGraph_DegreeDirectedAndUndirectedLoops verifies loop-aware degree components.
func TestGraph_DegreeDirectedAndUndirectedLoops(t *testing.T) {
	g := MustNewMixedGraph(t, core.WithLoops(), core.WithMultiEdges())

	_, err := g.AddEdge(VertexA, VertexA, Weight0)
	MustErrorNil(t, err, "AddEdge undirected self-loop A-A")
	_, err = g.AddEdge(VertexA, VertexA, Weight0, core.WithEdgeDirected(true))
	MustErrorNil(t, err, "AddEdge directed self-loop A->A")
	_, err = g.AddEdge(VertexB, VertexA, Weight0, core.WithEdgeDirected(true))
	MustErrorNil(t, err, "AddEdge directed incoming B->A")
	_, err = g.AddEdge(VertexA, VertexC, Weight0, core.WithEdgeDirected(true))
	MustErrorNil(t, err, "AddEdge directed outgoing A->C")
	_, err = g.AddEdge(VertexA, VertexB, Weight0)
	MustErrorNil(t, err, "AddEdge undirected A-B")

	in, out, undirected, err := g.Degree(VertexA)
	MustErrorNil(t, err, "Degree(A)")
	MustEqualInt(t, in, Count2, "Degree(A).in")
	MustEqualInt(t, out, Count2, "Degree(A).out")
	MustEqualInt(t, undirected, Count3, "Degree(A).undirected")
}

// TestGraph_DegreeValidationAndIsolatedVertex verifies degree behavior for missing and isolated vertices.
//
// Contract anchors:
//   - Degree(missing) returns ErrVertexNotFound.
//   - An explicitly added isolated vertex has zero in/out/undirected degree.
//   - Public vertex membership is independent from the sparse adjacency index.
func TestGraph_DegreeValidationAndIsolatedVertex(t *testing.T) {
	g := MustNewGraph(t)

	in, out, undirected, err := g.Degree(VertexMissing)
	MustErrorIs(t, err, core.ErrVertexNotFound, "Degree(missing)")
	MustEqualInt(t, in, Count0, "Degree(missing).in")
	MustEqualInt(t, out, Count0, "Degree(missing).out")
	MustEqualInt(t, undirected, Count0, "Degree(missing).undirected")

	MustErrorNil(t, g.AddVertex(VertexA), "AddVertex(A isolated)")

	in, out, undirected, err = g.Degree(VertexA)
	MustErrorNil(t, err, "Degree(A isolated)")
	MustEqualInt(t, in, Count0, "Degree(A isolated).in")
	MustEqualInt(t, out, Count0, "Degree(A isolated).out")
	MustEqualInt(t, undirected, Count0, "Degree(A isolated).undirected")
}
