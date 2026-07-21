package core_test

import (
	"testing"

	"github.com/lvlath/go/core"
)

// TestGraph_VertexLifecycle ASSERTS AddVertex/HasVertex/RemoveVertex invariants.
//
// Implementation:
//   - Stage 1: Create a graph.
//   - Stage 2: Reject empty ID on AddVertex.
//   - Stage 3: Add a vertex and validate presence.
//   - Stage 4: Duplicate AddVertex is no-op.
//   - Stage 5: RemoveVertex rejects empty and missing IDs.
//   - Stage 6: Remove existing vertex and validate absence.
//
// Behavior highlights:
//   - Locks in sentinel errors for empty/missing IDs.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal on mismatch.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(k log k) due to Vertices() sorting during count checks (implementation-dependent).
//
// Notes:
//   - This test relies on Vertices() being stable and safe.
//
// AI-Hints:
//   - Keep vertex IDs short and consistent to avoid noise in failure output.
func TestGraph_VertexLifecycle(t *testing.T) {
	g := NewGraphFull(t)

	err := g.AddVertex(VertexEmpty)
	MustErrorIs(t, err, core.ErrEmptyVertexID, "AddVertex(empty)")

	MustErrorNil(t, g.AddVertex(VertexV1), "AddVertex(V1)")
	MustEqualBool(t, g.HasVertex(VertexV1), true, "HasVertex(V1) after AddVertex(V1)")

	before := len(g.Vertices())
	MustErrorNil(t, g.AddVertex(VertexV1), "AddVertex(V1) duplicate")
	after := len(g.Vertices())
	MustEqualInt(t, after, before, "duplicate AddVertex(V1) must not change vertex count")

	err = g.RemoveVertex("Z")
	MustErrorIs(t, err, core.ErrVertexNotFound, "RemoveVertex(Z missing)")

	err = g.RemoveVertex(VertexEmpty)
	MustErrorIs(t, err, core.ErrEmptyVertexID, "RemoveVertex(empty)")

	MustErrorNil(t, g.RemoveVertex(VertexV1), "RemoveVertex(V1)")
	MustEqualBool(t, g.HasVertex(VertexV1), false, "HasVertex(V1) after RemoveVertex(V1)")
}

// TestGraph_VerticesMapReadOnly ASSERTS VerticesMap returns a safe snapshot.
//
// Implementation:
//   - Stage 1: Add vertex Z.
//   - Stage 2: Read VerticesMap snapshot.
//   - Stage 3: Mutate snapshot.
//   - Stage 4: Assert original graph is unchanged.
//
// Behavior highlights:
//   - Prevents external mutation through returned maps.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal if snapshot is not read-only.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(V) for snapshot copy, Space O(V).
//
// Notes:
//   - This locks in “defensive copy” behavior for maps.
//
// AI-Hints:
//   - Prefer snapshot APIs when you want a safe iteration without holding graph locks.
func TestGraph_VerticesMapReadOnly(t *testing.T) {
	g := NewGraphFull(t)

	MustErrorNil(t, g.AddVertex("Z"), "AddVertex(Z)")

	vm := g.VerticesMap()
	vm["NEW"] = &core.Vertex{ID: "NEW"}

	MustEqualBool(t, g.HasVertex("NEW"), false, "VerticesMap must be read-only snapshot")
}

// TestGraph_InternalVerticesDetachedSnapshot verifies that the compatibility accessor
// no longer exposes the live vertex catalog map.
func TestGraph_InternalVerticesDetachedSnapshot(t *testing.T) {
	g := NewGraphFull(t)
	MustErrorNil(t, g.AddVertex(VertexA), "AddVertex(A)")

	snapshot := g.InternalVertices()
	snapshot[VertexX] = &core.Vertex{ID: VertexX}
	delete(snapshot, VertexA)

	MustEqualBool(t, g.HasVertex(VertexA), true, "InternalVertices delete must not remove graph vertex")
	MustEqualBool(t, g.HasVertex(VertexX), false, "InternalVertices insert must not add graph vertex")
	MustEqualInt(t, g.VertexCount(), Count1, "InternalVertices detached map mutation must not change VertexCount")
}
