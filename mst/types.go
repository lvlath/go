// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mst defines public result and compatibility configuration types.
// Algorithm kernels, option assembly, sentinels, and graph adapters live in dedicated files.
package mst

import (
	"errors"
	"math"

	"github.com/lvlath/go/core"
)

// Algorithm identifies the MST algorithm selected by the canonical facade.
//
// Implementation:
//   - Stage 1: Option assembly freezes one Algorithm value.
//   - Stage 2: MinimumSpanningTree dispatches to exactly one kernel.
//
// Behavior highlights:
//   - AlgorithmKruskal is the default because it does not need a root.
//   - AlgorithmPrim requires a root in strict tree mode.
//
// Determinism:
//   - Algorithm selection is exact constant comparison.
//
// AI-Hints:
//   - Do not parse algorithm names from error strings; use Algorithm constants and errors.Is.
type Algorithm string

const (
	// AlgorithmKruskal selects Kruskal's algorithm.
	// It sorts candidate edges and accepts edges that connect different DSU components.
	AlgorithmKruskal Algorithm = "kruskal"

	// AlgorithmPrim selects Prim's algorithm.
	// It grows one or more components from explicit or forest-selected roots.
	AlgorithmPrim Algorithm = "prim"
)

// Mode identifies the connectivity policy used by the canonical facade.
//
// Implementation:
//   - Stage 1: Option assembly freezes the requested Mode.
//   - Stage 2: Kernels either require one spanning tree or publish a spanning forest.
//
// Behavior highlights:
//   - ModeStrictTree is the default and returns ErrDisconnected for disconnected graphs.
//   - ModeForest is explicit opt-in and returns a minimum spanning forest.
//
// AI-Hints:
//   - Do not silently switch strict tree mode to forest mode after ErrDisconnected.
type Mode string

const (
	// ModeStrictTree requires exactly one spanning tree over all vertices.
	ModeStrictTree Mode = "strict_tree"

	// ModeForest returns one minimum spanning tree per connected component.
	ModeForest Mode = "forest"
)

// Result is the canonical result artifact for minimum spanning tree and forest computation.
// It captures not only selected edges and total weight, but also the algorithm, mode,
// root policy, vertex count, and component count used to interpret the result.
//
// Implementation:
//   - Stage 1: Kernels append detached core.Edge values into Edges.
//   - Stage 2: Kernels accumulate TotalWeight from accepted finite edges.
//   - Stage 3: Finalization fills ComponentCount and ComponentRoots.
//
// Behavior highlights:
//   - In ModeStrictTree, ComponentCount is 1 on success.
//   - In ModeForest, ComponentCount can be greater than 1.
//   - Root is meaningful for AlgorithmPrim; for Kruskal it is empty.
//   - Edges never retains *core.Edge pointers.
//
// Inputs:
//   - This type is returned by MinimumSpanningTree.
//
// Returns:
//   - Helper methods return detached data or ErrNilResult for nil receivers.
//
// Errors:
//   - EdgeValues returns ErrNilResult on a nil receiver.
//
// Determinism:
//   - Edge order follows the selected kernel's documented deterministic order.
//   - ComponentRoots follows the selected mode's deterministic root policy.
//
// Complexity:
//   - Clone and EdgeValues run in O(E + C) time and allocate O(E + C) space.
//   - IsNil runs in O(1).
//
// Notes:
//   - There are no partial results in this phase; result is nil on errors.
//   - Future cancellation/time-limit semantics must explicitly extend this contract.
//
// Nilability:
//   - IsNil and Clone are nil-safe.
//   - EdgeValues classifies nil access with ErrNilResult.
//
// AI-Hints:
//   - Do not expose raw internal edge pointers through Result.
//   - Do not treat ComponentRoots as arbitrary DSU roots; they are public interpretation metadata.
type Result struct {
	// Algorithm is the algorithm selected by the canonical facade.
	Algorithm Algorithm

	// Mode is the connectivity policy used to produce this result.
	Mode Mode

	// Root is the explicit Prim root in strict mode, or the first Prim component root in forest mode.
	Root string

	// Edges contains detached edge values selected for the tree or forest.
	Edges []core.Edge

	// TotalWeight is the sum of selected edge weights.
	TotalWeight float64

	// VertexCount records the number of vertices in the validated input graph.
	VertexCount int

	// ComponentCount records how many connected components were represented in the result.
	ComponentCount int

	// ComponentRoots records deterministic roots used to describe result components.
	ComponentRoots []string
}

// IsNil reports whether r is nil.
//
// Implementation:
//   - Stage 1: Compare the receiver pointer with nil.
//   - Stage 2: Return the boolean result without reading any fields.
//
// Behavior highlights:
//   - Safe for nil receivers.
//   - Useful for callers that handle optional results after errors.
//
// Returns:
//   - bool: true when the receiver is nil.
//
// Determinism:
//   - Pure pointer check; no package state is read.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This method does not classify errors; methods that need data return ErrNilResult.
//
// AI-Hints:
//   - Prefer IsNil over comparing helper-returned data when result ownership is unclear.
func (r *Result) IsNil() bool {
	return r == nil
}

// Clone returns a detached copy of the result.
// It preserves scalar metadata and deep-copies slice fields so callers can mutate the clone safely.
//
// Implementation:
//   - Stage 1: Return nil for a nil receiver.
//   - Stage 2: Copy scalar fields by value.
//   - Stage 3: Deep-copy Edges and ComponentRoots when present.
//
// Behavior highlights:
//   - Edges and ComponentRoots in the clone do not share backing arrays with the source result.
//   - No *core.Edge pointers are introduced.
//
// Returns:
//   - *Result: nil for a nil receiver; otherwise a detached result copy.
//
// Determinism:
//   - Slice order is preserved exactly.
//
// Complexity:
//   - Time O(E + C), Space O(E + C), where C is the component root count.
//
// Notes:
//   - This is a result-copy helper, not a graph-copy helper.
//   - The input graph is not referenced by Result, so Clone never touches core.Graph.
//
// AI-Hints:
//   - Do not replace append-copy with direct slice assignment; that would reintroduce aliasing.
func (r *Result) Clone() *Result {
	if r == nil {
		return nil
	}

	clone := *r
	if r.Edges != nil {
		clone.Edges = append([]core.Edge(nil), r.Edges...)
	}
	if r.ComponentRoots != nil {
		clone.ComponentRoots = append([]string(nil), r.ComponentRoots...)
	}

	return &clone
}

// EdgeValues returns detached MST or MSF edge values.
// It gives callers a fresh slice while preserving the deterministic result order.
//
// Implementation:
//   - Stage 1: Reject nil receiver with ErrNilResult.
//   - Stage 2: Copy Edges into a caller-owned slice.
//   - Stage 3: Return the copied slice.
//
// Behavior highlights:
//   - The returned slice can be reordered or truncated by the caller.
//   - Edge values are copies, not live graph edge pointers.
//
// Returns:
//   - []core.Edge: caller-owned copy of selected edges.
//   - error: ErrNilResult for nil receiver.
//
// Errors:
//   - ErrNilResult when called on a nil *Result.
//
// Determinism:
//   - Edge order is identical to r.Edges.
//
// Complexity:
//   - Time O(E), Space O(E).
//
// Notes:
//   - Mutating returned edge values does not mutate the result or the source graph.
//
// AI-Hints:
//   - Do not expose r.Edges directly from helper methods; preserve result ownership guarantees.
func (r *Result) EdgeValues() ([]core.Edge, error) {
	if r == nil {
		return nil, ErrNilResult
	}

	return append([]core.Edge(nil), r.Edges...), nil
}

// mstSnapshot is the detached kernel-local view of a validated core.Graph.
// It stores vertex order, non-loop edge values, and undirected adjacency lists for MST kernels.
//
// Implementation:
//   - Stage 1: newMSTSnapshot fills vertices from core.Vertices().
//   - Stage 2: newMSTSnapshot fills edges from detached core.Edge values.
//   - Stage 3: newMSTSnapshot mirrors every non-loop edge into both endpoint adjacency lists.
//
// Behavior highlights:
//   - The snapshot owns all edge values; it stores no live *core.Edge pointers.
//   - Self-loops are absent because they cannot connect components.
//   - Parallel edges remain independent candidates.
//   - Adjacency lists preserve core.Edges() order.
//
// Fields:
//   - vertices: sorted vertex IDs from core.Vertices().
//   - edges: detached non-loop edge candidates from core.Edges().
//   - adj: undirected adjacency relation keyed by vertex ID.
//
// Determinism:
//   - Vertex order is inherited from core.Vertices().
//   - Edge and adjacency order are inherited from core.Edges().
//
// Complexity:
//   - Space O(V + E).
//
// AI-Hints:
//   - Do not store *core.Edge here; snapshot ownership must stay detached from core.Graph.
type mstSnapshot struct {
	// vertices stores deterministic vertex IDs copied from core.Vertices().
	vertices []string

	// edges stores detached non-loop candidate edges in core.Edges() order.
	edges []core.Edge

	// adj stores undirected adjacency lists; each edge value appears under both endpoints.
	adj map[string][]core.Edge
}

// newMSTSnapshot validates graph policy and builds detached kernel-local edge storage.
//
// Implementation:
//   - Stage 1: Validate graph-level MST policy.
//   - Stage 2: Read vertices in core.Vertices() order.
//   - Stage 3: Read edges in core.Edges() order, reject non-finite weights, and skip self-loops.
//   - Stage 4: Store detached core.Edge values and undirected adjacency lists.
//
// Behavior highlights:
//   - Negative finite weights are accepted.
//   - Self-loops are ignored because they cannot reduce component count.
//   - Parallel edges are preserved as independent candidates.
//   - Directed edge-level overrides are rejected.
//
// Inputs:
//   - graph: candidate *core.Graph.
//
// Returns:
//   - *mstSnapshot: detached local representation for kernels.
//   - error: sentinel-classified validation failure.
//
// Errors:
//   - errors.Join(ErrInvalidGraph, ErrNilGraph) for nil graph.
//   - errors.Join(ErrInvalidGraph, ErrUnweightedGraph) for unweighted graphs.
//   - errors.Join(ErrInvalidGraph, ErrDirectedGraph) for directed graph policy.
//   - errors.Join(ErrInvalidGraph, ErrDirectedEdge) for directed edge-level overrides.
//   - errors.Join(ErrDisconnected, ErrEmptyGraph) for empty graphs.
//   - ErrNaNInfWeight for NaN or infinite weights.
//
// Determinism:
//   - Vertex order comes from core.Vertices().
//   - Edge order comes from core.Edges().
//   - Adjacency lists inherit core.Edges() order.
//
// Complexity:
//   - Time O(V + E), Space O(V + E).
//
// Notes:
//   - The snapshot intentionally detaches edge values from core.Graph.
//   - Concurrent graph mutation during snapshot construction remains unsupported.
//
// AI-Hints:
//   - Do not use core.Neighbors directly inside kernels after this adapter exists.
//   - Do not preserve self-loops in candidate edges; they are never useful for MST/MSF.
func newMSTSnapshot(graph *core.Graph) (*mstSnapshot, error) {
	// Validation input graph
	if graph == nil {
		return nil, errors.Join(ErrInvalidGraph, ErrNilGraph)
	}
	if !graph.Weighted() {
		return nil, errors.Join(ErrInvalidGraph, ErrUnweightedGraph)
	}
	if graph.Directed() {
		return nil, errors.Join(ErrInvalidGraph, ErrDirectedGraph)
	}
	if graph.HasDirectedEdges() {
		return nil, errors.Join(ErrInvalidGraph, ErrDirectedEdge)
	}

	vertices := graph.Vertices()
	if len(vertices) == 0 {
		return nil, errors.Join(ErrDisconnected, ErrEmptyGraph)
	}

	snapshot := &mstSnapshot{
		vertices: append([]string(nil), vertices...),
		edges:    make([]core.Edge, 0, graph.EdgeCount()),
		adj:      make(map[string][]core.Edge, len(vertices)),
	}

	for _, vertexID := range snapshot.vertices {
		snapshot.adj[vertexID] = nil
	}

	// Scan the stable edge catalog once and convert live graph edge pointers into detached values.
	for _, edge := range graph.Edges() {
		// Reject non-finite weights before any sorting or heap usage can observe them.
		if math.IsNaN(edge.Weight) || math.IsInf(edge.Weight, 0) {
			return nil, ErrNaNInfWeight
		}

		// Reject edge-level direction overrides because MST consumes an undirected relation only.
		if edge.Directed {
			return nil, errors.Join(ErrInvalidGraph, ErrDirectedEdge)
		}

		// Skip self-loops because they cannot connect two different DSU/Prim components.
		if edge.From == edge.To {
			continue
		}

		candidate := *edge

		// Store one detached candidate globally and mirror it into both endpoint adjacency lists.
		snapshot.edges = append(snapshot.edges, candidate)
		snapshot.adj[candidate.From] = append(snapshot.adj[candidate.From], candidate)
		snapshot.adj[candidate.To] = append(snapshot.adj[candidate.To], candidate)
	}

	return snapshot, nil
}

// hasVertex reports whether id exists in the snapshot vertex set.
// It uses binary search because newMSTSnapshot preserves the sorted order returned by core.Vertices().
//
// Implementation:
//   - Stage 1: Search the sorted vertex slice with a lower-bound loop.
//   - Stage 2: Confirm that the found position is in range and equal to id.
//
// Behavior highlights:
//   - Empty id is treated like any other missing vertex.
//   - The method does not consult core.Graph after snapshot construction.
//
// Inputs:
//   - id: vertex ID to check.
//
// Returns:
//   - bool: true when id exists in the snapshot.
//
// Determinism:
//   - Pure read over immutable snapshot order.
//
// Complexity:
//   - Time O(log V), Space O(1).
//
// Notes:
//   - This method relies on core.Vertices() returning sorted IDs.
//
// AI-Hints:
//   - Do not replace this with map iteration; the snapshot already owns deterministic vertex order.
func (s *mstSnapshot) hasVertex(id string) bool {
	left := 0
	right := len(s.vertices)

	for left < right {
		mid := left + (right-left)/2
		if s.vertices[mid] < id {
			left = mid + 1
			continue
		}
		right = mid
	}

	return left < len(s.vertices) && s.vertices[left] == id
}
