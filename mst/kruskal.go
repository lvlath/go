// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mst contains the Kruskal MST/MSF kernel.
// Public wrappers live in api.go and delegate here through MinimumSpanningTree.
package mst

import (
	"sort"

	"github.com/lvlath/go/core"
)

// kruskalKernel computes a minimum spanning tree or forest over a validated snapshot.
//
// Implementation:
//   - Stage 1: Copy detached non-loop snapshot edges into local candidate storage.
//   - Stage 2: Stable-sort candidates by ascending finite weight.
//   - Stage 3: Use DSU to accept only edges joining different components.
//   - Stage 4: Enforce strict tree connectivity or publish forest metadata.
//
// Behavior highlights:
//   - Equal-weight candidates retain snapshot edge order, inherited from core.Edges().
//   - Negative finite weights are valid.
//   - Self-loops were removed by the adapter.
//
// Inputs:
//   - snapshot: validated MST snapshot.
//   - cfg: finalized option policy.
//
// Returns:
//   - *Result: canonical detached result.
//   - error: ErrDisconnected in strict tree mode when not all vertices connect.
//
// Errors:
//   - ErrDisconnected for disconnected strict tree mode.
//
// Determinism:
//   - Candidate edge order is stable by Weight, then original core.Edges() order for ties.
//   - ComponentRoots are deterministic lexicographic minima for Kruskal forests.
//
// Complexity:
//   - Time O(E log E + E·α(V)), Space O(E + V).
//
// AI-Hints:
//   - Do not replace stable sort with an unstable sort; equal-weight MST representatives would drift.
func kruskalKernel(snapshot *mstSnapshot, cfg Options) (*Result, error) {
	vertexCount := len(snapshot.vertices)

	result := &Result{
		Algorithm:   AlgorithmKruskal,
		Mode:        cfg.Mode,
		Edges:       make([]core.Edge, 0, maxMSTEdgeCapacity(vertexCount)),
		VertexCount: vertexCount,
	}

	// Publish the trivial tree for a single-vertex graph without sorting or DSU allocation.
	if vertexCount == 1 {
		result.ComponentCount = 1
		result.ComponentRoots = []string{snapshot.vertices[0]}
		return result, nil
	}

	// Copy candidates before sorting so snapshot edge order remains reusable by other kernels/tests.
	candidates := append([]core.Edge(nil), snapshot.edges...)
	sort.SliceStable(candidates, func(i int, j int) bool {
		return candidates[i].Weight < candidates[j].Weight
	})

	// Initialize one DSU component per vertex.
	set := newDisjointSet(snapshot.vertices)

	for _, edge := range candidates {
		// Accept only edges that join two previously separate components.
		if !set.union(edge.From, edge.To) {
			continue
		}

		// Publish the detached edge value and accumulate its finite weight.
		result.Edges = append(result.Edges, edge)
		result.TotalWeight += edge.Weight

		// Strict tree mode can stop as soon as |V|-1 accepted edges are present.
		if cfg.Mode == ModeStrictTree && len(result.Edges) == vertexCount-1 {
			break
		}
	}

	// Compute deterministic component metadata after all accepted unions.
	result.ComponentRoots = set.componentRoots(snapshot.vertices)
	result.ComponentCount = len(result.ComponentRoots)

	// Strict tree mode requires one connected component and exactly |V|-1 accepted edges.
	if cfg.Mode == ModeStrictTree && len(result.Edges) != vertexCount-1 {
		return nil, ErrDisconnected
	}

	return result, nil
}

// maxMSTEdgeCapacity returns the maximum number of edges in a spanning tree over vertexCount vertices.
// It centralizes the |V|-1 capacity rule and avoids negative capacities for empty or single-vertex graphs.
//
// Implementation:
//   - Stage 1: Return 0 for vertexCount <= 1.
//   - Stage 2: Return vertexCount-1 for all larger graphs.
//
// Behavior highlights:
//   - Used only for slice preallocation.
//   - Does not validate graph connectivity.
//
// Inputs:
//   - vertexCount: number of vertices in a validated snapshot.
//
// Returns:
//   - int: safe edge-slice capacity.
//
// Determinism:
//   - Pure arithmetic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// AI-Hints:
//   - Do not use vertexCount-1 directly in make capacity without guarding vertexCount <= 1.
func maxMSTEdgeCapacity(vertexCount int) int {
	if vertexCount <= 1 {
		return 0
	}
	return vertexCount - 1
}
