// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mst contains the shared Kruskal edge-selection kernel.
// The public MST kernel publishes a total; Steiner reuses selection without an intermediate total.
package mst

import (
	"context"
	"sort"

	"github.com/lvlath/go/core"
)

// lessMSTEdge compares finite MST candidates using their full deterministic ordering key.
// Weight has priority, and original Edge.ID is the strict tie-break for equal weights.
// Implementation:
//   - Stage 1: Compare the finite weights without epsilon or negation.
//   - Stage 2: Break equal-weight ties by unique edge identity.
//
// Behavior highlights:
//   - The same order is used by Prim's heap and Kruskal selection.
//
// Inputs:
//   - left, right: validated finite weighted core.Edge values.
//
// Returns:
//   - bool: true exactly when left precedes right in the order.
//
// Errors:
//   - None; numeric admission precedes this comparator.
//
// Determinism:
//   - Lexicographic tuple (Weight, Edge.ID) with exact float64 comparisons.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This matches legacy Kruskal: core.Edges() was already sorted by ID before stable sorting by Weight.
//
// AI-Hints:
//   - Do not introduce epsilon-based comparison; it breaks strict weak ordering.
func lessMSTEdge(left, right core.Edge) bool {
	if left.Weight != right.Weight {
		return left.Weight < right.Weight
	}
	return left.ID < right.ID
}

// selectKruskalEdges selects an MST or MSF without calculating any scalar total.
// Ordinary Kruskal, the terminal closure, and the original-edge union share one DSU/sort path.
// Implementation:
//   - Stage 1: Copy candidates and sort them by (Weight, Edge.ID).
//   - Stage 2: Merge distinct components through the existing disjointSet.
//   - Stage 3: Enforce strict connectivity and return canonical component roots.
//
// Behavior highlights:
//   - Strict mode stops at |V|-1 accepted edges; forest mode consumes all candidates.
//   - Valid finite negative weights are accepted here; Steiner enforces its stricter domain upstream.
//   - Selected edges retain input identities and original weights.
//
// Inputs:
//   - ctx: non-nil cancellation context; ordinary MST passes context.Background().
//   - vertices: sorted, unique local vertex IDs, read-only.
//   - candidates: validated undirected non-loop edges with endpoints in vertices.
//   - mode: ModeStrictTree or ModeForest.
//
// Returns:
//   - []core.Edge: detached selection in (Weight, Edge.ID) order.
//   - []string: lexicographically sorted component minima, detached.
//   - error: nil on success; no partial selection on error.
//
// Errors:
//   - ErrDisconnected in strict mode with multiple components.
//   - ErrInvalidOption for an unrecognized mode, ErrInvalidGraph for missing endpoints,
//     or context cancellation unchanged.
//
// Determinism:
//   - Full edge key (Weight, Edge.ID), DSU unions, and sorted component roots.
//
// Complexity:
//   - Time O(M log M + M*alpha(V) + V log V), Space O(M+V) for M candidates.
//
// Notes:
//   - No intermediate float64 TotalWeight is materialized; the caller owns publication policy.
//
// AI-Hints:
//   - Do not call the public MST facade on terminal closure: its unnecessary total can overflow.
//   - Do not duplicate union-find to optimize the Steiner union stage.
func selectKruskalEdges(
	ctx context.Context,
	vertices []string,
	candidates []core.Edge,
	mode Mode,
) ([]core.Edge, []string, error) {
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	if mode != ModeStrictTree && mode != ModeForest {
		return nil, nil, ErrInvalidOption
	}

	vertexCount := len(vertices)
	if vertexCount == 1 {
		// The ordinary Kruskal result historically publishes a non-nil empty
		// edge slice for a singleton. Preserve that observable slice state.
		return make([]core.Edge, 0), []string{vertices[0]}, nil
	}

	ordered := append([]core.Edge(nil), candidates...)
	sort.Slice(ordered, func(i, j int) bool {
		return lessMSTEdge(ordered[i], ordered[j])
	})

	set := newDisjointSet(vertices)
	selected := make([]core.Edge, 0, maxMSTEdgeCapacity(vertexCount))
	for _, edge := range ordered {
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}

		// The graph/closure adapter must declare every endpoint. Guard this
		// invariant because the existing DSU does not create missing vertices.
		if _, exists := set.parent[edge.From]; !exists {
			return nil, nil, ErrInvalidGraph
		}
		if _, exists := set.parent[edge.To]; !exists {
			return nil, nil, ErrInvalidGraph
		}
		if !set.union(edge.From, edge.To) {
			continue
		}
		selected = append(selected, edge)
		if mode == ModeStrictTree && len(selected) == vertexCount-1 {
			break
		}
	}

	// Determine connectivity before the caller attempts to publish a numeric sum.
	if mode == ModeStrictTree && len(selected) != vertexCount-1 {
		return nil, nil, ErrDisconnected
	}
	roots := set.componentRoots(vertices)
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	return selected, roots, nil
}

// kruskalKernel computes the ordinary minimum spanning tree or forest over a validated snapshot.
// It consumes the shared selection but preserves the existing Result type and numeric fold order.
// Implementation:
//   - Stage 1: Select edges and determine structural connectivity without aggregating cost.
//   - Stage 2: Compute the checked public total in accepted-edge order.
//   - Stage 3: Publish the Result and component metadata only if both stages succeed.
//
// Behavior highlights:
//   - Disconnected strict graphs return ErrDisconnected before any new arithmetic error.
//   - Finite negative weights, forest, isolates and historical edge selection are preserved.
//
// Inputs:
//   - snapshot: detached, validated graph snapshot.
//   - cfg: finalized old MST policy; no caller callbacks are executed here.
//
// Returns:
//   - *Result: the usual MST/MSF result, or nil on any failure.
//
// Errors:
//   - ErrDisconnected from selection; ErrWeightOverflow from total publication.
//
// Determinism:
//   - The same (Weight, Edge.ID) accepted-edge order as previous stable Kruskal.
//
// Complexity:
//   - Time O(E log E + E*alpha(V) + V log V), Space O(E+V).
//
// Notes:
//   - This is the sole ordinary Kruskal kernel and shares no numeric policy with closure selection.
//
// AI-Hints:
//   - Keep structural checks before checked weight publication and preserve negative MST weights.
func kruskalKernel(snapshot *mstSnapshot, cfg Options) (*Result, error) {
	edges, roots, err := selectKruskalEdges(
		context.Background(), snapshot.vertices, snapshot.edges, cfg.Mode,
	)
	if err != nil {
		return nil, err
	}
	total, err := checkedMSTEdgeTotal(edges)
	if err != nil {
		return nil, err
	}
	return &Result{
		Algorithm:      AlgorithmKruskal,
		Mode:           cfg.Mode,
		Edges:          edges,
		TotalWeight:    total,
		VertexCount:    len(snapshot.vertices),
		ComponentCount: len(roots),
		ComponentRoots: roots,
	}, nil
}

// maxMSTEdgeCapacity returns the maximum number of tree edges for vertexCount vertices.
// It protects slice capacity calculations for empty or singleton vertex sets.
// Implementation:
//   - Stage 1: Return zero for vertexCount at most one.
//   - Stage 2: Return vertexCount-1 otherwise.
//
// Behavior highlights:
//   - It does not validate graph connectivity.
//
// Inputs:
//   - vertexCount: count of local vertices, nonnegative.
//
// Returns:
//   - int: nonnegative safe upper bound on accepted tree edges.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pure arithmetic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - A spanning forest also has at most V-1 edges for nonempty V.
//
// AI-Hints:
//   - Guard V-1 when V may be zero; never make a negative slice capacity.
func maxMSTEdgeCapacity(vertexCount int) int {
	if vertexCount <= 1 {
		return 0
	}
	return vertexCount - 1
}
