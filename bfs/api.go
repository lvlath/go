// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package bfs

import (
	"context"
	"fmt"

	"github.com/lvlath/go/core"
)

// BFS performs deterministic, unweighted breadth-first search from startID.
//
// Implementation:
//   - Stage 1: Validate inputs and apply options (fail-fast).
//   - Stage 2: Validate graph constraints (unweighted, start exists).
//   - Stage 3: Delegate to the BFS kernel (no algorithmic loop here).
//
// Behavior highlights:
//   - Determinism: traversal preserves the neighbor order provided by core.Graph.NeighborIDs.
//   - Partial result: on early exit (cancel, neighbor fetch error, hook error), returns a partial Result with error.
//
// Inputs:
//   - g: graph instance; must be non-nil.
//   - startID: existing vertex ID to start from.
//   - opts: functional options; last-writer-wins.
//
// Returns:
//   - *Result: traversal result (may be partial on error).
//   - error: sentinel-classified error (use errors.Is).
//
// Errors:
//   - ErrGraphNil if g is nil (Stage 1).
//   - ErrOptionViolation if any option is invalid (Stage 1).
//   - ErrWeightedGraph if g is weighted (Stage 2).
//   - ErrStartVertexNotFound if startID is absent (Stage 2).
//   - ErrNeighborFetch on neighbor enumeration errors (Stage 3).
//   - context.Canceled / context.DeadlineExceeded on context termination (Stage 3).
//
// Determinism:
//   - FIFO order is fixed; per-vertex neighbor iteration order is preserved as surfaced by NeighborIDs.
//
// Complexity:
//   - Time O(|V|+|E|), Space O(|V|) in the kernel implementation.
//
// Notes:
//   - BFS treats the graph as read-only during the call.
//
// AI-Hints:
//   - Hooks are observers; do not mutate the graph or you risk violating graph invariants.
//   - If you need weighted shortest paths, use a dedicated algorithm (e.g., Dijkstra); BFS is edge-count distance.
//   - Avoid time-based cancellation in Examples; use WithCancel + deterministic trigger.
func BFS(g *core.Graph, startID string, opts ...Option) (*Result, error) {
	// Stage 1: Validate graph pointer first.
	if g == nil {
		return nil, ErrGraphNil
	}

	// Stage 1: Apply options with fail-fast semantics.
	o, err := applyOptions(opts...)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrOptionViolation, err)
	}

	// Stage 2: Reject weighted graphs (BFS is unweighted shortest path only).
	if g.Weighted() {
		return nil, ErrWeightedGraph
	}

	// Stage 2: Validate that the start vertex exists.
	if !g.HasVertex(startID) {
		return nil, ErrStartVertexNotFound
	}

	// Stage 3: Delegate to the kernel (must not live in api.go).
	return runBFS(g, startID, o)
}

// Components computes weakly-connected components under an undirected relation.
//
// Implementation:
//   - Stage 1: Validate inputs (g != nil, ctx != nil).
//   - Stage 2: Delegate to the components kernel (no topology mutation here).
//
// Behavior highlights:
//   - Weak connectivity: directed edges still connect their endpoints for component membership.
//   - Determinism: components and vertex IDs are returned in stable sorted order.
//
// Inputs:
//   - ctx: cancellation context; must be non-nil.
//   - g: graph instance; must be non-nil.
//
// Returns:
//   - *ComponentsResult: deterministic component listing (may be partial on ctx cancellation).
//   - error: ctx.Err() on cancellation; otherwise nil.
//
// Errors:
//   - ErrGraphNil if g is nil (Stage 1).
//   - ErrOptionViolation if ctx is nil (Stage 1).
//   - context.Canceled / context.DeadlineExceeded if ctx is done (Stage 2).
//
// Determinism:
//   - Components are stable for deterministic core enumeration; component IDs are lex-sorted.
//
// Complexity:
//   - Time O(V log V + E log E + Σ deg(v) log deg(v) + (V+E)), Space O(V+E).
//
// Notes:
//   - Components ≠ SCC: this is weak connectivity, not strong connectivity.
//
// AI-Hints:
//   - Use Components for undirected connectivity checks even on directed graphs.
//   - Avoid mutating the graph concurrently; cancellation is supported but topology mutation may yield partial snapshots.
func Components(ctx context.Context, g *core.Graph) (*ComponentsResult, error) {
	// Minimal validation here; kernel will be introduced in a later stage.
	if g == nil {
		return nil, ErrGraphNil
	}
	if ctx == nil {
		return nil, fmt.Errorf("%w: context is nil", ErrOptionViolation)
	}
	// Delegation point for the future kernel.
	return runComponents(ctx, g)
}
