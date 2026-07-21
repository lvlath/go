// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// DFS traversal implementation and runtime state for core.Graph.
package dfs

import (
	"fmt"

	"github.com/lvlath/go/core"
)

// dfsWalker owns runtime-only DFS state during a single traversal execution.
//
// Implementation:
//   - Stage 1: Hold immutable graph and traversal policy references.
//   - Stage 2: Accumulate runtime traversal state and diagnostics.
//   - Stage 3: Write observable results into Result as vertices are entered and finished.
//
// Behavior highlights:
//   - The struct is internal and per-execution.
//   - Runtime counters do not leak back into Option.
//
// Inputs:
//   - N/A.
//
// Returns:
//   - N/A.
//
// Errors:
//   - N/A.
//
// Determinism:
//   - Deterministic under deterministic graph order, neighbor order, and callbacks.
//
// Complexity:
//   - Space O(V) through the owned Result maps and slices.
//
// Notes:
//   - The struct is intentionally not reused across DFS calls.
//
// AI-Hints:
//   - Keep runtime state here, not inside Option.
//   - Diagnostics such as skipped-neighbor counts belong to execution state, not configuration.
type dfsWalker struct {
	// graph is the traversed graph.
	graph *core.Graph

	// opts is the finalized traversal policy for this execution.
	opts Options

	// res accumulates the observable traversal result.
	res *Result

	// skippedNeighbors counts neighbors rejected by FilterNeighbor during this execution.
	skippedNeighbors int
}

// runDFS performs depth-first traversal over g using the provided traversal policy.
//
// Implementation:
//   - Stage 1: Validate the input graph.
//   - Stage 2: Assemble and validate DFS options.
//   - Stage 3: Validate the starting-root policy.
//   - Stage 4: Allocate deterministic traversal result storage.
//   - Stage 5: Execute either single-tree traversal or DFS-forest traversal.
//   - Stage 6: Expose runtime diagnostics in the returned Result.
//
// Behavior highlights:
//   - Order is DFS finish order (post-order).
//   - FullTraversal produces a DFS forest rather than a single global DFS tree.
//   - Depth records DFS-tree depth, not shortest-path distance.
//   - Traversal direction for mixed graphs is interpreted per edge.
//
// Inputs:
//   - g: graph to traverse.
//   - startID: starting vertex ID for single-source traversal.
//   - opts: ordered DFS option builders.
//
// Returns:
//   - *Result: traversal result, including visited flags, tree parents, depths, and finish order.
//   - error: nil on success, or a traversal/configuration failure.
//
// Errors:
//   - ErrGraphNil: if g is nil.
//   - ErrStartVertexNotFound: if startID does not exist in single-source mode.
//   - ErrOptionViolation: if option assembly rejects explicit input.
//   - ErrNeighborFetch: if graph neighbor enumeration fails.
//   - context.Canceled / context.DeadlineExceeded: if traversal context is canceled.
//   - Any user callback error returned by OnVisit or OnExit.
//
// Determinism:
//   - Deterministic under deterministic graph root order, neighbor order, and deterministic callbacks.
//
// Complexity:
//   - Time O(V + E) for traversal itself, plus callback and filter costs.
//   - Space O(V) for result maps, recursion stack, and post-order output.
//
// Notes:
//   - On error, Visited, Depth, and Parent may contain partial traversal state.
//   - Order is cleared when traversal aborts because partial finish order is not exposed as authoritative output.
//
// AI-Hints:
//   - Order is post-order finish order, not discovery order.
//   - FullTraversal resets tree depth at each new DFS-tree root.
//   - Mixed-edge traversal must interpret direction per edge, not via edge.To.
//   - Invalid option input is a configuration failure, not a runtime traversal event.
func runDFS(g *core.Graph, startID string, opts ...Option) (*Result, error) {
	// Reject a nil graph immediately because traversal semantics require a concrete graph instance.
	if g == nil {
		return nil, ErrGraphNil
	}

	// Assemble the canonical DFS configuration before any traversal state is allocated.
	dopts, err := buildOption(opts...)
	if err != nil {
		return nil, err
	}

	// In single-source mode, the requested start vertex must exist explicitly.
	if !dopts.FullTraversal && !g.HasVertex(startID) {
		return nil, ErrStartVertexNotFound
	}

	// Use the graph's reliable vertex count for preallocation.
	vertexCount := g.VertexCount()

	// Preallocate result storage from the stable vertex-count hint.
	res := &Result{
		Order:   make([]string, 0, vertexCount),
		Depth:   make(map[string]int, vertexCount),
		Parent:  make(map[string]string, vertexCount),
		Visited: make(map[string]bool, vertexCount),
	}

	// Build the per-execution traversal runtime.
	walker := &dfsWalker{
		graph: g,
		opts:  dopts,
		res:   res,
	}

	// Full traversal visits every unvisited root in graph vertex order, forming a DFS forest.
	if dopts.FullTraversal {
		for _, vertexID := range g.Vertices() {
			if walker.res.Visited[vertexID] {
				continue
			}

			if err = walker.traverse(vertexID, 0, ""); err != nil {
				walker.res.SkippedNeighbors = walker.skippedNeighbors
				return walker.res, err
			}
		}
	} else {
		// Single-source traversal starts only from the validated start vertex.
		if err = walker.traverse(startID, 0, ""); err != nil {
			walker.res.SkippedNeighbors = walker.skippedNeighbors
			return walker.res, err
		}
	}

	// Expose runtime diagnostics in the caller-owned result.
	walker.res.SkippedNeighbors = walker.skippedNeighbors

	return walker.res, nil
}

// traverse visits vertexID at the given DFS-tree depth.
//
// Implementation:
//   - Stage 1: Observe context cancellation.
//   - Stage 2: Enforce depth policy before vertex entry.
//   - Stage 3: Record parent only when the vertex is actually entered.
//   - Stage 4: Mark the vertex as entered and record DFS-tree depth.
//   - Stage 5: Execute the pre-order hook.
//   - Stage 6: Enumerate graph neighbors exactly once.
//   - Stage 7: Resolve per-edge traversal semantics and recurse on eligible neighbors.
//   - Stage 8: Execute the post-order hook.
//   - Stage 9: Append the vertex to DFS finish order.
//
// Behavior highlights:
//   - Parent is assigned only when vertexID actually passes entry guards.
//   - FilterNeighbor applies after edge-to-neighbor semantics are resolved.
//   - Self-loop policy remains separate from neighbor resolution policy.
//
// Inputs:
//   - vertexID: current vertex being traversed.
//   - depth: DFS-tree depth of vertexID.
//   - parentID: DFS-tree parent candidate for vertexID.
//
// Returns:
//   - error: nil on success, or a traversal failure.
//
// Errors:
//   - ErrNeighborFetch: if neighbor enumeration fails.
//   - context.Canceled / context.DeadlineExceeded: if the traversal context is canceled.
//   - Any user callback error returned by OnVisit or OnExit.
//
// Determinism:
//   - Deterministic under deterministic neighbor order and deterministic callbacks.
//
// Complexity:
//   - Local overhead is O(out-degree(vertexID)) plus callback and filter cost.
//   - Overall DFS complexity remains O(V + E) across the full traversal, excluding callback and filter cost.
//
// Notes:
//   - On failure, Order is cleared because partial finish order is not exposed as authoritative output.
//
// AI-Hints:
//   - Never assign Parent before the child actually enters traversal.
//   - Never infer undirected neighbors via edge.To.
//   - Apply filtering only after neighbor semantics are resolved.
func (w *dfsWalker) traverse(vertexID string, depth int, parentID string) error {
	// Respect traversal cancellation before performing any new vertex work.
	select {
	case <-w.opts.Ctx.Done():
		w.res.Order = nil
		return w.opts.Ctx.Err()
	default:
	}

	// Reject traversal entry beyond the configured depth limit.
	if w.opts.MaxDepth != NoDepthLimit && depth > w.opts.MaxDepth {
		return nil
	}

	// Record the DFS-tree parent only after the vertex has passed all pre-entry guards.
	if depth > 0 {
		w.res.Parent[vertexID] = parentID
	}

	// Mark the vertex as entered and record its DFS-tree depth.
	w.res.Visited[vertexID] = true
	w.res.Depth[vertexID] = depth

	// Run the pre-order callback immediately after entry if one is configured.
	if w.opts.OnVisit != nil {
		if err := w.opts.OnVisit(vertexID); err != nil {
			w.res.Order = nil
			return fmt.Errorf("dfs: OnVisit(%q): %w", vertexID, err)
		}
	}

	// Read neighbor edges exactly once so traversal logic works from a stable local slice.
	neighbors, err := w.graph.Neighbors(vertexID)
	if err != nil {
		w.res.Order = nil
		return fmt.Errorf("%w: neighbors(%q): %w", ErrNeighborFetch, vertexID, err)
	}

	// Process neighbors in graph-provided order so traversal determinism follows graph determinism.
	for _, edge := range neighbors {
		// Resolve the actual traversal neighbor under per-edge directed or undirected semantics.
		neighborID, ok := neighborFromEdge(edge, vertexID)
		if !ok {
			continue
		}

		// Enforce self-loop policy independently from neighbor resolution semantics.
		if neighborID == vertexID && !w.graph.Looped() {
			continue
		}

		// Apply caller-provided neighbor filtering only after a valid candidate neighbor exists.
		if w.opts.FilterNeighbor != nil && !w.opts.FilterNeighbor(neighborID) {
			w.skippedNeighbors++
			continue
		}

		// Skip already-entered vertices because DFS-tree parent assignment is defined only on first entry.
		if w.res.Visited[neighborID] {
			continue
		}

		// Recurse to the child at the next DFS-tree depth level.
		if err = w.traverse(neighborID, depth+1, vertexID); err != nil {
			w.res.Order = nil
			return err
		}
	}

	// Run the post-order callback after all reachable descendants have been processed.
	if w.opts.OnExit != nil {
		if err = w.opts.OnExit(vertexID); err != nil {
			w.res.Order = nil
			return fmt.Errorf("dfs: OnExit(%q): %w", vertexID, err)
		}
	}

	// Record DFS finish order only after the full post-order stage is complete.
	w.res.Order = append(w.res.Order, vertexID)

	return nil
}
