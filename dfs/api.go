// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Role: Thin, deterministic public facade for DFS-based algorithms.
// Policy:
//   - No algorithmic loops or traversal kernels live here.
//   - Public functions delegate to canonical internal implementations.
//   - Convenience wrappers may improve discoverability, but must not silently change
//     algorithmic semantics beyond their explicitly documented policy.
//
// AI-Hints:
//   - Use Forest when the intent is component coverage rather than single-root traversal.
//   - Use HasCycle when only cyclicity matters and witness-cycle details are not needed.
//   - TopologicalSort is valid only for directed graphs.
//   - TopologicalSortContext is a convenience wrapper over WithCancelContext(ctx), not a distinct algorithm.

package dfs

import (
	"context"

	"github.com/lvlath/go/core"
)

// DFS performs deterministic depth-first traversal over g from startID.
//
// Implementation:
//   - Stage 1: Validate the public call shape by delegating to the canonical DFS kernel.
//   - Stage 2: Return the kernel result unchanged so all traversal semantics remain centralized.
//
// Behavior highlights:
//   - Order is DFS finish order (post-order), not discovery order.
//   - FullTraversal remains opt-in through WithFullTraversal.
//   - Mixed-edge traversal semantics are interpreted per edge by the kernel.
//
// Inputs:
//   - g: graph to traverse.
//   - startID: starting vertex ID for single-source traversal.
//   - opts: ordered DFS option builders; last-writer-wins by construction order.
//
// Returns:
//   - *Result: traversal result, including visited flags, parent links, depths, finish order,
//     and diagnostics.
//   - error: nil on success, or a traversal/configuration failure.
//
// Errors:
//   - ErrGraphNil: if g is nil.
//   - ErrStartVertexNotFound: if startID does not exist in single-source mode.
//   - ErrOptionViolation: if option assembly rejects explicit input.
//   - ErrNeighborFetch: if graph neighbor enumeration fails.
//   - context.Canceled / context.DeadlineExceeded: if the traversal context is canceled.
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
//   - Order is cleared when traversal aborts because partial finish order is not exposed as
//     authoritative output.
//
// AI-Hints:
//   - Order is post-order finish order, not shortest-path layering.
//   - Prefer Forest when the goal is to cover every disconnected component.
//   - Do not assume undirected or mixed-edge traversal can be reduced to edge.To semantics.
func DFS(g *core.Graph, startID string, opts ...Option) (*Result, error) {
	return runDFS(g, startID, opts...)
}

// Forest performs deterministic DFS-forest traversal over all graph components.
//
// Implementation:
//   - Stage 1: Prepend WithFullTraversal() to the caller-provided options.
//   - Stage 2: Delegate to the canonical DFS kernel without introducing new traversal logic.
//
// Behavior highlights:
//   - The traversal covers every unvisited component root in graph vertex order.
//   - Depth values reset at each DFS-tree root in the forest.
//
// Inputs:
//   - g: graph to traverse.
//   - opts: DFS option builders applied after enabling full traversal.
//
// Returns:
//   - *Result: DFS-forest traversal result.
//   - error: nil on success, or a traversal/configuration failure.
//
// Errors:
//   - Same as DFS.
//
// Determinism:
//   - Root order follows g.Vertices() through the canonical DFS kernel.
//
// Complexity:
//   - Same as DFS, plus O(len(opts)) for composed option assembly.
//
// Notes:
//   - startID is intentionally omitted because forest traversal is root-driven by graph order.
//
// AI-Hints:
//   - Prefer Forest when the intent is full component coverage rather than single-root reachability.
//   - In forest mode, Depth is measured from each DFS-tree root, not from a single global origin.
func Forest(g *core.Graph, opts ...Option) (*Result, error) {
	fullTraversalOptions := make([]Option, 0, len(opts)+1)
	fullTraversalOptions = append(fullTraversalOptions, WithFullTraversal())
	fullTraversalOptions = append(fullTraversalOptions, opts...)

	return runDFS(g, "", fullTraversalOptions...)
}

// DetectCycles reports graph cyclicity and returns a deterministic witness set of canonical cycles.
//
// Implementation:
//   - Stage 1: Delegate to the canonical cycle-detection kernel.
//   - Stage 2: Return the kernel result unchanged so witness-set semantics remain centralized.
//
// Behavior highlights:
//   - The result contains a deterministic witness set, not an exhaustive set of all simple cycles.
//   - Directed-cycle canonicalization preserves orientation.
//   - Undirected-cycle canonicalization may consider reversed orientation equivalent.
//
// Inputs:
//   - g: graph to inspect for cycles.
//
// Returns:
//   - *CycleDetectionResult: summary cyclicity flag plus canonical witness cycles.
//   - error: nil on success, or a graph/traversal failure.
//
// Errors:
//   - ErrGraphNil: if g is nil.
//   - ErrNeighborFetch: if graph neighbor enumeration fails.
//
// Determinism:
//   - Root order follows g.Vertices().
//   - Neighbor order follows g.Neighbors(id).
//   - Final witness-cycle order is deterministic by canonical signature.
//
// Complexity:
//   - Time O(V + E + W·L), where V = vertex count, E = edge count, W = witness-cycle count,
//     and L = average witness-cycle length used during reconstruction and canonicalization.
//   - Space O(V + Lmax + W·L), where Lmax is maximum active DFS path length.
//
// Notes:
//   - The returned cycles are closed sequences of the form [v0, v1, ..., v0].
//   - Absence of a specific cycle in the result does not prove exhaustive mathematical absence
//     under a different enumeration strategy.
//
// AI-Hints:
//   - This function reports witness cycles, not an exhaustive set of all simple cycles.
//   - Use HasCycle when only the summary boolean matters.
//   - Directed cycle canonicalization must preserve edge orientation.
func DetectCycles(g *core.Graph) (*CycleDetectionResult, error) {
	return runDetectCycles(g)
}

// HasCycle reports whether g contains at least one cycle witness.
//
// Implementation:
//   - Stage 1: Delegate to the canonical cycle-detection kernel.
//   - Stage 2: Return only the summary cyclicity flag from the result.
//
// Behavior highlights:
//   - The current implementation reuses the full witness-cycle detection path.
//   - The wrapper improves call-site clarity when only cyclicity matters.
//
// Inputs:
//   - g: graph to inspect for cycles.
//
// Returns:
//   - bool: true if at least one cycle witness is found.
//   - error: nil on success, or a cycle-detection failure.
//
// Errors:
//   - Same as DetectCycles.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Currently the same asymptotic cost as DetectCycles because it reuses the full witness-detection path.
//
// Notes:
//   - This is semantic sugar for callers interested only in cyclicity.
//
// AI-Hints:
//   - Use HasCycle for intent clarity.
//   - Do not assume this wrapper is cheaper than DetectCycles unless the kernel is later specialized.
func HasCycle(g *core.Graph) (bool, error) {
	result, err := runDetectCycles(g)
	if err != nil {
		return false, err
	}

	return result.HasCycle, nil
}

// TopologicalSort computes a deterministic topological ordering of a directed graph.
//
// Implementation:
//   - Stage 1: Delegate to the canonical topological-sort kernel.
//   - Stage 2: Return the kernel result unchanged so directed-graph policy remains centralized.
//
// Behavior highlights:
//   - Only globally directed graphs are accepted.
//   - Cycle detection is performed through DFS coloring.
//   - In mixed-edge contexts, undirected edges are ignored once the graph itself is accepted as directed.
//
// Inputs:
//   - g: graph to topologically sort.
//   - options: ordered topological option builders.
//
// Returns:
//   - []string: deterministic topological vertex order.
//   - error: nil on success, or a graph/configuration/traversal failure.
//
// Errors:
//   - ErrGraphNil: if g is nil.
//   - ErrGraphNotDirected: if g is not globally directed.
//   - ErrOptionViolation: if option assembly rejects explicit input.
//   - ErrCycleDetected: if a directed cycle is discovered.
//   - ErrNeighborFetch: if graph neighbor enumeration fails.
//   - context.Canceled / context.DeadlineExceeded: if traversal context is canceled.
//
// Determinism:
//   - Root order follows g.Vertices().
//   - Neighbor order follows g.Neighbors(id) after filtering to directed outgoing edges.
//   - Final output is deterministic under deterministic graph iteration order.
//
// Complexity:
//   - Time O(V + E), where V = vertex count and E = directed edge count examined by traversal.
//   - Space O(V) for visitation state, recursion stack, and post-order storage.
//
// Notes:
//   - Undirected edges are ignored rather than treated as an error once the graph itself is accepted
//     as directed by graph-level policy.
//
// AI-Hints:
//   - This algorithm is only valid for directed graphs.
//   - Never test the non-directed path via string matching; use errors.Is with ErrGraphNotDirected.
//   - Preserve wrapped causes for neighbor-fetch failures.
func TopologicalSort(g *core.Graph, options ...TopoOption) ([]string, error) {
	return runTopologicalSort(g, options...)
}

// TopologicalSortContext computes a deterministic topological ordering with explicit cancellation context.
//
// Implementation:
//   - Stage 1: Wrap ctx through WithCancelContext.
//   - Stage 2: Delegate to the canonical topological-sort kernel.
//
// Behavior highlights:
//   - The wrapper does not introduce new traversal semantics.
//   - It is a convenience entry point for the common single-context use case.
//
// Inputs:
//   - ctx: traversal context used for cancellation and timeout.
//   - g: graph to topologically sort.
//
// Returns:
//   - []string: deterministic topological vertex order.
//   - error: nil on success, or a graph/configuration/traversal failure.
//
// Errors:
//   - Same as TopologicalSort.
//   - ErrOptionViolation: if ctx is nil through WithCancelContext.
//
// Determinism:
//   - Same as TopologicalSort, assuming deterministic graph iteration order.
//
// Complexity:
//   - Same as TopologicalSort, plus O(1) wrapper composition cost.
//
// Notes:
//   - This wrapper is equivalent to calling TopologicalSort(g, WithCancelContext(ctx)).
//
// AI-Hints:
//   - Prefer this wrapper when a single context is all you need.
//   - Pass a non-nil context explicitly; nil remains invalid explicit input.
func TopologicalSortContext(ctx context.Context, g *core.Graph) ([]string, error) {
	return runTopologicalSort(g, WithCancelContext(ctx))
}
