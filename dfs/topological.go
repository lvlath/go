// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Topological sorting implementation and runtime state for directed core.Graph traversal.
package dfs

import (
	"context"
	"fmt"

	"github.com/lvlath/go/core"
)

// TopoOption configures TopologicalSort behavior before execution starts.
//
// Implementation:
//   - Stage 1: Each option mutates topoOptions during configuration.
//   - Stage 2: Each option validates its own input.
//   - Stage 3: TopologicalSort applies all options before graph traversal begins.
//
// Behavior highlights:
//   - Options follow fail-fast validation.
//   - Invalid option input returns ErrOptionViolation.
//
// Inputs:
//   - *topoOptions: mutable topological-sort configuration.
//
// Returns:
//   - error: nil on success, or ErrOptionViolation-wrapped failure on invalid input.
//
// Errors:
//   - ErrOptionViolation: returned when an option receives invalid explicit input.
//
// Determinism:
//   - Option application is deterministic in call order.
//
// Complexity:
//   - Time O(1) per option, Space O(1).
//
// Notes:
//   - Options configure traversal behavior only; they do not store runtime sort state.
//
// AI-Hints:
//   - Keep option validation local to the option.
//   - Treat explicitly provided nil context as invalid input.
type TopoOption func(*topoOptions) error

// topoOptions stores internal TopologicalSort configuration.
//
// Implementation:
//   - Stage 1: defaultTopoOptions provides the canonical baseline.
//   - Stage 2: Option builders selectively override supported fields.
//
// Behavior highlights:
//   - The current configuration surface contains only cancellation context.
//
// Inputs:
//   - N/A.
//
// Returns:
//   - N/A.
//
// Errors:
//   - Validation errors are produced by TopoOption builders.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Accessing fields is O(1).
//
// Notes:
//   - The zero-value is not the canonical configuration; use defaultTopoOptions.
//
// AI-Hints:
//   - Keep this struct as pure configuration.
//   - Do not store visitation state or output slices here.
type topoOptions struct {
	// ctx controls cancellation and timeout behavior for topological traversal.
	ctx context.Context
}

// defaultTopoOptions returns the canonical baseline configuration for topological sort.
//
// Implementation:
//   - Stage 1: Install the default background context.
//
// Behavior highlights:
//   - The returned configuration is ready for further TopoOption-based customization.
//
// Inputs:
//   - N/A.
//
// Returns:
//   - topoOptions: canonical baseline configuration.
//
// Errors:
//   - N/A.
//
// Determinism:
//   - Deterministic by construction.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Callers should not rely on the topoOptions zero-value directly.
//
// AI-Hints:
//   - Use this helper as the single baseline before applying topological options.
func defaultTopoOptions() topoOptions {
	return topoOptions{
		ctx: context.Background(),
	}
}

// WithCancelContext sets the traversal context for TopologicalSort.
//
// Implementation:
//   - Stage 1: Validate the provided context.
//   - Stage 2: Store it in topoOptions for later cancellation checks.
//
// Behavior highlights:
//   - A nil context is invalid explicit input.
//
// Inputs:
//   - ctx: traversal context used for cancellation and timeout.
//
// Returns:
//   - TopoOption: an option that stores the validated context.
//
// Errors:
//   - ErrOptionViolation: if ctx is nil.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Context cancellation semantics remain provided by the standard library.
//
// AI-Hints:
//   - Do not silently accept nil context when it was explicitly provided.
//   - Preserve context.Canceled and context.DeadlineExceeded as underlying causes.
func WithCancelContext(ctx context.Context) TopoOption {
	return func(options *topoOptions) error {
		if ctx == nil {
			return ErrOptionViolation
		}

		options.ctx = ctx
		return nil
	}
}

// buildTopoOptions assembles the canonical TopologicalSort configuration before traversal starts.
//
// Implementation:
//   - Stage 1: Start from defaultTopoOptions as the single baseline.
//   - Stage 2: Apply options in the exact order provided by the caller.
//   - Stage 3: Abort immediately on the first invalid or failing option.
//
// Behavior highlights:
//   - Configuration is fail-fast.
//   - A nil TopoOption value is invalid explicit input.
//
// Inputs:
//   - options: ordered topological option builders.
//
// Returns:
//   - topoOptions: assembled TopologicalSort configuration.
//   - error: nil on success, or an option-construction failure.
//
// Errors:
//   - ErrOptionViolation: if an option is nil or invalid.
//   - Any custom TopoOption error is returned unchanged so callers may preserve its category.
//
// Determinism:
//   - Deterministic in caller-provided option order.
//
// Complexity:
//   - Time O(k), Space O(1), where k = len(options).
//
// Notes:
//   - This helper centralizes option application for TopologicalSort.
//
// AI-Hints:
//   - Use this helper as the single entry point for topological option assembly.
//   - Treat invalid explicit option input as configuration failure, not runtime traversal failure.
func buildTopoOptions(options ...TopoOption) (topoOptions, error) {
	// Start from the canonical baseline configuration.
	config := defaultTopoOptions()

	// Apply options in caller order so later overrides remain explicit.
	for index, option := range options {
		// Reject a nil option value explicitly to avoid a panic on call.
		if option == nil {
			return topoOptions{}, fmt.Errorf("%w: topological option at index %d is nil", ErrOptionViolation, index)
		}

		// Apply the option and stop immediately if it rejects the input.
		if err := option(&config); err != nil {
			return topoOptions{}, err
		}
	}

	return config, nil
}

// topoSorter owns runtime-only state during a single topological-sort execution.
//
// Implementation:
//   - Stage 1: Hold immutable graph and configuration references.
//   - Stage 2: Track DFS visitation colors.
//   - Stage 3: Record DFS post-order for final reversal into topological order.
//
// Behavior highlights:
//   - The struct is internal and per-execution.
//   - Runtime state is not stored in configuration objects.
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
//   - Deterministic under deterministic graph root order and neighbor order.
//
// Complexity:
//   - Space O(V), where V = vertex count.
//
// Notes:
//   - The order slice stores post-order until the final reversal step.
//
// AI-Hints:
//   - Keep runtime traversal state here, not inside topoOptions.
//   - Gray state is the active recursion-path state used for cycle detection.
type topoSorter struct {
	// graph is the directed graph being ordered.
	graph *core.Graph

	// opts is the finalized configuration for this execution.
	opts topoOptions

	// state stores DFS visitation color for each vertex.
	state map[string]VertexState

	// order stores DFS post-order before final reversal.
	order []string
}

// runTopologicalSort computes a deterministic topological ordering of a directed graph.
//
// Implementation:
//   - Stage 1: Validate graph-level preconditions.
//   - Stage 2: Assemble and validate topological options.
//   - Stage 3: Allocate deterministic traversal storage.
//   - Stage 4: Launch DFS from each unvisited vertex in graph vertex order.
//   - Stage 5: Reverse DFS post-order into final topological order.
//
// Behavior highlights:
//   - Only globally directed graphs are accepted.
//   - Cycle detection is performed through DFS coloring.
//   - In mixed-edge contexts, undirected edges are ignored; only directed outgoing edges participate.
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
//   - Mixed-edge handling here ignores undirected edges and traverses only directed outgoing edges.
func runTopologicalSort(g *core.Graph, options ...TopoOption) ([]string, error) {
	// Reject a nil graph explicitly so topological sort follows package-wide nil-input policy.
	if g == nil {
		return nil, ErrGraphNil
	}

	// Topological ordering is defined only for globally directed graphs in this package contract.
	if !g.Directed() {
		return nil, ErrGraphNotDirected
	}

	// Assemble the canonical configuration before any traversal state is allocated.
	opts, err := buildTopoOptions(options...)
	if err != nil {
		return nil, err
	}

	// Capture deterministic vertex order once for stable DFS-root traversal.
	vertices := g.Vertices()

	// Use the graph's reliable vertex count for preallocation.
	vertexCount := g.VertexCount()

	// Build the per-execution runtime sorter.
	sorter := &topoSorter{
		graph: g,
		opts:  opts,
		state: make(map[string]VertexState, vertexCount),
		order: make([]string, 0, vertexCount),
	}

	// Launch DFS from each unvisited root in deterministic graph vertex order.
	for _, vertexID := range vertices {
		if sorter.state[vertexID] != White {
			continue
		}

		if err = sorter.visit(vertexID); err != nil {
			return nil, err
		}
	}

	// Reverse DFS post-order in place to obtain the final topological order.
	for left, right := 0, len(sorter.order)-1; left < right; left, right = left+1, right-1 {
		sorter.order[left], sorter.order[right] = sorter.order[right], sorter.order[left]
	}

	return sorter.order, nil
}

// visit performs DFS from vertexID for topological sorting and cycle detection.
//
// Implementation:
//   - Stage 1: Observe cancellation.
//   - Stage 2: Reject Gray back-edges as directed cycles.
//   - Stage 3: Skip already-completed Black vertices.
//   - Stage 4: Mark the current vertex Gray.
//   - Stage 5: Enumerate neighbors exactly once.
//   - Stage 6: Traverse only directed outgoing edges.
//   - Stage 7: Mark the vertex Black and append it to DFS post-order.
//
// Behavior highlights:
//   - Gray state detects active-path cycles.
//   - Undirected edges are ignored in mixed-edge contexts.
//   - Traversal uses the package-wide neighborFromEdge helper for per-edge semantics.
//
// Inputs:
//   - vertexID: current vertex being processed.
//
// Returns:
//   - error: nil on success, or a traversal failure.
//
// Errors:
//   - ErrCycleDetected: if a Gray back-edge is found.
//   - ErrNeighborFetch: if neighbor enumeration fails.
//   - context.Canceled / context.DeadlineExceeded: if traversal context is canceled.
//
// Determinism:
//   - Deterministic under deterministic neighbor order.
//
// Complexity:
//   - Local overhead is O(out-degree(vertexID)).
//   - Overall topological sort remains O(V + E).
//
// Notes:
//   - The order slice stores post-order and is reversed only once after traversal completes.
//
// AI-Hints:
//   - Use Gray to detect cycles in the active recursion path.
//   - Ignore undirected edges here by policy; do not reinterpret them as directed work.
//   - Use neighborFromEdge even in directed-only algorithms to keep traversal semantics centralized.
func (t *topoSorter) visit(vertexID string) error {
	// Respect cancellation before performing new work for this vertex.
	select {
	case <-t.opts.ctx.Done():
		return t.opts.ctx.Err()
	default:
	}

	// A Gray vertex indicates a back-edge into the active recursion path, which is a directed cycle.
	if t.state[vertexID] == Gray {
		return ErrCycleDetected
	}

	// A Black vertex has already been fully processed and contributes no new work.
	if t.state[vertexID] == Black {
		return nil
	}

	// Mark the vertex as active in the current DFS path.
	t.state[vertexID] = Gray

	// Read graph-provided neighbors exactly once for stable local traversal logic.
	neighbors, err := t.graph.Neighbors(vertexID)
	if err != nil {
		return fmt.Errorf("%w: neighbors(%q): %w", ErrNeighborFetch, vertexID, err)
	}

	// Traverse only directed outgoing edges in graph-provided neighbor order.
	for _, edge := range neighbors {
		// Ignore undirected edges by topological-sort policy.
		if edge == nil || !edge.Directed {
			continue
		}

		// Resolve the outgoing directed neighbor from the current vertex.
		neighborID, ok := neighborFromEdge(edge, vertexID)
		if !ok {
			continue
		}

		// Recurse into the directed neighbor.
		if err = t.visit(neighborID); err != nil {
			return err
		}
	}

	// Mark the vertex fully explored after all directed outgoing work is complete.
	t.state[vertexID] = Black

	// Record post-order; final reversal is performed once after the full traversal finishes.
	t.order = append(t.order, vertexID)

	return nil
}
