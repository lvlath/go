// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package bfs

import (
	"context"
	"sort"

	"github.com/lvlath/go/core"
)

const (
	// componentsAdjCapHint is a small initial capacity hint used when materializing adjacency slices.
	//
	// AI-HINTS:
	//   - This is a hint only; adjacency sizes depend on input topology.
	componentsAdjCapHint = 4
)

// runComponents computes weakly-connected components under an undirected relation.
//
// Implementation:
//   - Stage 1: Snapshot deterministic vertex IDs and edges.
//   - Stage 2: Build an undirected adjacency structure (treat every edge as connecting both endpoints).
//   - Stage 3: Traverse components in deterministic root order using a retention-safe queue.
//   - Stage 4: Sort vertices inside each component and sort components by a stable key.
//
// Behavior highlights:
//   - Weak connectivity: directed edges still connect their endpoints for component membership.
//   - Partial result: on ctx cancellation, returns accumulated components plus ctx.Err().
//
// Inputs:
//   - ctx: cancellation context; must be non-nil.
//   - g: graph instance; must be non-nil.
//
// Returns:
//   - *ComponentsResult: deterministic component listing (may be partial on ctx cancellation).
//   - error: nil on success; otherwise ctx.Err() (cancellation).
//
// Errors:
//   - context.Canceled / context.DeadlineExceeded when ctx is done.
//
// Determinism:
//   - Root selection follows g.Vertices() order.
//   - Neighbor processing is deterministic because adjacency lists are sorted lexicographically.
//
// Complexity:
//   - Time O(V log V + E log E + Σ deg(v) log deg(v) + (V+E)) due to sorting APIs and adjacency sorting.
//   - Space O(V+E) for adjacency and traversal state.
//
// Notes:
//   - Components ≠ SCC: this is weak connectivity, not strong connectivity.
//
// AI-Hints:
//   - Use this function when you need undirected connectivity even on directed graphs.
//   - Avoid mutating the graph concurrently; cancellation is supported, but topology mutation may yield partial snapshots.
func runComponents(ctx context.Context, g *core.Graph) (*ComponentsResult, error) {
	if ctx == nil {
		// Public facade currently treats nil ctx as an option violation.
		// This internal check keeps behavior safe for accidental direct calls.
		return nil, ErrOptionViolation
	}
	if g == nil {
		return nil, ErrGraphNil
	}

	// Stage 1: Deterministic snapshots.
	vertexIDs := g.Vertices()
	edges := g.Edges()

	// Stage 2: Build undirected adjacency as sets first (to deduplicate).
	adjSet := make(map[string]map[string]struct{}, len(vertexIDs))
	for _, id := range vertexIDs {
		adjSet[id] = make(map[string]struct{}, componentsAdjCapHint)
	}

	for _, e := range edges {
		u := e.From
		v := e.To

		// Ensure buckets exist even if topology changes concurrently (defensive).
		if _, ok := adjSet[u]; !ok {
			adjSet[u] = make(map[string]struct{}, componentsAdjCapHint)
		}
		if _, ok := adjSet[v]; !ok {
			adjSet[v] = make(map[string]struct{}, componentsAdjCapHint)
		}

		// Treat every edge as undirected for weak connectivity.
		adjSet[u][v] = struct{}{}
		adjSet[v][u] = struct{}{}
	}

	// Materialize sorted adjacency lists for deterministic traversal.
	adj := make(map[string][]string, len(adjSet))
	for id, set := range adjSet {
		out := make([]string, 0, len(set))
		for nbr := range set {
			out = append(out, nbr)
		}
		sort.Strings(out)
		adj[id] = out
	}

	// Stage 3: Traverse components by deterministic root order.
	visited := make(map[string]bool, len(vertexIDs))
	components := make([][]string, 0)

	// Retention-safe queue state.
	queue := make([]string, 0, len(vertexIDs))
	head := 0

	// Helper to finalize a component (sort IDs, append).
	finalizeComponent := func(comp []string) {
		sort.Strings(comp)
		components = append(components, comp)
	}

	for _, root := range vertexIDs {
		// Cancellation check before starting a new component.
		select {
		case <-ctx.Done():
			res := &ComponentsResult{
				Components:     components,
				Count:          len(components),
				UndirectedView: true,
			}
			return res, ctx.Err()
		default:
		}

		if visited[root] {
			continue
		}

		// Start a new component BFS.
		comp := make([]string, 0, componentsAdjCapHint)

		queue = queue[:0]
		head = 0

		visited[root] = true
		queue = append(queue, root)

		for head < len(queue) {
			// Cancellation check during traversal.
			select {
			case <-ctx.Done():
				// Partial component is included as partial result by contract.
				finalizeComponent(comp)
				res := &ComponentsResult{
					Components:     components,
					Count:          len(components),
					UndirectedView: true,
				}
				return res, ctx.Err()
			default:
			}

			u := queue[head]
			queue[head] = "" // release references
			head++

			comp = append(comp, u)

			for _, nbr := range adj[u] {
				// Optional fine-grained cancellation check inside neighbor scan.
				select {
				case <-ctx.Done():
					finalizeComponent(comp)
					res := &ComponentsResult{
						Components:     components,
						Count:          len(components),
						UndirectedView: true,
					}
					return res, ctx.Err()
				default:
				}

				if visited[nbr] {
					continue
				}
				visited[nbr] = true
				queue = append(queue, nbr)
			}
		}

		finalizeComponent(comp)
	}

	// Stage 4: Sort components by a stable key (first ID after per-component sort).
	sort.Slice(components, func(i, j int) bool {
		return components[i][0] < components[j][0]
	})

	return &ComponentsResult{
		Components:     components,
		Count:          len(components),
		UndirectedView: true,
	}, nil
}
