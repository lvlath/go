// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dijkstra implements target-directed shortest paths over effective graph arcs.
// This file owns single-frontier settlement and early target termination.
package dijkstra

import (
	"container/heap"
	"context"
	"fmt"
	"math"
)

// runTarget executes the canonical forward label-setting loop until its target settles.
// Dijkstra and consistent A* share its original-cost labels and predecessor arcs.
// Implementation:
//   - Stage 1: Initialize original-cost labels and a zero-priority normalized root.
//   - Stage 2: Pop valid minimum keys, settle them and relax outgoing effective arcs.
//   - Stage 3: Stop only after the target is settled or all valid candidates are exhausted.
//
// Behavior highlights:
//   - Uses exact represented-binary prefix arithmetic with explicit refusal on inexact storage.
//   - Out-of-cutoff candidates are skipped before classifying unneeded storage overflow.
//   - Equal-cost paths never replace the first predecessor; zero cycles stay finite.
//   - With offsets, key=g(v)+offset(v) is exactly materialized before heap push.
//
// Inputs:
//   - ctx: non-nil caller context checked during expansion and before publication.
//   - graph: already validated detached effective graph with original edge records.
//   - source, target: valid dense indices in the same graph.
//   - limit: validated original-cost inclusive cutoff or +Inf.
//   - offsets: nil for Dijkstra; for A*, exact normalized h(v)-h(source) values.
//
// Returns:
//   - *searchFrontier: mutable private state, with target settled iff reachable.
//   - error: nil on completion, never a partial public PathResult.
//
// Errors:
//   - ErrDistanceOverflow for an admitted non-finite original-cost candidate.
//   - ErrInexactArithmetic for an admitted sum or A* key not storable exactly.
//   - ErrSearchKeyOverflow for an overflowing A* priority.
//   - Context cancellation/deadline errors and distinct caller causes.
//
// Determinism:
//   - Sorted original arcs, heap order (priority, distance, vertex, sequence), strict improvements.
//   - Consistency ensures nondecreasing keys, so settled labels never reopen.
//
// Complexity:
//   - Time O(V + (V+A) log(A+1)), Space O(V+A) for lazy heap entries, A<=2E.
//   - Graph cloning, edge validation and sorting occur in prepareSearchGraph, not here.
//
// Notes:
//   - An unreachable target remains +Inf under the effective cutoff/wall policies.
//   - A source==target query still follows full graph and weight validation upstream.
//
// AI-Hints:
//   - Do not stop at the first target enqueue; only a non-stale target pop finalizes.
//   - Never publish the other tentative labels as a legacy Result.
//   - Check the wall before arithmetic and preserve original edge IDs in linkArc.
//   - Only candidate original costs use MaxDistance; never prune A* priority keys.
func runTarget(
	ctx context.Context,
	graph *searchGraph,
	source, target int,
	limit float64,
	offsets []float64,
) (*searchFrontier, error) {
	frontier := newSearchFrontier(len(graph.ids), source)
	if source == target {
		frontier.settled[source] = true
		return frontier, nil
	}

	for frontier.queue.Len() > 0 {
		if err := checkSearchContext(ctx); err != nil {
			return nil, err
		}

		// A lazy heap may contain superseded candidates. Only the current
		// lowest label can settle a vertex, and a settled vertex is immutable.
		item := heap.Pop(&frontier.queue).(searchItem)
		current := item.vertex
		if frontier.settled[current] || item.distance != frontier.distance[current] {
			continue
		}
		frontier.settled[current] = true

		// Once the target's minimum candidate is settled, all shorter paths
		// have already been explored under non-negative, exact prefix costs.
		if current == target {
			break
		}

		for _, arcIndex := range graph.out[current] {
			if err := checkSearchContext(ctx); err != nil {
				return nil, err
			}

			arc := graph.arcs[arcIndex]
			if frontier.settled[arc.to] {
				continue
			}

			// The original edge record is already validated and admitted by the
			// effective-graph adapter. A numerical failure is a search error,
			// not proof that the neighbor has no reachable route.
			edge := graph.edges[arc.edge]
			candidate, allowed, err := addSearchCost(frontier.distance[current], edge.Weight, limit)
			if err != nil {
				return nil, fmt.Errorf(
					"dijkstra: target relax edge_id=%q from=%q to=%q: %w",
					edge.ID, graph.ids[arc.from], graph.ids[arc.to], err,
				)
			}
			if !allowed || candidate >= frontier.distance[arc.to] {
				continue
			}

			// A* shifts only queue ordering, never the original-cost labels.
			// The offset is frozen and exact, so a stale distance uniquely
			// identifies its corresponding stale priority as well.
			priority := candidate
			if offsets != nil {
				priority, err = certifySearchKeySum(candidate, offsets[arc.to])
				if err != nil {
					return nil, fmt.Errorf(
						"dijkstra: priority edge_id=%q vertex_id=%q distance=%g: %w",
						edge.ID, graph.ids[arc.to], candidate, err,
					)
				}
			}

			frontier.distance[arc.to] = candidate
			frontier.linkArc[arc.to] = arcIndex
			heap.Push(&frontier.queue, searchItem{
				vertex: arc.to, distance: candidate, priority: priority,
				sequence: frontier.sequence,
			})
			frontier.sequence++
		}
	}

	if err := checkSearchContext(ctx); err != nil {
		return nil, err
	}
	if !frontier.settled[target] {
		// The infinity is an ordinary search-domain absence sentinel, never
		// an overflowed candidate admitted into the frontier.
		frontier.distance[target] = math.Inf(1)
	}
	return frontier, nil
}
