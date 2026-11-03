// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dijkstra implements settled-target A* with a consistent heuristic.
// This file prepares normalized keys and reuses the canonical forward search kernel.
package dijkstra

import "context"

// runAStar executes consistent-heuristic target search over the prepared graph.
// Its forward labels remain exact original costs; only queue priorities are shifted.
// Implementation:
//   - Stage 1: Evaluate and certify the immutable target heuristic and offsets.
//   - Stage 2: Delegate label setting and strict improvements to runTarget.
//   - Stage 3: Return private frontier state for the facade's shared reconstruction.
//
// Behavior highlights:
//   - The normalized key is g(v)+h(v)-h(source), evaluated via exact offsets.
//   - Consistency makes these keys nondecreasing along reachable original paths,
//     so the first non-stale target pop finalizes the optimum without reopening.
//
// Inputs:
//   - ctx: non-nil invocation context checked during preparation and traversal.
//   - graph: one frozen validated effective directed/mixed/undirected topology.
//   - source, target: valid dense endpoint indices.
//   - limit: validated inclusive original-cost cutoff or positive infinity.
//   - estimate: non-nil target heuristic, in the same units as original weights.
//
// Returns:
//   - *searchFrontier: target settlement plus original-cost predecessor state;
//     other labels are tentative, not a completed SSSP result.
//
// Errors:
//   - ErrNilHeuristic, ErrInvalidHeuristic or ErrInconsistentHeuristic in preprocessing.
//   - ErrSearchKeyOverflow, ErrInexactArithmetic or ErrDistanceOverflow in key/cost checks.
//   - Callback and cancellation errors with their original causes preserved.
//
// Determinism:
//   - Lexical one-time callback order; identical arc/heap ordering as runTarget.
//
// Complexity:
//   - Time O(V*H + A + V + (V+A)log(A+1)), Space O(V+A), A<=2E.
//   - Snapshot sorting and Clone are accounted for by the calling facade.
//
// Notes:
//   - No second A* relaxation engine is maintained and no heuristic is cached globally.
//   - Missing target paths retain the normal +Inf/Reachable=false publication law.
//
// AI-Hints:
//   - Do not call the callback from a heap comparator or per-relaxation loop.
//   - Do not use priority as the published distance or compare cutoff against priority.
//   - Do not allow inconsistent heuristics without a separately proven reopening mode.
func runAStar(
	ctx context.Context,
	graph *searchGraph,
	source, target int,
	limit float64,
	estimate Heuristic,
) (*searchFrontier, error) {
	offsets, err := prepareAStarOffsets(ctx, graph, source, target, estimate)
	if err != nil {
		return nil, err
	}
	return runTarget(ctx, graph, source, target, limit, offsets)
}
