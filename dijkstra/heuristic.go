// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dijkstra validates and freezes consistent directional heuristics.
// This file shares exact arc inequalities across ordinary and balanced A*.
package dijkstra

import (
	"context"
	"fmt"
	"math"
)

// searchHeuristicConsistent checks one exact target-potential inequality.
// It certifies h(from) <= weight + h(to) over the binary values represented by float64.
// Implementation:
//   - Stage 1: Form an error-free two-component sum for weight+h(to).
//   - Stage 2: Treat positive overflow as greater than every finite h(from).
//   - Stage 3: Resolve equality of rounded high parts with the residual's sign.
//
// Behavior highlights:
//   - Works for finite signed heuristics; weights are finite and non-negative.
//   - It does not insert rounded slacks into labels or relax consistency by epsilon.
//
// Inputs:
//   - from, to: frozen, finite heuristic values in original edge-cost units.
//   - weight: previously validated finite non-negative effective edge weight.
//
// Returns:
//   - bool: whether the represented-binary consistency inequality holds exactly.
//
// Errors:
//   - None; the caller reports ErrInconsistentHeuristic for a false result.
//
// Determinism:
//   - Exact high/residual comparison without traversal-dependent tolerances.
//
// Complexity:
//   - Time O(1), Space O(1) for one effective oriented arc.
//
// Notes:
//   - Negative infinity cannot arise from adding non-negative weight to finite h(to).
//
// AI-Hints:
//   - Do not replace this predicate with from <= weight+to; rounded equality can
//     accept a mathematically inconsistent edge and invalidate settlement.
func searchHeuristicConsistent(from, weight, to float64) bool {
	hi, lo, overflow := splitSearchSum(weight, to)
	return overflow || hi > from || (hi == from && lo >= 0)
}

// prepareAStarOffsets evaluates one complete, consistent target potential.
// It freezes every callback value before any priority queue is initialized.
// Implementation:
//   - Stage 1: Visit every vertex in lexical order exactly once and validate each value.
//   - Stage 2: Require h(target)=0 and certify consistency over every effective arc.
//   - Stage 3: Materialize exact normalized offsets h(v)-h(source) for all vertices.
//
// Behavior highlights:
//   - Negative finite heuristic values are legal when consistency holds.
//   - A* priorities use g(v)+offset(v), not unnormalized g(v)+h(v).
//   - Even disconnected vertices are evaluated; invalid callbacks cannot hide there.
//
// Inputs:
//   - ctx: non-nil invocation context; cancellation is checked around callbacks.
//   - graph: detached, validated effective topology with stable lexical vertex IDs.
//   - source, target: valid dense vertex indices in the same graph.
//   - estimate: non-nil caller callback; it must be safe for synchronous invocation.
//
// Returns:
//   - []float64: owned, exact normalized offsets; offset[source] is exactly zero.
//   - error: nil only after all values, inequalities and offsets are certified.
//
// Errors:
//   - ErrNilHeuristic for a nil callback.
//   - ErrInvalidHeuristic for NaN/Inf or a nonzero target estimate.
//   - ErrInconsistentHeuristic for a violated effective-arc inequality.
//   - ErrSearchKeyOverflow or ErrInexactArithmetic for unrepresentable offsets.
//   - Callback errors and cancellation causes are preserved with wrapping.
//
// Determinism:
//   - Callbacks follow graph.ids lexical order; arc checks follow graph.arcs order;
//     the first failure in that order wins.
//
// Complexity:
//   - Time O(V*H + A), Space O(V), H is the callback cost and A<=2E.
//   - The graph clone and sorted enumeration are owned by prepareSearchGraph.
//
// Notes:
//   - All offset storage is private to the invocation and is never exposed in PathResult.
//   - No re-evaluation occurs during heap operations, even for stale entries.
//
// AI-Hints:
//   - Validate against admitted arcs, not blocked graph edges or raw graph defaults.
//   - Check exact consistency before any settlement: admissible-only heuristics
//     require a distinct reopening algorithm and are not accepted by this kernel.
//   - h(target)=0 is required even for source==target and unreachable queries.
func prepareAStarOffsets(
	ctx context.Context,
	graph *searchGraph,
	source, target int,
	estimate Heuristic,
) ([]float64, error) {
	values, err := freezeSearchHeuristic(ctx, graph, estimate, "")
	if err != nil {
		return nil, err
	}
	if err := validateSearchHeuristic(ctx, graph, values, target, false); err != nil {
		return nil, err
	}

	// Normalize once at the source. The existing forward engine starts with a
	// zero key and never exposes potential-adjusted costs through PathResult.
	offsets := make([]float64, len(values))
	for vertex, value := range values {
		if err := checkSearchContext(ctx); err != nil {
			return nil, err
		}
		offset, err := certifySearchKeySum(value, -values[source])
		if err != nil {
			return nil, fmt.Errorf("dijkstra: heuristic offset vertex_id=%q source_id=%q: %w",
				graph.ids[vertex], graph.ids[source], err)
		}
		offsets[vertex] = offset
	}
	return offsets, nil
}

// freezeSearchHeuristic evaluates one callback against a stable lexical vertex domain.
// The returned values cannot change while the search queue is being processed.
// Implementation:
//   - Stage 1: Reject a missing callback before allocating the cache.
//   - Stage 2: Invoke the callback exactly once for each sorted original vertex ID.
//   - Stage 3: Reject non-finite estimates and publish the complete private cache.
//
// Behavior highlights:
//   - Disconnected vertices are evaluated; no invalid estimate is ignored.
//   - Callbacks are never called from heap comparators or during relaxation.
//
// Inputs:
//   - ctx: non-nil cancellation context checked around every callback.
//   - graph: detached, immutable effective graph sorted by original vertex ID.
//   - estimate: synchronous user-provided cost estimate; must be non-nil.
//   - role: diagnostic name; empty preserves the single-heuristic error prefix.
//
// Returns:
//   - []float64: fresh, finite, indexed callback values, ordered by graph.ids.
//   - error: nil only when all vertices were successfully evaluated.
//
// Errors:
//   - ErrNilHeuristic if estimate is nil; ErrInvalidHeuristic on NaN or infinity.
//   - Callback errors and context cancellation/cause are preserved via wrapping.
//
// Determinism:
//   - Exactly one callback per vertex in lexical order, no map iteration.
//
// Complexity:
//   - Time O(V*H), Space O(V), H being the callback cost.
//
// Notes:
//   - The caller owns the returned cache and must not mutate it during traversal.
//
// AI-Hints:
//   - Never re-evaluate a stateful callback during a later consistency check.
//   - Only effective graph arcs, not the source graph, are used by the next stage.
func freezeSearchHeuristic(
	ctx context.Context,
	graph *searchGraph,
	estimate Heuristic,
	role string,
) ([]float64, error) {
	if estimate == nil {
		return nil, ErrNilHeuristic
	}

	values := make([]float64, len(graph.ids))
	for vertex, vertexID := range graph.ids {
		if err := checkSearchContext(ctx); err != nil {
			return nil, err
		}
		value, err := estimate(vertexID)
		if err != nil {
			if role != "" {
				return nil, fmt.Errorf("dijkstra: heuristic role=%q vertex_id=%q: %w", role, vertexID, err)
			}
			return nil, fmt.Errorf("dijkstra: heuristic vertex_id=%q: %w", vertexID, err)
		}
		if err := checkSearchContext(ctx); err != nil {
			return nil, err
		}
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, fmt.Errorf("%w: role=%q vertex_id=%q value=%g", ErrInvalidHeuristic, role, vertexID, value)
		}
		values[vertex] = value
	}
	return values, nil
}

// validateSearchHeuristic proves one directional consistency law on effective arcs.
// It checks the selected zero anchor and every oriented relation before any settlement.
// Implementation:
//   - Stage 1: Require exact zero at the direction's designated terminal.
//   - Stage 2: Traverse all effective original oriented arcs in canonical order.
//   - Stage 3: Check h(u)<=w+h(v), or h(v)<=w+h(u) for reverse search.
//
// Behavior highlights:
//   - A forward and backward estimate may both be negative when consistent.
//   - Wall-filtered edges are not part of the consistency relation.
//
// Inputs:
//   - ctx: non-nil cancellation context checked during every arc inspection.
//   - graph: detached effective topology with validated non-negative weights.
//   - values: frozen finite estimates indexed identically to graph.ids.
//   - anchor: vertex index required to have heuristic value exactly zero.
//   - reverse: false for toward-target, true for from-source consistency.
//
// Returns:
//   - error: nil only when anchor and every directed constraint are valid.
//
// Errors:
//   - ErrInvalidHeuristic for an incorrect terminal anchor.
//   - ErrInconsistentHeuristic for a violated exact signed inequality.
//   - Context cancellation and a distinct context cause are preserved.
//
// Determinism:
//   - Exact anchor check followed by graph.arcs order and the first invalid edge.
//
// Complexity:
//   - Time O(A), Space O(1), with A<=2E effective directed arcs.
//
// Notes:
//   - Reverse consistency does NOT reverse the actual source graph's edge IDs.
//
// AI-Hints:
//   - For an original arc u->v, reverse checks h(v)<=weight+h(u).
//   - Do not replace exact residual comparison with a tolerance or rounded sum.
func validateSearchHeuristic(
	ctx context.Context,
	graph *searchGraph,
	values []float64,
	anchor int,
	reverse bool,
) error {
	if values[anchor] != 0 {
		return fmt.Errorf("%w: anchor_id=%q expected=0 got=%g",
			ErrInvalidHeuristic, graph.ids[anchor], values[anchor])
	}

	for _, arc := range graph.arcs {
		if err := checkSearchContext(ctx); err != nil {
			return err
		}
		from, to := arc.from, arc.to
		if reverse {
			from, to = to, from
		}
		edge := graph.edges[arc.edge]
		if !searchHeuristicConsistent(values[from], edge.Weight, values[to]) {
			return fmt.Errorf(
				"%w: edge_id=%q from=%q to=%q reversed=%t weight=%g h_from=%g h_to=%g",
				ErrInconsistentHeuristic, edge.ID, graph.ids[arc.from], graph.ids[arc.to], reverse,
				edge.Weight, values[from], values[to],
			)
		}
	}
	return nil
}
