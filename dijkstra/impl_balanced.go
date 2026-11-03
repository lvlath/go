// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dijkstra implements target-directed shortest paths with balanced potentials.
// This file prepares two certified heuristic directions for the shared two-frontier engine.
package dijkstra

import (
	"context"
	"fmt"
)

// searchBalancedPotentials contains normalized, exact key offsets for both directions.
// Its shift translates a complete original-cost incumbent into the common reduced-key scale.
// Implementation:
//   - Stage 1: Freeze two directional heuristics on the same detached graph.
//   - Stage 2: Derive p(v)=(hT(v)-hS(v))/2 and normalized offsets.
//   - Stage 3: Retain only the two offset arrays and the endpoint shift.
//
// Behavior highlights:
//   - forward[v]=p(v)-p(source), backward[v]=p(target)-p(v).
//   - shift=p(target)-p(source), shared by both frontier key scales.
//
// Inputs:
//   - forward, backward: owned exact per-vertex offsets aligned with graph.ids.
//   - shift: exact original-cost potential difference between target and source.
//
// Returns:
//   - searchBalancedPotentials: immutable private value used by bidirectional search.
//
// Errors:
//   - The type itself does not return errors; preparation certifies every field.
//
// Determinism:
//   - Vertex indices are fixed by the detached lexically sorted searchGraph.
//
// Complexity:
//   - Space O(V), with exactly two offset arrays of V floating-point values.
//
// Notes:
//   - Values are not exposed in PathResult and do not mutate source edge weights.
//
// AI-Hints:
//   - Never change sign of backward offsets: backward labels measure v->target.
//   - The stopping bound must include shift; comparing keys to original mu is unsound.
type searchBalancedPotentials struct {
	forward  []float64 // Original source-cost key shift at each vertex.
	backward []float64 // Original target-cost key shift at each vertex.
	shift    float64   // Common translation from original incumbent cost to reduced cost.
}

// prepareBalancedPotentials validates two consistent heuristics and constructs exact keys.
// Both callbacks are frozen before search begins, with no dependence on heap activity.
// Implementation:
//   - Stage 1: Evaluate all hT values, then all hS values in lexical vertex order.
//   - Stage 2: Require hT(target)=0, hS(source)=0 and both directional inequalities.
//   - Stage 3: Certify the balanced potential and exact normalized offset arrays.
//
// Behavior highlights:
//   - hT(u)<=w+hT(v) and hS(v)<=w+hS(u) for each admitted arc u->v.
//   - These inequalities imply w-p(u)+p(v)>=0, even on asymmetric graphs.
//   - No reduced weights are rounded, clamped, or copied into original core.Edge.
//
// Inputs:
//   - ctx: non-nil context, checked before/after callbacks and every arc.
//   - graph: one validated, detached searchGraph with stable vertex/arc order.
//   - source, target: valid dense indices for terminal anchors and normalization.
//   - toTarget: required finite target heuristic with a zero target anchor.
//   - fromSource: required finite source heuristic with a zero source anchor.
//
// Returns:
//   - *searchBalancedPotentials: private immutable normalized key representation.
//   - error: nil only after all callbacks, inequalities and numeric operations succeed.
//
// Errors:
//   - ErrNilHeuristic, ErrInvalidHeuristic, ErrInconsistentHeuristic from validation.
//   - ErrInexactArithmetic or ErrSearchKeyOverflow on potential/offset materialization.
//   - Callback errors and cancellation causes preserve errors.Is through wrapping.
//
// Determinism:
//   - All target callbacks first; all source callbacks second; both in lexical ID order.
//   - Inequalities use oriented original-arc order; offsets use lexical vertex order.
//
// Complexity:
//   - Time O(V*(HT+HS)+A), Space O(V), where A<=2E and H are callback costs.
//
// Notes:
//   - In the exact-only numeric model, some finite consistent potentials are rejected.
//   - A source==target query still validates both complete heuristics and all arcs.
//
// AI-Hints:
//   - Do not accept admissibility alone: settled frontiers require consistency.
//   - Do not divide an overflowing hT-hS directly; certify the half-difference.
//   - Original-cost MaxDistance and returned Distance must never include the shift.
func prepareBalancedPotentials(
	ctx context.Context,
	graph *searchGraph,
	source, target int,
	toTarget, fromSource Heuristic,
) (*searchBalancedPotentials, error) {
	// The two frozen passes share the same immutable topology but do not share
	// callback state. Diagnostic roles preserve which estimate caused a failure.
	toTargetValues, err := freezeSearchHeuristic(ctx, graph, toTarget, "to_target")
	if err != nil {
		return nil, err
	}
	fromSourceValues, err := freezeSearchHeuristic(ctx, graph, fromSource, "from_source")
	if err != nil {
		return nil, err
	}

	// A target estimate is consistent along original outgoing arcs; a source
	// estimate must be consistent along their reversed traversal relation.
	if err := validateSearchHeuristic(ctx, graph, toTargetValues, target, false); err != nil {
		return nil, err
	}
	if err := validateSearchHeuristic(ctx, graph, fromSourceValues, source, true); err != nil {
		return nil, err
	}

	// Exact half-differences are frozen separately from their offsets. Thus
	// neither search direction computes floating-point potentials in the hot loop.
	potentials := make([]float64, len(graph.ids))
	for vertex := range potentials {
		if err := checkSearchContext(ctx); err != nil {
			return nil, err
		}
		value, err := certifyBalancedPotential(toTargetValues[vertex], fromSourceValues[vertex])
		if err != nil {
			return nil, fmt.Errorf("dijkstra: balanced potential vertex_id=%q: %w", graph.ids[vertex], err)
		}
		potentials[vertex] = value
	}

	// Normalizing both roots to zero gives monotone non-negative queue keys.
	// The offset shift translates the incumbent to exactly the same cost scale:
	// keyF(v)+keyB(v) = distF(v)+distB(v)+shift.
	balanced := &searchBalancedPotentials{
		forward:  make([]float64, len(potentials)),
		backward: make([]float64, len(potentials)),
	}
	balanced.shift, err = certifySearchKeySum(potentials[target], -potentials[source])
	if err != nil {
		return nil, fmt.Errorf("dijkstra: balanced endpoint offset source_id=%q target_id=%q: %w",
			graph.ids[source], graph.ids[target], err)
	}
	for vertex, potential := range potentials {
		if err := checkSearchContext(ctx); err != nil {
			return nil, err
		}
		balanced.forward[vertex], err = certifySearchKeySum(potential, -potentials[source])
		if err != nil {
			return nil, fmt.Errorf("dijkstra: forward offset vertex_id=%q: %w", graph.ids[vertex], err)
		}
		balanced.backward[vertex], err = certifySearchKeySum(potentials[target], -potential)
		if err != nil {
			return nil, fmt.Errorf("dijkstra: backward offset vertex_id=%q: %w", graph.ids[vertex], err)
		}
	}
	if err := checkSearchContext(ctx); err != nil {
		return nil, err
	}
	return balanced, nil
}

// runBalanced runs certified balanced bidirectional A* on the existing two-frontier engine.
// It neither maintains a second relaxation loop nor translates the published distance.
// Implementation:
//   - Stage 1: Freeze, validate and normalize both directional potentials.
//   - Stage 2: Delegate two-heap expansion and exact stopping to runBidirectional.
//   - Stage 3: Return original oriented arcs and original-cost distance for publication.
//
// Behavior highlights:
//   - Crossing certificates are identical to bidirectional Dijkstra's original IDs.
//   - Only queue priorities and the stopping comparison use balanced potentials.
//
// Inputs:
//   - ctx: non-nil cancellation context covering preparation and search.
//   - graph: detached effective graph from the canonical Search facade.
//   - source, target: valid dense indices for requested original vertices.
//   - limit: finite inclusive original-cost cutoff or +Inf (never a reduced key).
//   - toTarget, fromSource: non-nil consistent callbacks in original cost units.
//
// Returns:
//   - []int: original oriented route arc indices, nil when unreachable or self-path.
//   - float64: original path cost or +Inf when the target is known unreachable.
//   - error: nil only for complete, publishable search outcomes.
//
// Errors:
//   - All heuristic, key, original-cost and context errors from delegated stages.
//
// Determinism:
//   - Lexical callbacks and original arcs; queue tie-breaks are inherited unchanged.
//
// Complexity:
//   - Time O(V*(HT+HS)+A+(V+A)log(A+1)), Space O(V+A), A<=2E.
//
// Notes:
//   - The graph clone/sorted enumeration are already accounted for by Search.
//
// AI-Hints:
//   - Never return a reduced distance as PathResult.Distance.
//   - Do not add an alternate A* crossing engine: shared bidirectional logic owns it.
func runBalanced(
	ctx context.Context,
	graph *searchGraph,
	source, target int,
	limit float64,
	toTarget, fromSource Heuristic,
) ([]int, float64, error) {
	potentials, err := prepareBalancedPotentials(ctx, graph, source, target, toTarget, fromSource)
	if err != nil {
		return nil, 0, err
	}
	return runBidirectional(ctx, graph, source, target, limit, potentials)
}
