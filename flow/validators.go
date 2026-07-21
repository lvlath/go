// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package flow

import (
	"errors"
	"fmt"
	"math"

	"github.com/lvlath/go/core"
)

// validateFlowGraphOnly validates graph-level capacity policy without terminals.
// It is used by matrix/diagnostic adapters that do not need source-sink vertices.
//
// Implementation:
//   - Stage 1: Reject nil graph.
//   - Stage 2: Require weighted graph because Edge.Weight carries capacity.
//   - Stage 3: Respect context cancellation before heavy adapter work.
//
// Behavior highlights:
//   - Does not validate source or sink.
//   - Preserves core.ErrBadWeight with ErrUnweightedGraph.
//
// Inputs:
//   - g: graph to adapt into capacity/residual structures.
//   - cfg: finalized runtime options.
//
// Returns:
//   - error: nil when graph-level policy is valid.
//
// Errors:
//   - ErrNilGraph.
//   - ErrUnweightedGraph joined with core.ErrBadWeight.
//   - cfg.ctx.Err() when canceled.
//
// Determinism:
//   - Validation has no traversal output.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Terminal-specific algorithms must use validateFlowInput instead.
//
// AI-Hints:
//   - Do not use this helper for MaxFlow; it intentionally omits source/sink checks.
func validateFlowGraphOnly(g *core.Graph, cfg options) error {
	if g == nil {
		return ErrNilGraph
	}
	if err := cfg.ctx.Err(); err != nil {
		return err
	}
	if !g.Weighted() {
		return errors.Join(ErrUnweightedGraph, core.ErrBadWeight)
	}

	return nil
}

// validateFlowInput validates graph and terminal preconditions.
//
// Implementation:
//   - Stage 1: Reject nil graph before any method call.
//   - Stage 2: Reject empty and identical terminal IDs.
//   - Stage 3: Validate terminal presence through core.Graph.
//   - Stage 4: Require weighted graphs because capacities are stored in Edge.Weight.
//
// Behavior highlights:
//   - Multi-layer errors preserve flow and core sentinels where applicable.
//
// Inputs:
//   - g: capacity graph.
//   - source: source vertex ID.
//   - sink: sink vertex ID.
//   - cfg: already finalized runtime options.
//
// Returns:
//   - error: nil when the graph is eligible for residual construction.
//
// Errors:
//   - ErrNilGraph, ErrEmptyTerminal, ErrSameTerminal.
//   - ErrSourceNotFound, ErrSinkNotFound.
//   - ErrUnweightedGraph joined with core.ErrBadWeight.
//
// Determinism:
//   - Validation has no traversal side effects.
//
// Complexity:
//   - Time O(1) expected for HasVertex, Space O(1).
//
// Notes:
//   - The input graph is read through core public API and is never mutated.
//
// AI-Hints:
//   - Do not call g.HasVertex before nil graph validation.
func validateFlowInput(g *core.Graph, source, sink string, cfg options) error {
	_ = cfg

	if g == nil {
		return ErrNilGraph
	}
	if source == "" || sink == "" {
		return errors.Join(ErrEmptyTerminal, core.ErrEmptyVertexID)
	}
	if source == sink {
		return ErrSameTerminal
	}
	if !g.HasVertex(source) {
		return errors.Join(ErrSourceNotFound, core.ErrVertexNotFound)
	}
	if !g.HasVertex(sink) {
		return errors.Join(ErrSinkNotFound, core.ErrVertexNotFound)
	}
	if !g.Weighted() {
		return errors.Join(ErrUnweightedGraph, core.ErrBadWeight)
	}

	return nil
}

// validateCapacity validates and normalizes one edge capacity for residual construction.
// It rejects non-finite and materially negative capacities before aggregation.
//
// Implementation:
//   - Stage 1: Reject nil edge references.
//   - Stage 2: Reject NaN and +/-Inf because residual arithmetic must be finite.
//   - Stage 3: Reject capacities below -epsilon as true negative capacities.
//   - Stage 4: Treat capacities <= epsilon as absent arcs.
//   - Stage 5: Return the positive finite capacity unchanged.
//
// Behavior highlights:
//   - Small positive capacities under epsilon are ignored, not rounded upward.
//   - Small negative capacities within epsilon are treated as numerical zero.
//   - Real negative capacities are sentinel-classified with ErrNegativeCapacity.
//
// Inputs:
//   - edge: core.Edge snapshot from core.Edges().
//   - epsilon: non-negative finite threshold already validated by applyOptions.
//
// Returns:
//   - float64: positive capacity, or 0 when the edge is absent under epsilon.
//   - error: nil when capacity can be used or ignored.
//
// Errors:
//   - ErrInvalidCapacity for nil edge or invalid numeric state.
//   - ErrNaNInf for NaN/Inf capacity.
//   - ErrNegativeCapacity for capacity < -epsilon.
//
// Determinism:
//   - Pure numeric validation; no ordering side effects.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This helper intentionally does not inspect directionality.
//   - Directional translation belongs to buildResidualNetwork.
//
// AI-Hints:
//   - Do not silently coerce NaN or Inf to zero.
//   - Do not accept negative capacities because max-flow residual proofs require
//     non-negative capacities.
func validateCapacity(edge *core.Edge, epsilon float64) (float64, error) {
	if edge == nil {
		return 0, ErrInvalidCapacity
	}

	capacity := edge.Weight
	if math.IsNaN(capacity) || math.IsInf(capacity, 0) {
		return 0, errors.Join(
			ErrInvalidCapacity,
			ErrNaNInf,
			fmt.Errorf("flow: edge %q %q->%q has capacity %v", edge.ID, edge.From, edge.To, capacity),
		)
	}
	if capacity < -epsilon {
		return 0, errors.Join(
			ErrInvalidCapacity,
			ErrNegativeCapacity,
			fmt.Errorf("flow: edge %q %q->%q has capacity %v", edge.ID, edge.From, edge.To, capacity),
		)
	}
	if capacity <= epsilon {
		return 0, nil
	}

	return capacity, nil
}
