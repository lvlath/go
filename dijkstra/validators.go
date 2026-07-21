// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package dijkstra

import (
	"fmt"
	"math"

	"github.com/lvlath/go/core"
)

// validateInputs validates the canonical kernel entry inputs before any traversal
// state is allocated or any shortest-path work begins.
// The function centralizes the mandatory graph and source checks required by the
// package contract.
//
// Implementation:
//   - Stage 1: Validate that the graph pointer is non-nil.
//   - Stage 2: Validate that the source vertex identifier is non-empty.
//   - Stage 3: Validate that the graph supports weights.
//   - Stage 4: Validate that the source vertex exists in the graph.
//
// Behavior highlights:
//   - The function performs fail-fast contract checks.
//   - No traversal state, heap state, or result maps are allocated here.
//
// Inputs:
//   - g: the graph that will be traversed by the kernel.
//   - sourceID: the source vertex identifier for the shortest-path run.
//
// Returns:
//   - error: nil when all kernel-entry input checks succeed.
//
// Errors:
//   - ErrNilGraph if g is nil.
//   - ErrEmptySourceID if sourceID is empty.
//   - ErrUnweightedGraph if the graph is not weighted.
//   - ErrSourceNotFound if the source vertex is absent from the graph.
//
// Determinism:
//   - Deterministic for the same graph pointer and source identifier.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This validator intentionally repeats facade-level checks to harden the kernel contract.
//   - The function does not inspect edge weights; numeric validation belongs to validateEdgeWeights.
//
// AI-Hints:
//   - Keep kernel validation explicit even if upper API layers already validate the same fields.
//   - Do not silently auto-create a missing source vertex; shortest-path kernels must not mutate graph topology.
func validateInputs(g *core.Graph, sourceID string) error {
	if g == nil {
		return ErrNilGraph
	}
	if sourceID == "" {
		return ErrEmptySourceID
	}
	if !g.Weighted() {
		return ErrUnweightedGraph
	}
	if !g.HasVertex(sourceID) {
		return ErrSourceNotFound
	}

	return nil
}

// classifyWeight validates one graph-edge weight under the package numeric policy.
// The classifier is the canonical source of truth for separating invalid
// non-finite input from mathematically unsupported finite negative input.
//
// Implementation:
//   - Stage 1: Reject NaN and both IEEE-754 infinities.
//   - Stage 2: Reject finite negative weights.
//   - Stage 3: Accept finite non-negative weights, including zero.
//
// Behavior highlights:
//   - Graph-edge weights must always remain finite.
//   - Positive infinity is not an edge-level wall representation.
//   - ErrInvalidWeight and ErrNegativeWeight remain separate protocol classes.
//
// Inputs:
//   - weight: the raw graph-edge weight to classify.
//
// Returns:
//   - error: nil when weight is finite and non-negative.
//
// Errors:
//   - ErrInvalidWeight if weight is NaN, +Inf, or -Inf.
//   - ErrNegativeWeight if weight is finite and less than zero.
//
// Determinism:
//   - Classification is deterministic for the same float64 bit pattern.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This function validates the graph-edge input domain only.
//   - It does not apply MaxDistance or InfEdgeThreshold.
//   - Positive infinity remains valid in separate result and option domains:
//     as an unreachable distance or as an unbounded policy value.
//
// AI-Hints:
//   - Do not reintroduce +Inf as a valid edge weight.
//   - InfEdgeThreshold applies to finite edges; it does not authorize non-finite graph data.
//   - Keep ErrInvalidWeight distinct from ErrNegativeWeight so callers can classify
//     numeric corruption separately from unsupported negative-cost mathematics.
func classifyWeight(weight float64) error {
	switch {
	case math.IsNaN(weight), math.IsInf(weight, 0):
		return ErrInvalidWeight
	case weight < 0:
		return ErrNegativeWeight
	default:
		return nil
	}
}

// validateEdgeWeights performs a deterministic defensive pre-scan of the graph
// edge catalog before traversal state is allocated.
// The validator rejects every edge that falls outside Dijkstra's finite,
// non-negative input domain.
//
// Implementation:
//   - Stage 1: Materialize the deterministic edge catalog through g.Edges.
//   - Stage 2: Classify every edge weight through classifyWeight.
//   - Stage 3: Stop at the first invalid edge.
//   - Stage 4: Preserve the sentinel with %w and attach complete edge diagnostics.
//
// Behavior highlights:
//   - Edge enumeration follows core.Edges order, which is sorted by Edge.ID.
//   - Validation fails before distance maps, predecessor maps, visited state,
//     or heap storage are allocated.
//   - The pre-scan and the runtime guard in relax intentionally enforce the
//     same numeric law at two different execution boundaries.
//
// Inputs:
//   - g: a non-nil weighted graph that has already passed validateInputs.
//
// Returns:
//   - error: nil when every edge weight is finite and non-negative.
//
// Errors:
//   - ErrInvalidWeight if an edge contains NaN, +Inf, or -Inf.
//   - ErrNegativeWeight if an edge contains a finite negative weight.
//   - Wrapped errors preserve the original sentinel and identify the exact edge.
//
// Determinism:
//   - The first reported invalid edge is stable because core.Edges returns
//     the edge catalog in deterministic Edge.ID order.
//
// Complexity:
//   - Effective time O(E log E) because core.Edges materializes and sorts E edges.
//   - The classification pass after materialization is O(E).
//   - Effective auxiliary space O(E) for the edge snapshot returned by core.
//
// Notes:
//   - MaxDistance and InfEdgeThreshold are traversal policies and are deliberately
//     absent from this validator.
//   - core.AddEdge normally prevents non-finite weights from being published,
//     but this defensive boundary remains necessary because caller-visible edge
//     objects can be corrupted after publication.
//   - Runtime reclassification in relax must remain in place because dijkstra
//     reads graph state progressively rather than through an immutable snapshot.
//
// AI-Hints:
//   - Do not add Options back to this signature; edge validity is policy-independent.
//   - Do not remove runtime classification from relax as “duplicate validation”.
//   - Preserve sentinels with %w when adding edge diagnostics.
func validateEdgeWeights(g *core.Graph) error {
	edges := g.Edges()

	for _, edge := range edges {
		if err := classifyWeight(edge.Weight); err != nil {
			return fmt.Errorf(
				"%w: edge_id=%q from=%q to=%q directed=%t weight=%g",
				err,
				edge.ID,
				edge.From,
				edge.To,
				edge.Directed,
				edge.Weight,
			)
		}
	}

	return nil
}

// validateMaxDistance validates one MaxDistance value under the option-domain
// numeric contract.
// The helper is shared by the public option constructor and canonical option
// assembly so custom Option values cannot bypass the same policy.
//
// Implementation:
//   - Stage 1: Reject NaN and negative infinity.
//   - Stage 2: Reject finite negative values.
//   - Stage 3: Accept finite non-negative values and positive infinity.
//
// Behavior highlights:
//   - Positive infinity means “no accumulated-distance cutoff”.
//   - Zero is valid and preserves zero-cost reachability.
//
// Inputs:
//   - distance: the candidate inclusive maximum path distance.
//
// Returns:
//   - error: nil when max belongs to the valid option domain.
//
// Errors:
//   - ErrBadMaxDistance if max is NaN, -Inf, or a finite negative value.
//
// Determinism:
//   - Deterministic for the same float64 bit pattern.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Positive infinity is valid only in the option domain.
//   - This helper does not validate graph-edge weights.
//
// AI-Hints:
//   - Keep public constructors and final option assembly on this same validator.
//   - Do not authorize +Inf graph-edge weights through this option-domain rule.
func validateMaxDistance(distance float64) error {
	switch {
	case math.IsNaN(distance):
		return ErrBadMaxDistance
	case math.IsInf(distance, -1):
		return ErrBadMaxDistance
	case distance < 0:
		return ErrBadMaxDistance
	default:
		return nil
	}
}

// validateInfEdgeThreshold validates one InfEdgeThreshold value under the
// option-domain numeric contract.
// The helper prevents built-in and custom options from drifting into different
// wall-threshold semantics.
//
// Implementation:
//   - Stage 1: Reject NaN and negative infinity.
//   - Stage 2: Reject zero and finite negative values.
//   - Stage 3: Accept finite strictly positive values and positive infinity.
//
// Behavior highlights:
//   - The wall comparison is inclusive: weight >= threshold is blocked.
//   - Positive infinity means that no valid finite edge is blocked.
//
// Inputs:
//   - threshold: the candidate inclusive finite-edge wall threshold.
//
// Returns:
//   - error: nil when threshold belongs to the valid option domain.
//
// Errors:
//   - ErrBadInfEdgeThreshold if threshold is NaN, -Inf, zero,
//     or a finite negative value.
//
// Determinism:
//   - Deterministic for the same float64 bit pattern.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Positive infinity is a policy-domain sentinel.
//   - It does not permit non-finite graph-edge weights.
//
// AI-Hints:
//   - Keep threshold validation independent from edge-weight validation.
//   - Do not silently normalize zero into another threshold.
func validateInfEdgeThreshold(threshold float64) error {
	switch {
	case math.IsNaN(threshold):
		return ErrBadInfEdgeThreshold
	case math.IsInf(threshold, -1):
		return ErrBadInfEdgeThreshold
	case threshold <= 0:
		return ErrBadInfEdgeThreshold
	default:
		return nil
	}
}

// validateOptions validates one fully assembled Options value.
// The function is the final invariant barrier after every functional option,
// including caller-defined custom Option implementations.
//
// Implementation:
//   - Stage 1: Validate MaxDistance.
//   - Stage 2: Validate InfEdgeThreshold.
//   - Stage 3: Accept the remaining boolean policy state.
//
// Behavior highlights:
//   - Custom options cannot bypass the package numeric contract.
//   - The first invalid finalized field determines the returned sentinel.
//
// Inputs:
//   - config: the current assembled runtime policy.
//
// Returns:
//   - error: nil when the complete policy is valid.
//
// Errors:
//   - ErrBadMaxDistance if MaxDistance is invalid.
//   - ErrBadInfEdgeThreshold if InfEdgeThreshold is invalid.
//
// Determinism:
//   - Field-validation order is fixed: MaxDistance, then InfEdgeThreshold.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - TrackPaths is boolean and requires no additional domain validation.
//
// AI-Hints:
//   - Do not rely only on built-in WithXxx constructors for validation.
//   - Option is publicly constructible, so canonical assembly must defend itself.
func validateOptions(config Options) error {
	if err := validateMaxDistance(config.MaxDistance); err != nil {
		return err
	}
	if err := validateInfEdgeThreshold(config.InfEdgeThreshold); err != nil {
		return err
	}

	return nil
}
