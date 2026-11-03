// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dijkstra reconstructs and publishes target-directed path witnesses.
// This file verifies original edge orientation, cost, ownership, and result structure.
package dijkstra

import (
	"context"
	"fmt"
	"math"
)

// reconstructSearchPath follows exactly the predecessor arcs of a settled target.
// It returns oriented arc indices, not merely vertex IDs, preserving multigraph identity.
// Implementation:
//   - Stage 1: Validate target settlement and handle the zero-length root path.
//   - Stage 2: Follow incoming parent arcs backward with bounded cycle detection.
//   - Stage 3: Reverse the chain into source-to-target order for publication.
//
// Behavior highlights:
//   - Rejects broken, cyclic, out-of-domain and directionally invalid parent state.
//
// Inputs:
//   - graph: detached effective graph whose arc indices anchor original edges.
//   - source, target: valid dense endpoint indices.
//   - forward: target kernel frontier with final predecessor links.
//
// Returns:
//   - []int: fresh ordered original-arc indices; nil for source==target.
//
// Errors:
//   - ErrNoPath for malformed or missing predecessor state.
//
// Determinism:
//   - Exactly the stored first-strict-improvement predecessor chain.
//
// Complexity:
//   - Time O(k), Space O(V+k) for cycle markers and the route, k<=V-1.
//
// Notes:
//   - This helper never publishes tentative distances or partially valid witnesses.
//
// AI-Hints:
//   - Never derive EdgeIDs from endpoint lookup; parallel edges require linkArc.
//   - Do not omit the cycle guard because malformed internal chains must terminate.
func reconstructSearchPath(
	graph *searchGraph,
	source, target int,
	forward *searchFrontier,
) ([]int, error) {
	if !forward.settled[target] {
		return nil, ErrNoPath
	}
	if source == target {
		return nil, nil
	}

	backwards := make([]int, 0)
	seen := make([]bool, len(graph.ids))
	current := target
	for current != source {
		if current < 0 || current >= len(graph.ids) || seen[current] {
			return nil, ErrNoPath
		}
		seen[current] = true
		arcIndex := forward.linkArc[current]
		if arcIndex < 0 || arcIndex >= len(graph.arcs) {
			return nil, ErrNoPath
		}
		arc := graph.arcs[arcIndex]
		if arc.to != current {
			return nil, ErrNoPath
		}
		backwards = append(backwards, arcIndex)
		current = arc.from
	}
	for left, right := 0, len(backwards)-1; left < right; left, right = left+1, right-1 {
		backwards[left], backwards[right] = backwards[right], backwards[left]
	}
	return backwards, nil
}

// publishSearchPath validates a fully reconstructed path and detaches its public fields.
// Every consecutive vertex pair is certified by one permitted original oriented arc.
// Implementation:
//   - Stage 1: Classify known unreachable and validated source==target outcomes.
//   - Stage 2: Verify arc bounds, direction, original IDs, cycle freedom and exact original cost.
//   - Stage 3: Check cancellation and publish caller-owned slices and scalar policy fields.
//
// Behavior highlights:
//   - Reachable target has len(EdgeIDs)+1 == len(Vertices).
//   - Unreachable target has +Inf distance, Reachable=false and nil witness slices.
//   - Finite costs refer to original edge weights, never potentials or queue priority.
//
// Inputs:
//   - ctx: non-nil caller context, checked during witness assembly and before return.
//   - graph: detached validated topology retaining all original edge identities.
//   - source, target: valid dense endpoint indices.
//   - config: complete target policy, copied by value.
//   - arcs: oriented original-arc indices produced by the kernel's predecessor relation.
//   - distance: finalized original target cost, or +Inf if unreachable.
//
// Returns:
//   - *PathResult: fully detached public witness, or nil on error.
//
// Errors:
//   - ErrNoPath for malformed witness structure or inconsistent original cost.
//   - ErrInexactArithmetic or ErrDistanceOverflow on invalid cost reconstruction arithmetic.
//   - Cancellation and its distinct context cause, without publishing an incumbent.
//
// Determinism:
//   - Retains exact oriented predecessor order and original edge IDs.
//
// Complexity:
//   - Time O(V+k), Space O(V+k) for cycle markers and copied witness, k<=V-1.
//
// Notes:
//   - This post-check is an internal solver certificate, not a public verifier after mutation.
//
// AI-Hints:
//   - Do not use edge.From/To as if all edges were directed.
//   - Do not return a partial route if any arc or cost check fails.
//   - Do not leak graph pointers, workspaces or callback ownership through PathResult.
//
// nolint:gocyclo
func publishSearchPath(
	ctx context.Context,
	graph *searchGraph,
	source, target int,
	config SearchOptions,
	arcs []int,
	distance float64,
) (*PathResult, error) {
	result := &PathResult{
		SourceID: graph.ids[source], TargetID: graph.ids[target],
		Method: config.Method, MaxDistance: config.MaxDistance,
		InfEdgeThreshold: config.InfEdgeThreshold,
	}

	if math.IsInf(distance, 1) {
		if len(arcs) != 0 || source == target {
			return nil, ErrNoPath
		}
		if err := checkSearchContext(ctx); err != nil {
			return nil, err
		}
		result.Distance = math.Inf(1)
		return result, nil
	}
	if math.IsNaN(distance) || math.IsInf(distance, 0) || distance < 0 {
		return nil, ErrNoPath
	}

	result.Reachable = true
	result.Vertices = make([]string, 1, len(arcs)+1)
	result.Vertices[0] = graph.ids[source]
	if len(arcs) > 0 {
		result.EdgeIDs = make([]string, 0, len(arcs))
	}

	visited := make([]bool, len(graph.ids))
	visited[source] = true
	current := source
	cost := float64(0)
	for _, arcIndex := range arcs {
		if err := checkSearchContext(ctx); err != nil {
			return nil, err
		}
		if arcIndex < 0 || arcIndex >= len(graph.arcs) {
			return nil, ErrNoPath
		}
		arc := graph.arcs[arcIndex]
		if arc.from != current || arc.to < 0 || arc.to >= len(graph.ids) ||
			arc.edge < 0 || arc.edge >= len(graph.edges) || visited[arc.to] {
			return nil, ErrNoPath
		}

		// Validate the original directedness and identity rather than merely
		// accepting the endpoint pair. A parallel slower edge is a different
		// witness even if it shares both vertices with the chosen arc.
		edge := graph.edges[arc.edge]
		fromID, toID := graph.ids[arc.from], graph.ids[arc.to]
		if edge.ID == "" || (edge.Directed && (edge.From != fromID || edge.To != toID)) ||
			(!edge.Directed && !((edge.From == fromID && edge.To == toID) ||
				(edge.From == toID && edge.To == fromID))) || edge.Weight >= config.InfEdgeThreshold {
			return nil, ErrNoPath
		}

		nextCost, allowed, err := addSearchCost(cost, edge.Weight, config.MaxDistance)
		if err != nil {
			return nil, fmt.Errorf("dijkstra: verify edge_id=%q: %w", edge.ID, err)
		}
		if !allowed {
			return nil, ErrNoPath
		}
		cost = nextCost
		visited[arc.to] = true
		current = arc.to
		result.Vertices = append(result.Vertices, toID)
		result.EdgeIDs = append(result.EdgeIDs, edge.ID)
	}

	if current != target || distance != cost ||
		(source == target && (len(arcs) != 0 || distance != 0)) ||
		(source != target && len(arcs) == 0) || cost > config.MaxDistance {
		return nil, ErrNoPath
	}
	if err := checkSearchContext(ctx); err != nil {
		return nil, err
	}
	result.Distance = cost
	return result, nil
}

// Clone returns a detached deep copy of the target result's slice fields.
// Scalar values and nil-versus-empty slice distinctions are preserved exactly.
// Implementation:
//   - Stage 1: Handle the nil receiver without panicking.
//   - Stage 2: Copy all scalar fields by value.
//   - Stage 3: Copy each non-nil slice into independent backing storage.
//
// Behavior highlights:
//   - Cloning does not validate or repair caller-mutated fields.
//
// Inputs:
//   - None; receiver may be nil and may contain caller-modified values.
//
// Returns:
//   - *PathResult: caller-owned deep slice copy, or nil for a nil receiver.
//
// Errors:
//   - None.
//
// Determinism:
//   - Exact scalar and ordered-slice copy.
//
// Complexity:
//   - Time O(k), Space O(k), k=number of witness vertices plus edges.
//
// Notes:
//   - Strings are immutable; no graph/workspace/context pointer is retained.
//
// AI-Hints:
//   - Do not use shallow assignment alone for the exported Vertices/EdgeIDs slices.
func (r *PathResult) Clone() *PathResult {
	if r == nil {
		return nil
	}
	copyResult := *r
	if r.Vertices != nil {
		copyResult.Vertices = append(make([]string, 0, len(r.Vertices)), r.Vertices...)
	}
	if r.EdgeIDs != nil {
		copyResult.EdgeIDs = append(make([]string, 0, len(r.EdgeIDs)), r.EdgeIDs...)
	}
	return &copyResult
}

// Path returns a structurally validated, independently owned witness copy.
// It cannot re-check historical edge weights or optimality without the source graph.
// Implementation:
//   - Stage 1: Reject nil, unreachable and malformed caller-owned result state.
//   - Stage 2: Check vertex/edge count, endpoints, IDs, repetition and scalar policy.
//   - Stage 3: Copy both valid slices to independently owned storage.
//
// Behavior highlights:
//   - A reachable self path contains exactly one vertex, no edges and cost zero.
//   - Unknown historical graph membership is not reconstructed from mutable fields.
//
// Inputs:
//   - None; receiver's public fields are caller-owned and may have been modified.
//
// Returns:
//   - vertices, edgeIDs: detached copies in source-to-target order.
//   - error: nil only when the current result structure is self-consistent.
//
// Errors:
//   - ErrNilResult for a nil receiver.
//   - ErrNoPath for unreachable or structurally invalid caller-owned state.
//
// Determinism:
//   - Checks and copies slices in stored order; no map enumeration is published.
//
// Complexity:
//   - Time O(k), Space O(k), k=number of returned vertices plus edges.
//
// Notes:
//   - A malicious caller can construct a plausible but historically false witness;
//     only the producer's verified publication certifies its original provenance.
//
// AI-Hints:
//   - Never claim this method proves optimality after callers edit exported fields.
//   - Reject repeated vertices rather than trusting a caller-supplied predecessor chain.
func (r *PathResult) Path() (vertices, edgeIDs []string, err error) {
	if r == nil {
		return nil, nil, ErrNilResult
	}
	if !r.Reachable {
		return nil, nil, ErrNoPath
	}
	if r.SourceID == "" || r.TargetID == "" || r.Distance < 0 ||
		math.IsNaN(r.Distance) || math.IsInf(r.Distance, 0) ||
		validateMaxDistance(r.MaxDistance) != nil ||
		validateInfEdgeThreshold(r.InfEdgeThreshold) != nil ||
		r.Distance > r.MaxDistance ||
		len(r.Vertices) == 0 || len(r.EdgeIDs)+1 != len(r.Vertices) ||
		r.Vertices[0] != r.SourceID || r.Vertices[len(r.Vertices)-1] != r.TargetID {
		return nil, nil, ErrNoPath
	}
	if r.SourceID == r.TargetID && (len(r.Vertices) != 1 || r.Distance != 0) {
		return nil, nil, ErrNoPath
	}

	seenVertices := make(map[string]struct{}, len(r.Vertices))
	for _, vertexID := range r.Vertices {
		if vertexID == "" {
			return nil, nil, ErrNoPath
		}
		if _, exists := seenVertices[vertexID]; exists {
			return nil, nil, ErrNoPath
		}
		seenVertices[vertexID] = struct{}{}
	}
	seenEdges := make(map[string]struct{}, len(r.EdgeIDs))
	for _, edgeID := range r.EdgeIDs {
		if edgeID == "" {
			return nil, nil, ErrNoPath
		}
		if _, exists := seenEdges[edgeID]; exists {
			return nil, nil, ErrNoPath
		}
		seenEdges[edgeID] = struct{}{}
	}

	vertices = append([]string{}, r.Vertices...)
	if r.EdgeIDs != nil {
		edgeIDs = append([]string{}, r.EdgeIDs...)
	}
	return vertices, edgeIDs, nil
}
