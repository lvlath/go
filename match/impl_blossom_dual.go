// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package match implements the dual/slack subsystem for weighted Blossom search.
// Dual movement exposes new tight edges when the current alternating forest has
// no immediately applicable grow, shrink, augment, or expand event.
//
// Responsibility:
//   - Compute edge slack under active top-level blossom duals.
//   - Select the next deterministic dual delta.
//   - Apply delta updates to outer and inner active nodes.
//   - Preserve non-negative dual feasibility within numeric tolerance.
//
// Boundaries:
//   - This file does not scan queues.
//   - This file does not mutate mate[].
//   - This file does not contract or expand blossom cycle metadata.
//
// AI-Hints:
//   - Do not compute final matching cost from dual values.
//   - Do not silently clamp large negative duals.
//   - Do not change tie-breaking without updating deterministic tests.
package match

import "math"

const (
	// blossomDualDeltaScale is the doubled-convention factor used when a labeled
	// contracted blossom changes state. Vertex duals move by delta; blossom duals
	// move by 2*delta to preserve internal edge slack.
	blossomDualDeltaScale = 2.0
)

// blossomDeltaKind names the dual-update reason selected by nextDelta.
type blossomDeltaKind uint8

const (
	// deltaNone means no finite dual update was found.
	deltaNone blossomDeltaKind = iota

	// deltaGrowToUnlabeled makes at least one outer-to-unlabeled edge tight.
	deltaGrowToUnlabeled

	// deltaJoinOuterTrees makes at least one outer-to-outer edge tight.
	deltaJoinOuterTrees

	// deltaExpandInnerBlossom reduces an inner blossom dual to zero.
	deltaExpandInnerBlossom
)

// blossomDelta stores one dense dual update candidate.
// The engine applies the smallest non-negative candidate to create a new tight
// edge or unlock a zero-dual expansion.
//
// AI-Hints:
//   - Do not accumulate matching cost from delta values.
//   - Cost is recomputed from matchingProblem after export.
type blossomDelta struct {
	// kind identifies why this delta is valid.
	kind blossomDeltaKind

	// value is the non-negative dual movement to apply.
	// Small negative values inside eps are clamped before selection.
	value float64

	// edge is the dense edge made tight by this delta.
	// noEdge is valid for inner-blossom expansion deltas.
	edge int

	// node is the node affected by this delta.
	// For grow it is the unlabeled endpoint; for expansion it is the blossom node.
	node int
}

// slack computes doubled reduced slack for one dense edge under the current
// vertex and allocated-blossom dual system.
//
// Mathematical convention:
//
//	slack(uv) = y(u) + y(v) + Σ z(B) - 2*profit(uv),
//	where B ranges over allocated blossoms containing both u and v.
//
// Implementation:
//   - Stage 1: Start with original endpoint vertex duals.
//   - Stage 2: Add every live blossom dual whose members contain both endpoints.
//   - Stage 3: Subtract doubled transformed profit.
//
// Behavior highlights:
//   - Does not use inBlossom as a dual lookup.
//   - Inactive nested blossoms still contribute while their cycle metadata exists.
//   - A tight edge has slack <= eps.
//
// Inputs:
//   - edgeID: dense local edge identifier.
//
// Returns:
//   - float64: doubled reduced slack.
//
// Errors:
//   - None. Callers validate edge IDs before use or rely on internal dense edge IDs.
//
// Determinism:
//   - Scans allocated blossom IDs in increasing order.
//
// Complexity:
//   - Time O(B*m), Space O(1), where B is allocated blossom count and m is membership scan cost.
//
// Notes:
//   - This correctness-first formula is intentionally explicit.
//   - Later optimization may cache blossom containment, but must preserve this contract.
//
// AI-Hints:
//   - Do not replace this with dual[inBlossom[u]] + dual[inBlossom[v]].
//   - inBlossom is forest ownership, not full laminar dual ownership.
func (e *blossomEngine) slack(edgeID int) float64 {
	edge := e.edges[edgeID]

	value := e.dual[edge.u] + e.dual[edge.v]

	for node := e.problem.n; node < e.nextNode; node++ {
		if !e.isAllocatedBlossom(node) {
			continue
		}
		if e.nodeContainsVertex(node, edge.u) && e.nodeContainsVertex(node, edge.v) {
			value += e.dual[node]
		}
	}

	return value - blossomDualDeltaScale*edge.profit
}

// isTight reports whether one dense edge is tight under the current laminar dual state.
// It increments TightScans as deterministic internal telemetry for scans and benchmarks.
//
// Implementation:
//   - Stage 1: Increment tight-scan telemetry.
//   - Stage 2: Compute reduced slack.
//   - Stage 3: Compare slack against eps tolerance.
//
// Behavior highlights:
//   - Does not mutate matching, forest, or dual values.
//   - Treats small positive slack within eps as tight.
//   - Uses full laminar slack, including nested allocated blossom duals.
//
// Inputs:
//   - edgeID: dense local edge identifier.
//
// Returns:
//   - bool: true when slack(edgeID) <= eps.
//
// Errors:
//   - None. Callers must pass a valid edge ID.
//
// Determinism:
//   - Pure computation plus deterministic counter increment.
//
// Complexity:
//   - Time O(B*m), Space O(1), inherited from slack.
//
// Notes:
//   - TightScans is diagnostic only and must not affect solver decisions.
//
// AI-Hints:
//   - Do not bypass slack() for speed unless a cached laminar ownership structure is introduced.
func (e *blossomEngine) isTight(edgeID int) bool {
	e.stats.TightScans++

	return e.slack(edgeID) <= e.dualTolerance()
}

// improveDelta updates best when candidate is a smaller valid non-negative dual movement.
// Equal values inside eps are tie-broken deterministically by smaller dense edge ID when
// both candidates are edge-backed.
//
// Implementation:
//   - Stage 1: Reject negative, NaN, or infinite candidate values.
//   - Stage 2: Accept candidate when best is empty or candidate value is smaller.
//   - Stage 3: Resolve eps ties by edge ID when both edge IDs are present.
//
// Behavior highlights:
//   - Mutates only the best delta value.
//   - Keeps deterministic event selection under equal slack.
//   - Ignores invalid candidates instead of publishing them.
//
// Inputs:
//   - best: pointer to current best delta.
//   - candidate: candidate dual movement.
//
// Returns:
//   - None.
//
// Errors:
//   - None. Invalid candidates are ignored.
//
// Determinism:
//   - Stable numeric and edge-ID tie-breaking.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - node-only expansion deltas use noEdge and are not edge-ID tie-broken.
//
// AI-Hints:
//   - Do not use random or map-order tie-breaking here.
/*func (e *blossomEngine) improveDelta(best *blossomDelta, candidate blossomDelta) {
	tol := e.dualTolerance()

	if math.IsNaN(candidate.value) || math.IsInf(candidate.value, 0) {
		return
	}

	if candidate.value < 0 {
		if candidate.value < -tol {
			return
		}

		// Tiny negative drift is equivalent to an already-tight candidate.
		candidate.value = 0
	}

	if best.kind == deltaNone || candidate.value < best.value-tol ||
		(math.Abs(candidate.value-best.value) <= tol && candidate.edge != noEdge &&
			(best.edge == noEdge || candidate.edge < best.edge)) {
		*best = candidate
	}
}*/
func (e *blossomEngine) improveDelta(best *blossomDelta, candidate blossomDelta) {
	feasTol := e.dualTolerance()
	selectTol := e.dualSelectionTolerance()

	if math.IsNaN(candidate.value) || math.IsInf(candidate.value, 0) {
		return
	}

	if candidate.value < 0 {
		if candidate.value < -feasTol {
			return
		}

		// Tiny negative drift means the candidate is already tight.
		candidate.value = 0
	}

	if best.kind == deltaNone {
		*best = candidate
		return
	}

	if candidate.value < best.value-selectTol {
		*best = candidate
		return
	}

	if math.Abs(candidate.value-best.value) <= selectTol && deltaTieLess(candidate, *best) {
		*best = candidate
	}
}

// isAllocatedBlossom reports whether node owns live contracted blossom metadata.
// Inactive nested blossoms still count as allocated because their blossom duals may
// contribute to slack while the outer contracted structure remains alive.
//
// Implementation:
//   - Stage 1: Require node to be in the contracted-blossom ID range.
//   - Stage 2: Require node to be below nextNode.
//   - Stage 3: Require non-empty cycle metadata.
//   - Stage 4: Require non-empty member metadata.
//
// Behavior highlights:
//   - Does not require active[node].
//   - Does not mutate engine state.
//   - Distinguishes allocated blossoms from original singleton vertices.
//
// Inputs:
//   - node: engine node ID.
//
// Returns:
//   - bool: true when node has live blossom metadata.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pure indexed checks.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This helper is central to laminar slack and final dual checks.
//
// AI-Hints:
//   - Do not add active[node] to this predicate.
func (e *blossomEngine) isAllocatedBlossom(node int) bool {
	return node >= e.problem.n &&
		node < e.nextNode &&
		len(e.cycles[node]) > 0 &&
		len(e.members[node]) > 0
}

// nextDelta selects the smallest deterministic dual update that can expose the next
// structural event when no currently tight grow, shrink, augment, or expand event exists.
//
// Implementation:
//   - Stage 1: Initialize best as +Inf / deltaNone.
//   - Stage 2: Consider outer-to-unlabeled grow deltas.
//   - Stage 3: Consider outer-to-outer join deltas.
//   - Stage 4: Consider zero-dual inner blossom expansion deltas.
//   - Stage 5: Reject when no finite candidate exists.
//
// Behavior highlights:
//   - Does not mutate duals.
//   - Uses deterministic tie-breaking through improveDelta.
//   - Separates candidate selection from applyDelta mutation.
//
// Inputs:
//   - None; reads labels, active nodes, duals, edges, and ownership.
//
// Returns:
//   - blossomDelta: selected non-negative dual movement.
//   - error: nil when a finite candidate exists.
//
// Errors:
//   - ErrInvalidMatching when no valid finite delta can be found or candidate scans fail.
//
// Determinism:
//   - Fixed scan order and deterministic tie-breaking.
//
// Complexity:
//   - Time O(k^2 * B*m) in dense worst case through slack checks, Space O(1).
//
// Notes:
//   - The returned delta must be applied by applyDelta before scanning resumes.
//
// AI-Hints:
//   - Do not apply dual changes inside this method.
func (e *blossomEngine) nextDelta() (blossomDelta, error) {
	best := blossomDelta{
		kind:  deltaNone,
		value: math.Inf(1),
		edge:  noEdge,
		node:  noNode,
	}

	if err := e.considerGrowDeltas(&best); err != nil {
		return best, err
	}
	if err := e.considerJoinDeltas(&best); err != nil {
		return best, err
	}
	if err := e.considerExpandDeltas(&best); err != nil {
		return best, err
	}

	if best.kind == deltaNone || math.IsInf(best.value, 1) {
		return best, ErrIncompleteGraph
	}

	return best, nil
}

// considerGrowDeltas finds dual movements that make an outer-to-unlabeled edge tight.
// Such an edge can grow the alternating forest once the selected delta is applied.
//
// Implementation:
//   - Stage 1: Scan active outer nodes.
//   - Stage 2: Scan incident dense edges of every original member.
//   - Stage 3: Resolve the opposite endpoint's active top-level node.
//   - Stage 4: Consider slack(edge) for unlabeled targets.
//
// Behavior highlights:
//   - Does not mutate duals or labels.
//   - Skips self edges inside the same top-level node.
//   - Publishes target node in the delta for later grow classification.
//
// Inputs:
//   - best: pointer to currently selected best delta.
//
// Returns:
//   - error: nil after all grow candidates are considered.
//
// Errors:
//   - ErrInvalidMatching for invalid ownership or malformed top-node lookup.
//
// Determinism:
//   - Active node order, member order, and incident edge order are fixed.
//
// Complexity:
//   - Time O(k^2 * B*m), Space O(1) in dense worst case.
//
// Notes:
//   - Candidate value is full slack, not slack/2.
//
// AI-Hints:
//   - Do not consider inner or already outer targets here.
func (e *blossomEngine) considerGrowDeltas(best *blossomDelta) error {
	for _, edge := range e.edges {
		uTop, err := e.topNodeOfVertex(edge.u)
		if err != nil {
			return err
		}
		vTop, err := e.topNodeOfVertex(edge.v)
		if err != nil {
			return err
		}
		if uTop == vTop {
			continue
		}

		uOuter := e.label[uTop] == blossomOuter
		vOuter := e.label[vTop] == blossomOuter

		if uOuter && e.label[vTop] == blossomUnlabeled {
			e.improveDelta(best, blossomDelta{kind: deltaGrowToUnlabeled, value: e.slack(edge.id), edge: edge.id, node: vTop})
		}
		if vOuter && e.label[uTop] == blossomUnlabeled {
			e.improveDelta(best, blossomDelta{kind: deltaGrowToUnlabeled, value: e.slack(edge.id), edge: edge.id, node: uTop})
		}
	}

	return nil
}

// considerJoinDeltas finds dual movements that make an outer-to-outer edge tight.
// After the selected delta is applied, same-root joins become shrink events and
// different-root joins become augment events.
//
// Implementation:
//   - Stage 1: Scan active outer nodes.
//   - Stage 2: Scan incident dense edges of represented original members.
//   - Stage 3: Resolve the opposite endpoint's active top-level node.
//   - Stage 4: Consider slack(edge)/2 for outer targets in canonical node order.
//
// Behavior highlights:
//   - Does not mutate duals, labels, or matching.
//   - Skips same-node edges.
//   - Uses half slack because both endpoints are outer and move toward tightness.
//
// Inputs:
//   - best: pointer to currently selected best delta.
//
// Returns:
//   - error: nil after all join candidates are considered.
//
// Errors:
//   - ErrInvalidMatching for invalid ownership or malformed top-node lookup.
//
// Determinism:
//   - Fixed scan order plus improveDelta tie-breaking.
//
// Complexity:
//   - Time O(k^2 * B*m), Space O(1) in dense worst case.
//
// Notes:
//   - This method does not decide shrink vs augment; event classification happens after tight scan.
//
// AI-Hints:
//   - Do not use full slack for outer/outer joins.
func (e *blossomEngine) considerJoinDeltas(best *blossomDelta) error {
	for _, edge := range e.edges {
		uTop, err := e.topNodeOfVertex(edge.u)
		if err != nil {
			return err
		}
		vTop, err := e.topNodeOfVertex(edge.v)
		if err != nil {
			return err
		}
		if uTop == vTop {
			continue
		}
		if e.label[uTop] == blossomOuter && e.label[vTop] == blossomOuter {
			e.improveDelta(best, blossomDelta{
				kind:  deltaJoinOuterTrees,
				value: e.slack(edge.id) / 2,
				edge:  edge.id,
				node:  noNode,
			})
		}
	}

	return nil
}

// considerExpandDeltas finds active inner contracted blossoms whose blossom dual can be
// reduced to zero. The selected delta triggers expansion when applied.
//
// Implementation:
//   - Stage 1: Scan allocated node IDs in deterministic order.
//   - Stage 2: Keep only active inner allocated blossoms.
//   - Stage 3: Convert blossom dual z(B) to vertex-dual movement z(B)/2.
//   - Stage 4: Publish an expansion delta for the best candidate.
//
// Behavior highlights:
//   - Does not mutate duals or expand nodes.
//   - Ignores inactive nested blossoms.
//   - Uses doubled-convention scaling consistently.
//
// Inputs:
//   - best: pointer to currently selected best delta.
//
// Returns:
//   - error: nil after expansion candidates are considered.
//
// Errors:
//   - None in normal operation.
//
// Determinism:
//   - Fixed node-ID scan and improveDelta tie-breaking.
//
// Complexity:
//   - Time O(k), Space O(1).
//
// Notes:
//   - applyDelta must be followed by expand(delta.node) for expansion deltas.
//
// AI-Hints:
//   - Do not use e.dual[node] directly; expansion movement is z(B)/2.
func (e *blossomEngine) considerExpandDeltas(best *blossomDelta) error {
	for node := e.problem.n; node < e.nextNode; node++ {
		if !e.active[node] || e.label[node] != blossomInner || !e.isAllocatedBlossom(node) {
			continue
		}

		e.improveDelta(best, blossomDelta{
			kind:  deltaExpandInnerBlossom,
			value: e.dual[node] / blossomDualDeltaScale,
			edge:  noEdge,
			node:  node,
		})
	}

	return nil
}

// applyDelta moves the laminar dual system by one selected non-negative delta.
// Vertex duals move for every original vertex inside labeled top-level nodes.
// Allocated top-level blossom duals move in the opposite doubled amount to keep
// their internal cycle slack stable.
//
// Implementation:
//   - Stage 1: Reject invalid delta values.
//   - Stage 2: For every active outer node, decrease member vertex duals.
//   - Stage 3: For every active inner node, increase member vertex duals.
//   - Stage 4: For allocated outer blossoms, increase blossom dual by 2*delta.
//   - Stage 5: For allocated inner blossoms, decrease blossom dual by 2*delta.
//   - Stage 6: Clamp tiny zero drift on blossom duals and reject negative blossom z.
//
// Behavior highlights:
//   - inBlossom is not used as a dual storage shortcut.
//   - Original vertex duals may become negative; they are equality duals and are not constrained non-negative.
//   - Allocated blossom duals must remain non-negative.
//   - The caller must rescan the forest after this method returns.
//
// Inputs:
//   - delta: selected dual movement from nextDelta.
//
// Returns:
//   - error: nil when blossom dual feasibility is preserved.
//
// Errors:
//   - ErrInvalidMatching for invalid delta values, corrupt labels, invalid members,
//     or negative allocated blossom duals.
//
// Determinism:
//   - Fixed active node-index scan and member order.
//
// Complexity:
//   - Time O(total members of active labeled nodes), Space O(1).
//
// Notes:
//   - This method preserves internal slack of contracted blossoms by moving z(B)
//     opposite to member vertex dual movement.
//
// AI-Hints:
//   - Do not update only dual[node] for top-level nodes.
//   - Do not require original vertex duals to be non-negative.
//   - Do not forget the 2*delta blossom-dual shift.
func (e *blossomEngine) applyDelta(delta blossomDelta) error {
	tol := e.dualTolerance()

	if math.IsNaN(delta.value) || math.IsInf(delta.value, 0) {
		return ErrInvalidMatching
	}

	if delta.value < 0 {
		if delta.value < -tol {
			return ErrInvalidMatching
		}

		// Only tiny negative floating-point drift may be clamped to zero.
		// Positive deltas, even smaller than tol, must still be applied because
		// they may be exactly the slack/2 movement needed to create a tight edge.
		delta.value = 0
	}

	for node := range e.active {
		if !e.active[node] {
			continue
		}

		switch e.label[node] {
		case blossomOuter:
			if err := e.shiftMemberVertexDuals(node, -delta.value); err != nil {
				return err
			}
			if e.isAllocatedBlossom(node) {
				e.dual[node] += blossomDualDeltaScale * delta.value
			}

		case blossomInner:
			if err := e.shiftMemberVertexDuals(node, delta.value); err != nil {
				return err
			}
			if e.isAllocatedBlossom(node) {
				e.dual[node] -= blossomDualDeltaScale * delta.value
				if math.Abs(e.dual[node]) <= tol {
					e.dual[node] = 0
				}
				if e.dual[node] < -tol {
					return ErrInvalidMatching
				}
			}

		case blossomUnlabeled:
			continue

		default:
			return ErrInvalidMatching
		}
	}

	e.stats.DualUpdates++

	return nil
}

// shiftMemberVertexDuals adds delta to every original vertex dual represented by one
// active top-level node. Singleton and contracted blossom nodes are both handled through
// members[], while vertex duals live only in dual[v] for v < problem.n.
//
// Implementation:
//   - Stage 1: Validate node membership storage.
//   - Stage 2: Scan original member vertices.
//   - Stage 3: Validate every original vertex ID.
//   - Stage 4: Add delta to dual[vertex].
//   - Stage 5: Clamp tiny numeric drift to exact zero.
//
// Behavior highlights:
//   - Mutates original vertex duals only.
//   - Does not update blossom dual z(B); applyDelta handles blossom dual shifts separately.
//   - Allows vertex duals to become negative.
//
// Inputs:
//   - node: active singleton or contracted blossom node.
//   - delta: signed vertex-dual change.
//
// Returns:
//   - error: nil after all represented vertex duals are shifted.
//
// Errors:
//   - ErrInvalidMatching for invalid node membership or invalid original member IDs.
//
// Determinism:
//   - Fixed member order scan.
//
// Complexity:
//   - Time O(|members(node)|), Space O(1).
//
// Notes:
//   - Original vertex duals are equality duals and are not constrained non-negative.
//
// AI-Hints:
//   - Do not apply blossomDualDeltaScale in this helper.
func (e *blossomEngine) shiftMemberVertexDuals(node int, delta float64) error {
	if node < 0 || node >= len(e.members) || len(e.members[node]) == 0 {
		return ErrInvalidMatching
	}

	tol := e.dualTolerance()

	for _, vertex := range e.members[node] {
		if vertex < 0 || vertex >= e.problem.n {
			return ErrInvalidMatching
		}

		e.dual[vertex] += delta
		//if math.Abs(e.dual[vertex]) <= e.eps {
		if math.Abs(e.dual[vertex]) <= tol {
			e.dual[vertex] = 0
		}
	}

	return nil
}

// dualSelectionTolerance returns the tiny tolerance used only for deterministic
// tie-breaking between dual-delta candidates.
//
// Mathematical contract:
//   - Delta selection must choose the smallest available non-negative movement.
//   - Feasibility tolerance may be wider, but selection tolerance must stay narrow
//     to avoid overshooting near-tie slacks.
//
// Implementation:
//   - Stage 1: Compute machine epsilon around 1.
//   - Stage 2: Scale it by the largest local matching cost.
//   - Stage 3: Return a small ULP budget, with e.eps as a lower bound only when
//     e.eps is smaller than the feasibility policy would make unsafe.
//
// Behavior highlights:
//   - Does not change slack feasibility.
//   - Does not change matching objective.
//   - Used only by improveDelta tie-breaking.
//
// Inputs:
//   - None; reads e.scale.
//
// Returns:
//   - float64: narrow numeric tolerance for delta ordering.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pure numeric function.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This must remain much smaller than dualTolerance.
//
// AI-Hints:
//   - Do not use dualTolerance for delta ordering.
//   - Do not turn this into a percentage tolerance.
func (e *blossomEngine) dualSelectionTolerance() float64 {
	machineEps := math.Nextafter(1, 2) - 1
	scale := e.scale
	if scale < 1 || math.IsNaN(scale) || math.IsInf(scale, 0) {
		scale = 1
	}

	tol := blossomDeltaSelectionToleranceMultiplier * machineEps * scale
	if tol < 0 {
		return 0
	}

	return tol
}

// deltaTieLess reports whether candidate should replace best when both dual movements
// are numerically indistinguishable under dualSelectionTolerance. It exists only as a
// deterministic tie-breaker; improveDelta still owns numeric ordering.
//
// Implementation:
//   - Stage 1: Prefer edge-backed events over no-edge expansion events.
//   - Stage 2: Prefer the smaller dense edge ID when both candidates are edge-backed.
//   - Stage 3: Prefer the smaller target node ID when edge IDs are equal.
//   - Stage 4: Prefer the smaller delta kind as the final stable fallback.
//
// Behavior highlights:
//   - Does not compare candidate.value.
//   - Does not mutate engine state.
//   - Keeps near-tie dual selection deterministic without widening numeric ordering.
//   - Treats noEdge/noNode as lower-priority sentinels.
//
// Inputs:
//   - candidate: challenger delta candidate.
//   - best: currently selected delta candidate.
//
// Returns:
//   - bool: true when candidate should replace best under equal numeric movement.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pure field comparison with fixed priority order.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This helper must be called only after improveDelta has established
//     |candidate.value-best.value| <= dualSelectionTolerance.
//
// AI-Hints:
//   - Do not use dualTolerance here; it is intentionally too wide for delta ordering.
//   - Do not make expansion events win over edge-backed events unless tests and proof are updated.
func deltaTieLess(candidate blossomDelta, best blossomDelta) bool {
	if candidate.edge != best.edge {
		if candidate.edge == noEdge {
			return false
		}
		if best.edge == noEdge {
			return true
		}

		return candidate.edge < best.edge
	}

	if candidate.node != best.node {
		if candidate.node == noNode {
			return false
		}
		if best.node == noNode {
			return true
		}

		return candidate.node < best.node
	}

	return candidate.kind < best.kind
}
