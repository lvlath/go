// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package match manages the alternating forest for dense weighted Blossom search.
// The forest discovers tight-edge events: grow, shrink, augment, and expand.
// It is rebuilt for each augmentation attempt from the current committed matching.
//
// Responsibility:
//   - Reset and seed alternating forests from unmatched vertices.
//   - Scan tight edges in deterministic FIFO outer-node order.
//   - Classify events by endpoint labels and tree roots.
//   - Route events to grow, shrink, augment, or expand handlers.
//
// Boundaries:
//   - Slack and dual updates live in impl_blossom_dual.go.
//   - Path realization and lifting live in impl_blossom_path.go.
//   - Matching mutation lives in impl_blossom_augment.go.
//   - Cycle contraction details live in impl_blossom_contract.go.
//
// AI-Hints:
//   - Do not mutate mate[] from forest scanning.
//   - Do not classify same-tree outer/outer edges as augmenting paths.
//   - Do not reuse forest labels across augmentation attempts.
package match

import "fmt"

// blossomEventKind names the next structural action found by tight-edge scanning.
// Events are applied immediately by applyBlossomEvent.
type blossomEventKind uint8

const (
	// eventNone means no applicable tight-edge event was found.
	eventNone blossomEventKind = iota

	// eventGrow grows an alternating tree from an outer node to an unlabeled node.
	eventGrow

	// eventShrink contracts an odd cycle found inside one alternating tree.
	eventShrink

	// eventAugment applies an augmenting path between two outer roots.
	eventAugment

	// eventExpand expands a zero-dual inner blossom.
	eventExpand
)

// blossomEvent describes one structural action discovered by tight-edge scanning.
// It preserves both top-level node ownership and original endpoint ownership,
// which is required for contraction metadata and augmenting-path lifting.
//
// Implementation:
//   - Stage 1: scanOuterNodeTightEdges detects a tight dense edge.
//   - Stage 2: orientEdgeForNodes maps the dense edge endpoints into active top-level nodes.
//   - Stage 3: The event stores both node IDs and original local vertices.
//   - Stage 4: applyBlossomEvent dispatches one mutation without re-deriving endpoint ownership.
//
// Behavior highlights:
//   - Transient value; never stored across dual updates.
//   - a and b are active top-level node IDs.
//   - aVertex is an original local vertex currently owned by a.
//   - bVertex is an original local vertex currently owned by b.
//
// Inputs:
//   - Produced by tight-edge scanning.
//
// Returns:
//   - Consumed by grow, shrink, augment, or expand.
//
// Errors:
//   - Invalid endpoint ownership is rejected before the event is returned.
//
// Determinism:
//   - The first event follows queue order, member order, incident edge order, and edge ID order.
//
// Complexity:
//   - Storage O(1).
//   - Endpoint orientation is O(|members(a)|+|members(b)|) through membership checks.
//
// Notes:
//   - Path lifting must not infer endpoints from node IDs alone.
//   - A contracted blossom may contain many original vertices, so endpoint ownership is essential.
//
// AI-Hints:
//   - Do not remove aVertex/bVertex.
//   - Do not apply events directly inside incident-edge loops.
//   - Do not store original matrix vertices in a/b.
//   - Do not store original matrix indices here; these are local matchingProblem vertices.
type blossomEvent struct {
	// kind selects which structural mutation must be applied.
	kind blossomEventKind

	// edge is the dense local edge that triggered grow, shrink, or augment.
	edge int

	// a is the first active top-level node participating in the event.
	a int

	// b is the second active top-level node participating in the event.
	b int

	// aVertex is the original local endpoint of edge owned by node a.
	aVertex int

	// bVertex is the original local endpoint of edge owned by node b.
	bVertex int
}

// solve repeatedly finds augmenting paths until every original local vertex is matched.
// Each successful outer iteration must increase the number of committed matching pairs.
//
// Implementation:
//   - Stage 1: Compute the target number of pairs from the local matching problem size.
//   - Stage 2: Search for one complete augmenting path.
//   - Stage 3: Verify that the committed matching size strictly increased.
//   - Stage 4: Repeat until the perfect matching cardinality is reached.
//   - Stage 5: Verify final structural, dual, and matching invariants.
//
// Behavior highlights:
//   - Grow, shrink, and expand events do not count as completed augmentations.
//   - ErrInvalidMatching is returned when a search iteration reports success without progress.
//   - The method never downgrades to a heuristic matching.
//
// Inputs:
//   - None; uses the engine initialized by newBlossomEngine.
//
// Returns:
//   - error: nil when a perfect matching is found.
//
// Errors:
//   - ErrIncompleteGraph from search when no augmenting path can be exposed.
//   - ErrInvalidMatching when the engine reports a no-progress augmentation.
//   - Verification sentinels from verifyOptimalState.
//
// Determinism:
//   - Fixed root seeding, edge scanning, delta tie-breaking, and event routing order.
//
// Complexity:
//   - Dense correctness-first implementation is polynomial but intentionally not heap-optimized.
//   - Space is governed by preallocated engine arrays and lifted path buffers.
//
// Notes:
//   - stats.Augmentations counts committed augmenting-path flips, not forest events.
//
// AI-Hints:
//   - Do not increment Augmentations after grow or shrink.
//   - Do not reset the forest between intermediate events of the same augmentation search.
func (e *blossomEngine) solve() error {
	targetPairs := e.problem.n / 2

	for e.matchedPairs() < targetPairs {
		before := e.matchedPairs()

		if err := e.findAndApplyAugmentation(); err != nil {
			return err
		}

		after := e.matchedPairs()
		if after <= before {
			return ErrInvalidMatching
		}

		e.stats.Augmentations++
	}

	return e.verifyOptimalState()
}

// findAndApplyAugmentation finds and applies one complete augmenting path.
// It may process many grow, shrink, expand, and dual-update events before the
// actual augmenting path is found.
//
// Implementation:
//   - Stage 1: Rebuild the alternating forest from currently unmatched vertices.
//   - Stage 2: Scan tight edges for immediate structural events.
//   - Stage 3: Apply grow/shrink/expand events in-place and continue the same search.
//   - Stage 4: Apply dual deltas when no tight event is currently available.
//   - Stage 5: Return only after an augmenting path has been flipped.
//
// Behavior highlights:
//   - Does not restart the forest after grow or shrink.
//   - Keeps the same search context until augmentation is committed.
//   - Uses dual movement only when the current tight-edge frontier is exhausted.
//
// Inputs:
//   - None; uses the current committed matching and contraction state.
//
// Returns:
//   - error: nil only after one successful matching augmentation.
//
// Errors:
//   - ErrIncompleteGraph when no valid delta/event can expose progress.
//   - ErrInvalidMatching for malformed events, contraction state, or path lifting.
//   - Dual and path sentinels propagated from called subsystems.
//
// Determinism:
//   - Roots are assigned in increasing original-vertex order.
//   - Queue scanning is FIFO over deterministic root insertion.
//   - Delta tie-breaking remains delegated to improveDelta.
//
// Complexity:
//   - One search may scan dense edges many times; correctness is prioritized over sparse optimization.
//   - Space is O(k) plus temporary lifted path buffers.
//
// Notes:
//   - This method is the boundary between “search progress” and “committed matching progress”.
//
// AI-Hints:
//   - Returning after grow/shrink causes infinite restarts.
//   - Only eventAugment should make this function return nil.
func (e *blossomEngine) findAndApplyAugmentation() error {
	e.resetForest()

	for vertex := 0; vertex < e.problem.n; vertex++ {
		if e.mate[vertex] != noVertex {
			continue
		}

		node := e.inBlossom[vertex]
		e.assignOuterRoot(node)
	}

	maxSteps := e.augmentationStepLimit()
	for step := 0; step < maxSteps; step++ {
		event, ok, err := e.scanTightEdges()
		if err != nil {
			return fmt.Errorf("search step=%d scan tight edges queue=%v head=%d label=%v parent=%v root=%v: %w",
				step, e.queue, e.head, e.label, e.parent, e.treeRoot, err)
		}

		if ok {
			done, applyErr := e.applyBlossomEvent(event)
			if applyErr != nil {
				return fmt.Errorf("search step=%d apply event=%+v mate=%v mateEdge=%v label=%v parent=%v root=%v base=%v: %w",
					step, event, e.mate, e.mateEdge, e.label, e.parent, e.treeRoot, e.base, applyErr)
			}
			if done {
				return nil
			}

			continue
		}

		delta, err := e.nextDelta()
		if err != nil {
			return fmt.Errorf("search step=%d next delta mate=%v label=%v parent=%v root=%v dual=%v base=%v: %w",
				step, e.mate, e.label, e.parent, e.treeRoot, e.dual, e.base, err)
		}

		if err = e.applyDelta(delta); err != nil {
			return fmt.Errorf("search step=%d apply delta=%+v dual=%v label=%v base=%v: %w",
				step, delta, e.dual, e.label, e.base, err)
		}

		if delta.kind == deltaExpandInnerBlossom {
			if err = e.expand(delta.node); err != nil {
				return fmt.Errorf("search step=%d expand delta=%+v label=%v parent=%v root=%v dual=%v base=%v: %w",
					step, delta, e.label, e.parent, e.treeRoot, e.dual, e.base, err)
			}
		}

		e.rewindForestScan()
	}

	return fmt.Errorf("search exceeded step limit=%d mate=%v mateEdge=%v label=%v parent=%v root=%v dual=%v base=%v: %w",
		maxSteps, e.mate, e.mateEdge, e.label, e.parent, e.treeRoot, e.dual, e.base, ErrInvalidMatching)
}

// resetForest clears the alternating forest while preserving matching and contraction state.
// Each augmentation search starts from the current committed matching, then builds a fresh
// forest rooted at unmatched original vertices.
//
// Implementation:
//   - Stage 1: Reset every node label to blossomUnlabeled.
//   - Stage 2: Clear labelEdge, treeRoot, and parent links.
//   - Stage 3: Reuse the existing queue backing array and reset the FIFO head cursor.
//
// Behavior highlights:
//   - Does not mutate mate[] or mateEdge[].
//   - Does not mutate active[], base[], cycles[], members[], or inBlossom[].
//   - Keeps slice capacity for queue reuse across augmentation attempts.
//   - Makes old same-tree/cross-tree classifications impossible to reuse accidentally.
//
// Inputs:
//   - None; operates on the current engine state.
//
// Returns:
//   - None.
//
// Errors:
//   - None.
//
// Determinism:
//   - Fixed increasing node-index reset order.
//   - Queue is rebuilt later by assignOuterRoot in deterministic vertex order.
//
// Complexity:
//   - Time O(k), Space O(1), where k is the allocated Blossom node capacity.
//
// Notes:
//   - This is a forest reset, not an engine reset.
//   - Contracted blossoms remain contracted across searches until expansion logic changes them.
//
// AI-Hints:
//   - Do not clear matching arrays here.
//   - Do not allocate a new queue; reuse preserves predictable allocation behavior.
func (e *blossomEngine) resetForest() {
	for node := range e.label {
		e.label[node] = blossomUnlabeled
		e.labelEdge[node] = noEdge
		e.treeRoot[node] = noNode
		e.parent[node] = noNode
	}

	e.queue = e.queue[:0]
	e.head = 0
}

// rewindForestScan restarts tight-edge scanning over the current alternating forest queue.
// It is used after dual movement because previously scanned outer nodes may now expose
// newly tight grow, shrink, or augment edges.
//
// Implementation:
//   - Stage 1: Keep all forest labels, parents, roots, and queued outer nodes unchanged.
//   - Stage 2: Reset only the FIFO scan cursor.
//   - Stage 3: Let scanTightEdges revisit queued outer nodes under the updated dual state.
//
// Behavior highlights:
//   - Does not clear the forest.
//   - Does not mutate matching state.
//   - Does not remove queued nodes.
//   - Makes newly tight edges visible after applyDelta.
//
// Inputs:
//   - None; operates on the current engine queue.
//
// Returns:
//   - None.
//
// Errors:
//   - None.
//
// Determinism:
//   - Replays queued outer nodes in their existing deterministic order.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This is not resetForest.
//   - Calling resetForest here would discard the alternating tree built before the delta.
//
// AI-Hints:
//   - Call this after every successful applyDelta.
//   - Do not call this after grow; grow appends new outer nodes and scanTightEdges can continue naturally.
func (e *blossomEngine) rewindForestScan() {
	e.head = 0
}

// assignOuterRoot inserts an active top-level node as an outer root of the current
// alternating forest. Roots have no parent edge and scan tight outgoing edges.
//
// Implementation:
//   - Stage 1: Assign outer label.
//   - Stage 2: Clear parent and labelEdge.
//   - Stage 3: Set treeRoot[node]=node.
//   - Stage 4: Append node to the FIFO scan queue.
//
// Behavior highlights:
//   - Mutates only forest label/root/queue state.
//   - Does not mutate mate[], duals, or contraction metadata.
//   - Used when seeding search from unmatched original vertices.
//
// Inputs:
//   - node: active top-level node to seed as root.
//
// Returns:
//   - None.
//
// Errors:
//   - None. Caller must pass an active valid node.
//
// Determinism:
//   - Queue append order follows caller vertex scan order.
//
// Complexity:
//   - Time O(1), Space amortized O(1).
//
// Notes:
//   - Multiple roots are valid during one augmentation search.
//
// AI-Hints:
//   - Do not assign a parent to a root.
func (e *blossomEngine) assignOuterRoot(node int) {
	e.label[node] = blossomOuter
	e.treeRoot[node] = node
	e.labelEdge[node] = noEdge
	e.parent[node] = noNode
	e.queue = append(e.queue, node)
}

// topNodeOfVertex returns the active top-level node currently owning an original local vertex.
// It is the safe bridge from dense edge endpoints into the contracted Blossom forest.
//
// Implementation:
//   - Stage 1: Validate original vertex bounds.
//   - Stage 2: Read inBlossom[vertex].
//   - Stage 3: Validate node bounds and active status.
//   - Stage 4: Return the owning top-level node.
//
// Behavior highlights:
//   - Does not mutate engine state.
//   - Rejects stale ownership left behind by failed shrink/expand operations.
//   - Used by tight-edge scanning and grow logic.
//
// Inputs:
//   - vertex: original local vertex ID.
//
// Returns:
//   - int: active top-level owner node.
//   - error: nil when ownership is valid.
//
// Errors:
//   - ErrInvalidMatching for invalid vertex, invalid owner, or inactive owner.
//
// Determinism:
//   - Pure indexed lookup.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This is different from base[node]; base is only the pseudo-node interface.
//
// AI-Hints:
//   - Do not read inBlossom without validating active ownership.
func (e *blossomEngine) topNodeOfVertex(vertex int) (int, error) {
	if vertex < 0 || vertex >= e.problem.n {
		return noNode, ErrInvalidMatching
	}

	node := e.inBlossom[vertex]
	if node < 0 || node >= len(e.active) || !e.active[node] {
		return noNode, ErrInvalidMatching
	}

	return node, nil
}

// nodeContainsVertex reports whether node currently represents vertex.
// It checks explicit membership instead of assuming singleton identity, because
// contracted blossoms own multiple original local vertices.
//
// Implementation:
//   - Stage 1: Validate vertex range.
//   - Stage 2: Validate node range.
//   - Stage 3: Scan members[node] for the original vertex.
//
// Behavior highlights:
//   - Does not mutate engine state.
//   - Works for singleton and contracted blossom nodes.
//   - Treats missing membership as false rather than panicking.
//
// Inputs:
//   - node: engine node ID.
//   - vertex: original local matchingProblem vertex.
//
// Returns:
//   - bool: true when vertex belongs to node.
//
// Errors:
//   - None.
//
// Determinism:
//   - Fixed member scan order.
//
// Complexity:
//   - Time O(|members(node)|), Space O(1).
//
// Notes:
//   - This helper is intentionally simple; membership arrays are already explicit.
//
// AI-Hints:
//   - Do not replace this with inBlossom[vertex]==node when checking inactive child ownership.
func (e *blossomEngine) nodeContainsVertex(node int, vertex int) bool {
	if vertex < 0 || vertex >= e.problem.n {
		return false
	}
	if node < 0 || node >= len(e.members) {
		return false
	}

	for _, member := range e.members[node] {
		if member == vertex {
			return true
		}
	}

	return false
}

// orientEdgeForNodes returns the original endpoints of edgeID as owned by a and b.
// It is the canonical bridge from dense edge endpoints to contracted top-level
// blossom ownership.
//
// Implementation:
//   - Stage 1: Validate edge and active node IDs.
//   - Stage 2: Read the dense edge endpoints.
//   - Stage 3: Test both endpoint orientations against node memberships.
//   - Stage 4: Return endpoints in the same order as nodes a and b.
//
// Behavior highlights:
//   - Does not mutate engine state.
//   - Accepts singleton and contracted nodes.
//   - Rejects edges whose endpoints are not split across the requested nodes.
//
// Inputs:
//   - edgeID: dense local edge ID.
//   - a: first active top-level node.
//   - b: second active top-level node.
//
// Returns:
//   - int: original local vertex owned by a.
//   - int: original local vertex owned by b.
//   - error: nil when endpoint ownership is valid.
//
// Errors:
//   - ErrInvalidMatching for invalid edge ID, inactive nodes, same-node ownership,
//     or endpoint membership mismatch.
//
// Determinism:
//   - Checks the two possible orientations in fixed order.
//
// Complexity:
//   - Time O(|members(a)|+|members(b)|), Space O(1).
//
// Notes:
//   - This helper is used by event creation and cycle metadata construction.
//
// AI-Hints:
//   - Do not guess orientation from edge.u<edge.v.
//   - Do not use this helper with original matrix vertex IDs.
func (e *blossomEngine) orientEdgeForNodes(edgeID int, a int, b int) (int, int, error) {
	if edgeID < 0 || edgeID >= len(e.edges) {
		return noVertex, noVertex, ErrInvalidMatching
	}
	if a < 0 || a >= len(e.active) || b < 0 || b >= len(e.active) {
		return noVertex, noVertex, ErrInvalidMatching
	}
	if !e.active[a] || !e.active[b] || a == b {
		return noVertex, noVertex, ErrInvalidMatching
	}

	edge := e.edges[edgeID]

	if e.nodeContainsVertex(a, edge.u) && e.nodeContainsVertex(b, edge.v) {
		return edge.u, edge.v, nil
	}
	if e.nodeContainsVertex(a, edge.v) && e.nodeContainsVertex(b, edge.u) {
		return edge.v, edge.u, nil
	}

	return noVertex, noVertex, ErrInvalidMatching
}

// scanTightEdges scans queued outer nodes until it finds one applicable tight-edge event.
// If the FIFO queue is exhausted, the caller must select/apply a dual delta and rewind scanning.
//
// Implementation:
//   - Stage 1: Consume queued nodes from head.
//   - Stage 2: Skip nodes that are no longer active outer nodes.
//   - Stage 3: Scan tight edges incident to the outer node.
//   - Stage 4: Return the first grow, shrink, augment, or expand event found.
//
// Behavior highlights:
//   - Does not apply events.
//   - Does not mutate mate[] or dual values.
//   - Advances the queue cursor.
//   - Preserves deterministic event discovery order.
//
// Inputs:
//   - None; reads queue/head and forest state.
//
// Returns:
//   - blossomEvent: discovered event when ok is true.
//   - bool: true when an event was found.
//   - error: nil when scanning succeeds.
//
// Errors:
//   - ErrInvalidMatching from malformed ownership or edge orientation during node scans.
//
// Determinism:
//   - FIFO queue order plus deterministic incident edge order.
//
// Complexity:
//   - Time O(number of scanned incident edges), Space O(1).
//
// Notes:
//   - Caller owns dual movement when ok is false.
//
// AI-Hints:
//   - Do not call applyDelta inside this method.
func (e *blossomEngine) scanTightEdges() (blossomEvent, bool, error) {
	for e.head < len(e.queue) {
		node := e.queue[e.head]
		e.head++

		if node < 0 || node >= len(e.active) || !e.active[node] || e.label[node] != blossomOuter {
			continue
		}

		event, ok, err := e.scanOuterNodeTightEdges(node)
		if err != nil || ok {
			return event, ok, err
		}
	}

	return blossomEvent{kind: eventNone, edge: noEdge, a: noNode, b: noNode}, false, nil
}

// scanOuterNodeTightEdges scans all dense edges incident to original members of one outer node.
// Tight edges to unlabeled nodes grow the alternating tree; tight edges to outer nodes either
// shrink a same-tree odd cycle or augment across two different roots.
//
// Implementation:
//   - Stage 1: Validate active outer node.
//   - Stage 2: Iterate represented original members.
//   - Stage 3: Iterate incident dense edges in deterministic edge-ID order.
//   - Stage 4: Resolve opposite endpoint top-level ownership.
//   - Stage 5: Classify tight target labels into grow, shrink, or augment events.
//
// Behavior highlights:
//   - Does not mutate engine state.
//   - Skips non-tight edges.
//   - Preserves endpoint ownership in returned events.
//   - Uses treeRoot to distinguish shrink from augment.
//
// Inputs:
//   - node: active outer top-level node.
//
// Returns:
//   - blossomEvent: first applicable event when ok is true.
//   - bool: true when an event was found.
//   - error: nil when scanning succeeds.
//
// Errors:
//   - ErrInvalidMatching for invalid node, invalid ownership, or invalid edge orientation.
//
// Determinism:
//   - Member order and incident edge order are fixed.
//
// Complexity:
//   - Time O(|members(node)| * k * B*m) in dense worst case through tight/slack checks,
//     Space O(1).
//
// Notes:
//   - Same-tree outer/outer is never an augmenting event; it must shrink.
//
// AI-Hints:
//   - Do not infer endpoints after event creation; preserve aVertex/bVertex.
func (e *blossomEngine) scanOuterNodeTightEdges(node int) (blossomEvent, bool, error) {
	if node < 0 || node >= len(e.members) || !e.active[node] {
		return blossomEvent{}, false, ErrInvalidMatching
	}

	for _, original := range e.members[node] {
		if original < 0 || original >= e.problem.n {
			return blossomEvent{}, false, ErrInvalidMatching
		}

		for _, edgeID := range e.incident[original] {
			if edgeID < 0 || edgeID >= len(e.edges) {
				return blossomEvent{}, false, ErrInvalidMatching
			}
			if !e.isTight(edgeID) {
				continue
			}

			edge := e.edges[edgeID]
			other := edge.v
			if other == original {
				other = edge.u
			}

			otherTop, err := e.topNodeOfVertex(other)
			if err != nil {
				return blossomEvent{}, false, err
			}
			if otherTop == node {
				continue
			}

			aVertex, bVertex, err := e.orientEdgeForNodes(edgeID, node, otherTop)
			if err != nil {
				return blossomEvent{}, false, err
			}

			switch e.label[otherTop] {
			case blossomUnlabeled:
				return blossomEvent{
					kind:    eventGrow,
					edge:    edgeID,
					a:       node,
					b:       otherTop,
					aVertex: aVertex,
					bVertex: bVertex,
				}, true, nil

			case blossomOuter:
				if e.treeRoot[node] == e.treeRoot[otherTop] {
					return blossomEvent{
						kind:    eventShrink,
						edge:    edgeID,
						a:       node,
						b:       otherTop,
						aVertex: aVertex,
						bVertex: bVertex,
					}, true, nil
				}

				return blossomEvent{
					kind:    eventAugment,
					edge:    edgeID,
					a:       node,
					b:       otherTop,
					aVertex: aVertex,
					bVertex: bVertex,
				}, true, nil

			case blossomInner:
				continue

			default:
				return blossomEvent{}, false, ErrInvalidMatching
			}
		}
	}

	return blossomEvent{}, false, nil
}

// applyBlossomEvent routes one tight-edge event and reports whether augmentation completed.
// Structural events update the active search state and return done=false; only an actual
// augmenting-path flip returns done=true.
//
// Implementation:
//   - Stage 1: Dispatch grow events to alternating-forest label extension.
//   - Stage 2: Dispatch shrink events to odd-cycle contraction.
//   - Stage 3: Dispatch augment events to lifted path mutation.
//   - Stage 4: Dispatch expand events to forest-preserving blossom expansion.
//   - Stage 5: Reject unknown events as invalid matching state.
//
// Behavior highlights:
//   - Grow, shrink, and expand are intermediate search events.
//   - Augment is the only terminal event for findAndApplyAugmentation.
//   - The function does not reset the forest.
//
// Inputs:
//   - event: classified tight-edge or expansion event.
//
// Returns:
//   - bool: true only when one augmenting path was applied.
//   - error: nil on valid event processing.
//
// Errors:
//   - ErrInvalidMatching for unknown event kinds or malformed event payloads.
//   - Event-specific errors from grow, shrink, augment, or expansion.
//
// Determinism:
//   - Dispatch is a fixed switch on event.kind.
//
// Complexity:
//   - Event dispatch is O(1), excluding delegated event handling.
//
// Notes:
//   - stats.Augmentations is intentionally not changed here.
//
// AI-Hints:
//   - Do not return done=true from eventGrow, eventShrink, or eventExpand.
//   - Do not swallow shrink/expand errors; they protect matching correctness.
func (e *blossomEngine) applyBlossomEvent(event blossomEvent) (bool, error) {
	switch event.kind {
	case eventGrow:
		if err := e.grow(event); err != nil {
			return false, fmt.Errorf("event grow %+v: %w", event, err)
		}
		return false, nil

	case eventShrink:
		if err := e.shrink(event); err != nil {
			return false, fmt.Errorf("event shrink %+v: %w", event, err)
		}
		return false, nil

	case eventAugment:
		if err := e.augment(event); err != nil {
			return false, fmt.Errorf("event augment %+v: %w", event, err)
		}
		return true, nil

	case eventExpand:
		if err := e.expand(event.a); err != nil {
			return false, fmt.Errorf("event expand node=%d event=%+v: %w", event.a, event, err)
		}
		return false, nil

	case eventNone:
		return false, fmt.Errorf("event none: %w", ErrInvalidMatching)

	default:
		return false, fmt.Errorf("unknown event kind %d event=%+v: %w", event.kind, event, ErrInvalidMatching)
	}
}

// grow extends an alternating tree through a tight edge to an unlabeled top-level node.
// The entered original vertex becomes the inner side of the grow step; its committed
// mate determines the next outer node.
//
// Implementation:
//   - Stage 1: Validate grow event and endpoint ownership.
//   - Stage 2: Label the reached top-level node as inner.
//   - Stage 3: Follow the matched edge of event.bVertex, not the blossom base.
//   - Stage 4: Label the matched partner's top-level node as outer and enqueue it.
//
// Behavior highlights:
//   - Works for singleton vertices and contracted blossoms.
//   - Does not assume that the blossom base is the entered endpoint.
//   - Preserves endpoint ownership for later path lifting.
//
// Inputs:
//   - event: eventGrow from an outer node to an unlabeled top-level node.
//
// Returns:
//   - error: nil after forest extension.
//
// Errors:
//   - ErrInvalidMatching for malformed event, missing mate edge, corrupt endpoint ownership,
//     or inconsistent active top-level ownership.
//
// Determinism:
//   - Fixed event endpoint ownership and one committed mate edge.
//
// Complexity:
//   - Time O(|members(event.a)|+|members(event.b)|), Space amortized O(1).
//
// Notes:
//   - This method does not mutate mate[].
//
// AI-Hints:
//   - Do not use e.base[toUnlabeled] as the matched endpoint.
//   - The base is for blossom contraction semantics, not for every grow entry.
func (e *blossomEngine) grow(event blossomEvent) error {
	if event.kind != eventGrow {
		return ErrInvalidMatching
	}

	fromOuter := event.a
	toUnlabeled := event.b

	if fromOuter < 0 || fromOuter >= len(e.active) || toUnlabeled < 0 || toUnlabeled >= len(e.active) {
		return ErrInvalidMatching
	}
	if !e.active[fromOuter] || !e.active[toUnlabeled] {
		return ErrInvalidMatching
	}
	if e.label[fromOuter] != blossomOuter || e.label[toUnlabeled] != blossomUnlabeled {
		return ErrInvalidMatching
	}
	if !e.nodeContainsVertex(fromOuter, event.aVertex) || !e.nodeContainsVertex(toUnlabeled, event.bVertex) {
		return ErrInvalidMatching
	}

	entry, err := e.growMateEndpoint(toUnlabeled, event.bVertex)
	if err != nil {
		return err
	}

	mate := e.mate[entry]
	mateEdge := e.mateEdge[entry]
	if mate == noVertex || mateEdge == noEdge {
		return ErrInvalidMatching
	}

	mateTop, err := e.topNodeOfVertex(mate)
	if err != nil {
		return err
	}
	if mateTop == toUnlabeled {
		return ErrInvalidMatching
	}
	if e.label[mateTop] != blossomUnlabeled {
		return ErrInvalidMatching
	}

	e.label[toUnlabeled] = blossomInner
	e.labelEdge[toUnlabeled] = event.edge
	e.parent[toUnlabeled] = fromOuter
	e.treeRoot[toUnlabeled] = e.treeRoot[fromOuter]

	e.label[mateTop] = blossomOuter
	e.labelEdge[mateTop] = mateEdge
	e.parent[mateTop] = toUnlabeled
	e.treeRoot[mateTop] = e.treeRoot[fromOuter]
	e.queue = append(e.queue, mateTop)

	return nil
}

// growMateEndpoint returns the original vertex whose committed mate represents the matched
// continuation of an unlabeled top-level node during grow.
//
// For singleton nodes, the entering endpoint is the only possible continuation endpoint.
// For contracted blossoms, the blossom base is the pseudo-node interface; a non-base entry
// vertex may be internally matched and must not be used as the top-level continuation.
//
// Implementation:
//   - Stage 1: Validate active target node.
//   - Stage 2: Validate that entryVertex belongs to the node.
//   - Stage 3: Return entryVertex for singleton nodes.
//   - Stage 4: Validate and return base[node] for contracted blossoms.
//
// Behavior highlights:
//   - Does not mutate engine state.
//   - Separates edge-entry endpoint from pseudo-node matching interface.
//   - Protects grow from following internal blossom matches.
//
// Inputs:
//   - node: unlabeled active top-level target node.
//   - entryVertex: original endpoint where the tight grow edge enters node.
//
// Returns:
//   - int: original vertex whose mate defines the matched continuation.
//   - error: nil when a valid continuation endpoint exists.
//
// Errors:
//   - ErrInvalidMatching for invalid nodes, missing ownership, invalid base,
//     or base not contained in node.
//
// Determinism:
//   - Pure indexed/membership checks.
//
// Complexity:
//   - Time O(|members(node)|), Space O(1).
//
// Notes:
//   - refreshAllocatedBlossomBases keeps base[node] current after augmentation.
//
// AI-Hints:
//   - Do not use entryVertex for contracted blossoms.
func (e *blossomEngine) growMateEndpoint(node int, entryVertex int) (int, error) {
	if node < 0 || node >= len(e.active) || !e.active[node] {
		return noVertex, ErrInvalidMatching
	}
	if !e.nodeContainsVertex(node, entryVertex) {
		return noVertex, ErrInvalidMatching
	}

	if node < e.problem.n {
		return entryVertex, nil
	}

	base := e.base[node]
	if base < 0 || base >= e.problem.n || !e.nodeContainsVertex(node, base) {
		return noVertex, ErrInvalidMatching
	}

	return base, nil
}

// augmentationStepLimit returns a deterministic defensive upper bound for one augmenting
// search phase. Correct Blossom progress should always discover an event or apply a
// valid dual movement before the bound is reached; hitting the bound indicates an internal
// progress invariant failure rather than a normal no-solution condition.
//
// Implementation:
//   - Stage 1: Scale the limit by dense edge count.
//   - Stage 2: Scale the limit by allocated node capacity.
//   - Stage 3: Return the larger bound.
//
// Behavior highlights:
//   - Prevents infinite loops from future forest/dual regressions.
//   - Does not participate in optimality decisions.
//   - Should never fire for valid complete even-order matching problems.
//
// Inputs:
//   - None; reads edge and node-capacity sizes.
//
// Returns:
//   - int: maximum scan/delta/event iterations for one augmentation phase.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pure arithmetic over fixed engine sizes.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Reaching this bound must be treated as ErrInvalidMatching.
//
// AI-Hints:
//   - Do not replace this with an unbounded for loop in production.
//   - Do not silently increase the bound to hide missing-progress bugs.
func (e *blossomEngine) augmentationStepLimit() int {
	limit := len(e.edges) * 32
	minimum := len(e.active) * 8
	if limit < minimum {
		limit = minimum
	}
	if limit < 1 {
		limit = 1
	}

	return limit
}
