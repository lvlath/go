// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package match realizes endpoint-aware augmenting paths for dense Blossom search.
// Search events identify top-level nodes, but matching mutation requires original
// dense edge IDs. This file lifts contracted-blossom paths back into original edges.
//
// Responsibility:
//   - Represent oriented top-level path steps with endpoint ownership.
//   - Compose root-to-root augmenting paths in deterministic orientation.
//   - Lift paths through contracted blossom cycles.
//   - Preserve alternating parity before mate[] mutation.
//
// Boundaries:
//   - This file does not mutate mate[].
//   - This file does not choose dual deltas.
//   - This file reads cycle metadata but does not allocate or destroy blossoms.
//
// AI-Hints:
//   - Do not reduce path steps to bare edge IDs before lifting.
//   - Do not choose internal blossom direction by shortest path; parity decides validity.
//   - Do not infer endpoint ownership from edge IDs alone.
package match

// blossomPathStep stores one oriented edge traversal between top-level nodes.
// It is the unit consumed by augmenting-path lifting before mate[] is mutated.
//
// Implementation:
//   - Stage 1: pathToRoot emits steps from an outer node toward its root.
//   - Stage 2: composeAugmentingPath reverses/joins steps into root-to-root order.
//   - Stage 3: liftAugmentingPath expands steps that pass through contracted blossoms.
//
// Behavior highlights:
//   - edge identifies the dense original edge.
//   - fromNode/toNode are top-level nodes at the time the path is composed.
//   - fromVertex/toVertex are original local endpoints owned by those nodes.
//
// Inputs:
//   - Built from alternating forest parent links and event endpoint ownership.
//
// Returns:
//   - Consumed by liftAugmentingPath.
//
// Errors:
//   - Invalid ownership is rejected by step constructors and validation helpers.
//
// Determinism:
//   - Step order follows deterministic parent links and event order.
//
// Complexity:
//   - Storage O(1).
//
// Notes:
//   - This type prevents augment from losing endpoint ownership.
//
// AI-Hints:
//   - Do not collapse this back to []int edge IDs before lifting is complete.
type blossomPathStep struct {
	// edge is the dense local edge traversed by this step.
	edge int

	// fromNode is the active top-level node where this oriented step starts.
	fromNode int

	// toNode is the active top-level node where this oriented step ends.
	toNode int

	// fromVertex is the original local endpoint owned by fromNode.
	fromVertex int

	// toVertex is the original local endpoint owned by toNode.
	toVertex int
}

// reversePathStep returns the same dense edge traversal in the opposite orientation.
// It swaps both top-level nodes and original endpoint vertices, preserving endpoint ownership.
//
// Implementation:
//   - Stage 1: Copy the dense edge ID.
//   - Stage 2: Swap fromNode/toNode.
//   - Stage 3: Swap fromVertex/toVertex.
//
// Behavior highlights:
//   - Pure value transformation.
//   - Does not validate ownership.
//   - Used when composeAugmentingPath turns a node-to-root chain into root-to-node order.
//
// Inputs:
//   - step: oriented top-level path step.
//
// Returns:
//   - blossomPathStep: reversed orientation.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pure field swap.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Validation belongs to makePathStep and verifyPathStepContinuity.
//
// AI-Hints:
//   - Do not reverse only nodes; vertex endpoints must be reversed too.
func reversePathStep(step blossomPathStep) blossomPathStep {
	return blossomPathStep{
		edge:       step.edge,
		fromNode:   step.toNode,
		toNode:     step.fromNode,
		fromVertex: step.toVertex,
		toVertex:   step.fromVertex,
	}
}

// makePathStep constructs one validated oriented path step between two active top-level nodes.
// It resolves original endpoint ownership immediately so later Blossom lifting does not need
// to guess which local vertices an edge used.
//
// Implementation:
//   - Stage 1: Orient edgeID between fromNode and toNode.
//   - Stage 2: Store top-level node orientation.
//   - Stage 3: Store original endpoint ownership.
//   - Stage 4: Return the assembled blossomPathStep.
//
// Behavior highlights:
//   - Does not mutate engine state.
//   - Rejects edges that do not connect the requested top-level nodes.
//   - Preserves information needed by contracted-blossom lifting.
//
// Inputs:
//   - edgeID: dense local edge identifier.
//   - fromNode: active top-level source node.
//   - toNode: active top-level target node.
//
// Returns:
//   - blossomPathStep: validated oriented step.
//   - error: nil when endpoint ownership is valid.
//
// Errors:
//   - ErrInvalidMatching from orientEdgeForNodes for invalid edge or ownership mismatch.
//
// Determinism:
//   - Pure deterministic endpoint orientation.
//
// Complexity:
//   - Time O(|members(fromNode)| + |members(toNode)|), Space O(1).
//
// Notes:
//   - The returned step is still top-level and may require lifting.
//
// AI-Hints:
//   - Do not derive fromVertex/toVertex later from edge ID alone.
func (e *blossomEngine) makePathStep(edgeID int, fromNode int, toNode int) (blossomPathStep, error) {
	fromVertex, toVertex, err := e.orientEdgeForNodes(edgeID, fromNode, toNode)
	if err != nil {
		return blossomPathStep{}, err
	}

	return blossomPathStep{
		edge:       edgeID,
		fromNode:   fromNode,
		toNode:     toNode,
		fromVertex: fromVertex,
		toVertex:   toVertex,
	}, nil
}

// pathToRoot returns oriented path steps from an outer node toward its alternating-tree root.
// It records endpoint ownership at every parent edge so contracted blossoms can later be lifted.
//
// Implementation:
//   - Stage 1: Validate active outer start node.
//   - Stage 2: Follow parent links until the root.
//   - Stage 3: Convert every labelEdge into an oriented blossomPathStep.
//   - Stage 4: Stop when the current node is its own tree root.
//
// Behavior highlights:
//   - Does not mutate engine state.
//   - Does not expand blossoms.
//   - Returns node/endpoint-aware steps, not bare edge IDs.
//
// Inputs:
//   - node: active outer top-level node.
//
// Returns:
//   - []blossomPathStep: path from node toward root.
//   - error: nil on valid forest chain.
//
// Errors:
//   - ErrInvalidMatching for inactive nodes, non-outer start node,
//     missing label edges, or corrupt parent links.
//
// Determinism:
//   - Follows exactly one parent chain.
//
// Complexity:
//   - Time O(k*m), Space O(k), where m is average membership scan cost.
//
// Notes:
//   - composeAugmentingPath owns final orientation.
//
// AI-Hints:
//   - Do not return []int here; augment needs endpoint ownership.
func (e *blossomEngine) pathToRoot(node int) ([]blossomPathStep, error) {
	if node < 0 || node >= len(e.active) || !e.active[node] {
		return nil, ErrInvalidMatching
	}
	if e.label[node] != blossomOuter {
		return nil, ErrInvalidMatching
	}

	steps := make([]blossomPathStep, 0, e.problem.n)

	for current := node; current != noNode; current = e.parent[current] {
		if current < 0 || current >= len(e.active) {
			return nil, ErrInvalidMatching
		}

		if e.treeRoot[current] == current {
			break
		}

		parent := e.parent[current]
		if parent == noNode || e.labelEdge[current] == noEdge {
			return nil, ErrInvalidMatching
		}

		step, err := e.makePathStep(e.labelEdge[current], current, parent)
		if err != nil {
			return nil, err
		}

		steps = append(steps, step)
	}

	return steps, nil
}

// composeAugmentingPath creates a root-to-root endpoint-aware augmenting path.
// pathToRoot returns node-to-root paths; therefore the left side must be reversed,
// while the right side is already in the direction needed after the bridge.
//
// Implementation:
//   - Stage 1: Validate the bridge step.
//   - Stage 2: Reverse and reorient the left node-to-root path into root-to-node order.
//   - Stage 3: Append the bridge from left outer node to right outer node.
//   - Stage 4: Append the right node-to-root path unchanged.
//   - Stage 5: Validate endpoint continuity before lifting.
//
// Behavior highlights:
//   - Does not mutate matching state.
//   - Produces the only orientation in which first/last edges can be unmatched.
//   - Keeps endpoint ownership attached to every step.
//
// Inputs:
//   - left: path from event.a toward its root.
//   - bridge: tight edge from event.a to event.b.
//   - right: path from event.b toward its root.
//
// Returns:
//   - []blossomPathStep: root-to-root endpoint-aware path.
//   - error: nil when path continuity is valid.
//
// Errors:
//   - ErrInvalidMatching for invalid bridge, invalid step orientation, or broken continuity.
//
// Determinism:
//   - Fixed left reversal and right append order.
//
// Complexity:
//   - Time O(len(left)+len(right)), Space O(len(left)+len(right)+1).
//
// Notes:
//   - Do not pass this path directly to flipAugmentingPath; it still must be lifted.
//
// AI-Hints:
//   - pathToRoot returns node->root, not root->node.
//   - Reversing the right side here is a common bug.
func (e *blossomEngine) composeAugmentingPath(
	left []blossomPathStep,
	bridge blossomPathStep,
	right []blossomPathStep,
) ([]blossomPathStep, error) {
	if bridge.edge < 0 || bridge.edge >= len(e.edges) {
		return nil, ErrInvalidMatching
	}

	out := make([]blossomPathStep, 0, len(left)+1+len(right))

	for index := len(left) - 1; index >= 0; index-- {
		out = append(out, reversePathStep(left[index]))
	}

	out = append(out, bridge)
	out = append(out, right...)

	if err := e.verifyPathStepContinuity(out); err != nil {
		return nil, err
	}

	return out, nil
}

// verifyPathStepContinuity checks that consecutive oriented top-level path steps meet
// at the same node. Singleton joins must also meet at the same original vertex;
// contracted blossom joins may use different original vertices because lifting reconstructs
// the internal route.
//
// Implementation:
//   - Stage 1: Scan adjacent step pairs.
//   - Stage 2: Require prev.toNode == next.fromNode.
//   - Stage 3: For singleton shared nodes, require prev.toVertex == next.fromVertex.
//   - Stage 4: Allow contracted shared nodes to defer vertex continuity to lifting.
//
// Behavior highlights:
//   - Does not mutate engine state.
//   - Rejects impossible singleton path joins early.
//   - Preserves contracted-blossom flexibility.
//
// Inputs:
//   - steps: root-to-root oriented top-level path.
//
// Returns:
//   - error: nil when top-level continuity is valid.
//
// Errors:
//   - ErrInvalidMatching for broken node continuity or singleton endpoint mismatch.
//
// Determinism:
//   - Fixed left-to-right scan.
//
// Complexity:
//   - Time O(len(steps)), Space O(1).
//
// Notes:
//   - This is not full augmenting-path validation; it runs before lifting.
//
// AI-Hints:
//   - Do not require original endpoint equality for contracted blossom nodes.
func (e *blossomEngine) verifyPathStepContinuity(steps []blossomPathStep) error {
	for index := 1; index < len(steps); index++ {
		prev := steps[index-1]
		next := steps[index]

		if prev.toNode != next.fromNode {
			return ErrInvalidMatching
		}

		if prev.toNode < e.problem.n && prev.toVertex != next.fromVertex {
			return ErrInvalidMatching
		}
	}

	return nil
}

// liftAugmentingPathChoices expands an endpoint-aware top-level augmenting path into
// every deterministic original-edge candidate that satisfies strict augmenting-path
// alternation under the current committed matching.
//
// Mathematical contract:
//   - Local internal blossom choices are not enough.
//   - A lifted candidate is eligible only if verifyAugmentingEdgeSequence accepts it.
//   - Persistent blossom-base preservation is checked later by augment through a
//     transactional flip + refreshAllocatedBlossomBases attempt.
//
// Implementation:
//   - Stage 1: Build the same candidate set as liftAugmentingPath.
//   - Stage 2: Keep every candidate accepted by verifyAugmentingEdgeSequence.
//   - Stage 3: Return all candidates in deterministic order.
//
// Behavior highlights:
//   - Does not mutate mate[], mateEdge[], base[], labels, or duals.
//   - Preserves alternative nested blossom routes.
//   - Does not stop at the first locally valid candidate.
//
// Inputs:
//   - steps: root-to-root oriented top-level path.
//
// Returns:
//   - [][]int: fully alternating lifted edge candidates.
//   - error: nil when at least one candidate is available.
//
// Errors:
//   - ErrInvalidMatching for malformed steps, broken ownership, exhausted candidates,
//     or absence of any valid augmenting edge sequence.
//
// Determinism:
//   - Candidate order follows liftAugmentingPath construction order.
//
// Complexity:
//   - Worst-case O(2^b * L), where b is crossed/nested blossoms and L is lifted path length.
//
// AI-Hints:
//   - Do not mutate matching state here.
//   - Do not return only the first candidate.
//   - Do not replace refreshAllocatedBlossomBases; this method cannot prove persistent bases.
func (e *blossomEngine) liftAugmentingPathChoices(steps []blossomPathStep) ([][]int, error) {
	if len(steps) == 0 {
		return nil, ErrInvalidMatching
	}

	candidates := [][]int{{}}

	first := steps[0]
	if first.fromNode >= e.problem.n {
		choices, err := e.liftThroughBlossomChoices(first.fromNode, e.base[first.fromNode], first.fromVertex)
		if err != nil {
			return nil, err
		}

		candidates = e.extendAlternatingCandidates(candidates, choices)
		if len(candidates) == 0 {
			return nil, ErrInvalidMatching
		}
	}

	for index, step := range steps {
		if step.edge < 0 || step.edge >= len(e.edges) {
			return nil, ErrInvalidMatching
		}

		if index > 0 {
			prev := steps[index-1]
			if prev.toNode != step.fromNode {
				return nil, ErrInvalidMatching
			}

			if prev.toVertex != step.fromVertex {
				if step.fromNode < e.problem.n {
					return nil, ErrInvalidMatching
				}

				choices, err := e.liftThroughBlossomChoices(step.fromNode, prev.toVertex, step.fromVertex)
				if err != nil {
					return nil, err
				}

				candidates = e.extendAlternatingCandidates(candidates, choices)
				if len(candidates) == 0 {
					return nil, ErrInvalidMatching
				}
			}
		}

		candidates = e.extendAlternatingCandidates(candidates, [][]int{{step.edge}})
		if len(candidates) == 0 {
			return nil, ErrInvalidMatching
		}
	}

	last := steps[len(steps)-1]
	if last.toNode >= e.problem.n {
		choices, err := e.liftThroughBlossomChoices(last.toNode, last.toVertex, e.base[last.toNode])
		if err != nil {
			return nil, err
		}

		candidates = e.extendAlternatingCandidates(candidates, choices)
		if len(candidates) == 0 {
			return nil, ErrInvalidMatching
		}
	}

	valid := make([][]int, 0, len(candidates))
	for _, candidate := range candidates {
		if err := e.verifyAugmentingEdgeSequence(candidate); err == nil {
			next := append([]int(nil), candidate...)
			valid = append(valid, next)
		}
	}

	if len(valid) == 0 {
		return nil, ErrInvalidMatching
	}

	return valid, nil
}

// liftThroughBlossomChoices returns all deterministic internal edge sequences between two
// original vertices represented by one contracted blossom.
//
// Implementation:
//   - Stage 1: Validate the contracted blossom and same-vertex neutral route.
//   - Stage 2: Locate entry and exit child indices.
//   - Stage 3: Build every forward-direction recursive lift candidate.
//   - Stage 4: Build every backward-direction recursive lift candidate.
//   - Stage 5: Keep only candidates that alternate locally.
//
// Behavior highlights:
//   - Does not mutate mate[], base[], labels, or duals.
//   - Preserves all nested child blossom choices.
//   - Returns candidates in deterministic forward-then-backward order.
//   - Does not decide full augmenting-path validity; liftAugmentingPath validates the whole path.
//
// Inputs:
//   - node: contracted blossom node.
//   - entryVertex: original local vertex where the route enters node.
//   - exitVertex: original local vertex where the route leaves node.
//
// Returns:
//   - [][]int: candidate internal dense edge sequences.
//   - error: nil when at least one structurally valid local route exists.
//
// Errors:
//   - ErrInvalidMatching for singleton misuse, missing cycle metadata, missing ownership,
//     or absence of any locally alternating internal route.
//
// Determinism:
//   - Forward direction is considered before backward direction.
//   - Recursive child choices preserve their deterministic order.
//
// Complexity:
//   - Worst-case O(2^b * L) for nested blossoms crossed by this local route.
//   - Space O(2^b * L).
//
// Notes:
//   - This method deliberately returns all viable local choices.
//   - The caller filters them against surrounding external edges.
//
// AI-Hints:
//   - Do not collapse recursive child choices to choices[0] here.
//   - Do not choose by shortest path; alternation decides validity.
func (e *blossomEngine) liftThroughBlossomChoices(
	node int,
	entryVertex int,
	exitVertex int,
) ([][]int, error) {
	if entryVertex == exitVertex {
		return [][]int{{}}, nil
	}
	if node < e.problem.n {
		return nil, ErrInvalidMatching
	}
	if node < 0 || node >= len(e.cycles) || len(e.cycles[node]) == 0 {
		return nil, ErrInvalidMatching
	}

	fromChild, err := e.cycleChildIndexContainingVertex(node, entryVertex)
	if err != nil {
		return nil, err
	}

	toChild, err := e.cycleChildIndexContainingVertex(node, exitVertex)
	if err != nil {
		return nil, err
	}

	choices := make([][]int, 0, 4)

	forward, err := e.cyclePathBetweenDirection(node, fromChild, toChild, true, entryVertex, exitVertex)
	if err == nil {
		for _, candidate := range forward {
			if e.edgeSequenceAlternates(candidate) {
				choices = append(choices, candidate)
			}
		}
	}

	backward, err := e.cyclePathBetweenDirection(node, fromChild, toChild, false, entryVertex, exitVertex)
	if err == nil {
		for _, candidate := range backward {
			if e.edgeSequenceAlternates(candidate) {
				choices = append(choices, candidate)
			}
		}
	}

	if len(choices) == 0 {
		return nil, ErrInvalidMatching
	}

	return choices, nil
}

// cycleChildIndexContainingVertex finds the cycle child that owns an original local vertex
// inside a contracted blossom. Ownership is checked through members[] so inactive nested
// children remain searchable.
//
// Implementation:
//   - Stage 1: Validate contracted blossom node bounds.
//   - Stage 2: Scan cycle steps in stored order.
//   - Stage 3: Return the first child whose members contain vertex.
//   - Stage 4: Reject when no child owns the vertex.
//
// Behavior highlights:
//   - Does not mutate engine state.
//   - Works for singleton and nested contracted child nodes.
//   - Does not rely on inBlossom, because nested children may be inactive.
//
// Inputs:
//   - node: contracted blossom node.
//   - vertex: original local vertex to locate.
//
// Returns:
//   - int: child cycle index containing vertex.
//   - error: nil when ownership is found.
//
// Errors:
//   - ErrInvalidMatching for invalid node or missing child ownership.
//
// Determinism:
//   - Fixed stored cycle order.
//
// Complexity:
//   - Time O(c*m), Space O(1), where c is cycle length and m is average child membership.
//
// Notes:
//   - This method is used by lifting and expansion, not top-level ownership.
//
// AI-Hints:
//   - Do not replace this with inBlossom[vertex] while node is nested.
func (e *blossomEngine) cycleChildIndexContainingVertex(node int, vertex int) (int, error) {
	if node < e.problem.n || node >= len(e.cycles) {
		return -1, ErrInvalidMatching
	}

	for index, step := range e.cycles[node] {
		if e.nodeContainsVertex(step.node, vertex) {
			return index, nil
		}
	}

	return -1, ErrInvalidMatching
}

// cyclePathBetweenDirection builds every candidate lift sequence along one directed walk
// of a contracted blossom cycle. It preserves recursive alternatives from nested child
// blossoms instead of collapsing them to the first local route.
//
// Implementation:
//   - Stage 1: Validate contracted blossom metadata and child indices.
//   - Stage 2: Start with one empty candidate.
//   - Stage 3: Walk forward or backward through adjacent cycle children.
//   - Stage 4: Append every valid internal child route choice.
//   - Stage 5: Append the boundary edge to the next child.
//   - Stage 6: Lift inside the final child to exitVertex.
//
// Behavior highlights:
//   - Does not mutate engine state.
//   - Preserves nested Blossom alternatives as [][]int.
//   - Performs local alternation pruning after every appended segment.
//   - Supports neutral empty child routes.
//
// Inputs:
//   - node: contracted blossom node.
//   - fromChild: starting child index.
//   - toChild: ending child index.
//   - forward: true for increasing cycle direction, false for decreasing direction.
//   - entryVertex: original vertex where traversal enters fromChild.
//   - exitVertex: original vertex where traversal leaves toChild.
//
// Returns:
//   - [][]int: candidate dense edge sequences for this directed route.
//   - error: nil when at least one candidate survives local pruning.
//
// Errors:
//   - ErrInvalidMatching for malformed cycle metadata, invalid indices,
//     invalid child ownership, or exhausted local candidates.
//
// Determinism:
//   - Direction is fixed by the forward flag; nested choices preserve their own order.
//
// Complexity:
//   - Worst-case O(2^b * L), Space O(2^b * L), where b is nested blossoms crossed.
//
// Notes:
//   - The caller still validates full augmenting-path endpoints and parity.
//
// AI-Hints:
//   - Do not call liftThroughBlossom wrapper here; use liftInsideChildChoices.
func (e *blossomEngine) cyclePathBetweenDirection(
	node int,
	fromChild int,
	toChild int,
	forward bool,
	entryVertex int,
	exitVertex int,
) ([][]int, error) {
	if node < e.problem.n || node >= len(e.cycles) {
		return nil, ErrInvalidMatching
	}

	cycle := e.cycles[node]
	if len(cycle) < 3 || (len(cycle)&1) == 0 {
		return nil, ErrInvalidMatching
	}
	if fromChild < 0 || fromChild >= len(cycle) || toChild < 0 || toChild >= len(cycle) {
		return nil, ErrInvalidMatching
	}

	candidates := [][]int{{}}
	currentIndex := fromChild
	currentVertex := entryVertex

	for currentIndex != toChild {
		currentNode := cycle[currentIndex].node

		if forward {
			step := cycle[currentIndex]

			internalChoices, err := e.liftInsideChildChoices(currentNode, currentVertex, step.vertexToNext)
			if err != nil {
				return nil, err
			}
			candidates = e.extendAlternatingCandidates(candidates, internalChoices)
			if len(candidates) == 0 {
				return nil, ErrInvalidMatching
			}

			candidates = e.extendAlternatingCandidates(candidates, [][]int{{step.edgeToNext}})
			if len(candidates) == 0 {
				return nil, ErrInvalidMatching
			}

			currentVertex = step.nextVertex
			currentIndex++
			if currentIndex == len(cycle) {
				currentIndex = 0
			}

			continue
		}

		previousIndex := currentIndex - 1
		if previousIndex < 0 {
			previousIndex = len(cycle) - 1
		}

		previousStep := cycle[previousIndex]

		internalChoices, err := e.liftInsideChildChoices(currentNode, currentVertex, previousStep.nextVertex)
		if err != nil {
			return nil, err
		}
		candidates = e.extendAlternatingCandidates(candidates, internalChoices)
		if len(candidates) == 0 {
			return nil, ErrInvalidMatching
		}

		candidates = e.extendAlternatingCandidates(candidates, [][]int{{previousStep.edgeToNext}})
		if len(candidates) == 0 {
			return nil, ErrInvalidMatching
		}

		currentVertex = previousStep.vertexToNext
		currentIndex = previousIndex
	}

	finalNode := cycle[currentIndex].node

	internalChoices, err := e.liftInsideChildChoices(finalNode, currentVertex, exitVertex)
	if err != nil {
		return nil, err
	}
	candidates = e.extendAlternatingCandidates(candidates, internalChoices)
	if len(candidates) == 0 {
		return nil, ErrInvalidMatching
	}

	return candidates, nil
}

// liftInsideChildChoices returns every candidate internal route inside one cycle child.
// Singleton children allow only a neutral same-vertex route; contracted child blossoms
// recursively return all deterministic lift choices.
//
// Implementation:
//   - Stage 1: Validate child node bounds.
//   - Stage 2: Validate entry and exit vertex ownership through members[].
//   - Stage 3: For singleton children, require entryVertex==exitVertex.
//   - Stage 4: For contracted child blossoms, delegate to liftThroughBlossomChoices.
//
// Behavior highlights:
//   - Does not mutate engine state.
//   - Preserves nested alternatives.
//   - Returns empty edge sequence for neutral singleton routes.
//
// Inputs:
//   - child: singleton or contracted blossom child node.
//   - entryVertex: original vertex where traversal enters child.
//   - exitVertex: original vertex where traversal leaves child.
//
// Returns:
//   - [][]int: candidate internal edge sequences.
//   - error: nil when at least one route exists.
//
// Errors:
//   - ErrInvalidMatching for invalid child, missing ownership, singleton endpoint mismatch,
//     or recursive lifting failure.
//
// Determinism:
//   - Singleton returns one neutral candidate.
//   - Contracted blossoms preserve deterministic recursive choice order.
//
// Complexity:
//   - Time O(1) for singleton children; recursive O(2^b * L) for contracted children.
//
// Notes:
//   - This helper is the safe choice for recursive Blossom path lifting.
//
// AI-Hints:
//   - Do not collapse recursive choices to choices[0].
func (e *blossomEngine) liftInsideChildChoices(child int, entryVertex int, exitVertex int) ([][]int, error) {
	if child < 0 || child >= len(e.members) {
		return nil, ErrInvalidMatching
	}
	if !e.nodeContainsVertex(child, entryVertex) || !e.nodeContainsVertex(child, exitVertex) {
		return nil, ErrInvalidMatching
	}

	if child < e.problem.n {
		if entryVertex != exitVertex {
			return nil, ErrInvalidMatching
		}

		return [][]int{{}}, nil
	}

	return e.liftThroughBlossomChoices(child, entryVertex, exitVertex)
}
