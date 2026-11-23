// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package match contracts and expands odd blossoms in dense weighted matching.
// Contraction turns a same-tree outer/outer odd cycle into one active top-level node;
// expansion restores cycle children when weighted search requires reopening a blossom.
//
// Responsibility:
//   - Locate blossom bases and ordered odd cycles.
//   - Build full edge-to-next cycle metadata.
//   - Remap original vertex ownership through inBlossom.
//   - Preserve forest labels during zero-dual inner expansion.
//   - Provide structural cleanup for non-search restoration paths.
//
// Boundaries:
//   - Path lifting through contracted cycles lives in impl_blossom_path.go.
//   - mate[] mutation lives in impl_blossom_augment.go.
//   - Tight-edge discovery lives in impl_blossom_forest.go.
//
// AI-Hints:
//   - Do not store only the closing shrink edge.
//   - Do not clear forest labels during search expansion.
//   - Do not use a separate children slice; cycles are the child-order source of truth.
package match

// shrink contracts an odd alternating cycle into one active top-level blossom node.
// The new blossom replaces the cycle base node in the alternating forest, so root
// and parent metadata must be copied from the base node before child nodes are deactivated.
//
// Implementation:
//   - Stage 1: Locate base vertex, base node, and both cycle arms.
//   - Stage 2: Snapshot the base node forest position.
//   - Stage 3: Allocate and contract the blossom cycle.
//   - Stage 4: Publish the new blossom as an outer node at the base node position.
//   - Stage 5: Enqueue the new blossom for further tight-edge scanning.
//
// Behavior highlights:
//   - Does not mutate mate[] directly.
//   - Preserves a valid active-root invariant after root-cycle contraction.
//   - Preserves parent linkage when the contracted cycle is not rooted.
//
// Inputs:
//   - event: same-tree outer/outer shrink event with endpoint ownership.
//
// Returns:
//   - error: nil after successful contraction.
//
// Errors:
//   - ErrInvalidMatching for malformed event, invalid cycle, corrupted base position,
//     exhausted blossom capacity, or invalid post-contraction forest placement.
//
// Determinism:
//   - Base detection and cycle order follow deterministic parent paths.
//
// Complexity:
//   - Time O(k), Space O(k).
//
// Notes:
//   - A contracted blossom must replace the base node, not merely point at the old inactive root.
//
// AI-Hints:
//   - If the base node was a root, set treeRoot[newNode]=newNode.
//   - Do not leave parent[newNode]==noNode with treeRoot[newNode]!=newNode.
func (e *blossomEngine) shrink(event blossomEvent) error {
	if event.kind != eventShrink {
		return ErrInvalidMatching
	}

	baseVertex, baseNode, pathA, pathB, err := e.findBlossomBaseAndCycle(event.a, event.b)
	if err != nil {
		return err
	}

	baseParent := e.parent[baseNode]
	baseLabelEdge := e.labelEdge[baseNode]
	baseRoot := e.treeRoot[baseNode]

	newNode, err := e.allocateBlossomNode(baseVertex, pathA, pathB, event.edge)
	if err != nil {
		return err
	}

	if err = e.contractCycleIntoBlossom(newNode); err != nil {
		return err
	}

	e.label[newNode] = blossomOuter

	if baseParent == noNode {
		if baseLabelEdge != noEdge {
			return ErrInvalidMatching
		}

		e.parent[newNode] = noNode
		e.labelEdge[newNode] = noEdge
		e.treeRoot[newNode] = newNode
	} else {
		if baseParent < 0 || baseParent >= len(e.active) || !e.active[baseParent] {
			return ErrInvalidMatching
		}
		if baseLabelEdge == noEdge || baseRoot == noNode {
			return ErrInvalidMatching
		}

		e.parent[newNode] = baseParent
		e.labelEdge[newNode] = baseLabelEdge
		e.treeRoot[newNode] = baseRoot
	}

	if err = e.redirectForestThroughContractedBlossom(newNode); err != nil {
		return err
	}

	e.queue = append(e.queue, newNode)

	e.stats.Shrinks++

	return nil
}

// nodePathToRoot returns active node IDs from a node toward its alternating-tree root.
// shrink uses this path to find the common base node of a same-tree outer/outer odd cycle.
//
// Implementation:
//   - Stage 1: Validate that the start node is active.
//   - Stage 2: Follow parent links toward the root.
//   - Stage 3: Append every visited active node.
//   - Stage 4: Stop when treeRoot[current]==current.
//
// Behavior highlights:
//   - Does not mutate engine state.
//   - Requires active forest closure; contracted inactive children must not appear.
//   - Returns node-to-root order, not root-to-node order.
//
// Inputs:
//   - node: active top-level forest node.
//
// Returns:
//   - []int: path from node to its tree root, inclusive.
//   - error: nil when the parent chain is structurally valid.
//
// Errors:
//   - ErrInvalidMatching for inactive nodes, out-of-range nodes, or broken parent chains.
//
// Determinism:
//   - Follows exactly one stored parent chain.
//
// Complexity:
//   - Time O(k), Space O(k).
//
// Notes:
//   - redirectForestThroughContractedBlossom protects this method after shrink.
//
// AI-Hints:
//   - Do not skip inactive nodes here; their presence means contraction cleanup failed.
func (e *blossomEngine) nodePathToRoot(node int) ([]int, error) {
	if node < 0 || node >= len(e.active) || !e.active[node] {
		return nil, ErrInvalidMatching
	}

	path := make([]int, 0, e.problem.n)

	for current := node; current != noNode; current = e.parent[current] {
		if current < 0 || current >= len(e.active) {
			return nil, ErrInvalidMatching
		}

		path = append(path, current)

		if e.treeRoot[current] == current {
			break
		}
	}

	return path, nil
}

// findBlossomBaseAndCycle locates the alternating-tree base and ordered cycle arms.
// The base vertex is used as the contracted blossom base; the base node is used
// to place the new blossom back into the alternating forest.
//
// Implementation:
//   - Stage 1: Build node-to-root paths for both outer endpoints.
//   - Stage 2: Find the first common active top-level node.
//   - Stage 3: Use that common node as the blossom base node.
//   - Stage 4: Return local original base vertex plus both cycle arms.
//
// Behavior highlights:
//   - Requires a and b to be in the same alternating tree.
//   - Does not mutate engine state.
//   - Keeps base node separate from base original vertex.
//
// Inputs:
//   - a: first active outer top-level node.
//   - b: second active outer top-level node in the same tree.
//
// Returns:
//   - int: original local base vertex.
//   - int: active top-level base node.
//   - []int: left path from a to base node.
//   - []int: right path from b toward, but excluding, base node.
//   - error: nil on valid odd-cycle discovery.
//
// Errors:
//   - ErrInvalidMatching for different roots, malformed parent paths, missing common base,
//     or corrupted base ownership.
//
// Determinism:
//   - Uses deterministic parent-path intersection order.
//
// Complexity:
//   - Time O(k^2), Space O(k).
//
// Notes:
//   - baseNode is required to publish the contracted blossom into the forest correctly.
//
// AI-Hints:
//   - Do not derive forest placement from baseVertex alone.
//   - baseVertex belongs to original matching vertices; baseNode belongs to active forest nodes.
func (e *blossomEngine) findBlossomBaseAndCycle(a, b int) (baseVertex, baseNode int, left, right []int, err error) {
	if a < 0 || a >= len(e.active) || b < 0 || b >= len(e.active) {
		return noVertex, noNode, nil, nil, ErrInvalidMatching
	}
	if !e.active[a] || !e.active[b] {
		return noVertex, noNode, nil, nil, ErrInvalidMatching
	}
	if e.treeRoot[a] != e.treeRoot[b] {
		return noVertex, noNode, nil, nil, ErrInvalidMatching
	}

	pathA, err := e.nodePathToRoot(a)
	if err != nil {
		return noVertex, noNode, nil, nil, err
	}

	pathB, err := e.nodePathToRoot(b)
	if err != nil {
		return noVertex, noNode, nil, nil, err
	}

	commonA := -1
	commonB := -1

	for indexA, nodeA := range pathA {
		for indexB, nodeB := range pathB {
			if nodeA == nodeB {
				commonA = indexA
				commonB = indexB
				break
			}
		}
		if commonA != -1 {
			break
		}
	}

	if commonA == -1 || commonB == -1 {
		return noVertex, noNode, nil, nil, ErrInvalidMatching
	}

	baseNode = pathA[commonA]
	if baseNode < 0 || baseNode >= len(e.base) || e.base[baseNode] == noVertex {
		return noVertex, noNode, nil, nil, ErrInvalidMatching
	}

	left = append([]int(nil), pathA[:commonA+1]...)
	right = append([]int(nil), pathB[:commonB]...)

	return e.base[baseNode], baseNode, left, right, nil
}

// allocateBlossomNode reserves and initializes one contracted blossom node with deterministic
// child order, original base vertex, full edge-to-next cycle metadata, member vertices,
// and zero initial blossom dual.
//
// Implementation:
//   - Stage 1: Validate base vertex, closing edge, and node capacity.
//   - Stage 2: Build deterministic cycle order from left arm plus reversed right arm.
//   - Stage 3: Validate odd cycle length.
//   - Stage 4: Merge original member vertices from all child nodes.
//   - Stage 5: Build edge-to-next metadata for every adjacent cycle pair.
//   - Stage 6: Store base, cycles, members, and initial blossom dual.
//
// Behavior highlights:
//   - Does not activate the new node.
//   - Does not mutate mate[].
//   - Stores enough metadata for future lifting and expansion.
//   - Uses cycles[] as the only child-order source of truth.
//
// Inputs:
//   - base: original local vertex that represents the blossom base.
//   - left: active node path from one outer endpoint to the base node.
//   - right: active node path from the other outer endpoint toward, but excluding, the base node.
//   - edgeID: closing tight edge that completes the odd cycle.
//
// Returns:
//   - int: allocated blossom node ID.
//   - error: nil after metadata is initialized.
//
// Errors:
//   - ErrInvalidMatching for invalid base, invalid edge, exhausted capacity,
//     inactive children, malformed cycle shape, or invalid cycle metadata.
//
// Determinism:
//   - Child order is left followed by reversed right.
//
// Complexity:
//   - Time O(k), Space O(k), where k is cycle/member count.
//
// Notes:
//   - contractCycleIntoBlossom performs ownership remapping after allocation.
//
// AI-Hints:
//   - Do not store only the closing edge.
//   - Do not publish the new node before cycle metadata is complete.
func (e *blossomEngine) allocateBlossomNode(base int, left, right []int, edgeID int) (int, error) {
	if base < 0 || base >= e.problem.n {
		return noNode, ErrInvalidMatching
	}
	if edgeID < 0 || edgeID >= len(e.edges) {
		return noNode, ErrInvalidMatching
	}
	if e.nextNode >= len(e.active) {
		return noNode, ErrInvalidMatching
	}

	newNode := e.nextNode
	e.nextNode++

	cycle := make([]int, 0, len(left)+len(right))
	cycle = append(cycle, left...)

	for index := len(right) - 1; index >= 0; index-- {
		cycle = append(cycle, right[index])
	}

	if len(cycle) < 3 || (len(cycle)&1) == 0 {
		return noNode, ErrInvalidMatching
	}

	members := make([]int, 0, e.problem.n)

	for _, child := range cycle {
		if child < 0 || child >= len(e.active) || !e.active[child] {
			return noNode, ErrInvalidMatching
		}

		members = append(members, e.members[child]...)
	}

	cycleSteps, err := e.buildBlossomCycleSteps(cycle, edgeID)
	if err != nil {
		return noNode, err
	}

	e.base[newNode] = base
	e.cycles[newNode] = cycleSteps
	e.members[newNode] = members
	e.dual[newNode] = 0

	return newNode, nil
}

// contractCycleIntoBlossom publishes an allocated blossom as the active top-level owner
// of all original member vertices. Cycle children become inactive, and every original
// member is remapped through inBlossom.
//
// Implementation:
//   - Stage 1: Validate allocated contracted node and cycle metadata.
//   - Stage 2: Deactivate every child node in the stored cycle.
//   - Stage 3: Remap each original member vertex to the new blossom node.
//   - Stage 4: Mark the new blossom node active.
//
// Behavior highlights:
//   - Does not mutate mate[] or mateEdge[].
//   - Does not assign forest label/root/parent; shrink does that after placement snapshot.
//   - Preserves cycle metadata for lifting and future expansion.
//
// Inputs:
//   - node: allocated contracted blossom node.
//
// Returns:
//   - error: nil after ownership remapping succeeds.
//
// Errors:
//   - ErrInvalidMatching for invalid node, malformed cycle metadata,
//     inactive children, empty members, or invalid original member IDs.
//
// Determinism:
//   - Cycle and member scans follow stored order.
//
// Complexity:
//   - Time O(k), Space O(1), where k is total cycle/member count.
//
// Notes:
//   - shrink must call redirectForestThroughContractedBlossom after forest placement.
//
// AI-Hints:
//   - Do not clear child metadata here.
//   - Do not forget to update inBlossom for every original member.
func (e *blossomEngine) contractCycleIntoBlossom(node int) error {
	if node < e.problem.n || node >= len(e.active) {
		return ErrInvalidMatching
	}

	steps, err := e.cycleSteps(node)
	if err != nil {
		return err
	}
	if len(e.members[node]) == 0 {
		return ErrInvalidMatching
	}

	for _, step := range steps {
		child := step.node
		if child < 0 || child >= len(e.active) || !e.active[child] {
			return ErrInvalidMatching
		}

		e.active[child] = false
	}

	for _, vertex := range e.members[node] {
		if vertex < 0 || vertex >= e.problem.n {
			return ErrInvalidMatching
		}

		e.inBlossom[vertex] = node
	}

	e.active[node] = true

	return nil
}

// expand expands a zero-dual inner blossom while preserving alternating-forest semantics.
// It is invoked during weighted search when an inner contracted blossom reaches zero dual
// and must be reopened so tight-edge scanning can continue on its children.
//
// Implementation:
//   - Stage 1: Validate active contracted blossom shape.
//   - Stage 2: Ignore non-inner or positive-dual blossoms.
//   - Stage 3: Expand children through expandInnerBlossomInForest.
//   - Stage 4: Deactivate the contracted node and record expansion telemetry.
//
// Behavior highlights:
//   - Does not clear forest labels blindly.
//   - Preserves the represented matching cost.
//   - Reactivates child nodes in deterministic cycle order.
//   - Keeps inBlossom ownership consistent for original vertices.
//
// Inputs:
//   - node: active contracted blossom node.
//
// Returns:
//   - error: nil when expansion is not needed or succeeds.
//
// Errors:
//   - ErrInvalidMatching for invalid node, missing cycle metadata, corrupted children,
//     or broken ownership restoration.
//
// Determinism:
//   - Children are restored in cycle order.
//
// Complexity:
//   - Time O(c + members), Space O(1).
//
// Notes:
//   - Contracted node storage is cleared only after forest-aware expansion succeeds.
//   - Search expansion must preserve enough labels for continued alternating-tree processing.
//
// AI-Hints:
//   - Do not reset all child labels to blossomUnlabeled here.
func (e *blossomEngine) expand(node int) error {
	if node < e.problem.n || node >= len(e.active) {
		return ErrInvalidMatching
	}
	if !e.active[node] {
		return ErrInvalidMatching
	}
	if _, err := e.cycleSteps(node); err != nil {
		return err
	}

	if e.label[node] != blossomInner || e.dual[node] > e.eps {
		return nil
	}

	if err := e.expandInnerBlossomInForest(node); err != nil {
		return err
	}

	e.active[node] = false
	e.stats.Expansions++

	return nil
}

// expandInnerBlossomInForest reopens a zero-dual inner blossom inside the active forest.
// The old contracted node is replaced by an alternating child path from the entry child
// of the parent edge to the base child. Children not on that path become active but
// unlabeled top-level nodes.
//
// Implementation:
//   - Stage 1: Validate the contracted inner blossom and read its old forest attachments.
//   - Stage 2: Locate the entry child from labelEdge[node], not from base.
//   - Stage 3: Locate the base child from e.base[node].
//   - Stage 4: Capture the old outer continuation child whose parent was node.
//   - Stage 5: Reactivate all cycle children and restore inBlossom ownership.
//   - Stage 6: Choose the valid alternating path from entry child to base child.
//   - Stage 7: Label only that path; leave off-path children unlabeled.
//   - Stage 8: Reparent the old outer continuation child to the base child.
//   - Stage 9: Clear the contracted node storage.
//
// Behavior highlights:
//   - Does not mutate mate[].
//   - Does not label the whole odd cycle.
//   - Does not attach parent edge to base unless the parent edge actually enters base.
//   - Does not leave active forest nodes pointing to the cleared contracted node.
//
// Inputs:
//   - node: active inner contracted blossom node with zero dual.
//
// Returns:
//   - error: nil after forest-preserving expansion.
//
// Errors:
//   - ErrInvalidMatching for malformed cycle metadata, invalid parent edge,
//     missing entry/base ownership, impossible alternating path, or stale forest children.
//
// Determinism:
//   - Tries forward cycle direction first, then backward direction.
//
// Complexity:
//   - Time O(c + k), Space O(c), where c is blossom cycle length.
//
// AI-Hints:
//   - Do not start from baseChildIndex blindly.
//   - Do not label all cycle children.
//   - Reparent the existing outer child from the contracted node to the base child.
//   - Edmonds' Blossom algorithm is inherently branch-heavy due to multiple pointer and
//     index manipulations for contracted/expanded nodes. Breaking this method apart
//     introduces high risk of violating delicate graph invariants.
//
// nolint:gocyclo
func (e *blossomEngine) expandInnerBlossomInForest(node int) error {
	if node < e.problem.n || node >= len(e.cycles) {
		return ErrInvalidMatching
	}
	if !e.active[node] || e.label[node] != blossomInner {
		return ErrInvalidMatching
	}
	if e.dual[node] > e.eps {
		return ErrInvalidMatching
	}

	steps, err := e.cycleSteps(node)
	if err != nil {
		return err
	}

	root := e.treeRoot[node]
	oldParent := e.parent[node]
	oldParentEdge := e.labelEdge[node]
	if root == noNode || oldParent == noNode || oldParentEdge == noEdge {
		return ErrInvalidMatching
	}
	if oldParent < 0 || oldParent >= len(e.active) || !e.active[oldParent] {
		return ErrInvalidMatching
	}
	if e.label[oldParent] != blossomOuter {
		return ErrInvalidMatching
	}

	_, entryVertex, err := e.orientEdgeForNodes(oldParentEdge, oldParent, node)
	if err != nil {
		return err
	}

	entryIndex, err := e.cycleChildIndexContainingVertex(node, entryVertex)
	if err != nil {
		return err
	}

	baseIndex, err := e.baseChildIndex(node)
	if err != nil {
		return err
	}

	outerChild, outerChildEdge, hasOuterChild, err := e.outerChildOfContractedBlossom(node)
	if err != nil {
		return err
	}

	path, err := e.chooseInnerExpansionPath(node, entryIndex, baseIndex, oldParentEdge, outerChildEdge, hasOuterChild)
	if err != nil {
		return err
	}

	// Restore every child as an active top-level node, but keep it outside the
	// forest unless it belongs to the selected alternating path.
	for _, step := range steps {
		child := step.node
		if child < 0 || child >= len(e.active) {
			return ErrInvalidMatching
		}

		e.active[child] = true
		e.label[child] = blossomUnlabeled
		e.parent[child] = noNode
		e.labelEdge[child] = noEdge
		e.treeRoot[child] = noNode

		for _, vertex := range e.members[child] {
			if vertex < 0 || vertex >= e.problem.n {
				return ErrInvalidMatching
			}
			e.inBlossom[vertex] = child
		}
	}

	// Label the selected entry->base path.
	for offset, index := range path {
		child := steps[index].node

		e.treeRoot[child] = root

		if offset == 0 {
			e.label[child] = blossomInner
			e.parent[child] = oldParent
			e.labelEdge[child] = oldParentEdge
			continue
		}

		prevIndex := path[offset-1]
		prevChild := steps[prevIndex].node

		edgeID, edgeErr := cycleEdgeBetweenAdjacentIndices(steps, prevIndex, index)
		if edgeErr != nil {
			return edgeErr
		}

		e.parent[child] = prevChild
		e.labelEdge[child] = edgeID

		if (offset & 1) == 1 {
			e.label[child] = blossomOuter
			e.queue = append(e.queue, child)
			continue
		}

		e.label[child] = blossomInner
	}

	baseChild := steps[baseIndex].node

	// Reattach the old outer continuation child, if the contracted node had one.
	if hasOuterChild {
		if outerChild < 0 || outerChild >= len(e.active) || !e.active[outerChild] {
			return ErrInvalidMatching
		}
		if e.label[outerChild] != blossomOuter {
			return ErrInvalidMatching
		}
		if outerChildEdge == noEdge {
			return ErrInvalidMatching
		}

		e.parent[outerChild] = baseChild
		e.labelEdge[outerChild] = outerChildEdge
		e.treeRoot[outerChild] = root
		e.queue = append(e.queue, outerChild)
	}

	e.cycles[node] = nil
	e.members[node] = nil
	e.base[node] = noVertex
	e.dual[node] = 0
	e.active[node] = false
	e.parent[node] = noNode
	e.label[node] = blossomUnlabeled
	e.labelEdge[node] = noEdge
	e.treeRoot[node] = noNode

	return nil
}

// outerChildOfContractedBlossom finds the active outer forest child whose parent is
// a contracted inner blossom. During inner-blossom expansion this child represents
// the matched continuation leaving the blossom base.
//
// Implementation:
//   - Stage 1: Scan forest parent links.
//   - Stage 2: Ignore inactive nodes and the contracted node itself.
//   - Stage 3: Select the unique active outer child whose parent is node.
//   - Stage 4: Return its label edge when present.
//
// Behavior highlights:
//   - Does not mutate engine state.
//   - Accepts absence of such child.
//   - Rejects multiple continuation children as forest corruption.
//   - Requires the continuation child to be outer.
//
// Inputs:
//   - node: contracted blossom node being expanded.
//
// Returns:
//   - int: found outer child node, or noNode.
//   - int: dense label edge from the child to the contracted blossom, or noEdge.
//   - bool: true when a continuation child exists.
//   - error: nil when the relation is structurally valid.
//
// Errors:
//   - ErrInvalidMatching for multiple children, non-outer children, or missing label edge.
//
// Determinism:
//   - Fixed node-index scan.
//
// Complexity:
//   - Time O(k), Space O(1).
//
// Notes:
//   - At most one such child is valid in an alternating forest.
//
// AI-Hints:
//   - Do not silently pick the first child if multiple are found.
func (e *blossomEngine) outerChildOfContractedBlossom(node int) (int, int, bool, error) {
	foundChild := noNode
	foundEdge := noEdge

	for child := range e.parent {
		if child == node || e.parent[child] != node {
			continue
		}
		if child < 0 || child >= len(e.active) || !e.active[child] {
			continue
		}
		if e.label[child] != blossomOuter || e.labelEdge[child] == noEdge {
			return noNode, noEdge, false, ErrInvalidMatching
		}

		if foundChild != noNode {
			return noNode, noEdge, false, ErrInvalidMatching
		}

		foundChild = child
		foundEdge = e.labelEdge[child]
	}

	if foundChild == noNode {
		return noNode, noEdge, false, nil
	}

	return foundChild, foundEdge, true, nil
}

// chooseInnerExpansionPath selects the valid alternating child-index path used to replace
// a zero-dual inner contracted blossom in the active forest. It tries both cycle directions
// and returns the one that satisfies matched/unmatched parity constraints.
//
// Implementation:
//   - Stage 1: Build the forward entry-to-base index path.
//   - Stage 2: Validate forest parity through innerExpansionPathValid.
//   - Stage 3: Build the backward entry-to-base index path if needed.
//   - Stage 4: Validate backward parity.
//   - Stage 5: Reject when neither direction is valid.
//
// Behavior highlights:
//   - Does not mutate engine state.
//   - Uses entry child, not blindly the base child, as the expansion start.
//   - Handles optional outer continuation child parity.
//
// Inputs:
//   - node: contracted blossom node being expanded.
//   - entryIndex: child index entered by the old parent edge.
//   - baseIndex: child index containing the blossom base vertex.
//   - oldParentEdge: edge from old outer parent into the blossom.
//   - outerChildEdge: optional continuation edge from blossom base to outer child.
//   - hasOuterChild: whether outerChildEdge is meaningful.
//
// Returns:
//   - []int: child indices from entry child to base child, inclusive.
//   - error: nil when a valid direction exists.
//
// Errors:
//   - ErrInvalidMatching for malformed cycle paths or absence of a valid parity direction.
//
// Determinism:
//   - Forward direction is tried before backward direction.
//
// Complexity:
//   - Time O(c), Space O(c).
//
// Notes:
//   - The returned path is consumed by expandInnerBlossomInForest.
//
// AI-Hints:
//   - Do not choose by shortest path.
//   - Do not ignore outerChildEdge parity when hasOuterChild is true.
func (e *blossomEngine) chooseInnerExpansionPath(
	node int,
	entryIndex int,
	baseIndex int,
	oldParentEdge int,
	outerChildEdge int,
	hasOuterChild bool,
) ([]int, error) {
	for _, forward := range []bool{true, false} {
		path, err := e.cycleIndexPath(node, entryIndex, baseIndex, forward)
		if err != nil {
			return nil, err
		}

		if e.innerExpansionPathValid(node, path, oldParentEdge, outerChildEdge, hasOuterChild) {
			return path, nil
		}
	}

	return nil, ErrInvalidMatching
}

// innerExpansionPathValid checks whether a proposed entry-to-base child-index path
// can replace an inner contracted blossom while preserving alternating-forest parity.
//
// Implementation:
//   - Stage 1: Reject empty paths.
//   - Stage 2: Require even internal edge count so entry and base children are both inner.
//   - Stage 3: Require the old parent edge into the entry child to be unmatched.
//   - Stage 4: Check alternating matched/unmatched status along internal cycle edges.
//   - Stage 5: If present, require the outer continuation edge to be matched.
//
// Behavior highlights:
//   - Does not mutate engine state.
//   - Validates forest-label parity, not full augmenting-path parity.
//   - Uses current committed matching state through isMatchedEdge.
//
// Inputs:
//   - node: contracted blossom node.
//   - path: child indices from entry child to base child.
//   - oldParentEdge: unmatched edge from old outer parent into entry child.
//   - outerChildEdge: matched continuation edge to old outer child, when present.
//   - hasOuterChild: whether outerChildEdge should be checked.
//
// Returns:
//   - bool: true when the path is a valid inner-blossom forest replacement.
//
// Errors:
//   - None. Malformed inputs return false.
//
// Determinism:
//   - Fixed path-order scan.
//
// Complexity:
//   - Time O(len(path)), Space O(1).
//
// Notes:
//   - This is a predicate used before expansion mutates forest labels.
//
// AI-Hints:
//   - Do not relax the even-length requirement.
//   - Do not treat oldParentEdge as matched for inner expansion.
func (e *blossomEngine) innerExpansionPathValid(
	node int,
	path []int,
	oldParentEdge int,
	outerChildEdge int,
	hasOuterChild bool,
) bool {
	if len(path) == 0 {
		return false
	}

	// Entry child is inner, base child must also be inner. Therefore the number
	// of internal cycle edges from entry to base must be even.
	if ((len(path) - 1) & 1) != 0 {
		return false
	}

	// Parent edge into an inner node must be an unmatched edge.
	if e.isMatchedEdge(oldParentEdge) {
		return false
	}

	steps := e.cycles[node]

	for offset := 1; offset < len(path); offset++ {
		edgeID, err := cycleEdgeBetweenAdjacentIndices(steps, path[offset-1], path[offset])
		if err != nil {
			return false
		}

		// offset 1 labels an outer child, so the edge Inner->Outer must be matched.
		// offset 2 labels an inner child, so the edge Outer->Inner must be unmatched.
		wantMatched := (offset & 1) == 1
		if e.isMatchedEdge(edgeID) != wantMatched {
			return false
		}
	}

	if hasOuterChild {
		if outerChildEdge == noEdge || !e.isMatchedEdge(outerChildEdge) {
			return false
		}
	}

	return true
}

// cycleIndexPath returns child indices from fromIndex to toIndex, inclusive, walking one
// deterministic direction around a stored blossom cycle.
//
// Implementation:
//   - Stage 1: Validate cycle metadata and index bounds.
//   - Stage 2: Start from fromIndex.
//   - Stage 3: Move one step forward or backward with wraparound.
//   - Stage 4: Stop at toIndex, rejecting loops longer than the cycle.
//
// Behavior highlights:
//   - Does not mutate engine state.
//   - Includes both endpoints.
//   - Supports forward and backward deterministic paths.
//
// Inputs:
//   - node: contracted blossom node.
//   - fromIndex: starting child index.
//   - toIndex: target child index.
//   - forward: true for increasing indices, false for decreasing indices.
//
// Returns:
//   - []int: inclusive index path.
//   - error: nil when indices are valid and path is bounded.
//
// Errors:
//   - ErrInvalidMatching for invalid node, malformed cycle metadata, invalid indices,
//     or impossible cycle traversal.
//
// Determinism:
//   - Fixed wraparound arithmetic.
//
// Complexity:
//   - Time O(c), Space O(c).
//
// Notes:
//   - This helper works on child indices, not dense edge IDs.
//
// AI-Hints:
//   - Do not omit the start or end index; expansion parity depends on both.
func (e *blossomEngine) cycleIndexPath(node int, fromIndex int, toIndex int, forward bool) ([]int, error) {
	steps, err := e.cycleSteps(node)
	if err != nil {
		return nil, err
	}

	if fromIndex < 0 || fromIndex >= len(steps) || toIndex < 0 || toIndex >= len(steps) {
		return nil, ErrInvalidMatching
	}

	path := []int{fromIndex}
	current := fromIndex

	for current != toIndex {
		if forward {
			current++
			if current == len(steps) {
				current = 0
			}
		} else {
			current--
			if current < 0 {
				current = len(steps) - 1
			}
		}

		path = append(path, current)
		if len(path) > len(steps) {
			return nil, ErrInvalidMatching
		}
	}

	return path, nil
}

// cycleEdgeBetweenAdjacentIndices returns the stored dense edge connecting two adjacent
// child indices in either direction around a contracted blossom cycle.
//
// Implementation:
//   - Stage 1: Validate odd cycle metadata and index bounds.
//   - Stage 2: Check forward adjacency from fromIndex to toIndex.
//   - Stage 3: Check backward adjacency from fromIndex to toIndex.
//   - Stage 4: Return noEdge with ErrInvalidMatching if indices are not adjacent.
//
// Behavior highlights:
//   - Pure helper; no engine state required.
//   - Supports both cycle directions.
//   - Uses stored edgeToNext metadata only.
//
// Inputs:
//   - steps: ordered blossom cycle metadata.
//   - fromIndex: source child index.
//   - toIndex: adjacent target child index.
//
// Returns:
//   - int: dense edge connecting the two adjacent children.
//   - error: nil when adjacency is valid.
//
// Errors:
//   - ErrInvalidMatching for malformed cycle metadata, invalid indices, or non-adjacent indices.
//
// Determinism:
//   - Fixed arithmetic over cycle length.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This helper does not validate endpoint ownership; that is done at cycle construction.
//
// AI-Hints:
//   - Do not search all edges here.
//   - Do not allow non-adjacent indices to succeed.
func cycleEdgeBetweenAdjacentIndices(steps []blossomCycleStep, fromIndex int, toIndex int) (int, error) {
	if len(steps) < 3 || (len(steps)&1) == 0 {
		return noEdge, ErrInvalidMatching
	}
	if fromIndex < 0 || fromIndex >= len(steps) || toIndex < 0 || toIndex >= len(steps) {
		return noEdge, ErrInvalidMatching
	}

	if toIndex == (fromIndex+1)%len(steps) {
		return steps[fromIndex].edgeToNext, nil
	}

	prev := fromIndex - 1
	if prev < 0 {
		prev = len(steps) - 1
	}
	if toIndex == prev {
		return steps[toIndex].edgeToNext, nil
	}

	return noEdge, ErrInvalidMatching
}

// edgeBetweenAdjacentTreeNodes returns the label edge between parent-adjacent forest nodes.
// It is used while building a contracted blossom cycle from two paths to their common base.
//
// Implementation:
//   - Stage 1: Check whether left is the child of right.
//   - Stage 2: Check whether right is the child of left.
//   - Stage 3: Return the child label edge when exactly one parent relation matches.
//
// Behavior highlights:
//   - Does not mutate engine state.
//   - Rejects non-adjacent nodes.
//   - Uses forest parent links, not graph adjacency scans.
//
// Inputs:
//   - left: first active top-level node.
//   - right: second active top-level node.
//
// Returns:
//   - int: dense edge ID connecting the two forest-adjacent nodes.
//   - error: nil when the forest relation is valid.
//
// Errors:
//   - ErrInvalidMatching for invalid node IDs, non-adjacent nodes, or missing label edge.
//
// Determinism:
//   - Fixed parent-direction checks.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - The closing shrink edge is handled separately; this helper handles only parent-chain edges.
//
// AI-Hints:
//   - Do not search all dense edges here; parent links already define the alternating-tree edge.
func (e *blossomEngine) edgeBetweenAdjacentTreeNodes(left, right int) (int, error) {
	if left < 0 || left >= len(e.parent) || right < 0 || right >= len(e.parent) {
		return noEdge, ErrInvalidMatching
	}

	if e.parent[left] == right {
		if e.labelEdge[left] == noEdge {
			return noEdge, ErrInvalidMatching
		}

		return e.labelEdge[left], nil
	}

	if e.parent[right] == left {
		if e.labelEdge[right] == noEdge {
			return noEdge, ErrInvalidMatching
		}

		return e.labelEdge[right], nil
	}

	return noEdge, ErrInvalidMatching
}

// buildBlossomCycleSteps builds ordered edge-to-next metadata for one blossom cycle.
// The resulting slice has the same length as cycle, and step i connects cycle[i]
// to cycle[(i+1)%len(cycle)].
//
// Implementation:
//   - Stage 1: Validate odd cycle length.
//   - Stage 2: Resolve parent-chain edges for adjacent cycle nodes.
//   - Stage 3: Use closingEdgeID for the final cycle edge.
//   - Stage 4: Orient every dense edge by child-node ownership.
//
// Behavior highlights:
//   - Stores enough metadata for path lifting.
//   - Does not mutate engine state.
//   - Rejects incomplete cycle edge descriptions immediately.
//
// Inputs:
//   - cycle: active child nodes in deterministic blossom cycle order.
//   - closingEdgeID: tight edge that closes the odd cycle.
//
// Returns:
//   - []blossomCycleStep: ordered cycle metadata.
//   - error: nil when every cycle edge is valid.
//
// Errors:
//   - ErrInvalidMatching for malformed cycle shape, invalid child nodes,
//     invalid edge IDs, or endpoint ownership mismatch.
//
// Determinism:
//   - Fixed cycle-index scan.
//   - Closing edge is always the last step.
//
// Complexity:
//   - Time O(c * m), where c is cycle length and m is average child membership.
//   - Space O(c).
//
// Notes:
//   - This replaces the invalid single-edge childEdges model.
//
// AI-Hints:
//   - Do not store only closingEdgeID.
//   - Do not allow len(steps) != len(cycle).
func (e *blossomEngine) buildBlossomCycleSteps(cycle []int, closingEdgeID int) ([]blossomCycleStep, error) {
	if len(cycle) < 3 || (len(cycle)&1) == 0 {
		return nil, ErrInvalidMatching
	}

	steps := make([]blossomCycleStep, len(cycle))

	for index, node := range cycle {
		if node < 0 || node >= len(e.active) || !e.active[node] {
			return nil, ErrInvalidMatching
		}

		nextIndex := index + 1
		if nextIndex == len(cycle) {
			nextIndex = 0
		}

		nextNode := cycle[nextIndex]

		edgeID := closingEdgeID
		if index+1 < len(cycle) {
			var err error
			edgeID, err = e.edgeBetweenAdjacentTreeNodes(node, nextNode)
			if err != nil {
				return nil, err
			}
		}

		vertexToNext, nextVertex, err := e.orientEdgeForNodes(edgeID, node, nextNode)
		if err != nil {
			return nil, err
		}

		steps[index] = blossomCycleStep{
			node:         node,
			edgeToNext:   edgeID,
			vertexToNext: vertexToNext,
			nextVertex:   nextVertex,
		}
	}

	return steps, nil
}

// baseChildIndex returns the child cycle index that owns the current base original vertex
// of a contracted blossom.
//
// Implementation:
//   - Stage 1: Validate node is a contracted blossom node.
//   - Stage 2: Validate base[node] is an original local vertex.
//   - Stage 3: Scan cycle children in stored order.
//   - Stage 4: Return the first child whose members contain the base vertex.
//
// Behavior highlights:
//   - Does not mutate engine state.
//   - Works for singleton and nested child blossoms through members[].
//   - Used by expansion and compatibility path helpers.
//
// Inputs:
//   - node: contracted blossom node.
//
// Returns:
//   - int: cycle child index owning base[node].
//   - error: nil when ownership is found.
//
// Errors:
//   - ErrInvalidMatching for invalid node, invalid base, or missing base ownership.
//
// Determinism:
//   - Fixed stored cycle order.
//
// Complexity:
//   - Time O(c*m), Space O(1), where c is cycle length and m is average child membership size.
//
// Notes:
//   - base[node] must be kept fresh after augmentation through refreshAllocatedBlossomBases.
//
// AI-Hints:
//   - Do not infer the base child from cycle position alone.
//   - Do not use inBlossom here; inactive nested children may still own base through members[].
func (e *blossomEngine) baseChildIndex(node int) (int, error) {
	if node < e.problem.n || node >= len(e.cycles) {
		return -1, ErrInvalidMatching
	}

	base := e.base[node]
	if base < 0 || base >= e.problem.n {
		return -1, ErrInvalidMatching
	}

	for index, step := range e.cycles[node] {
		if e.nodeContainsVertex(step.node, base) {
			return index, nil
		}
	}

	return -1, ErrInvalidMatching
}

// cycleSteps returns the ordered cycle metadata for one contracted blossom node.
// It is the canonical accessor for child order and edge-to-next data; callers must
// not read a separate child list because no separate child list exists.
//
// Implementation:
//   - Stage 1: Validate that node is a contracted blossom node, not an original vertex.
//   - Stage 2: Validate bounds against the allocated cycle storage.
//   - Stage 3: Reject empty, too-short, or even-length cycle metadata.
//   - Stage 4: Return the stored slice without allocation.
//
// Behavior highlights:
//   - Does not mutate engine state.
//   - Preserves the deterministic stored cycle order.
//   - Treats malformed cycle metadata as ErrInvalidMatching.
//
// Inputs:
//   - node: contracted blossom node ID.
//
// Returns:
//   - []blossomCycleStep: ordered odd cycle metadata.
//   - error: nil when the cycle is structurally valid.
//
// Errors:
//   - ErrInvalidMatching for singleton nodes, out-of-range nodes, empty cycles,
//     even-length cycles, or cycles shorter than three children.
//
// Determinism:
//   - Pure indexed access; no map iteration or reordering.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - The returned slice aliases engine storage and must be treated as read-only.
//
// AI-Hints:
//   - Do not append to the returned slice.
//   - Do not reconstruct children from members; members do not encode cycle order.
func (e *blossomEngine) cycleSteps(node int) ([]blossomCycleStep, error) {
	if node < e.problem.n || node >= len(e.cycles) {
		return nil, ErrInvalidMatching
	}

	steps := e.cycles[node]
	if len(steps) < 3 || (len(steps)&1) == 0 {
		return nil, ErrInvalidMatching
	}

	return steps, nil
}

// redirectForestThroughContractedBlossom rewires active forest nodes that still point
// through child nodes hidden by a fresh blossom contraction.
//
// Mathematical invariant:
//   - After contraction, inactive cycle children must not appear on active forest
//     parent chains.
//   - Any active node whose parent was a contracted child must now point to the
//     new blossom node.
//   - Any active node whose tree root was a contracted child must now use the
//     new blossom's tree root.
//
// Implementation:
//   - Stage 1: Build the contracted child-node set from cycle metadata.
//   - Stage 2: Validate the new blossom root.
//   - Stage 3: Redirect active nodes whose parent is one of the contracted children.
//   - Stage 4: Redirect active nodes whose treeRoot is one of the contracted children.
//   - Stage 5: Validate redirected parent edges against the new blossom ownership.
//
// Behavior highlights:
//   - Does not mutate mate[].
//   - Does not clear child cycle metadata.
//   - Preserves labelEdge[] because the original edge is still valid against the
//     new blossom's member ownership.
//   - Prevents pathToRoot/nodePathToRoot from walking into inactive children.
//
// Inputs:
//   - node: newly active contracted blossom node.
//
// Returns:
//   - error: nil after forest parent/root closure is restored.
//
// Errors:
//   - ErrInvalidMatching for malformed cycle metadata, invalid roots, invalid parent
//     references, or label edges that no longer orient to the contracted blossom.
//
// Determinism:
//   - Fixed node-index scan.
//
// Complexity:
//   - Time O(k + n*m), Space O(n), where k is blossom cycle size and m is membership scan cost.
//
// AI-Hints:
//   - Do not reset the whole forest here.
//   - Do not rewrite mate[] here.
//   - Do not clear inactive child labels; expansion/lifting use cycle metadata, not active labels.
func (e *blossomEngine) redirectForestThroughContractedBlossom(node int) error {
	if node < e.problem.n || node >= len(e.active) || !e.active[node] {
		return ErrInvalidMatching
	}

	steps, err := e.cycleSteps(node)
	if err != nil {
		return err
	}

	newRoot := e.treeRoot[node]
	if newRoot == noNode || newRoot < 0 || newRoot >= len(e.active) {
		return ErrInvalidMatching
	}
	if !e.active[newRoot] {
		return ErrInvalidMatching
	}

	contracted := make([]bool, len(e.active))
	for _, step := range steps {
		child := step.node
		if child < 0 || child >= len(e.active) {
			return ErrInvalidMatching
		}
		if child == node {
			return ErrInvalidMatching
		}

		contracted[child] = true
	}

	for current := range e.active {
		if current == node || !e.active[current] {
			continue
		}

		parent := e.parent[current]
		if parent != noNode {
			if parent < 0 || parent >= len(e.active) {
				return ErrInvalidMatching
			}

			if contracted[parent] {
				if e.labelEdge[current] == noEdge {
					return ErrInvalidMatching
				}
				if _, _, err = e.orientEdgeForNodes(e.labelEdge[current], current, node); err != nil {
					return err
				}

				e.parent[current] = node
			}
		}

		root := e.treeRoot[current]
		if root != noNode {
			if root < 0 || root >= len(e.active) {
				return ErrInvalidMatching
			}

			if contracted[root] {
				e.treeRoot[current] = newRoot
			}
		}
	}

	return nil
}
