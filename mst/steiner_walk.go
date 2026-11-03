// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mst defines deterministic edge-covering walks on validated Steiner trees.
// This file implements iterative spine traversal and backtracking over off-spine branches.
package mst

import (
	"fmt"
	"slices"
	"unsafe"

	"github.com/lvlath/go/core"
)

// steinerWalkUnvisitedParent distinguishes vertices not yet discovered by spine search.
// A valid local vertex index is nonnegative, so this sentinel cannot equal a real parent.
const steinerWalkUnvisitedParent = -1

// steinerWalkFrame stores the resumption point of one iterative off-spine DFS level.
// Each frame remembers the only edge leading back toward its parent and its next arc.
// Implementation:
//   - Stage 1: Push a child after its forward edge passage is recorded.
//   - Stage 2: Advance nextNeighbor while scanning canonical sparse adjacency.
//   - Stage 3: Record one return passage when the frame exhausts its children.
//
// Behavior highlights:
//   - The inbound edge is excluded during child expansion, preventing immediate backtracking.
//
// Inputs:
//   - vertex, parent: indexes of adjacent vertices in the prepared tree.
//   - inboundEdge: original edge index used to enter vertex; nextNeighbor: adjacency cursor.
//
// Returns:
//   - steinerWalkFrame: private traversal state with no heap pointers into the original graph.
//
// Errors:
//   - None in the frame itself; passage arithmetic may fail in the traversal.
//
// Determinism:
//   - Each cursor advances through sorted adjacency exactly once.
//
// Complexity:
//   - Space O(1) per active DFS depth, O(V) at maximum depth.
//
// Notes:
//   - Frames are reused between branches; they do not accumulate across completed subtrees.
//
// AI-Hints:
//   - Do not replace the explicit stack with recursion on potentially deep user-supplied trees.
type steinerWalkFrame struct {
	// vertex is the current branch vertex whose children remain to be processed.
	vertex int
	// parent is the vertex to revisit when this branch frame finishes.
	parent int
	// inboundEdge identifies the original edge connecting vertex to parent.
	inboundEdge int
	// nextNeighbor is the next canonical adjacency index that has not yet been inspected.
	nextNeighbor int
}

// SteinerWalk produces a minimum-distance edge-covering walk on the supplied tree.
// It covers every tree edge and never asserts that the route is globally optimal in the original graph.
// Implementation:
//   - Stage 1: Validate explicit endpoints and reconstruct a detached, canonical tree workspace.
//   - Stage 2: Find the unique start-to-end spine and iteratively traverse every off-spine branch.
//   - Stage 3: Record each original edge passage, check its numeric cost, and publish a complete route.
//
// Behavior highlights:
//   - Equal endpoints produce a closed walk traversing each edge in both directions.
//   - Different endpoints traverse spine edges once and every other edge twice.
//   - The input may be any structurally valid tree, including one with nonterminal leaves.
//   - NetworkWeight counts unique edges; Distance is a checked fold over the actual passage sequence.
//
// Inputs:
//   - tree: caller-owned, non-nil SteinerResult; not mutated during this call or by this function.
//   - startID, endID: explicit, nonempty original vertex IDs present in tree.Vertices.
//
// Returns:
//   - *WalkResult: owned ordered Vertices/EdgeIDs with len(Vertices)=len(EdgeIDs)+1;
//     nil if any validation or arithmetic step fails, without publishing a partial route.
//
// Errors:
//   - ErrNilResult or core.ErrEmptyVertexID for invalid request shape;
//     ErrInvalidSteinerResult and original core/domain causes for a malformed tree;
//     core.ErrVertexNotFound for a missing endpoint; ErrWeightOverflow for a nonfinite
//     actual passage sum; ErrSizeOverflow when output or index sizes are unrepresentable.
//
// Determinism:
//   - Canonical local edge-ID order and per-vertex (neighbor ID, Edge.ID) adjacency;
//     branches are exhausted before the next spine edge is traversed.
//
// Complexity:
//   - Time O(V log V + E log E + sum_v deg(v)log(deg(v)+1) + S),
//     Space O(V+E+S), including validation/copies and S output passages, S<=2E.
//
// Notes:
//   - The mathematical cost is 2W-d_tree(start,end), but no such potentially
//     overflowing intermediate formula is used to generate or price the route.
//   - This is a synchronous operation without context; callers must not mutate the tree concurrently.
//
// AI-Hints:
//   - Never deduplicate repeated EdgeIDs or replace actual traversals with a scalar distance formula.
//   - Do not prune nonterminal leaves: covering the given tree requires visiting their edges.
//   - Original graph shortcuts are outside the tree-covering problem and are not consulted.
func SteinerWalk(tree *SteinerResult, startID, endID string) (*WalkResult, error) {
	if tree == nil {
		return nil, ErrNilResult
	}
	if startID == "" || endID == "" {
		return nil, fmt.Errorf("mst: walk endpoints must be nonempty: %w", core.ErrEmptyVertexID)
	}

	// The public SteinerResult is mutable. Copy and verify its complete shape,
	// not merely its TotalWeight or a possibly stale set of terminal IDs.
	prepared, err := prepareSteinerWalkTree(tree)
	if err != nil {
		return nil, err
	}
	start, startFound := prepared.indices[startID]
	if !startFound {
		return nil, fmt.Errorf("mst: walk start vertex %q: %w", startID, core.ErrVertexNotFound)
	}
	end, endFound := prepared.indices[endID]
	if !endFound {
		return nil, fmt.Errorf("mst: walk end vertex %q: %w", endID, core.ErrVertexNotFound)
	}

	// The unique spine separates edges needed only once (on the spine) from
	// attached branches that must be entered and left along the same edges.
	spineVertices, spineEdges, onSpine, err := findSteinerWalkSpine(prepared, start, end)
	if err != nil {
		return nil, err
	}
	edgeCount := len(prepared.edges)
	if len(spineEdges) > edgeCount {
		return nil, ErrInvalidSteinerResult
	}

	// A tree edge is visited once plus one additional time if it is not on
	// the fixed-end spine. Checked addition avoids overflowing an unused 2*E.
	passages, err := checkedSteinerCountAdd(edgeCount, edgeCount-len(spineEdges))
	if err != nil {
		return nil, err
	}
	vertexVisits, err := checkedSteinerCountAdd(passages, 1)
	if err != nil {
		return nil, err
	}
	if _, err := checkedSteinerSliceCapacity(passages, unsafe.Sizeof("")); err != nil {
		return nil, err
	}
	if _, err := checkedSteinerSliceCapacity(vertexVisits, unsafe.Sizeof("")); err != nil {
		return nil, err
	}

	result := &WalkResult{
		StartID:       startID,
		EndID:         endID,
		NetworkWeight: prepared.networkWeight,
		Vertices:      make([]string, 1, vertexVisits),
		EdgeIDs:       make([]string, 0, passages),
	}
	result.Vertices[0] = startID

	// Reuse the same explicit DFS storage across all branches. In particular,
	// a star must not allocate one new traversal stack for each leaf.
	var frames []steinerWalkFrame
	for spineIndex, vertex := range spineVertices {
		for _, arc := range prepared.adjacency[vertex] {
			// Spine passages are scheduled separately and exactly once. All
			// remaining incident edges lead into disjoint off-spine subtrees.
			if onSpine[arc.edge] {
				continue
			}
			if err := traverseSteinerWalkBranch(prepared, result, vertex, arc, &frames); err != nil {
				return nil, err
			}
		}

		// Moving to the next spine vertex is deliberately not followed by a
		// return: this single passage is what fixes the requested end vertex.
		if spineIndex < len(spineEdges) {
			if err := appendSteinerWalkPassage(prepared, result, vertex,
				spineVertices[spineIndex+1], spineEdges[spineIndex]); err != nil {
				return nil, err
			}
		}
	}

	// These invariants are inexpensive publication guards against traversal
	// mistakes, not an attempt to repair or silently truncate a partial walk.
	if len(result.EdgeIDs) != passages || len(result.Vertices) != vertexVisits ||
		result.Vertices[len(result.Vertices)-1] != endID {
		return nil, ErrInvalidSteinerResult
	}
	return result, nil
}

// findSteinerWalkSpine locates the unique path between two vertices in a validated tree.
// It marks each spine edge so that the covering traversal can omit its return passage.
// Implementation:
//   - Stage 1: Handle equal endpoints with a one-vertex spine and no spine edges.
//   - Stage 2: Discover parents using one iterative queue over canonical adjacency.
//   - Stage 3: Reverse the unique predecessor chain and mark its original edge indexes.
//
// Behavior highlights:
//   - Since the input is a tree, each discovered parent is the unique path predecessor.
//   - Discovery stops as soon as end is reached; no repeated BFS is performed per branch.
//
// Inputs:
//   - tree: validated and detached tree; start, end: existing local vertex indexes.
//
// Returns:
//   - []int: ordered spine vertices; []int: corresponding edge indexes;
//     []bool: membership mask for all edges; error: non-nil only on violated invariants/size.
//
// Errors:
//   - ErrInvalidSteinerResult if the verified tree's endpoint connection cannot be reconstructed;
//     ErrSizeOverflow for unrepresentable index-array backing storage.
//
// Determinism:
//   - First discovery in canonical neighbor order, though a tree has only one simple path.
//
// Complexity:
//   - Time O(V+E), Space O(V+E), including BFS and the spine-edge mask.
//
// Notes:
//   - No shortest-path package is needed: every pair in a tree has a unique simple path.
//
// AI-Hints:
//   - Never search the original graph: shortcuts would change the edge-covering tree objective.
//   - Do not recompute a route for each off-spine branch.
func findSteinerWalkSpine(
	tree *steinerWalkTree,
	start, end int,
) ([]int, []int, []bool, error) {
	spineMask := make([]bool, len(tree.edges))
	if start == end {
		return []int{start}, nil, spineMask, nil
	}
	vertexCount := len(tree.vertices)
	if _, err := checkedSteinerSliceCapacity(vertexCount, unsafe.Sizeof(int(0))); err != nil {
		return nil, nil, nil, err
	}

	parent := make([]int, vertexCount)
	parentEdge := make([]int, vertexCount)
	for index := range parent {
		parent[index] = steinerWalkUnvisitedParent
	}
	parent[start] = start
	queue := make([]int, 1, vertexCount)
	queue[0] = start

	// A head cursor prevents O(V^2) shifting and avoids retaining popped
	// queue prefixes. Parent assignment doubles as the visited marker.
	for head := 0; head < len(queue) && parent[end] == steinerWalkUnvisitedParent; head++ {
		vertex := queue[head]
		for _, arc := range tree.adjacency[vertex] {
			if parent[arc.to] != steinerWalkUnvisitedParent {
				continue
			}
			parent[arc.to] = vertex
			parentEdge[arc.to] = arc.edge
			queue = append(queue, arc.to)
			if arc.to == end {
				break
			}
		}
	}
	if parent[end] == steinerWalkUnvisitedParent {
		return nil, nil, nil, ErrInvalidSteinerResult
	}

	// Reconstruct backwards once. Every parent edge is marked for the later
	// branch DFS, which must never cross a spine edge twice.
	spineVertices := make([]int, 0)
	spineEdges := make([]int, 0)
	for vertex := end; vertex != start; vertex = parent[vertex] {
		spineVertices = append(spineVertices, vertex)
		edgeIndex := parentEdge[vertex]
		spineEdges = append(spineEdges, edgeIndex)
		spineMask[edgeIndex] = true
	}
	spineVertices = append(spineVertices, start)
	slices.Reverse(spineVertices)
	slices.Reverse(spineEdges)
	return spineVertices, spineEdges, spineMask, nil
}

// traverseSteinerWalkBranch explores an off-spine subtree and returns to its attachment.
// Every branch edge contributes exactly two original-ID passages without recursive calls.
// Implementation:
//   - Stage 1: Record the incoming attachment edge and push its child frame.
//   - Stage 2: Expand each non-parent adjacency edge in deterministic order.
//   - Stage 3: Record one backtrack passage for every exhausted child and reuse the stack.
//
// Behavior highlights:
//   - The validated tree has no alternate cycles, so excluding the inbound edge suffices.
//   - Frames from finished branches are discarded without discarding the reusable backing array.
//
// Inputs:
//   - tree: immutable validated workspace; result: private route under construction.
//   - attachment: spine vertex index; first: arc leading to a distinct branch subtree.
//   - scratch: caller-owned stack reused across branches, not retained in the returned result.
//
// Returns:
//   - error: nil only after returning to attachment; result has two passages per branch edge.
//
// Errors:
//   - ErrWeightOverflow for an actual passage overflow;
//     ErrInvalidSteinerResult if an internal traversal-incidence invariant fails.
//
// Determinism:
//   - Depth-first traversal in sorted adjacency order, then the reverse parent edge.
//
// Complexity:
//   - Time O(Vb+Eb), additional live stack O(depth) for one branch subtree.
//
// Notes:
//   - The caller discards the entire private result on any failure; no partial walk escapes.
//
// AI-Hints:
//   - Do not append all outward edges first and then manufacture return IDs without traversing.
//   - Reuse scratch so a high-degree hub does not create one allocation per leaf.
func traverseSteinerWalkBranch(
	tree *steinerWalkTree,
	result *WalkResult,
	attachment int,
	first steinerWalkArc,
	scratch *[]steinerWalkFrame,
) error {
	if err := appendSteinerWalkPassage(tree, result, attachment, first.to, first.edge); err != nil {
		return err
	}
	frames := (*scratch)[:0]
	frames = append(frames, steinerWalkFrame{
		vertex: first.to, parent: attachment, inboundEdge: first.edge,
	})

	for len(frames) > 0 {
		top := len(frames) - 1
		vertex := frames[top].vertex
		if frames[top].nextNeighbor < len(tree.adjacency[vertex]) {
			arc := tree.adjacency[vertex][frames[top].nextNeighbor]
			frames[top].nextNeighbor++
			if arc.edge == frames[top].inboundEdge {
				continue
			}
			if err := appendSteinerWalkPassage(tree, result, vertex, arc.to, arc.edge); err != nil {
				return err
			}
			frames = append(frames, steinerWalkFrame{
				vertex: arc.to, parent: vertex, inboundEdge: arc.edge,
			})
			continue
		}

		// All children of this vertex have returned; follow the same original
		// edge back toward the attachment and then release the completed frame.
		completed := frames[top]
		frames = frames[:top]
		if err := appendSteinerWalkPassage(tree, result,
			completed.vertex, completed.parent, completed.inboundEdge); err != nil {
			return err
		}
	}
	*scratch = frames
	return nil
}

// appendSteinerWalkPassage records one correctly oriented original-edge traversal.
// It calculates the cost from the recorded edge before appending either route slice.
// Implementation:
//   - Stage 1: Confirm that the current route vertex and requested edge endpoints agree.
//   - Stage 2: Apply the existing checked float64 edge-cost addition to the travel prefix.
//   - Stage 3: Append exactly one original Edge.ID and its reached vertex, then update Distance.
//
// Behavior highlights:
//   - Either orientation of an undirected edge is admitted; the stored Edge.To is not privileged.
//   - The successful path cost uses the actual visitation order, never a 2W-distance formula.
//
// Inputs:
//   - tree: validated edge/vertex inventory; result: private mutable walk state.
//   - from, to: indexes of consecutive vertices; edgeIndex: original edge used to connect them.
//
// Returns:
//   - error: nil after one valid passage; no partial append if the numeric check fails.
//
// Errors:
//   - ErrInvalidSteinerResult for inconsistent route position or original-edge incidence;
//     ErrWeightOverflow for a nonfinite next travel prefix.
//
// Determinism:
//   - The caller fixes passage order; this helper records one event without reordering.
//
// Complexity:
//   - Time amortized O(1), Space O(1) beyond preallocated route storage.
//
// Notes:
//   - All public output slices belong to the private result until full publication.
//
// AI-Hints:
//   - Always append both the Edge.ID and reached vertex, or the witness lengths diverge.
//   - Do not reverse Edge.From/To in a detached record merely because the passage is backward.
func appendSteinerWalkPassage(
	tree *steinerWalkTree,
	result *WalkResult,
	from, to, edgeIndex int,
) error {
	if from < 0 || to < 0 || from >= len(tree.vertices) || to >= len(tree.vertices) ||
		edgeIndex < 0 || edgeIndex >= len(tree.edges) || len(result.Vertices) == 0 {
		return ErrInvalidSteinerResult
	}
	fromID, toID := tree.vertices[from], tree.vertices[to]
	edge := tree.edges[edgeIndex]
	if result.Vertices[len(result.Vertices)-1] != fromID ||
		!((edge.From == fromID && edge.To == toID) ||
			(edge.To == fromID && edge.From == toID)) {
		return ErrInvalidSteinerResult
	}

	// Compute before mutating output: a mathematically valid tree may still
	// need an unrepresentable repeated passage sum for this endpoint choice.
	nextDistance, err := checkedMSTWeightAdd(result.Distance, edge.Weight)
	if err != nil {
		return fmt.Errorf("mst: walk passage edge_id=%q from=%q to=%q: %w",
			edge.ID, fromID, toID, err)
	}
	result.EdgeIDs = append(result.EdgeIDs, edge.ID)
	result.Vertices = append(result.Vertices, toID)
	result.Distance = nextDistance
	return nil
}
