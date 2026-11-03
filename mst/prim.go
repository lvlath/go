// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mst contains Prim's Minimum Spanning Tree implementation in this file.
// The algorithm consumes incident undirected edges from core.Graph without assuming edge.To is the next vertex.
package mst

import (
	"container/heap"

	"github.com/lvlath/go/core"
)

// primCandidate stores one frontier edge together with the unvisited endpoint it reaches.
// It is the heap payload used by Prim to avoid assuming that core.Edge.To is always the next vertex.
//
// Implementation:
//   - Stage 1: enqueuePrimSnapshotFrontier resolves target relative to the current source.
//   - Stage 2: heap.Push stores the detached edge value and resolved target together.
//   - Stage 3: primKernel discards stale candidates whose target was already visited.
//
// Behavior highlights:
//   - edge is a detached core.Edge value from mstSnapshot.
//   - target is the endpoint that was unvisited at enqueue time.
//
// Fields:
//   - edge: candidate edge value.
//   - target: resolved unvisited endpoint.
//
// Determinism:
//   - Heap ordering is controlled by primFrontier.Less.
//
// Complexity:
//   - Space O(1) per queued candidate.
//
// AI-Hints:
//   - Do not remove target and later use edge.To; undirected traversal depends on the source vertex.
type primCandidate struct {
	// edge is the detached candidate edge considered for inclusion in the MST/MSF.
	edge core.Edge

	// target is the endpoint that was unvisited when this candidate entered the heap.
	target string
}

// primFrontier implements heap.Interface for Prim frontier candidates.
// It orders candidates by finite Weight and then unique Edge.ID for deterministic equal-weight selection.
//
// Implementation:
//   - Stage 1: Len, Less, Swap provide heap ordering.
//   - Stage 2: Push appends a primCandidate supplied by heap.Push.
//   - Stage 3: Pop removes the tail after heap.Pop moves the minimum there.
//
// Behavior highlights:
//   - Non-finite weights are rejected by newMSTSnapshot before candidates reach the heap.
//   - Equal-weight choices are deterministic because Edge.ID is unique.
//   - The target field is traversal metadata, not a tie-break key.
//
// Determinism:
//   - Less uses Weight, then Edge.ID.
//   - No map iteration participates in heap ordering.
//
// Complexity:
//   - Len/Less/Swap are O(1); heap Push/Pop are O(log E).
//
// AI-Hints:
//   - Do not add source/target tie-breaks after Edge.ID; unique edge IDs already close the order.
type primFrontier []primCandidate

// Len returns the number of candidate edges currently stored in the heap.
//
// Complexity:
//   - Time O(1), Space O(1).
func (pf primFrontier) Len() int { return len(pf) }

// Less reports whether candidate i has higher heap priority than candidate j.
// Lower finite weight wins; equal weights are broken by unique Edge.ID.
//
// Complexity:
//   - Time O(1), Space O(1).
func (pf primFrontier) Less(i, j int) bool {
	left := pf[i].edge
	right := pf[j].edge

	return lessMSTEdge(left, right)
}

// Swap exchanges two heap candidates in O(1) time.
//
// Complexity:
//   - Time O(1), Space O(1).
func (pf primFrontier) Swap(i, j int) { pf[i], pf[j] = pf[j], pf[i] }

// Push appends one primCandidate to the frontier.
// The container/heap package calls Push and then restores heap order.
//
// Complexity:
//   - Amortized append O(1); heap.Push as a whole is O(log E).
func (pf *primFrontier) Push(value any) {
	*pf = append(*pf, value.(primCandidate))
}

// Pop removes and returns the heap tail after heap.Pop moves the minimum candidate there.
// It clears the removed slot to avoid retaining detached edge data longer than needed.
//
// Complexity:
//   - Time O(1) for this method; heap.Pop as a whole is O(log E).
func (pf *primFrontier) Pop() any {
	old := *pf
	lastIndex := len(old) - 1
	candidate := old[lastIndex]
	old[lastIndex] = primCandidate{}
	*pf = old[:lastIndex]

	return candidate
}

// primKernel computes a minimum spanning tree or forest using Prim's frontier policy.
//
// Implementation:
//   - Stage 1: Validate root policy against the snapshot.
//   - Stage 2: Grow the explicit root component in strict mode.
//   - Stage 3: In forest mode, grow remaining components in vertex order.
//   - Stage 4: Enforce strict connectivity before checked ordered total publication.
//
// Behavior highlights:
//   - Strict Prim requires a root.
//   - Forest Prim may omit root and then starts from the first vertex in core.Vertices() order.
//   - Heap ordering is by finite Weight, then Edge.ID.
//   - Stale heap candidates are discarded after their target is already visited.
//
// Inputs:
//   - snapshot: validated MST snapshot.
//   - cfg: finalized option policy.
//
// Returns:
//   - *Result: canonical detached result.
//   - error: ErrDisconnected in strict tree mode when not all vertices connect.
//
// Errors:
//   - core.ErrVertexNotFound when the requested root is absent.
//   - ErrDisconnected when strict mode cannot reach all vertices.
//   - ErrWeightOverflow when the selected-edge total has a non-finite prefix.
//
// Determinism:
//   - Primary root is explicit or the first unvisited vertex in snapshot order.
//   - Secondary forest roots follow snapshot vertex order.
//   - Frontier ties are resolved by Edge.ID.
//
// Complexity:
//   - Time O(V + E log(E+1)), Space O(V + E), after snapshot preparation.
//
// AI-Hints:
//   - Do not derive target from edge.to alone; the same undirected edge is stored in both adjacency lists.
func primKernel(snapshot *mstSnapshot, cfg Options) (*Result, error) {
	vertexCount := len(snapshot.vertices)

	result := &Result{
		Algorithm:      AlgorithmPrim,
		Mode:           cfg.Mode,
		Root:           cfg.Root,
		Edges:          make([]core.Edge, 0, maxMSTEdgeCapacity(vertexCount)),
		VertexCount:    vertexCount,
		ComponentRoots: make([]string, 0, 1),
	}

	// A single-vertex graph has an empty spanning tree; optional forest mode may infer the only root.
	if vertexCount == 1 {
		root := cfg.Root
		if root == "" {
			root = snapshot.vertices[0]
		}

		// A non-empty root still must refer to the only vertex in the snapshot.
		if !snapshot.hasVertex(root) {
			return nil, core.ErrVertexNotFound
		}

		result.Root = root
		result.ComponentCount = 1
		result.ComponentRoots = []string{root}
		return result, nil
	}

	// Validate the explicit Prim root before any traversal starts.
	if cfg.Root != "" && !snapshot.hasVertex(cfg.Root) {
		return nil, core.ErrVertexNotFound
	}

	visited := make(map[string]bool, vertexCount)

	// Grow the requested root component first when a root was supplied.
	if cfg.Root != "" {
		growPrimComponent(snapshot, cfg.Root, visited, result)
	}

	// Strict mode must produce exactly one spanning tree over all vertices.
	if cfg.Mode == ModeStrictTree {
		if len(result.Edges) != vertexCount-1 {
			return nil, ErrDisconnected
		}
		result.ComponentCount = 1
		if len(result.ComponentRoots) == 0 {
			result.ComponentRoots = []string{cfg.Root}
		}
		// Structural disconnection wins over an unrepresentable scalar total.
		total, err := checkedMSTEdgeTotal(result.Edges)
		if err != nil {
			return nil, err
		}
		result.TotalWeight = total
		return result, nil
	}

	// Forest mode resumes from every still-unvisited vertex in deterministic snapshot order.
	for _, root := range snapshot.vertices {
		// Skip vertices already covered by the explicit root component or an earlier forest component.
		if visited[root] {
			continue
		}

		// Grow one minimum spanning tree for this connected component.
		growPrimComponent(snapshot, root, visited, result)
	}

	// Finalize forest metadata after every component has been grown.
	result.ComponentCount = len(result.ComponentRoots)
	if result.Root == "" && len(result.ComponentRoots) > 0 {
		result.Root = result.ComponentRoots[0]
	}

	// Publish only a finite total in the same edge-acceptance order as before.
	total, err := checkedMSTEdgeTotal(result.Edges)
	if err != nil {
		return nil, err
	}
	result.TotalWeight = total
	return result, nil
}

// growPrimComponent grows one Prim component from root and appends selected edges into result.
// It owns the frontier traversal and leaves cost publication to primKernel after connectivity.
//
// Implementation:
//   - Stage 1: Register root as a public component root.
//   - Stage 2: Initialize an empty frontier heap.
//   - Stage 3: Mark root visited and enqueue all frontier candidates.
//   - Stage 4: Pop candidates by priority, skipping stale targets.
//   - Stage 5: Accept the lightest valid edge without summing and expand its target.
//
// Behavior highlights:
//   - Stale candidates are expected when multiple edges reach a vertex before it is visited.
//   - Accepted edges are detached values copied from mstSnapshot.
//   - Accepted edge order is preserved; TotalWeight is checked after the component is built.
//
// Inputs:
//   - snapshot: validated MST snapshot.
//   - root: component root to grow from.
//   - visited: shared visited set across strict/forest traversal.
//   - result: mutable result artifact under construction.
//
// Determinism:
//   - Initial adjacency scan follows snapshot adjacency order.
//   - Candidate extraction follows primFrontier ordering.
//   - Component root registration follows caller-provided root order.
//
// Complexity:
//   - Time O(Ec log E), where Ec is the scanned edge count for the component.
//   - Space O(Ec) in the frontier at worst.
//
// Notes:
//   - This helper is a kernel helper and assumes inputs were validated by primKernel.
//
// AI-Hints:
//   - Do not append result edges before checking visited[target]; stale heap candidates must not form cycles.
func growPrimComponent(snapshot *mstSnapshot, root string, visited map[string]bool, result *Result) {
	result.ComponentRoots = append(result.ComponentRoots, root)

	frontier := &primFrontier{}
	heap.Init(frontier)

	visited[root] = true
	enqueuePrimSnapshotFrontier(snapshot, root, visited, frontier)

	for frontier.Len() > 0 {
		candidate := heap.Pop(frontier).(primCandidate)
		if visited[candidate.target] {
			continue
		}

		visited[candidate.target] = true
		result.Edges = append(result.Edges, candidate.edge)

		enqueuePrimSnapshotFrontier(snapshot, candidate.target, visited, frontier)
	}
}

// enqueuePrimSnapshotFrontier pushes all candidate edges from source to unvisited endpoints.
// It resolves the opposite endpoint relative to source, which is required for undirected edge values.
//
// Implementation:
//   - Stage 1: Scan snapshot.adj[source] in deterministic order.
//   - Stage 2: Resolve the endpoint opposite to source.
//   - Stage 3: Skip already visited targets.
//   - Stage 4: Push a primCandidate carrying the edge value and resolved target.
//
// Behavior highlights:
//   - No graph calls happen here; all data comes from mstSnapshot.
//   - The helper does not allocate a secondary edge list.
//   - Invalid endpoint pairs are ignored because the snapshot adapter owns adjacency construction.
//
// Inputs:
//   - snapshot: validated MST snapshot.
//   - source: visited vertex whose incident edges are scanned.
//   - visited: current Prim visited set.
//   - frontier: heap receiving candidates.
//
// Determinism:
//   - Scan order follows snapshot adjacency order.
//   - Extraction order is controlled later by primFrontier.Less.
//
// Complexity:
//   - Time O(deg(source) log E), Space O(k) for pushed candidates.
//
// AI-Hints:
//   - Keep endpoint resolution relative to source; simplifying this to edge.To reintroduces the P0 bug.
func enqueuePrimSnapshotFrontier(snapshot *mstSnapshot, source string, visited map[string]bool, frontier *primFrontier) {
	for _, edge := range snapshot.adj[source] {
		var target string

		switch source {
		case edge.From:
			target = edge.To
		case edge.To:
			target = edge.From
		default:
			continue
		}

		if visited[target] {
			continue
		}

		heap.Push(frontier, primCandidate{
			edge:   edge,
			target: target,
		})
	}
}
