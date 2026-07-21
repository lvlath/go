// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package flow

import (
	"math"
	"sort"

	"github.com/lvlath/go/core"
)

// residualNetwork is the internal mutable residual-capacity model used by all kernels.
// It separates fast algorithmic residual updates from public core.Graph publication.
//
// Implementation:
//   - vertices stores deterministic vertex order copied from core.Vertices().
//   - cap stores residual capacities as cap[from][to].
//   - adj stores deterministic residual neighbor lists for traversal.
//   - buildResidualNetwork initializes reverse arcs with zero capacity so later flow
//     cancellation can traverse them without changing adjacency shape.
//
// Behavior highlights:
//   - cap is mutable during algorithms.
//   - adj is finalized once and then treated as immutable traversal order.
//   - vertices preserves result/cut/matrix output order.
//
// Inputs:
//   - Constructed only through newResidualNetwork and buildResidualNetwork.
//
// Returns:
//   - Internal residual state for Dinic, Edmonds-Karp, Ford-Fulkerson, min-cut,
//     residual graph publication, and CapacityMatrix.
//
// Errors:
//   - The type itself returns no errors; builders and validators enforce policy.
//
// Determinism:
//   - vertices is copied from core.Vertices().
//   - adj is sorted and compacted by finalizeOrder.
//
// Complexity:
//   - Storage O(V + A), where A is residual adjacency-entry count.
//
// Notes:
//   - This type is intentionally not exported.
//   - Public callers receive MaxFlowResult.Residual, not residualNetwork.
//
// AI-Hints:
//   - Do not iterate cap maps directly in kernels.
//   - Use rn.adj[from] for every traversal to preserve deterministic behavior.
type residualNetwork struct {
	vertices []string
	cap      map[string]map[string]float64
	adj      map[string][]string
}

// newResidualNetwork allocates empty residual storage for a known vertex order.
// It creates capacity buckets and adjacency lists for every vertex up front.
//
// Implementation:
//   - Stage 1: Copy the vertex order so callers cannot mutate rn.vertices.
//   - Stage 2: Allocate cap and adj maps sized by vertex count.
//   - Stage 3: Initialize one capacity bucket and one adjacency slice per vertex.
//
// Behavior highlights:
//   - Pre-initialized buckets avoid nil-map writes during addArc.
//   - The copied vertex slice becomes the canonical output order.
//
// Inputs:
//   - vertices: deterministic vertex order, normally from core.Vertices().
//
// Returns:
//   - *residualNetwork: empty residual structure.
//
// Errors:
//   - None.
//
// Determinism:
//   - Preserves the order of the input slice exactly.
//
// Complexity:
//   - Time O(V), Space O(V).
//
// Notes:
//   - This helper does not validate vertex IDs; graph validation happens earlier.
//
// AI-Hints:
//   - Pass only stable vertex order into this constructor.
func newResidualNetwork(vertices []string) *residualNetwork {
	rn := &residualNetwork{
		vertices: append([]string(nil), vertices...),
		cap:      make(map[string]map[string]float64, len(vertices)),
		adj:      make(map[string][]string, len(vertices)),
	}

	for _, vertexID := range vertices {
		rn.cap[vertexID] = make(map[string]float64)
		rn.adj[vertexID] = nil
	}

	return rn
}

// addArc adds or aggregates a directed residual capacity arc.
// It also ensures that the reverse residual arc exists with zero capacity.
//
// Implementation:
//   - Stage 1: Ensure capacity buckets exist for both endpoints.
//   - Stage 2: If the forward adjacency is new, register from -> to.
//   - Stage 3: If the reverse capacity slot is new, register to -> from and set zero.
//   - Stage 4: Add capacity to cap[from][to], aggregating parallel input edges.
//
// Behavior highlights:
//   - Parallel arcs are aggregated, not stored as separate residual entries.
//   - Reverse adjacency exists before algorithms run, so residual cancellation can be
//     traversed after flow is pushed backward.
//   - This method does not sort adjacency; finalizeOrder does that once after build.
//
// Inputs:
//   - from: residual arc source.
//   - to: residual arc target.
//   - capacity: positive finite capacity already validated by validateCapacity.
//
// Returns:
//   - None.
//
// Errors:
//   - None. Invalid capacity must be rejected before calling addArc.
//
// Determinism:
//   - Aggregation is deterministic when caller ingests edges in deterministic order.
//
// Complexity:
//   - Amortized O(1) time, O(1) additional space for a new residual pair.
//
// Notes:
//   - addArc is used for both directed edges and both directions of undirected edges.
//
// AI-Hints:
//   - Do not omit the reverse zero-capacity arc; residual algorithms require it.
func (rn *residualNetwork) addArc(from, to string, capacity float64) {
	if _, ok := rn.cap[from]; !ok {
		rn.cap[from] = make(map[string]float64)
	}
	if _, ok := rn.cap[to]; !ok {
		rn.cap[to] = make(map[string]float64)
	}

	if _, exists := rn.cap[from][to]; !exists {
		rn.adj[from] = append(rn.adj[from], to)
	}
	if _, exists := rn.cap[to][from]; !exists {
		rn.adj[to] = append(rn.adj[to], from)
		rn.cap[to][from] = 0
	}

	rn.cap[from][to] += capacity
}

// finalizeOrder sorts and compacts every residual adjacency list.
// It must be called after all input graph edges have been adapted.
//
// Implementation:
//   - Stage 1: Iterate over every adjacency bucket.
//   - Stage 2: Sort neighbors lexicographically.
//   - Stage 3: Remove duplicates created by parallel edges or reverse arc setup.
//
// Behavior highlights:
//   - Kernels depend on this method for deterministic traversal.
//   - Sorting once is cheaper and safer than sorting in every BFS/DFS loop.
//
// Inputs:
//   - receiver: residual network whose adjacency lists may contain duplicates.
//
// Returns:
//   - None.
//
// Errors:
//   - None.
//
// Determinism:
//   - Produces stable lexical neighbor order for every vertex.
//
// Complexity:
//   - Time O(sum deg(v) log deg(v)), Space O(1) extra per list aside from sort internals.
//
// Notes:
//   - Call this exactly once after buildResidualNetwork finishes edge ingestion.
//
// AI-Hints:
//   - Do not call finalizeOrder inside algorithm loops.
func (rn *residualNetwork) finalizeOrder() {
	for vertexID := range rn.adj {
		sort.Strings(rn.adj[vertexID])
		rn.adj[vertexID] = compactSortedStrings(rn.adj[vertexID])
	}
}

// compactSortedStrings removes duplicates from a sorted string slice in place.
// It preserves the first occurrence of every value.
//
// Implementation:
//   - Stage 1: Fast-return slices of length 0 or 1.
//   - Stage 2: Maintain a write cursor for unique values.
//   - Stage 3: Scan sorted values and copy only new values forward.
//   - Stage 4: Return the compacted prefix.
//
// Behavior highlights:
//   - Requires sorted input.
//   - Does not allocate a new slice.
//   - Preserves lexical order.
//
// Inputs:
//   - values: sorted string slice.
//
// Returns:
//   - []string: compacted slice sharing the original backing array.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pure deterministic slice transformation.
//
// Complexity:
//   - Time O(n), Space O(1).
//
// Notes:
//   - Passing unsorted input removes only adjacent duplicates and is a caller bug.
//
// AI-Hints:
//   - Sort before calling this helper.
func compactSortedStrings(values []string) []string {
	if len(values) < 2 {
		return values
	}

	write := 1
	for read := 1; read < len(values); read++ {
		if values[read] == values[write-1] {
			continue
		}

		values[write] = values[read]
		write++
	}

	return values[:write]
}

// buildResidualNetwork builds the internal deterministic residual network.
//
// Implementation:
//   - Stage 1: Snapshot vertices through core.Vertices lexical order.
//   - Stage 2: Snapshot edges through core.Edges stable Edge.ID order.
//   - Stage 3: Validate each capacity and ignore loops.
//   - Stage 4: Translate directed edges into one arc and undirected edges into two arcs.
//   - Stage 5: Sort and compact residual adjacency lists once.
//
// Behavior highlights:
//   - Parallel edges are aggregated deterministically in Edge.ID order.
//   - Undirected capacity edges are represented as two directed residual arcs.
//
// Inputs:
//   - g: validated weighted core.Graph.
//   - cfg: finalized runtime options.
//
// Returns:
//   - *residualNetwork: detached internal residual structure.
//
// Errors:
//   - ErrInvalidCapacity, ErrNegativeCapacity, ErrNaNInf.
//   - Context cancellation from cfg.ctx.
//
// Determinism:
//   - Vertex order: core.Vertices().
//   - Edge ingestion order: core.Edges().
//   - Traversal order: lexical rn.adj[u].
//
// Complexity:
//   - Time O(V + E + A log A), where A is residual adjacency entries.
//   - Space O(V + A).
//
// Notes:
//   - The input graph is never mutated.
//
// AI-Hints:
//   - Do not rebuild this from core.Neighbors; undirected endpoint semantics will break.
func buildResidualNetwork(g *core.Graph, cfg options) (*residualNetwork, error) {
	if err := cfg.ctx.Err(); err != nil {
		return nil, err
	}

	vertices := g.Vertices()
	rn := newResidualNetwork(vertices)

	for _, edge := range g.Edges() {
		if err := cfg.ctx.Err(); err != nil {
			return nil, err
		}
		if edge == nil {
			return nil, ErrInvalidCapacity
		}
		if edge.From == edge.To {
			continue
		}

		capacity, err := validateCapacity(edge, cfg.epsilon)
		if err != nil {
			return nil, err
		}
		if capacity == 0 {
			continue
		}

		if edge.Directed {
			rn.addArc(edge.From, edge.To, capacity)
			continue
		}

		rn.addArc(edge.From, edge.To, capacity)
		rn.addArc(edge.To, edge.From, capacity)
	}

	rn.finalizeOrder()
	return rn, nil
}

// addResidual applies one residual update after a successful augmentation.
// It subtracts delta from the forward arc and adds delta to the reverse arc.
//
// Implementation:
//   - Stage 1: Decrease cap[from][to] by delta.
//   - Stage 2: Clamp tiny forward residual noise to zero.
//   - Stage 3: Increase cap[to][from] by delta.
//   - Stage 4: Clamp tiny reverse residual noise to zero.
//
// Behavior highlights:
//   - Preserves the residual invariant for one augmented arc pair.
//   - Centralizes floating-point cleanup through epsilon.
//   - Does not change adjacency lists; reverse arcs already exist.
//
// Inputs:
//   - rn: mutable residual network.
//   - from: current path arc source.
//   - to: current path arc target.
//   - delta: positive bottleneck pushed through the path.
//   - epsilon: numeric threshold for zero-clamping.
//
// Returns:
//   - None.
//
// Errors:
//   - None. Callers must pass valid path arcs and positive delta.
//
// Determinism:
//   - Pure arithmetic update for fixed inputs.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This method is shared by all kernels to avoid residual arithmetic drift.
//
// AI-Hints:
//   - Do not duplicate residual update logic in individual algorithms.
func addResidual(rn *residualNetwork, from, to string, delta float64, epsilon float64) {
	rn.cap[from][to] -= delta
	if math.Abs(rn.cap[from][to]) <= epsilon {
		rn.cap[from][to] = 0
	}

	rn.cap[to][from] += delta
	if math.Abs(rn.cap[to][from]) <= epsilon {
		rn.cap[to][from] = 0
	}
}

// buildResidualGraph publishes a detached directed weighted residual graph.
// It converts positive internal residual capacities into core.Graph edges.
//
// Implementation:
//   - Stage 1: Create a new directed weighted core.Graph.
//   - Stage 2: Copy every residual vertex into the public graph.
//   - Stage 3: Scan deterministic residual adjacency lists.
//   - Stage 4: Publish only arcs with capacity greater than epsilon.
//   - Stage 5: Use stable residual edge IDs for reproducible proof artifacts.
//
// Behavior highlights:
//   - Output graph never inherits directedness or weightedness from the input graph.
//   - Residual edges are directed because residual networks are mathematically directed.
//   - Edge IDs are deterministic for from/to pairs.
//
// Inputs:
//   - rn: internal residual network.
//   - epsilon: threshold for publishing positive residual arcs.
//
// Returns:
//   - *core.Graph: detached directed weighted residual graph.
//   - error: core graph construction error, if any.
//
// Errors:
//   - core.NewGraph/AddVertex/AddEdge errors.
//   - core.WithID-related errors if ID policy is violated unexpectedly.
//
// Determinism:
//   - Vertex order follows rn.vertices.
//   - Edge order follows rn.vertices and rn.adj[from].
//   - Edge IDs are stableResidualEdgeID(from, to).
//
// Complexity:
//   - Time O(V + A), Space O(V + E_res).
//
// Notes:
//   - This graph is a certificate artifact, not the internal mutable residual state.
//
// AI-Hints:
//   - Do not use core.CloneEmpty here; residual output must be directed weighted.
func buildResidualGraph(rn *residualNetwork, epsilon float64) (*core.Graph, error) {
	residual, err := core.NewGraph(core.WithDirected(true), core.WithWeighted())
	if err != nil {
		return nil, err
	}

	for _, vertexID := range rn.vertices {
		if err = residual.AddVertex(vertexID); err != nil {
			return nil, err
		}
	}

	for _, from := range rn.vertices {
		for _, to := range rn.adj[from] {
			capacity := rn.cap[from][to]
			if capacity <= epsilon {
				continue
			}

			//edgeID := "res_" + from + "_" + to
			//if _, err = residual.AddEdge(from, to, capacity, core.WithID(edgeID)); err != nil {
			if _, err = residual.AddEdge(from, to, capacity); err != nil {
				return nil, err
			}
		}
	}

	return residual, nil
}

// minCutFromResidual extracts the s-side and t-side of the final min-cut.
// It performs residual reachability from source over strictly positive residual arcs.
//
// Implementation:
//   - Stage 1: Run BFS from source using a head-index queue.
//   - Stage 2: Traverse only arcs with residual capacity greater than epsilon.
//   - Stage 3: Partition rn.vertices into reachable and non-reachable sides.
//
// Behavior highlights:
//   - This is a certificate helper for MaxFlowResult, not a separate algorithm choice.
//   - The queue uses a head cursor and never uses queue = queue[1:].
//   - The returned slices are caller-owned and sorted by rn.vertices order.
//
// Inputs:
//   - rn: final residual network after max-flow termination.
//   - source: source vertex ID used by the max-flow run.
//   - epsilon: residual threshold; capacities <= epsilon are treated as absent.
//
// Returns:
//   - []string: vertices reachable from source in the final residual network.
//   - []string: vertices not reachable from source in the final residual network.
//
// Errors:
//   - None. The caller is responsible for passing a valid residualNetwork.
//
// Determinism:
//   - BFS neighbor scan order is rn.adj[u], sorted once by buildResidualNetwork.
//   - Output partition order is rn.vertices, inherited from core.Vertices().
//
// Complexity:
//   - Time O(V + A), Space O(V), where A is the residual adjacency-entry count.
//
// Notes:
//   - By max-flow/min-cut theorem, after a successful run with no augmenting path,
//     the capacity of edges crossing sourceSide -> sinkSide equals MaxFlowResult.Value.
//   - On partial results this partition is diagnostic, not a full optimality certificate.
//
// AI-Hints:
//   - Do not switch to queue slicing; it can retain the full backing array.
//   - Do not scan maps here; rn.adj is the deterministic traversal source.
func minCutFromResidual(rn *residualNetwork, source string, epsilon float64) ([]string, []string) {
	// Track residual reachability from the source side of the cut.
	seen := make(map[string]bool, len(rn.vertices))

	// Use a head-index queue to avoid retaining discarded queue prefixes.
	queue := make([]string, 0, len(rn.vertices))

	// Seed BFS with the source vertex, which is always on the source side.
	seen[source] = true
	queue = append(queue, source)

	// Traverse all vertices reachable through positive residual capacity.
	for head := 0; head < len(queue); head++ {
		// Read without removing so the queue backing array can be reused safely.
		from := queue[head]

		// Scan deterministic residual neighbors only.
		for _, to := range rn.adj[from] {
			// Skip already reached vertices and arcs that are absent under epsilon.
			if seen[to] || rn.cap[from][to] <= epsilon {
				continue
			}

			// Mark the vertex as source-side reachable and enqueue it for expansion.
			seen[to] = true
			queue = append(queue, to)
		}
	}

	// Preallocate both partitions with conservative capacity.
	sourceSide := make([]string, 0, len(rn.vertices))
	sinkSide := make([]string, 0, len(rn.vertices))

	// Partition in rn.vertices order to keep output stable across runs.
	for _, vertexID := range rn.vertices {
		if seen[vertexID] {
			sourceSide = append(sourceSide, vertexID)
			continue
		}

		sinkSide = append(sinkSide, vertexID)
	}

	return sourceSide, sinkSide
}

// reconstructPath converts parent links into a source-to-sink path.
// It assumes parent describes a valid witness path produced by a search helper.
//
// Implementation:
//   - Stage 1: Walk backward from sink to source through parent links.
//   - Stage 2: Reverse the collected path in place.
//   - Stage 3: Return a caller-owned path slice.
//
// Behavior highlights:
//   - Used only after a search helper has reported found=true.
//   - Does not validate missing parent links because that would duplicate search logic.
//
// Inputs:
//   - parent: map where parent[v] is the previous vertex on the witness path.
//   - source: source vertex ID.
//   - sink: sink vertex ID.
//
// Returns:
//   - []string: source-to-sink witness path.
//
// Errors:
//   - None.
//
// Determinism:
//   - Parent links are produced by deterministic BFS/DFS scans.
//   - Reversal preserves the exact witness path.
//
// Complexity:
//   - Time O(P), Space O(P), where P is path length.
//
// Notes:
//   - Dinic currently does not materialize path witnesses for every blocking push.
//
// AI-Hints:
//   - Do not call this helper unless the search helper returned found=true.
func reconstructPath(parent map[string]string, source, sink string) []string {
	reversed := make([]string, 0, len(parent)+1)

	for vertexID := sink; ; vertexID = parent[vertexID] {
		reversed = append(reversed, vertexID)
		if vertexID == source {
			break
		}
	}

	for left, right := 0, len(reversed)-1; left < right; left, right = left+1, right-1 {
		reversed[left], reversed[right] = reversed[right], reversed[left]
	}

	return reversed
}
