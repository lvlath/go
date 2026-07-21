// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package dijkstra

import (
	"container/heap"
	"fmt"
	"math"

	"github.com/lvlath/go/core"
)

// runDijkstra executes the canonical Dijkstra kernel for one source vertex
// over a weighted graph and returns a detached shortest-path result.
// The kernel assumes that the public API surface has already assembled the
// finalized runtime policy and delegated the computation here.
//
// Implementation:
//   - Stage 1: Validate kernel-entry inputs and numeric edge policy.
//   - Stage 2: Allocate traversal state exactly once.
//   - Stage 3: Initialize distances, predecessor storage, and heap state.
//   - Stage 4: Run the visited-finalization loop.
//   - Stage 5: Publish the detached result object.
//
// Behavior highlights:
//   - The kernel uses a lazy decrease-key heap strategy.
//   - Predecessor storage is allocated only when path tracking is enabled.
//   - Edge weights must be finite and non-negative.
//   - Positive infinity is reserved for unreachable result distances and unbounded
//     option values; it is not accepted as graph edge data.
//
// Inputs:
//   - g: the weighted graph to traverse.
//   - sourceID: the source vertex identifier for the shortest-path run.
//   - config: the finalized runtime policy for this execution.
//
// Returns:
//   - *Result: the detached shortest-path result for the requested source.
//
// Errors:
//   - Any error returned by validateInputs.
//   - ErrInvalidWeight or ErrNegativeWeight, wrapped with edge context, if defensive edge validation fails.
//   - ErrDistanceOverflow, wrapped with edge and arithmetic context,
//     if a required candidate sum cannot be represented as finite float64.
//   - Any graph-surface error returned by g.Neighbors.
//   - On every error, the function returns nil result.
//
// Determinism:
//   - Deterministic for the same graph state, sourceID, and runtime policy.
//   - Finalization order is governed by graph neighbor order, heap tie-break by vertex ID,
//     and strict-improvement-only updates.
//
// Complexity:
//   - Effective time O(V log V + E log V + E_surface), where graph-surface enumeration costs
//     depend on the core API and include sorted materialization where applicable.
//   - Effective space O(V + E_heap), excluding graph-owned storage.
//
// Notes:
//   - This kernel intentionally uses visited-finalization and does not rely on stale-distance comparison.
//   - Concurrent graph mutation during execution is unsupported by the package contract.
//
// AI-Hints:
//   - Do not reintroduce source lookup through options.
//   - Do not move endpoint resolution logic out of the canonical helper path or simplify it to edge.To.
//   - Do not remove finite MaxDistance subtraction guard before candidate addition.
//   - Do not convert ErrDistanceOverflow into +Inf unreachable publication.
func runDijkstra(g *core.Graph, sourceID string, config Options) (*Result, error) {
	if err := validateInputs(g, sourceID); err != nil {
		return nil, err
	}
	if err := validateEdgeWeights(g); err != nil {
		return nil, err
	}

	vertexCount := g.VertexCount()

	distances := make(map[string]float64, vertexCount)
	visited := make(map[string]bool, vertexCount)

	var previous map[string]string
	if config.TrackPaths {
		previous = make(map[string]string, vertexCount)
	}

	frontierCapacity := vertexCount
	if frontierCapacity < 1 {
		frontierCapacity = 1
	}

	runnerState := &runner{
		graph:     g,
		sourceID:  sourceID,
		options:   config,
		distances: distances,
		previous:  previous,
		visited:   visited,
		frontier:  make(nodePQ, 0, frontierCapacity),
	}

	runnerState.init()

	if err := runnerState.process(); err != nil {
		return nil, err
	}

	return &Result{
		SourceID:  sourceID,
		Distances: runnerState.distances,
		Prev:      runnerState.previous,
	}, nil
}

// runner stores the mutable traversal state for one Dijkstra execution.
// The structure is private to the kernel and is discarded after result publication.
//
// Implementation:
//   - Stage 1: Hold graph and policy references.
//   - Stage 2: Hold mutable shortest-path state maps.
//   - Stage 3: Hold the frontier heap used by the lazy decrease-key loop.
//
// Behavior highlights:
//   - The graph is treated as read-only.
//   - Distances and predecessor data are owned by the runner until publication.
//
// Inputs:
//   - graph: the traversed graph.
//   - sourceID: the source vertex identifier.
//   - options: the finalized runtime policy.
//
// Returns:
//   - runner: internal-only traversal state.
//
// Errors:
//   - None directly; methods on runner may return errors.
//
// Determinism:
//   - The stored state is consumed through deterministic methods and ordering rules.
//
// Complexity:
//   - Storage cost is O(V) for maps plus O(H) for the heap, where H is the number of queued heap items.
//
// Notes:
//   - The struct is not concurrency-safe and is intentionally confined to a single execution.
//
// AI-Hints:
//   - Keep this as the single mutable kernel state carrier.
//   - Do not duplicate distance/predecessor state in parallel structs.
type runner struct {
	graph     *core.Graph
	sourceID  string
	options   Options
	distances map[string]float64
	previous  map[string]string
	visited   map[string]bool
	frontier  nodePQ
}

// init initializes the full traversal state for a single Dijkstra execution.
// All vertices receive an initial distance of +Inf, the source receives 0,
// and the source is pushed into the heap as the first frontier item.
//
// Implementation:
//   - Stage 1: Enumerate the deterministic vertex domain.
//   - Stage 2: Initialize distances and visited flags.
//   - Stage 3: Initialize predecessor storage when enabled.
//   - Stage 4: Initialize the frontier heap with the source vertex.
//
// Behavior highlights:
//   - All known vertices are present in the distance map before processing begins.
//   - Predecessors are initialized only when tracking is enabled.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - None.
//
// Determinism:
//   - Initialization order follows core.Vertices() lexicographic order.
//
// Complexity:
//   - Time O(V log V) in effective graph-surface cost because core.Vertices() returns a sorted slice.
//   - Additional initialization work after enumeration is O(V).
//   - Space O(V).
//
// Notes:
//   - +Inf is the canonical unreachable value for all known vertices before relaxation begins.
//
// AI-Hints:
//   - Keep the initial source distance at exactly 0.
//   - Do not leave vertices absent from distances; missing keys mean unknown target, not unreachable vertex.
func (r *runner) init() {
	vertexIDs := r.graph.Vertices()

	for _, vertexID := range vertexIDs {
		r.distances[vertexID] = math.Inf(1)
		r.visited[vertexID] = false

		if r.previous != nil {
			r.previous[vertexID] = ""
		}
	}

	r.distances[r.sourceID] = 0

	heap.Init(&r.frontier)
	heap.Push(&r.frontier, &nodeItem{
		id:   r.sourceID,
		dist: 0,
	})
}

// process runs the main visited-finalization loop of Dijkstra's algorithm.
// The loop repeatedly extracts the smallest frontier item, finalizes it once,
// and relaxes the corresponding outgoing relation surface.
//
// Implementation:
//   - Stage 1: Pop the minimum-distance frontier item.
//   - Stage 2: Skip stale duplicate heap entries through the visited map.
//   - Stage 3: Apply MaxDistance cutoff.
//   - Stage 4: Finalize the vertex.
//   - Stage 5: Relax its neighbor relation.
//
// Behavior highlights:
//   - The loop stops early when the current minimum exceeds MaxDistance.
//   - The implementation intentionally keeps the visited-finalization model.
//
// Inputs:
//   - None.
//
// Returns:
//   - error: nil on success, or a wrapped runtime/graph-surface error.
//
// Errors:
//   - Any error returned by relax.
//
// Determinism:
//   - Frontier order is deterministic under the heap tie-break and graph neighbor order.
//
// Complexity:
//   - Effective kernel loop cost is O((V + E_push) log V), excluding graph-surface enumeration costs inside relax.
//
// Notes:
//   - A popped item with distance greater than MaxDistance is not finalized and terminates the loop.
//
// AI-Hints:
//   - Do not replace the visited-finalization model unless you can prove full contract equivalence.
//   - Keep the cutoff comparison on the popped minimum item, not on arbitrary neighbors.
func (r *runner) process() error {
	for r.frontier.Len() > 0 {
		item := heap.Pop(&r.frontier).(*nodeItem)

		currentID := item.id
		currentDistance := item.dist

		if r.visited[currentID] {
			continue
		}
		if currentDistance > r.options.MaxDistance {
			break
		}

		r.visited[currentID] = true

		if err := r.relax(currentID); err != nil {
			return err
		}
	}

	return nil
}

// relax scans the neighbor relation of the current vertex and performs
// strict-improvement relaxation for all traversable outgoing candidates.
// The method applies endpoint resolution, runtime numeric guards, wall policy,
// cutoff policy, and deterministic predecessor updates.
//
// Implementation:
//   - Stage 1: Enumerate the graph neighbor relation of the current vertex.
//   - Stage 2: Resolve the effective neighbor endpoint for each edge.
//   - Stage 3: Apply runtime numeric validation and wall policy.
//   - Stage 4: Compute candidate distances and apply strict-improvement relaxation.
//   - Stage 5: Record predecessor state when tracking is enabled and push the new heap item.
//
// Behavior highlights:
//   - Undirected and mixed edges are handled through canonical endpoint resolution.
//   - Equal candidate distances do not overwrite predecessor state.
//   - Already finalized neighbors are skipped eagerly.
//
// Inputs:
//   - currentID: the currently finalized vertex whose neighbor relation will be relaxed.
//
// Returns:
//   - error: nil on success, or a wrapped runtime/graph-surface error.
//
// Errors:
//   - Wrapped graph-surface errors from g.Neighbors.
//   - Wrapped ErrInvalidWeight if runtime observation finds NaN or either infinity.
//   - Wrapped ErrNegativeWeight if runtime observation finds a finite negative weight.
//   - Wrapped ErrDistanceOverflow if currentDistance + weight cannot be represented
//     as a finite float64 under the active MaxDistance policy.
//
// Determinism:
//   - Neighbor scanning order follows core.Neighbors(currentID).
//   - Heap insertion order is stabilized by nodePQ.Less through (dist, id).
//
// Complexity:
//   - Effective time is O(deg(currentID) * push_cost) plus graph-surface neighbor enumeration cost.
//   - Extra space is O(1) beyond heap growth for successful relaxations.
//
// Notes:
//   - InfEdgeThreshold applies to finite edge weights.
//   - A threshold of +Inf disables finite-weight wall filtering.
//   - This method assumes currentID has already been finalized.
//
// AI-Hints:
//   - Never replace endpoint resolution with edge.To for all cases; undirected edges require relative endpoint logic.
//   - Keep strict-improvement-only updates to preserve deterministic predecessor selection.
func (r *runner) relax(currentID string) error {
	neighborEdges, err := r.graph.Neighbors(currentID)
	if err != nil {
		return fmt.Errorf("dijkstra: failed to get neighbors of %q: %w", currentID, err)
	}

	for _, edge := range neighborEdges {
		neighborID, ok := otherEndpoint(edge, currentID)
		if !ok {
			continue
		}
		if r.visited[neighborID] {
			continue
		}

		weight := edge.Weight
		if err = classifyWeight(weight); err != nil {
			return fmt.Errorf(
				"%w: edge_id=%q from=%q to=%q directed=%t weight=%g",
				err,
				edge.ID,
				edge.From,
				edge.To,
				edge.Directed,
				weight,
			)
		}
		if weight >= r.options.InfEdgeThreshold {
			continue
		}

		currentDistance := r.distances[currentID]

		// Apply a finite MaxDistance cutoff before addition. This prevents an
		// out-of-policy candidate from overflowing even though the candidate
		// would be discarded by the traversal policy anyway.
		if !math.IsInf(r.options.MaxDistance, 1) &&
			weight > r.options.MaxDistance-currentDistance {
			continue
		}

		candidateDistance := currentDistance + weight
		if math.IsInf(candidateDistance, 1) {
			return fmt.Errorf(
				"%w: edge_id=%q from=%q to=%q current_distance=%g weight=%g",
				ErrDistanceOverflow,
				edge.ID,
				edge.From,
				edge.To,
				currentDistance,
				weight,
			)
		}

		if candidateDistance > r.options.MaxDistance {
			continue
		}
		if candidateDistance >= r.distances[neighborID] {
			continue
		}

		r.distances[neighborID] = candidateDistance

		if r.previous != nil {
			r.previous[neighborID] = currentID
		}

		heap.Push(&r.frontier, &nodeItem{
			id:   neighborID,
			dist: candidateDistance,
		})
	}

	return nil
}

// nodeItem stores one frontier candidate for the priority queue.
// The item contains the target vertex identifier and the candidate distance
// currently associated with that heap entry.
//
// Implementation:
//   - Stage 1: Store the candidate vertex identifier.
//   - Stage 2: Store the candidate distance.
//
// Behavior highlights:
//   - Multiple items for the same vertex may coexist under lazy decrease-key.
//
// Inputs:
//   - id: the candidate vertex identifier.
//   - dist: the candidate source distance associated with this heap item.
//
// Returns:
//   - nodeItem: an internal heap payload type.
//
// Errors:
//   - None.
//
// Determinism:
//   - The item itself is plain data; ordering is defined by nodePQ.Less.
//
// Complexity:
//   - Storage cost O(1).
//
// Notes:
//   - Stale items are discarded by the visited-finalization logic in process.
//
// AI-Hints:
//   - Keep this type minimal; extra fields tend to create duplicate sources of truth.
type nodeItem struct {
	id   string
	dist float64
}

// nodePQ is the canonical min-heap used by the Dijkstra kernel.
// Items are ordered first by candidate distance and then by vertex identifier
// to guarantee deterministic tie-breaking under equal distances.
//
// Implementation:
//   - Stage 1: Maintain container/heap compatibility.
//   - Stage 2: Compare items by (dist, id).
//   - Stage 3: Support lazy decrease-key through duplicate entries.
//
// Behavior highlights:
//   - Smaller distance has higher priority.
//   - Equal distances are broken by lexicographically smaller vertex IDs.
//
// Inputs:
//   - None.
//
// Returns:
//   - nodePQ: an internal priority-queue type.
//
// Errors:
//   - None.
//
// Determinism:
//   - Deterministic for the same inserted heap contents because Less uses (dist, id).
//
// Complexity:
//   - Heap primitives remain O(log N) under container/heap semantics.
//
// Notes:
//   - The type is internal and should not leak into the public surface.
//
// AI-Hints:
//   - Do not remove the secondary ID tie-break; that would weaken determinism and predecessor stability.
type nodePQ []*nodeItem

// Len returns the number of items currently stored in the priority queue.
//
// Implementation:
//   - Stage 1: Return the slice length.
//
// Behavior highlights:
//   - Pure query with no side effects.
//
// Inputs:
//   - None.
//
// Returns:
//   - int: the current heap size.
//
// Errors:
//   - None.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
func (pq nodePQ) Len() int {
	return len(pq)
}

// Less reports whether item i must sort before item j in the min-heap ordering.
// The comparison first uses candidate distance and then uses vertex ID as the
// deterministic tie-break for equal distances.
//
// Implementation:
//   - Stage 1: Compare distances.
//   - Stage 2: Break equal-distance ties by vertex ID.
//
// Behavior highlights:
//   - Smaller distance wins.
//   - Equal distances prefer smaller vertex IDs.
//
// Inputs:
//   - i: the first heap index.
//   - j: the second heap index.
//
// Returns:
//   - bool: true when item i has higher priority than item j.
//
// Errors:
//   - None.
//
// Determinism:
//   - Fully deterministic for equal-distance items because the secondary key is explicit.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This tie-break is part of the package determinism contract, not an implementation accident.
//
// AI-Hints:
//   - Do not simplify this to distance-only comparison.
//   - Equal-distance tie-breaking affects predecessor stability and must remain explicit.
func (pq nodePQ) Less(i, j int) bool {
	if pq[i].dist < pq[j].dist {
		return true
	}
	if pq[i].dist > pq[j].dist {
		return false
	}

	return pq[i].id < pq[j].id
}

// Swap exchanges two heap items in place.
//
// Implementation:
//   - Stage 1: Swap the two slice entries.
//
// Behavior highlights:
//   - container/heap uses this method to restore heap invariants.
//
// Inputs:
//   - i: the first heap index.
//   - j: the second heap index.
//
// Returns:
//   - None.
//
// Errors:
//   - None.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
func (pq nodePQ) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
}

// Push appends one heap item to the underlying slice.
// The method is part of the container/heap interface and expects x to hold
// a *nodeItem value.
//
// Implementation:
//   - Stage 1: Type-assert x to *nodeItem.
//   - Stage 2: Append the item to the slice.
//
// Behavior highlights:
//   - The method does not perform ordering logic itself; heap invariants are managed by container/heap.
//
// Inputs:
//   - x: the heap item to append.
//
// Returns:
//   - None.
//
// Errors:
//   - A bad internal type would panic, which is acceptable here because this method is package-internal and controlled.
//
// Determinism:
//   - Deterministic for the same appended item sequence.
//
// Complexity:
//   - Time O(1) amortized, Space O(1) amortized.
func (pq *nodePQ) Push(x interface{}) {
	*pq = append(*pq, x.(*nodeItem))
}

// Pop removes and returns the last slice item as required by container/heap.
// The heap package itself ensures that the last element is the current minimum
// after it has restored the heap invariant.
//
// Implementation:
//   - Stage 1: Read the tail item.
//   - Stage 2: Shrink the slice by one.
//   - Stage 3: Return the removed item.
//
// Behavior highlights:
//   - The method operates on the internal heap storage only.
//
// Inputs:
//   - None.
//
// Returns:
//   - interface{}: the removed *nodeItem value.
//
// Errors:
//   - None directly; container/heap governs call correctness.
//
// Determinism:
//   - Deterministic for the same heap state.
//
// Complexity:
//   - Time O(1), Space O(1).
func (pq *nodePQ) Pop() interface{} {
	oldItems := *pq
	lastIndex := len(oldItems) - 1
	item := oldItems[lastIndex]
	*pq = oldItems[:lastIndex]

	return item
}

////////////////////////////////////////////////////////////////////
//
// helpers

// otherEndpoint resolves the effective neighbor vertex for an edge relative
// to the currently processed vertex identifier.
// The helper is the canonical endpoint-resolution law for directed, undirected,
// and mixed edge traversal inside this package.
//
// Implementation:
//   - Stage 1: Reject a nil edge.
//   - Stage 2: Apply directed edge semantics.
//   - Stage 3: Apply undirected edge semantics relative to currentID.
//   - Stage 4: Reject non-incident edges.
//
// Behavior highlights:
//   - Directed edges are traversable only from From to To.
//   - Undirected edges resolve to the opposite endpoint relative to currentID.
//   - Edges that do not contain currentID are rejected.
//
// Inputs:
//   - e: the edge whose endpoint relation is being resolved.
//   - currentID: the vertex currently being expanded by the traversal.
//
// Returns:
//   - string: the effective neighbor vertex identifier.
//   - bool: true when the edge yields a valid traversable neighbor from currentID.
//
// Errors:
//   - None.
//
// Determinism:
//   - Deterministic for the same edge and currentID.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This helper defines the package endpoint law and must remain the only
//     canonical implementation of that law.
//   - The helper performs relation resolution only; it does not classify weight
//     validity or traversal policy.
//
// AI-Hints:
//   - Do not replace this helper with edge.To-only logic.
//   - Do not duplicate endpoint semantics elsewhere in the package with a second implementation.
func otherEndpoint(e *core.Edge, currentID string) (string, bool) {
	if e == nil {
		return "", false
	}

	if e.Directed {
		if e.From != currentID {
			return "", false
		}

		return e.To, true
	}

	if e.From == currentID {
		return e.To, true
	}
	if e.To == currentID {
		return e.From, true
	}

	return "", false
}
