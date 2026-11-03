// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dijkstra supplies deterministic lazy heaps for target-directed search.
// This file isolates frontier ordering from the pre-existing SSSP heap contract.
package dijkstra

import (
	"container/heap"
	"math"
)

// noSearchArc denotes a root vertex with no predecessor or an unset link.
// Valid oriented arc indices are non-negative and refer to searchGraph.arcs.
const noSearchArc = -1

// searchItem records one enqueued candidate and its deterministic priority.
// A stale item is discarded when its vertex has settled or its distance changed.
// Implementation:
//   - Stage 1: Carry a dense vertex index and an original-cost candidate.
//   - Stage 2: Carry the method's priority, which equals distance for Dijkstra.
//   - Stage 3: Carry a per-frontier insertion sequence for final ties.
//
// Behavior highlights:
//   - All admitted heap distances and priorities are finite and non-NaN.
//
// Inputs:
//   - vertex: dense vertex index; distance: original cost; priority: ordering key.
//   - sequence: insertion ordinal in its owning frontier.
//
// Returns:
//   - searchItem: internal immutable queue value.
//
// Errors:
//   - None; validation occurs before insertion.
//
// Determinism:
//   - Heap ordering is (priority, distance, vertex, sequence).
//
// Complexity:
//   - Time O(1), Space O(1) for a value.
//
// Notes:
//   - This layout deliberately differs from legacy nodeItem's vertex-ID comparator.
//
// AI-Hints:
//   - Do not use a tolerance-based heap comparator; it can violate transitivity.
type searchItem struct {
	vertex   int     // Dense vertex index, ordered lexically through searchGraph.ids.
	distance float64 // Finite original-cost label when enqueued.
	priority float64 // Finite method-specific heap key; original cost in Dijkstra.
	sequence uint64  // Monotone enqueue ordinal for otherwise identical tuples.
}

// searchQueue implements container/heap with a strict, stable four-part order.
// Lazy duplicates are permitted, so heap space may reach O(A) entries.
// Implementation:
//   - Stage 1: Maintain searchItem values without pointers to graph objects.
//   - Stage 2: Use exact key comparisons and deterministic final ties.
//   - Stage 3: Remove stale values in the caller, not inside the heap interface.
//
// Behavior highlights:
//   - The heap's public interface remains private to target-search kernels.
//
// Inputs:
//   - []searchItem: internal value slice of the active search frontier.
//
// Returns:
//   - searchQueue: min-priority heap storage.
//
// Errors:
//   - None; only correctly typed values are pushed internally.
//
// Determinism:
//   - Fixed tuple order independent of Go's map iteration order.
//
// Complexity:
//   - Heap.Push and Heap.Pop are O(log H), Space O(H) with H <= O(A).
//
// Notes:
//   - Zero-value searchQueue is a valid empty heap.
//
// AI-Hints:
//   - Do not reuse nodePQ: its ordering contract is distinct.
type searchQueue []searchItem

// Len returns the number of queued candidates, including stale entries.
// A candidate's presence alone does not establish a finalized shortest-path label.
// Implementation:
//   - Stage 1: Return the length of the internal heap slice.
//
// Behavior highlights:
//   - Compatible with container/heap.Interface.
//
// Inputs:
//   - None; receiver is the internal queue slice.
//
// Returns:
//   - int: current number of heap entries.
//
// Errors:
//   - None.
//
// Determinism:
//   - Deterministic for the current queue state.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - No traversal is performed by Len.
//
// AI-Hints:
//   - Heap length is not a count of unsettled vertices.
func (pq searchQueue) Len() int { return len(pq) }

// Less orders two heap entries by priority and stable, strict tie-breaks.
// It uses original distance and lexical dense index after the numerical key.
// Implementation:
//   - Stage 1: Compare exact finite priorities.
//   - Stage 2: Compare original costs, then vertex indices.
//   - Stage 3: Break repeated-vertex ties by insertion sequence.
//
// Behavior highlights:
//   - No epsilon, rounding tolerance or implicit ID conversions.
//
// Inputs:
//   - i, j: valid positions in the queue slice.
//
// Returns:
//   - bool: true when entry i precedes entry j.
//
// Errors:
//   - None; heap owns valid indices.
//
// Determinism:
//   - Total tuple order for admitted finite, non-NaN entries.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Dense vertex indices inherit core.Vertices lexical order.
//
// AI-Hints:
//   - Do not substitute legacy heap's ID tie-break or a fuzzy comparator.
func (pq searchQueue) Less(i, j int) bool {
	left, right := pq[i], pq[j]
	if left.priority != right.priority {
		return left.priority < right.priority
	}
	if left.distance != right.distance {
		return left.distance < right.distance
	}
	if left.vertex != right.vertex {
		return left.vertex < right.vertex
	}
	return left.sequence < right.sequence
}

// Swap exchanges two queue entries without altering their contents.
// The container/heap implementation calls it while restoring heap order.
// Implementation:
//   - Stage 1: Swap the entries at valid positions i and j.
//
// Behavior highlights:
//   - No allocations or graph accesses.
//
// Inputs:
//   - i, j: valid heap positions.
//
// Returns:
//   - None.
//
// Errors:
//   - None.
//
// Determinism:
//   - Fixed exchange for the same state and positions.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Heap ordering is determined exclusively by Less.
//
// AI-Hints:
//   - Do not revise sequence values during swaps.
func (pq searchQueue) Swap(i, j int) { pq[i], pq[j] = pq[j], pq[i] }

// Push appends a new queue value for container/heap to reorder.
// Kernel code provides a searchItem only after numeric admission succeeds.
// Implementation:
//   - Stage 1: Assert the internal searchItem type.
//   - Stage 2: Append the value to the queue slice.
//   - Stage 3: Leave heap restoration to container/heap.
//
// Behavior highlights:
//   - It does not replace older entries when a vertex improves.
//
// Inputs:
//   - value: internal searchItem supplied by the kernel.
//
// Returns:
//   - None.
//
// Errors:
//   - None for correctly typed internal use.
//
// Determinism:
//   - Appends in the exact order requested by the kernel.
//
// Complexity:
//   - Amortized O(1) append work, excluding heap restoration.
//
// Notes:
//   - A wrong value type is a private programming error, not a public input.
//
// AI-Hints:
//   - Do not implement decrease-key here without changing stale-item handling.
func (pq *searchQueue) Push(value any) { *pq = append(*pq, value.(searchItem)) }

// Pop removes the tail value prepared by container/heap as the minimum.
// The emptied slot is cleared to keep the backing array free of obsolete state.
// Implementation:
//   - Stage 1: Save the last entry as the result.
//   - Stage 2: Clear the vacated slot and shorten the heap slice.
//   - Stage 3: Return the saved searchItem.
//
// Behavior highlights:
//   - Only the heap package decides which entry becomes the removed tail.
//
// Inputs:
//   - None; receiver points to the heap-managed queue.
//
// Returns:
//   - any: the popped searchItem.
//
// Errors:
//   - None for valid heap calls.
//
// Determinism:
//   - Preserves the strict Less tuple order.
//
// Complexity:
//   - Time O(1), Space O(1) excluding heap restoration.
//
// Notes:
//   - This value queue has no graph pointers; clearing also aids future layout changes.
//
// AI-Hints:
//   - Never perform a naked slice-head drop that retains a large backing array.
func (pq *searchQueue) Pop() any {
	items := *pq
	last := len(items) - 1
	item := items[last]
	items[last] = searchItem{}
	*pq = items[:last]
	return item
}

// searchFrontier stores mutable labels and parent arcs for one search direction.
// Only settled labels are final; the target facade never publishes other distances.
// Implementation:
//   - Stage 1: Own distance, linkArc and settled vectors over dense vertex indices.
//   - Stage 2: Own a lazy searchQueue and its insertion sequence.
//   - Stage 3: Discard mutable workspace after publication of PathResult.
//
// Behavior highlights:
//   - +Inf marks a missing tentative label, not an arithmetic overflow.
//
// Inputs:
//   - root: dense source vertex index; all arrays share searchGraph.ids length.
//
// Returns:
//   - searchFrontier: private state for one kernel invocation.
//
// Errors:
//   - None directly; allocation is managed by Go.
//
// Determinism:
//   - Labels follow strict improvements; heap follows its defined tuple ordering.
//
// Complexity:
//   - Space O(V+H), H<=O(A) due to lazy heap duplicates.
//
// Notes:
//   - This state is not safe for concurrent mutation by multiple goroutines.
//
// AI-Hints:
//   - Never treat an un-settled finite label as an SSSP result.
type searchFrontier struct {
	root     int         // Immutable dense source index for this frontier.
	distance []float64   // Best tentative original costs, or +Inf for absent labels.
	linkArc  []int       // Original oriented predecessor arc, or noSearchArc.
	settled  []bool      // True only after a non-stale minimum was removed.
	queue    searchQueue // Lazy duplicate min-heap over admitted candidate labels.
	sequence uint64      // Next local enqueue ordinal; not a public work counter.
}

// newSearchFrontier initializes all labels and the one certified root witness.
// The root distance is zero and has no incoming predecessor arc.
// Implementation:
//   - Stage 1: Allocate dense labels, predecessor links and settled bits.
//   - Stage 2: Set every unknown label to +Inf and parent to noSearchArc.
//   - Stage 3: Insert the zero-cost root candidate into the min-heap.
//
// Behavior highlights:
//   - No other vertex is finalized or treated as reachable at initialization.
//
// Inputs:
//   - vertexCount: positive size of the effective vertex domain.
//   - root: valid dense index for the source vertex.
//
// Returns:
//   - *searchFrontier: newly allocated, invocation-owned mutable workspace.
//
// Errors:
//   - None for validated counts and indices.
//
// Determinism:
//   - Root entry uses the first insertion sequence; all other distances are +Inf.
//
// Complexity:
//   - Time O(V), Space O(V), excluding later heap growth.
//
// Notes:
//   - The initial root candidate has exactly representable priority zero.
//
// AI-Hints:
//   - Do not initialize all labels to zero; that destroys reachability semantics.
func newSearchFrontier(vertexCount, root int) *searchFrontier {
	frontier := &searchFrontier{
		root:     root,
		distance: make([]float64, vertexCount),
		linkArc:  make([]int, vertexCount),
		settled:  make([]bool, vertexCount),
		queue:    make(searchQueue, 0, vertexCount),
	}
	for vertex := range frontier.distance {
		frontier.distance[vertex] = math.Inf(1)
		frontier.linkArc[vertex] = noSearchArc
	}
	frontier.distance[root] = 0
	heap.Push(&frontier.queue, searchItem{vertex: root, distance: 0, priority: 0, sequence: frontier.sequence})
	frontier.sequence++
	return frontier
}
