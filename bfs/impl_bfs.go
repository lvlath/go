// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package bfs

import (
	"fmt"

	"github.com/lvlath/go/core"
)

const (
	// rootDepth is the canonical depth value for a BFS root vertex.
	//
	// AI-HINTS:
	//   - Depth is measured as edge count (unweighted).
	//   - The root is always at depth 0, even under FullTraversal (forest mode).
	rootDepth = 0
)

// queueItem is a single work unit in the BFS frontier queue.
//
// Notes:
//   - parent is NOT stored here because the parent is always the current vertex
//     when enqueuing its neighbors; parent links are written at enqueue time.
type queueItem struct {
	// id is the vertex identifier to process.
	id string

	// depth is the shortest distance in edges from the current component root.
	depth int
}

// walker owns the mutable state of a single BFS execution.
//
// The walker is internal and not concurrency-safe by itself; the graph provides
// concurrency safety for its own storage. Callbacks are treated as observers.
type walker struct {
	// g is the graph being traversed (read-only by convention).
	g *core.Graph

	// o is the already-finalized effective options value.
	o Options

	// res is the result object that accumulates output incrementally.
	res *Result

	// q is the FIFO frontier queue. Dequeue uses head-index to avoid slice-shift retention.
	q []queueItem

	// head is the dequeue index into q.
	head int
}

// runBFS executes the BFS kernel using already-finalized options.
//
// Implementation:
//   - Stage 1: (performed by the public facade) validate inputs and apply options.
//   - Stage 2: Allocate working sets once using O(V) capacity hints.
//   - Stage 3: Run FIFO frontier expansion with fixed neighbor iteration order.
//   - Stage 4: Optionally run deterministic forest continuation (FullTraversal).
//
// Behavior highlights:
//   - Determinism: traversal order is deterministic if NeighborIDs is deterministic.
//   - Visit definition: "visit" happens at dequeue; Order is dequeue/visit order.
//   - Partial result: on early exit (ctx cancel, neighbor fetch error, hook error),
//     the current *Result is returned alongside the error.
//
// Inputs:
//   - g: graph instance (assumed non-nil and validated by the facade).
//   - startID: existing vertex ID (assumed validated by the facade).
//   - o: finalized options (ctx and callbacks are assumed non-nil).
//
// Returns:
//   - *Result: accumulated result (may be partial if an error is returned).
//   - error: nil on success; otherwise ctx.Err(), ErrNeighbors-wrapped error, or hook error.
//
// Errors:
//   - context.Canceled / context.DeadlineExceeded when o.ctx is done.
//   - ErrNeighbors when NeighborIDs fails (double-wrapped with the underlying cause).
//   - Any error returned by OnVisit (wrapped with contextual text and %w).
//
// Determinism:
//   - FIFO queue order is fixed.
//   - Neighbor processing follows the order returned by g.NeighborIDs(u).
//
// Complexity:
//   - Time O(|V|+|E|), Space O(|V|) for a single-source traversal.
//   - WithFullTraversal adds the cost of iterating all vertices to find roots.
//
// Notes:
//   - Callbacks must not mutate the graph; treat them as observers.
//
// AI-Hints:
//   - Mark visited on enqueue to guarantee each vertex is enqueued once.
//   - Use head-index queue + clear slots to avoid memory retention on large traversals.
//   - Keep hooks allocation-free; they run in hot paths.
func runBFS(g *core.Graph, startID string, o Options) (*Result, error) {
	// Stage 2: Allocate Once.
	//
	// AI-HINT: VertexCount() is O(1) and avoids sorting costs of Vertices().
	n := g.VertexCount()

	w := &walker{
		g: g,
		o: o,
		res: &Result{
			StartID: startID,
			Order:   make([]string, 0, n),
			Depth:   make(map[string]int, n),
			Parent:  make(map[string]string, n),
			Visited: make(map[string]bool, n),
		},
		q:    make([]queueItem, 0, n),
		head: 0,
	}

	// Stage 3: Seed the queue with the start root (no parent).
	w.enqueueRoot(startID)

	// Stage 3: Core traversal of the start component.
	if err := w.loop(); err != nil {
		return w.res, err
	}

	// Stage 4: Optional forest continuation.
	if w.o.fullTraversal {
		if err := w.fullTraversal(); err != nil {
			return w.res, err
		}
	}

	return w.res, nil
}

// enqueueRoot enqueues a component root at depth 0 without writing a Parent entry.
//
// AI-HINTS:
//   - Roots have no parent link; Parent[root] must be absent.
//   - Visited is set at enqueue time to avoid duplicate queue entries.
func (w *walker) enqueueRoot(id string) {
	w.res.Visited[id] = true
	w.res.Depth[id] = rootDepth

	w.o.onEnqueue(id, rootDepth)

	w.q = append(w.q, queueItem{id: id, depth: rootDepth})
}

// enqueueChild enqueues a newly discovered vertex and writes its Parent/Depth.
//
// AI-HINTS:
//   - Parent is recorded at discovery time (enqueue) to ensure shortest-path correctness.
//   - Visited is set at enqueue time to guarantee at-most-once enqueue semantics.
func (w *walker) enqueueChild(id string, depth int, parent string) {
	w.res.Visited[id] = true
	w.res.Depth[id] = depth
	w.res.Parent[id] = parent

	w.o.onEnqueue(id, depth)

	w.q = append(w.q, queueItem{id: id, depth: depth})
}

// dequeue returns the next item from the frontier queue using head-index semantics.
//
// Behavior highlights:
//   - Clears the slot to avoid retaining references in the underlying array.
//   - Does not shrink the slice; head-index controls consumption.
//
// AI-HINTS:
//   - Avoid queue = queue[1:] in Go; it can retain the entire backing array.
func (w *walker) dequeue() queueItem {
	item := w.q[w.head]
	w.q[w.head] = queueItem{} // release references (retention-safe)
	w.head++
	return item
}

// loop processes the current queue until exhausted or until an early-exit condition occurs.
//
// Implementation:
//   - Stage 1: Check ctx cancellation at dequeue granularity.
//   - Stage 2: Dequeue and run hooks (OnDequeue, OnVisit).
//   - Stage 3: Expand neighbors (subject to MaxDepth and FilterNeighbor).
//
// Behavior highlights:
//   - Order is dequeue/visit order.
//   - Expansion respects MaxDepth inclusive rule.
//
// AI-HINTS:
//   - Do not allocate inside neighbor loops; all working sets are preallocated.
func (w *walker) loop() error {
	for w.head < len(w.q) {
		// Early exit: cancellation check once per dequeued vertex.
		select {
		case <-w.o.ctx.Done():
			return w.o.ctx.Err()
		default:
		}

		item := w.dequeue()

		// Hook: dequeue observation happens before visit.
		w.o.onDequeue(item.id, item.depth)

		// "visit" is defined as dequeue time.
		w.res.Order = append(w.res.Order, item.id)

		// Hook: visit may stop traversal by returning an error.
		if err := w.o.onVisit(item.id); err != nil {
			return fmt.Errorf("bfs: OnVisit error at %q: %w", item.id, err)
		}

		// MaxDepth inclusive: visit at depth == maxDepth, but do not expand.
		if w.o.maxDepth != MaxDepthUnlimited && item.depth >= w.o.maxDepth {
			continue
		}

		neighbors, err := w.g.NeighborIDs(item.id)
		if err != nil {
			// Double-wrap preserves both ErrNeighbors classification and the underlying cause.
			return fmt.Errorf("%w: failed to get neighbors of %q: %w", ErrNeighbors, item.id, err)
		}

		for _, nbr := range neighbors {
			// Cancellation check inside neighbor iteration for responsive aborts.
			select {
			case <-w.o.ctx.Done():
				return w.o.ctx.Err()
			default:
			}

			// Relation-level filter: (currID -> nbrID).
			if !w.o.filterNeighbor(item.id, nbr) {
				w.res.Skipped++
				continue
			}

			// Enqueue only once.
			if w.res.Visited[nbr] {
				continue
			}

			w.enqueueChild(nbr, item.depth+1, item.id)
		}
	}

	return nil
}

// fullTraversal continues BFS as a forest over all vertices.
//
// Implementation:
//   - Stage 1: Obtain a deterministic vertex iteration order.
//   - Stage 2: For each unvisited vertex, run a fresh BFS rooted at that vertex.
//
// Behavior highlights:
//   - Produces a forest: multiple roots, each with depth 0 and no parent.
//   - Preserves determinism by selecting secondary roots in deterministic vertex order.
//
// Determinism:
//   - Secondary roots are chosen in the order returned by g.Vertices().
//
// Complexity:
//   - Adds O(V log V) due to g.Vertices() sorting, plus O(V+E) traversal work overall.
//
// AI-HINTS:
//   - FullTraversal is useful for building a forest; PathTo must still anchor to StartID.
func (w *walker) fullTraversal() error {
	// Reset the queue state so each component traversal starts from a clean frontier.
	w.q = w.q[:0]
	w.head = 0

	// core.Graph.Vertices() returns IDs sorted lexicographically for deterministic order.
	vertexIDs := w.g.Vertices()

	for _, id := range vertexIDs {
		if w.res.Visited[id] {
			continue
		}

		// Early exit check before starting a new component.
		select {
		case <-w.o.ctx.Done():
			return w.o.ctx.Err()
		default:
		}

		// Secondary root: depth 0, no parent.
		w.enqueueRoot(id)

		if err := w.loop(); err != nil {
			return err
		}

		// Prepare for the next component.
		w.q = w.q[:0]
		w.head = 0
	}

	return nil
}
