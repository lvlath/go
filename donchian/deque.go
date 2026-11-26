// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package donchian stores rolling extrema in fixed-capacity index rings.
// This file explains candidate lifetime, latest-equal eviction, and bounded cancellation polling.
package donchian

import "context"

// indexDeque retains candidate extrema in a fixed-capacity circular index buffer.
// Indices increase in logical order; their values are strictly monotonic after insertion.
//
// Implementation:
//   - Stage 1: allocate exactly Period indices only when some result can be ready.
//   - Stage 2: expire the oldest candidate and remove dominated tail candidates.
//   - Stage 3: read the head to obtain the selected latest-tie extremum.
//
// Behavior highlights:
//   - size distinguishes a full ring from an empty ring when head equals tail.
//   - Maximum and minimum queues share the structure but never share mutable storage.
//
// Inputs:
//   - indices: owning nonempty buffer, sized by the admitted window period.
//   - head, tail, size: private cursor state maintained only by expire and insert.
//
// Returns:
//   - An internal candidate set; no buffer is exposed or retained after Compute.
//
// Errors:
//   - insert can propagate ctx.Err(); a partially changed queue is discarded with the call.
//
// Determinism:
//   - New equal values remove older equal candidates, preserving the latest index.
//
// Complexity:
//   - O(P) storage per ring; O(n) total queue work over n sequential insertions.
//
// Notes:
//   - Zero-value queues are not usable: the private kernel always allocates their buffers.
//   - Private invariants replace redundant bounds checks; no public mutation surface exists.
//
// AI-Hints:
//   - Expire before insert so capacity P suffices without a P+1 sentinel slot.
//   - Do not replace the fixed ring by a growing append-only index history.
type indexDeque struct {
	// indices owns exactly Period slots; only size logical entries are active.
	indices []int
	// head identifies the oldest candidate and hence the active extremum.
	head int
	// tail is the next free insertion slot, wrapping at len(indices).
	tail int
	// size is in [0,len(indices)] and prevents full/empty ambiguity.
	size int
}

// expire removes candidates strictly to the left of the next inclusive window.
// It preserves candidate order and leaves tail unchanged when the head advances.
//
// Implementation:
//   - Stage 1: compare the oldest retained index with the new left boundary.
//   - Stage 2: advance and wrap head for each expired candidate.
//   - Stage 3: decrease size so the subsequent insertion has a free slot.
//
// Behavior highlights:
//   - An index exactly equal to first remains eligible.
//   - The kernel advances first by one, so at most one candidate expires per call.
//
// Inputs:
//   - d: initialized private ring with increasing logical indices.
//   - first: inclusive left boundary; negative values during warmup expire nothing.
//
// Returns:
//   - No value; the ring is changed in place and remains private.
//
// Errors:
//   - None; shape and ring invariants are established by the sole kernel.
//
// Determinism:
//   - Oldest candidates are removed first; no comparisons of price values occur.
//
// Complexity:
//   - O(k) time for k removed indices, O(1) space; k <= 1 in the kernel.
//
// Notes:
//   - No polling is needed in this bounded expiration path under kernel preconditions.
//
// AI-Hints:
//   - Changing the kernel to skip multiple positions requires rechecking this bound.
//   - Use a strict index comparison; expiring first itself shortens the window.
func (d *indexDeque) expire(first int) {
	// Guard the head read by size; indices at first remain inside the closed window.
	for d.size > 0 && d.indices[d.head] < first {
		// Move past the expired candidate without shifting any buffer elements.
		d.head++
		// Wrap by comparison instead of potentially overflowing head+size arithmetic.
		if d.head == len(d.indices) {
			d.head = 0
		}
		// Decrementing size preserves full/empty meaning even when cursors coincide.
		d.size--
	}
}

// insert appends one observation after removing older dominated candidates.
// The newest index wins equal-value ties, including ties between signed zeros.
//
// Implementation:
//   - Stage 1: inspect the logical tail, handling its circular predecessor.
//   - Stage 2: remove dominated candidates and poll after each bounded removal block.
//   - Stage 3: write the new index, wrap tail, and increase size.
//
// Behavior highlights:
//   - maximum=true retains decreasing values; false retains increasing values.
//   - A strictly better old tail survives; equal or worse old tails do not.
//
// Inputs:
//   - d: private nonempty-capacity ring, expired before this insertion.
//   - ctx: non-nil context; the outer kernel also polls once per input block.
//   - values: immutable finite high or low channel corresponding to this ring.
//   - index: next sequential input position; maximum: selects high versus low comparison.
//
// Returns:
//   - error: nil after insertion; ctx.Err() aborts the entire private computation.
//
// Errors:
//   - Only context cancellation or deadline expiration; no alternate numerical path.
//
// Determinism:
//   - Strict retention comparisons guarantee latest-index ties in fixed input order.
//
// Complexity:
//   - O(1) amortized time per insertion, O(P) worst-case burst, O(1) extra space.
//   - Every observation is inserted once and removed at most once in this ring.
//
// Notes:
//   - At most Period-1 active candidates precede insertion, so the fixed buffer fits.
//   - Cancellation can leave a partially shortened ring; Compute never publishes or reuses it.
//
// AI-Hints:
//   - Poll inside tail bursts: per-bar polling alone does not bound a large pop loop.
//   - Do not use strict removal of equals; that changes the observable zero-sign tie rule.
func (d *indexDeque) insert(ctx context.Context, values []float64, index int, maximum bool) error {
	// Count actual removals rather than bars to bound a single expensive insertion.
	removed := 0
	// Stop when the new observation dominates every retained candidate.
	for d.size > 0 {
		// tail is the next insertion position; its predecessor is the last active slot.
		last := d.tail - 1
		if last < 0 {
			last = len(d.indices) - 1
		}
		// Read the old candidate without copying or transforming its price bits.
		previous := values[d.indices[last]]
		// Keep a strictly better tail; monotonicity then protects every earlier tail.
		if (maximum && previous > values[index]) || (!maximum && previous < values[index]) {
			break
		}
		// Forget an equal or worse candidate; its newer replacement expires later.
		d.tail = last
		d.size--
		removed++
		// A large monotonic history can be emptied by one spike; remain cancellable.
		if removed%pollInterval == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
	}
	// Expiration before insertion guarantees space even when no tail was removed.
	d.indices[d.tail] = index
	// Advance the insertion cursor and wrap at the fixed capacity.
	d.tail++
	if d.tail == len(d.indices) {
		d.tail = 0
	}
	// The new index is now the newest candidate and may also be the sole candidate.
	d.size++
	return nil
}
