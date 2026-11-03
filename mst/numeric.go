// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mst defines checked arithmetic for spanning-network publication.
// The arithmetic helpers distinguish unrepresentable costs from disconnected topology.
package mst

import (
	"math"
	"unsafe"

	"github.com/lvlath/go/core"
)

// checkedMSTWeightAdd adds two validated, finite edge-cost operands.
// It preserves Go's ordinary left-to-right float64 rounding and rejects a non-finite result.
// Implementation:
//   - Stage 1: Add the next edge weight to the accumulated prefix.
//   - Stage 2: Classify arithmetic overflow before publishing the prefix.
//   - Stage 3: Return the finite sum without rounding adjustments.
//
// Behavior highlights:
//   - Signed finite weights are supported by ordinary MST/MSF.
//   - A signed prefix can overflow even if a differently ordered sum would be finite.
//
// Inputs:
//   - prefix, weight: finite float64 values in the same edge-cost units.
//
// Returns:
//   - float64: the next finite left-fold prefix, or zero on error.
//
// Errors:
//   - ErrWeightOverflow if the addition produces an infinite or NaN value.
//
// Determinism:
//   - Exactly one float64 addition in the caller-specified order.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - The caller validates input operands; this routine does not sanitize them.
//
// AI-Hints:
//   - Do not replace the ordered fold with Kahan, sorting, or an exact accumulator silently.
func checkedMSTWeightAdd(prefix, weight float64) (float64, error) {
	next := prefix + weight
	if math.IsNaN(next) || math.IsInf(next, 0) {
		return 0, ErrWeightOverflow
	}
	return next, nil
}

// checkedMSTEdgeTotal computes a publishable network cost in the supplied edge order.
// It deliberately does not define the selection order or sort the caller's edge slice.
// Implementation:
//   - Stage 1: Initialize the zero prefix.
//   - Stage 2: Accumulate every selected edge with the shared checked addition.
//   - Stage 3: Publish a finite cost only after all prefixes succeed.
//
// Behavior highlights:
//   - Negative finite weights remain valid for ordinary MST/MSF.
//
// Inputs:
//   - edges: ordered, detached, finite core.Edge values, read without mutation.
//
// Returns:
//   - float64: the final checked cost, or zero on failure.
//
// Errors:
//   - ErrWeightOverflow from a non-finite intermediate sum.
//
// Determinism:
//   - Iteration follows the existing slice order exactly.
//
// Complexity:
//   - Time O(E), Space O(1).
//
// Notes:
//   - Steiner callers sort final edges by ID before calling this function.
//
// AI-Hints:
//   - Do not evaluate closure or union-tree totals when only selected edges are needed.
func checkedMSTEdgeTotal(edges []core.Edge) (float64, error) {
	var total float64
	for _, edge := range edges {
		var err error
		total, err = checkedMSTWeightAdd(total, edge.Weight)
		if err != nil {
			return 0, err
		}
	}
	return total, nil
}

// checkedSteinerPairCount computes the number of unordered terminal pairs safely.
// It prevents integer overflow and impossible edge-slice sizes before allocating closure storage.
// Implementation:
//   - Stage 1: Reject negative terminal counts.
//   - Stage 2: Divide the even factor of k*(k-1) by two before multiplication.
//   - Stage 3: Check both integer representability and core.Edge backing-store size.
//
// Behavior highlights:
//   - For k<=1 no closure is needed.
//
// Inputs:
//   - terminalCount: a non-negative number of canonical terminal IDs.
//
// Returns:
//   - int: safe pair capacity, or zero on error.
//
// Errors:
//   - ErrSizeOverflow for overflowing cardinality or backing-store size.
//
// Determinism:
//   - Pure checked integer operations.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This is a representability guard, not a hard RSS or free-memory promise.
//
// AI-Hints:
//   - Never multiply k*(k-1) before checking; an overflow can precede the division.
func checkedSteinerPairCount(terminalCount int) (int, error) {
	if terminalCount < 0 {
		return 0, ErrSizeOverflow
	}
	if terminalCount <= 1 {
		return 0, nil
	}
	left, right := terminalCount, terminalCount-1
	if left%2 == 0 {
		left /= 2
	} else {
		right /= 2
	}
	if left > math.MaxInt/right {
		return 0, ErrSizeOverflow
	}
	pairs := left * right
	if uintptr(pairs) > uintptr(math.MaxInt)/unsafe.Sizeof(core.Edge{}) {
		return 0, ErrSizeOverflow
	}
	return pairs, nil
}

// checkedSteinerSliceCapacity validates a count before constructing a typed slice.
// It centralizes byte-size representability without inventing a process-memory budget.
// Implementation:
//   - Stage 1: Reject negative element counts.
//   - Stage 2: Check multiplication against the maximum representable int size.
//   - Stage 3: Return the original count for the caller's allocation.
//
// Behavior highlights:
//   - No actual allocation takes place here.
//
// Inputs:
//   - count: proposed element count; elementBytes: positive element width.
//
// Returns:
//   - int: unchanged safe count, or zero on error.
//
// Errors:
//   - ErrSizeOverflow for invalid or overflowing sizes.
//
// Determinism:
//   - Pure integer arithmetic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Runtime allocations can still fail because available memory is not controlled.
//
// AI-Hints:
//   - Keep checking size products even when explicit resource caps are zero.
func checkedSteinerSliceCapacity(count int, elementBytes uintptr) (int, error) {
	if count < 0 || elementBytes == 0 || uintptr(count) > uintptr(math.MaxInt)/elementBytes {
		return 0, ErrSizeOverflow
	}
	return count, nil
}

// checkedSteinerCountAdd adds two nonnegative element counts before an allocation.
// It makes composite map/slice capacity calculations safe even without caller limits.
// Implementation:
//   - Stage 1: Reject negative counts.
//   - Stage 2: Check the remaining representable int capacity before adding.
//   - Stage 3: Return the exact nonnegative total.
//
// Behavior highlights:
//   - Prevents overflow before make() sees an invalid capacity.
//
// Inputs:
//   - left, right: nonnegative element counts.
//
// Returns:
//   - int: exact sum, or zero on failure.
//
// Errors:
//   - ErrSizeOverflow if either count is negative or the sum exceeds MaxInt.
//
// Determinism:
//   - Pure integer comparison and addition.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Does not reserve memory or implement a byte budget.
//
// AI-Hints:
//   - Do not form len(a)+len(b) as a make capacity before checking overflow.
func checkedSteinerCountAdd(left, right int) (int, error) {
	if left < 0 || right < 0 || left > math.MaxInt-right {
		return 0, ErrSizeOverflow
	}
	return left + right, nil
}
