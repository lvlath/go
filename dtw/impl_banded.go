// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw implements the specialized fixed-length squared-cost banded scorer.
// This file shares classified-state arithmetic and predecessor selection with the DTW kernel.
package dtw

import (
	"context"
	"errors"
	"math"
	"sync/atomic"
)

// bandWidth returns the maximum number of admitted columns in a square DTW band.
// The width bounds the worker's two row buffers independently of the query length.
//
// Implementation:
//   - Stage 1: treat -1 or sufficiently wide nonnegative radii as full width.
//   - Stage 2: compute 2*radius+1 only when it is known to fit below length.
//   - Stage 3: return the positive row capacity.
//
// Behavior highlights:
//   - A strict-diagonal radius of zero needs one DP cell per row.
//
// Inputs:
//   - length: positive shared length; radius: valid -1 or nonnegative band radius.
//
// Returns:
//   - int: maximum live row width, in DP cells.
//
// Errors:
//   - None after caller validation.
//
// Determinism:
//   - Pure integer arithmetic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Use this width for workspace accounting before allocating per-worker rows.
//
// AI-Hints:
//   - Do not compute 2*radius+1 for an unbounded positive int radius.
//   - This width controls storage; the actual row boundaries still require clipping.
func bandWidth(length, radius int) int {
	if radius < 0 || radius >= length/2 {
		return length
	}
	return 2*radius + 1
}

// reserveSearchCell charges exactly one global cell credit before computing a local cost.
// Each worker checks the same atomic counter, so a hard MaxCells limit cannot be exceeded.
//
// Implementation:
//   - Stage 1: use a single atomic increment without a finite limit.
//   - Stage 2: for a finite limit, read the current count and reject exhaustion.
//   - Stage 3: reserve one credit through compare-and-swap or retry a concurrent update.
//
// Behavior highlights:
//   - Unused credits are never pre-reserved; the count includes interrupted candidates.
//
// Inputs:
//   - visited: shared counter; limit: zero means unlimited, otherwise a hard cap.
//
// Returns:
//   - error: nil for one reserved cell credit.
//
// Errors:
//   - ErrBudget: attempting to evaluate a cell beyond the hard limit.
//
// Determinism:
//   - Single-worker execution has a deterministic counted prefix.
//
// Complexity:
//   - Expected O(1) per credit; CAS contention may cause retries with workers.
//
// Notes:
//   - Atomic counter is per Search invocation; no global budget state is shared.
//
// AI-Hints:
//   - Never increment first and check later: that could exceed MaxCells.
//   - Count a cell even if its cost calculation returns an error after reservation.
func reserveSearchCell(visited *atomic.Uint64, limit uint64) error {
	if limit == 0 {
		visited.Add(1)
		return nil
	}
	for {
		before := visited.Load()
		if before >= limit {
			return ErrBudget
		}
		if visited.CompareAndSwap(before, before+1) {
			return nil
		}
	}
}

// scoreSquaredBand evaluates one fixed-length squared-DTW candidate with two band-only rows.
// It uses the same classified-state and strict-minimum predecessor arithmetic as Align,
// but visits only cells satisfying |i-j|<=radius and publishes no path or matrix artifacts.
//
// Implementation:
//   - Stage 1: initialize a virtual origin and two row slices of bandWidth cells.
//   - Stage 2: scan row i and admitted columns j in ascending order, reserving one cell credit.
//   - Stage 3: compare diagonal/up/left states, add squared local cost, and classify the final state.
//
// Behavior highlights:
//   - For radius=0 exactly L cells are evaluated, rather than L*L rectangular positions.
//   - Overflowed discarded transitions are not errors; terminal optimum overflow is.
//   - The fixed scalar scorer has zero slope penalty and no configurable CostFunc.
//
// Inputs:
//   - ctx: cooperative per-cell cancellation context; query, candidate: finite equal-length samples.
//   - radius: -1 or nonnegative window; prev, curr: private worker buffers of bandWidth size.
//   - visited, maxCells: global atomic exact-work accounting and hard limit.
//
// Returns:
//   - float64: finite exact-to-binary64 accumulated score for this recurrence.
//   - error: context, budget, shape, or numeric failure; no partial distance.
//
// Errors:
//   - ErrShape for mismatched sequences or undersized row buffers.
//   - ErrBudget for exhausted global credits; ErrOverflow joined with ErrNaNInf for numeric overflow.
//   - context cancellation/deadline error when observed before a cell.
//
// Determinism:
//   - Increasing i and j, strictly smaller predecessors only: diagonal, up, then left.
//
// Complexity:
//   - Time O(L*b), Space O(b) scratch, where b=min(L,2r+1), or b=L for radius=-1.
//
// Notes:
//   - This is a specialized geometry of the canonical DTW recurrence, not a second public engine.
//   - Caller owns and reuses rows; neither input slice nor row storage escapes in a result.
//
// AI-Hints:
//   - Do not replace actual j-loop bounds with a full-grid scan and a conditional continue.
//   - Do not silently turn an overflowing local squared difference into an unreachable path.
func scoreSquaredBand(
	ctx context.Context,
	query, candidate []float64,
	radius int,
	prev, curr []dpCell,
	visited *atomic.Uint64,
	maxCells uint64,
) (float64, error) {
	length := len(query)
	if length == 0 || len(candidate) != length ||
		len(prev) < bandWidth(length, radius) || len(curr) < bandWidth(length, radius) {
		return 0, ErrShape
	}

	previousStart, previousEnd := 0, 0 // Half-open predecessor row range.
	for i := 0; i < length; i++ {
		// Compute a clipped half-open band. Use length-i-1 to prevent int overflow
		// when the user supplies a very large but valid nonnegative Radius.
		start, end := 0, length
		if radius >= 0 {
			if i > radius {
				start = i - radius
			}
			if radius < length-1-i {
				end = i + radius + 1
			}
		}

		for j := start; j < end; j++ {
			if err := ctx.Err(); err != nil {
				return 0, err
			}
			if err := reserveSearchCell(visited, maxCells); err != nil {
				return 0, err
			}

			// Square a difference, never the original coordinates before subtraction.
			// A non-representable admitted local cost is a numerical failure even if
			// the optimal path might later discard this cell.
			difference := query[i] - candidate[j]
			localCost := difference * difference
			if math.IsInf(localCost, 0) || math.IsNaN(localCost) {
				return 0, errors.Join(ErrOverflow, ErrNaNInf)
			}

			if i == 0 && j == 0 {
				curr[j-start] = dpCell{class: dpFinite, cost: localCost}
				continue
			}

			// Missing predecessors remain dpUnreachable, independently of a
			// finite or unrepresentable prefix. No old-buffer cell is read outside
			// the previous row's half-open range.
			var diagonal, up, left dpCell
			if i > 0 && j > 0 && j-1 >= previousStart && j-1 < previousEnd {
				diagonal = prev[j-1-previousStart]
			}
			if i > 0 && j >= previousStart && j < previousEnd {
				up = prev[j-previousStart]
			}
			if j > start {
				left = curr[j-1-start]
			}
			best, _ := choosePredecessor(diagonal, up, left, 0)
			curr[j-start] = best.addCost(localCost)
		}
		previousStart, previousEnd = start, end
		prev, curr = curr, prev
	}

	final := prev[length-1-previousStart]
	switch final.class {
	case dpFinite:
		return final.cost, nil
	case dpOverflow:
		return 0, errors.Join(ErrOverflow, ErrNaNInf)
	default:
		// A square alignment always has an admitted diagonal for radius>=0.
		// Treat a missing terminal state as a violated internal invariant.
		return 0, ErrNoPath
	}
}
