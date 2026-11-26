// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package donchian builds aligned channels with one causal rolling-window kernel.
// This file keeps prior-window publication separate from current-bar insertion.
package donchian

import "context"

// compute fills private output channels using one pair of monotonic index queues.
// The two modes differ only in when an already complete window is read.
//
// Implementation:
//   - Stage 1: skip deque allocation when the entire input is warmup.
//   - Stage 2: scan rows, polling context and publishing prior windows before insertion
//     or inclusive windows afterward.
//   - Stage 3: return to Compute, which alone may expose the complete artifact.
//
// Behavior highlights:
//   - Exclude-current first becomes ready at P; include-current at P-1.
//   - Warmup rows are never written and retain allocation-provided positive zeros.
//
// Inputs:
//   - ctx: non-nil context already admitted by the facade.
//   - high, low: equal-length finite read-only prices with valid per-row geometry.
//   - opts: Period >= 1; result: private owning zeroed channels of the input length.
//
// Returns:
//   - error: nil after completion; the caller discards result on any failure.
//
// Errors:
//   - ctx.Err(): from per-row polling or either deque's long removal burst.
//
// Determinism:
//   - Increasing observation index, upper deque before lower deque, latest-index ties.
//
// Complexity:
//   - O(n) total time; each index is inserted and removed at most once per deque.
//   - O(min(P,n)) index workspace; zero deque allocations when no result is ready.
//
// Notes:
//   - At entry to iteration i, retained candidates cover the prior window ending i-1.
//   - The lower bound i-(P-1) cannot overflow with i >= 0 and P >= 1.
//   - A partial private result is not a partial public result; no caller aliases it.
//
// AI-Hints:
//   - Moving expiration before the exclude read drops the oldest valid prior bar.
//   - Moving insertion before that read contaminates the threshold with the tested bar.
//   - Do not allocate by Period until the warmup-only path has returned.
func compute(ctx context.Context, high, low []float64, opts Options, result *Result) error {
	// Validation guarantees both input and every output channel have this length.
	n := len(high)
	// No row can be ready: leave the owning outputs masked without allocating rings.
	if opts.Period > n || (opts.ExcludeCurrent && opts.Period == n) {
		return nil
	}
	// Here 1 <= Period <= n. Each ring has exactly Period slots, no sentinel
	// slot and no Period+1 or 2*Period allocation arithmetic.
	upper := indexDeque{indices: make([]int, opts.Period)}
	lower := indexDeque{indices: make([]int, opts.Period)}
	// At loop entry, rings describe retained candidates through observation i-1.
	for i := range high {
		// This also covers repeated bars with no long tail-removal bursts.
		if i%pollInterval == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		// Read BEFORE expiration/insertion: the prior state is exactly [i-P,i-1].
		if opts.ExcludeCurrent && i >= opts.Period {
			writeRow(result, i, high[upper.indices[upper.head]], low[lower.indices[lower.head]])
		}
		// i-(Period-1) cannot overflow: i >= 0 and Period >= 1.
		first := i - (opts.Period - 1)
		// Retain [i-P+1,i-1] so adding i constructs the next complete inclusive window.
		upper.expire(first)
		lower.expire(first)
		// The rings use their own price channel and share no candidate storage.
		if err := upper.insert(ctx, high, i, true); err != nil {
			return err
		}
		if err := lower.insert(ctx, low, i, false); err != nil {
			return err
		}
		// Read AFTER insertion: the current state is exactly [i-P+1,i].
		if !opts.ExcludeCurrent && i >= opts.Period-1 {
			writeRow(result, i, high[upper.indices[upper.head]], low[lower.indices[lower.head]])
		}
	}
	// The public facade still performs the final cancellation/publication check.
	return nil
}

// writeRow stores the observed extrema, their midpoint, and readiness for one row.
// It is called only after both queue heads represent a complete admitted window.
//
// Implementation:
//   - Stage 1: store the selected Upper without transforming its bit pattern.
//   - Stage 2: calculate Middle once and store the selected Lower.
//   - Stage 3: mark the three-channel row valid.
//
// Behavior highlights:
//   - Ready zero-valued rows are valid; untouched warmup zeros remain invalid.
//
// Inputs:
//   - result: private owning output; index: an in-range row known to be ready.
//   - upper, lower: finite selected extrema satisfying upper >= lower.
//
// Returns:
//   - No value; updates only the indicated row in the private result.
//
// Errors:
//   - None; midpoint cannot overflow on the validated domain.
//
// Determinism:
//   - Upper, Middle, Lower, then Valid are assigned in fixed order.
//
// Complexity:
//   - O(1) time and O(1) extra space.
//
// Notes:
//   - Setting Valid last is a local invariant, not a concurrent publication mechanism.
//   - Compute publishes the whole result after its final context check.
//
// AI-Hints:
//   - Never mark readiness by testing whether a numerical boundary is nonzero.
func writeRow(result *Result, index int, upper, lower float64) {
	// Keep the actual selected high, including its signed-zero tie result.
	result.Upper[index] = upper
	// Round the exact boundary mean without forming an overflowing sum or difference.
	result.Middle[index] = midpoint(upper, lower)
	// Keep the actual selected low rather than reconstructing it from width.
	result.Lower[index] = lower
	// This ready row now has all three payloads; warmup rows remain untouched.
	result.Valid[index] = true
}
