// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package sessionlevels provides previous completed session High/Low/Close levels without look-ahead bias.
// This file defines the caller-owned segmentation and output contracts.
package sessionlevels

// Session identifies one nonempty half-open interval of observed bars.
// Complete is the caller's assertion of finality and accepted coverage, not a bar-count inference.
//
// Implementation:
//   - Stage 1: Compute checks bounds, adjacency, and completeness before reading prices.
//   - Stage 2: bars receive the preceding completed interval's summary.
//   - Stage 3: only a complete interval may become the next interval's predecessor.
//
// Behavior highlights:
//   - Only the last interval may be incomplete, including a single-interval input.
//
// Inputs:
//   - Start and End: positions in the supplied price slices, with 0 <= Start < End <= n.
//   - Complete: external assertion; the package does not authenticate a calendar.
//
// Returns:
//   - Session is a value descriptor and owns no buffers.
//
// Errors:
//   - ErrSession from Compute for invalid bounds, coverage, or an early incomplete interval.
//
// Determinism:
//   - Slice order is chronological; Compute never sorts or repairs it.
//
// Complexity:
//   - O(1) storage per descriptor; validation takes O(s) time and O(1) workspace.
//
// Notes:
//   - The caller must not mutate the sessions slice during Compute.
//
// AI-Hints:
//   - Clip End to the supplied history and set Complete=false for an intrainterval cut.
//   - Do not infer completeness from 1440 bars or from UTC midnight.
type Session struct {
	// Start is the inclusive index of the first observed bar in this session.
	// An empty session or a missing earlier prefix is rejected as ErrSession.
	Start int

	// End is the exclusive index immediately after the final observed bar.
	// End-1 is read as final close only after strict bounds validation.
	End int

	// Complete asserts that the caller has finalized both this interval's
	// bar availability and its session policy. Only the final interval may be
	// false; it still receives levels from the preceding complete interval.
	Complete bool
}

// Options reserves an explicit configuration position in the batch API.
// Its zero value is the only configuration and is valid without normalization.
//
// Implementation:
//   - Stage 1: accept the empty value without introducing hidden defaults.
//   - Stage 2: use the explicit sessions slice as the sole segmentation policy.
//   - Stage 3: publish the same channels for every call.
//
// Behavior highlights:
//   - There is no period, exchange, timezone, or include-current mode.
//
// Inputs:
//   - Pass Options{} to Compute.
//
// Returns:
//   - Options is a stateless value with no retained resources.
//
// Errors:
//   - None; there is no invalid configuration in this version.
//
// Determinism:
//   - Options introduces no alternate traversal or numeric policy.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Daily and weekly levels require separate caller-supplied partitions.
//
// AI-Hints:
//   - Do not invent a default session duration or an unreachable invalid-option error.
type Options struct{}

// Result contains aligned levels from the strictly preceding completed session.
// A false Valid entry means all three payloads are positive zero, not a measured zero.
//
// Implementation:
//   - Stage 1: allocate four independent zeroed output slices after validation.
//   - Stage 2: fill ready rows from previously committed scalar state.
//   - Stage 3: publish only after a final context check succeeds.
//
// Behavior highlights:
//   - The first session is entirely masked; later sessions are entirely ready.
//   - A successful empty result has four non-nil empty slices.
//
// Inputs:
//   - Result is normally produced by Compute; its zero value is not an empty Compute artifact.
//
// Returns:
//   - Three float64 slices and one bool slice, each of length n and owned by the caller.
//
// Errors:
//   - None from this carrier; Compute returns a nil result on every error.
//
// Determinism:
//   - Extrema retain the earliest equal value's bits; closing prices retain the final bar's bits.
//
// Complexity:
//   - O(n) storage and initialization; IsNil is O(1).
//
// Notes:
//   - Channels do not alias inputs, each other, or other calls. A value copy of Result is shallow.
//   - Sharing read-only results is safe; concurrent writes need caller synchronization.
//
// AI-Hints:
//   - Check Valid, not a zero payload or IsNil, before using a level.
//   - Copy a channel explicitly when a second independently mutable result is needed.
type Result struct {
	// PreviousHigh is the earliest equal maximum selected in the prior completed
	// session. On Valid=false every entry must be positive binary64 zero.
	PreviousHigh []float64

	// PreviousLow is the earliest equal minimum selected in the prior completed
	// session, with its exact float64 bits preserved, including signed zero.
	PreviousLow []float64

	// PreviousClose is the exact closing price of the final bar of the previous
	// completed session, never its high, low, midpoint or an inferred settlement.
	PreviousClose []float64

	// Valid distinguishes a completed predecessor from the first-session warmup.
	// A ready numerical level equal to positive or negative zero remains valid.
	Valid []bool
}

// IsNil reports whether the result pointer itself is nil.
// It implements the UES nilable-result convention without inspecting any channel.
//
// Implementation:
//   - Stage 1: accept the receiver, including a typed nil pointer.
//   - Stage 2: compare the pointer with nil.
//   - Stage 3: return that comparison unchanged.
//
// Behavior highlights:
//   - A non-nil zero-value, empty, or wholly masked result returns false.
//
// Inputs:
//   - r: any *Result, including nil.
//
// Returns:
//   - bool: true exactly when r == nil.
//
// Errors:
//   - None; the receiver is never dereferenced.
//
// Determinism:
//   - A pure pointer comparison, including through a nilable interface.
//
// Complexity:
//   - Time O(1), Space O(1), no allocation.
//
// Notes:
//   - This is neither readiness nor shape validation.
//
// AI-Hints:
//   - Do not replace the pointer comparison with reflection or slice inspection.
func (r *Result) IsNil() bool {
	return r == nil
}
