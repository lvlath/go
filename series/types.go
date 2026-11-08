// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package series represents ordered observations with explicit validity masks.
// This file defines the public data carriers and their ownership invariants.

package series

// GridOptions configures exact timestamp-grid validation without changing data.
// Its zero value requires strictly increasing, representable nanosecond intervals
// but intentionally does not infer cadence, gaps, UTC, or a calendar.
//
// Behavior highlights:
//   - A positive cadence requires every adjacent delta to be an exact multiple.
//   - AllowGaps permits whole missing slots; it cannot allow off-grid intervals.
//
// Notes:
//   - Configuration is passed by value; no retained policy state is created.
//
// AI-Hints:
//   - CadenceNS == 0 means order-only, not an automatically inferred cadence.
//   - Validate is not a resampling operation; missing values are never inserted.
type GridOptions struct {
	// CadenceNS is the expected positive grid step in nanoseconds.
	// Zero disables cadence checking but retains strict order and delta-overflow checks;
	// negative values are invalid options.
	CadenceNS int64

	// AllowGaps reports whole missing cadence slots rather than rejecting them.
	// This flag has no effect without a positive cadence and never permits
	// nonmultiple or nonpositive timestamp differences.
	AllowGaps bool
}

// Gap locates missing whole cadence slots between two supplied observations.
// It describes only input-index metadata; a Gap does not hold interpolated
// timestamps, new observations, or elapsed wall-clock calendar intervals.
//
// Notes:
//   - All returned Gap entries have MissingCount > 0 and increasing AfterIndex.
//   - Gap entries are caller-owned values stored in a detached result slice.
//
// AI-Hints:
//   - AfterIndex is an index in the original input, not a missing timestamp.
type Gap struct {
	// AfterIndex is the index immediately before the gap; the next supplied
	// observation is at AfterIndex+1, preserving original input order.
	AfterIndex int

	// MissingCount is the number of absent full cadence slots, computed as
	// (next-previous)/CadenceNS - 1; it is always strictly positive.
	MissingCount int64
}

// ValidationResult reports the shape and gap metadata of a validated time grid.
// Its successful fields describe the original input without adding timestamps
// or samples; no partial ValidationResult is returned on error.
//
// Notes:
//   - Gaps is always non-nil on success, including order-only validation.
//   - The caller owns the result and may mutate Gaps without changing inputs.
//
// AI-Hints:
//   - Samples counts supplied observations, not theoretical cadence slots.
type ValidationResult struct {
	// Samples is the number of validated timestamp/value pairs; it equals
	// the original len(values) and excludes all missing cadence slots.
	Samples int

	// Gaps contains detected whole-slot omissions in increasing AfterIndex order.
	// An empty successful grid has a non-nil, zero-length slice; entries do
	// not refer to any shared input storage.
	Gaps []Gap
}

// TransformResult is a length-preserving causal values-and-validity artifact.
// At each index i, Valid[i] declares whether Values[i] is a computed result;
// invalid warmup cells are exactly zero/false, never missing slice entries.
//
// Notes:
//   - len(Values) == len(Valid) == len(input); valid cells are finite float64.
//   - Both slices are detached, caller-owned, and independent of input memory.
//
// AI-Hints:
//   - Read Valid before interpreting a zero: zero may be a valid computed value.
//   - An output mask encodes insufficient history, not numeric NaN or a gap fill.
type TransformResult struct {
	// Values holds one float64 per original input position, including zeros
	// at invalid warmup indices. Every Valid[i] == true value is finite.
	Values []float64

	// Valid marks which corresponding Values entries contain computed output.
	// All missing-history positions are false, even when their zero value
	// coincides with a legitimate numeric return or difference.
	Valid []bool
}

// Range represents a complete half-open interval of sample indices.
// Every Range emitted by WindowRanges obeys 0 <= Start < End <= length and
// End-Start == width; End itself is not an included sample index.
//
// Notes:
//   - Ranges are index intervals, not timestamp/calendar windows.
//   - A Range is a value without retained storage or input aliases.
//
// AI-Hints:
//   - Iterate samples using i < End; i == End belongs to the next interval.
type Range struct {
	// Start is the inclusive index of the first sample in the window.
	// It begins at zero for the first emitted range and advances by stride.
	Start int

	// End is the exclusive index immediately after the last sample.
	// It equals Start+width and never exceeds the supplied length.
	End int
}
