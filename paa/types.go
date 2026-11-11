// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
//
// Package paa publishes exact fractional segment coordinates and aggregated
// series metadata. This file defines the caller-owned geometric result model;
// no data container in this file retains an input slice or a private workspace.

package paa

// Segment stores one half-open PAA interval on the original sample-time axis.
// It represents [LeftNumerator/Denominator, RightNumerator/Denominator) exactly,
// allowing fractional boundaries without rounding sample indices or times.
//
// Implementation:
//   - Stage 1: use the validated original length n and output count w.
//   - Stage 2: represent segment j by [j*n,(j+1)*n) in common units of 1/w.
//   - Stage 3: publish the unsimplified representation in result order.
//
// Behavior highlights:
//   - Denominator is always the requested w (>0), even for reducible fractions.
//   - Adjacent successful segments share exact endpoints and cover [0,n).
//
// Inputs:
//   - Segment values are constructed by Transform or TransformRange; no
//     caller-provided Segment is required when invoking the algorithm.
//
// Returns:
//   - A value-type interval descriptor, copied with ordinary struct assignment.
//
// Errors:
//   - None from Segment itself; coordinate overflow is rejected by Transform
//     before geometry is constructed (ErrOverflow).
//
// Determinism:
//   - Result.Segments[j] describes segment j; indices increase left to right.
//
// Complexity:
//   - Time O(1) to read or copy; space O(1), exactly three int64 fields.
//
// Notes:
//   - Coordinates are sample-position units, not timestamps or durations.
//   - TransformRange expresses coordinates relative to its selected window.
//   - A zero-value Segment does not describe a successful output interval.
//
// AI-Hints:
//   - Do not simplify fractions: downstream SAX consumers can compare grids
//     reliably using the common denominator instead of rounding float indices.
//   - To recover an absolute source offset for a range, preserve start outside
//     Result and check any arithmetic used to add it to rational coordinates.
//   - This struct has no pointers; editing one returned element does not mutate
//     its source input or any other independently computed result.
type Segment struct {
	// LeftNumerator is the inclusive left endpoint j*n on the common 1/w grid.
	// Divide by Denominator to obtain the exact window-local sample coordinate.
	LeftNumerator int64

	// RightNumerator is the exclusive endpoint (j+1)*n on the same grid.
	// Its value equals the next segment's LeftNumerator, when one exists.
	RightNumerator int64

	// Denominator is the original requested segment count w (always positive).
	// It remains unsimplified to preserve a canonical shared coordinate grid.
	Denominator int64
}

// Result describes an entire fractional Piecewise Aggregate Approximation.
// Values[j] is the correctly-rounded average of the input step function over
// Segments[j], while OriginalLength records the number of original samples.
//
// Implementation:
//   - Stage 1: validate the complete public input domain before allocation.
//   - Stage 2: allocate independent Values and Segments buffers of length w.
//   - Stage 3: publish the complete artifact after the canonical kernel succeeds.
//
// Behavior highlights:
//   - len(Values) == len(Segments) == w and OriginalLength == n on success.
//   - Values are finite, lie within their contributing inputs' convex hull,
//     and represent nearest-even rounded exact binary-weighted means.
//   - The identity partition w=n copies input bits, including negative zero.
//
// Inputs:
//   - Created from a read-only finite []float64 and a required w in [1,n].
//
// Returns:
//   - *Result: a detached, caller-owned snapshot with no retained input view.
//   - No partial Result is returned by the public facades on error.
//
// Errors:
//   - The container itself does not return errors; facade validation and
//     arithmetic failures return the package-owned sentinels and nil results.
//
// Determinism:
//   - Values and Segments are ordered by increasing segment index, with exact
//     integer boundary comparisons and no sorting or adaptive partitioning.
//
// Complexity:
//   - Space O(w) for the two slices; field access and IsNil run in O(1).
//
// Nilability:
//   - *Result implements core.Nilable structurally, without importing core.
//   - A nil receiver reports true; every non-nil receiver reports false.
//
// Notes:
//   - Both slices may be independently mutated by their owning caller after
//     return. Such mutation changes that Result, not the original observations.
//   - Independent calls share no mutable state; concurrent mutation of a
//     Result's exported fields needs caller synchronization.
//   - IsNil checks pointer presence only, including a typed-nil receiver;
//     it does not validate a zero-valued or caller-modified Result.
//   - Only successful facade calls establish the documented result invariants.
//
// AI-Hints:
//   - Never infer absolute timestamps from Segments: the caller owns cadence
//     and the origin timestamp, and must validate temporal uniformity upstream.
//   - Keep the full Result when SAX or a distance calculation needs exact
//     geometry; Means intentionally discards that information.
//   - Conservation is exact for the underlying step function; summing rounded
//     output Values in float64 need not reproduce the original sum bit-for-bit.
type Result struct {
	// Values contains w segment averages in ascending segment order.
	// It is detached from input and Segments, and is fully caller-owned.
	Values []float64

	// Segments contains w exact rational half-open boundaries for Values.
	// It is detached and uses the same constant Denominator across all cells.
	Segments []Segment

	// OriginalLength is the number n of input samples used for this result.
	// For TransformRange it is end-start, not the length of the parent slice.
	OriginalLength int
}

// IsNil reports whether the Result receiver is a nil pointer.
// It implements core.Nilable so callers can recognize a typed-nil *Result even
// when the pointer is stored inside a non-nil interface value.
//
// Implementation:
//   - Stage 1: Compare the receiver pointer directly with nil.
//   - Stage 2: Return that comparison without reading or changing result data.
//
// Behavior highlights:
//   - A nil receiver returns true, including through core.Nilable.
//   - Every non-nil pointer returns false, including a zero-valued Result.
//
// Inputs:
//   - r: possibly nil *Result; no Values, Segments or metadata are accessed.
//
// Returns:
//   - bool: true exactly when r == nil.
//
// Errors:
//   - None; this predicate cannot panic and does not allocate.
//
// Determinism:
//   - A pure pointer comparison, independent of segment values and mutation.
//
// Complexity:
//   - Time O(1), additional space O(1), zero allocations.
//
// Notes:
//   - IsNil checks pointer presence, not successful Transform invariants.
//   - It does not classify an empty or manually modified Result as nil.
//   - Concurrent invocations are safe, since no mutable fields are read.
//
// AI-Hints:
//   - Never test len(r.Values)==0 as a substitute for nil: a non-nil
//     zero-valued Result must report false.
//   - Do not use reflect or validate numerical fields inside IsNil; callers
//     must use the public facades to obtain a contract-valid Result.
func (r *Result) IsNil() bool {
	return r == nil
}
