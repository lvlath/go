// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
//
// Package paa provides exact fractional averaging of uniformly spaced observations.
// This file exposes the canonical public facade and its narrow convenience wrappers.

package paa

// Transform computes the exact-geometry fractional Piecewise Aggregate
// Approximation of uniformly spaced, finite observations. Every observation
// contributes its full interval measure; fractional boundary cells are split.
//
// Implementation:
//   - Stage 1: validate segment count, slice presence/shape, all finite values,
//     and the full integer endpoint-product range in strict precedence order.
//   - Stage 2: perform one two-pointer exact-overlap sweep with exact
//     fixed-width binary accumulation and one correctly-rounded conversion.
//   - Stage 3: publish detached means and unsimplified rational segments.
//
// Behavior highlights:
//   - 1 <= segments <= n is required; zero does not request a default.
//   - Each input value covers [i,i+1) sample-time units. Segment j covers
//     [j*n/segments,(j+1)*n/segments), including when n%segments != 0.
//   - Averaged values remain finite and inside their contributing input hull.
//   - When segments == n, a detached identity preserves all original bits,
//     including negative zero, without passing through numeric rounding.
//
// Inputs:
//   - values: read-only-for-call non-nil, nonempty finite float64 slice;
//     no irregular sampling, missing-values policy or timestamp units.
//   - segments: positive number of equal-duration output cells, at most n.
//
// Returns:
//   - *Result: n in OriginalLength, two independent caller-owned slices of
//     length segments; Segments contain checked exact rational boundaries.
//   - error: nil on success, otherwise a nil result and classified sentinel.
//
// Errors:
//   - ErrInvalidOption: segments<=0 first; segments>n after nil/empty checks.
//   - ErrNilInput: values==nil; ErrEmptyInput: non-nil empty values.
//   - ErrNonFinite: any NaN or infinity, checked before geometry overflow.
//   - ErrOverflow: n*segments cannot fit int64, or numeric output cannot be
//     represented by finite float64.
//
// Determinism:
//   - Scan indices i and j advance monotonically from zero; equal right
//     boundaries advance both, with no sorting or floating boundary compares.
//
// Complexity:
//   - Time O(n+segments), space O(segments), including validation and copies.
//
// Notes:
//   - The input is never modified or retained; concurrent caller mutation
//     is unsupported. Independent calls on unmodified inputs are concurrent-safe.
//   - Exact geometry and weighted-integral conservation are mathematical laws;
//     segment means are rounded once to binary64; reductions of these
//     rounded means may accumulate further floating-point error.
//
// AI-Hints:
//   - Never replace fractional overlap with floor(n/segments) block slicing.
//   - Never multiply a huge *float64* sample by a raw integer overlap; the
//     kernel instead multiplies exact mantissas as bounded-width integers.
//   - SAX consumers must use the same segment geometry and distance scaling.
func Transform(values []float64, segments int) (*Result, error) {
	// Validate the complete public input domain before allocating output storage.
	if err := validateTransform(values, segments); err != nil {
		return nil, err
	}

	// The single canonical kernel owns all fractional geometry and arithmetic.
	return transformKernel(values, segments)
}

// TransformRange computes PAA over one contiguous half-open sample-index range.
// It provides checked window selection without copying observations or changing
// the fractional geometry used by Transform.
//
// Implementation:
//   - Stage 1: reject a nonpositive segment count, validate source presence,
//     then check the half-open window bounds and count within that window.
//   - Stage 2: delegate to Transform(values[start:end], segments), the sole
//     validating facade and exact fractional averaging implementation.
//   - Stage 3: publish the delegated detached Result without retaining the input.
//
// Behavior highlights:
//   - Segment bounds are LOCAL to [start,end): segment zero begins at 0/w,
//     not at start/w; OriginalLength is end-start, not len(values).
//   - Observations outside the selected range are neither read nor validated.
//   - No timestamp gaps are synthesized, and no partial tail is discarded.
//
// Inputs:
//   - values: non-nil, nonempty read-only-for-call source observations.
//   - start: inclusive zero-based source sample index; must be nonnegative.
//   - end: exclusive source sample index; must satisfy start<end<=len(values).
//   - segments: required output count in [1,end-start], with no implicit default.
//
// Returns:
//   - *Result: exact window-local rational boundaries, detached Values and
//     Segments of length segments, OriginalLength=end-start.
//   - error: nil on complete success; otherwise nil *Result and a sentinel.
//
// Errors:
//   - ErrInvalidOption: nonpositive segments are rejected before source
//     presence; malformed bounds or segments>end-start follow nil/empty checks.
//   - ErrNilInput, ErrEmptyInput: nil or non-nil empty original source slice.
//   - ErrNonFinite: NaN or infinity inside the selected window only.
//   - ErrOverflow: checked integer geometry, output size or numeric failure.
//
// Determinism:
//   - Identical to Transform on the exact values[start:end] view, traversing
//     selected samples and emitting local segments in ascending index order.
//
// Complexity:
//   - Time O(end-start+segments), space O(segments), including validation,
//     output construction and the detached copy for an identity window.
//
// Notes:
//   - The window slice is read only for the duration of this call; the
//     caller must not concurrently modify its selected elements.
//   - Use the original start separately when mapping local segment coordinates
//     to a larger indexed series; this result does not store absolute offsets.
//
// AI-Hints:
//   - Do not shift Segment numerators alone: absolute coordinates are
//     (start*Denominator + numerator)/Denominator when representable.
//   - Use TransformRange instead of unchecked slicing when indices can come
//     from external input. Use Transform for an already validated window.
func TransformRange(values []float64, start, end, segments int) (*Result, error) {
	// The segment count is the only input-independent option. Its mandatory
	// positive-domain gate precedes any inspection of the source slice.
	if segments <= 0 {
		return nil, ErrInvalidOption
	}

	// Preserve Transform's nil-before-empty-before-shape precedence. In
	// particular, an empty original source is not a valid empty window.
	if values == nil {
		return nil, ErrNilInput
	}
	if len(values) == 0 {
		return nil, ErrEmptyInput
	}

	// Shape checks come after source presence and before any numeric scan.
	// The short-circuit order ensures end-start cannot underflow or exceed int.
	if start < 0 || start >= end || end > len(values) {
		return nil, ErrInvalidOption
	}
	if segments > end-start {
		return nil, ErrInvalidOption
	}

	// Delegation is intentionally the only computation path; a subrange is a
	// bounded view over the original series, not a second PAA algorithm.
	return Transform(values[start:end], segments)
}

// Means returns only the PAA segment means for callers that do not need geometry.
// It is a value-only projection of Transform, not a separate or faster numeric
// algorithm; callers requiring SAX-style boundary metadata must use Transform.
//
// Implementation:
//   - Stage 1: forward values and segment count to canonical Transform.
//   - Stage 2: retain only its Values slice on success, discarding the result
//     carrier and the temporary exact Segment metadata.
//   - Stage 3: return the detached means or nil with the original error.
//
// Behavior highlights:
//   - Uses the same fractional weighting and nearest-even final rounding,
//     except for the identical bit-preserving copy when segments == n.
//   - No truncation, padding, alternate numeric modes or hidden segment default.
//
// Inputs:
//   - values: non-nil, nonempty, finite read-only-for-call source samples.
//   - segments: required number of equal-duration averages, 1<=segments<=n.
//
// Returns:
//   - []float64: caller-owned detached means of length segments on success;
//     nil on failure. Exact segment boundaries are intentionally not returned.
//   - error: the unmodified error from Transform, or nil on success.
//
// Errors:
//   - ErrInvalidOption, ErrNilInput, ErrEmptyInput, ErrNonFinite, ErrOverflow:
//     the same sentinel identity and precedence as Transform.
//
// Determinism:
//   - Identical input-index traversal and ascending output order to Transform.
//
// Complexity:
//   - Time O(n+segments), peak additional space O(segments); the temporary
//     Segments slice is allocated even though this wrapper does not return it.
//
// Notes:
//   - Returned values do not alias input or other calls. Input must remain
//     stable throughout the operation, including the full validation pass.
//
// AI-Hints:
//   - Do not infer original length or rational segment positions from this
//     value-only slice: retain the full Transform Result for distance proofs.
//   - A result containing only means does not by itself define an absolute
//     time axis or qualify as a SAX lower-bound certificate.
func Means(values []float64, segments int) ([]float64, error) {
	// Retain the full public validation, geometry and numeric reduction policy.
	result, err := Transform(values, segments)
	if err != nil {
		return nil, err
	}

	// The result's Values slice is already detached. Returning it directly
	// avoids a second copy while discarding only unrequested metadata.
	return result.Values, nil
}
