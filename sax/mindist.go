// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// File: mindist.go
// Role: Exact symbolic-bin gap formula and scaled Euclidean lower-bound evaluation.
// AI-Hint: Positive Euclidean MINDIST does not certify arbitrary-DTW pruning.

package sax

import "math"

// Mindist computes the SAX Euclidean lower bound between two encoded windows.
// Both results must share the same original sample count and mathematical
// EncodingID, and must match the receiver's immutable encoding configuration.
//
// Implementation:
//   - Stage 1: validate the receiver, each independent Result and comparison
//     dimensions/identity; do not trust caller-mutable identity strings alone.
//   - Stage 2: scan matching segments in order, measuring the smallest distance
//     between corresponding disjoint quantization intervals.
//   - Stage 3: use a scaled Euclidean norm and multiply by sqrt(n/w), then
//     reject unrepresentable final distances without publishing a partial value.
//
// Behavior highlights:
//   - Equal/adjacent bins contribute zero; separated bins contribute the
//     difference between their closest finite boundary endpoints.
//   - The bound is on Euclidean distance in the declared transformed space:
//     Raw observations or separately Z-normalized observations, never arbitrary DTW.
//   - The fractional segment weight is exactly n/w, not floor(n/w).
//
// Inputs:
//   - left, right: non-nil readable Results of equal n, w and alphabet;
//     callers must not mutate their exported fields during the comparison.
//
// Returns:
//   - float64: finite, nonnegative symbolic Euclidean lower bound on success;
//     exactly 0 with a classified error on any failure.
//
// Errors:
//   - ErrNilInput: nil receiver/result; ErrInvalidOption: zero-value receiver;
//     ErrShape: malformed symbols/slices or inconsistent fitted metadata;
//     ErrNonFinite: corrupted non-finite metadata/optional PAA;
//     ErrIncompatible: different shape, identity or normalization version;
//     ErrOverflow: non-representable gap, norm or final distance.
//
// Determinism:
//   - Results are scanned from segment 0 to w-1 with fixed adjacent-bin rules.
//   - math.Hypot accumulates in that order; the final scale is sqrt(n/w).
//
// Complexity:
//   - Time O(w) without PAA; O(w log a) when optional diagnostic PAA is
//     checked against symbols; extra space O(1). Alphabet is bounded to 256.
//
// Notes:
//   - Floating-point rounding is not a substitute for a real-number proof;
//     downstream certified Euclidean pruning should account for its numerical
//     error budget before using an equality or distance cutoff.
//   - EncodingID establishes compatibility, not cryptographic authenticity
//     of Result objects that callers can construct or edit directly.
//
// AI-Hints:
//   - Never pass this positive lower bound as a DTW rejection certificate:
//     q=[0,0,1], c=[0,1,1] yield MINDIST 0.5 but unconstrained DTW 0.
//   - If ReturnPAA is disabled, validate symbols and metadata independently;
//     no public API can rederive their original observations from a word.
func (e *Encoder) Mindist(left, right *Result) (float64, error) {
	if err := e.validateEncoder(); err != nil {
		return 0, err
	}
	if left == nil || right == nil {
		return 0, ErrNilInput
	}

	// Validate each publicly mutable artifact before comparing its claimed
	// identity. A forged hash must never bypass symbol range or finite checks.
	if err := validateResult(left); err != nil {
		return 0, err
	}
	if err := validateResult(right); err != nil {
		return 0, err
	}

	// The scalar n/w belongs to the observation grid; comparing different
	// original lengths would produce a meaningless Euclidean scale.
	if left.OriginalLength != right.OriginalLength ||
		left.Segments != right.Segments || left.Alphabet != right.Alphabet ||
		left.Segments != e.segments || left.Alphabet != e.alphabet ||
		left.EncodingID != right.EncodingID || left.EncodingID != e.encodingID {
		return 0, ErrIncompatible
	}
	if err := e.validateResultForEncoder(left); err != nil {
		return 0, err
	}
	if err := e.validateResultForEncoder(right); err != nil {
		return 0, err
	}

	// With x in the lower bin and y in the higher bin, adjacent intervals
	// touch and contribute no positive separation. For disjoint intervals,
	// only the closest thresholds define the symbolic gap.
	norm := 0.0
	for index, leftSymbol := range left.Symbols {
		lower := int(leftSymbol)
		upper := int(right.Symbols[index])
		if lower > upper {
			lower, upper = upper, lower
		}
		if upper-lower <= 1 {
			continue
		}
		gap := e.breakpoints[upper-1] - e.breakpoints[lower]
		if !finite(gap) || gap < 0 {
			return 0, ErrOverflow
		}

		// Hypot scales its arguments internally; direct sum(gap*gap) could
		// overflow even when the final norm fits into float64.
		norm = math.Hypot(norm, gap)
		if !finite(norm) {
			return 0, ErrOverflow
		}
	}

	// All segments occupy equal continuous duration n/w, including when n
	// is not divisible by w. Since n>=w, the multiplier is at least one.
	factor := math.Sqrt(float64(left.OriginalLength) / float64(left.Segments))
	if norm > math.MaxFloat64/factor {
		return 0, ErrOverflow
	}
	answer := norm * factor
	if !finite(answer) {
		return 0, ErrOverflow
	}
	return answer, nil
}
