// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package series verifies exact signed-nanosecond timestamp intervals.
// This file owns grid traversal, overflow-safe deltas, and gap classification.
package series

import "math"

// validateGrid enforces one finite observation per ascending timestamp.
// It checks representability of every positive integer-nanosecond delta before
// applying the optional exact-cadence policy or recording missing slots.
//
// Implementation:
//   - Stage 1: Validate cadence, nil inputs, lengths, emptiness, and all values.
//   - Stage 2: Scan adjacent times, reject invalid order and overflowing deltas.
//   - Stage 3: When cadence is positive, classify nonmultiples and whole gaps.
//   - Stage 4: Publish sample count with detached, ordered gap records.
//
// Behavior highlights:
//   - CadenceNS == 0 skips only cadence division, never delta representation checks.
//   - AllowGaps allows missing whole slots, not arbitrary elapsed intervals.
//
// Inputs:
//   - timesNS: caller-owned signed int64 timestamps in nanoseconds, read-only.
//   - values: read-only finite float64 observations with identical input length.
//   - opts: nonnegative cadence and explicit whole-gap permission.
//
// Returns:
//   - *ValidationResult: count of supplied samples and non-nil detached Gaps.
//   - error: a failed call publishes nil, never partially accumulated gap metadata.
//
// Errors:
//   - ErrInvalidOption: negative cadence.
//   - ErrNilInput, ErrShape, ErrEmptyInput, ErrNonFinite: structural/finite checks.
//   - ErrOrder: equal or descending adjacent timestamps.
//   - ErrOverflow: a positive elapsed interval exceeds MaxInt64 nanoseconds.
//   - ErrGap: a nonmultiple of cadence or a forbidden missing whole slot.
//
// Determinism:
//   - Traverse adjacent pairs from (0,1) onward; gap records use original indices.
//
// Complexity:
//   - Time O(n), additional space O(g), with g reported gaps.
//
// Notes:
//   - Caller mutation during validation is unsupported; no input is sorted or copied.
//   - All delta arithmetic is exact integer arithmetic, without float conversion.
//
// AI-Hints:
//   - Signed subtraction before an overflow check is unsafe: use unsigned distance.
//   - Do not infer or fill cadence positions from a zero-valued GridOptions.
//
// nolint:gosec
func validateGrid(timesNS []int64, values []float64, opts GridOptions) (*ValidationResult, error) {
	// Reject invalid policy before dereferencing or inspecting either input slice.
	if opts.CadenceNS < 0 {
		return nil, ErrInvalidOption
	}

	// Nil inputs take precedence over shape checks, including nil/empty mixtures.
	if timesNS == nil || values == nil {
		return nil, ErrNilInput
	}
	// A missing observation is a shape error even when one slice is empty.
	if len(timesNS) != len(values) {
		return nil, ErrShape
	}
	// Equal, allocated-empty slices have their own error class.
	if len(timesNS) == 0 {
		return nil, ErrEmptyInput
	}
	// Reject all nonfinite values before reporting any timestamp-domain failure.
	if err := validateFinite(values); err != nil {
		return nil, err
	}

	// A non-nil Gaps slice is required even for order-only validation.
	// Its records are built in input-index order and never alias caller storage.
	result := &ValidationResult{Samples: len(values), Gaps: make([]Gap, 0)}

	// Each sample after the first contributes precisely one adjacent interval.
	for i := 1; i < len(timesNS); i++ {
		// Keep the signed operands for the strict ordering check.
		prev, next := timesNS[i-1], timesNS[i]
		if next <= prev {
			return nil, ErrOrder
		}

		// Subtracting after conversion to uint64 yields the exact positive delta
		// modulo 2^64. Since next > prev, it is the true mathematical distance
		// in [1, 2^64-1], even across the signed zero boundary.
		span := uint64(next) - uint64(prev)
		if span > uint64(math.MaxInt64) {
			return nil, ErrOverflow
		}
		// The checked cast is now safe, including MinInt64 -> -1.
		delta := int64(span)

		// A zero cadence validates only order and representable deltas;
		// it never guesses an expected interval or publishes a gap.
		if opts.CadenceNS == 0 {
			continue
		}

		// Exact grid alignment has no epsilon tolerance: a nonmultiple is
		// off-grid regardless of permission to skip whole cadence slots.
		if delta%opts.CadenceNS != 0 {
			return nil, ErrGap
		}

		// The positive quotient counts complete cadence intervals.
		// The first interval reaches the next supplied sample; the remaining
		// intervals represent missing samples, hence subtract one.
		missing := delta/opts.CadenceNS - 1
		if missing > 0 {
			// Rejection occurs before publishing any partially collected result.
			if !opts.AllowGaps {
				return nil, ErrGap
			}
			// The hole follows i-1 in the original input array.
			result.Gaps = append(result.Gaps, Gap{AfterIndex: i - 1, MissingCount: missing})
		}
	}

	// All adjacent intervals passed; publish only now.
	return result, nil
}
