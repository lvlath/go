// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw exposes the squared LB_Keogh envelope score as a diagnostic function.
// This file does not authorize machine-level pruning without a certified float64 enclosure.
package dtw

import (
	"errors"
	"fmt"
	"math"
)

// LBKeoghSquared computes the squared envelope deviation of a finite query sequence.
// For valid DTW radius-derived envelopes, the real-arithmetic sum is a mathematical
// lower bound on squared-local-cost DTW; the returned float64 is rounded, not certified.
//
// Implementation:
//   - Stage 1: validate nil, lengths, all finite samples, and lower<=upper domain.
//   - Stage 2: compute squared distance from each query sample to its envelope interval.
//   - Stage 3: accumulate with the classified checked nonnegative binary64 addition.
//
// Behavior highlights:
//   - A sample inside [lower[i],upper[i]] contributes zero.
//   - No window radius is inferred; callers must supply a mathematically valid envelope.
//   - Search uses an independently certified enclosure, never this rounded diagnostic as a direct cutoff.
//
// Inputs:
//   - query: nonempty finite scalar samples, borrowed read-only.
//   - lower, upper: finite same-length interval bounds with lower[i]<=upper[i].
//
// Returns:
//   - float64: non-negative squared envelope deviation in query sample units squared.
//   - error: nil only if every computed local term and sum fits finite float64.
//
// Errors:
//   - ErrNilInput: any slice is nil; ErrEmptyInput: zero query length.
//   - ErrShape: envelope size mismatch or a reversed interval.
//   - ErrNonFinite joined with ErrNaNInf: NaN/Inf in the supplied envelope or query.
//   - ErrOverflow joined with ErrNaNInf: unrepresentable squared deviation or sum.
//
// Determinism:
//   - Increasing-index accumulation with the same checked-add numerical policy as DTW.
//
// Complexity:
//   - Time O(L), Space O(1) where L=len(query).
//
// Notes:
//   - Mathematical LB_Keogh validity depends on correct radius-indexed envelopes.
//   - Rounding of the sum does not prove a conservative binary64 lower enclosure;
//     comparing it directly to a top-K cutoff can create false prunes near ties.
//
// AI-Hints:
//   - Do not silently use this rounded diagnostic score as a safe pruning certificate.
//   - Do not apply the bound to absolute-cost, asymmetric, or penalty-modified DTW.
func LBKeoghSquared(query, lower, upper []float64) (float64, error) {
	if query == nil || lower == nil || upper == nil {
		return 0, ErrNilInput
	}
	if len(query) == 0 {
		return 0, ErrEmptyInput
	}
	if len(query) != len(lower) || len(query) != len(upper) {
		return 0, ErrShape
	}

	for i := range query {
		if math.IsNaN(query[i]) || math.IsInf(query[i], 0) ||
			math.IsNaN(lower[i]) || math.IsInf(lower[i], 0) ||
			math.IsNaN(upper[i]) || math.IsInf(upper[i], 0) {
			return 0, fmt.Errorf("dtw: envelope index %d: %w", i, errors.Join(ErrNonFinite, ErrNaNInf))
		}
	}

	var total float64
	for i := range query {
		if lower[i] > upper[i] {
			return 0, fmt.Errorf("dtw: reversed envelope index %d: %w", i, ErrShape)
		}
		var deviation float64
		if query[i] < lower[i] {
			deviation = lower[i] - query[i]
		} else if query[i] > upper[i] {
			deviation = query[i] - upper[i]
		}
		local := deviation * deviation
		if math.IsNaN(local) || math.IsInf(local, 0) {
			return 0, fmt.Errorf("dtw: envelope squared deviation index %d: %w", i,
				errors.Join(ErrOverflow, ErrNaNInf))
		}
		var overflow bool
		total, overflow = checkedNonnegativeAdd(total, local)
		if overflow {
			return 0, fmt.Errorf("dtw: envelope accumulated deviation index %d: %w", i,
				errors.Join(ErrOverflow, ErrNaNInf))
		}
	}
	return total, nil
}
