// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package stats implements the Welford and Chan central-moment transitions.
// This file owns every mutable accumulator transition and its commit boundary.
package stats

import "math"

// NewMoments allocates an empty moments accumulator with valid zero state.
// A plain zero-valued Moments variable has exactly the same behavior, so this
// constructor is only a convenience for callers wanting pointer ownership.
// Implementation:
//   - Stage 1: allocate the zero-valued private state.
//   - Stage 2: return it without allocating any matrix or sample history.
//
// Behavior highlights:
//   - The first successful Update initializes the observed extrema.
//
// Inputs: none.
// Returns: a unique mutable *Moments owned by the caller.
// Errors: none.
// Determinism: empty state is identical on every invocation.
// Complexity: O(1) time and space.
// Notes: Moments mutation is not concurrency-safe; use independent partitions.
// AI-Hints: use one accumulator per goroutine, then Merge in a fixed order.
func NewMoments() *Moments {
	return &Moments{}
}

// Update adds a finite observation using compensated, scaled Welford moments.
// Instead of materializing delta*delta in float64, it maintains extended-range
// M2 and preserves small center corrections that otherwise disappear at offset.
// Implementation:
//   - Stage 1: validate receiver, value, and available uint64 count capacity.
//   - Stage 2: construct new mean and M2 as local candidate values.
//   - Stage 3: commit all fields in a single assignment after success.
//
// Behavior highlights:
//   - M2' = M2 + (x-mean)^2*n/(n+1); no cancellation-prone E[x^2]-E[x]^2.
//   - The first observation sets a zero M2 and initializes exact extrema.
//
// Inputs:
//   - value: one finite float64 observation; it is never retained by reference.
//
// Returns:
//   - nil if one observation has been committed atomically.
//
// Errors:
//   - ErrNilInput: nil receiver.
//   - ErrNonFinite: NaN or either infinity, before any mutation.
//   - ErrOverflow: Count already equals MaxUint64.
//
// Determinism:
//   - Each update consumes exactly one observation in caller-specified order.
//
// Complexity:
//   - Time O(1), Space O(1); no big integers or per-sample allocations.
//
// Notes:
//   - On error every field is unchanged; a successful Update may hold M2>MaxFloat64.
//   - Callers must serialize mutations of a shared accumulator.
//
// AI-Hints:
//   - Do not compute the delta as raw float64 subtraction before scaling.
//   - Materialize only through Summary after selecting ddof.
func (m *Moments) Update(value float64) error {
	// Stage 1: validation happens before even reading private numeric state.
	if m == nil {
		return ErrNilInput
	}
	if !finite(value) {
		return ErrNonFinite
	}
	if m.count == math.MaxUint64 {
		return ErrOverflow
	}

	// Stage 2a: the empty accumulator has an exact one-sample identity state.
	candidate := *m
	if m.count == 0 {
		candidate.count = 1
		candidate.mean = scaledFromFloat(value)
		candidate.m2 = scaledNumber{}
		candidate.exactMeanSum.addFloat(value)
		candidate.min = value
		candidate.max = value
		*m = candidate
		return nil
	}

	// Stage 2b: preserve n/(n+1) in extended arithmetic, including counts
	// above 2^53 where a plain float64 count loses low-order integer bits.
	nextCount := m.count + 1
	oldCountScale := scaledFromCount(m.count)
	nextCountScale := scaledFromCount(nextCount)
	delta := scaledFromFloat(value).sub(m.mean)

	// Welford's centered mean shift is delta/(n+1). The mathematically
	// equivalent nonnegative M2 increment is delta^2*n/(n+1).
	candidate.mean = m.mean.add(delta.div(nextCountScale))
	increment := delta.mul(delta).mul(oldCountScale).div(nextCountScale)
	candidate.m2 = m.m2.add(increment)
	candidate.count = nextCount
	// An exact fixed-point sum tracks cancellation across the entire float64
	// exponent range; it is detached because candidate is a value copy.
	candidate.exactMeanSum.addFloat(value)

	// Extrema apply to the ORIGINAL observations, not the rounded mean.
	if value < candidate.min {
		candidate.min = value
	}
	if value > candidate.max {
		candidate.max = value
	}

	// Stage 3: one publication point guarantees complete rollback on failures.
	*m = candidate
	return nil
}

// Merge combines two independent accumulators via Chan-Golub-LeVeque moments.
// The source remains read-only and detached; the destination changes exactly
// once after the count and alias checks have succeeded.
// Implementation:
//   - Stage 1: reject nil receivers, self-merge, and counter overflow.
//   - Stage 2: apply empty identities or compute the scaled Chan cross term.
//   - Stage 3: commit the complete candidate to the destination only.
//
// Behavior highlights:
//   - M2 = M2A+M2B+delta^2*nA*nB/(nA+nB) without float64 overflow.
//   - Empty source is a no-op; empty destination adopts source by value.
//
// Inputs:
//   - other: read-only *Moments with independent identity; it must not mutate concurrently.
//
// Returns:
//   - nil on success, including an empty-source no-op.
//
// Errors:
//   - ErrNilInput: nil receiver or nil other.
//   - ErrInvalidOption: destination and source are the same pointer.
//   - ErrOverflow: adding the uint64 counts exceeds MaxUint64.
//
// Determinism:
//   - Dest-then-source combination order is fixed; floating grouping is not bitwise associative.
//
// Complexity:
//   - Time O(1), Space O(1) with bounded-width compensated arithmetic.
//
// Notes:
//   - Every failed merge preserves both operands, including compensation and extrema.
//   - Concurrent Update/Merge on either operand requires caller synchronization.
//
// AI-Hints:
//   - Do not compute nA*nB as uint64: it can overflow before division by n.
//   - Merge partition trees in a reproducible order when results are persisted.
func (m *Moments) Merge(other *Moments) error {
	// Stage 1: the self-merge rule precedes the other-input nil check.
	if m == nil {
		return ErrNilInput
	}
	if m == other {
		return ErrInvalidOption
	}
	if other == nil {
		return ErrNilInput
	}
	if other.count > math.MaxUint64-m.count {
		return ErrOverflow
	}

	// Stage 2a: the empty source contributes no observations or extrema.
	if other.count == 0 {
		return nil
	}
	if m.count == 0 {
		// All internal scaled fields are values, so this copy cannot alias.
		*m = *other
		return nil
	}

	// Stage 2b: delay construction of nA*nB until it is a scaled value.
	// The sum of counts has already been proved representable in uint64.
	combinedCount := m.count + other.count
	countA := scaledFromCount(m.count)
	countB := scaledFromCount(other.count)
	countTotal := scaledFromCount(combinedCount)
	delta := other.mean.sub(m.mean)

	// The merged mean is meanA + delta*(nB/nTotal); the M2 correction
	// measures between-partition separation, not within-partition noise.
	candidate := *m
	candidate.mean = m.mean.add(delta.mul(countB).div(countTotal))
	cross := delta.mul(delta).mul(countA).mul(countB).div(countTotal)
	candidate.m2 = m.m2.add(other.m2).add(cross)
	candidate.count = combinedCount
	// Exact summation composes in a fixed low-to-high limb order and never
	// shares storage between independent accumulators.
	candidate.exactMeanSum.add(other.exactMeanSum)

	// Merging extrema is independent of the partition weighting.
	if other.min < candidate.min {
		candidate.min = other.min
	}
	if other.max > candidate.max {
		candidate.max = other.max
	}

	// Stage 3: publish a fully assembled state, never a partially merged one.
	*m = candidate
	return nil
}
