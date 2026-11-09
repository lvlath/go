// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package stats materializes fitted central moments as public float64 results.
// This file keeps all denominator and final-overflow handling in one kernel.
package stats

// Summary publishes a finite snapshot using population or sample variance.
// Intermediate M2 is held outside float64's exponent range, so overflow is
// reported only when the requested variance or mean cannot be represented.
// Implementation:
//   - Stage 1: validate ddof, receiver, and sample-count sufficiency.
//   - Stage 2: divide extended-range M2 by the exact integer denominator.
//   - Stage 3: materialize only finite values in an independent Summary.
//
// Behavior highlights:
//   - A mathematically positive variance may round to +0 in float64 underflow.
//   - A successful summary never contains NaN or infinity.
//
// Inputs:
//   - ddof: 0 (population) or 1 (sample); Count must exceed ddof.
//
// Returns:
//   - *Summary: independently owned count, mean, variance, and extrema.
//
// Errors:
//   - ErrInvalidOption: unsupported ddof, checked before receiver.
//   - ErrNilInput: nil receiver.
//   - ErrInsufficientSamples: empty or Count<=ddof.
//   - ErrOverflow: mean or final variance is not finite in float64.
//
// Determinism:
//   - Fixed denominator law and nearest-even rounding, with no state mutation.
//
// Complexity:
//   - Time O(1), Space O(1), including materialization.
//
// Notes:
//   - M2 may exceed MaxFloat64 without constituting an Update/Merge failure.
//   - An underflowed variance must not be used to classify a constant feature.
//
// AI-Hints:
//   - Do not convert M2 to float64 before dividing by Count-ddof.
//   - Use FitStandardizer's internal M2 for zero-variance and scale decisions.
func (m *Moments) Summary(ddof int) (*Summary, error) {
	// Validate the option before the receiver to preserve error precedence.
	if !validDDOF(ddof) {
		return nil, ErrInvalidOption
	}
	if m == nil {
		return nil, ErrNilInput
	}
	if m.count <= uint64(ddof) {
		return nil, ErrInsufficientSamples
	}

	// The denominator is an exact scaled uint64: float64(count) would lose
	// integer bits for large merged datasets with Count > 2^53.
	denominator := scaledFromCount(m.count - uint64(ddof))
	if m.m2.hi < 0 {
		return nil, ErrOverflow
	}
	mean, err := m.exactMeanSum.meanFloat(m.count)
	if err != nil {
		return nil, err
	}
	variance, err := m.m2.div(denominator).toFloat()
	if err != nil {
		return nil, err
	}

	// All output fields are copied by value; neither the scaled storage nor
	// the mutable receiver can alias this result.
	return &Summary{
		Count:    m.count,
		Mean:     mean,
		Variance: variance,
		Min:      m.min,
		Max:      m.max,
	}, nil
}

// summarizeKernel streams observations through the canonical Welford state.
// The public Summarize facade validates options and nil/empty input first;
// this private worker handles the ordered numerical reduction only.
// Implementation:
//   - Stage 1: initialize a local zero-valued accumulator.
//   - Stage 2: Update once per original slice element, in index order.
//   - Stage 3: call Summary with the already validated ddof.
//
// Inputs: values is non-nil and nonempty; ddof is 0 or 1.
// Returns: a detached Summary or nil on failure.
// Errors: ErrNonFinite, ErrOverflow, or ErrInsufficientSamples from the kernel.
// Determinism: strict left-to-right input sequence; no map or goroutine ordering.
// Complexity: Time O(len(values)), Space O(1), excluding the returned result.
// AI-Hints: do not replace the kernel with a sum-of-squares formula.
func summarizeKernel(values []float64, ddof int) (*Summary, error) {
	var accumulator Moments
	for _, value := range values {
		if err := accumulator.Update(value); err != nil {
			return nil, err
		}
	}
	return accumulator.Summary(ddof)
}
