// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package montecarlo checks the exact optional backing-storage budget.
// This file rejects arithmetic overflow and excess allocation before sampling.
package montecarlo

import "math"

// checkedMul computes an unsigned byte/element product without wrapping.
// It is a private building block for all optional sample-size calculations.
// Implementation:
//   - Stage 1: compare the first factor against the unsigned quotient.
//   - Stage 2: multiply only after representability is established.
//
// Inputs: a and b are nonnegative uint64 counts.
// Returns: the product or ErrOverflow when no uint64 representation exists.
// Determinism: pure integer arithmetic. Complexity: Time O(1), Space O(1).
// AI-Hints: never construct a slice length from an unchecked product.
func checkedMul(a, b uint64) (uint64, error) {
	// A zero factor has a zero product and must not become a divisor.
	if b != 0 && a > math.MaxUint64/b {
		return 0, ErrOverflow
	}
	return a * b, nil
}

// checkedAdd combines independently verified byte counts without wrapping.
// It keeps the memory budget a monotonic sum of all named buffers.
// Implementation:
//   - Stage 1: compare the first size with the remaining unsigned capacity.
//   - Stage 2: add only when the exact sum is representable.
//
// Inputs: a and b are already verified uint64 byte counts.
// Returns: exact sum or ErrOverflow; no allocation takes place.
// Determinism: pure integer arithmetic. Complexity: O(1) time and space.
// AI-Hints: a wrapped total could falsely appear below MaxSamplesBytes.
func checkedAdd(a, b uint64) (uint64, error) {
	if a > math.MaxUint64-b {
		return 0, ErrOverflow
	}
	return a + b, nil
}

// checkedBytes computes the addressable float64 backing-storage size in bytes.
// It checks both the element product and the native-int byte capacity.
// Implementation:
//   - Stage 1: reject an unrepresentable unsigned element product.
//   - Stage 2: bound the product by MaxInt divided by the float64 width.
//   - Stage 3: return the exact byte count without another fallible conversion.
//
// Behavior highlights:
//   - An int-representable element count alone is not sufficient for make.
//   - A zero-element buffer requires no backing storage.
//
// Inputs: a and b are nonnegative uint64 dimensions.
// Returns: exact native-addressable byte count, or ErrOverflow.
// Errors: ErrOverflow for an element or byte product outside its checked domain.
// Determinism: fixed unsigned quotient checks on the target architecture.
// Complexity: Time O(1), Space O(1), without allocating a slice.
// Notes: representable arithmetic does not guarantee available physical memory.
// AI-Hints: apply the byte check to every independently allocated float64 buffer,
// not only to the row count or the aggregate user budget.
func checkedBytes(a, b uint64) (uint64, error) {
	// The product may exceed uint64 even before accounting for element width.
	elements, err := checkedMul(a, b)
	if err != nil {
		return 0, err
	}
	// A byte size beyond native int is not an admitted backing allocation.
	if elements > uint64(math.MaxInt)/bytesPerFloat64 {
		return 0, ErrOverflow
	}
	// The preceding bound proves both this multiplication and native indexing safe.
	return elements * bytesPerFloat64, nil
}

// checkBudget validates every full-size buffer required by requested output modes.
// It accounts for one Paths*d raw owner, one Paths sorting scratch column and
// one q*d quantile output. No extra raw copy is allocated for retention.
// Implementation:
//   - Stage 1: check one feature vector even in streaming-only mode.
//   - Stage 2: check raw and optional quantile output sizes independently.
//   - Stage 3: sum the backing byte costs and compare with the resolved limit.
//
// Behavior highlights:
//   - Sorting scratch is one raw column, not a second sample matrix.
//   - Retention and quantiles share one raw owner.
//
// Inputs: c has validated modes and level order; dimension is its positive unsigned size.
// Returns: ErrOverflow for unaddressable or wrapped sizes, ErrBudget for
// allocation demand above MaxSamplesBytes; otherwise nil.
// Errors: integer representability precedes the configured byte ceiling.
// Determinism: the fixed raw, scratch, output summation order determines priority.
// Complexity: Time O(1), Space O(1) beyond earlier level validation.
// Notes: worker vectors, statistics objects and scheduler metadata are separate
// from MaxSamplesBytes. This checks arithmetic, not runtime allocator availability.
// AI-Hints: never silently approximate exact quantiles to bypass the budget;
// do not charge the raw owner twice when both modes are requested.
func checkBudget(c configuration, dimension uint64) error {
	// All modes allocate feature vectors, including the published summaries.
	if _, err := checkedBytes(dimension, 1); err != nil {
		return err
	}
	if !c.keepRaw {
		// Streaming-only mode never materializes a Paths-by-d array.
		return nil
	}

	// Validate the one raw owner independently of the caller's byte ceiling.
	rawBytes, err := checkedBytes(c.plan.Paths, dimension)
	if err != nil {
		return err
	}
	// Quantile levels are already validated and preserve the original order.
	levelCount, err := nonnegativeUint64(len(c.levels))
	if err != nil {
		return err
	}
	// Check the quantile output even when the raw owner itself fits.
	quantileBytes, err := checkedBytes(levelCount, dimension)
	if err != nil {
		return err
	}
	// One sorting column is exactly rawBytes/dimension. The positive dimension
	// and the admitted raw owner prove this derived byte size addressable.
	var sortBytes uint64
	if levelCount != 0 {
		sortBytes = rawBytes / dimension
	}
	// Each independent allocation is <= MaxInt, so the first sum fits uint64.
	// The third buffer may exceed the aggregate unsigned domain; check it once.
	requested, err := checkedAdd(rawBytes+sortBytes, quantileBytes)
	if err != nil {
		return err
	}
	// Zero selects the documented default, not an unlimited budget.
	limit := c.plan.MaxSamplesBytes
	if limit == 0 {
		limit = defaultSamplesBytes
	}
	if requested > limit {
		return ErrBudget
	}
	return nil
}
