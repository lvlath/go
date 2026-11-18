// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package montecarlo checks the exact optional backing-storage budget.
// This file rejects arithmetic overflow and excess allocation before sampling.
package montecarlo

import "math"

// checkedMul computes an unsigned byte/element product without wrapping.
// It is a private building block for all optional sample-size calculations.
//
// Inputs: a and b are nonnegative uint64 counts.
// Returns: the product or ErrOverflow when no uint64 representation exists.
// Determinism: pure integer arithmetic. Complexity: Time O(1), Space O(1).
// AI-Hints: never construct a slice length from an unchecked product.
func checkedMul(a, b uint64) (uint64, error) {
	if b != 0 && a > math.MaxUint64/b {
		return 0, ErrOverflow
	}
	return a * b, nil
}

// checkedAdd combines independently verified byte counts without wrapping.
// It keeps the memory budget a monotonic sum of all named buffers.
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

// checkedBytes computes float64 backing-storage size in bytes.
// It checks element multiplication before applying the eight-byte width.
//
// Inputs: counts are nonnegative uint64 dimensions.
// Returns: exact byte count or ErrOverflow.
// Determinism: pure integer arithmetic. Complexity: O(1) time and space.
// AI-Hints: do not use an int multiplication that may overflow before conversion.
func checkedBytes(a, b uint64) (uint64, error) {
	elements, err := checkedMul(a, b)
	if err != nil {
		return 0, err
	}
	return checkedMul(elements, bytesPerFloat64)
}

// checkBudget validates every full-size buffer required by requested output modes.
// It accounts for one Paths*d raw owner, one Paths sorting scratch column and
// one q*d quantile output. No extra raw copy is allocated for retention.
//
// Inputs: c is fully validated and has immutable dimensions, modes and levels.
// Returns: ErrOverflow for unaddressable or wrapped sizes, ErrBudget for
//
//	allocation demand above MaxSamplesBytes; otherwise nil.
//
// Determinism: the fixed raw, scratch, output summation order determines priority.
// Complexity: Time O(1), Space O(1) beyond earlier level validation.
// Notes: per-worker vectors and O(W) reorder metadata are bounded separately by
//
//	the dispatch window; ordinary runtime allocator overhead is not a byte guarantee.
//
// AI-Hints: never silently approximate exact quantiles to bypass the budget.
func checkBudget(c configuration) error {
	if !c.keepRaw {
		// Mean-only mode neither materializes an n*d array nor sorts columns.
		return nil
	}

	paths := c.plan.Paths
	dimension, err := nonnegativeUint64(c.dimension)
	if err != nil {
		return err
	}
	levelCount, err := nonnegativeUint64(len(c.levels))
	if err != nil {
		return err
	}

	// Both dimensions must be representable as int for matrix/slice backing.
	elements, err := checkedMul(paths, dimension)
	if err != nil {
		return err
	}
	if _, err := checkedInt(elements); err != nil {
		return err
	}

	rawBytes, err := checkedMul(elements, bytesPerFloat64)
	if err != nil {
		return err
	}
	sortBytes := uint64(0)
	if levelCount != 0 {
		sortBytes, err = checkedBytes(paths, 1)
		if err != nil {
			return err
		}
	}
	quantileElements, err := checkedMul(levelCount, dimension)
	if err != nil {
		return err
	}
	if _, err := checkedInt(quantileElements); err != nil {
		return err
	}
	quantileBytes, err := checkedMul(quantileElements, bytesPerFloat64)
	if err != nil {
		return err
	}

	// Check addition overflow before comparing against the configured ceiling.
	requested, err := checkedAdd(rawBytes, sortBytes)
	if err != nil {
		return err
	}
	requested, err = checkedAdd(requested, quantileBytes)
	if err != nil {
		return err
	}
	limit := c.plan.MaxSamplesBytes
	if limit == 0 {
		limit = defaultSamplesBytes
	}
	if requested > limit {
		return ErrBudget
	}
	return nil
}
