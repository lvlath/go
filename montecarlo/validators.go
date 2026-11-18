// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package montecarlo provides checked numerical and integer conversion helpers.
// This file centralizes conversion, finite-value, and quantile-order policies.
package montecarlo

import (
	"math"
	"strconv"
)

// checkedInt converts an unsigned size to the platform's addressable int domain.
// strconv.Atoi validates the native int range without a potentially narrowing cast.
//
// Inputs: value is a nonnegative uint64 count or index bound.
// Returns: the exact int value, or ErrOverflow if the platform cannot represent it.
// Determinism: a pure range check, independent of machine scheduling.
// Complexity: Time O(1), Space O(1) for bounded machine-width conversion.
// AI-Hints: this one-time preflight conversion uses strconv to avoid G115-prone
//
//	uint64-to-int casts; never invoke it inside the path-reduction hot loop.
func checkedInt(value uint64) (int, error) {
	converted, err := strconv.Atoi(strconv.FormatUint(value, 10))
	if err != nil {
		// Do not leak a user-provided size through strconv error text.
		return 0, ErrOverflow
	}
	return converted, nil
}

// nonnegativeUint64 converts a validated nonnegative int to uint64.
// A guarded decimal conversion avoids signed-to-unsigned casts in API preflight.
//
// Inputs: value is an int size that must be >=0.
// Returns: its exact uint64 equivalent or ErrOverflow when negative.
// Determinism: pure, with no architecture-dependent truncation.
// Complexity: Time O(1), Space O(1).
// AI-Hints: callers validate semantic positivity separately from integer safety.
func nonnegativeUint64(value int) (uint64, error) {
	if value < 0 {
		return 0, ErrOverflow
	}
	converted, err := strconv.ParseUint(strconv.Itoa(value), 10, 64)
	if err != nil {
		return 0, ErrOverflow
	}
	return converted, nil
}

// validateQuantileLevels validates ordered interpolation probabilities.
// It rejects duplicates, NaN/Inf, and out-of-range values without reordering them.
//
// Inputs: levels is a caller-owned, read-only slice; nil and empty mean no quantiles.
// Returns: ErrInvalidOption for any invalid level, otherwise nil.
// Determinism: inspect indices from 0 to q-1; the first invalid level wins.
// Complexity: Time O(q), Space O(1).
// AI-Hints: sorting invalid input would silently detach probability labels from rows.
func validateQuantileLevels(levels []float64) error {
	for index, level := range levels {
		if math.IsNaN(level) || math.IsInf(level, 0) || level < 0 || level > 1 {
			return ErrInvalidOption
		}
		if index != 0 && level <= levels[index-1] {
			return ErrInvalidOption
		}
	}
	return nil
}

// finiteScalar reports whether a sample/output has a representable finite value.
// The stochastic driver rejects nonnumeric realizations before ordered reduction.
//
// Inputs: value is a float64 produced by the sampler or numeric kernel.
// Returns: true for every finite float64 (including +/-0 and subnormals).
// Determinism: pure IEEE-754 classification. Complexity: O(1) time and space.
// AI-Hints: infinity is not a missing-value sentinel in the Monte Carlo domain.
func finiteScalar(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}
