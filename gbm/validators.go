// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gbm centralizes finite-domain, unit, ordering, and nil checks.
// Validation is side-effect free and runs before any path state is allocated.
package gbm

import (
	"context"
	"math"

	"github.com/lvlath/go/rng"
)

// finite reports whether a float64 belongs to the finite real input domain.
// This predicate excludes both signs of infinity and every NaN payload.
//
// Inputs: any float64 value; Returns: true exactly for finite values.
// Determinism: pure IEEE-754 classification; Complexity: O(1) time and space.
// AI-Hints: a finite input can still cause ErrOverflow during arithmetic.
func finite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

// validateInputs enforces the canonical priority of GBM preflight failures.
// It rejects invalid state before drawing entropy or allocating trajectory data.
//
// Implementation:
//   - Stage 1: check context and exact physical-unit compatibility.
//   - Stage 2: identify absent streams or grids, then validate grid order.
//   - Stage 3: reject nonfinite coefficients and invalid physical domains.
//
// Behavior highlights:
//   - An optional IsNil implementation detects recognized typed-nil streams.
//   - No reflection or speculative method calls are performed on a foreign RNG.
//   - Nonfinite timestamps are classified before comparing their ordering.
//
// Inputs:
//   - ctx: non-nil context, checked for cancellation before inspecting data.
//   - p: parameters with finite Initial>0, finite Drift, finite sigma>=0.
//   - grid: nonnil, nonempty, finite timestamps starting at zero, strictly rising.
//   - random: non-nil RNG; recognized optional nilable pointers are rejected.
//
// Returns: nil on success, or the earliest applicable sentinel/context error.
// Errors: ErrInvalidOption, ErrNilInput, ErrEmptyInput, ErrOrder,
//
//	ErrNonFinite, ErrDomain, and the unchanged ctx.Err().
//
// Determinism: check order is fixed, with timestamps scanned at ascending index.
// Complexity: Time O(n) for n timestamps, Space O(1), no allocations.
// Notes: a foreign typed-nil RNG lacking IsNil cannot be identified reflect-free.
// AI-Hints: never probe a foreign pointer's representation via unsafe or reflect.
func validateInputs(ctx context.Context, p Parameters, grid TimeGrid, random rng.RNG) error {
	// A nil context has no Err method. A canceled context wins over every
	// other invalid input so no user data or RNG implementation is touched.
	if ctx == nil {
		return ErrInvalidOption
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	// Enum values have an intentionally closed domain; equality alone is not
	// enough because two identical unsupported enums must still be rejected.
	if p.Unit > TimeSeconds || grid.Unit > TimeSeconds || p.Unit != grid.Unit {
		return ErrInvalidOption
	}

	// Interface equality catches an untyped nil. IsNil extends validation to
	// cooperative typed-nil RNGs without violating the reflect-free protocol.
	if random == nil {
		return ErrNilInput
	}
	if nilable, ok := random.(interface{ IsNil() bool }); ok && nilable.IsNil() {
		return ErrNilInput
	}
	if grid.Values == nil {
		return ErrNilInput
	}
	if len(grid.Values) == 0 {
		return ErrEmptyInput
	}

	// NaN does not have a meaningful ordering relation, so classify its
	// numeric invalidity explicitly rather than letting != 0 imply ErrOrder.
	if !finite(grid.Values[0]) {
		return ErrNonFinite
	}
	if grid.Values[0] != 0 {
		return ErrOrder
	}
	for index := 1; index < len(grid.Values); index++ {
		if !finite(grid.Values[index]) {
			return ErrNonFinite
		}
		if grid.Values[index] <= grid.Values[index-1] {
			return ErrOrder
		}
	}

	// Input-domain errors must not be conflated with a later arithmetic
	// overflow: every coefficient is valid in isolation before a step begins.
	if !finite(p.Initial) || !finite(p.Drift) || !finite(p.Volatility) {
		return ErrNonFinite
	}
	if p.Initial <= 0 || p.Volatility < 0 {
		return ErrDomain
	}
	return nil
}
