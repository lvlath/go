// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package stats centralizes finite-number, dimension, and matrix-nil guards.
// This file delegates interface nil recognition to matrix's core.Nilable policy.
package stats

import (
	"fmt"
	"math"

	"github.com/lvlath/go/matrix"
)

// finite reports whether a scalar is neither NaN nor positive/negative infinity.
// This predicate applies to all observed and inferred matrix values; no option
// can weaken the finite-only numerical domain of stats.
// Implementation:
//   - Stage 1: classify NaN using IEEE-754 math.IsNaN.
//   - Stage 2: exclude both signs of infinity.
//
// Inputs: value is any float64, including non-finite values.
// Returns: true exactly for finite float64 numbers, including signed zero.
// Errors: none.
// Determinism: pure IEEE-754 comparisons.
// Complexity: O(1) time and space.
// AI-Hints: do not use value==value alone; that rejects NaN but accepts Inf.
func finite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

// validDDOF validates the two supported variance denominator conventions.
// Population variance divides M2 by Count; sample variance divides M2 by
// Count-1, and no negative or larger degrees of freedom are supported.
// Implementation: compare ddof against both explicit allowed constants.
// Inputs: ddof is the requested nonnegative integer degrees of freedom.
// Returns: true exactly for ddof 0 or 1.
// Errors: none; the caller maps false to ErrInvalidOption.
// Determinism: pure integer comparisons.
// Complexity: O(1) time and space.
// AI-Hints: do not clamp ddof to the available sample count.
func validDDOF(ddof int) bool {
	return ddof == 0 || ddof == 1
}

// validateScaleOptions rejects invalid fitting configurations before reading data.
// This is the only configuration gate for FitStandardizer and its API facades,
// so convenience wrappers cannot accidentally weaken the option contract.
// Implementation:
//   - Stage 1: verify DDOF is population or sample convention.
//   - Stage 2: validate the complete ConstantPolicy enum, including zero value.
//
// Inputs: opts is a by-value copy; it is never modified.
// Returns: nil for a supported exact configuration.
// Errors: ErrInvalidOption for unsupported DDOF or ConstantPolicy.
// Determinism: fixed DDOF check before policy check.
// Complexity: O(1) time and space.
// AI-Hints: remember that ScaleOptions{} rejects constant features by default.
func validateScaleOptions(opts ScaleOptions) error {
	if !validDDOF(opts.DDOF) {
		return ErrInvalidOption
	}
	if opts.ConstantPolicy != RejectConstant && opts.ConstantPolicy != UnitScaleConstant {
		return ErrInvalidOption
	}
	return nil
}

// matrixShape checks matrix presence and allocation-safe dimensions.
// Presence is delegated to matrix.ValidateNotNil, which recognizes typed-nil
// implementations conforming to the established core.Nilable convention.
// Implementation:
//   - Stage 1: reject nil or supported typed-nil without using reflection.
//   - Stage 2: read Rows and Cols only after the presence check.
//   - Stage 3: reject negative dimensions and rows*cols integer overflow.
//
// Inputs: input is an immutable-for-the-call matrix.Matrix reference.
// Returns: nonnegative (rows, cols) suitable for shape-safe allocation.
// Errors:
//   - ErrNilInput: nil matrix, joined with matrix's underlying nil sentinel.
//   - ErrShape: malformed dimensions or an unrepresentable element count.
//
// Determinism: one presence check followed by Rows then Cols.
// Complexity: O(1) time and space.
// Notes:
//   - Custom pointer Matrix implementations must implement IsNil for typed-nil safety.
//   - This does not verify contents or guarantee available system memory.
//
// AI-Hints:
//   - Do not introduce reflect: matrix.ValidateNotNil is the canonical adapter.
//   - Validate count multiplication before calling matrix.NewDense.
func matrixShape(input matrix.Matrix) (int, int, error) {
	if err := matrix.ValidateNotNil(input); err != nil {
		return 0, 0, fmt.Errorf("stats: matrix presence: %w: %w", ErrNilInput, err)
	}

	rows, cols := input.Rows(), input.Cols()
	maxInt := int(^uint(0) >> 1)
	if rows < 0 || cols < 0 || (cols != 0 && rows > maxInt/cols) {
		return 0, 0, ErrShape
	}
	return rows, cols, nil
}

// matrixCell reads an element without losing the matrix's original error identity.
// Stats does not reclassify arbitrary At failures as shape errors: a custom
// Matrix can return its own meaningful error, which callers may inspect.
// Implementation:
//   - Stage 1: read exactly the requested row and column.
//   - Stage 2: return the scalar, or wrap the underlying error with indices.
//
// Inputs: input is a valid Matrix; row/col are in validated shape range.
// Returns: the reported scalar value when At succeeds.
// Errors: wrapped underlying matrix.At error, preserving errors.Is.
// Determinism: exactly one At call at the given coordinate.
// Complexity: O(1) plus the Matrix.At implementation cost.
// AI-Hints: do not ignore a custom Matrix.At error or silently substitute zero.
func matrixCell(input matrix.Matrix, row, col int) (float64, error) {
	value, err := input.At(row, col)
	if err != nil {
		return 0, fmt.Errorf("stats: matrix.At(%d,%d): %w", row, col, err)
	}
	return value, nil
}
