// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package bootstrap verifies policy domains and native-architecture result budgets.
// Validation precedes stream construction and any substantial allocation.
package bootstrap

import (
	"context"
	"fmt"
	"unsafe"
)

// validateContext enforces the first validation stage of public sampling calls.
// A missing context is an invalid option; a canceled one retains its native cause.
//
// Inputs:
//   - ctx: non-nil context.Context owned and controlled by the caller.
//
// Errors:
//   - ErrInvalidOption for nil; ctx.Err() for a canceled or expired context.
//
// Complexity:
//   - Time O(1), Space O(1).
func validateContext(ctx context.Context) error {
	if ctx == nil {
		return fmt.Errorf("bootstrap: missing context: %w", ErrInvalidOption)
	}
	return ctx.Err()
}

// validatePlan checks all public sampling dimensions and resource admission.
// Invalid values are rejected before any memory calculation or random draw.
//
// Inputs:
//   - rows: number of available original observations, at least 1.
//   - plan: immutable value-copy request; positive dimensions with BlockLength<=rows.
//
// Errors:
//   - ErrInvalidOption for an unknown Kind or an illegal dimension.
//   - ErrOverflow for native-int size overflow, before ErrBudget admission.
//   - ErrBudget for a representable allocation above MaxResultBytes.
//
// Determinism:
//   - Kind/positivity precedes block-vs-rows, then overflow, then budget.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// AI-Hints:
//   - Do not allow Circular or Stationary BlockLength>rows as an implicit mode shift.
func validatePlan(rows int, plan Plan) error {
	if plan.Kind > StationaryBlocks || rows <= 0 || plan.Length <= 0 ||
		plan.BlockLength <= 0 || plan.Replicates <= 0 {
		return fmt.Errorf("bootstrap: method and dimensions: %w", ErrInvalidOption)
	}
	if plan.BlockLength > rows {
		return fmt.Errorf("bootstrap: block length exceeds source rows: %w", ErrInvalidOption)
	}
	return validateBudget(plan.Length, plan.Replicates)
}

// validateBudget computes the exact logical matrix payload on the current architecture.
// It includes all row-header slots in the outer slice and every int in the row data.
//
// Inputs:
//   - length: positive output indices per replicate (validated by validatePlan).
//   - replicates: positive output rows (validated by validatePlan).
//
// Errors:
//   - ErrOverflow if any byte multiplication or addition overflows native int.
//   - ErrBudget if the representable byte count exceeds MaxResultBytes.
//
// Determinism:
//   - Overflow is diagnosed before budget, regardless of integer widths.
//
// Complexity:
//   - Time O(1), Space O(1), with no allocations.
//
// Notes:
//   - Go runtime allocation overhead, allocator metadata and RSS are excluded.
//   - A bounded logical allocation still depends on available system memory.
//
// AI-Hints:
//   - Never form replicates*length before checking both multiplicands.
//   - unsafe.Sizeof is used only for native value layout, not pointer manipulation.
func validateBudget(length, replicates int) error {
	maxInt := int(^uint(0) >> 1)
	intBytes := int(unsafe.Sizeof(int(0)))
	rowHeaderBytes := int(unsafe.Sizeof([]int(nil)))

	// A single row must fit, including its outer-slice descriptor.
	if length > maxInt/intBytes {
		return ErrOverflow
	}
	rowBytes := length * intBytes
	if rowBytes > maxInt-rowHeaderBytes {
		return ErrOverflow
	}

	// Calculate each replica's complete logical cost before multiplying by R.
	bytesPerReplicate := rowBytes + rowHeaderBytes
	if replicates > maxInt/bytesPerReplicate {
		return ErrOverflow
	}

	// Once representable, enforce the immutable output ceiling.
	if replicates*bytesPerReplicate > MaxResultBytes {
		return ErrBudget
	}
	return nil
}
