// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw adapts flat, row-major local costs to the classified DTW recurrence.
// This file owns validation and index translation; no second recurrence is maintained.
package dtw

// computeFromFlatLocal validates precomputed local costs and delegates to the shared kernel.
// It preserves the matrix adapter's complete validation, including out-of-band
// cells that will not participate in the selected optimal path.
//
// Implementation:
//   - Stage 1: validate positive dimensions and safe row-major product length.
//   - Stage 2: validate every local cost in row-major order before any DP execution.
//   - Stage 3: pass a constant-time flat-index source to the shared recurrence.
//
// Behavior highlights:
//   - The flat input is read-only, and no CostFunc callback runs in this adapter.
//   - Errors from malformed off-band costs are not hidden by the window.
//
// Inputs:
//   - localCosts: row-major finite, non-negative local cost values, caller-owned.
//   - rows, cols: positive time-step counts, with rows*cols == len(localCosts).
//   - cfg: finalized DTW configuration, including memory and window policy.
//
// Returns:
//   - *Result: detached alignment result following shared DTW result semantics.
//   - error: sentinel-classified shape, numeric, overflow, or path failure.
//
// Errors:
//   - ErrEmptyInput: zero-sized local-cost domain.
//   - ErrBadInput: negative, overflowing, or inconsistent dimensions.
//   - ErrNaNInf / ErrNegativeCost: invalid local input cost.
//   - ErrOverflow joined with ErrNaNInf: unrepresentable terminal optimum.
//   - ErrNoPath: structurally unreachable terminal with path tracking.
//
// Determinism:
//   - Validation and subsequent DP use ascending row-major cell order.
//   - Path tie order is diagonal, up, left in the delegated recurrence.
//
// Complexity:
//   - Validation O(rows*cols) time; recurrence O(rows*cols) time.
//   - Rolling DP O(cols) space, plus explicitly requested artifacts/path storage.
//
// Notes:
//   - The matrix facade already materializes and owns the flat local costs.
//   - This adapter intentionally has no alternate mathematical implementation.
//
// AI-Hints:
//   - Do not skip validation of costs outside a narrow window.
//   - Do not silently interpret non-finite local input as absent topology.
func computeFromFlatLocal(localCosts []float64, rows, cols int, cfg options) (*Result, error) {
	if rows == 0 || cols == 0 {
		return nil, ErrEmptyInput
	}
	if rows < 0 || cols < 0 || rows > maxIntValue/cols || rows*cols != len(localCosts) {
		return nil, ErrBadInput
	}
	for _, cost := range localCosts {
		if err := validateLocalCost(cost); err != nil {
			return nil, err
		}
	}

	local := func(i, j int) (float64, error) {
		return localCosts[i*cols+j], nil
	}

	// The matrix facades already own the complete validated local surface.
	// They attach a detached Dense copy if requested; allocating a second
	// full local artifact inside the shared scalar kernel would be redundant.
	kernelConfig := cfg
	kernelConfig.returnLocalCost = false
	return computeFromCostSource(rows, cols, kernelConfig, local)
}
