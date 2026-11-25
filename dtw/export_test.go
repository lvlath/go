// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw exposes narrow test-build adapters to the private matrix kernels.
// This file declares no tests and is excluded from normal builds. All assertions
// and fixtures belong to the external dtw_test package.
package dtw

import "github.com/lvlath/go/matrix"

// DenseCopyOfMatrixForTest invokes the private finite Dense copy boundary.
// It permits independent verification of the adapter's error and ownership laws.
//
// Implementation:
//   - Delegate directly to the existing copy routine without adapting its errors.
//
// Behavior highlights:
//   - No production symbol, callback, or mutable global is introduced.
//
// Inputs:
//   - source: borrowed Matrix, including invalid or failing backend fixtures.
//
// Returns:
//   - Detached Dense copy or nil and the original sentinel-classified error.
//
// Errors:
//   - The errors returned by denseCopyOfMatrix are preserved unchanged.
//
// Determinism:
//   - The underlying copy reads cells in row-major order.
//
// Complexity:
//   - Time O(rows*cols), Space O(rows*cols) for a valid source.
//
// Notes:
//   - Available only to the test build of dtw.
//
// AI-Hints:
//   - Use AlignCostMatrix for the public alignment contract, not this bridge.
func DenseCopyOfMatrixForTest(source matrix.Matrix) (*matrix.Dense, error) {
	return denseCopyOfMatrix(source) // Exercise the real implementation, not a duplicate.
}

// ComputeFromFlatLocalForTest invokes the flat-cost adapter with ordinary DTW options.
// It tests malformed buffers independently of the matrix facade's prior validation.
//
// Implementation:
//   - Stage 1: assemble options using the canonical functional-option entry point.
//   - Stage 2: delegate the supplied buffer and dimensions to the private kernel.
//   - Stage 3: return its result and error unchanged.
//
// Behavior highlights:
//   - No private configuration type or option-assembly result is exposed.
//
// Inputs:
//   - costs: borrowed row-major local costs; the caller must not mutate them during use.
//   - rows, cols: time-axis lengths; invalid products are allowed as test inputs.
//   - opts: public DTW functional options.
//
// Returns:
//   - The kernel result, without the facade-owned LocalCost artifact, and its error.
//
// Errors:
//   - Canonical option errors and all flat-kernel sentinels are preserved.
//
// Determinism:
//   - The kernel validates row-major input and uses its standard predecessor policy.
//
// Complexity:
//   - Time O(rows*cols), Space O(cols) plus requested path/accumulated storage.
//
// Notes:
//   - The test bridge does not create a new production flat-alignment API.
//
// AI-Hints:
//   - Pass malformed dimensions with tiny buffers to test rejection without huge allocations.
func ComputeFromFlatLocalForTest(costs []float64, rows, cols int, opts ...Option) (*Result, error) {
	// Preserve the same finalized option policy used by the production facades.
	cfg, err := applyOptions(opts...)
	if err != nil {
		return nil, err // Invalid options must not reach the kernel.
	}
	return computeFromFlatLocal(costs, rows, cols, cfg) // Exercise the private validation boundary.
}

// NewAccumulatedCostMatrixForTest invokes the private DP artifact initializer.
// It separates the allocation and empty-prefix contract from the alignment engine.
//
// Implementation:
//   - Delegate directly to the existing accumulated-cost constructor.
//
// Behavior highlights:
//   - The constructor's distance policy and sentinel identities are unchanged.
//
// Inputs:
//   - rows, cols: dimensions including the boundary row and column.
//
// Returns:
//   - Initialized Dense storage or nil and the original initialization error.
//
// Errors:
//   - ErrBadInput or ErrNaNInf with the original Matrix cause, when applicable.
//
// Determinism:
//   - Every cell starts at +Inf except the finite zero origin.
//
// Complexity:
//   - Time O(rows*cols), Space O(rows*cols) for valid dimensions.
//
// Notes:
//   - Only the test build exports this name.
//
// AI-Hints:
//   - Do not use this bridge to allocate giant matrices merely to trigger OOM.
func NewAccumulatedCostMatrixForTest(rows, cols int) (*matrix.Dense, error) {
	return newAccumulatedCostMatrix(rows, cols) // Preserve the actual boundary implementation.
}
