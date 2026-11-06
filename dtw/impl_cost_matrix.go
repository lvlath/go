// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw adapts matrix.Matrix sequences into finite local-cost surfaces.
// This file implements squared L2 construction and detached flat input materialization.
package dtw

import (
	"errors"
	"fmt"
	"math"

	"github.com/lvlath/go/matrix"
)

// localSquaredL2Matrix constructs the complete row-to-row squared Euclidean cost surface.
// It subtracts coordinates before scaling and squaring, preventing catastrophic
// cancellation in the identity ||x||²+||y||²-2*x·y for large near-equal rows.
//
// Implementation:
//   - Stage 1: validate matrix nil-state, time axes, matching feature dimensions,
//     and finite source coordinates using the published matrix validators.
//   - Stage 2: for each row pair, read corresponding feature coordinates in order
//     and accumulate their squared differences in a shared exponent domain.
//   - Stage 3: reject an unrepresentable local cost and publish a fully populated
//     finite Dense local matrix, independent of the active DTW window.
//
// Behavior highlights:
//   - Equal finite coordinate vectors produce exactly zero cost, including 1e200.
//   - A genuinely unrepresentable squared distance reports ErrOverflow and ErrNaNInf.
//   - Subnormal squared distances follow binary64 rounding, without clamping.
//
// Inputs:
//   - x: read-only nx-by-d matrix, each row a multivariate time step.
//   - y: read-only ny-by-d matrix, matching feature dimension d.
//
// Returns:
//   - *matrix.Dense: detached nx-by-ny matrix of finite non-negative squared costs.
//   - error: sentinel-classified validation, access, allocation, or numeric failure.
//
// Errors:
//   - ErrBadInput: matrix shape, allocation, or access failure.
//   - matrix.ErrDimensionMismatch: unequal feature dimensions, joined with ErrBadInput.
//   - ErrEmptyInput: either time axis has zero rows.
//   - ErrNaNInf: invalid coordinate or numeric artifact.
//   - ErrOverflow joined with ErrNaNInf: a local squared cost exceeds float64 range.
//
// Determinism:
//   - Pair order is row-major (i, j); feature order is k ascending.
//   - Each pair uses a deterministic ordered float64 exponent accumulator.
//
// Complexity:
//   - Time O(nx*ny*d + nx*d + ny*d), Space O(nx*ny) for the output.
//   - No extra dot-product, transposed matrix, or squared-norm matrices are allocated.
//
// Notes:
//   - Inputs and their storage are never modified or aliased by the result.
//   - Concurrent mutation of x or y during this call is unsupported.
//
// AI-Hints:
//   - Never reconstruct this cost via x²+y²-2xy: high-magnitude cancellation is real.
//   - Generic matrix.Matrix implementations must obey the same coordinate cost law as Dense.
func localSquaredL2Matrix(x, y matrix.Matrix) (*matrix.Dense, error) {
	if err := matrix.ValidateNotNil(x); err != nil {
		return nil, errors.Join(ErrBadInput, err)
	}
	if err := matrix.ValidateNotNil(y); err != nil {
		return nil, errors.Join(ErrBadInput, err)
	}

	if x.Rows() == 0 || y.Rows() == 0 {
		return nil, ErrEmptyInput
	}
	if x.Cols() != y.Cols() {
		return nil, errors.Join(ErrBadInput, matrix.ErrDimensionMismatch)
	}
	if err := matrix.ValidateAllFinite(x); err != nil {
		return nil, errors.Join(ErrNaNInf, err)
	}
	if err := matrix.ValidateAllFinite(y); err != nil {
		return nil, errors.Join(ErrNaNInf, err)
	}

	cost, err := matrix.NewPreparedDense(x.Rows(), y.Rows())
	if err != nil {
		return nil, errors.Join(ErrBadInput, err)
	}

	for i := 0; i < x.Rows(); i++ {
		for j := 0; j < y.Rows(); j++ {
			var sum squaredSumOfDifferences
			for feature := 0; feature < x.Cols(); feature++ {
				left, readErr := x.At(i, feature)
				if readErr != nil {
					return nil, errors.Join(ErrBadInput, readErr)
				}
				right, readErr := y.At(j, feature)
				if readErr != nil {
					return nil, errors.Join(ErrBadInput, readErr)
				}
				sum.addDifference(left, right)
			}

			local := sum.value()
			if math.IsNaN(local) {
				return nil, fmt.Errorf("dtw: non-finite local squared L2 at (%d,%d): %w", i, j, ErrNaNInf)
			}
			if math.IsInf(local, 1) {
				return nil, fmt.Errorf("dtw: unrepresentable squared L2 at (%d,%d): %w", i, j, errors.Join(ErrOverflow, ErrNaNInf))
			}
			if err = cost.Set(i, j, local); err != nil {
				return nil, errors.Join(ErrNaNInf, err)
			}
		}
	}
	return cost, nil
}

// materializeLocalCostMatrix copies and validates a caller-owned local cost matrix.
// Its flat row-major representation keeps arbitrary Matrix.At dispatch and
// mutation concerns out of the DP hot loop.
//
// Implementation:
//   - Stage 1: validate nil, dimensions, and integer-safe cell count.
//   - Stage 2: allocate a flat slice and read every source cell in row-major order.
//   - Stage 3: validate and store each finite non-negative local cost.
//
// Behavior highlights:
//   - All cells are validated, including those excluded by a future window policy.
//   - Caller input remains untouched and detached from the resulting flat slice.
//
// Inputs:
//   - local: read-only n-by-m matrix of finite non-negative local costs.
//
// Returns:
//   - []float64: detached row-major local costs indexed by i*cols+j.
//   - int, int: validated row and column counts.
//   - error: sentinel-preserving validation or access failure.
//
// Errors:
//   - ErrBadInput: invalid nil-state, dimension product, or At error.
//   - ErrEmptyInput: zero rows or zero columns.
//   - ErrNaNInf / ErrNegativeCost: invalid caller local cost.
//
// Determinism:
//   - Strict ascending row then column validation, first failure wins.
//
// Complexity:
//   - Time O(rows*cols), Space O(rows*cols).
//
// Notes:
//   - Cannot guarantee a stable snapshot if the caller concurrently mutates input.
//
// AI-Hints:
//   - Do not defer validation of off-band cells until after recurrence execution.
//   - Check integer products before allocation or indexing.
func materializeLocalCostMatrix(local matrix.Matrix) ([]float64, int, int, error) {
	if err := matrix.ValidateNotNil(local); err != nil {
		return nil, 0, 0, errors.Join(ErrBadInput, err)
	}
	rows, cols := local.Rows(), local.Cols()
	if rows == 0 || cols == 0 {
		return nil, 0, 0, ErrEmptyInput
	}
	if rows < 0 || cols < 0 || rows > maxIntValue/cols {
		return nil, 0, 0, ErrBadInput
	}

	costs := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			value, err := local.At(i, j)
			if err != nil {
				return nil, 0, 0, errors.Join(ErrBadInput, err)
			}
			if err = validateLocalCost(value); err != nil {
				return nil, 0, 0, err
			}
			costs[i*cols+j] = value
		}
	}
	return costs, rows, cols, nil
}

// denseCopyOfMatrix creates a detached Dense copy of a matrix.Matrix.
//
// Implementation:
//   - Stage 1: validate non-nil matrix input.
//   - Stage 2: allocate matrix.Dense with default finite-value policy.
//   - Stage 3: copy values in row-major order using At and Set.
//
// Behavior highlights:
//   - The copy never aliases caller-owned matrix storage.
//   - The helper does not require the input to be *matrix.Dense.
//
// Inputs:
//   - m: source matrix.
//
// Returns:
//   - *matrix.Dense: detached dense copy.
//
// Errors:
//   - ErrBadInput joined with matrix validation, allocation, At, or Set errors.
//   - ErrNaNInf joined with Set errors when non-finite values violate Dense policy.
//
// Determinism:
//   - Copies cells in row-major order.
//
// Complexity:
//   - Time O(r*c), Space O(r*c).
//
// AI-Hints:
//   - Do not use type assertion as the only clone path; matrix implementations may differ.
func denseCopyOfMatrix(m matrix.Matrix) (*matrix.Dense, error) {
	if err := matrix.ValidateNotNil(m); err != nil {
		return nil, errors.Join(ErrBadInput, err)
	}

	out, err := matrix.NewPreparedDense(m.Rows(), m.Cols())
	if err != nil {
		return nil, errors.Join(ErrBadInput, err)
	}

	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < m.Cols(); j++ {
			value, err := m.At(i, j)
			if err != nil {
				return nil, errors.Join(ErrBadInput, err)
			}

			if err = out.Set(i, j, value); err != nil {
				return nil, errors.Join(ErrNaNInf, err)
			}
		}
	}

	return out, nil
}
