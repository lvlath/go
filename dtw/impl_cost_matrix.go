// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw adapts matrix.Matrix inputs into local-cost matrices for DTW.
// This file keeps matrix-backed cost construction separate from the DP recurrence.
package dtw

import (
	"errors"
	"fmt"
	"math"

	"github.com/lvlath/go/matrix"
)

const (
	// squaredL2RoundoffTolerance clamps tiny negative squared distances caused by floating-point roundoff.
	squaredL2RoundoffTolerance = 1e-12
)

// localSquaredL2Matrix computes pairwise squared L2 row distances.
//
// Implementation:
//   - Stage 1: validate non-nil matrix inputs.
//   - Stage 2: validate equal feature dimensions and non-empty time axes.
//   - Stage 3: reject NaN and ±Inf through matrix.ValidateAllFinite.
//   - Stage 4: compute dot products using matrix.Mul(x, Transpose(y)).
//   - Stage 5: compute row squared norms with Hadamard and RowSums.
//   - Stage 6: assemble a dense local-cost matrix in row-major order.
//
// Behavior highlights:
//   - Rows are time steps.
//   - Columns are features.
//   - The returned cost is squared L2, not sqrt L2.
//   - Tiny negative values caused by roundoff are clamped to zero.
//
// Inputs:
//   - x: matrix with rows as time steps and columns as feature dimensions.
//   - y: matrix with rows as time steps and columns as feature dimensions.
//
// Returns:
//   - *matrix.Dense: local-cost matrix C where C[i,j]=||x_i-y_j||².
//
// Errors:
//   - ErrBadInput joined with matrix nil/shape/operation/access errors.
//   - matrix.ErrDimensionMismatch is preserved when x.Cols()!=y.Cols().
//   - ErrEmptyInput when x or y has zero rows.
//   - ErrNaNInf joined with matrix finite-validation failures.
//   - ErrBadInput when roundoff exceeds tolerance and produces a negative cost.
//
// Determinism:
//   - Uses deterministic matrix operations and row-major assembly.
//
// Complexity:
//   - Time O(nx*ny*d) conceptually, where d=x.Cols()=y.Cols().
//   - Space O(nx*ny + nx*d + ny*d) depending on matrix operation internals.
//
// Notes:
//   - Caller-side normalization should happen before this function.
//   - This function does not mutate x or y.
//
// AI-Hints:
//   - Do not silently normalize or clip inputs here.
//   - Do not take sqrt unless the API explicitly promises Euclidean L2 instead of squared L2.
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

	yT, err := matrix.Transpose(y)
	if err != nil {
		return nil, errors.Join(ErrBadInput, err)
	}

	dot, err := matrix.Mul(x, yT)
	if err != nil {
		return nil, errors.Join(ErrBadInput, err)
	}

	x2, err := matrix.Hadamard(x, x)
	if err != nil {
		return nil, errors.Join(ErrBadInput, err)
	}

	y2, err := matrix.Hadamard(y, y)
	if err != nil {
		return nil, errors.Join(ErrBadInput, err)
	}

	xNorms, err := matrix.RowSums(x2)
	if err != nil {
		return nil, errors.Join(ErrBadInput, err)
	}

	yNorms, err := matrix.RowSums(y2)
	if err != nil {
		return nil, errors.Join(ErrBadInput, err)
	}

	cost, err := matrix.NewPreparedDense(x.Rows(), y.Rows())
	if err != nil {
		return nil, errors.Join(ErrBadInput, err)
	}

	for i := 0; i < x.Rows(); i++ {
		for j := 0; j < y.Rows(); j++ {
			dotValue, err := dot.At(i, j)
			if err != nil {
				return nil, errors.Join(ErrBadInput, err)
			}

			localCost := xNorms[i] + yNorms[j] - 2*dotValue
			if localCost < 0 && math.Abs(localCost) <= squaredL2RoundoffTolerance {
				localCost = 0
			}

			if localCost < 0 || math.IsNaN(localCost) || math.IsInf(localCost, 0) {
				return nil, fmt.Errorf("dtw: local squared L2 cost[%d,%d]=%v: %w", i, j, localCost, ErrBadInput)
			}

			if err = cost.Set(i, j, localCost); err != nil {
				return nil, errors.Join(ErrNaNInf, err)
			}
		}
	}

	return cost, nil
}

// materializeLocalCostMatrix validates and flattens a local-cost matrix.
//
// Implementation:
//   - Stage 1: validate matrix nil-state.
//   - Stage 2: reject zero-row or zero-column DTW domains.
//   - Stage 3: read every cell in row-major order.
//   - Stage 4: validate each local cost and store it into a flat slice.
//
// Behavior highlights:
//   - The flat layout is row-major: cost[i*cols+j].
//   - The DP hot loop reads the flat slice without interface dispatch.
//   - The source matrix is never mutated.
//
// Inputs:
//   - local: matrix of finite non-negative local alignment costs.
//
// Returns:
//   - []float64: row-major local costs.
//   - rows: number of time steps in sequence A.
//   - cols: number of time steps in sequence B.
//   - error: sentinel-classified failure.
//
// Errors:
//   - ErrBadInput joined with matrix validation/access errors.
//   - ErrEmptyInput when rows==0 or cols==0.
//   - ErrNaNInf when a cost is NaN or ±Inf.
//   - ErrNegativeCost when a cost is negative.
//
// Determinism:
//   - Reads cells in row-major order and reports the first invalid cell.
//
// Complexity:
//   - Time O(rows*cols), Space O(rows*cols).
//
// AI-Hints:
//   - Do not run DP directly through Matrix.At in the hot loop unless benchmarking proves it acceptable.
//   - Do not allow +Inf local costs; +Inf is reserved for unreachable accumulated states.
func materializeLocalCostMatrix(local matrix.Matrix) ([]float64, int, int, error) {
	if err := matrix.ValidateNotNil(local); err != nil {
		return nil, 0, 0, errors.Join(ErrBadInput, err)
	}

	rows := local.Rows()
	cols := local.Cols()

	if rows == 0 || cols == 0 {
		return nil, 0, 0, ErrEmptyInput
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
