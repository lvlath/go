// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package stats performs immutable fitted affine inference on matrix inputs.
// This file contains only inference-time transformations, never training code.
package stats

import (
	"math"

	"github.com/lvlath/go/matrix"
)

// Transform standardizes an inference matrix using fixed fitted parameters.
// Each output cell is (input-trainingMean)/trainingScale; no heldout values
// contribute to parameter estimation or change the stored model.
// Implementation:
//   - Stage 1: validate model, matrix nil, dimensions, and all input finiteness.
//   - Stage 2: allocate an independent matrix of identical shape.
//   - Stage 3: evaluate stable affine divisions in row-major order.
//
// Behavior highlights:
//   - Uses extended arithmetic when subtracting two finite numbers would overflow.
//   - Accepts zero-row matrices with exactly the fitted feature count.
//
// Inputs:
//   - x: read-only matrix, rows>=0 and Cols()==fitted feature count.
//
// Returns: detached matrix.Matrix (concrete *matrix.Dense) of standardized values.
// Errors:
//   - ErrNilInput: nil/unfitted model or nil input.
//   - ErrShape: feature count mismatch or invalid matrix dimensions.
//   - ErrNonFinite: an input cell is NaN or ±Inf.
//   - ErrOverflow: a final standardized value cannot fit float64.
//   - Wrapped matrix.At or constructor error with original sentinel identity.
//
// Determinism: validates and transforms row-major, with fixed per-column parameters.
// Complexity: Time O(rows*cols), Space O(rows*cols) for detached output.
// Notes: concurrent calls are safe if their input matrices are not mutated concurrently.
// AI-Hints: never refit test data or modify fitted Means/Scales during Transform.
func (m *Standardizer) Transform(x matrix.Matrix) (matrix.Matrix, error) {
	return m.apply(x, false)
}

// InverseTransform restores the training-coordinate representation of a matrix.
// Each result is z*trainingScale+trainingMean; FMA evaluates this as one fused
// operation so a huge intermediate product cannot overflow before cancellation.
// Implementation:
//   - Stage 1: validate model, shape, and finite normalized input.
//   - Stage 2: allocate a caller-owned matrix of the same dimensions.
//   - Stage 3: use IEEE fused multiply-add for each reconstructed cell.
//
// Behavior highlights:
//   - Uses the same immutable training parameters as Transform.
//   - Supports a zero-row inference matrix without special output aliasing.
//
// Inputs:
//   - z: read-only normalized matrix with Cols()==fitted features.
//
// Returns: new matrix.Matrix (concrete *matrix.Dense) of reconstructed values.
// Errors:
//   - ErrNilInput: nil/unfitted model or nil input.
//   - ErrShape: invalid or incompatible matrix dimensions.
//   - ErrNonFinite: nonfinite input cell.
//   - ErrOverflow: the final fused result is not finite.
//   - Wrapped underlying matrix errors are preserved.
//
// Determinism: fixed row-major input/output order.
// Complexity: Time O(rows*cols), Space O(rows*cols) for detached output.
// Notes: rounding may prevent bitwise round-trip equality for arbitrary samples.
// AI-Hints: do not replace FMA with z*scale+mean; the product can overflow early.
func (m *Standardizer) InverseTransform(z matrix.Matrix) (matrix.Matrix, error) {
	return m.apply(z, true)
}

// apply is the single validated matrix transformation kernel for both directions.
// It intentionally scans the whole input for non-finite values before doing any
// potentially overflowing calculation, preserving the documented precedence.
// Implementation:
//   - Stage 1: validate the immutable model and matrix shape.
//   - Stage 2: validate every input value, then allocate a detached result.
//   - Stage 3: compute and store transformed cells, returning nil on failure.
//
// Inputs:
//   - input: an unchanged matrix for the duration of the method call.
//   - inverse: false for standardization; true for inverse affine conversion.
//
// Returns: caller-owned matrix.Matrix or nil if any cell fails.
// Errors: ErrNilInput, ErrShape, ErrNonFinite, ErrOverflow, or wrapped matrix errors.
// Determinism: validation pass and computation pass are both row-major.
// Complexity: Time O(rows*cols), Space O(rows*cols); O(1) extra scalar workspace.
// AI-Hints: do not cache intermediate matrices in Standardizer: that creates races.
func (m *Standardizer) apply(input matrix.Matrix, inverse bool) (matrix.Matrix, error) {
	// Reject a nil or otherwise unfitted model before inspecting input values.
	if m == nil || len(m.scales) == 0 {
		return nil, ErrNilInput
	}
	rows, cols, err := matrixShape(input)
	if err != nil {
		return nil, err
	}
	if cols != len(m.scales) {
		return nil, ErrShape
	}

	// An entire prevalidation pass ensures a late NaN is reported instead of
	// an earlier arithmetic overflow; no output has yet been published.
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			value, readErr := matrixCell(input, row, col)
			if readErr != nil {
				return nil, readErr
			}
			if !finite(value) {
				return nil, ErrNonFinite
			}
		}
	}

	// matrix.NewDense explicitly supports a zero-row matrix and owns its
	// storage; the model never hands its parameter slices to the result.
	out, err := matrix.NewDense(rows, cols)
	if err != nil {
		return nil, err
	}
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			value, readErr := matrixCell(input, row, col)
			if readErr != nil {
				return nil, readErr
			}

			var result float64
			if inverse {
				// One fused operation rounds AFTER multiplication and addition.
				result = math.FMA(value, m.scales[col], m.means[col])
			} else {
				// The direct path is the fast common case; the stable path
				// handles finite operands whose subtraction becomes infinity.
				result, err = stableStandardize(value, m.means[col], m.scales[col])
				if err != nil {
					return nil, err
				}
			}

			if !finite(result) {
				return nil, ErrOverflow
			}
			if err = out.Set(row, col, result); err != nil {
				return nil, err
			}
		}
	}
	return out, nil
}

// stableStandardize divides a centered observation without premature overflow.
// Ordinary finite differences use fast float64 division; when x-mean overflows,
// scaled compensated subtraction precedes division by the fitted positive scale.
// Implementation:
//   - Stage 1: calculate the direct delta and check its finiteness.
//   - Stage 2: divide finite deltas or compute an extended-range ratio.
//   - Stage 3: classify any unrepresentable final result as ErrOverflow.
//
// Inputs: value and mean are finite; scale is finite and strictly positive.
// Returns: (value-mean)/scale rounded to finite float64.
// Errors: ErrOverflow when the final quotient cannot be represented.
// Determinism: direct arithmetic, then a fixed extended fallback if necessary.
// Complexity: O(1) time and space; no allocation in either path.
// AI-Hints: do not rewrite the fallback as value/scale-mean/scale: both
// quotients might overflow even when their difference is representable.
func stableStandardize(value, mean, scale float64) (float64, error) {
	difference := value - mean
	if finite(difference) {
		result := difference / scale
		if !finite(result) {
			return 0, ErrOverflow
		}
		return result, nil
	}

	// The extended path preserves a large finite ratio without ever
	// materializing the unrepresentable raw difference.
	ratio := scaledFromFloat(value).sub(scaledFromFloat(mean)).div(scaledFromFloat(scale))
	return ratio.toFloat()
}
