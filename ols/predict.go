// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ols evaluates frozen fitted regressions on arbitrary inference batches.
// This file isolates the row-level FMA dot product from batch-level validation.
package ols

import (
	"context"
	"math"

	"github.com/lvlath/go/matrix"
)

// rowPrediction evaluates one original-unit affine dot product in feature order.
// It reads model state and caller observations without allocating or mutating
// any shared workspace, so independent concurrent inference calls are safe.
//
// Implementation:
//   - Stage 1: seed the accumulator with the frozen fitted intercept.
//   - Stage 2: read each feature and accumulate x_j*beta_j with math.FMA.
//   - Stage 3: reject unrepresentable partial/final prediction values.
//
// Behavior highlights:
//   - math.FMA performs each multiply-add with one final rounding.
//   - Even standardized models predict using original-unit coefficients.
//
// Inputs: ctx active; x read-only with matching p; row validated by caller; m fitted.
// Returns: one finite predicted response in y units, or zero/error.
// Errors: ErrOverflow for nonfinite accumulated values; ctx.Err(), Matrix.At errors.
// Determinism: feature columns j=0..p-1 in their original order.
// Complexity: Time O(p), Space O(1), no model-owned mutable buffers.
// Notes: Upstream Predict prevalidates every cell before starting this computation.
// AI-Hints: Do not refit feature means or use returned Coefficients without ownership awareness.
func (m *Model) rowPrediction(ctx context.Context, x matrix.Matrix, row int) (float64, error) {
	prediction := m.intercept
	for col, slope := range m.beta {
		if col%contextCheckRowInterval == 0 {
			if err := ctx.Err(); err != nil {
				return 0, err
			}
		}
		feature, err := x.At(row, col)
		if err != nil {
			return 0, err
		}
		// One fused multiply-add avoids separately rounding the product.
		// The ordering is fixed; the model does not maintain a cached dot buffer.
		prediction = math.FMA(feature, slope, prediction)
		if !finite(prediction) {
			return 0, ErrOverflow
		}
	}
	return prediction, nil
}

// Predict calculates out-of-sample responses using a fully frozen fitted Model.
// It accepts zero-row batches with the trained feature width and never refits
// training scales, changes coefficients, or shares mutable output between calls.
//
// Implementation:
//   - Stage 1: check context, fitted receiver, matrix presence, and feature count.
//   - Stage 2: validate all input cells before allocating or publishing predictions.
//   - Stage 3: compute each row by rowPrediction and return a detached output.
//
// Behavior highlights:
//   - For x with shape 0-by-p, returns a non-nil zero-length slice and no error.
//   - An error at any point returns nil, not partially filled predictions.
//   - No hidden training, centering, or mutable model workspace is used.
//
// Inputs:
//   - ctx: active non-nil context, checked during both passes over rows.
//   - x: read-only n-by-p matrix with exactly the fitted feature width p.
//
// Returns: independent []float64 of length n in original response units, or nil/error.
// Errors: ErrInvalidOption for nil ctx; ErrNilInput for nil/untrained model or x;
//
//	ErrShape for mismatched features/shape overflow; ErrNonFinite for invalid cells;
//	ErrOverflow for unrepresentable predictions; ctx.Err(), Matrix.At errors.
//
// Determinism: two ascending row-major passes; each dot product scans columns ascending.
// Complexity: Time O(n*p), Space O(n) for the detached output.
// Notes: Concurrent calls are safe if callers do not concurrently mutate input matrices.
// AI-Hints: A new inference batch MUST NOT trigger stats.FitStandardizer or scale refitting.
func (m *Model) Predict(ctx context.Context, x matrix.Matrix) ([]float64, error) {
	if err := contextErr(ctx); err != nil {
		return nil, err
	}
	if m == nil || !m.fitted || nilMatrix(x) {
		return nil, ErrNilInput
	}
	n, p := x.Rows(), x.Cols()
	if !checkedShape(n, p) || p != m.p || !checkedShape(n, 1) {
		return nil, ErrShape
	}

	// All finite checks precede prediction arithmetic so invalid data cannot
	// produce an unrelated overflow classification in an earlier row.
	for row := 0; row < n; row++ {
		if row%contextCheckRowInterval == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		for col := 0; col < p; col++ {
			if col%contextCheckRowInterval == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			value, err := x.At(row, col)
			if err != nil {
				return nil, err
			}
			if !finite(value) {
				return nil, ErrNonFinite
			}
		}
	}

	// Allocate only after all validation succeeds; no prefix is returned on error.
	out := make([]float64, n)
	for row := 0; row < n; row++ {
		if row%contextCheckRowInterval == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		value, err := m.rowPrediction(ctx, x, row)
		if err != nil {
			return nil, err
		}
		out[row] = value
	}
	return out, nil
}
