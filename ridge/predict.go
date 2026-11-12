// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ridge performs deterministic physical-unit inference from immutable models.
// A rare exact-arithmetic path rescues representable results after intermediate overflow.
package ridge

import (
	"context"
	"math"
	"math/big"

	"github.com/lvlath/go/matrix"
)

// exactRowPrediction evaluates a dot product as exact binary64 rationals.
// It is used only after an intermediate floating-point overflow in the fast path.
//
// Implementation:
//   - Stage 1: initialize an exact rational accumulator from the fitted intercept.
//   - Stage 2: multiply each binary64 feature and slope exactly in column order.
//   - Stage 3: round the exact result once to float64 and check representability.
//
// Behavior highlights:
//   - Opposing unrepresentable products may cancel into a finite final prediction.
//   - The fallback changes only arithmetic precision, never the fitted model or loss.
//
// Inputs:
//   - ctx: active context; x: read-only finite n-by-p Matrix; row: valid row index.
//   - m: private fitted model whose coefficients are finite and immutable.
//
// Returns:
//   - float64: one correctly rounded exact dot product, or zero/error.
//
// Errors:
//   - ErrOverflow when the final exact prediction is outside finite float64.
//   - ctx.Err() or a Matrix.At error unchanged.
//
// Determinism:
//   - Feature columns ascend; big.Rat operations are exact for binary64 inputs.
//
// Complexity:
//   - Time O(p*B^2) in a conservative big-integer bit-cost model, Space O(p*B),
//     where B covers binary exponents, coefficient precision, and accumulated bits.
//
// Notes:
//   - The ordinary finite path does not allocate big numbers.
//   - Input is reread only when the fast path's arithmetic would overflow.
//
// AI-Hints:
//   - Do not use a float64 Inf as a valid prediction or as a missing-value sentinel.
//   - Do not silently skip the feature responsible for intermediate overflow.
func (m *Model) exactRowPrediction(ctx context.Context, x matrix.Matrix, row int) (float64, error) {
	var sum, term, featureRat, slopeRat big.Rat
	sum.SetFloat64(m.intercept)
	for j, slope := range m.beta {
		if j%contextCheckInterval == 0 {
			if err := ctx.Err(); err != nil {
				return 0, err
			}
		}
		feature, err := x.At(row, j)
		if err != nil {
			return 0, err
		}
		if !finite(feature) {
			return 0, ErrNonFinite
		}
		featureRat.SetFloat64(feature)
		slopeRat.SetFloat64(slope)
		term.Mul(&featureRat, &slopeRat)
		sum.Add(&sum, &term)
	}
	prediction, _ := sum.Float64()
	if !finite(prediction) {
		return 0, ErrOverflow
	}
	return prediction, nil
}

// rowPrediction calculates one physical-unit response with compensated summation.
// Intermediate overflow is retried in exact rational arithmetic if it may cancel.
//
// Implementation:
//   - Stage 1: seed a Neumaier accumulator with the fitted original-unit intercept.
//   - Stage 2: multiply and compensate sum/product rounding in feature-column order.
//   - Stage 3: on intermediate overflow use exactRowPrediction; otherwise finish finite.
//
// Behavior highlights:
//   - The common path is allocation-free and uses fused multiply-add for product error.
//   - The fallback does not change regularization or reapply the training transform.
//
// Inputs:
//   - ctx: active context; x: read-only finite matrix; row: valid index.
//   - m: private fitted model with physical-unit coefficients.
//
// Returns:
//   - float64: finite predicted response, or zero/error.
//
// Errors:
//   - ErrNonFinite for a changed/invalid input cell, ErrOverflow for final range loss.
//   - ctx.Err() or Matrix.At error from either execution path.
//
// Determinism:
//   - Original feature indices ascend; the exact fallback uses the same order.
//
// Complexity:
//   - Time O(p), Space O(1) on the finite path; see exactRowPrediction on fallback.
//
// Notes:
//   - Input mutation during either pass is unsupported; no Model state is modified.
//
// AI-Hints:
//   - Do not evaluate standardized coordinates during prediction: beta and b are physical.
//   - Never replace the compensation term with a hidden mutable Model field.
func (m *Model) rowPrediction(ctx context.Context, x matrix.Matrix, row int) (float64, error) {
	sum, correction := m.intercept, 0.0
	for j, slope := range m.beta {
		if j%contextCheckInterval == 0 {
			if err := ctx.Err(); err != nil {
				return 0, err
			}
		}
		feature, err := x.At(row, j)
		if err != nil {
			return 0, err
		}
		if !finite(feature) {
			return 0, ErrNonFinite
		}
		term := feature * slope
		if !finite(term) {
			return m.exactRowPrediction(ctx, x, row)
		}
		next := sum + term
		if !finite(next) {
			return m.exactRowPrediction(ctx, x, row)
		}
		// Neumaier compensates the rounded addition; FMA contributes the
		// residual of the independently rounded multiplication.
		if math.Abs(sum) >= math.Abs(term) {
			correction += (sum - next) + term
		} else {
			correction += (term - next) + sum
		}
		correction += math.FMA(feature, slope, -term)
		if !finite(correction) {
			return m.exactRowPrediction(ctx, x, row)
		}
		sum = next
	}
	result := sum + correction
	if !finite(result) {
		return m.exactRowPrediction(ctx, x, row)
	}
	return result, nil
}

// Predict applies a fitted physical-unit ridge model to an inference matrix.
// It never refits scaling statistics, mutates its Model, or returns partial output.
//
// Implementation:
//   - Stage 1: validate context, fitted receiver, matrix presence, and dimensions.
//   - Stage 2: check every input cell for finiteness before any row arithmetic.
//   - Stage 3: predict rows using a per-call output slice and publish only on success.
//
// Behavior highlights:
//   - A valid 0-by-p matrix returns an independently allocated non-nil empty slice.
//   - Nonfinite observations have precedence over row prediction overflow.
//   - Ordinary predictions are compensated; an intermediate overflow can be rescued.
//
// Inputs:
//   - ctx: non-nil active context; m: fitted immutable Model receiver.
//   - x: caller-owned read-only m-by-p Matrix with p equal to training feature width.
//
// Returns:
//   - []float64: detached original-unit response predictions, or nil/error.
//
// Errors:
//   - ErrInvalidOption for nil context; ErrNilInput for nil/unfitted model or matrix.
//   - ErrShape for negative/wrong dimensions; ErrOverflow for invalid byte size or math.
//   - ErrNonFinite for any NaN or infinity; ctx.Err() and Matrix.At errors retained.
//
// Determinism:
//   - Complete validation is row-major; prediction follows increasing rows/features.
//
// Complexity:
//   - Time O(m*p) and Space O(m) on the ordinary finite path.
//   - An exceptional row can require exact arithmetic, documented by its helper.
//
// Notes:
//   - Concurrent read-only calls on a fitted Model are safe with stable input matrices.
//   - Caller mutation of x while Predict executes is unsupported.
//
// AI-Hints:
//   - Do not compute predictions on partially validated batches if error order matters.
//   - *matrix.Dense inputs use its O(1) At access; no unsafe raw buffer is required.
func (m *Model) Predict(ctx context.Context, x matrix.Matrix) ([]float64, error) {
	if err := validateContext(ctx); err != nil {
		return nil, err
	}
	if m == nil || !m.fitted {
		return nil, ErrNilInput
	}
	if err := validateMatrixPresence(x); err != nil {
		return nil, err
	}
	rows, cols := x.Rows(), x.Cols()
	if rows < 0 || cols < 0 || cols != m.p {
		return nil, ErrShape
	}
	if rows == 0 {
		return []float64{}, nil
	}
	if cols > (math.MaxInt/bytesPerFloat64)/rows {
		return nil, ErrOverflow
	}

	// Preflight the entire supplied domain before multiplying any row.
	// This preserves the required nonfinite-before-compute error ordering.
	for i := 0; i < rows; i++ {
		if i%contextCheckInterval == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		for j := 0; j < cols; j++ {
			if j%contextCheckInterval == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			value, err := x.At(i, j)
			if err != nil {
				return nil, err
			}
			if !finite(value) {
				return nil, ErrNonFinite
			}
		}
	}

	output := make([]float64, rows)
	for i := range output {
		if i%contextCheckInterval == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		value, err := m.rowPrediction(ctx, x, i)
		if err != nil {
			return nil, err
		}
		output[i] = value
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return output, nil
}
