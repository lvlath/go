// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ols evaluates immutable fitted regressions in original physical units.
// This file implements compensated affine prediction, exact exceptional arithmetic,
// and batch validation without storing mutable numerical state in the model.
package ols

import (
	"context"
	"math"
	"math/big"

	"github.com/lvlath/go/matrix"
)

// The common path uses half the significant bits as a cancellation warning.
// Losing more than 26 of 53 binary significand bits justifies the exact path;
// it is not a rank tolerance, a prediction cutoff, or a way to change the model.
const affineCancellationThreshold = 0x1p-26

// A subnormal rounded product can lose the rounding error of an individual
// multiplication, even when that error would contribute to the final sum.
// Exact arithmetic is therefore used for these exceptionally small products.
const affineSmallestNormalProduct = 0x1p-1022

// exactRowPrediction evaluates the physical-unit affine response exactly over
// the finite binary64 inputs, then rounds the complete rational result once.
// The method rescues finite predictions whose intermediate products or sums
// cannot be represented in float64, without modifying fitted coefficients.
//
// Implementation:
//   - Stage 1: initialize an exact accumulator from the original-unit intercept.
//   - Stage 2: reread the stable input row and accumulate exact rational products.
//   - Stage 3: round the entire sum once, rejecting only nonfinite final output.
//
// Behavior highlights:
//   - Opposing products larger than MaxFloat64 may cancel to a finite result.
//   - Subnormal product contributions are retained until the final rounding.
//
// Inputs:
//   - ctx: active non-nil context, checked periodically and at the end.
//   - x: stable read-only n-by-p matrix with finite original-unit features.
//   - row: previously validated observation index; m: immutable fitted model.
//
// Returns:
//   - float64: rounded, finite physical-unit prediction, or zero with an error.
//
// Errors:
//   - ErrOverflow: only when the final rounded prediction is nonfinite.
//   - ErrNonFinite: a matrix cell became NaN or infinity between input passes.
//   - ctx.Err() or the original Matrix.At error, unchanged.
//
// Determinism:
//   - Feature columns are visited in their original, ascending order.
//   - big.Rat represents each finite binary64 operand exactly, with no rounding
//     until conversion of the completed affine sum to float64.
//
// Complexity:
//   - Time O(p*B^2), Space O(p*B) as conservative big-integer bounds,
//     where B includes fixed binary64 exponent widths and accumulated bits.
//
// Notes:
//   - This exceptional path allocates local big.Rat scratch registers.
//   - It rereads x, which the caller must not mutate during Predict or Fit.
//   - It never stores scratch values in Model or exposes a partial prediction.
//
// AI-Hints:
//   - Never substitute an intermediate float64 infinity for the true sum.
//   - Keep this fallback independent of QR, ridge penalties, and train scalers.
func (m *Model) exactRowPrediction(ctx context.Context, x matrix.Matrix, row int) (float64, error) {
	// Preserve the intercept exactly as the first affine term. A zero-intercept
	// fit stores positive zero, and no centering is performed during inference.
	var sum, product, featureRat, slopeRat big.Rat
	sum.SetFloat64(m.intercept)

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
		if !finite(feature) {
			return 0, ErrNonFinite
		}

		// SetFloat64 exactly captures the original finite binary64 values.
		// Multiply them before adding: no intermediate float64 product exists.
		featureRat.SetFloat64(feature)
		slopeRat.SetFloat64(slope)
		product.Mul(&featureRat, &slopeRat)
		sum.Add(&sum, &product)
	}

	if err := ctx.Err(); err != nil {
		return 0, err
	}
	prediction, _ := sum.Float64()
	if !finite(prediction) {
		return 0, ErrOverflow
	}
	return prediction, nil
}

// rowPrediction evaluates a frozen physical-unit affine response using
// compensated sums of rounded products and their FMA rounding residuals.
// Its fast path avoids allocations; exceptional ranges use exactRowPrediction.
//
// Implementation:
//   - Stage 1: seed the Neumaier accumulator with the fitted intercept.
//   - Stage 2: accumulate products, addition errors, and FMA product errors.
//   - Stage 3: verify the final value; retry exact arithmetic when overflow,
//     subnormal product loss, or severe cancellation threatens correctness.
//
// Behavior highlights:
//   - A small term between large canceling terms is not silently discarded.
//   - Intermediate infinity never by itself determines ErrOverflow.
//   - Standardization is NOT recomputed; coefficients already use original units.
//
// Inputs:
//   - ctx: non-nil active context; x: stable read-only finite n-by-p matrix.
//   - row: valid observation index; m: immutable successfully fitted model.
//
// Returns:
//   - float64: one finite response in original y units, or zero/error.
//
// Errors:
//   - ErrNonFinite for an input changed after prevalidation.
//   - ErrOverflow for a genuinely unrepresentable final exact response.
//   - ctx.Err() or the original Matrix.At error from either arithmetic path.
//
// Determinism:
//   - Columns ascend; rounding errors and the intercept use fixed order.
//   - Severe cancellation is checked against the sum of absolute terms.
//
// Complexity:
//   - Time O(p), Space O(1) and zero heap allocations in the ordinary row path;
//     exactRowPrediction documents the exceptional higher-precision cost.
//
// Notes:
//   - This helper is shared by public Predict and training SSE in fit.go.
//   - Neither Model nor x is mutated; concurrent stable-input inference is safe.
//
// AI-Hints:
//   - FMA on a running total is NOT compensated summation: retain both the
//     rounded-addition error and rounded-product error separately.
//   - Never stop at a nonfinite partial product; later terms can cancel it.
func (m *Model) rowPrediction(ctx context.Context, x matrix.Matrix, row int) (float64, error) {
	sum, correction := m.intercept, 0.0
	absoluteTerms := math.Abs(m.intercept)

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
		if !finite(feature) {
			return 0, ErrNonFinite
		}

		// An unrepresentable or subnormal individual product needs the exact
		// path: materializing it now would lose information before cancellation.
		term := feature * slope
		if !finite(term) || (feature != 0 && slope != 0 && math.Abs(term) < affineSmallestNormalProduct) {
			return m.exactRowPrediction(ctx, x, row)
		}

		// Track the conditioning of this particular affine sum, not the QR
		// factor. Overflow here means cancellation could still save the result.
		absoluteTerms += math.Abs(term)
		if !finite(absoluteTerms) {
			return m.exactRowPrediction(ctx, x, row)
		}

		// Neumaier's residual captures rounding when adding very different
		// magnitudes. Keep it separate from the rounded running sum.
		next := sum + term
		if !finite(next) {
			return m.exactRowPrediction(ctx, x, row)
		}
		var additionError float64
		if math.Abs(sum) >= math.Abs(term) {
			additionError = (sum - next) + term
		} else {
			additionError = (term - next) + sum
		}

		// FMA computes the residual of an individually rounded multiplication.
		// It is NOT the same error as the addition residual above.
		correction += additionError
		if !finite(correction) {
			return m.exactRowPrediction(ctx, x, row)
		}
		correction += math.FMA(feature, slope, -term)
		if !finite(correction) {
			return m.exactRowPrediction(ctx, x, row)
		}
		sum = next
	}

	if err := ctx.Err(); err != nil {
		return 0, err
	}
	prediction := sum + correction
	if !finite(prediction) {
		return m.exactRowPrediction(ctx, x, row)
	}

	// A heavily canceled response is exceptionally sensitive to cumulative
	// rounding errors, including errors in the correction itself. Recompute it
	// exactly instead of relying on an arbitrary near-zero float64 threshold.
	if absoluteTerms != 0 && math.Abs(prediction) <= affineCancellationThreshold*absoluteTerms {
		return m.exactRowPrediction(ctx, x, row)
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
// Complexity: Time O(n*p), Space O(n) for ordinary finite rows and detached output;
//
//	exceptional rows use exactRowPrediction with its documented big-integer cost.
//
// Notes: Concurrent calls are safe if callers do not concurrently mutate input matrices.
//
//	Compensated rounding may change last bits relative to historical sequential FMA.
//
// AI-Hints: A new inference batch MUST NOT trigger stats.FitStandardizer or scale refitting.
//
//	Zero-row batches allocate only the detached empty result; never attempt QR on them.
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
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return out, nil
}
