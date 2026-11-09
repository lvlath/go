// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package stats_test checks fitted scaling against exact rational statistics.
// The test oracle retains huge and tiny variances until square-root rounding.
package stats_test

import (
	"errors"
	"math"
	"math/big"
	"testing"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/stats"
)

const (
	// oracleScalePrecisionBits supplies substantially more than the 106-bit
	// production compensated significand and the 53-bit output scale.
	oracleScalePrecisionBits = uint(512)
)

// oracleRationalScale computes sqrt of a sample's EXACT central variance.
// The variance is never rounded to float64 before taking the square root,
// allowing the reference to validate fitted scales beyond float64 M2 range.
//
// Implementation:
//   - Stage 1: compute exact binary rational mean from the input float64 values.
//   - Stage 2: accumulate squared rational deviations and apply ddof.
//   - Stage 3: compute a high-precision square root and round once to float64.
//
// Behavior highlights:
//   - Handles a positive rational variance whose float64 conversion underflows.
//
// Inputs:
//   - values: nonempty finite observations; ddof: 0 or 1, with len>ddof.
//
// Returns:
//   - float64: correctly rounded independent reference within 512-bit precision.
//
// Errors:
//   - None; test fixture preconditions exclude undefined denominators.
//
// Determinism:
//   - Fixed rational summation order and nearest-even output rounding.
//
// Complexity:
//   - Time O(len(values)) arbitrary-precision arithmetic; space O(len(values)).
//
// Notes:
//   - This oracle is intentionally test-only and is not suitable for fit hot loops.
//
// AI-Hints:
//   - Do not call math.Sqrt on the rounded float64 variance as an oracle.
//   - The reference must retain exact binary input values, not decimal strings.
func oracleRationalScale(values []float64, ddof int) float64 {
	count := big.NewRat(int64(len(values)), 1)
	sum := new(big.Rat)
	for _, value := range values {
		sum.Add(sum, newRat(value))
	}
	mean := new(big.Rat).Quo(sum, count)

	centralSum := new(big.Rat)
	for _, value := range values {
		deviation := new(big.Rat).Sub(newRat(value), mean)
		centralSum.Add(centralSum, deviation.Mul(deviation, deviation))
	}
	variance := centralSum.Quo(centralSum, big.NewRat(int64(len(values)-ddof), 1))
	root := new(big.Float).SetPrec(oracleScalePrecisionBits).SetMode(big.ToNearestEven).SetRat(variance)
	root.Sqrt(root)
	scale, _ := root.Float64()
	return scale
}

// oracleAffine evaluates an affine transform with exact binary float64 inputs.
// It uses the *rounded fitted parameters* as part of the public model law,
// rather than silently substituting an unrounded training mean or scale.
//
// Implementation:
//   - Stage 1: lift input and fitted parameters to exact rational values.
//   - Stage 2: compute either (value-mean)/scale or value*scale+mean.
//   - Stage 3: round once to float64, including Inf for out-of-range results.
//
// Behavior highlights:
//   - The inverse reference matches mathematical fused multiply-add semantics.
//
// Inputs:
//   - value, mean, scale: finite float64 inputs; scale strictly positive.
//   - inverse: true for reconstruction, false for standardization.
//
// Returns:
//   - float64: rounded exact affine result; infinity signals overflow.
//
// Errors:
//   - None; inputs obey the trained Standardizer preconditions.
//
// Determinism:
//   - Exact rational arithmetic, one final nearest-even conversion.
//
// Complexity:
//   - O(1) bounded-size rational operations; no sample-count dependence.
//
// Notes:
//   - The reference is independent of production scaledNumber and math.FMA.
//
// AI-Hints:
//   - Never infer an affine oracle by calling the package's Transform.
//   - The two divisions value/scale - mean/scale are not algebraically safe in float64.
func oracleAffine(value, mean, scale float64, inverse bool) float64 {
	result := newRat(value)
	if inverse {
		result.Mul(result, newRat(scale))
		result.Add(result, newRat(mean))
	} else {
		result.Sub(result, newRat(mean))
		result.Quo(result, newRat(scale))
	}
	converted, _ := result.Float64()
	return converted
}

// TestStandardizer_ExtremeFittedScaleOracle evaluates four independent columns.
// It verifies the exact training means, compensated scales, and constant mask
// against rational references at large offset, huge variance, and subnormal M2.
//
// Implementation:
//   - Stage 1: construct a rectangular training matrix with contrasting ranges.
//   - Stage 2: fit with explicit UnitScaleConstant policy for one constant column.
//   - Stage 3: compare each fitted parameter with a separate rational oracle.
//
// Behavior highlights:
//   - Rounded variance zero must not silently imply constant training data.
//
// Inputs:
//   - t: owning test; matrix entries are original finite float64 samples.
//
// Returns:
//   - No value; failures identify the offending column and ddof policy.
//
// Errors:
//   - No error expected for representable positive scale or selected unit scale.
//
// Determinism:
//   - Feature columns are visited in ascending column order.
//
// Complexity:
//   - O(rows*cols) production work and independent per-column rational work.
//
// Notes:
//   - The model retains means/scales, but public ScaleFitResult slices are detached.
//
// AI-Hints:
//   - Do not round an underflowing variance and then classify it as constant.
//   - Sample ddof changes only variance, not arithmetic means or extrema.
func TestStandardizer_ExtremeFittedScaleOracle(t *testing.T) {
	tiny := math.Ldexp(1, -537)
	train := mustDense(t, [][]float64{
		{1e12 + 1, -1e154, 5, -tiny},
		{1e12 + 2, 0, 5, tiny},
		{1e12 + 3, 1e154, 5, -tiny},
		{1e12 + 4, -1e154, 5, tiny},
		{1e12 + 5, 0, 5, -tiny},
		{1e12 + 6, 1e154, 5, tiny},
	})
	for _, ddof := range []int{0, 1} {
		fit, err := stats.FitStandardizer(train, stats.ScaleOptions{DDOF: ddof, ConstantPolicy: stats.UnitScaleConstant})
		if err != nil {
			t.Fatal(err)
		}
		if len(fit.Means) != train.Cols() || len(fit.Scales) != train.Cols() || len(fit.Constant) != train.Cols() {
			t.Fatalf("ddof=%d: incorrect model feature metadata lengths", ddof)
		}

		for column := 0; column < train.Cols(); column++ {
			values := make([]float64, train.Rows())
			for row := 0; row < train.Rows(); row++ {
				values[row] = mustCell(t, train, row, column)
			}
			wantMean, _ := oracleRatSummary(values, ddof)
			if fit.Means[column] != wantMean {
				t.Fatalf("ddof=%d column=%d: mean=%g oracle=%g", ddof, column, fit.Means[column], wantMean)
			}
			if column == 2 {
				if !fit.Constant[column] || fit.Scales[column] != 1 {
					t.Fatalf("ddof=%d constant column lost unit scale", ddof)
				}
				continue
			}
			if fit.Constant[column] {
				t.Fatalf("ddof=%d column=%d: positive variance misclassified as constant", ddof, column)
			}
			wantScale := oracleRationalScale(values, ddof)
			if wantScale == 0 || math.IsInf(wantScale, 0) {
				t.Fatalf("ddof=%d column=%d: invalid fixture scale %g", ddof, column, wantScale)
			}
			// Scales are computed with double-double M2 and sqrt rounding;
			// a scale-aware tolerance is permitted, not bitwise equality.
			if math.Abs(fit.Scales[column]-wantScale) > varianceRelativeErrorBudget*wantScale {
				t.Fatalf("ddof=%d column=%d: scale=%g oracle=%g", ddof, column, fit.Scales[column], wantScale)
			}
		}
	}
}

// TestStandardizer_AffineOverflowAndFMAOracle isolates two numeric hazards.
// Forward subtraction and inverse multiplication may exceed MaxFloat64 even
// when their final affine results are finite and must be returned correctly.
//
// Implementation:
//   - Stage 1: fit two independent bounded training models with wide offsets.
//   - Stage 2: transform/invert observations requiring extended or fused arithmetic.
//   - Stage 3: compare returned cells against exact rational affine results.
//
// Behavior highlights:
//   - Valid representable affine results must not be classified ErrOverflow.
//
// Inputs:
//   - t: owning test; all input matrix values are individually finite.
//
// Returns:
//   - No value; exact reference confirms both transformations at float64 boundary.
//
// Errors:
//   - ErrOverflow is required only when the exact final affine result is nonfinite.
//
// Determinism:
//   - Matrices are traversed row-major with fixed model parameters.
//
// Complexity:
//   - O(1) fitted features and cells for these focused regression fixtures.
//
// Notes:
//   - Inference never modifies or refits either model.
//
// AI-Hints:
//   - Do not replace FMA with a two-operation product followed by addition.
//   - Do not compute value/scale-mean/scale as a fallback for huge deltas.
func TestStandardizer_AffineOverflowAndFMAOracle(t *testing.T) {
	// The training mean is negative and the fitted scale positive. A new
	// positive observation creates an unrepresentable raw subtraction.
	forward, err := stats.FitStandardizer(mustDense(t, [][]float64{{-1.6e308}, {0.4e308}}), stats.ScaleOptions{})
	if err != nil {
		t.Fatal(err)
	}
	newValue := 1.7e308
	inference := mustDense(t, [][]float64{{newValue}})
	standardized, err := forward.Model.Transform(inference)
	if err != nil {
		t.Fatal(err)
	}
	wantForward := oracleAffine(newValue, forward.Means[0], forward.Scales[0], false)
	if math.IsInf(wantForward, 0) || wantForward == 0 {
		t.Fatalf("forward oracle setup incorrect: %g", wantForward)
	}
	if math.Abs(mustCell(t, standardized, 0, 0)-wantForward) > varianceRelativeErrorBudget*math.Abs(wantForward) {
		t.Fatalf("forward overflow fallback mismatch: got %g want %g", mustCell(t, standardized, 0, 0), wantForward)
	}

	// A negative normalized input overflows when multiplied by the scale,
	// but a fused product-plus-mean restores a finite original coordinate.
	inverseFit, err := stats.FitStandardizer(mustDense(t, [][]float64{{0.65e308}, {1.75e308}}), stats.ScaleOptions{})
	if err != nil {
		t.Fatal(err)
	}
	const negativeNormalized = -4.0
	reconstructed, err := inverseFit.Model.InverseTransform(mustDense(t, [][]float64{{negativeNormalized}}))
	if err != nil {
		t.Fatal(err)
	}
	wantInverse := oracleAffine(negativeNormalized, inverseFit.Means[0], inverseFit.Scales[0], true)
	if wantInverse != mustCell(t, reconstructed, 0, 0) {
		t.Fatalf("inverse FMA mismatch: got %.17g, exact rational oracle %.17g", mustCell(t, reconstructed, 0, 0), wantInverse)
	}
}

// TestStandardizer_MatrixValidationPrecedence asserts the public error law.
// A malformed input or bad option must fail before matrix traversal or output
// publication, and a late nonfinite value must outrank an earlier overflow.
//
// Implementation:
//   - Stage 1: exercise invalid options and nil/typed-nil matrix inputs.
//   - Stage 2: verify shape and size validation precedes cell access.
//   - Stage 3: verify exact sentinel identity on At errors and nonfinite cells.
//
// Behavior highlights:
//   - Fits and transforms return nil results on every validation error.
//
// Inputs:
//   - t: owning test; scriptedMatrix is a read-only external interface fixture.
//
// Returns:
//   - No result; errors and absence of partial model/matrix are asserted.
//
// Errors:
//   - ErrInvalidOption, ErrNilInput, ErrEmptyInput, ErrShape, ErrNonFinite,
//     ErrInsufficientSamples, and wrapped matrix.At error are distinguished.
//
// Determinism:
//   - Canonical order: options, nil, shape, nonfinite, domain, compute.
//
// Complexity:
//   - O(1) validation fixtures; at most two scripted cell reads.
//
// Notes:
//   - A typed-nil custom Matrix must implement IsNil per the Matrix convention.
//
// AI-Hints:
//   - Use errors.Is rather than examining diagnostic error text.
//   - Do not allocate a result or read cells after options/shape failure.
func TestStandardizer_MatrixValidationPrecedence(t *testing.T) {
	var nilMatrix *scriptedMatrix
	fit, err := stats.FitStandardizer(nilMatrix, stats.ScaleOptions{DDOF: 2})
	mustErrorIs(t, err, stats.ErrInvalidOption)
	mustNilState(t, fit, true, "options_before_typed_nil")

	fit, err = stats.FitStandardizer(nilMatrix, stats.ScaleOptions{})
	mustErrorIs(t, err, stats.ErrNilInput)
	mustErrorIs(t, err, matrix.ErrNilMatrix)
	mustNilState(t, fit, true, "typed_nil")

	fit, err = stats.FitStandardizer(&scriptedMatrix{rows: 0, cols: 3}, stats.ScaleOptions{})
	mustErrorIs(t, err, stats.ErrEmptyInput)
	mustNilState(t, fit, true, "empty_matrix")

	readCount := 0
	fit, err = stats.FitStandardizer(&scriptedMatrix{rows: -1, cols: 4, readCount: &readCount}, stats.ScaleOptions{})
	mustErrorIs(t, err, stats.ErrShape)
	mustNilState(t, fit, true, "negative_shape")
	if readCount != 0 {
		t.Fatal("invalid shape triggered a matrix read")
	}

	fit, err = stats.FitStandardizer(&scriptedMatrix{rows: 1, cols: 1, value: math.NaN()}, stats.ScaleOptions{DDOF: 1})
	mustErrorIs(t, err, stats.ErrNonFinite)
	mustNilState(t, fit, true, "nonfinite_before_domain")

	fit, err = stats.FitStandardizer(&scriptedMatrix{rows: 1, cols: 1, value: 5}, stats.ScaleOptions{DDOF: 1})
	mustErrorIs(t, err, stats.ErrInsufficientSamples)
	mustNilState(t, fit, true, "insufficient_rows")

	// Matrix-specific errors remain available through errors.Is rather
	// than being collapsed into an unrelated stats domain sentinel.
	readFailure := errors.New("scripted At failure")
	fit, err = stats.FitStandardizer(&scriptedMatrix{rows: 2, cols: 1, readErr: readFailure}, stats.ScaleOptions{})
	if !errors.Is(err, readFailure) || fit != nil {
		t.Fatalf("lost matrix.At error identity: model=%v err=%v", fit, err)
	}
}

// TestStandardizer_ZeroRowsPreserveShape exercises inference's empty-domain law.
// A zero-row matrix with the fitted feature count must round-trip as an
// independent, concrete matrix, and a different feature count must fail.
//
// Implementation:
//   - Stage 1: fit on finite nonconstant training columns.
//   - Stage 2: apply forward/inverse inference to 0-by-d input matrices.
//   - Stage 3: validate shapes, ownership and feature mismatch classification.
//
// Behavior highlights:
//   - Zero observations are legal at inference, not during fitting.
//
// Inputs:
//   - t: owning test; all matrices are real concrete matrix.Dense values.
//
// Returns:
//   - No result; shape and detached matrix identities are asserted.
//
// Errors:
//   - ErrShape for incorrect inference feature count.
//
// Determinism:
//   - Empty element domains cause no row/column read or write calls.
//
// Complexity:
//   - O(1) allocation for zero-row output, excluding training setup.
//
// Notes:
//   - Matrix.Clone or empty data buffers do not imply output pointer aliasing.
//
// AI-Hints:
//   - Do not collapse 0-by-d matrices into nil or lose the nonzero column shape.
//   - The zero-row law does not permit zero-row fitting.
func TestStandardizer_ZeroRowsPreserveShape(t *testing.T) {
	train := mustDense(t, [][]float64{{1, 10}, {3, 20}})
	fit, err := stats.FitStandardizer(train, stats.ScaleOptions{})
	if err != nil {
		t.Fatal(err)
	}
	input, err := matrix.NewDense(0, train.Cols())
	if err != nil {
		t.Fatal(err)
	}
	standardized, err := fit.Model.Transform(input)
	if err != nil {
		t.Fatal(err)
	}
	if standardized.Rows() != 0 || standardized.Cols() != train.Cols() || standardized == input {
		t.Fatal("forward zero-row output has wrong shape or identity")
	}
	reconstructed, err := fit.Model.InverseTransform(standardized)
	if err != nil {
		t.Fatal(err)
	}
	if reconstructed.Rows() != 0 || reconstructed.Cols() != train.Cols() || reconstructed == standardized {
		t.Fatal("inverse zero-row output has wrong shape or identity")
	}
	wrongColumns, err := matrix.NewDense(0, train.Cols()+1)
	if err != nil {
		t.Fatal(err)
	}
	result, err := fit.Model.Transform(wrongColumns)
	mustErrorIs(t, err, stats.ErrShape)
	mustNilState(t, result, true, "empty_row_wrong_feature_count")
}
