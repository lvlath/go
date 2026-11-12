// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ridge_test exercises exceptional numerical behavior through public APIs.
// Independent exact-binary arithmetic distinguishes finite cancellation from overflow.
package ridge_test

import (
	"context"
	"errors"
	"math"
	"math/big"
	"testing"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/ridge"
)

// TestRidge_PredictFiniteAfterIntermediateOverflow demonstrates why rejecting
// an individual x_j*beta_j infinity would be a false positive: two huge terms
// can cancel exactly before the correctly rounded final prediction is formed.
func TestRidge_PredictFiniteAfterIntermediateOverflow(t *testing.T) {
	train := dense(t, [][]float64{{1, 0}, {0, 1}})
	fit, err := ridge.FitThroughOrigin(context.Background(), train, []float64{10, -10}, .1)
	if err != nil {
		t.Fatal(err)
	}
	beta, _ := fit.Model.Coefficients()
	if len(beta) != 2 || math.Abs(beta[0]) <= 1 || math.Abs(beta[1]) <= 1 {
		t.Fatalf("fixture must trigger intermediate product overflow: beta=%v", beta)
	}
	future, err := matrix.NewDense(1, 2)
	if err != nil {
		t.Fatal(err)
	}
	for j := 0; j < 2; j++ {
		if err = future.Set(0, j, math.MaxFloat64); err != nil {
			t.Fatal(err)
		}
	}

	// Independent exact binary64 dot oracle: fitted slopes and features are
	// interpreted as exact binary numbers before one final float64 rounding.
	var sum, product, operand big.Rat
	for _, slope := range beta {
		operand.SetFloat64(slope)
		product.Mul(new(big.Rat).SetFloat64(math.MaxFloat64), &operand)
		sum.Add(&sum, &product)
	}
	want, _ := sum.Float64()
	if math.IsInf(want, 0) || math.IsNaN(want) {
		t.Fatal("fixture did not produce a representable final prediction")
	}
	predictions, err := fit.Model.Predict(context.Background(), future)
	if err != nil {
		t.Fatal(err)
	}
	if len(predictions) != 1 || math.Float64bits(predictions[0]) != math.Float64bits(want) {
		t.Fatalf("exact-fallback prediction=%.17g want=%.17g", predictions[0], want)
	}
}

// TestRidge_LargePositiveLambdaScalarOracle verifies nonzero shrinkage despite
// very small final slopes and checks that penalty scaling stays representable.
func TestRidge_LargePositiveLambdaScalarOracle(t *testing.T) {
	const lambda = 1e100
	fit, err := ridge.FitThroughOrigin(context.Background(), dense(t, [][]float64{{1}, {2}}), []float64{1, 2}, lambda)
	if err != nil {
		t.Fatal(err)
	}
	beta, intercept := fit.Model.Coefficients()
	if intercept != 0 || len(beta) != 1 {
		t.Fatalf("invalid fitted physical coordinates: %v %.17g", beta, intercept)
	}
	wantBeta := 5 / (5 + 2*lambda)
	wantPenalty := (lambda / 2) * wantBeta * wantBeta
	if math.Abs(beta[0]/wantBeta-1) > 1e-8 {
		t.Fatalf("coefficient %.17g want %.17g", beta[0], wantBeta)
	}
	if math.Abs(fit.Penalty/wantPenalty-1) > 1e-8 {
		t.Fatalf("penalty %.17g want %.17g", fit.Penalty, wantPenalty)
	}
	if fit.ResidualSumSquares <= 0 || fit.Objective <= 0 {
		t.Fatalf("expected nonzero residual cost: SSE=%g Objective=%g", fit.ResidualSumSquares, fit.Objective)
	}
}

// TestRidge_OriginScaleMatchesManualCalibration independently verifies that
// training population scales are used without subtracting feature means.
func TestRidge_OriginScaleMatchesManualCalibration(t *testing.T) {
	x := dense(t, [][]float64{{10, 5}, {20, 7}, {30, 9}, {40, 11}})
	y := []float64{2, 4, 6, 8}
	// Population standard deviations: sqrt(125) and sqrt(5).
	s0, s1 := math.Sqrt(125), math.Sqrt(5)
	manual := dense(t, [][]float64{
		{10 / s0, 5 / s1}, {20 / s0, 7 / s1},
		{30 / s0, 9 / s1}, {40 / s0, 11 / s1},
	})
	standard, err := ridge.FitThroughOriginStandardized(context.Background(), x, y, .4)
	if err != nil {
		t.Fatal(err)
	}
	explicit, err := ridge.FitThroughOrigin(context.Background(), manual, y, .4)
	if err != nil {
		t.Fatal(err)
	}
	beta, intercept := standard.Model.Coefficients()
	theta, _ := explicit.Model.Coefficients()
	if intercept != 0 {
		t.Fatal("scale-only training introduced a nonzero intercept")
	}
	closeTo(t, beta[0], theta[0]/s0)
	closeTo(t, beta[1], theta[1]/s1)
	closeTo(t, standard.Penalty, explicit.Penalty)
	closeTo(t, standard.Objective, explicit.Objective)
}

// TestRidge_CancelDuringInputScan propagates cancellation rather than replacing
// it by a matrix, shape, numeric, or regularization error.
func TestRidge_CancelDuringInputScan(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	input := &syntheticMatrix{rows: 2, cols: 2}
	input.at = func(_, _ int) (float64, error) {
		cancel()
		return 1, nil
	}
	result, err := ridge.Fit(ctx, input, []float64{1, 2}, ridge.Options{Lambda: 1})
	if result != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("result=%v error=%v, want canceled and nil", result, err)
	}
}

// TestRidge_CustomMatrixErrorPreserved proves that matrix access failures are
// not converted into unrelated package errors or matched by message strings.
func TestRidge_CustomMatrixErrorPreserved(t *testing.T) {
	marker := errors.New("source: unavailable reading")
	input := &syntheticMatrix{rows: 1, cols: 1, at: func(int, int) (float64, error) { return 0, marker }}
	result, err := ridge.Fit(context.Background(), input, []float64{1}, ridge.Options{Lambda: 1})
	if result != nil || !errors.Is(err, marker) {
		t.Fatalf("expected original Matrix.At root: result=%v err=%v", result, err)
	}
}

// TestRidge_PredictContextAndShapeBoundary checks inference validation before
// any value-dependent computation or result allocation.
func TestRidge_PredictContextAndShapeBoundary(t *testing.T) {
	fit, err := ridge.FitWithLambda(context.Background(), dense(t, [][]float64{{1}, {2}, {3}}), []float64{1, 2, 3}, 1)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if output, err := fit.Model.Predict(ctx, nil); output != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel precedence: %v %v", output, err)
	}
	wrong := &syntheticMatrix{rows: -1, cols: 1}
	if output, err := fit.Model.Predict(context.Background(), wrong); output != nil || !errors.Is(err, ridge.ErrShape) {
		t.Fatalf("negative prediction rows: %v %v", output, err)
	}
	broken := &syntheticMatrix{rows: 1, cols: 1, at: func(int, int) (float64, error) { return 0, matrix.ErrOutOfRange }}
	if output, err := fit.Model.Predict(context.Background(), broken); output != nil || !errors.Is(err, matrix.ErrOutOfRange) {
		t.Fatalf("inference At error: %v %v", output, err)
	}
}

// TestRidge_InternalCenterOverflowIsExplicit checks an unscaled affine design
// whose mathematically needed centered coordinate is outside float64 range.
// Ridge must reject it instead of inserting Inf into the augmented QR matrix.
func TestRidge_InternalCenterOverflowIsExplicit(t *testing.T) {
	maxV := math.MaxFloat64
	x := dense(t, [][]float64{{-maxV}, {-maxV}, {maxV}})
	fit, err := ridge.FitWithLambda(context.Background(), x, []float64{-1, -1, 1}, 1)
	if fit != nil || !errors.Is(err, ridge.ErrOverflow) {
		t.Fatalf("unrepresentable centered design: fit=%v err=%v", fit, err)
	}
}

// TestRidge_StrictRelativeRankTolerance requests a cutoff so high that even
// the leading QR diagonal is declared deficient. It must never trigger OLS.
func TestRidge_StrictRelativeRankTolerance(t *testing.T) {
	x := dense(t, [][]float64{{1, 0}, {0, 1}, {1, 1}})
	fit, err := ridge.Fit(context.Background(), x, []float64{1, 2, 3}, ridge.Options{
		Lambda: 1, RankTolerance: 2,
	})
	if fit != nil || !errors.Is(err, ridge.ErrRankDeficient) {
		t.Fatalf("relative rank cutoff: fit=%v err=%v", fit, err)
	}
}

// TestRidge_PredictAllocationPrecedence rejects an impossible prediction buffer
// without dereferencing a user-defined virtual Matrix cell.
func TestRidge_PredictAllocationPrecedence(t *testing.T) {
	fit, err := ridge.FitWithLambda(context.Background(), dense(t, [][]float64{{1}, {2}}), []float64{1, 2}, 1)
	if err != nil {
		t.Fatal(err)
	}
	read := false
	input := &syntheticMatrix{rows: math.MaxInt / 4, cols: 1, at: func(int, int) (float64, error) {
		read = true
		return math.NaN(), nil
	}}
	predicted, err := fit.Model.Predict(context.Background(), input)
	if predicted != nil || !errors.Is(err, ridge.ErrOverflow) || read {
		t.Fatalf("impossible prediction allocation: pred=%v err=%v read=%t", predicted, err, read)
	}
}
