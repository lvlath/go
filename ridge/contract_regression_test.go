// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ridge_test protects the public ridge contract and published regression fixes.
// All tests invoke only exported ridge APIs; no test-only production entry points exist.
package ridge_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/ridge"
)

// syntheticMatrix is an external Matrix implementation for boundary testing.
// It validates error precedence without constructing unrepresentable Dense storage.
type syntheticMatrix struct {
	rows, cols int
	at         func(row, col int) (float64, error)
}

func (s *syntheticMatrix) IsNil() bool { return s == nil }
func (s *syntheticMatrix) Rows() int   { return s.rows }
func (s *syntheticMatrix) Cols() int   { return s.cols }
func (s *syntheticMatrix) At(row, col int) (float64, error) {
	if s.at != nil {
		return s.at(row, col)
	}
	return 0, nil
}
func (s *syntheticMatrix) Set(_, _ int, _ float64) error { return matrix.ErrOutOfRange }
func (s *syntheticMatrix) Clone() matrix.Matrix {
	return &syntheticMatrix{rows: s.rows, cols: s.cols, at: s.at}
}

var _ matrix.Matrix = (*syntheticMatrix)(nil)
var _ core.Nilable = (*syntheticMatrix)(nil)

// TestRidge_FacadesAreCanonical asserts exact option selection, not merely fit success.
func TestRidge_FacadesAreCanonical(t *testing.T) {
	ctx := context.Background()
	x := dense(t, [][]float64{{1, 3}, {2, 2}, {3, 5}, {4, 8}, {5, 7}, {6, 9}, {7, 6}})
	y := []float64{2, 3, 4, 7, 6, 8, 9}
	const lambda = 0.75
	tests := []struct {
		name string
		fit  func() (*ridge.FitResult, error)
		opts ridge.Options
	}{
		{"with lambda", func() (*ridge.FitResult, error) { return ridge.FitWithLambda(ctx, x, y, lambda) }, ridge.Options{Lambda: lambda}},
		{"standardized", func() (*ridge.FitResult, error) { return ridge.FitStandardized(ctx, x, y, lambda) }, ridge.Options{Lambda: lambda, Standardize: true}},
		{"through origin", func() (*ridge.FitResult, error) { return ridge.FitThroughOrigin(ctx, x, y, lambda) }, ridge.Options{Lambda: lambda, NoIntercept: true}},
		{"origin standardized", func() (*ridge.FitResult, error) { return ridge.FitThroughOriginStandardized(ctx, x, y, lambda) }, ridge.Options{Lambda: lambda, NoIntercept: true, Standardize: true}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			actual, err := tc.fit()
			if err != nil {
				t.Fatal(err)
			}
			reference, err := ridge.Fit(ctx, x, y, tc.opts)
			if err != nil {
				t.Fatal(err)
			}
			a, interceptA := actual.Model.Coefficients()
			b, interceptB := reference.Model.Coefficients()
			if len(a) != len(b) {
				t.Fatal("different coefficient dimensions")
			}
			for j := range a {
				if math.Float64bits(a[j]) != math.Float64bits(b[j]) {
					t.Fatalf("coefficient[%d] wrapper=%.17g Fit=%.17g", j, a[j], b[j])
				}
			}
			if math.Float64bits(interceptA) != math.Float64bits(interceptB) ||
				math.Float64bits(actual.Objective) != math.Float64bits(reference.Objective) ||
				math.Float64bits(actual.Penalty) != math.Float64bits(reference.Penalty) ||
				math.Float64bits(actual.ResidualSumSquares) != math.Float64bits(reference.ResidualSumSquares) {
				t.Fatal("wrapper deviated from canonical Fit")
			}
		})
	}
}

// TestRidge_LambdaZeroRejectedThroughAllFacades forbids bypassing Lambda validation.
func TestRidge_LambdaZeroRejectedThroughAllFacades(t *testing.T) {
	ctx := context.Background()
	x := dense(t, [][]float64{{1}, {2}})
	y := []float64{1, 2}
	for _, lambda := range []float64{0, math.Copysign(0, -1), -1, math.NaN(), math.Inf(1), math.Inf(-1)} {
		fits := []func() (*ridge.FitResult, error){
			func() (*ridge.FitResult, error) { return ridge.FitWithLambda(ctx, x, y, lambda) },
			func() (*ridge.FitResult, error) { return ridge.FitStandardized(ctx, x, y, lambda) },
			func() (*ridge.FitResult, error) { return ridge.FitThroughOrigin(ctx, x, y, lambda) },
			func() (*ridge.FitResult, error) { return ridge.FitThroughOriginStandardized(ctx, x, y, lambda) },
		}
		for index, fit := range fits {
			result, err := fit()
			if result != nil || !errors.Is(err, ridge.ErrInvalidOption) {
				t.Fatalf("Lambda=%g wrapper=%d: result=%v error=%v", lambda, index, result, err)
			}
		}
	}
}

// TestRidge_SizePrecedesFiniteAndDataAccess exercises a virtual enormous design.
// The rejected matrix must never receive an At call or trigger a large allocation.
func TestRidge_SizePrecedesFiniteAndDataAccess(t *testing.T) {
	accessed := false
	x := &syntheticMatrix{rows: 1, cols: math.MaxInt / 8}
	x.at = func(_, _ int) (float64, error) {
		accessed = true
		return math.NaN(), nil
	}
	result, err := ridge.Fit(context.Background(), x, []float64{1}, ridge.Options{Lambda: 1})
	if result != nil || !errors.Is(err, ridge.ErrOverflow) || accessed {
		t.Fatalf("expected pre-allocation overflow before reads, got result=%v error=%v read=%t", result, err, accessed)
	}
}

// TestRidge_FitErrorPrecedence independently checks context/options/nil/shape/value.
func TestRidge_FitErrorPrecedence(t *testing.T) {
	x := &syntheticMatrix{rows: 2, cols: 1, at: func(_, _ int) (float64, error) { return math.NaN(), nil }}
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	tests := []struct {
		name string
		ctx  context.Context
		x    matrix.Matrix
		y    []float64
		opts ridge.Options
		want error
	}{
		{"context before options", cancelled, nil, nil, ridge.Options{}, context.Canceled},
		{"options before nil", context.Background(), nil, nil, ridge.Options{}, ridge.ErrInvalidOption},
		{"nil before shape", context.Background(), nil, []float64{1}, ridge.Options{Lambda: 1}, ridge.ErrNilInput},
		{"shape before nonfinite", context.Background(), x, []float64{1}, ridge.Options{Lambda: 1}, ridge.ErrShape},
		{"nonfinite", context.Background(), x, []float64{1, 2}, ridge.Options{Lambda: 1}, ridge.ErrNonFinite},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result, err := ridge.Fit(tc.ctx, tc.x, tc.y, tc.opts)
			if result != nil || !errors.Is(err, tc.want) {
				t.Fatalf("result=%v err=%v expected=%v", result, err, tc.want)
			}
		})
	}
}

// TestRidge_PredictCompleteFiniteValidation ensures no computed-row error can
// hide a later NaN and that the method never leaks partial predictions.
func TestRidge_PredictCompleteFiniteValidation(t *testing.T) {
	fit, err := ridge.FitThroughOrigin(context.Background(), dense(t, [][]float64{{1}, {2}}), []float64{4, 8}, 1)
	if err != nil {
		t.Fatal(err)
	}
	x := &syntheticMatrix{rows: 2, cols: 1}
	x.at = func(row, _ int) (float64, error) {
		if row == 0 {
			return math.MaxFloat64, nil // would overflow when multiplied by fitted beta
		}
		return math.NaN(), nil // must win before any prediction calculation
	}
	predictions, err := fit.Model.Predict(context.Background(), x)
	if predictions != nil || !errors.Is(err, ridge.ErrNonFinite) {
		t.Fatalf("prediction=%v err=%v, want ErrNonFinite", predictions, err)
	}
}

// TestRidge_PredictReceiverAndTypedNilMatrix checks absence and zero-state contracts.
func TestRidge_PredictReceiverAndTypedNilMatrix(t *testing.T) {
	ctx := context.Background()
	var untrained ridge.Model
	_, err := untrained.Predict(ctx, nil)
	if !errors.Is(err, ridge.ErrNilInput) {
		t.Fatalf("untrained receiver: %v", err)
	}
	var typedNil *syntheticMatrix
	_, err = ridge.Fit(ctx, typedNil, []float64{1}, ridge.Options{Lambda: 1})
	if !errors.Is(err, ridge.ErrNilInput) || !errors.Is(err, matrix.ErrNilMatrix) {
		t.Fatalf("typed nil matrix: %v", err)
	}
	_, err = untrained.Predict(nil, typedNil)
	if !errors.Is(err, ridge.ErrInvalidOption) {
		t.Fatalf("nil context must precede model check: %v", err)
	}
}

// TestRidge_PredictRejectsUnrepresentableFinalValue requires a true overflow
// instead of Inf in the output domain, even after exact fallback arithmetic.
func TestRidge_PredictRejectsUnrepresentableFinalValue(t *testing.T) {
	fit, err := ridge.FitThroughOrigin(context.Background(), dense(t, [][]float64{{1}, {2}}), []float64{4, 8}, 1)
	if err != nil {
		t.Fatal(err)
	}
	x, err := matrix.NewDense(1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err = x.Set(0, 0, math.MaxFloat64); err != nil {
		t.Fatal(err)
	}
	pred, err := fit.Model.Predict(context.Background(), x)
	if pred != nil || !errors.Is(err, ridge.ErrOverflow) {
		t.Fatalf("final overflow: prediction=%v error=%v", pred, err)
	}
}

// TestRidge_NoInterceptNoHiddenCentering checks origin after scale-only training
// using distinct nonzero feature means and a strictly zero future feature row.
func TestRidge_NoInterceptNoHiddenCentering(t *testing.T) {
	train := dense(t, [][]float64{{120, 4}, {126, 5}, {132, 6}, {138, 7}, {144, 8}, {150, 9}})
	response := []float64{1.2, 2.2, 3.1, 4.1, 5.3, 6.2}
	result, err := ridge.FitThroughOriginStandardized(context.Background(), train, response, .3)
	if err != nil {
		t.Fatal(err)
	}
	_, intercept := result.Model.Coefficients()
	if math.Float64bits(intercept) != math.Float64bits(0) {
		t.Fatalf("intercept must be positive zero, got %v", intercept)
	}
	predictions, err := result.Model.Predict(context.Background(), dense(t, [][]float64{{0, 0}, {120, 4}}))
	if err != nil {
		t.Fatal(err)
	}
	if math.Float64bits(predictions[0]) != math.Float64bits(0) {
		t.Fatalf("origin must map to exact zero: %.17g", predictions[0])
	}
}
