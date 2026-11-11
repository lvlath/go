// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ols_test locks numerical edge laws with independent full-rank data.
// These fixtures target meaningful overflow and scale-only transformations.
package ols_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/ols"
)

func TestOLS_StandardizeAvoidsIntermediateCenteringOverflow(t *testing.T) {
	// With population mean near -8.5e307, the positive sample's x-mean
	// exceeds MaxFloat64 although the normalized ratio is close to sqrt(3).
	// Rejecting that intermediate as an arithmetic overflow would be wrong.
	data := [][]float64{{-1.7e308}, {-1.7e308}, {-1.7e308}, {1.7e308}}
	result, err := ols.FitStandardized(context.Background(), mustDense(t, data),
		[]float64{4, 4, 4, 4})
	if err != nil {
		t.Fatalf("finite standardized design rejected: %v", err)
	}
	mustEqualInt(t, result.Rank, 2, "large-magnitude standardized rank")
	beta, intercept := result.Model.Coefficients()
	if !finiteForTest(beta[0]) || !finiteForTest(intercept) {
		t.Fatalf("published nonfinite coefficients: %v, %v", beta, intercept)
	}
	predicted, err := result.Model.Predict(context.Background(), mustDense(t, data))
	if err != nil {
		t.Fatal(err)
	}
	for i, value := range predicted {
		mustNear(t, value, 4, 1e-10)
		if math.IsNaN(value) {
			t.Fatalf("NaN prediction %d", i)
		}
	}
}

func TestOLS_SSEIsNotResidualNormOrMeanSquare(t *testing.T) {
	// With x=[-2,-1,0,1,2], constant response bias plus alternating noise
	// produces nonzero residuals even when the fitted slope is close to known.
	x := mustDense(t, [][]float64{{-2}, {-1}, {0}, {1}, {2}})
	y := []float64{-1, 2, 0, 4, 6}
	result, err := ols.FitDefault(context.Background(), x, y)
	if err != nil {
		t.Fatal(err)
	}
	predicted, err := result.Model.Predict(context.Background(), x)
	if err != nil {
		t.Fatal(err)
	}
	independent := 0.0
	for i, value := range predicted {
		residual := y[i] - value
		independent = math.FMA(residual, residual, independent)
	}
	mustNear(t, result.ResidualSumSquares, independent, 5e-14)
	if result.ResidualSumSquares <= 0 {
		t.Fatal("SSE unexpectedly zero")
	}
}

func TestOLS_NoInterceptScaledNeverCreatesHiddenOffset(t *testing.T) {
	// An asymmetric strictly positive feature has a nonzero mean; any
	// unauthorized centering would change this physical origin-constrained fit.
	x := mustDense(t, [][]float64{{2}, {5}, {9}, {15}, {26}, {41}})
	y := []float64{4, 10, 18, 30, 52, 82}
	result, err := ols.FitThroughOriginStandardized(context.Background(), x, y)
	if err != nil {
		t.Fatal(err)
	}
	beta, intercept := result.Model.Coefficients()
	mustNear(t, beta[0], 2, 1e-12)
	if intercept != 0 || math.Signbit(intercept) {
		t.Fatalf("implicit intercept: %g", intercept)
	}
	hold := mustDense(t, [][]float64{{0}, {1000}})
	predicted, err := result.Model.Predict(context.Background(), hold)
	if err != nil {
		t.Fatal(err)
	}
	mustNear(t, predicted[0], 0, 0)
	mustNear(t, predicted[1], 2000, 1e-12)
}

func TestOLS_SSEOverflowNeverPublishesFit(t *testing.T) {
	// A numerically valid design and finite responses have unrepresentable SSE.
	x := mustDense(t, [][]float64{{0}, {1}, {2}, {3}})
	result, err := ols.FitDefault(context.Background(), x,
		[]float64{1e200, -1e200, -1e200, 1e200})
	if result != nil || !errors.Is(err, ols.ErrOverflow) {
		t.Fatalf("SSE overflow must return nil/ErrOverflow, got %v/%v", result, err)
	}
}

// TestOLS_UnscaledCenteringRejectsUnrepresentableDesign distinguishes genuine
// overflow from the standardize-enabled case that can normalize the same data.
func TestOLS_UnscaledCenteringRejectsUnrepresentableDesign(t *testing.T) {
	x := mustDense(t, [][]float64{{-1.7e308}, {-1.7e308}, {-1.7e308}, {1.7e308}})
	result, err := ols.FitDefault(context.Background(), x, []float64{4, 4, 4, 4})
	if result != nil {
		t.Fatal("unrepresentable unscaled design published a Model")
	}
	mustErrorIs(t, err, ols.ErrOverflow)
}

// TestOLS_OriginalCoefficientOverflowRejectsFit prevents publishing a model
// whose QR-space slope is finite but physical-unit slope cannot fit float64.
func TestOLS_OriginalCoefficientOverflowRejectsFit(t *testing.T) {
	x := mustDense(t, [][]float64{{-2e-320}, {-1e-320}, {1e-320}, {2e-320}})
	result, err := ols.FitStandardized(context.Background(), x,
		[]float64{-2, -1, 1, 2})
	if result != nil {
		t.Fatal("unrepresentable physical coefficient published a Model")
	}
	mustErrorIs(t, err, ols.ErrOverflow)
}

// TestOLS_ExplicitRelativeRankTolerance refuses almost dependent columns
// when the caller deliberately selects a stricter numerical-rank threshold.
func TestOLS_ExplicitRelativeRankTolerance(t *testing.T) {
	data := [][]float64{
		{0, 0}, {1, 1.00000001}, {2, 2.00000001}, {3, 3.00000002},
		{4, 4.00000003}, {5, 5.00000001}, {6, 6.00000005}, {7, 7.00000001},
	}
	response := []float64{1, 3, 5, 7, 9, 11, 13, 15}
	result, err := ols.Fit(context.Background(), mustDense(t, data), response,
		ols.Options{RankTolerance: 1e-4})
	if result != nil {
		t.Fatal("rank cutoff ignored and partial model published")
	}
	mustErrorIs(t, err, ols.ErrRankDeficient)
}
