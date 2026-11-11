// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ols_test verifies facade option equivalence and mathematical invariants.
// Every test deliberately uses only the exported ols and matrix APIs.
package ols_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/ols"
)

func TestFacade_ExactOptionEquivalence(t *testing.T) {
	ctx := context.Background()
	data := [][]float64{{1, 5}, {2, 2}, {3, 8}, {4, 3}, {5, 11}, {6, 5}, {7, 13}, {8, 6}}
	x := mustDense(t, data)
	y := make([]float64, len(data))
	for i, row := range data {
		y[i] = 2.5*row[0] - 1.2*row[1]
	}
	for _, tt := range []struct {
		name string
		opts ols.Options
		fit  func(context.Context, matrix.Matrix, []float64) (*ols.FitResult, error)
	}{
		{"default", ols.Options{}, ols.FitDefault},
		{"standardized", ols.Options{Standardize: true}, ols.FitStandardized},
		{"through_origin", ols.Options{NoIntercept: true}, ols.FitThroughOrigin},
		{"origin_scaled", ols.Options{NoIntercept: true, Standardize: true}, ols.FitThroughOriginStandardized},
	} {
		t.Run(tt.name, func(t *testing.T) {
			canonical, err := ols.Fit(ctx, x, y, tt.opts)
			if err != nil {
				t.Fatal(err)
			}
			wrapped, err := tt.fit(ctx, x, y)
			if err != nil {
				t.Fatal(err)
			}
			mustEqualInt(t, wrapped.Rank, canonical.Rank, "wrapper rank")
			if wrapped.ConditionEstimate != canonical.ConditionEstimate || wrapped.ResidualSumSquares != canonical.ResidualSumSquares {
				t.Fatal("wrapper altered canonical QR diagnostic")
			}
			beta, intercept := wrapped.Model.Coefficients()
			wantBeta, wantIntercept := canonical.Model.Coefficients()
			for j := range wantBeta {
				if beta[j] != wantBeta[j] {
					t.Fatalf("wrapper slope[%d] mismatch", j)
				}
			}
			if intercept != wantIntercept {
				t.Fatalf("wrapper intercept mismatch: %g != %g", intercept, wantIntercept)
			}
			if tt.opts.NoIntercept {
				mustEqualBool(t, intercept == 0 && !math.Signbit(intercept), true, "positive-zero intercept")
			}
		})
	}
}

// TestFitResult_PresenceAndDetachedModel distinguishes nilable pointer presence
// from fitted model lifetime and compares rounded QR outputs numerically.
func TestFitResult_PresenceAndDetachedModel(t *testing.T) {
	var absent *ols.FitResult
	var nilable core.Nilable = absent
	mustEqualBool(t, nilable.IsNil(), true, "typed nil FitResult")
	nilable = &ols.FitResult{}
	mustEqualBool(t, nilable.IsNil(), false, "allocated FitResult")

	x := mustDense(t, [][]float64{{1}, {2}, {3}, {4}})
	result, err := ols.FitDefault(context.Background(), x, []float64{3, 5, 7, 9})
	if err != nil {
		t.Fatal(err)
	}
	// The public FitResult is mutable metadata, while the previously retained
	// fitted Model has independent lifetime. Clearing the exported field must
	// not invalidate the Model or change its coefficients.
	originalModel := result.Model
	betaBefore, interceptBefore := originalModel.Coefficients()
	result.Model = nil
	result.Rank = 0
	result.ResidualSumSquares = 0
	result.ConditionEstimate = 0

	predicted, err := originalModel.Predict(context.Background(), x)
	if err != nil {
		t.Fatalf("retained Model.Predict: %v", err)
	}
	mustEqualInt(t, len(predicted), 4, "retained Model prediction count")
	for i, want := range []float64{3, 5, 7, 9} {
		// Householder QR and the original-unit FMA dot product can differ
		// by a few ULPs even when the underlying linear law is exact.
		mustNear(t, predicted[i], want, 1e-12)
	}
	betaAfter, interceptAfter := originalModel.Coefficients()
	mustEqualInt(t, len(betaAfter), len(betaBefore), "retained coefficient count")
	for j := range betaBefore {
		if betaAfter[j] != betaBefore[j] {
			t.Fatalf("retained coefficient %d changed: %g != %g", j, betaAfter[j], betaBefore[j])
		}
	}
	if interceptAfter != interceptBefore {
		t.Fatalf("retained intercept changed: %g != %g", interceptAfter, interceptBefore)
	}
}

func TestOLS_ExactSquareFullRankAndConstantPolicies(t *testing.T) {
	ctx := context.Background()
	// The boundary n=q is valid; a perfect two-point affine line has no
	// residual degrees of freedom but does not require invented t-statistics.
	x := mustDense(t, [][]float64{{-2}, {4}})
	r, err := ols.FitDefault(ctx, x, []float64{-3, 9})
	if err != nil {
		t.Fatal(err)
	}
	mustEqualInt(t, r.Rank, 2, "square rank")
	slopes, intercept := r.Model.Coefficients()
	mustNear(t, slopes[0], 2, 1e-12)
	mustNear(t, intercept, 1, 1e-12)
	mustNear(t, r.ResidualSumSquares, 0, 1e-12)

	constant := mustDense(t, [][]float64{{5}, {5}, {5}, {5}})
	if _, err = ols.FitStandardized(ctx, constant, []float64{2, 2, 2, 2}); !errors.Is(err, ols.ErrRankDeficient) {
		t.Fatalf("constant plus intercept must be deficient: %v", err)
	}
	origin, err := ols.FitThroughOriginStandardized(ctx, constant, []float64{2, 2, 2, 2})
	if err != nil {
		t.Fatal(err)
	}
	mustEqualInt(t, origin.Rank, 1, "origin-constant rank")
	beta, intercept := origin.Model.Coefficients()
	mustNear(t, beta[0], 0.4, 1e-12)
	mustEqualBool(t, intercept == 0 && !math.Signbit(intercept), true, "origin constant")
}

func TestOLS_ReadOnlyInputAndDetachedPredictions(t *testing.T) {
	data := [][]float64{{10, 1}, {12, 4}, {14, 2}, {16, 8}, {18, 5}, {20, 10}}
	responses := []float64{31, 40, 39, 52, 51, 64} // 5 + 2*x0 + 6*x1 (not exact on purpose).
	backup := append([]float64(nil), responses...)
	x := mustDense(t, data)
	r, err := ols.FitStandardized(context.Background(), x, responses)
	if err != nil {
		t.Fatal(err)
	}
	mustMatrixValues(t, x, data)
	for i, want := range backup {
		if responses[i] != want {
			t.Fatal("Fit mutated caller responses")
		}
	}
	a, err := r.Model.Predict(context.Background(), x)
	if err != nil {
		t.Fatal(err)
	}
	b, err := r.Model.Predict(context.Background(), x)
	if err != nil {
		t.Fatal(err)
	}
	if a[0] != b[0] {
		t.Fatal("predictions not deterministic")
	}
	a[0] += 9000
	c, err := r.Model.Predict(context.Background(), x)
	if err != nil || c[0] != b[0] {
		t.Fatalf("mutable output aliased Model: %v, %v", c, err)
	}
}

func TestOLS_TrainScalerDoesNotTouchHeldout(t *testing.T) {
	trainRows := [][]float64{{10, 1}, {12, 2}, {14, 4}, {16, 8}, {18, 16}, {20, 32}}
	y := []float64{22, 27, 33, 41, 53, 73} // Smooth but not manufactured as a perfect fit.
	model, err := ols.FitStandardized(context.Background(), mustDense(t, trainRows), y)
	if err != nil {
		t.Fatal(err)
	}
	beta, intercept := model.Model.Coefficients()
	holdRows := [][]float64{{1000, 300}, {2000, 400}, {3000, 500}}
	hold := mustDense(t, holdRows)
	first, err := model.Model.Predict(context.Background(), hold)
	if err != nil {
		t.Fatal(err)
	}
	for i, row := range holdRows {
		want := math.FMA(row[1], beta[1], math.FMA(row[0], beta[0], intercept))
		mustNear(t, first[i], want, 1e-11)
	}
	second, err := model.Model.Predict(context.Background(), hold)
	if err != nil {
		t.Fatal(err)
	}
	for i := range first {
		if first[i] != second[i] {
			t.Fatal("heldout evaluation refitted or mutated the model")
		}
	}
	mustMatrixValues(t, hold, holdRows)
}
