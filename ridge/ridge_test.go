// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ridge_test tests the public interface and original-unit invariants.
package ridge_test

import (
	"context"
	"errors"
	"math"
	"sync"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/ridge"
)

func TestRidge_ScalarAnalyticOracle(t *testing.T) {
	x := dense(t, [][]float64{{1}, {2}})
	r, err := ridge.Fit(context.Background(), x, []float64{1, 2}, ridge.Options{Lambda: 1, NoIntercept: true})
	if err != nil {
		t.Fatal(err)
	}
	b, _, p, o := scalarOracle(t, []int64{1, 2}, []int64{1, 2}, 1)
	coeff, intercept := r.Model.Coefficients()
	v, _ := b.Float64()
	closeTo(t, coeff[0], v)
	closeTo(t, intercept, 0)
	closeTo(t, r.ResidualSumSquares, 20.0/49)
	closeTo(t, r.Penalty, 25.0/98)
	w, _ := o.Float64()
	closeTo(t, r.Objective, w)
	w, _ = p.Float64()
	closeTo(t, r.Penalty, w)
}

func TestRidge_InterceptUnpenalized(t *testing.T) {
	x := dense(t, [][]float64{{-1}, {1}})
	r, err := ridge.Fit(context.Background(), x, []float64{4, 4}, ridge.Options{Lambda: 1e6})
	if err != nil {
		t.Fatal(err)
	}
	slopes, b := r.Model.Coefficients()
	closeTo(t, slopes[0], 0)
	closeTo(t, b, 4)
	closeTo(t, r.Objective, 0)
	r, err = ridge.Fit(context.Background(), x, []float64{-1, 1}, ridge.Options{Lambda: 1})
	if err != nil {
		t.Fatal(err)
	}
	slopes, b = r.Model.Coefficients()
	closeTo(t, slopes[0], .5)
	closeTo(t, b, 0)
	closeTo(t, r.ResidualSumSquares, .5)
	closeTo(t, r.Penalty, .125)
	closeTo(t, r.Objective, .25)
}
func TestRidge_CollinearFeaturesSolvable(t *testing.T) {
	r, err := ridge.Fit(context.Background(), dense(t, [][]float64{{1, 1}, {2, 2}, {3, 3}, {4, 4}}), []float64{2, 4, 6, 8}, ridge.Options{Lambda: .2})
	if err != nil {
		t.Fatal(err)
	}
	beta, _ := r.Model.Coefficients()
	closeTo(t, beta[0], beta[1])
}
func TestRidge_WideMatrixRecovery(t *testing.T) {
	r, err := ridge.Fit(context.Background(), dense(t, [][]float64{{1, 0, 1, 2, 0}, {0, 1, 0, 1, 2}}), []float64{3, 4}, ridge.Options{Lambda: 1})
	if err != nil {
		t.Fatal(err)
	}
	beta, _ := r.Model.Coefficients()
	if len(beta) != 5 {
		t.Fatalf("len=%d", len(beta))
	}
	if r.Objective < 0 || !isFinite(r.Objective) {
		t.Fatal(r.Objective)
	}
}
func TestRidge_LambdaZeroRejected(t *testing.T) {
	for _, lambda := range []float64{0, -1, math.NaN(), math.Inf(1), math.Inf(-1)} {
		result, err := ridge.Fit(context.Background(), nil, nil, ridge.Options{Lambda: lambda})
		if result != nil {
			t.Fatal("partial result")
		}
		errorIs(t, err, ridge.ErrInvalidOption)
	}
}
func TestRidge_FitResult_IsNilInterface(t *testing.T) {
	var absent *ridge.FitResult
	var n core.Nilable = absent
	mustNilState(t, n, true, "typed nil FitResult")
	n = &ridge.FitResult{}
	mustNilState(t, n, false, "allocated FitResult")
}
func TestRidge_Predict_TrainScalerFrozen(t *testing.T) {
	x := dense(t, [][]float64{{2, 4}, {4, 8}, {6, 12}, {8, 16}})
	r, err := ridge.Fit(context.Background(), x, []float64{3, 5, 7, 9}, ridge.Options{Lambda: .1, Standardize: true})
	if err != nil {
		t.Fatal(err)
	}
	predX := dense(t, [][]float64{{10, 20}, {12, 24}})
	a, err := r.Model.Predict(context.Background(), predX)
	if err != nil {
		t.Fatal(err)
	}
	_ = x.Set(0, 0, 9999)
	b, err := r.Model.Predict(context.Background(), predX)
	if err != nil {
		t.Fatal(err)
	}
	for i := range a {
		closeTo(t, a[i], b[i])
	}
	beta, intercept := r.Model.Coefficients()
	for i := range a {
		closeTo(t, a[i], intercept+float64(10+2*i)*beta[0]+float64(20+4*i)*beta[1])
	}
}
func TestRidge_NoInterceptScaleOnly(t *testing.T) {
	r, err := ridge.Fit(context.Background(), dense(t, [][]float64{{10}, {20}, {30}}), []float64{10, 20, 30}, ridge.Options{Lambda: .2, NoIntercept: true, Standardize: true})
	if err != nil {
		t.Fatal(err)
	}
	_, b := r.Model.Coefficients()
	if b != 0 {
		t.Fatal("implicit intercept", b)
	}
	p, err := r.Model.Predict(context.Background(), dense(t, [][]float64{{0}, {40}}))
	if err != nil {
		t.Fatal(err)
	}
	closeTo(t, p[0], 0)
}
func TestRidge_CoefficientsDetached(t *testing.T) {
	r, err := ridge.Fit(context.Background(), dense(t, [][]float64{{1}, {2}, {3}}), []float64{2, 4, 6}, ridge.Options{Lambda: 1, NoIntercept: true})
	if err != nil {
		t.Fatal(err)
	}
	a, _ := r.Model.Coefficients()
	a[0] = 999
	b, _ := r.Model.Coefficients()
	if b[0] == 999 {
		t.Fatal("model aliases returned coefficients")
	}
}
func TestRidge_Validation(t *testing.T) {
	ctx := context.Background()
	x := dense(t, [][]float64{{1}, {2}})
	cases := []struct {
		name string
		ctx  context.Context
		x    matrix.Matrix
		y    []float64
		opts ridge.Options
		want error
	}{
		{"nil context", nil, x, []float64{1, 2}, ridge.Options{Lambda: 1}, ridge.ErrInvalidOption},
		{"nil X", ctx, nil, []float64{1, 2}, ridge.Options{Lambda: 1}, ridge.ErrNilInput},
		{"nil y", ctx, x, nil, ridge.Options{Lambda: 1}, ridge.ErrNilInput},
		{"empty y", ctx, x, []float64{}, ridge.Options{Lambda: 1}, ridge.ErrEmptyInput},
		{"shape", ctx, x, []float64{1}, ridge.Options{Lambda: 1}, ridge.ErrShape},
		{"invalid rank", ctx, x, []float64{1, 2}, ridge.Options{Lambda: 1, RankTolerance: math.NaN()}, ridge.ErrInvalidOption},
		{"nonfinite response", ctx, x, []float64{1, math.Inf(1)}, ridge.Options{Lambda: 1}, ridge.ErrNonFinite},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r, err := ridge.Fit(tc.ctx, tc.x, tc.y, tc.opts)
			if r != nil {
				t.Fatal("partial result")
			}
			errorIs(t, err, tc.want)
		})
	}
	c, cancel := context.WithCancel(ctx)
	cancel()
	r, err := ridge.Fit(c, nil, nil, ridge.Options{})
	if r != nil {
		t.Fatal("partial")
	}
	errorIs(t, err, context.Canceled)
}
func TestRidge_ConcurrentPredictRace(t *testing.T) {
	r, err := ridge.Fit(context.Background(), dense(t, [][]float64{{1, 2}, {2, 3}, {3, 4}, {4, 5}}), []float64{2, 3, 4, 5}, ridge.Options{Lambda: 1})
	if err != nil {
		t.Fatal(err)
	}
	x := dense(t, [][]float64{{5, 6}, {7, 8}})
	var wg sync.WaitGroup
	errCh := make(chan error, 64)
	for i := 0; i < 64; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); _, e := r.Model.Predict(context.Background(), x); errCh <- e }()
	}
	wg.Wait()
	close(errCh)
	for e := range errCh {
		if e != nil {
			t.Fatal(e)
		}
	}
}
func TestRidge_PredictNilAndEmpty(t *testing.T) {
	var m *ridge.Model
	_, err := m.Predict(context.Background(), nil)
	errorIs(t, err, ridge.ErrNilInput)
	r, err := ridge.Fit(context.Background(), dense(t, [][]float64{{1}, {2}}), []float64{1, 2}, ridge.Options{Lambda: 1})
	if err != nil {
		t.Fatal(err)
	}
	empty, err := matrix.NewDense(0, 1)
	if err != nil {
		t.Fatal(err)
	}
	pred, err := r.Model.Predict(context.Background(), empty)
	if err != nil {
		t.Fatal(err)
	}
	if pred == nil || len(pred) != 0 {
		t.Fatal("detached empty")
	}
	_, err = r.Model.Predict(context.Background(), dense(t, [][]float64{{1, 2}}))
	errorIs(t, err, ridge.ErrShape)
}
func TestRidge_AugmentedWeightOverflowProtection(t *testing.T) {
	// n=2 and lambda=MaxFloat64: n*lambda overflows although sqrt(n*lambda) is finite.
	lambdaMax := math.MaxFloat64
	w := math.Sqrt(2) * math.Sqrt(lambdaMax)
	if !isFinite(w) || !math.IsInf(2*lambdaMax, 1) {
		t.Fatal("fixture invalid")
	}
	// Solver may classify the condition as deficient; it must not silently fall back.
	r, err := ridge.Fit(context.Background(), dense(t, [][]float64{{1}, {2}}), []float64{1, 2}, ridge.Options{Lambda: math.MaxFloat64, NoIntercept: true})
	if err != nil && !errors.Is(err, ridge.ErrRankDeficient) && !errors.Is(err, ridge.ErrOverflow) && !errors.Is(err, matrix.ErrIllConditioned) {
		t.Fatal(err)
	}
	if err != nil && r != nil {
		t.Fatal("partial result")
	}
}
func isFinite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

// TestRidge_ReplicationInvariant protects the 1/(2*n) normalization: duplicating
// the complete observation set must not change optimum at fixed lambda.
func TestRidge_ReplicationInvariant(t *testing.T) {
	base := dense(t, [][]float64{{1, 2}, {2, 3}, {4, 1}, {5, 2}})
	repeated := dense(t, [][]float64{{1, 2}, {2, 3}, {4, 1}, {5, 2}, {1, 2}, {2, 3}, {4, 1}, {5, 2}})
	opts := ridge.Options{Lambda: 0.4, Standardize: true}
	first, err := ridge.Fit(context.Background(), base, []float64{2, 3, 5, 6}, opts)
	if err != nil {
		t.Fatal(err)
	}
	second, err := ridge.Fit(context.Background(), repeated, []float64{2, 3, 5, 6, 2, 3, 5, 6}, opts)
	if err != nil {
		t.Fatal(err)
	}
	b1, a1 := first.Model.Coefficients()
	b2, a2 := second.Model.Coefficients()
	closeTo(t, a1, a2)
	for j := range b1 {
		closeTo(t, b1[j], b2[j])
	}
	closeTo(t, first.Objective, second.Objective)
	closeTo(t, 2*first.ResidualSumSquares, second.ResidualSumSquares)
}

// TestRidge_StandardizeManualEquivalence independently constructs the
// DDOF=0 transformed design and verifies the penalty coordinate convention.
func TestRidge_StandardizeManualEquivalence(t *testing.T) {
	x := dense(t, [][]float64{{1}, {2}, {3}, {4}})
	y := []float64{1, 2, 3, 4}
	// Population mean 2.5, population variance 5/4.
	scale := math.Sqrt(1.25)
	transformed := dense(t, [][]float64{{-1.5 / scale}, {-0.5 / scale}, {0.5 / scale}, {1.5 / scale}})
	a, err := ridge.Fit(context.Background(), x, y, ridge.Options{Lambda: 1, Standardize: true})
	if err != nil {
		t.Fatal(err)
	}
	b, err := ridge.Fit(context.Background(), transformed, y, ridge.Options{Lambda: 1})
	if err != nil {
		t.Fatal(err)
	}
	originalCoefficients, originalIntercept := a.Model.Coefficients()
	theta, normalizedIntercept := b.Model.Coefficients()
	closeTo(t, originalCoefficients[0], theta[0]/scale)
	closeTo(t, originalIntercept, normalizedIntercept-2.5*theta[0]/scale)
	closeTo(t, a.Penalty, b.Penalty)
	closeTo(t, a.Objective, b.Objective)
}

// TestRidge_NilModelCoefficients validates the nil-receiver surface.
func TestRidge_NilModelCoefficients(t *testing.T) {
	var model *ridge.Model
	beta, b := model.Coefficients()
	if beta != nil || b != 0 {
		t.Fatalf("unexpected coefficients %v, %g", beta, b)
	}
}
