// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package ols_test

import (
	"context"
	"errors"
	"math"
	"sync"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/ols"
)

// TestOLS_ExactLinearRecovery checks Exact coefficients and zero SSE follow from an affine one-factor training fixture.
func TestOLS_ExactLinearRecovery(t *testing.T) {
	x := mustDense(t, [][]float64{{1}, {2}, {3}, {4}, {5}})
	r, err := ols.Fit(context.Background(), x, []float64{5, 8, 11, 14, 17}, ols.Options{})
	if err != nil {
		t.Fatal(err)
	}
	b, i := r.Model.Coefficients()
	mustNear(t, b[0], 3, 1e-12)
	mustNear(t, i, 2, 1e-12)
	mustNear(t, r.ResidualSumSquares, 0, 1e-12)
	if r.Rank != 2 {
		t.Fatalf("rank %d", r.Rank)
	}
}

// TestOLS_CollinearRejected checks Two identical predictors must retain both the OLS and matrix QR rank sentinels.
func TestOLS_CollinearRejected(t *testing.T) {
	x := mustDense(t, [][]float64{{1, 1}, {2, 2}, {3, 3}, {4, 4}})
	r, err := ols.Fit(context.Background(), x, []float64{3, 5, 7, 9}, ols.Options{})
	if r != nil {
		t.Fatal("partial result")
	}
	mustErrorIs(t, err, ols.ErrRankDeficient)
	if !errors.Is(err, matrix.ErrRankDeficient) {
		t.Fatal("lost underlying rank sentinel")
	}
}

// TestOLS_NoInterceptStandardize_NoImplicitConstant checks Scale-only training preserves physical positive-zero intercept at inference.
func TestOLS_NoInterceptStandardize_NoImplicitConstant(t *testing.T) {
	x := mustDense(t, [][]float64{{1}, {2}, {3}, {4}})
	r, err := ols.Fit(context.Background(), x, []float64{2, 4, 6, 8}, ols.Options{NoIntercept: true, Standardize: true})
	if err != nil {
		t.Fatal(err)
	}
	b, i := r.Model.Coefficients()
	if i != 0 || math.Signbit(i) {
		t.Fatalf("intercept %v", i)
	}
	mustNear(t, b[0], 2, 1e-12)
}

// TestPredict_TrainScalerFrozen checks A distant inference distribution must not change the already-fitted coefficients.
func TestPredict_TrainScalerFrozen(t *testing.T) {
	train := mustDense(t, [][]float64{{10, 1}, {12, 2}, {14, 4}, {16, 8}, {18, 16}, {20, 32}})
	y := []float64{22, 27, 33, 41, 53, 73} // Paired training responses with nonzero residuals.
	r, err := ols.Fit(context.Background(), train, y, ols.Options{Standardize: true})
	if err != nil {
		t.Fatal(err)
	}
	hold := mustDense(t, [][]float64{{1000, 300}, {2000, 400}})
	a, err := r.Model.Predict(context.Background(), hold)
	if err != nil {
		t.Fatal(err)
	}
	b, i := r.Model.Coefficients()
	for row := 0; row < 2; row++ {
		v0, err := hold.At(row, 0)
		if err != nil {
			t.Fatal(err)
		}
		v1, err := hold.At(row, 1)
		if err != nil {
			t.Fatal(err)
		}
		mustNear(t, a[row], i+b[0]*v0+b[1]*v1, 1e-11)
	}
	again, err := r.Model.Predict(context.Background(), hold)
	if err != nil {
		t.Fatal(err)
	}
	for j := range a {
		if a[j] != again[j] {
			t.Fatal("prediction drift")
		}
	}
}

// TestCoefficients_Detached checks Mutating returned coefficients must never corrupt the trained Model.
func TestCoefficients_Detached(t *testing.T) {
	x := mustDense(t, [][]float64{{1}, {2}, {3}})
	r, err := ols.Fit(context.Background(), x, []float64{5, 8, 11}, ols.Options{})
	if err != nil {
		t.Fatal(err)
	}
	a, _ := r.Model.Coefficients()
	a[0] = 123456
	b, _ := r.Model.Coefficients()
	mustNear(t, b[0], 3, 1e-12)
}

// TestFitResult_IsNilInterface checks Typed-nil results and model receivers obey the library nil-state contract.
func TestFitResult_IsNilInterface(t *testing.T) {
	var p *ols.FitResult
	var n core.Nilable = p
	if !n.IsNil() {
		t.Fatal("typed nil result")
	}
	n = &ols.FitResult{}
	if n.IsNil() {
		t.Fatal("non-nil empty result")
	}
	var m *ols.Model
	b, i := m.Coefficients()
	if b != nil || i != 0 {
		t.Fatal("nil getter")
	}
	_, err := m.Predict(context.Background(), mustDense(t, [][]float64{{1}}))
	mustErrorIs(t, err, ols.ErrNilInput)
}

// TestModel_ConcurrentPredictRace checks Many concurrent readers share immutable parameters and isolated output slices.
func TestModel_ConcurrentPredictRace(t *testing.T) {
	x := mustDense(t, [][]float64{{1}, {2}, {3}, {4}})
	r, err := ols.Fit(context.Background(), x, []float64{5, 8, 11, 14}, ols.Options{})
	if err != nil {
		t.Fatal(err)
	}
	errs := make(chan error, 64)
	var wg sync.WaitGroup
	for worker := 0; worker < 64; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			pred, e := r.Model.Predict(context.Background(), x)
			if e == nil && len(pred) != 4 {
				e = ols.ErrShape
			}
			errs <- e
		}()
	}
	wg.Wait()
	close(errs)
	for e := range errs {
		if e != nil {
			t.Fatal(e)
		}
	}
}

// TestOLS_ValidationAndSpecial checks Invalid arguments obey sentinel precedence and constant-feature rank rules.
func TestOLS_ValidationAndSpecial(t *testing.T) {
	x := mustDense(t, [][]float64{{1}, {2}, {3}})
	y := []float64{1, 2, 3}
	cases := []struct {
		name string
		ctx  context.Context
		x    matrix.Matrix
		y    []float64
		opts ols.Options
		want error
	}{
		{"nil context", nil, x, y, ols.Options{}, ols.ErrInvalidOption},
		{"options first", context.Background(), nil, nil, ols.Options{RankTolerance: -1}, ols.ErrInvalidOption},
		{"nil x", context.Background(), nil, y, ols.Options{}, ols.ErrNilInput},
		{"nil y", context.Background(), x, nil, ols.Options{}, ols.ErrNilInput},
		{"shape", context.Background(), x, []float64{1, 2}, ols.Options{}, ols.ErrShape},
		{"nonfinite y", context.Background(), x, []float64{1, math.Inf(1), 3}, ols.Options{}, ols.ErrNonFinite},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			r, e := ols.Fit(tt.ctx, tt.x, tt.y, tt.opts)
			if r != nil {
				t.Fatal("partial result")
			}
			mustErrorIs(t, e, tt.want)
		})
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, e := ols.Fit(ctx, x, y, ols.Options{})
	mustErrorIs(t, e, context.Canceled)
	constant := mustDense(t, [][]float64{{7}, {7}, {7}})
	_, e = ols.Fit(context.Background(), constant, y, ols.Options{Standardize: true})
	mustErrorIs(t, e, ols.ErrRankDeficient)
	r, e := ols.Fit(context.Background(), constant, []float64{2, 2, 2}, ols.Options{NoIntercept: true, Standardize: true})
	if e != nil {
		t.Fatal(e)
	}
	b, i := r.Model.Coefficients()
	mustNear(t, b[0], 2.0/7, 1e-12)
	if i != 0 {
		t.Fatal(i)
	}
	empty, e := matrix.NewDense(0, 1)
	if e != nil {
		t.Fatal(e)
	}
	pred, e := r.Model.Predict(context.Background(), empty)
	if e != nil || pred == nil || len(pred) != 0 {
		t.Fatalf("empty inference: %v %v", pred, e)
	}
}

// TestOLS_RowAndColumnPermutation checks Permutation of training observations/features preserves the model mathematics.
func TestOLS_RowAndColumnPermutation(t *testing.T) {
	x := mustDense(t, [][]float64{{1, 2}, {2, 1}, {3, 5}, {4, 2}, {5, 7}, {6, 4}})
	y := []float64{10, 10, 25, 19, 37, 31} // 1+3*x0+3*x1
	first, err := ols.Fit(context.Background(), x, y, ols.Options{Standardize: true})
	if err != nil {
		t.Fatal(err)
	}
	xp, err := matrix.NewDense(6, 2)
	if err != nil {
		t.Fatal(err)
	}
	yp := make([]float64, 6)
	for i := 0; i < 6; i++ {
		original := 5 - i
		yp[i] = y[original]
		v0, err := x.At(original, 1)
		if err != nil {
			t.Fatal(err)
		}
		v1, err := x.At(original, 0)
		if err != nil {
			t.Fatal(err)
		}
		if err = xp.Set(i, 0, v0); err != nil {
			t.Fatal(err)
		}
		if err = xp.Set(i, 1, v1); err != nil {
			t.Fatal(err)
		}
	}
	second, err := ols.Fit(context.Background(), xp, yp, ols.Options{Standardize: true})
	if err != nil {
		t.Fatal(err)
	}
	a, ia := first.Model.Coefficients()
	b, ib := second.Model.Coefficients()
	mustNear(t, ia, ib, 1e-11)
	mustNear(t, a[0], b[1], 1e-11)
	mustNear(t, a[1], b[0], 1e-11)
	mustNear(t, first.ResidualSumSquares, second.ResidualSumSquares, 1e-10)
}

// TestOLS_OverflowSSERejected checks An unrepresentable residual sum of squares cannot publish a partial fit.
func TestOLS_OverflowSSERejected(t *testing.T) {
	x := mustDense(t, [][]float64{{0}, {1}, {2}, {3}})
	// A large response contrast can have representable LS coefficients yet
	// produce an unrepresentable sum of squared residuals.
	r, err := ols.Fit(context.Background(), x, []float64{1e200, -1e200, -1e200, 1e200}, ols.Options{})
	if r != nil {
		t.Fatal("published unrepresentable SSE")
	}
	mustErrorIs(t, err, ols.ErrOverflow)
}
