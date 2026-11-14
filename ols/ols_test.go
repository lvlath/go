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

// TestOLSPredictLostSmallTerm protects the small term that sequential
// multiply-add destroys between two large, cancelling terms.
func TestOLSPredictLostSmallTerm(t *testing.T) {
	x := mustDense(t, [][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}})
	fit, err := ols.FitThroughOrigin(context.Background(), x, []float64{1, 1, 1})
	if err != nil {
		t.Fatal(err)
	}
	beta, intercept := fit.Model.Coefficients()
	if intercept != 0 || math.Signbit(intercept) || beta[0] != 1 || beta[1] != 1 || beta[2] != 1 {
		t.Fatalf("identity fit altered exact model: beta=%v intercept=%g", beta, intercept)
	}
	row := []float64{1e16, 1, -1e16}
	out, err := fit.Model.Predict(context.Background(), mustDense(t, [][]float64{row}))
	if err != nil || len(out) != 1 || out[0] != 1 {
		t.Fatalf("lost-small-term prediction=%v, err=%v, want [1]", out, err)
	}
}

// TestOLSPredictRepresentableCancellation rejects an intermediate
// overflow as a reason to discard a representable final affine response.
func TestOLSPredictRepresentableCancellation(t *testing.T) {
	x := mustDense(t, [][]float64{{1, 0}, {0, 1}})
	fit, err := ols.FitThroughOrigin(context.Background(), x, []float64{10, -10})
	if err != nil {
		t.Fatal(err)
	}
	beta, _ := fit.Model.Coefficients()
	if beta[0] != 10 || beta[1] != -10 {
		t.Fatalf("identity model did not recover exact slopes: %v", beta)
	}
	row := []float64{math.MaxFloat64, math.MaxFloat64}
	out, err := fit.Model.Predict(context.Background(), mustDense(t, [][]float64{row}))
	if err != nil || len(out) != 1 || out[0] != 0 {
		t.Fatalf("false overflow: prediction=%v, err=%v, want [0]", out, err)
	}
}

// TestOLSPredictSubnormalProductCarry verifies that individually
// unrepresentable half-subnormal products are combined before rounding.
func TestOLSPredictSubnormalProductCarry(t *testing.T) {
	fit, err := ols.FitThroughOrigin(context.Background(),
		mustDense(t, [][]float64{{1, 0}, {0, 1}}), []float64{0.5, 0.5})
	if err != nil {
		t.Fatal(err)
	}
	row := []float64{math.SmallestNonzeroFloat64, math.SmallestNonzeroFloat64}
	out, err := fit.Model.Predict(context.Background(), mustDense(t, [][]float64{row}))
	if err != nil || len(out) != 1 || out[0] != math.SmallestNonzeroFloat64 {
		t.Fatalf("subnormal terms lost: prediction=%v, err=%v", out, err)
	}
}

// TestOLSPredictNonzeroInterceptOracle retains the fitted affine offset
// in the same exact sum as its numerically hostile feature products.
func TestOLSPredictNonzeroInterceptOracle(t *testing.T) {
	train := make([][]float64, 0, 8)
	y := make([]float64, 0, 8)
	for a := -1.0; a <= 1; a += 2 {
		for b := -1.0; b <= 1; b += 2 {
			for c := -1.0; c <= 1; c += 2 {
				train = append(train, []float64{a, b, c})
				y = append(y, 3+a+b+c)
			}
		}
	}
	fit, err := ols.FitDefault(context.Background(), mustDense(t, train), y)
	if err != nil {
		t.Fatal(err)
	}
	scenarios := [][]float64{
		{1e16, 1, -1e16},
		{-1e16, 1e16, 1},
		{math.MaxFloat64, -math.MaxFloat64, 2},
		{0.5, -1.5, 3.25},
	}
	out, err := fit.Model.Predict(context.Background(), mustDense(t, scenarios))
	if err != nil {
		t.Fatalf("representable oracle predictions rejected: %v", err)
	}
	for index, row := range scenarios {
		want := exactOLSResponse(t, fit.Model, row)
		if out[index] != want {
			// The compensated path is an approximation; require a small ULP budget
			// only when the independent exact oracle is non-exceptional.
			if !finiteForTest(out[index]) || !finiteForTest(want) ||
				math.Abs(out[index]-want) > 2*math.Abs(math.Nextafter(want, math.Inf(1))-want) {
				t.Fatalf("row %d: got %.17g, exact binary64 oracle %.17g", index, out[index], want)
			}
		}
	}
}

// TestOLSPredictTrueOverflow ensures the exact path does not turn a
// genuinely unrepresentable finished affine response into a false success.
func TestOLSPredictTrueOverflow(t *testing.T) {
	fit, err := ols.FitThroughOrigin(context.Background(),
		mustDense(t, [][]float64{{1, 0}, {0, 1}}), []float64{1, 1})
	if err != nil {
		t.Fatal(err)
	}
	out, err := fit.Model.Predict(context.Background(), mustDense(t,
		[][]float64{{math.MaxFloat64, math.MaxFloat64}}))
	if out != nil || !errors.Is(err, ols.ErrOverflow) {
		t.Fatalf("true overflow must publish nil/ErrOverflow, got %v/%v", out, err)
	}
}

// TestOLSPredictFinitePrecedence verifies that full-batch input validation
// wins over an earlier row that would overflow during arithmetic.
func TestOLSPredictFinitePrecedence(t *testing.T) {
	fit, err := ols.FitThroughOrigin(context.Background(),
		mustDense(t, [][]float64{{1, 0}, {0, 1}}), []float64{1, 1})
	if err != nil {
		t.Fatal(err)
	}
	x := &fixtureMatrix{
		rows: 2, cols: 2,
		values: []float64{math.MaxFloat64, math.MaxFloat64, 0, math.NaN()},
	}
	out, err := fit.Model.Predict(context.Background(), x)
	if out != nil || !errors.Is(err, ols.ErrNonFinite) {
		t.Fatalf("nonfinite validation precedence violated: %v/%v", out, err)
	}
}

// TestOLSPredictPermutedCancellation checks that feature order changes
// neither the fitted physical law nor a correctly rounded hostile prediction.
func TestOLSPredictPermutedCancellation(t *testing.T) {
	for _, row := range [][]float64{
		{1e16, 1, -1e16},
		{-1e16, 1e16, 1},
		{-1e16, 1, 1e16},
	} {
		fit, err := ols.FitThroughOrigin(context.Background(),
			mustDense(t, [][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}),
			[]float64{1, 1, 1})
		if err != nil {
			t.Fatal(err)
		}
		out, err := fit.Model.Predict(context.Background(), mustDense(t, [][]float64{row}))
		if err != nil || len(out) != 1 || out[0] != 1 {
			t.Fatalf("order %v: predicted %v, err=%v; want [1]", row, out, err)
		}
	}
}

// TestOLSPredictConcurrentExactFallback verifies that the same frozen
// model can allocate private exact-arithmetic scratch in 64 goroutines.
func TestOLSPredictConcurrentExactFallback(t *testing.T) {
	fit, err := ols.FitThroughOrigin(context.Background(),
		mustDense(t, [][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}),
		[]float64{1, 1, 1})
	if err != nil {
		t.Fatal(err)
	}
	hold := mustDense(t, [][]float64{
		{1e16, 1, -1e16},
		{math.MaxFloat64, 0, -math.MaxFloat64},
	})
	const workers = 64
	var wg sync.WaitGroup
	failures := make(chan error, workers)
	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			out, err := fit.Model.Predict(context.Background(), hold)
			if err != nil {
				failures <- err
				return
			}
			if len(out) != 2 || out[0] != 1 || out[1] != 0 {
				failures <- errors.New("concurrent exact predictions differ from [1, 0]")
			}
		}()
	}
	wg.Wait()
	close(failures)
	for failure := range failures {
		t.Error(failure)
	}
}

// TestOLSPredictZeroRowsAndDetachedOutput retains the documented
// dimension-preserving empty-batch behavior after introducing exact fallback.
func TestOLSPredictZeroRowsAndDetachedOutput(t *testing.T) {
	fit, err := ols.FitThroughOrigin(context.Background(),
		mustDense(t, [][]float64{{1}, {2}}), []float64{2, 4})
	if err != nil {
		t.Fatal(err)
	}
	zero, err := matrix.NewDense(0, 1)
	if err != nil {
		t.Fatal(err)
	}
	out, err := fit.Model.Predict(context.Background(), zero)
	if err != nil || out == nil || len(out) != 0 {
		t.Fatalf("zero-row inference: %v/%v", out, err)
	}
}

// TestOLSFitSSEUsesPublicPrediction checks that training diagnostics and
// heldout inference share the same compensated/exact row-level response law.
// It also distinguishes the unnormalized SSE from a QR residual L2 norm.
func TestOLSFitSSEUsesPublicPrediction(t *testing.T) {
	features := [][]float64{
		{-2, -1, 0}, {-1, 2, 1}, {0, 1, -2},
		{1, -2, 3}, {2, 3, -1}, {3, -1, 2},
		{4, 2, 0}, {5, -3, 1}, {6, 0, -2},
	}
	responses := []float64{-3.2, 1.4, 2.1, 0.7, 3.8, -0.3, 6.5, 3.2, 7.1}
	training := mustDense(t, features)
	fit, err := ols.FitStandardized(context.Background(), training, responses)
	if err != nil {
		t.Fatal(err)
	}
	predicted, err := fit.Model.Predict(context.Background(), training)
	if err != nil {
		t.Fatal(err)
	}
	independentSSE := 0.0
	for index, value := range predicted {
		residual := responses[index] - value
		independentSSE = math.FMA(residual, residual, independentSSE)
	}
	if independentSSE <= 0 {
		t.Fatal("fixture unexpectedly has no training residuals")
	}
	mustNear(t, fit.ResidualSumSquares, independentSSE, 1e-12)
	for index, row := range features {
		oracle := exactOLSResponse(t, fit.Model, row)
		mustNear(t, predicted[index], oracle, 2e-13)
	}
}
