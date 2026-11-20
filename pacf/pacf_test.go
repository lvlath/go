// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pacf_test checks sample estimands, boundary errors, and ownership.
// It exercises only the public PACF facade and never reaches into the private kernel.
package pacf_test

import (
	"context"
	"errors"
	"math"
	"sync"
	"testing"

	"github.com/lvlath/go/pacf"
)

// TestPACF01_Centered The centered [1,2,3] fixture anchors the exact gamma and Durbin-Levinson signs.
// The second lag must equal -1/2 rather than a pair-adjusted or overwritten value.
func TestPACF01_Centered(t *testing.T) {
	r, e := pacf.Compute(context.Background(), []float64{1, 2, 3}, pacf.Options{MaxLag: 2, Center: true})
	if e != nil {
		t.Fatal(e)
	}
	mustSlices(t, r.Values, []float64{1, 0, -0.5}, 1e-14)
	mustSlices(t, r.PredictionVariance, []float64{2.0 / 3, 2.0 / 3, 0.5}, 1e-14)
	mustEqualInt(t, r.Count, 3, "sample count")
}

// TestPACF02_UncenteredConstant A nonzero constant preserves a positive raw second moment.
// The common n denominator makes lag-one correlation exactly one half for n=2.
func TestPACF02_UncenteredConstant(t *testing.T) {
	r, e := pacf.Compute(context.Background(), []float64{1, 1}, pacf.Options{MaxLag: 1})
	if e != nil {
		t.Fatal(e)
	}
	mustSlices(t, r.Values, []float64{1, .5}, 1e-14)
	mustSlices(t, r.PredictionVariance, []float64{1, .75}, 1e-14)
}

// TestPACF03_ZeroCentered Any constant centered window has exactly zero energy after mean removal.
// The result must be nil and the error must retain its dedicated sentinel.
func TestPACF03_ZeroCentered(t *testing.T) {
	for _, v := range [][]float64{{1, 1}, {2, 2, 2}, {0, 0}} {
		r, e := pacf.Compute(context.Background(), v, pacf.Options{Center: true})
		mustNilState(t, r, true, "constant")
		mustErrorIs(t, e, pacf.ErrZeroVariance)
	}
}

// TestPACF04_Singleton One nonzero uncentered observation supports only lag zero.
// Count, anchored coefficient, and physical prediction variance are exact.
func TestPACF04_Singleton(t *testing.T) {
	r, e := pacf.Compute(context.Background(), []float64{2}, pacf.Options{})
	if e != nil {
		t.Fatal(e)
	}
	mustSlices(t, r.Values, []float64{1}, 0)
	mustSlices(t, r.PredictionVariance, []float64{4}, 0)
	mustEqualInt(t, r.Count, 1, "count")
}

// TestPACF06_Scale Nonzero signed scaling preserves PACF and scales prediction variance quadratically.
// Both centered and raw-second-moment estimands must obey this law.
func TestPACF06_Scale(t *testing.T) {
	x := []float64{3, -2, 5, 1, -4, 2, 8, -1, 0, 4, -3, 2}
	y := make([]float64, len(x))
	for i, v := range x {
		y[i] = -8 * v
	}
	for _, center := range []bool{false, true} {
		a, e := pacf.Compute(context.Background(), x, pacf.Options{MaxLag: 6, Center: center})
		if e != nil {
			t.Fatal(e)
		}
		b, e := pacf.Compute(context.Background(), y, pacf.Options{MaxLag: 6, Center: center})
		if e != nil {
			t.Fatal(e)
		}
		mustSlices(t, a.Values, b.Values, 1e-12)
		for i := range a.PredictionVariance {
			if !closeEnough(b.PredictionVariance[i], 64*a.PredictionVariance[i], 1e-12) {
				t.Fatalf("variance lag %d", i)
			}
		}
	}
}

// TestPACF07_Offset An exactly representable large additive offset does not alter centered PACF.
// A single stable fitted mean must serve every lag.
func TestPACF07_Offset(t *testing.T) {
	x := []float64{4, -3, 5, 2, -6, 8, 1, 3, -2, 7, -1, 0, 6, -4, 2}
	y := make([]float64, len(x))
	for i, v := range x {
		y[i] = v + 1e9
	}
	a, e := pacf.Compute(context.Background(), x, pacf.Options{MaxLag: 9, Center: true})
	if e != nil {
		t.Fatal(e)
	}
	b, e := pacf.Compute(context.Background(), y, pacf.Options{MaxLag: 9, Center: true})
	if e != nil {
		t.Fatal(e)
	}
	mustSlices(t, b.Values, a.Values, 2e-12)
	mustSlices(t, b.PredictionVariance, a.PredictionVariance, 2e-12)
}

// TestPACF08_ValidationPrecedence Invalid contexts, slices, shapes, and numbers are classified by errors.Is.
// Every failure must return a nil result without partially exposed arrays.
func TestPACF08_ValidationPrecedence(t *testing.T) {
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	cases := []struct {
		name string
		ctx  context.Context
		x    []float64
		opts pacf.Options
		want error
	}{
		{"nilContext", nil, nil, pacf.Options{}, pacf.ErrInvalidOption},
		{"canceled", canceled, nil, pacf.Options{}, context.Canceled},
		{"nilInput", context.Background(), nil, pacf.Options{}, pacf.ErrNilInput},
		{"empty", context.Background(), []float64{}, pacf.Options{}, pacf.ErrEmptyInput},
		{"negativeLag", context.Background(), []float64{1, 2}, pacf.Options{MaxLag: -1}, pacf.ErrInvalidOption},
		{"lagTooLarge", context.Background(), []float64{1, 2}, pacf.Options{MaxLag: 2}, pacf.ErrInvalidOption},
		{"nan", context.Background(), []float64{math.NaN()}, pacf.Options{}, pacf.ErrNonFinite},
		{"positiveInfinity", context.Background(), []float64{math.Inf(1)}, pacf.Options{}, pacf.ErrNonFinite},
		{"negativeInfinity", context.Background(), []float64{math.Inf(-1)}, pacf.Options{}, pacf.ErrNonFinite},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r, e := pacf.Compute(tc.ctx, tc.x, tc.opts)
			mustNilState(t, r, true, "error result")
			mustErrorIs(t, e, tc.want)
		})
	}
}

// TestResult_IsNilInterface A typed nil pointer inside an interface must be classified as absent.
// An allocated zero-value Result is present even though its slices are nil.
func TestResult_IsNilInterface(t *testing.T) {
	var r *pacf.Result
	var nilable interface{ IsNil() bool } = r
	mustEqualBool(t, nilable.IsNil(), true, "typed nil")
	r = &pacf.Result{}
	mustEqualBool(t, r.IsNil(), false, "allocated zero result")
}

// TestResult_OwnershipAndInput Compute must not mutate or retain the input window or alias results across calls.
// Mutating public slices from one call must not affect independently computed results.
func TestResult_OwnershipAndInput(t *testing.T) {
	x := []float64{5, 1, -2, 7, 2, 4, 8, -1, 3}
	before := append([]float64(nil), x...)
	a, e := pacf.Compute(context.Background(), x, pacf.Options{MaxLag: 4, Center: true})
	if e != nil {
		t.Fatal(e)
	}
	b, e := pacf.Compute(context.Background(), x, pacf.Options{MaxLag: 4, Center: true})
	if e != nil {
		t.Fatal(e)
	}
	for i := range x {
		if x[i] != before[i] {
			t.Fatal("mutated input")
		}
	}
	a.Values[0] = 123
	a.PredictionVariance[0] = 234
	if b.Values[0] != 1 || b.PredictionVariance[0] == 234 {
		t.Fatal("aliased outputs")
	}
}

// TestPACF_ConcurrentImmutableInput Independent calls may concurrently read a shared immutable input window.
// Worker errors flow through a channel and are asserted by the owning test goroutine.
func TestPACF_ConcurrentImmutableInput(t *testing.T) {
	x := make([]float64, 400)
	for i := range x {
		x[i] = math.Sin(float64(i)*.13) + .2*math.Cos(float64(i)*.09)
	}
	var wg sync.WaitGroup
	errs := make(chan error, 16)
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r, e := pacf.Compute(context.Background(), x, pacf.Options{MaxLag: 30, Center: true})
			if e != nil {
				errs <- e
				return
			}
			if r.Count != 400 {
				errs <- errors.New("invalid count")
			}
		}()
	}
	wg.Wait()
	close(errs)
	for e := range errs {
		t.Fatal(e)
	}
}

// TestPACF_VarianceMonotonicity Every successful recursive prediction variance remains positive and nonincreasing.
// Reflection coefficients must stay strictly between -1 and 1 after lag zero.
func TestPACF_VarianceMonotonicity(t *testing.T) {
	x := make([]float64, 120)
	for i := range x {
		x[i] = math.Sin(float64(i)*.19) + .3*math.Cos(float64(i)*.37)
	}
	r, e := pacf.Compute(context.Background(), x, pacf.Options{MaxLag: 20, Center: true})
	if e != nil {
		t.Fatal(e)
	}
	for k := 1; k < len(r.Values); k++ {
		if math.Abs(r.Values[k]) >= 1 || r.PredictionVariance[k] <= 0 || r.PredictionVariance[k] > r.PredictionVariance[k-1]*(1+1e-13) {
			t.Fatalf("lag %d", k)
		}
	}
}

// TestPACF_ExtremeScale Representable large and small signal magnitudes preserve positive finite variance.
// The estimator must scale before squaring to avoid intermediate overflow.
func TestPACF_ExtremeScale(t *testing.T) {
	for _, v := range []float64{1e154, 1e-155} {
		x := []float64{v, v, -v, -v, v, v, -v, -v, v}
		r, e := pacf.Compute(context.Background(), x, pacf.Options{MaxLag: 3})
		if e != nil {
			t.Fatalf("scale %g: %v", v, e)
		}
		for _, z := range r.PredictionVariance {
			if z <= 0 || math.IsNaN(z) || math.IsInf(z, 0) {
				t.Fatalf("invalid variance %g", z)
			}
		}
	}
}

// TestPACF_UnrepresentableVariance Finite observations can have an unrepresentable physical second moment.
// Such a domain-valid input must fail with ErrOverflow instead of ErrNonFinite.
func TestPACF_UnrepresentableVariance(t *testing.T) {
	r, e := pacf.Compute(context.Background(), []float64{1e200, 1e200}, pacf.Options{})
	mustNilState(t, r, true, "overflow")
	mustErrorIs(t, e, pacf.ErrOverflow)
}
