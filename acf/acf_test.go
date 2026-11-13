// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package acf_test anchors the public ACF validation, mathematical identity, and ownership laws.
// Its tests use only exported acf APIs, without coupling to private numerical helpers.
package acf_test

import (
	"errors"
	"fmt"
	"math"
	"sync"
	"testing"

	"github.com/lvlath/go/acf"
)

// TestCompute_ExactFixtures anchors the published common/adjusted estimator vectors.
// Exact lag-zero identity, pair counts, and mean metadata are checked on each case.
func TestCompute_ExactFixtures(t *testing.T) {
	cases := []struct {
		name string
		v    []float64
		lag  int
		opts acf.Options
		want []float64
		mean float64
	}{
		{"trend-common", []float64{1, 2, 3}, 2, acf.Options{}, []float64{1, 0, -.5}, 2},
		{"trend-adjusted", []float64{1, 2, 3}, 2, acf.Options{Denominator: acf.LagAdjusted}, []float64{1, 0, -1.5}, 2},
		{"alternating-common", []float64{1, -1, 1, -1}, 3, acf.Options{}, []float64{1, -.75, .5, -.25}, 0},
		{"alternating-adjusted", []float64{1, -1, 1, -1}, 3, acf.Options{Denominator: acf.LagAdjusted}, []float64{1, -1, 1, -1}, 0},
		{"constant-raw-common", []float64{2, 2, 2}, 2, acf.Options{NoCenter: true}, []float64{1, 2.0 / 3, 1.0 / 3}, 0},
		{"constant-raw-adjusted", []float64{2, 2, 2}, 2, acf.Options{NoCenter: true, Denominator: acf.LagAdjusted}, []float64{1, 1, 1}, 0},
		{"singleton-raw", []float64{1}, 0, acf.Options{NoCenter: true}, []float64{1}, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r, e := acf.Compute(tc.v, tc.lag, tc.opts)
			if e != nil {
				t.Fatal(e)
			}
			mustValues(t, r.Values, tc.want)
			mustClose(t, r.Mean, tc.mean)
			mustPairCounts(t, r, len(tc.v))
			mustEqualInt(t, int(r.Denominator), int(tc.opts.Denominator), "denominator")
			if r.Values[0] != 1 {
				t.Fatal("lag0")
			}
		})
	}
	for _, tc := range []struct {
		v    []float64
		lag  int
		opts acf.Options
	}{{[]float64{2, 2, 2}, 2, acf.Options{}}, {[]float64{1}, 0, acf.Options{}}, {[]float64{0}, 0, acf.Options{NoCenter: true}}} {
		r, e := acf.Compute(tc.v, tc.lag, tc.opts)
		mustErrorIs(t, e, acf.ErrZeroVariance)
		if r != nil {
			t.Fatal("partial result")
		}
	}
}

// TestCompute_ValidationPrecedence proves that invalid options precede nil,
// empty, invalid shape, and non-finite data, with no partial result on failure.
func TestCompute_ValidationPrecedence(t *testing.T) {
	cases := []struct {
		v    []float64
		lag  int
		opts acf.Options
		err  error
	}{
		{nil, -1, acf.Options{}, acf.ErrInvalidOption},
		{nil, 0, acf.Options{Denominator: 99}, acf.ErrInvalidOption},
		{nil, 0, acf.Options{}, acf.ErrNilInput},
		{[]float64{}, 0, acf.Options{}, acf.ErrEmptyInput},
		{[]float64{math.NaN()}, 1, acf.Options{}, acf.ErrInvalidOption},
		{[]float64{math.NaN()}, 0, acf.Options{}, acf.ErrNonFinite},
		{[]float64{math.Inf(1)}, 0, acf.Options{}, acf.ErrNonFinite},
		{[]float64{math.Inf(-1)}, 0, acf.Options{}, acf.ErrNonFinite},
	}
	for _, tc := range cases {
		r, e := acf.Compute(tc.v, tc.lag, tc.opts)
		mustErrorIs(t, e, tc.err)
		if r != nil {
			t.Fatal("result on error")
		}
	}
}

// TestCompute_ReversedSequenceSymmetry reverses an irregular sample window.
// Every lag must match for both estimators and both centering choices.
func TestCompute_ReversedSequenceSymmetry(t *testing.T) {
	x := []float64{5, -3, 2, 8, 0, -7, 4, 9, -2, 1, 11, 6}
	y := append([]float64(nil), x...)
	for i, j := 0, len(y)-1; i < j; i, j = i+1, j-1 {
		y[i], y[j] = y[j], y[i]
	}
	for _, den := range []acf.Denominator{acf.CommonDenominator, acf.LagAdjusted} {
		for _, nc := range []bool{false, true} {
			o := acf.Options{Denominator: den, NoCenter: nc}
			a, e := acf.Compute(x, len(x)-1, o)
			if e != nil {
				t.Fatal(e)
			}
			b, e := acf.Compute(y, len(y)-1, o)
			if e != nil {
				t.Fatal(e)
			}
			mustValues(t, a.Values, b.Values)
		}
	}
}

// TestCompute_ScalingInvariance checks that a large positive multiplier
// cancels from the numerator and denominator of both valid estimators.
func TestCompute_ScalingInvariance(t *testing.T) {
	x := []float64{-8, 3, 7, -2, 5, 9, 1, -4, 6, -11, 2}
	y := make([]float64, len(x))
	for i, v := range x {
		y[i] = v * 1e150
	}
	for _, den := range []acf.Denominator{acf.CommonDenominator, acf.LagAdjusted} {
		for _, nc := range []bool{false, true} {
			o := acf.Options{Denominator: den, NoCenter: nc}
			a, e := acf.Compute(x, 10, o)
			if e != nil {
				t.Fatal(e)
			}
			b, e := acf.Compute(y, 10, o)
			if e != nil {
				t.Fatal(e)
			}
			mustValues(t, a.Values, b.Values)
		}
	}
}

// TestCompute_ExtremeLargeOffsetSmallSpread tests unit-scale variation
// around a much larger fitted offset without losing centered deviations.
func TestCompute_ExtremeLargeOffsetSmallSpread(t *testing.T) {
	for _, den := range []acf.Denominator{acf.CommonDenominator, acf.LagAdjusted} {
		o := acf.Options{Denominator: den}
		a, e := acf.Compute([]float64{1, 2, 3}, 2, o)
		if e != nil {
			t.Fatal(e)
		}
		b, e := acf.Compute([]float64{1e12 + 1, 1e12 + 2, 1e12 + 3}, 2, o)
		if e != nil {
			t.Fatal(e)
		}
		mustValues(t, a.Values, b.Values)
		mustClose(t, b.Mean, 1e12+2)
	}
}

// TestCompute_ExtremeMagnitudeNoOverflow exercises scaled S0 and cross-products.
// Finite extremes must remain computable when unscaled squares would overflow.
func TestCompute_ExtremeMagnitudeNoOverflow(t *testing.T) {
	x := []float64{1e154, -1e154, 1e154, -1e154}
	y := []float64{math.MaxFloat64, -math.MaxFloat64, math.MaxFloat64, -math.MaxFloat64}
	for _, v := range [][]float64{x, y, {math.MaxFloat64, math.MaxFloat64}, {math.SmallestNonzeroFloat64, 0, 0}} {
		for _, nc := range []bool{false, true} {
			if !nc && v[0] == math.MaxFloat64 && v[1] == math.MaxFloat64 {
				continue
			}
			r, e := acf.Compute(v, len(v)-1, acf.Options{NoCenter: nc})
			if e != nil {
				t.Fatalf("%v noCenter=%v: %v", v, nc, e)
			}
			for _, a := range r.Values {
				if math.IsInf(a, 0) || math.IsNaN(a) {
					t.Fatal("nonfinite result")
				}
			}
		}
	}
}

// TestCompute_ResultDetached confirms input immutability and output ownership.
// Mutating an earlier Result must not contaminate a later independent call.
func TestCompute_ResultDetached(t *testing.T) {
	x := []float64{1, 4, 2, 8, 6}
	original := append([]float64(nil), x...)
	a, e := acf.Compute(x, 3, acf.Options{})
	if e != nil {
		t.Fatal(e)
	}
	a.Values[0] = 77
	a.PairCounts[0] = 88
	b, e := acf.Compute(x, 3, acf.Options{})
	if e != nil {
		t.Fatal(e)
	}
	if b.Values[0] != 1 || b.PairCounts[0] != 5 {
		t.Fatal("retained result")
	}
	mustValues(t, x, original)
}

// TestResult_IsNilInterface checks typed-nil classification without reflection.
// A pointer to an allocated zero-value Result is explicitly non-nil.
func TestResult_IsNilInterface(t *testing.T) {
	var r *acf.Result
	var nilable interface{ IsNil() bool } = r
	mustNilState(t, nilable, true, "typed nil")
	nilable = &acf.Result{}
	mustNilState(t, nilable, false, "non-nil")
}

// TestCompute_NonCausalPrefixAndTranslation distinguishes fitted-window
// autocorrelation from prefix-causal transforms and raw offset-invariant laws.
func TestCompute_NonCausalPrefixAndTranslation(t *testing.T) {
	a, e := acf.Compute([]float64{1, 2, 3}, 1, acf.Options{})
	if e != nil {
		t.Fatal(e)
	}
	b, e := acf.Compute([]float64{1, 2, 3, 9}, 1, acf.Options{})
	if e != nil {
		t.Fatal(e)
	}
	if a.Values[1] == b.Values[1] {
		t.Fatal("unexpected prefix-invariance")
	}
	c, e := acf.Compute([]float64{2, 2, 2}, 1, acf.Options{NoCenter: true})
	if e != nil {
		t.Fatal(e)
	}
	d, e := acf.Compute([]float64{3, 3, 3}, 1, acf.Options{NoCenter: true})
	if e != nil {
		t.Fatal(e)
	}
	mustValues(t, c.Values, d.Values)
	e1, e := acf.Compute([]float64{1, 2, 3}, 2, acf.Options{NoCenter: true})
	if e != nil {
		t.Fatal(e)
	}
	e2, e := acf.Compute([]float64{11, 12, 13}, 2, acf.Options{NoCenter: true})
	if e != nil {
		t.Fatal(e)
	}
	if math.Abs(e1.Values[2]-e2.Values[2]) < .01 {
		t.Fatal("uncentered translation incorrectly invariant")
	}
}

// Nil-state compatibility is structural: the repository's core.Nilable
// consists of exactly this one method, so importing core is unnecessary.
var _ interface{ IsNil() bool } = (*acf.Result)(nil)

// TestCompute_WrapperEquivalence confirms that convenience entry points do not
// implement independent arithmetic or silently switch the estimator.
func TestCompute_WrapperEquivalence(t *testing.T) {
	values := []float64{2, 4, 1, -3, 5, 0, 7, 9, -2, 3, 6, 1}
	tests := []struct {
		name string
		call func([]float64, int) (*acf.Result, error)
		opts acf.Options
	}{
		{"default", acf.ComputeDefault, acf.Options{}},
		{"centered-lag-adjusted", acf.ComputeLagAdjusted, acf.Options{Denominator: acf.LagAdjusted}},
		{"raw-common", acf.ComputeUncentered, acf.Options{NoCenter: true}},
		{"raw-lag-adjusted", acf.ComputeUncenteredLagAdjusted, acf.Options{NoCenter: true, Denominator: acf.LagAdjusted}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			for _, lag := range []int{0, 3, len(values) - 1} {
				want, err := acf.Compute(values, lag, tc.opts)
				if err != nil {
					t.Fatal(err)
				}
				got, err := tc.call(values, lag)
				if err != nil {
					t.Fatal(err)
				}
				mustValues(t, got.Values, want.Values)
				mustPairCounts(t, got, len(values))
				mustClose(t, got.Mean, want.Mean)
				mustEqualInt(t, int(got.Denominator), int(tc.opts.Denominator), "effective denominator")
				if tc.opts.NoCenter && (got.Mean != 0 || math.Signbit(got.Mean)) {
					t.Fatal("NoCenter must publish exactly positive zero")
				}
			}
		})
	}
}

// TestCompute_WrappersErrorPrecedence locks the same failure classes and
// nil-result law for every shortcut, including the invalid-lag-before-nil gate.
func TestCompute_WrappersErrorPrecedence(t *testing.T) {
	wrappers := []func([]float64, int) (*acf.Result, error){
		acf.ComputeDefault,
		acf.ComputeLagAdjusted,
		acf.ComputeUncentered,
		acf.ComputeUncenteredLagAdjusted,
	}
	for _, wrapper := range wrappers {
		for _, tc := range []struct {
			values []float64
			lag    int
			want   error
		}{
			{nil, -1, acf.ErrInvalidOption},
			{nil, 0, acf.ErrNilInput},
			{[]float64{}, 0, acf.ErrEmptyInput},
			{[]float64{math.NaN()}, 1, acf.ErrInvalidOption},
			{[]float64{math.Inf(-1)}, 0, acf.ErrNonFinite},
		} {
			got, err := wrapper(tc.values, tc.lag)
			mustErrorIs(t, err, tc.want)
			if got != nil {
				t.Fatalf("partial result for %v", tc.want)
			}
		}
	}
}

// TestResult_CloneDetached covers deep ownership, nil receivers, and
// the non-nil empty slice distinction without reflective comparisons.
func TestResult_CloneDetached(t *testing.T) {
	var missing *acf.Result
	if missing.Clone() != nil || !missing.IsNil() {
		t.Fatal("nil receiver must preserve nil on Clone and IsNil")
	}

	original := &acf.Result{
		Values:      []float64{1, -0.25, 0.125},
		PairCounts:  []int{8, 7, 6},
		Mean:        17.125,
		Denominator: acf.LagAdjusted,
	}
	copyResult := original.Clone()
	if copyResult == nil || copyResult == original {
		t.Fatal("Clone must allocate a new result")
	}
	mustValues(t, copyResult.Values, original.Values)
	mustEqualInt(t, len(copyResult.PairCounts), len(original.PairCounts), "clone pair counts")
	mustClose(t, copyResult.Mean, original.Mean)
	mustEqualInt(t, int(copyResult.Denominator), int(original.Denominator), "clone estimator")

	copyResult.Values[0] = 19
	copyResult.PairCounts[0] = 31
	if original.Values[0] != 1 || original.PairCounts[0] != 8 {
		t.Fatal("Clone shares mutable slices with the original")
	}
	original.Values[1] = -2
	original.PairCounts[1] = 99
	if copyResult.Values[1] != -0.25 || copyResult.PairCounts[1] != 7 {
		t.Fatal("original mutation propagated into clone")
	}

	empty := (&acf.Result{Values: []float64{}, PairCounts: []int{}}).Clone()
	if empty == nil || empty.Values == nil || empty.PairCounts == nil {
		t.Fatal("non-nil empty slices must remain non-nil")
	}
	zero := (&acf.Result{}).Clone()
	if zero == nil || zero.Values != nil || zero.PairCounts != nil {
		t.Fatal("zero-value result must clone without inventing slices")
	}
}

// TestCompute_ParallelIndependentCalls confirms the absence of shared kernel
// state and inspects worker errors only on the owning test goroutine.
func TestCompute_ParallelIndependentCalls(t *testing.T) {
	const workers = 12
	const observations = 257
	x := make([]float64, observations)
	for i := range x {
		x[i] = math.Sin(float64(i)*0.17) + float64(i%7)*0.125
	}

	want, err := acf.Compute(x, 64, acf.Options{})
	if err != nil {
		t.Fatal(err)
	}
	failures := make(chan error, workers)
	var group sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		group.Add(1)
		go func() {
			defer group.Done()
			got, err := acf.ComputeDefault(x, 64)
			if err != nil {
				failures <- err
				return
			}
			for k := range want.Values {
				if got.Values[k] != want.Values[k] || got.PairCounts[k] != want.PairCounts[k] {
					failures <- fmt.Errorf("lag %d: concurrent result differs", k)
					return
				}
			}
		}()
	}
	group.Wait()
	close(failures)
	for err = range failures {
		t.Error(err)
	}
}

// TestCompute_ValidatedErrorIdentity confirms errors.Is as the protocol for
// each package-owned sentinel; strings are never compared or parsed.
func TestCompute_ValidatedErrorIdentity(t *testing.T) {
	tests := []struct {
		name   string
		values []float64
		lag    int
		opts   acf.Options
		want   error
	}{
		{"invalid-enum", []float64{1, 2}, 1, acf.Options{Denominator: 255}, acf.ErrInvalidOption},
		{"negative-lag", nil, -1, acf.Options{}, acf.ErrInvalidOption},
		{"nil", nil, 0, acf.Options{}, acf.ErrNilInput},
		{"empty", []float64{}, 0, acf.Options{}, acf.ErrEmptyInput},
		{"shape", []float64{math.NaN()}, 1, acf.Options{}, acf.ErrInvalidOption},
		{"nonfinite", []float64{0, math.Inf(1)}, 0, acf.Options{}, acf.ErrNonFinite},
		{"constant", []float64{3, 3, 3}, 2, acf.Options{}, acf.ErrZeroVariance},
		{"all-zero", []float64{0, 0, 0}, 2, acf.Options{NoCenter: true}, acf.ErrZeroVariance},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := acf.Compute(tc.values, tc.lag, tc.opts)
			if got != nil {
				t.Fatal("result was published on failure")
			}
			if !errors.Is(err, tc.want) {
				t.Fatalf("error class %v does not match %v", err, tc.want)
			}
		})
	}
}
