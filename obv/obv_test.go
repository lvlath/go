// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package obv_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/obv"
)

func TestOBV01_IndependentOracle(t *testing.T) {
	r, e := obv.Compute(bars([]float64{10, 11, 11, 9}, []float64{5, 3, 8, 2}))
	if e != nil {
		t.Fatal(e)
	}
	mustValues(t, r, []float64{0, 3, 3, 1})
}
func TestOBV02_FirstBarNeverContributes(t *testing.T) {
	r, e := obv.Compute(bars([]float64{10}, []float64{math.MaxFloat64}))
	if e != nil {
		t.Fatal(e)
	}
	mustValues(t, r, []float64{0})
}
func TestOBV03_NeumaierCancellation(t *testing.T) {
	r, e := obv.Compute(bars([]float64{1, 2, 3, 2}, []float64{0, 1e16, 1, 1e16}))
	if e != nil {
		t.Fatal(e)
	}
	mustValues(t, r, []float64{0, 1e16, 1e16, 1})
}
func TestOBV04_EqualCloseHugeVolume(t *testing.T) {
	r, e := obv.Compute(bars([]float64{1, 2, 2, 3}, []float64{1, 2, math.MaxFloat64, 1}))
	if e != nil {
		t.Fatal(e)
	}
	mustValues(t, r, []float64{0, 2, 2, 3})
}
func TestOBV06_VolumeScaling(t *testing.T) {
	a := bars([]float64{10, 11, 9, 9, 12, 8}, []float64{7, 3, 2, 5, 9, 4})
	x, e := obv.Compute(a)
	if e != nil {
		t.Fatal(e)
	}
	for i := range a.Volume {
		a.Volume[i] *= 2
	}
	y, e := obv.Compute(a)
	if e != nil {
		t.Fatal(e)
	}
	for i := range x.Values {
		if y.Values[i] != 2*x.Values[i] {
			t.Fatalf("index %d", i)
		}
	}
}
func TestShapeAndDomainPrecedence(t *testing.T) {
	b := bars([]float64{1, 2}, []float64{0, 1})
	b.Volume = nil
	_, e := obv.Compute(b)
	mustErrorIs(t, e, obv.ErrNilInput)
	b = obv.Bars{Open: []float64{}, High: []float64{}, Low: []float64{}, Close: []float64{}, Volume: []float64{}}
	_, e = obv.Compute(b)
	mustErrorIs(t, e, obv.ErrEmptyInput)
	b = bars([]float64{1, 2}, []float64{0, 1})
	b.Volume = b.Volume[:1]
	_, e = obv.Compute(b)
	mustErrorIs(t, e, obv.ErrShape)
	b = bars([]float64{1, 2}, []float64{0, 1})
	b.High[1] = math.NaN()
	b.Open[1] = 0
	_, e = obv.Compute(b)
	mustErrorIs(t, e, obv.ErrNonFinite)
	b = bars([]float64{1, 2}, []float64{0, 1})
	b.Open[1] = 3
	_, e = obv.Compute(b)
	mustErrorIs(t, e, obv.ErrDomain)
}

// TestResult_IsNilInterface distinguishes nil receiver, typed-nil interface and
// an allocated zero-value Result without inspecting the output slice lengths.
func TestResult_IsNilInterface(t *testing.T) {
	var result *obv.Result
	var nilable interface{ IsNil() bool } = result
	mustNilState(t, nilable, true, "typed nil Result interface")
	result = &obv.Result{}
	nilable = result
	mustNilState(t, nilable, false, "allocated empty Result interface")
}
func TestResultDetached(t *testing.T) {
	b := bars([]float64{1, 2, 3}, []float64{1, 2, 3})
	r, e := obv.Compute(b)
	if e != nil {
		t.Fatal(e)
	}
	b.Close[1] = 100
	r.Values[0] = 42
	r.Valid[1] = false
	other, e := obv.Compute(bars([]float64{1, 2, 3}, []float64{1, 2, 3}))
	if e != nil {
		t.Fatal(e)
	}
	mustValues(t, other, []float64{0, 2, 5})
}
func TestPositiveAffinePriceTransformation(t *testing.T) {
	c := []float64{10, 11, 12, 10, 13}
	v := []float64{9, 7, 2, 3, 4}
	a, e := obv.Compute(bars(c, v))
	if e != nil {
		t.Fatal(e)
	}
	for i := range c {
		c[i] = c[i]*2 + 50
	}
	b, e := obv.Compute(bars(c, v))
	if e != nil {
		t.Fatal(e)
	}
	for i := range c {
		if math.Float64bits(a.Values[i]) != math.Float64bits(b.Values[i]) {
			t.Fatalf("index %d", i)
		}
	}
}
