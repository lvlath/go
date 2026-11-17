// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// MFI01-MFI10: external acceptance tests against hand-computed and independent oracles.
package mfi_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/mfi"
)

func TestMFI01_KnownFixture(t *testing.T) {
	b := sampleBars([]float64{1, 2, 1, 3}, []float64{1, 1, 1, 1})
	r, e := mfi.Compute(b, mfi.Options{Period: 2})
	if e != nil {
		t.Fatal(e)
	}
	want := []float64{0, 0, 200.0 / 3, 75}
	mask := []bool{false, false, true, true}
	for i := range want {
		mustNear(t, r.Values[i], want[i])
		mustEqualBool(t, r.Valid[i], mask[i], "mask")
	}
}
func TestMFI02_EqualityConsumesSlot(t *testing.T) {
	b := sampleBars([]float64{1, 2, 1, 1}, []float64{1, 1, 1, 100})
	r, e := mfi.Compute(b, mfi.Options{Period: 2})
	if e != nil {
		t.Fatal(e)
	}
	mustNear(t, r.Values[3], 0)
}
func TestMFI03_ZeroSums(t *testing.T) {
	for _, prices := range [][]float64{{3, 3, 3, 3, 3}, {1, 2, 1, 4, 2}} {
		volumes := []float64{0, 0, 0, 0, 0}
		r, e := mfi.Compute(sampleBars(prices, volumes), mfi.Options{Period: 2})
		if e != nil {
			t.Fatal(e)
		}
		for i := 2; i < 5; i++ {
			mustNear(t, r.Values[i], 50)
		}
	}
}
func TestMFI04_OneSided(t *testing.T) {
	for _, tc := range []struct {
		p    []float64
		want float64
	}{{[]float64{1, 2, 3, 4}, 100}, {[]float64{4, 3, 2, 1}, 0}} {
		r, e := mfi.Compute(sampleBars(tc.p, []float64{1, 1, 1, 1}), mfi.Options{Period: 2})
		if e != nil {
			t.Fatal(e)
		}
		mustNear(t, r.Values[3], tc.want)
	}
}
func TestMFI05_VolumeScaleInvariance(t *testing.T) {
	p := []float64{10, 11, 10, 12, 9, 10, 8, 13}
	v := []float64{1, 5, 2, 3, 4, 1, 5, 2}
	scaled := make([]float64, len(v))
	for i, x := range v {
		scaled[i] = x * 17
	}
	a, e := mfi.Compute(sampleBars(p, v), mfi.Options{Period: 3})
	if e != nil {
		t.Fatal(e)
	}
	b, e := mfi.Compute(sampleBars(p, scaled), mfi.Options{Period: 3})
	if e != nil {
		t.Fatal(e)
	}
	for i := range p {
		mustNear(t, a.Values[i], b.Values[i])
	}
}
func TestMFI06_ZeroVolumeUpdatesPrevious(t *testing.T) {
	b := sampleBars([]float64{1, 2, 3, 2}, []float64{1, 1, 0, 1})
	r, e := mfi.Compute(b, mfi.Options{Period: 2})
	if e != nil {
		t.Fatal(e)
	}
	mustNear(t, r.Values[3], 0) // price drop 3->2 after zero-volume rise
}
func TestMFI07_DomainAndShape(t *testing.T) {
	good := sampleBars([]float64{1, 2}, []float64{1, 1})
	cases := []struct {
		b    mfi.Bars
		want error
	}{
		{mfi.Bars{}, mfi.ErrNilInput},
		{mfi.Bars{Open: []float64{}, High: []float64{}, Low: []float64{}, Close: []float64{}, Volume: []float64{}}, mfi.ErrEmptyInput},
		{mfi.Bars{Open: []float64{1}, High: []float64{1, 2}, Low: []float64{1}, Close: []float64{1}, Volume: []float64{1}}, mfi.ErrShape},
	}
	for _, tc := range cases {
		_, e := mfi.Compute(tc.b, mfi.Options{})
		mustErrorIs(t, e, tc.want)
	}
	good.Volume[0] = -1
	_, e := mfi.Compute(good, mfi.Options{})
	mustErrorIs(t, e, mfi.ErrDomain)
	good.Volume[0] = math.NaN()
	_, e = mfi.Compute(good, mfi.Options{})
	mustErrorIs(t, e, mfi.ErrNonFinite)
}
func TestMFI08_IndependentLongOracle(t *testing.T) {
	const count = 3017
	prices, v := make([]float64, count), make([]float64, count)
	for i := range prices {
		prices[i] = float64(100 + (i % 11))
		v[i] = float64(i % 17)
	}
	b := sampleBars(prices, v)
	actual, e := mfi.Compute(b, mfi.Options{Period: 73})
	if e != nil {
		t.Fatal(e)
	}
	want, valid := oracle(b, 73)
	for i := range want {
		mustEqualBool(t, actual.Valid[i], valid[i], "oracle valid")
		mustNear(t, actual.Values[i], want[i])
	}
}
func TestMFI10_LimitAndOverflow(t *testing.T) {
	_, e := mfi.New(mfi.Options{Period: (1 << 20) + 1})
	mustErrorIs(t, e, mfi.ErrInvalidOption)
	_, e = mfi.Compute(mfi.Bars{}, mfi.Options{Period: -1})
	mustErrorIs(t, e, mfi.ErrInvalidOption)
	s, e := mfi.New(mfi.Options{Period: 1})
	if e != nil {
		t.Fatal(e)
	}
	_, _, e = s.Push(sampleBar(1, 1))
	if e != nil {
		t.Fatal(e)
	}
	_, _, e = s.Push(sampleBar(2, math.MaxFloat64))
	mustErrorIs(t, e, mfi.ErrOverflow)
	mustEqualInt(t, int(s.Snapshot().Count), 1, "atomic count")
}
func TestResult_IsNilInterface(t *testing.T) {
	var nilResult *mfi.Result
	mustNilState(t, nilResult, true, "typed nil")
	mustNilState(t, &mfi.Result{}, false, "allocated")
}
func TestMFI_DefaultAndWarmup(t *testing.T) {
	prices := make([]float64, 14)
	volumes := make([]float64, 14)
	for i := range prices {
		prices[i] = float64(i + 1)
		volumes[i] = 1
	}
	r, e := mfi.Compute(sampleBars(prices, volumes), mfi.Options{})
	if e != nil {
		t.Fatal(e)
	}
	for _, ok := range r.Valid {
		if ok {
			t.Fatal("unexpected valid warmup")
		}
	}
}
func TestMFI_ExtremeTypicalPrice(t *testing.T) {
	s, e := mfi.New(mfi.Options{Period: 1})
	if e != nil {
		t.Fatal(e)
	}
	x := math.MaxFloat64
	if _, _, e = s.Push(sampleBar(x, 0)); e != nil {
		t.Fatal(e)
	}
	value, valid, e := s.Push(sampleBar(x, 0))
	if e != nil {
		t.Fatal(e)
	}
	if !valid {
		t.Fatal("not valid")
	}
	mustNear(t, value, 50)
}
