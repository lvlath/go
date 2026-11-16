// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package macd_test protects the independent arithmetic oracles, exact warmup
// masks, and option/error admission laws through the public facade.
package macd_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/macd"
)

// TestMACD01_LinearRampIndependentOracle Checks the exact half-unit line and zero histogram on a five-bar rising price ramp.
func TestMACD01_LinearRampIndependentOracle(t *testing.T) {
	r, err := macd.Compute([]float64{1, 2, 3, 4, 5}, macd.Options{Fast: 2, Slow: 3, Signal: 2})
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 5; i++ {
		wantLine := i >= 2
		wantSignal := i >= 3
		mustEqualBool(t, r.Line.Valid[i], wantLine, "line mask")
		mustEqualBool(t, r.Signal.Valid[i], wantSignal, "signal mask")
		mustEqualBool(t, r.Histogram.Valid[i], wantSignal, "hist mask")
		if wantLine {
			mustClose(t, r.Line.Values[i], 0.5, "line")
		} else {
			mustBits(t, r.Line.Values[i], 0, "invalid line")
		}
		if wantSignal {
			mustClose(t, r.Signal.Values[i], 0.5, "signal")
			mustBits(t, r.Histogram.Values[i], 0, "hist")
		} else {
			mustBits(t, r.Signal.Values[i], 0, "invalid signal")
			mustBits(t, r.Histogram.Values[i], 0, "invalid hist")
		}
	}
}

// TestMACD02_ChangingDirectionRationalOracle Pins independently derived 1/6, 5/9, 13/27 and 2/27 fractions across a trend reversal.
func TestMACD02_ChangingDirectionRationalOracle(t *testing.T) {
	r, err := macd.Compute([]float64{1, 2, 3, 2, 5}, macd.Options{Fast: 2, Slow: 3, Signal: 2})
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range map[int]float64{2: 0.5, 3: 1.0 / 6, 4: 5.0 / 9} {
		mustClose(t, r.Line.Values[i], v, "line")
	}
	mustClose(t, r.Signal.Values[3], 1.0/3, "signal[3]")
	mustClose(t, r.Signal.Values[4], 13.0/27, "signal[4]")
	mustClose(t, r.Histogram.Values[3], -1.0/6, "hist[3]")
	mustClose(t, r.Histogram.Values[4], 2.0/27, "hist[4]")
}

// TestMACD03_ExtremeConstant Ensures online SMA warmup never sums large constant closes into a false overflow.
func TestMACD03_ExtremeConstant(t *testing.T) {
	for _, price := range []float64{math.MaxFloat64 / 2, math.MaxFloat64} {
		v := make([]float64, 48)
		for i := range v {
			v[i] = price
		}
		r, err := macd.Compute(v, macd.Options{Fast: 2, Slow: 3, Signal: 2})
		if err != nil {
			t.Fatal(err)
		}
		for i := 2; i < len(v); i++ {
			mustBits(t, r.Line.Values[i], 0, "constant line")
			if i >= 3 {
				mustBits(t, r.Signal.Values[i], 0, "constant signal")
				mustBits(t, r.Histogram.Values[i], 0, "constant hist")
			}
		}
	}
}

// TestMACD04_SignalOne Anchors the Signal=1 identity without relying on numeric zero as a validity test.
func TestMACD04_SignalOne(t *testing.T) {
	r, err := macd.Compute([]float64{7, 9, 3, 11, 2, 13, 5}, macd.Options{Fast: 2, Slow: 3, Signal: 1})
	if err != nil {
		t.Fatal(err)
	}
	for i := range r.Line.Values {
		mustEqualBool(t, r.Line.Valid[i], r.Signal.Valid[i], "valid")
		mustBits(t, r.Line.Values[i], r.Signal.Values[i], "signal=1")
		mustBits(t, r.Histogram.Values[i], 0, "hist=0")
	}
}

// TestMACD05_OptionsAndValidationPrecedence Checks option-over-input precedence and distinct nil/empty/nonfinite/domain failures.
func TestMACD05_OptionsAndValidationPrecedence(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for _, opts := range []macd.Options{{Fast: 3, Slow: 3}, {Fast: -1}, {Signal: -1}, {Fast: 1, Slow: maxInt, Signal: maxInt}} {
		r, err := macd.Compute(nil, opts)
		mustNilState(t, r, true, "failed compute")
		mustErrorIs(t, err, macd.ErrInvalidOption)
	}
	_, err := macd.Compute(nil, macd.Options{})
	mustErrorIs(t, err, macd.ErrNilInput)
	_, err = macd.Compute([]float64{}, macd.Options{})
	mustErrorIs(t, err, macd.ErrEmptyInput)
	for _, x := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		_, err = macd.Compute([]float64{1, x, 0}, macd.Options{})
		mustErrorIs(t, err, macd.ErrNonFinite)
	}
	for _, x := range []float64{0, -1} {
		_, err = macd.Compute([]float64{1, x}, macd.Options{})
		mustErrorIs(t, err, macd.ErrDomain)
	}
}

// TestMACD05_GlobalNonFinitePrecedence checks the batch-wide canonical error
// policy: a late NaN/Inf wins over an earlier invalid finite close.
func TestMACD05_GlobalNonFinitePrecedence(t *testing.T) {
	for _, closeR := range [][]float64{
		{-3, 2, math.NaN()},
		{0, 1, math.Inf(1)},
		{1, -1, math.Inf(-1)},
	} {
		got, err := macd.Compute(closeR, macd.Options{})
		mustNilState(t, got, true, "nonfinite precedence")
		mustErrorIs(t, err, macd.ErrNonFinite)
	}
}

// TestMACD08_Scaling Tests representable positive input scaling against the independent metamorphic law.
func TestMACD08_Scaling(t *testing.T) {
	v := []float64{7, 9, 3, 11, 2, 13, 5, 17, 3, 19, 4, 20, 21}
	scaled := make([]float64, len(v))
	for i, x := range v {
		scaled[i] = x * 8
	}
	o := macd.Options{Fast: 2, Slow: 4, Signal: 3}
	a, e := macd.Compute(v, o)
	if e != nil {
		t.Fatal(e)
	}
	b, e := macd.Compute(scaled, o)
	if e != nil {
		t.Fatal(e)
	}
	for i := range v {
		mustEqualBool(t, a.Line.Valid[i], b.Line.Valid[i], "line mask")
		mustEqualBool(t, a.Signal.Valid[i], b.Signal.Valid[i], "signal mask")
		if a.Line.Valid[i] {
			mustClose(t, b.Line.Values[i], a.Line.Values[i]*8, "scaled line")
		}
		if a.Signal.Valid[i] {
			mustClose(t, b.Signal.Values[i], a.Signal.Values[i]*8, "scaled signal")
			mustClose(t, b.Histogram.Values[i], a.Histogram.Values[i]*8, "scaled histogram")
		}
	}
}

// TestResult_IsNilInterface Checks the pointer-presence convention for typed-nil and allocated results.
func TestResult_IsNilInterface(t *testing.T) {
	var r *macd.Result
	mustEqualBool(t, r.IsNil(), true, "nil")
	r = &macd.Result{}
	mustEqualBool(t, r.IsNil(), false, "non-nil")
}

// TestShortWarmup Rejects premature publication of a Line, Signal or Histogram on short histories.
func TestShortWarmup(t *testing.T) {
	r, e := macd.Compute([]float64{1, 2}, macd.Options{Fast: 2, Slow: 3, Signal: 2})
	if e != nil {
		t.Fatal(e)
	}
	for i := 0; i < 2; i++ {
		if r.Line.Valid[i] || r.Signal.Valid[i] || r.Histogram.Valid[i] {
			t.Fatalf("premature validity at %d", i)
		}
	}
}
