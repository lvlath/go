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
	// A typed-nil pointer stored in an interface is not a nil interface.
	var r *macd.Result
	var nilable interface{ IsNil() bool } = r
	mustEqualBool(t, nilable == nil, false, "typed-nil interface")
	mustEqualBool(t, nilable.IsNil(), true, "nil result")
	nilable = &macd.Result{}
	mustEqualBool(t, nilable.IsNil(), false, "allocated result")
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

// TestFIXMACD_PeriodOneCheckpoint verifies that a period-one price EMA preserves
// the exact current sample and produces a restorable checkpoint at every step.
// It covers both orders of extreme magnitudes, including the binary64 endpoints.
//
// Implementation:
//   - Stage 1: construct the Fast=1, Slow=2, Signal=1 stream and empty standby.
//   - Stage 2: push each price, compare Fast to the input bits and restore the checkpoint.
//   - Stage 3: compare continuation against the previously restored standby and batch.
//
// Complexity:
//   - Time O(n), Space O(n) for the batch result; each stream operation is O(1).
//
// AI-Hints:
//   - A tolerance against a huge preceding price would hide complete sample loss.
//   - A successful Push must not export a checkpoint that fails its own Restore.
func TestFIXMACD_PeriodOneCheckpoint(t *testing.T) {
	for _, tc := range []struct {
		name   string
		prices []float64
	}{
		{"huge_to_unit", []float64{1e308, 1, 3, 2}},
		{"unit_to_huge", []float64{1, 1e308, 1}},
		{"max_to_min", []float64{math.MaxFloat64, math.SmallestNonzeroFloat64, 1}},
		{"min_to_max", []float64{math.SmallestNonzeroFloat64, math.MaxFloat64, math.SmallestNonzeroFloat64}},
		{"subnormal", []float64{2 * math.SmallestNonzeroFloat64, math.SmallestNonzeroFloat64, 3 * math.SmallestNonzeroFloat64}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			opts := macd.Options{Fast: 1, Slow: 2, Signal: 1}
			stream, err := macd.New(opts)
			if err != nil {
				t.Fatal(err)
			}
			standby, err := macd.New(opts)
			if err != nil {
				t.Fatal(err)
			}
			batch, err := macd.Compute(tc.prices, opts)
			if err != nil {
				t.Fatal(err)
			}
			for i, price := range tc.prices {
				point, pushErr := stream.Push(price)
				if pushErr != nil {
					t.Fatalf("price[%d]: %v", i, pushErr)
				}
				// Fast=1 is an exact projection, even when Slow is still warming.
				saved := stream.Snapshot()
				if math.Float64bits(saved.FloatState[0]) != math.Float64bits(price) {
					t.Errorf("index=%d fast mean=%.17g want=%.17g", i, saved.FloatState[0], price)
				}
				other, pushErr := standby.Push(price)
				if pushErr != nil {
					t.Fatal(pushErr)
				}
				mustBits(t, other.Line, point.Line, "checkpoint continuation line")
				mustBits(t, other.Signal, point.Signal, "checkpoint continuation signal")
				mustBits(t, other.Histogram, point.Histogram, "checkpoint continuation histogram")
				mustEqualBool(t, other.LineValid, point.LineValid, "checkpoint line mask")
				mustEqualBool(t, other.SignalValid, point.SignalValid, "checkpoint signal mask")
				if err = standby.Restore(saved); err != nil {
					t.Fatalf("index=%d own checkpoint rejected: %v", i, err)
				}
				mustSnapshotsEqual(t, saved, standby.Snapshot())
				mustBits(t, point.Line, batch.Line.Values[i], "batch line")
				mustBits(t, point.Signal, batch.Signal.Values[i], "batch signal")
				mustBits(t, point.Histogram, batch.Histogram.Values[i], "batch histogram")
				mustEqualBool(t, point.LineValid, i >= 1, "Line ready")
				mustEqualBool(t, point.SignalValid, i >= 1, "Signal ready")
				if point.SignalValid {
					mustBits(t, point.Signal, point.Line, "exact Signal=1")
					if point.Histogram != 0 {
						t.Fatalf("nonzero Histogram at index %d: %.17g", i, point.Histogram)
					}
				}
			}
		})
	}
}

// TestFIXMACD_SignalIdentity exercises the audited extreme signed-line history.
// Repeating the eight-sample motif crosses both the 26-close warmup and the
// first mature signal transition at index 26; eight closes alone cannot do so.
//
// Implementation:
//   - Stage 1: construct a positive, finite, 64-sample history and compute batch MACD.
//   - Stage 2: compare exact identity and readiness for every streaming and batch point.
//   - Stage 3: require both line signs, and restore every successful state.
//
// Complexity:
//   - Time O(n), Space O(n) for batch; O(1) stream state.
//
// AI-Hints:
//   - Signal=1 is an identity on signed MACD values, not a positive-price filter.
//   - Never use relative tolerance for the algebraic zero histogram guarantee.
func TestFIXMACD_SignalIdentity(t *testing.T) {
	motif := [...]float64{1, 1e270, 1e-270, 1e-100, 1e100, 1, 3, 2}
	prices := make([]float64, 8*len(motif))
	for i := range prices {
		prices[i] = motif[i%len(motif)]
	}
	opts := macd.Options{Fast: 2, Slow: 26, Signal: 1}
	batch, err := macd.Compute(prices, opts)
	if err != nil {
		t.Fatal(err)
	}
	stream, err := macd.New(opts)
	if err != nil {
		t.Fatal(err)
	}
	standby, err := macd.New(opts)
	if err != nil {
		t.Fatal(err)
	}
	positive, negative := false, false
	for i, price := range prices {
		point, pushErr := stream.Push(price)
		if pushErr != nil {
			t.Fatal(pushErr)
		}
		other, pushErr := standby.Push(price)
		if pushErr != nil {
			t.Fatal(pushErr)
		}
		mustBits(t, other.Line, point.Line, "restored Line")
		mustBits(t, other.Signal, point.Signal, "restored Signal")
		mustBits(t, other.Histogram, point.Histogram, "restored Histogram")
		mustEqualBool(t, other.LineValid, point.LineValid, "restored Line mask")
		mustEqualBool(t, other.SignalValid, point.SignalValid, "restored Signal mask")
		mustBits(t, point.Line, batch.Line.Values[i], "batch Line")
		mustBits(t, point.Signal, batch.Signal.Values[i], "batch Signal")
		mustBits(t, point.Histogram, batch.Histogram.Values[i], "batch Histogram")
		ready := i >= opts.Slow-1
		mustEqualBool(t, point.LineValid, ready, "Line warmup")
		mustEqualBool(t, point.SignalValid, ready, "Signal warmup")
		mustEqualBool(t, batch.Histogram.Valid[i], ready, "Histogram warmup")
		if ready {
			positive = positive || point.Line > 0
			negative = negative || point.Line < 0
			if math.Float64bits(point.Signal) != math.Float64bits(point.Line) || point.Histogram != 0 {
				t.Errorf("index=%d Line=%.17g Signal=%.17g Histogram=%.17g", i, point.Line, point.Signal, point.Histogram)
			}
		}
		if err = standby.Restore(stream.Snapshot()); err != nil {
			t.Fatalf("index=%d: %v", i, err)
		}
	}
	if !positive || !negative {
		t.Fatal("fixture must contain positive and negative valid lines")
	}
}
