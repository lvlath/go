// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package macd_test verifies that convenience facades remain projections of
// canonical Compute/New and never drift into a second numerical algorithm.
package macd_test

import (
	"errors"
	"math"
	"sync"
	"testing"

	"github.com/lvlath/go/macd"
)

// TestFacade_DefaultAndExplicitPeriods checks every added wrapper, including
// field-by-field bit equality and equal error classifications on invalid periods.
func TestFacade_DefaultAndExplicitPeriods(t *testing.T) {
	defaultOpts := macd.DefaultOptions()
	if defaultOpts.Fast != 12 || defaultOpts.Slow != 26 || defaultOpts.Signal != 9 {
		t.Fatalf("unexpected defaults: %+v", defaultOpts)
	}
	prices := make([]float64, 56)
	for i := range prices {
		prices[i] = 117.3 + 0.019*float64(i*i) + float64(i%7-3)*0.12
	}
	canonical, err := macd.Compute(prices, macd.Options{})
	if err != nil {
		t.Fatal(err)
	}
	for _, call := range []struct {
		name string
		fn   func() (*macd.Result, error)
	}{
		{"default", func() (*macd.Result, error) { return macd.ComputeDefault(prices) }},
		{"explicit", func() (*macd.Result, error) { return macd.ComputeWithPeriods(prices, 12, 26, 9) }},
		{"zero", func() (*macd.Result, error) { return macd.ComputeWithPeriods(prices, 0, 0, 0) }},
	} {
		t.Run(call.name, func(t *testing.T) {
			got, err := call.fn()
			if err != nil {
				t.Fatal(err)
			}
			for i := range prices {
				mustBits(t, got.Line.Values[i], canonical.Line.Values[i], "line")
				mustBits(t, got.Signal.Values[i], canonical.Signal.Values[i], "signal")
				mustBits(t, got.Histogram.Values[i], canonical.Histogram.Values[i], "histogram")
				mustEqualBool(t, got.Line.Valid[i], canonical.Line.Valid[i], "line ready")
				mustEqualBool(t, got.Signal.Valid[i], canonical.Signal.Valid[i], "signal ready")
				mustEqualBool(t, got.Histogram.Valid[i], canonical.Histogram.Valid[i], "hist ready")
			}
		})
	}
	canonicalStream, err := macd.New(macd.Options{})
	if err != nil {
		t.Fatal(err)
	}
	byDefault, err := macd.NewDefault()
	if err != nil {
		t.Fatal(err)
	}
	byPeriods, err := macd.NewWithPeriods(12, 26, 9)
	if err != nil {
		t.Fatal(err)
	}
	for _, price := range prices {
		for _, stream := range []*macd.Stream{canonicalStream, byDefault, byPeriods} {
			if _, err = stream.Push(price); err != nil {
				t.Fatal(err)
			}
		}
	}
	mustSnapshotsEqual(t, canonicalStream.Snapshot(), byDefault.Snapshot())
	mustSnapshotsEqual(t, canonicalStream.Snapshot(), byPeriods.Snapshot())
	if _, err = macd.NewWithPeriods(6, 6, 2); !errors.Is(err, macd.ErrInvalidOption) {
		t.Fatalf("expected ErrInvalidOption from stream wrapper, got %v", err)
	}
	if got, err := macd.ComputeWithPeriods(prices, 6, 6, 2); got != nil || !errors.Is(err, macd.ErrInvalidOption) {
		t.Fatalf("batch wrapper: got=%v err=%v", got, err)
	}
}

// TestMACD09_SignedLineRationalOracle pins a fast-period-one and signal-period-one
// interpretation independently from the implementation's general EMA helper.
func TestMACD09_SignedLineRationalOracle(t *testing.T) {
	result, err := macd.ComputeWithPeriods([]float64{5, 10, 5, 10}, 1, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	for index, want := range []float64{0, 2.5, -5.0 / 6, 25.0 / 18} {
		if index == 0 {
			if result.Line.Valid[index] || result.Signal.Valid[index] {
				t.Fatal("premature readiness at index zero")
			}
			continue
		}
		mustClose(t, result.Line.Values[index], want, "signed line")
		mustBits(t, result.Line.Values[index], result.Signal.Values[index], "signal 1")
		mustBits(t, result.Histogram.Values[index], 0, "histogram 1")
	}
}

// TestCompute_ReadOnlyInputAndIndependentSlices covers caller ownership of the
// full six-slice artifact, including safe concurrent read-only consumption.
func TestCompute_ReadOnlyInputAndIndependentSlices(t *testing.T) {
	prices := make([]float64, 50)
	for index := range prices {
		prices[index] = 70 + float64(index%11) + float64(index)*0.01
	}
	before := append([]float64(nil), prices...)
	result, err := macd.ComputeDefault(prices)
	if err != nil {
		t.Fatal(err)
	}
	for index := range prices {
		mustBits(t, prices[index], before[index], "unchanged close")
	}
	originalSignal := append([]float64(nil), result.Signal.Values...)
	originalHistogram := append([]float64(nil), result.Histogram.Values...)
	originalSignalMask := append([]bool(nil), result.Signal.Valid...)
	originalHistogramMask := append([]bool(nil), result.Histogram.Valid...)
	prices[49] = 1
	result.Line.Values[49] = 9.1e7
	result.Line.Valid[49] = false
	for i := range prices {
		mustBits(t, result.Signal.Values[i], originalSignal[i], "detached signal")
		mustBits(t, result.Histogram.Values[i], originalHistogram[i], "detached histogram")
		mustEqualBool(t, result.Signal.Valid[i], originalSignalMask[i], "detached signal mask")
		mustEqualBool(t, result.Histogram.Valid[i], originalHistogramMask[i], "detached hist mask")
	}

	var wg sync.WaitGroup
	for worker := 0; worker < 4; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for index := 0; index < 50; index++ {
				_ = result.Histogram.Values[index]
				_ = result.Histogram.Valid[index]
			}
		}()
	}
	wg.Wait()
}

// TestSnapshot_RejectsNonCanonicalInactiveSignedZero locks the state format's
// single canonical zero representation during each pre-seed warmup phase.
func TestSnapshot_RejectsNonCanonicalInactiveSignedZero(t *testing.T) {
	stream, err := macd.NewWithPeriods(2, 4, 3)
	if err != nil {
		t.Fatal(err)
	}
	for _, source := range []float64{15, 17} {
		if _, err = stream.Push(source); err != nil {
			t.Fatal(err)
		}
	}
	unchanged := stream.Snapshot()
	for _, stateIndex := range []int{1, 2} {
		invalid := stream.Snapshot()
		invalid.FloatState[stateIndex] = math.Copysign(0, -1)
		mustErrorIs(t, stream.Restore(invalid), macd.ErrStateVersion)
		mustSnapshotsEqual(t, unchanged, stream.Snapshot())
	}
	initial, err := macd.NewWithPeriods(2, 4, 3)
	if err != nil {
		t.Fatal(err)
	}
	invalid := initial.Snapshot()
	invalid.FloatState[0] = math.Copysign(0, -1)
	mustErrorIs(t, initial.Restore(invalid), macd.ErrStateVersion)
}

// TestStream_ResetMatchesFreshStream demonstrates full phase reset across
// a completed signal EMA and an entirely different replay trajectory.
func TestStream_ResetMatchesFreshStream(t *testing.T) {
	stream, err := macd.NewWithPeriods(3, 7, 4)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 25; i++ {
		if _, err = stream.Push(110 + float64(i)); err != nil {
			t.Fatal(err)
		}
	}
	stream.Reset()
	fresh, err := macd.NewWithPeriods(3, 7, 4)
	if err != nil {
		t.Fatal(err)
	}
	for _, value := range []float64{24, 23, 25, 28, 27, 30, 32, 29, 33, 31, 35, 37, 32} {
		actual, err := stream.Push(value)
		if err != nil {
			t.Fatal(err)
		}
		want, err := fresh.Push(value)
		if err != nil {
			t.Fatal(err)
		}
		mustBits(t, actual.Line, want.Line, "reset line")
		mustBits(t, actual.Signal, want.Signal, "reset signal")
		mustBits(t, actual.Histogram, want.Histogram, "reset histogram")
		mustEqualBool(t, actual.LineValid, want.LineValid, "reset line mask")
		mustEqualBool(t, actual.SignalValid, want.SignalValid, "reset signal mask")
	}
	mustSnapshotsEqual(t, stream.Snapshot(), fresh.Snapshot())
}
