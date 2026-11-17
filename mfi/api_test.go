// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// External API tests: additive facade parity, error precedence and ownership.
package mfi_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/mfi"
)

func TestAPI_ConvenienceWrappersAreCanonical(t *testing.T) {
	const count = 109
	rows := make([]mfi.Bar, count)
	prices, volumes := make([]float64, count), make([]float64, count)
	for i := range rows {
		prices[i] = float64(120 + (i*19)%27)
		volumes[i] = float64((i * 7) % 31)
		rows[i] = sampleBar(prices[i], volumes[i])
	}
	columns := sampleBars(prices, volumes)

	for _, period := range []int{0, 1, 2, 14, 37, 1000} {
		canonical, err := mfi.Compute(columns, mfi.Options{Period: period})
		if err != nil {
			t.Fatalf("period=%d: %v", period, err)
		}
		byPeriod, err := mfi.ComputePeriod(columns, period)
		if err != nil {
			t.Fatal(err)
		}
		byRows, err := mfi.ComputeCandles(rows, mfi.Options{Period: period})
		if err != nil {
			t.Fatal(err)
		}
		mustResultBits(t, byPeriod, canonical)
		mustResultBits(t, byRows, canonical)

		latest, valid, err := mfi.Latest(columns, mfi.Options{Period: period})
		if err != nil {
			t.Fatal(err)
		}
		if math.Float64bits(latest) != math.Float64bits(canonical.Values[count-1]) || valid != canonical.Valid[count-1] {
			t.Fatalf("period=%d: Latest diverged", period)
		}
		stream, err := mfi.NewPeriod(period)
		if err != nil {
			t.Fatal(err)
		}
		for i, row := range rows {
			value, warmed, pushErr := stream.Push(row)
			if pushErr != nil {
				t.Fatal(pushErr)
			}
			if warmed != canonical.Valid[i] || math.Float64bits(value) != math.Float64bits(canonical.Values[i]) {
				t.Fatalf("period=%d bar=%d: stream diverged", period, i)
			}
		}
	}

	byDefault, err := mfi.ComputeDefault(columns)
	if err != nil {
		t.Fatal(err)
	}
	canonical, err := mfi.Compute(columns, mfi.Options{Period: mfi.DefaultPeriod})
	if err != nil {
		t.Fatal(err)
	}
	mustResultBits(t, byDefault, canonical)
	defaultStream, err := mfi.NewDefault()
	if err != nil {
		t.Fatal(err)
	}
	mustEqualInt(t, len(defaultStream.Snapshot().FloatState), 5+2*mfi.DefaultPeriod, "default window")
}

func TestAPI_ValidationPrecedence(t *testing.T) {
	_, err := mfi.Compute(mfi.Bars{}, mfi.Options{Period: -1})
	mustErrorIs(t, err, mfi.ErrInvalidOption)
	_, err = mfi.ComputeCandles(nil, mfi.Options{Period: -1})
	mustErrorIs(t, err, mfi.ErrInvalidOption)
	_, err = mfi.ComputeCandles(nil, mfi.Options{})
	mustErrorIs(t, err, mfi.ErrNilInput)
	_, err = mfi.ComputeCandles([]mfi.Bar{}, mfi.Options{})
	mustErrorIs(t, err, mfi.ErrEmptyInput)
	_, err = mfi.NewPeriod(mfi.MaxPeriod + 1)
	mustErrorIs(t, err, mfi.ErrInvalidOption)
	_, err = mfi.NewPeriod(-1)
	mustErrorIs(t, err, mfi.ErrInvalidOption)

	columns := sampleBars([]float64{2, 3}, []float64{1, 1})
	columns.High[0] = math.NaN()
	columns.Volume = []float64{1}
	_, err = mfi.Compute(columns, mfi.Options{})
	mustErrorIs(t, err, mfi.ErrShape)
	columns.Volume = []float64{1, 1}
	_, err = mfi.Compute(columns, mfi.Options{})
	mustErrorIs(t, err, mfi.ErrNonFinite)

	bad := sampleBar(2, 1)
	bad.Volume = -1
	bad.Close = math.Inf(1)
	_, err = mfi.ComputeCandles([]mfi.Bar{bad}, mfi.Options{})
	mustErrorIs(t, err, mfi.ErrNonFinite)
}

func TestAPI_CallerOwnershipAndWarmup(t *testing.T) {
	rows := []mfi.Bar{sampleBar(100, 2), sampleBar(101, 4), sampleBar(102, 0)}
	result, err := mfi.ComputeCandles(rows, mfi.Options{Period: 10})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Valid) != len(rows) {
		t.Fatalf("got %d bars", len(result.Valid))
	}
	for i := range rows {
		if result.Valid[i] || result.Values[i] != 0 {
			t.Fatalf("warmup at %d: value=%g valid=%t", i, result.Values[i], result.Valid[i])
		}
	}
	rows[1].High = math.NaN()
	if result.Values[1] != 0 || result.Valid[1] {
		t.Fatal("result aliased caller's rows")
	}
	result.Valid[0] = true
	result.Values[0] = 77
	again, err := mfi.ComputeCandles([]mfi.Bar{sampleBar(100, 2)}, mfi.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if again.Valid[0] || again.Values[0] != 0 {
		t.Fatal("results retained mutable previous output")
	}
}

func mustResultBits(t *testing.T, got, want *mfi.Result) {
	t.Helper()
	if got == nil || want == nil || len(got.Values) != len(want.Values) || len(got.Valid) != len(want.Valid) {
		t.Fatal("result shape mismatch")
	}
	for i := range got.Values {
		if got.Valid[i] != want.Valid[i] || math.Float64bits(got.Values[i]) != math.Float64bits(want.Values[i]) {
			t.Fatalf("bar %d: got=%g %t want=%g %t", i, got.Values[i], got.Valid[i], want.Values[i], want.Valid[i])
		}
	}
}
