// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package obv_test verifies independent arithmetic and failure invariants.
// These cases exercise the public API without exposing private production helpers.
package obv_test

import (
	"math"
	"math/big"
	"testing"

	"github.com/lvlath/go/obv"
)

// TestOverflowRebaseCancellationFirst covers a valid, representable checkpoint
// whose high component would overflow when the next contribution is added.
func TestOverflowRebaseCancellationFirst(t *testing.T) {
	s := obv.New()
	if _, _, err := s.Push(obv.Bar{Open: 100, High: 100, Low: 100, Close: 100}); err != nil {
		t.Fatal(err)
	}
	checkpoint := s.Snapshot()
	checkpoint.Count = 4
	checkpoint.FloatState = []float64{100, math.MaxFloat64, -math.MaxFloat64}
	if err := s.Restore(checkpoint); err != nil {
		t.Fatal(err)
	}
	value, valid, err := s.Push(obv.Bar{Open: 101, High: 101, Low: 101, Close: 101, Volume: math.MaxFloat64})
	if err != nil || !valid || math.Float64bits(value) != math.Float64bits(math.MaxFloat64) {
		t.Fatalf("finite cancellation rejected: value=%v valid=%v err=%v", value, valid, err)
	}
	got := s.Snapshot()
	if got.Count != 5 || got.FloatState[0] != 101 || !finiteSnapshot(got) {
		t.Fatalf("bad committed state: %+v", got)
	}
}

func finiteSnapshot(state obv.Snapshot) bool {
	for _, x := range state.FloatState {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return false
		}
	}
	return true
}

// TestIndependentRationalOracle checks cancellation against exact rational arithmetic
// on a representable integer volume sequence, not merely batch/stream parity.
func TestIndependentRationalOracle(t *testing.T) {
	closes := []float64{100, 101, 102, 101, 100, 103, 103, 104, 102, 105}
	volumes := []float64{4000, 1e16, 1, 1e16, 17, 1e16, math.MaxFloat64, 5, 1e16, 7}
	got, err := obv.Compute(bars(closes, volumes))
	if err != nil {
		t.Fatal(err)
	}
	balance := new(big.Rat)
	for i := range closes {
		if i > 0 {
			contribution := new(big.Rat).SetFloat64(volumes[i])
			if closes[i] < closes[i-1] {
				contribution.Neg(contribution)
			} else if closes[i] == closes[i-1] {
				contribution.SetInt64(0)
			}
			balance.Add(balance, contribution)
		}
		want, _ := balance.Float64()
		if math.Float64bits(got.Values[i]) != math.Float64bits(want) {
			t.Fatalf("exact-rational oracle at %d: got %v, want %v", i, got.Values[i], want)
		}
	}
}

// TestComputeCandlesContracts anchors the row facade's equivalence, isolation,
// nil/empty classification, and no-partial-result contract.
func TestComputeCandlesContracts(t *testing.T) {
	candles := []obv.Bar{
		{Open: 100, High: 102, Low: 99, Close: 101, Volume: 99},
		{Open: 101, High: 105, Low: 100, Close: 104, Volume: 450},
		{Open: 104, High: 106, Low: 102, Close: 103, Volume: 250},
		{Open: 103, High: 104, Low: 99, Close: 101, Volume: 100},
	}
	got, err := obv.ComputeCandles(candles)
	if err != nil {
		t.Fatal(err)
	}
	mustValues(t, got, []float64{0, 450, 200, 100})
	columns := obv.Bars{Open: []float64{100, 101, 104, 103}, High: []float64{102, 105, 106, 104}, Low: []float64{99, 100, 102, 99}, Close: []float64{101, 104, 103, 101}, Volume: []float64{99, 450, 250, 100}}
	batch, err := obv.Compute(columns)
	if err != nil {
		t.Fatal(err)
	}
	for i := range got.Values {
		if math.Float64bits(got.Values[i]) != math.Float64bits(batch.Values[i]) {
			t.Fatalf("row/column divergence at %d", i)
		}
	}
	candles[1].Volume = 123456
	if got.Values[1] != 450 {
		t.Fatal("result aliased caller input")
	}
	_, err = obv.ComputeCandles(nil)
	mustErrorIs(t, err, obv.ErrNilInput)
	_, err = obv.ComputeCandles([]obv.Bar{})
	mustErrorIs(t, err, obv.ErrEmptyInput)
	candles[2].Low = 105
	broken, err := obv.ComputeCandles(candles)
	mustErrorIs(t, err, obv.ErrDomain)
	if broken != nil {
		t.Fatal("partial result leaked from malformed late candle")
	}
}

// TestAllFieldsNonFiniteAndDomain tests every OHLCV position independently.
func TestAllFieldsNonFiniteAndDomain(t *testing.T) {
	for field := 0; field < 5; field++ {
		for _, bad := range []float64{math.NaN(), math.Inf(-1), math.Inf(1)} {
			bar := obv.Bar{100, 101, 99, 100, 2}
			switch field {
			case 0:
				bar.Open = bad
			case 1:
				bar.High = bad
			case 2:
				bar.Low = bad
			case 3:
				bar.Close = bad
			case 4:
				bar.Volume = bad
			}
			var s obv.Stream
			_, _, err := s.Push(bar)
			mustErrorIs(t, err, obv.ErrNonFinite)
			if s.Snapshot().Count != 0 {
				t.Fatal("invalid bar mutated stream")
			}
		}
	}
	for _, bar := range []obv.Bar{
		{0, 101, 99, 100, 2},
		{100, 0, 99, 100, 2},
		{100, 101, 0, 100, 2},
		{100, 101, 99, 0, 2},
		{100, 101, 99, 100, -1},
		{102, 101, 99, 100, 2},
		{100, 98, 99, 100, 2},
		{98, 101, 99, 100, 2},
		{100, 101, 99, 102, 2},
		{100, 101, 99, 98, 2},
	} {
		var s obv.Stream
		_, _, err := s.Push(bar)
		mustErrorIs(t, err, obv.ErrDomain)
	}
}

// TestSnapshotCanonicalSchemaAndAtomicity rejects malformed seed and empty states
// without compromising an already running stream.
func TestSnapshotCanonicalSchemaAndAtomicity(t *testing.T) {
	s := obv.New()
	_, _, _ = s.Push(obv.Bar{100, 100, 100, 100, 5})
	before := s.Snapshot()
	cases := []obv.Snapshot{
		{Algorithm: before.Algorithm, Config: before.Config, Count: 0, FloatState: []float64{100, 0, 0}},
		{Algorithm: before.Algorithm, Config: before.Config, Count: 0, FloatState: []float64{0, 1, 0}},
		{Algorithm: before.Algorithm, Config: before.Config, Count: 1, FloatState: []float64{100, 1, 0}},
		{Algorithm: before.Algorithm, Config: before.Config, Count: 2, FloatState: []float64{100, math.MaxFloat64, math.MaxFloat64}},
	}
	for i, checkpoint := range cases {
		mustErrorIs(t, s.Restore(checkpoint), obv.ErrStateVersion)
		if !sameSnapshot(s.Snapshot(), before) {
			t.Fatalf("failed checkpoint %d mutated stream", i)
		}
	}
}

// TestPushZeroAllocsNonzeroTransitions exercises the compensated arithmetic path,
// rather than only the flat-price or zero-volume fast path.
func TestPushZeroAllocsNonzeroTransitions(t *testing.T) {
	var stream obv.Stream
	seed := obv.Bar{100, 100, 100, 100, 0}
	if _, _, err := stream.Push(seed); err != nil {
		t.Fatal(err)
	}
	up := obv.Bar{101, 101, 101, 101, 1e16}
	down := obv.Bar{100, 100, 100, 100, 1e16}
	allocs := testing.AllocsPerRun(1000, func() {
		if _, _, err := stream.Push(up); err != nil {
			t.Fatal(err)
		}
		if _, _, err := stream.Push(down); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("positive and negative delta allocated %v times", allocs)
	}
}
