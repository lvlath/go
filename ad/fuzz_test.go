// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ad_test fuzzes public OHLCV boundaries, partial-result policy and replay safety.
// No internal kernels, reflection or private testing-only exports are required.
package ad_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/ad"
)

// FuzzPushDomainAndReplay verifies error classification and transaction safety.
// It then checks that an accepted checkpoint continues identically on a second owner.
func FuzzPushDomainAndReplay(f *testing.F) {
	for _, seed := range [][5]float64{
		{1.5, 2, 1, 2, 3},
		{1, 1, 1, 1, 100},
		{0.5, 2, 1, 1.5, 1},
		{math.NaN(), 2, 1, 1, 1},
		{1.5, 2, 1, 1, math.MaxFloat64},
		{math.MaxFloat64, math.MaxFloat64, math.Nextafter(math.MaxFloat64, 0), math.MaxFloat64, 17},
		{math.SmallestNonzeroFloat64, math.SmallestNonzeroFloat64, math.SmallestNonzeroFloat64, math.SmallestNonzeroFloat64, 0},
	} {
		f.Add(seed[0], seed[1], seed[2], seed[3], seed[4])
	}

	f.Fuzz(func(t *testing.T, open, high, low, closePrice, volume float64) {
		bar := ad.Bar{Open: open, High: high, Low: low, Close: closePrice, Volume: volume}
		stream := ad.New()
		before := stream.Snapshot()
		value, valid, err := stream.Push(bar)

		// Independent domain categorization must agree with the public facade;
		// the oracle is based on field predicates, not on implementation sentinels.
		finite := func(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) }
		allFinite := finite(open) && finite(high) && finite(low) && finite(closePrice) && finite(volume)
		validGeometry := low > 0 && high > 0 && open > 0 && closePrice > 0 &&
			volume >= 0 && high >= low && low <= open && open <= high &&
			low <= closePrice && closePrice <= high

		if !allFinite || !validGeometry {
			want := ad.ErrDomain
			if !allFinite {
				want = ad.ErrNonFinite
			}
			if !errors.Is(err, want) || valid || math.Float64bits(value) != math.Float64bits(0) {
				t.Fatalf("invalid bar: got value=%g valid=%v err=%v want %v", value, valid, err, want)
			}
			mustSnapshotEqual(t, stream.Snapshot(), before)
			return
		}

		if err != nil || !valid || !finite(value) || math.Abs(value) > volume {
			t.Fatalf("valid bar: value=%g volume=%g valid=%v err=%v", value, volume, valid, err)
		}
		if stream.Snapshot().Count != 1 {
			t.Fatal("first accepted candle must increment Count")
		}

		replica := ad.New()
		if restoreErr := replica.Restore(stream.Snapshot()); restoreErr != nil {
			t.Fatal(restoreErr)
		}
		replayValue, replayValid, replayErr := replica.Push(bar)
		referenceValue, referenceValid, referenceErr := stream.Push(bar)
		if !errors.Is(replayErr, referenceErr) || replayValid != referenceValid ||
			math.Float64bits(replayValue) != math.Float64bits(referenceValue) {
			t.Fatal("restored continuation is not bit-identical")
		}
		mustSnapshotEqual(t, replica.Snapshot(), stream.Snapshot())
	})
}

// FuzzBatchShapeAndParity varies row count, malformed structure and arithmetic outcomes.
// Successful arrays are compared with their row and stream forms; failures never publish prefixes.
func FuzzBatchShapeAndParity(f *testing.F) {
	for _, seed := range [][]byte{
		{1, 4, 8, 12, 16, 20},
		{2, 0, 0, 0},
		{3, 0, 255, 11, 12},
		{4, 2, 15, 31, 47},
		{5, 3, 4, 5, 6},
		{6, 21, 44, 66, 88, 101},
		{7, 12, 9, 8, 7},
	} {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, payload []byte) {
		if len(payload) == 0 {
			return
		}
		count := 1 + len(payload)%32
		candles := make([]ad.Bar, count)
		for i := range candles {
			value := payload[i%len(payload)]
			low := 50 + float64(value%100)
			candles[i] = ad.Bar{
				Open: low + 2, Low: low, High: low + 4,
				Close: low + float64(value%5), Volume: float64(value),
			}
		}
		columns := barsFrom(candles)
		mode := payload[0] % 8
		switch mode {
		case 0:
			columns.Volume = nil
		case 1:
			columns.High = columns.High[:0]
		case 2:
			candles[0].Open = 0
			columns.Open[0] = 0
		case 3:
			candles[0].Close = math.NaN()
			columns.Close[0] = math.NaN()
		case 4:
			candles[0].Volume = -1
			columns.Volume[0] = -1
		case 5:
			candles[0].Volume = math.MaxFloat64
			columns.Volume[0] = math.MaxFloat64
			candles[0].Close = candles[0].High
			columns.Close[0] = columns.High[0]
		}

		columnResult, columnErr := ad.Compute(columns)
		rowResult, rowErr := ad.ComputeCandles(candles)
		if mode == 0 || mode == 1 {
			if !columnResult.IsNil() || columnErr == nil {
				t.Fatalf("invalid shape accepted: mode=%d", mode)
			}
			return // The rows remain independently well-formed in these modes.
		}

		if (columnErr == nil) != (rowErr == nil) {
			t.Fatalf("row/column error mismatch: rows=%v columns=%v", rowErr, columnErr)
		}
		if columnErr != nil {
			if !columnResult.IsNil() || !rowResult.IsNil() {
				t.Fatal("failed batch published partial Result")
			}
			if !errors.Is(columnErr, rowErr) {
				t.Fatalf("different error categories: %v vs %v", columnErr, rowErr)
			}
			return
		}
		for i := range columnResult.Values {
			if !columnResult.Valid[i] || !rowResult.Valid[i] ||
				math.Float64bits(columnResult.Values[i]) != math.Float64bits(rowResult.Values[i]) {
				t.Fatalf("row/column divergence at index %d", i)
			}
		}
	})
}
