// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ad_test benchmarks real signed-volume recurrence regimes, not trivial flat bars.
// All fixture construction precedes the timed loop; failures are reported immediately.
package ad_test

import (
	"testing"

	"github.com/lvlath/go/ad"
)

// benchmarkOHLCV produces reproducible, nonflat, changing close locations and volumes.
// A mix of extremes, middle values and signs exercises multiplier and Neumaier paths.
func benchmarkOHLCV(count int) []ad.Bar {
	candles := make([]ad.Bar, count)
	for i := range candles {
		low := 95 + float64((i*19)%47)
		candles[i] = ad.Bar{
			Open: low + 3, Low: low, High: low + 8,
			Close: low + float64((i*7+3)%9), Volume: float64(100 + (i*37)%4000),
		}
	}
	return candles
}

// benchmarkComputeColumns measures a column-oriented complete historical scan.
// It intentionally includes two validation passes and two new result arrays.
func benchmarkComputeColumns(b *testing.B, count int) {
	columns := barsFrom(benchmarkOHLCV(count))
	if _, err := ad.Compute(columns); err != nil {
		b.Fatal(err) // Assert setup shape and arithmetic before resetting the clock.
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ad.Compute(columns); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkCompute_1e3 measures 1000 nonflat, mixed-pressure historical bars.
func BenchmarkCompute_1e3(b *testing.B) { benchmarkComputeColumns(b, 1000) }

// BenchmarkCompute_1e6 measures the same recurrence on one million historical bars.
func BenchmarkCompute_1e6(b *testing.B) { benchmarkComputeColumns(b, 1_000_000) }

// benchmarkComputeRows isolates []Bar input cost without a transposition stage.
func benchmarkComputeRows(b *testing.B, count int) {
	candles := benchmarkOHLCV(count)
	if _, err := ad.ComputeCandles(candles); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ad.ComputeCandles(candles); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkComputeCandles_1e3 compares 1000 row-oriented bars against column scans.
func BenchmarkComputeCandles_1e3(b *testing.B) { benchmarkComputeRows(b, 1000) }

// BenchmarkComputeCandles_1e6 tests a million rows with no intermediate OHLCV columns.
func BenchmarkComputeCandles_1e6(b *testing.B) { benchmarkComputeRows(b, 1_000_000) }

// BenchmarkPush_VariablePressure measures the hot path for alternating realistic bar positions.
// The input buffer and Stream are prepared before timing; each iteration accepts nonzero volume.
func BenchmarkPush_VariablePressure(b *testing.B) {
	candles := benchmarkOHLCV(1024)
	stream := ad.New()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := stream.Push(candles[i%len(candles)]); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkPush_SignedCancellation isolates low-order correction under alternating huge contributions.
// The stream receives +1e16,+1,-1e16 repeatedly; it must preserve the small residue.
func BenchmarkPush_SignedCancellation(b *testing.B) {
	candles := []ad.Bar{
		{Open: 1.5, High: 2, Low: 1, Close: 2, Volume: 1e16},
		{Open: 1.5, High: 2, Low: 1, Close: 2, Volume: 1},
		{Open: 1.5, High: 2, Low: 1, Close: 1, Volume: 1e16},
	}
	stream := ad.New()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := stream.Push(candles[i%len(candles)]); err != nil {
			b.Fatal(err)
		}
	}
}
