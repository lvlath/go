// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package macd_test measures actual work regimes rather than constructor
// setup: default and Signal=1 batches plus the single-price stream hot path.
package macd_test

import (
	"testing"

	"github.com/lvlath/go/macd"
)

// benchmarkCompute Builds n real positive close observations outside the benchmark timer, then measures complete Compute calls.
func benchmarkCompute(b *testing.B, n int, o macd.Options) {
	prices := make([]float64, n)
	for i := range prices {
		prices[i] = 100 + float64(i%101)/10
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := macd.Compute(prices, o); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkCompute_Default_1e3 Measures conventional 12/26/9 over one thousand closes.
func BenchmarkCompute_Default_1e3(b *testing.B) { benchmarkCompute(b, 1000, macd.Options{}) }

// BenchmarkCompute_Default_1e6 Measures conventional 12/26/9 over one million closes.
func BenchmarkCompute_Default_1e6(b *testing.B) { benchmarkCompute(b, 1000000, macd.Options{}) }

// BenchmarkCompute_SignalOne_1e3 Measures Signal=1 over one thousand closes, including full result materialization.
func BenchmarkCompute_SignalOne_1e3(b *testing.B) { benchmarkCompute(b, 1000, macd.Options{Signal: 1}) }

// BenchmarkCompute_SignalOne_1e6 Measures Signal=1 over one million closes, including full result materialization.
func BenchmarkCompute_SignalOne_1e6(b *testing.B) {
	benchmarkCompute(b, 1000000, macd.Options{Signal: 1})
}

// BenchmarkStream_Push Measures the successful per-price hot path without snapshot or feed-adapter allocations.
func BenchmarkStream_Push(b *testing.B) {
	s, err := macd.New(macd.Options{})
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err = s.Push(100 + float64(i%101)/10); err != nil {
			b.Fatal(err)
		}
	}
}
