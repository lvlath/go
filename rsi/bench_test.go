// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package rsi_test benchmarks explicit feed regimes, not arbitrary series sizes alone.
// All generation, initialization, and warmup happen outside timed regions; sinks prevent dead-result elimination.
package rsi_test

import (
	"fmt"
	"testing"

	"github.com/lvlath/go/rsi"
)

// Global sinks keep the constructed stream/result and final scalar observable.
// Benchmark-only state is never imported into the production package.
var (
	benchmarkResultSink *rsi.Result
	benchmarkStreamSink *rsi.Stream
	benchmarkScalarSink float64
)

// benchmarkClosingPrices builds one reproducible feed regime of exactly n positive prices.
// The names describe the numeric branches exercised by the benchmark rather than fake market provenance.
func benchmarkClosingPrices(n int, regime string) []float64 {
	prices := make([]float64, n)
	for i := range prices {
		switch regime {
		case "flat":
			prices[i] = 64120
		case "drift":
			prices[i] = 64120 + float64(i)/10
		default: // alternating: repeated gain/loss updates representative of a choppy feed.
			prices[i] = 64120 + float64((i*127)%997) - float64((i*29)%203)
		}
	}
	return prices
}

// BenchmarkCompute measures end-to-end validation, seed/recurrence, and detached publication.
// n=1000 with Period=1000 intentionally measures a complete but never-ready seed regime.
// n=1000000 measures steady state overwhelmingly; regimes exercise flat, drift, and two-sided means.
func BenchmarkCompute(b *testing.B) {
	for _, n := range []int{1000, 1000000} {
		for _, period := range []int{1, 14, 1000} {
			for _, regime := range []string{"alternating", "drift", "flat"} {
				name := fmt.Sprintf("n=%d/P=%d/%s", n, period, regime)
				b.Run(name, func(b *testing.B) {
					prices := benchmarkClosingPrices(n, regime)
					opts := rsi.Options{Period: period}
					b.ReportAllocs()
					b.ResetTimer()
					var last *rsi.Result
					for i := 0; i < b.N; i++ {
						var err error
						last, err = rsi.Compute(prices, opts)
						if err != nil {
							b.Fatal(err)
						}
					}
					b.StopTimer()
					benchmarkResultSink = last
				})
			}
		}
	}
}

// BenchmarkStreamPush measures only an initialized, fully warmed streaming transition.
// Construction, checkpoint hashing, and warmup are excluded; report allocations per Push.
func BenchmarkStreamPush(b *testing.B) {
	for _, period := range []int{1, 14, 1000} {
		for _, regime := range []string{"alternating", "flat"} {
			b.Run(fmt.Sprintf("P=%d/%s", period, regime), func(b *testing.B) {
				stream, err := rsi.NewPeriod(period)
				if err != nil {
					b.Fatal(err)
				}
				prices := benchmarkClosingPrices(period+1, regime)
				for _, price := range prices {
					if _, _, err := stream.Push(price); err != nil {
						b.Fatal(err)
					}
				}
				b.ReportAllocs()
				b.ResetTimer()
				var value float64
				for i := 0; i < b.N; i++ {
					price := 64120.0
					if regime == "alternating" {
						price += float64((i%17)*7) - float64((i%5)*2)
					}
					var valid bool
					value, valid, err = stream.Push(price)
					if err != nil || !valid {
						b.Fatalf("hot path not ready: valid=%t err=%v", valid, err)
					}
				}
				b.StopTimer()
				benchmarkScalarSink = value
				benchmarkStreamSink = stream
			})
		}
	}
}

// BenchmarkNew measures configured stream allocation and SHA-256 setup independently of Push.
// Every new stream is published to a global sink so an optimizer cannot erase construction.
func BenchmarkNew(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stream, err := rsi.NewDefault()
		if err != nil {
			b.Fatal(err)
		}
		benchmarkStreamSink = stream
	}
}
