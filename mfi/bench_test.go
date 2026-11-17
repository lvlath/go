// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Shape-aware, external benchmarks separate batch, steady Push and scheduled rebase.
// Benchmarks never use an erroneous/partial result as a performance measurement.
package mfi_test

import (
	"fmt"
	"sort"
	"testing"
	"time"

	"github.com/lvlath/go/mfi"
)

// benchmarkBars creates a deterministic workload with both directions and zero flow.
// Inputs are built outside the timer to avoid benchmarking fixture generation.
func benchmarkBars(count int) mfi.Bars {
	prices := make([]float64, count)
	volumes := make([]float64, count)
	for i := range prices {
		prices[i] = float64(900 + (i*19)%43)
		volumes[i] = float64((i * 13) % 67)
	}
	return sampleBars(prices, volumes)
}

// BenchmarkCompute measures O(n+Period) batch work for short and long histories.
// No market-data generation or coercion occurs inside the measured loop.
func BenchmarkCompute(b *testing.B) {
	for _, n := range []int{1000, 1000000} {
		for _, period := range []int{14, 1000} {
			b.Run(fmt.Sprintf("n=%d/P=%d", n, period), func(b *testing.B) {
				bars := benchmarkBars(n)
				opts := mfi.Options{Period: period}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if _, err := mfi.Compute(bars, opts); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// BenchmarkStreamPush measures steady-state, already warmed-up rolling Push calls.
// The stimulus alternates prices/volumes; the large-period workload is not
// accidentally measuring the entirely unwarmed initialization branch.
func BenchmarkStreamPush(b *testing.B) {
	for _, period := range []int{14, 1000, mfi.MaxPeriod} {
		b.Run(fmt.Sprintf("P=%d", period), func(b *testing.B) {
			stream, err := mfi.NewPeriod(period)
			if err != nil {
				b.Fatal(err)
			}
			bars := []mfi.Bar{sampleBar(900, 4), sampleBar(903, 9), sampleBar(899, 0), sampleBar(904, 6)}
			for i := 0; i <= period; i++ {
				if _, _, err = stream.Push(bars[i%len(bars)]); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, _, err = stream.Push(bars[i%len(bars)]); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkStreamRebaseLatency samples *only scheduled* O(Period) transitions.
// Restore of the identical pre-boundary checkpoint is excluded from each sample;
// observed p99 is emitted only when at least 100 samples have been collected.
//
// Run with -benchtime=120x to obtain p99 even for P=1<<20; this is an empirical
// latency distribution, not a scheduling guarantee or the amortized Push cost.
func BenchmarkStreamRebaseLatency(b *testing.B) {
	for _, period := range []int{14, 1000, mfi.MaxPeriod} {
		b.Run(fmt.Sprintf("P=%d", period), func(b *testing.B) {
			stream, err := mfi.NewPeriod(period)
			if err != nil {
				b.Fatal(err)
			}
			initial := sampleBar(100, 20)
			for i := 0; i < period; i++ {
				if _, _, err = stream.Push(initial); err != nil {
					b.Fatal(err)
				}
			}
			beforeRebase := stream.Snapshot() // Count=Period: next Push triggers scheduled rebase.
			trigger := sampleBar(101, 30)
			samples := make([]time.Duration, 0, b.N)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				if err = stream.Restore(beforeRebase); err != nil {
					b.Fatal(err)
				}
				b.StartTimer()
				start := time.Now()
				if _, _, err = stream.Push(trigger); err != nil {
					b.Fatal(err)
				}
				samples = append(samples, time.Since(start))
			}
			b.StopTimer()
			sort.Slice(samples, func(i, j int) bool { return samples[i] < samples[j] })
			b.ReportMetric(float64(len(samples)), "scheduled-samples")
			if len(samples) >= 100 {
				// The index is the nearest-rank p99: ceil(0.99*N)-1.
				index := (99*len(samples)+99)/100 - 1
				b.ReportMetric(float64(samples[index].Nanoseconds()), "observed-p99-ns")
			} else if len(samples) > 0 {
				b.ReportMetric(float64(samples[len(samples)-1].Nanoseconds()), "observed-max-ns")
			}
		})
	}
}
