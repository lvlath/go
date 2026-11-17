// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package obv_test

import (
	"testing"

	"github.com/lvlath/go/obv"
)

// benchmarkCompute measures a complete five-column OBV batch for one price regime.
// It isolates deterministic fixture construction from timed validation, accumulation
// and detached result publication at the requested observation count.
//
// Implementation:
//   - Stage 1: generate coherent positive OHLCV prices and finite volumes.
//   - Stage 2: materialize all five columns before timing begins.
//   - Stage 3: repeatedly measure the public Compute facade and reject failures.
//
// Inputs:
//   - b: the benchmark owner; n: exact candle count; regime: rising, flat or alternating.
//
// Determinism:
//   - Close and Volume sequences use index-only arithmetic without RNG or clocks.
//
// Complexity:
//   - Fixture preparation O(n) time/space, outside the timer; each measured
//     Compute call is O(n) time and O(n) detached output space.
//
// AI-Hints:
//   - Do not move column preparation inside the timed loop or use an invalid
//     candle fixture that accidentally benchmarks only the error fast-path.
func benchmarkCompute(b *testing.B, n int, regime string) {
	c := make([]float64, n)
	v := make([]float64, n)
	for i := range c {
		switch regime {
		case "rising":
			c[i] = float64(i + 1)
		case "flat":
			c[i] = 100
		default:
			if i%2 == 0 {
				c[i] = 100
			} else {
				c[i] = 101
			}
		}
		v[i] = float64(i%10 + 1)
	}
	input := bars(c, v)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := obv.Compute(input)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkCompute1e3Rising measures one thousand consecutive upward closes.
// All accepted transitions add volume, so the accumulator grows monotonically.
func BenchmarkCompute1e3Rising(b *testing.B) { benchmarkCompute(b, 1000, "rising") }

// BenchmarkCompute1e6Rising measures one million upward-close observations.
// It exposes linear scanning and detached output-allocation costs at scale.
func BenchmarkCompute1e6Rising(b *testing.B) { benchmarkCompute(b, 1000000, "rising") }

// BenchmarkCompute1e3Flat measures one thousand equal-close observations.
// It isolates the zero-contribution path while still validating every candle.
func BenchmarkCompute1e3Flat(b *testing.B) { benchmarkCompute(b, 1000, "flat") }

// BenchmarkCompute1e6Flat measures one million equal-close observations.
// It checks the throughput regime where each bar updates only Count and close.
func BenchmarkCompute1e6Flat(b *testing.B) { benchmarkCompute(b, 1000000, "flat") }

// BenchmarkCompute1e3Alternating measures one thousand direction reversals.
// It exercises compensated additions of alternating positive/negative volume.
func BenchmarkCompute1e3Alternating(b *testing.B) { benchmarkCompute(b, 1000, "alternating") }

// BenchmarkCompute1e6Alternating measures one million direction reversals.
// It stresses fixed-size Neumaier state without a growing per-stream ring.
func BenchmarkCompute1e6Alternating(b *testing.B) { benchmarkCompute(b, 1000000, "alternating") }

// BenchmarkPushZeroAlloc measures accepted flat-close Push operations.
// Stream construction is excluded; this baseline primarily checks the hot-path
// overhead of validation, Count advancement and transactional publication.
func BenchmarkPushZeroAlloc(b *testing.B) {
	s := obv.New()
	bar := obv.Bar{1, 1, 1, 1, 0}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, err := s.Push(bar)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkPushAlternatingZeroAlloc measures accepted up/down volume transitions.
// Unlike a flat-close-only benchmark, this traverses the Neumaier arithmetic
// path on every call and excludes stream construction and seeding from the timer.
func BenchmarkPushAlternatingZeroAlloc(b *testing.B) {
	stream := obv.New()
	seed := obv.Bar{Open: 100, High: 100, Low: 100, Close: 100, Volume: 1000}
	if _, _, err := stream.Push(seed); err != nil {
		b.Fatal(err)
	}
	up := obv.Bar{Open: 101, High: 101, Low: 101, Close: 101, Volume: 1e16}
	down := obv.Bar{Open: 100, High: 100, Low: 100, Close: 100, Volume: 1e16}

	b.ReportAllocs()
	b.ResetTimer()
	for index := 0; index < b.N; index++ {
		bar := up
		if index&1 != 0 {
			bar = down
		}
		if _, _, err := stream.Push(bar); err != nil {
			b.Fatal(err)
		}
	}
}
