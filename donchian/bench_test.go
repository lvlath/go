// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package donchian_test measures declared workloads without timing fixture generation.
// This file uses only exported production operations and preserves explicit fixture ownership.
// AI-Hints: Keep all construction and oracle work outside b.ResetTimer and the timed loop.
package donchian_test

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/lvlath/go/donchian"
)

// BenchmarkCompute measures the specified batch sizes and ordinary window periods.
// Generated input is outside timing; all returned allocations remain part of the measured call.
//
// Implementation:
//   - Stage 1: generate and retain deterministic input outside the timer.
//   - Stage 2: reset timing and repeatedly execute the named public operation.
//   - Stage 3: keep the owning result live and report time, bytes, and allocations.
//
// Behavior highlights:
//   - Generated input is outside timing; all returned allocations remain part of the measured call.
//
// Inputs:
//   - n=1000,100000,1000000 and P=14,64,512 in both modes.
//
// Returns:
//   - Benchmark measurements; computation failures stop the benchmark.
//
// Errors:
//   - Every unexpected API error calls b.Fatal; failing-fast input is not measured as success.
//
// Determinism:
//   - Each repeated measurement reuses the same fixed input; timing itself is environment-dependent.
//
// Complexity:
//   - O(n) per measured operation, including output initialization.
//
// Notes:
//   - Measurements depend on toolchain, CPU, scheduler, and allocator; record the environment.
//
// AI-Hints:
//   - Do not include fixture generation or the quadratic oracle inside the timed loop.
func BenchmarkCompute(b *testing.B) {
	for _, n := range []int{1000, 100000, 1000000} {
		high, low := syntheticBars(n)
		for _, period := range []int{14, 64, 512} {
			for _, exclude := range []bool{false, true} {
				b.Run(fmt.Sprintf("n=%d/P=%d/exclude=%t", n, period, exclude), func(b *testing.B) {
					opts := donchian.Options{Period: period, ExcludeCurrent: exclude}
					ctx := context.Background()
					b.ReportAllocs()
					b.SetBytes(int64(n) * pricePairBytes)
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						result, err := donchian.Compute(ctx, high, low, opts)
						if err != nil {
							b.Fatal(err)
						}
						benchmarkResult = result
					}
				})
			}
		}
	}
}

// BenchmarkComputeMaskedMaxInt measures the wholly masked maximal-period path.
// Two observations must not request period-sized deque storage.
//
// Implementation:
//   - Stage 1: generate and retain deterministic input outside the timer.
//   - Stage 2: reset timing and repeatedly execute the named public operation.
//   - Stage 3: keep the owning result live and report time, bytes, and allocations.
//
// Behavior highlights:
//   - Two observations must not request period-sized deque storage.
//
// Inputs:
//   - a fixed two-row history with Period=MaxInt and ExcludeCurrent=true.
//
// Returns:
//   - Benchmark measurements; computation failures stop the benchmark.
//
// Errors:
//   - Every unexpected API error calls b.Fatal; failing-fast input is not measured as success.
//
// Determinism:
//   - Each repeated measurement reuses the same fixed input; timing itself is environment-dependent.
//
// Complexity:
//   - O(n) work and O(n) outputs, without queue allocation.
//
// Notes:
//   - Measurements depend on toolchain, CPU, scheduler, and allocator; record the environment.
//
// AI-Hints:
//   - Report bytes as well as allocations; object counts do not describe buffer lengths.
func BenchmarkComputeMaskedMaxInt(b *testing.B) {
	high, low := []float64{2, 4}, []float64{1, 3}
	opts := donchian.Options{Period: math.MaxInt, ExcludeCurrent: true}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result, err := donchian.Compute(ctx, high, low, opts)
		if err != nil {
			b.Fatal(err)
		}
		benchmarkResult = result
	}
}

// BenchmarkComputeDequeRegimes measures candidate occupancy and repeated eviction bursts.
// Ascending, descending, equal, and bursting histories load distinct queue regimes.
//
// Implementation:
//   - Stage 1: generate and retain deterministic input outside the timer.
//   - Stage 2: reset timing and repeatedly execute the named public operation.
//   - Stage 3: keep the owning result live and report time, bytes, and allocations.
//
// Behavior highlights:
//   - Ascending, descending, equal, and bursting histories load distinct queue regimes.
//
// Inputs:
//   - n=100000, P=4096, five deterministic regimes, and both modes.
//
// Returns:
//   - Benchmark measurements; computation failures stop the benchmark.
//
// Errors:
//   - Every unexpected API error calls b.Fatal; failing-fast input is not measured as success.
//
// Determinism:
//   - Each repeated measurement reuses the same fixed input; timing itself is environment-dependent.
//
// Complexity:
//   - O(n) amortized work per call; a single insertion may cost O(P).
//
// Notes:
//   - Measurements depend on toolchain, CPU, scheduler, and allocator; record the environment.
//
// AI-Hints:
//   - Benchmark names must describe the actual queue regime, not merely an input size.
func BenchmarkComputeDequeRegimes(b *testing.B) {
	const n = 100000
	const period = 4096
	ascendingHigh, ascendingLow := monotoneBars(n, false)
	descendingHigh, descendingLow := monotoneBars(n, true)
	upperHigh, upperLow := burstCycleBars(n, period, true)
	lowerHigh, lowerLow := burstCycleBars(n, period, false)
	cases := []struct {
		name      string
		high, low []float64
	}{
		{"ascending-full-min", ascendingHigh, ascendingLow},
		{"descending-full-max", descendingHigh, descendingLow},
		{"equal-latest-only", make([]float64, n), make([]float64, n)},
		{"repeated-max-burst", upperHigh, upperLow},
		{"repeated-min-burst", lowerHigh, lowerLow},
	}
	for _, test := range cases {
		for _, exclude := range []bool{false, true} {
			b.Run(fmt.Sprintf("%s/exclude=%t", test.name, exclude), func(b *testing.B) {
				opts := donchian.Options{Period: period, ExcludeCurrent: exclude}
				ctx := context.Background()
				b.ReportAllocs()
				b.SetBytes(int64(n) * pricePairBytes)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					result, err := donchian.Compute(ctx, test.high, test.low, opts)
					if err != nil {
						b.Fatal(err)
					}
					benchmarkResult = result
				}
			})
		}
	}
}

// BenchmarkComputeModeWrappers measures the explicit mode wrappers on the same ordinary workload.
// Each facade receives identical price inputs and the canonical benchmark period.
//
// Implementation:
//   - Stage 1: generate and retain deterministic input outside the timer.
//   - Stage 2: reset timing and repeatedly execute the named public operation.
//   - Stage 3: keep the owning result live and report time, bytes, and allocations.
//
// Behavior highlights:
//   - Each facade receives identical price inputs and the canonical benchmark period.
//
// Inputs:
//   - n=100000, P=64, and both named wrappers.
//
// Returns:
//   - Benchmark measurements; computation failures stop the benchmark.
//
// Errors:
//   - Every unexpected API error calls b.Fatal; failing-fast input is not measured as success.
//
// Determinism:
//   - Each repeated measurement reuses the same fixed input; timing itself is environment-dependent.
//
// Complexity:
//   - O(n) per operation including canonical output allocation.
//
// Notes:
//   - Measurements depend on toolchain, CPU, scheduler, and allocator; record the environment.
//
// AI-Hints:
//   - The function-value call belongs to this harness; do not infer nanosecond overhead from one run.
func BenchmarkComputeModeWrappers(b *testing.B) {
	const n = 100000
	const period = 64
	high, low := syntheticBars(n)
	for _, facade := range modeFacades() {
		b.Run(facade.name, func(b *testing.B) {
			ctx := context.Background()
			b.ReportAllocs()
			b.SetBytes(int64(n) * pricePairBytes)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				result, err := facade.call(ctx, high, low, period)
				if err != nil {
					b.Fatal(err)
				}
				benchmarkResult = result
			}
		})
	}
}
