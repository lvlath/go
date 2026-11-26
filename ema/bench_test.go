// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ema_test benchmarks exact EMA with all fixture generation outside timing.
// Allocation counts exclude New, input generation, snapshot I/O, and reporting.
package ema_test

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/lvlath/go/ema"
)

// BenchmarkCompute measures owning exact batch histories across the requested size/period grid.
// It isolates the named numerical regime rather than mixing setup with processing.
//
// Implementation:
//   - Stage 1: Generate n=1e3/1e5/1e6 finite observations before subbenchmark timing.
//   - Stage 2: report allocations and reset the measurement timer.
//   - Stage 3: Call Compute with P=14/64/512 and retain the result in a benchmark sink.
//
// Behavior highlights:
//   - SMA seeding plus mature traversal, including owning output allocation.
//
// Inputs:
//   - b: benchmark controller; b.N counts actual successful operations.
//
// Returns:
//   - ns/op, B/op, and allocs/op from the Go benchmark harness.
//
// Errors:
//   - Any constructor or timed operation error fails the benchmark rather than measuring rejection.
//
// Determinism:
//   - Fixed deterministic fixtures and ascending replay order; no hidden random setup.
//
// Complexity:
//   - Time O(n*W), Space O(n+W) per operation; W=34.
//
// Notes:
//   - CPU, toolchain, and runtime settings must accompany comparisons.
//
// AI-Hints:
//   - New input generation, fmt names, and configuration are excluded by ResetTimer.
func BenchmarkCompute(b *testing.B) {
	ctx := context.Background()
	for _, n := range []int{1000, 100000, 1000000} {
		values := benchmarkHistory(n)
		for _, period := range []int{14, 64, 512} {
			b.Run(fmt.Sprintf("n=%d/P=%d", n, period), func(b *testing.B) {
				opts := ema.Options{Period: period}
				b.ReportAllocs()
				b.SetBytes(int64(n) * 8)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					result, err := ema.Compute(ctx, values, opts)
					if err != nil {
						b.Fatal(err)
					}
					benchmarkResult = result
				}
			})
		}
	}
}

// BenchmarkPush measures mature streaming on a fixed price-like observation replay.
// It isolates the named numerical regime rather than mixing setup with processing.
//
// Implementation:
//   - Stage 1: Create an initialized SMA stream and complete its warmup before timing.
//   - Stage 2: report allocations and reset the measurement timer.
//   - Stage 3: Call Push once per timed operation and retain the detached output.
//
// Behavior highlights:
//   - Periods 14, 64, and 512 after readiness.
//
// Inputs:
//   - b: benchmark controller; b.N counts actual successful operations.
//
// Returns:
//   - ns/op, B/op, and allocs/op from the Go benchmark harness.
//
// Errors:
//   - Any constructor or timed operation error fails the benchmark rather than measuring rejection.
//
// Determinism:
//   - Fixed deterministic fixtures and ascending replay order; no hidden random setup.
//
// Complexity:
//   - Time O(W), Space O(W) fixed candidate, W=34; measure B/op and allocs/op.
//
// Notes:
//   - CPU, toolchain, and runtime settings must accompany comparisons.
//
// AI-Hints:
//   - Keep Snapshot, formatting, and constructor calls outside the timed loop.
func BenchmarkPush(b *testing.B) {
	ctx := context.Background()
	values := benchmarkHistory(256)
	for _, period := range []int{14, 64, 512} {
		b.Run(fmt.Sprintf("P=%d", period), func(b *testing.B) {
			stream := newStream(b, ema.Options{Period: period})
			for i := 0; i < period; i++ {
				pushValue(b, stream, values[i%len(values)])
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				value, err := stream.Push(ctx, values[i%len(values)])
				if err != nil {
					b.Fatal(err)
				}
				benchmarkValue = value
			}
		})
	}
}

// BenchmarkPushWarmup measures exact accumulation before SMA readiness.
// It isolates the named numerical regime rather than mixing setup with processing.
//
// Implementation:
//   - Stage 1: Select the maximum admitted period so realistic benchmark runs stay in warmup.
//   - Stage 2: report allocations and reset the measurement timer.
//   - Stage 3: Push the same exactly represented finite observation once per operation.
//
// Behavior highlights:
//   - Long exact seed accumulation, not a mature EMA update.
//
// Inputs:
//   - b: benchmark controller; b.N counts actual successful operations.
//
// Returns:
//   - ns/op, B/op, and allocs/op from the Go benchmark harness.
//
// Errors:
//   - Any constructor or timed operation error fails the benchmark rather than measuring rejection.
//
// Determinism:
//   - Fixed deterministic fixtures and ascending replay order; no hidden random setup.
//
// Complexity:
//   - Time O(W), Space O(W), W=34; no history-length growth.
//
// Notes:
//   - CPU, toolchain, and runtime settings must accompany comparisons.
//
// AI-Hints:
//   - This regime is different from mature affine division and must be labeled separately.
func BenchmarkPushWarmup(b *testing.B) {
	ctx := context.Background()
	stream := newStream(b, ema.Options{Period: math.MaxInt - 1})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		value, err := stream.Push(ctx, 62000.125)
		if err != nil {
			b.Fatal(err)
		}
		benchmarkValue = value
	}
}

// BenchmarkPushAdversarial measures mature recurrence over sign and exponent extremes.
// It isolates the named numerical regime rather than mixing setup with processing.
//
// Implementation:
//   - Stage 1: Prepare signed extreme/subnormal inputs and initialize a First-seeded stream.
//   - Stage 2: report allocations and reset the measurement timer.
//   - Stage 3: Alternate operands through half-weight, moderate, and maximal-period recurrences.
//
// Behavior highlights:
//   - Cancellation, large exponent gaps, and very small exact new-sample weights.
//
// Inputs:
//   - b: benchmark controller; b.N counts actual successful operations.
//
// Returns:
//   - ns/op, B/op, and allocs/op from the Go benchmark harness.
//
// Errors:
//   - Any constructor or timed operation error fails the benchmark rather than measuring rejection.
//
// Determinism:
//   - Fixed deterministic fixtures and ascending replay order; no hidden random setup.
//
// Complexity:
//   - Time O(W), Space O(W), W=34; output allocation is excluded by value return.
//
// Notes:
//   - CPU, toolchain, and runtime settings must accompany comparisons.
//
// AI-Hints:
//   - Near-equal positive operands alone cannot reveal all exact-arithmetic workload regimes.
func BenchmarkPushAdversarial(b *testing.B) {
	ctx := context.Background()
	values := []float64{math.MaxFloat64, -math.MaxFloat64, math.SmallestNonzeroFloat64, -math.SmallestNonzeroFloat64, 1, -1, 0}
	for _, period := range []int{3, 64, math.MaxInt - 1} {
		b.Run(fmt.Sprintf("P=%d", period), func(b *testing.B) {
			stream := newStream(b, ema.Options{Period: period, Seed: ema.SeedFirst})
			pushValue(b, stream, values[0])
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				value, err := stream.Push(ctx, values[i%len(values)])
				if err != nil {
					b.Fatal(err)
				}
				benchmarkValue = value
			}
		})
	}
}
