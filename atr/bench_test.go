// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package atr_test benchmarks owning batch histories and fixed-state stream updates.
// AI-Hints: all fixture building and warmup precede ResetTimer; retain error checks.
package atr_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/lvlath/go/atr"
)

// BenchmarkCompute_OrderedGapSeries measures owning batch computation across the declared size/period matrix.
// n=1e3/1e5/1e6 and P=14/64/512 with changing prior-close gaps.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - n=1e3/1e5/1e6 and P=14/64/512 with changing prior-close gaps.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - Do not include fixture generation inside the timed loop.
func BenchmarkCompute_OrderedGapSeries(b *testing.B) {
	for _, n := range []int{1000, 100000, 1000000} {
		for _, period := range []int{14, 64, 512} {
			b.Run(fmt.Sprintf("n%d/P%d", n, period), func(b *testing.B) {
				high, low, closed := marketBars(n)
				opts := atr.Options{Period: period}
				ctx := context.Background()
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					result, err := atr.Compute(ctx, high, low, closed, opts)
					if err != nil {
						b.Fatal(err)
					}
					benchmarkResult = result
				}
			})
		}
	}
}

// BenchmarkStreamPush_Mature measures mature constant-range recurrence without setup allocation.
// independent warmed streams at P=14/64/512.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - independent warmed streams at P=14/64/512.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - Report this constant workload separately from changing gaps.
func BenchmarkStreamPush_Mature(b *testing.B) {
	for _, period := range []int{14, 64, 512} {
		b.Run(fmt.Sprintf("P%d", period), func(b *testing.B) {
			stream := newStream(b, period)
			ctx := context.Background()
			for i := 0; i < period; i++ {
				push(b, stream, candle{103, 99, 101})
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				value, err := stream.Push(ctx, 103, 99, 101)
				if err != nil {
					b.Fatal(err)
				}
				benchmarkValue = value
			}
		})
	}
}

// BenchmarkStreamPush_Warmup measures exact seed accumulation before readiness.
// a very large period and fixed representable ranges.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - a very large period and fixed representable ranges.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - The measured operation is Push, not New or Snapshot.
func BenchmarkStreamPush_Warmup(b *testing.B) {
	stream := newStream(b, int(^uint(0)>>1))
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		value, err := stream.Push(ctx, 103, 99, 101)
		if err != nil {
			b.Fatal(err)
		}
		benchmarkValue = value
	}
}

// BenchmarkStreamPush_ChangingGap measures a mature stream whose range changes every bar.
// The bounded deterministic fixture exercises prior-close selection, not a constant steady value.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - The bounded deterministic fixture exercises prior-close selection, not a constant steady value.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - Preserve exact payload, mask, error identity, and ownership assertions.
func BenchmarkStreamPush_ChangingGap(b *testing.B) {
	const fixtureSize = 128
	high, low, closed := marketBars(fixtureSize)
	stream := newStream(b, 14)
	ctx := context.Background()
	for i := 0; i < 14; i++ {
		push(b, stream, candle{high[i], low[i], closed[i]})
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		index := i % fixtureSize
		value, err := stream.Push(ctx, high[index], low[index], closed[index])
		if err != nil {
			b.Fatal(err)
		}
		benchmarkValue = value
	}
}
