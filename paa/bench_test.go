// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
//
// File: bench_test.go. Linear-sweep workload/allocations in six shape regimes.
// AI-Hints: construct data before timer; do not measure malformed-input fast errors.
package paa_test

import (
	"fmt"
	"testing"

	"github.com/lvlath/go/paa"
)

var benchmarkResult *paa.Result

// BenchmarkTransform compares one global cell, 64 cells, and detached identity
// for both short and million-sample signals. Reported allocations must scale
// with output length, not with a materialized n*w weight matrix.
func BenchmarkTransform(b *testing.B) {
	for _, n := range []int{1_000, 1_000_000} {
		values := make([]float64, n)
		for i := range values {
			values[i] = float64(i%257) - 128
		}
		for _, w := range []int{1, 64, n} {
			b.Run(fmt.Sprintf("n%d_w%d", n, w), func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					result, err := paa.Transform(values, w)
					if err != nil {
						b.Fatal(err)
					}
					benchmarkResult = result
				}
			})
		}
	}
}

var benchmarkMeans []float64

// BenchmarkTransformRange measures checked subrange selection and its canonical
// PAA call, excluding construction of the full caller-owned telemetry trace.
func BenchmarkTransformRange(b *testing.B) {
	values := make([]float64, 1_000)
	for i := range values {
		values[i] = float64(i%127) - 63
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result, err := paa.TransformRange(values, 73, 927, 64)
		if err != nil {
			b.Fatal(err)
		}
		benchmarkResult = result
	}
}

// BenchmarkMeans measures the honest value-only projection, including the
// O(w) geometry allocation performed by the underlying canonical facade.
func BenchmarkMeans(b *testing.B) {
	values := make([]float64, 1_000)
	for i := range values {
		values[i] = float64(i%127) - 63
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result, err := paa.Means(values, 64)
		if err != nil {
			b.Fatal(err)
		}
		benchmarkMeans = result
	}
}
