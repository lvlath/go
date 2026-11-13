// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package bootstrap_test benchmarks allocation and index throughput across
// realistic time-series sizes, requested row lengths and replicate counts.
package bootstrap_test

import (
	"context"
	"strconv"
	"testing"

	"github.com/lvlath/go/bootstrap"
)

// benchmarkResult is a package-scoped sink to preserve all measured results.
var benchmarkResult *bootstrap.Result

// BenchmarkSampleIndices isolates the canonical sampling facade from fixture
// creation and verifies that every timed call completes the requested workload.
//
// Regimes:
//   - Short and large source histories: rows=100 or rows=10,000.
//   - Short and medium output paths: Length=100 or Length=1,000.
//   - Small and large ensembles: Replicates=10 or Replicates=1,000.
//   - All three block transition laws with BlockLength=min(rows,32).
//
// Complexity:
//   - Each iteration emits exactly Length*Replicates ints; B/op includes owned rows.
//
// AI-Hints:
//   - Report ns/op, B/op and allocs/op; do not benchmark allocation failure.
//   - Compare against the same RNG version, Go toolchain and CPU.
func BenchmarkSampleIndices(b *testing.B) {
	for _, kind := range []bootstrap.Kind{
		bootstrap.MovingBlocks, bootstrap.CircularBlocks, bootstrap.StationaryBlocks,
	} {
		for _, rows := range []int{100, 10_000} {
			for _, length := range []int{100, 1000} {
				for _, replicates := range []int{10, 1000} {
					label := kindName(kind) + "/n=" + strconv.Itoa(rows) +
						"/L=" + strconv.Itoa(length) + "/R=" + strconv.Itoa(replicates)
					b.Run(label, func(b *testing.B) {
						plan := bootstrap.Plan{
							Kind: kind, Length: length, BlockLength: 32,
							Replicates: replicates, Seed: 0,
						}
						ctx := context.Background()
						b.ReportAllocs()
						b.ResetTimer()
						for iteration := 0; iteration < b.N; iteration++ {
							result, err := bootstrap.SampleIndices(ctx, rows, plan)
							if err != nil {
								b.Fatal(err)
							}
							if result.Completed != replicates {
								b.Fatalf("completed %d of %d", result.Completed, replicates)
							}
							benchmarkResult = result
						}
					})
				}
			}
		}
	}
}

// BenchmarkSampleIndices_IntegerMapping measures the B=1 regime in which
// every destination requires a separately mapped bounded integer.
// This is the workload most sensitive to native-int reconstruction overhead.
//
// Implementation:
//   - Stage 1: Fix source domains from a small catalog up to native MaxInt.
//   - Stage 2: Generate an exact number of indices via the canonical facade.
//   - Stage 3: Retain the result to prevent elimination by the compiler.
//
// Behavior highlights:
//   - Each call produces 4*1024 indices, with one accepted integer per index.
//   - Large source domains increase conversion width without increasing output.
//
// Determinism:
//   - A fixed Seed and serial per-replicate streams make measured work repeatable.
//
// Complexity:
//   - Expected O(4096) RNG words per call and O(4096) caller-owned output space;
//     rejection can consume more words for non-power-of-two bounds.
//
// Notes:
//   - Compare only under the same Go toolchain, hardware and RNG factory revision.
//
// AI-Hints:
//   - Do not benchmark the rejected-allocation branch or use a time-based seed.
func BenchmarkSampleIndices_IntegerMapping(b *testing.B) {
	maxInt := int(^uint(0) >> 1)
	for _, rows := range []int{17, 10_000, maxInt} {
		b.Run("n="+strconv.Itoa(rows), func(b *testing.B) {
			plan := bootstrap.Plan{
				Kind: bootstrap.MovingBlocks, Length: 1024,
				BlockLength: 1, Replicates: 4, Seed: 0,
			}
			ctx := context.Background()
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				result, err := bootstrap.SampleIndices(ctx, rows, plan)
				if err != nil {
					b.Fatal(err)
				}
				benchmarkResult = result
			}
		})
	}
}
