// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw_test benchmarks fully validated exact fixed-window searches.
// Each case fixes query length, candidate count, radius, and worker policy.
package dtw_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/lvlath/go/dtw"
)

// BenchmarkSearch_ExactBands measures exact search, including finite validation,
// candidate enumeration, row workspace, ranking, and publication.
// It reports visited DP cells as an algorithmic workload counter in addition to ns/op.
//
// Implementation:
//   - Stage 1: build deterministic scalar query/data outside the timed region.
//   - Stage 2: execute Search with fixed radius, TopK, Workers, and normalization.
//   - Stage 3: report cells/op and compiler-visible result consumption.
//
// Behavior highlights:
//   - Query and data setup are excluded; all required Search validation is included.
//
// Inputs:
//   - testing.B: benchmark harness with stable CPU/toolchain configuration.
//
// Returns:
//   - Standard benchmark metrics and exact work cells/op.
//
// Errors:
//   - Any failed search or incomplete result fails the case immediately.
//
// Determinism:
//   - Data and ranking are fixed; no randomness or time-based selection.
//
// Complexity:
//   - Workload per run O(C*L*b), sorting/heap O(C log TopK).
//
// Notes:
//   - L1024 unconstrained cases use a short dataset to avoid impractical C*L² work.
//   - Compare only identical toolchain/CPU, input, radius, and normalization.
//
// AI-Hints:
//   - Do not use L1024,N100000,r=-1 as a routine benchmark default.
//   - A lower ns/op is not admissible if the oracle or visited-cell counts change.
func BenchmarkSearch_ExactBands(b *testing.B) {
	cases := []struct {
		length, samples, radius int
	}{
		{64, 1000, 0}, {64, 1000, 8}, {64, 1000, 32}, {64, 1000, -1},
		{256, 1000, 0}, {256, 1000, 8}, {256, 1000, 32},
		{1024, 1056, 0}, {1024, 1056, 8}, {1024, 1056, 32},
		{64, 100000, 0}, {64, 100000, 8},
	}
	for _, tc := range cases {
		b.Run(fmt.Sprintf("L%d_N%d_R%d", tc.length, tc.samples, tc.radius), func(b *testing.B) {
			query := make([]float64, tc.length)
			data := make([]float64, tc.samples)
			for i := range query {
				query[i] = float64((i*7)%23 - 11)
			}
			for i := range data {
				data[i] = float64((i*11)%29 - 14)
			}
			opts := dtw.DefaultSearchOptions()
			opts.Radius = tc.radius
			opts.TopK = 3
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				result, err := dtw.Search(context.Background(), query, data, opts)
				if err != nil || result == nil || !result.Complete {
					b.Fatalf("search incomplete: result=%+v err=%v", result, err)
				}
				b.ReportMetric(float64(result.CellsVisited), "cells/op")
			}
		})
	}
}
