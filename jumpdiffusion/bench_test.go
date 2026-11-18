// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package jumpdiffusion_test benchmarks distinct Poisson and grid regimes.
// Each timed call includes a new per-path seed-derived stream, as in Monte Carlo.
package jumpdiffusion_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/lvlath/go/gbm"
	"github.com/lvlath/go/jumpdiffusion"
)

const (
	benchmarkSeed          = uint64(345)
	benchmarkJumpMean      = -0.001
	benchmarkJumpDeviation = 0.002
)

// BenchmarkSimulate measures complete trajectory generation, not fast-fail preflight.
// Mean 0.1 selects inversion; mean 50 selects PTRS for every uniform grid interval.
//
// Implementation:
//   - Stage 1: build fixed valid grids and physical parameters outside timed loops.
//   - Stage 2: sample independent logical path IDs using the public seeded facade.
//   - Stage 3: reject any unexpected incomplete trajectory without hiding failures.
//
// Behavior highlights: jump deviation is positive, so both count and Gaussian
// aggregation are exercised, rather than silently benchmarking deterministic jumps.
// Inputs: b.N is controlled by testing; grid has 100 or 10000 intervals.
// Returns: benchmark time and allocation data; Errors: b.Fatal on simulation failure.
// Determinism: fixed base seed and monotonically increasing uint64 path IDs.
// Complexity: O(n*B) worst-case per path, O(n) detached output per path.
// Notes: stream construction time is included, consistent with montecarlo.Run.
// AI-Hints: keep model and grid construction outside ResetTimer and verify success.
func BenchmarkSimulate(b *testing.B) {
	for _, intervals := range []int{100, 10_000} {
		for _, meanPerStep := range []float64{0.1, 50} {
			b.Run(fmt.Sprintf("steps=%d/mean=%.1f", intervals, meanPerStep), func(b *testing.B) {
				grid := gbm.TimeGrid{
					Values: make([]float64, intervals+1), Unit: gbm.TimeYears,
				}
				for index := 1; index <= intervals; index++ {
					grid.Values[index] = float64(index) / float64(intervals)
				}

				p := jdParams()
				p.Intensity = meanPerStep * float64(intervals)
				p.LogJumpMean = benchmarkJumpMean
				p.LogJumpStdDev = benchmarkJumpDeviation
				p.Convention = jumpdiffusion.Compensated
				ctx := context.Background()

				b.ReportAllocs()
				b.ResetTimer()
				var pathID uint64
				for iteration := 0; iteration < b.N; iteration++ {
					result, err := jumpdiffusion.SimulateSeeded(ctx, p, grid, benchmarkSeed, pathID)
					if err != nil || result == nil || !result.Complete {
						b.Fatalf("invalid measured path: %v", err)
					}
					pathID++
				}
			})
		}
	}
}
