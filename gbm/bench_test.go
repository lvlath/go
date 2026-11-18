// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gbm_test benchmarks actual irregular-grid trajectory generation.
// Construction and RNG creation occur before the measured call sequence.
package gbm_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/lvlath/go/gbm"
	"github.com/lvlath/go/rng"
)

// BenchmarkSimulate measures complete trajectory publication at two useful sizes.
// Small grids approximate burst/option scenarios; 10k points stress retention,
// wide numeric operations, and the output copy on a long irregular time series.
//
// Implementation:
//   - Stage 1: prepare explicit strictly increasing nonuniform second timestamps.
//   - Stage 2: construct a reproducible RNG stream before timing starts.
//   - Stage 3: measure only repeated GBM calls with error checks.
//
// Behavior highlights:
//   - Sigma=0 benchmarks the analytic zero-entropy mode.
//   - Sigma>0 measures the real Normal interface and lognormal numeric kernel.
//
// Inputs: standard testing.B runner; fixed lengths 100 and 10000, fixed seed.
// Returns: ns/op, B/op, allocs/op measured by the Go benchmark harness.
// Errors: any simulation error fails the benchmark; invalid setup is forbidden.
// Determinism: fixed grid and stream seed; measured loop uses the same stream
//
//	sequentially to isolate hot-path cost, not to define independent MC paths.
//
// Complexity: Each iteration O(n) time and O(n) owned result memory.
// Notes: output buffers are intentionally allocated because Result owns them.
// AI-Hints: benchmark results from different Go architectures are not interchangeable.
func BenchmarkSimulate(b *testing.B) {
	for _, length := range []int{100, 10000} {
		for _, sigma := range []float64{0, 0.0004} {
			name := fmt.Sprintf("irregular_seconds/n=%d/sigma=%g", length, sigma)
			b.Run(name, func(b *testing.B) {
				times := make([]float64, length)
				for index := 1; index < length; index++ {
					// The increments cycle from 1 ms to 7 ms, representing an
					// irregular but strictly ordered event-time observation grid.
					times[index] = times[index-1] + float64(1+index%7)/1000
				}
				grid := gbm.TimeGrid{Values: times, Unit: gbm.TimeSeconds}
				params := gbm.Parameters{
					Initial: 120.25, Drift: 0.000001, Volatility: sigma, Unit: gbm.TimeSeconds,
				}
				stream := rng.NewStream(20261118, 17)
				ctx := context.Background()
				b.ReportAllocs()
				b.ResetTimer()
				for iteration := 0; iteration < b.N; iteration++ {
					_, err := gbm.Simulate(ctx, params, grid, stream)
					if err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
