// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pacf_test benchmarks a stationary finite-sample AR(2) workload.
// Sample generation and benchmark naming occur outside measured Compute calls.
package pacf_test

import (
	"context"
	"fmt"
	"math/rand"
	"testing"

	"github.com/lvlath/go/pacf"
)

// BenchmarkCompute measures validation, scaled common-denominator covariance
// scans, and Levinson recursion for short/long input windows and lag profiles.
// The fixed-seed AR(2) input is well conditioned and representative of a
// stationary forecast-residual identification workload; all setups are unmeasured.
//
// Regimes:
//   - n=1e3 or 1e5 observations; MaxLag=1,20,200 (inclusive).
//   - Both Center=false and Center=true; same signal and statistical convention.
//
// AI-Hints:
//   - Do not move fixture generation into the b.N loop or measure an error fast-path.
//   - Keep b.ReportAllocs and never compare benchmarks with different lag/sample shapes.
func BenchmarkCompute(b *testing.B) {
	for _, sampleCount := range []int{1000, 100000} {
		// A private seed fixes the innovation stream without modifying package-global RNG.
		rng := rand.New(rand.NewSource(2710))
		values := make([]float64, sampleCount)
		for i := 2; i < sampleCount; i++ {
			values[i] = 0.6*values[i-1] - 0.2*values[i-2] + rng.NormFloat64()
		}

		for _, maxLag := range []int{1, 20, 200} {
			for _, centered := range []bool{false, true} {
				b.Run(fmt.Sprintf("n%d_l%d_center%t", sampleCount, maxLag, centered), func(b *testing.B) {
					options := pacf.Options{MaxLag: maxLag, Center: centered}
					ctx := context.Background()
					b.ReportAllocs()
					b.ResetTimer()
					for iteration := 0; iteration < b.N; iteration++ {
						// Compute is the only measured operation; error branches fail the benchmark.
						if _, err := pacf.Compute(ctx, values, options); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		}
	}
}
