// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package acf_test benchmarks direct scalar ACF across explicit sample/lag regimes.
// The full n=100000, maxLag=n-1 experiment is opt-in because it scans ~5e9 pairs.
package acf_test

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/lvlath/go/acf"
)

var benchResult *acf.Result

// BenchmarkCompute_Direct measures scalar direct computations with all input
// preparation excluded from timing and the public result retained as a sink.
//
// Regimes:
//   - lag 0: centering, validation, and zero-lag energy dominate (linear).
//   - lag 16/128: bounded-lag scans test near-linear dependence on sample count.
//   - maxLag=n-1: triangular full autocorrelation scan tests quadratic work.
//
// The largest full-window run requires ACF_BENCH_FULL=1 and -benchtime=1x.
// All other cases use the normal Go benchmarking harness and report allocations.
func BenchmarkCompute_Direct(b *testing.B) {
	const runFullBenchmarkEnvironment = "ACF_BENCH_FULL"
	const fullBenchmarkOptIn = "1"

	for _, sampleCount := range []int{1000, 100000} {
		// Deterministic, nonconstant signals combine a short carrier and a slower
		// oscillation. Building them is not part of the measured algorithm.
		samples := make([]float64, sampleCount)
		for i := range samples {
			samples[i] = math.Sin(float64(i)*0.137) + 0.1*math.Cos(float64(i)*0.011)
		}
		for _, maxLag := range []int{0, 16, 128, sampleCount - 1} {
			for _, estimator := range []acf.Denominator{acf.CommonDenominator, acf.LagAdjusted} {
				name := fmt.Sprintf("n=%d/lag=%d/den=%d", sampleCount, maxLag, estimator)
				b.Run(name, func(b *testing.B) {
					if sampleCount == 100000 && maxLag == sampleCount-1 && os.Getenv(runFullBenchmarkEnvironment) != fullBenchmarkOptIn {
						b.Skip("opt in with ACF_BENCH_FULL=1 -benchtime=1x; full scan is quadratic")
					}

					// Reset after fixture and policy selection; only the public
					// computation (including its required allocations) is measured.
					options := acf.Options{Denominator: estimator}
					b.ReportAllocs()
					b.ResetTimer()
					for iteration := 0; iteration < b.N; iteration++ {
						result, err := acf.Compute(samples, maxLag, options)
						if err != nil {
							b.Fatal(err)
						}
						benchResult = result
					}
				})
			}
		}
	}
}
