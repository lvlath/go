// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gjrgarch_test benchmarks causal filtering, one-step updates,
// constrained-fit budget use, and expected-variance forecast allocation.
// Input generation and model construction occur before each timed region.
package gjrgarch_test

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/lvlath/go/gjrgarch"
)

// benchmarkReturns creates a reproducible two-regime financial return path.
// Large negative shock clusters exercise the asymmetric recurrence without RNG.
func benchmarkReturns(length int) []float64 {
	returns := make([]float64, length)
	for i := range returns {
		returnValue := float64((i*37)%101-50) * 0.00045
		if withinCluster := i%127 >= 96 && i%127 < 104; withinCluster {
			returnValue *= 2.1
		}
		if i%257 == 121 {
			returnValue -= 0.041
		}
		returns[i] = returnValue
	}
	return returns
}

// benchmarkModel constructs the same stationary asymmetric candidate for all
// regimes so differences measure workload shape rather than parameter changes.
func benchmarkModel(b *testing.B) *gjrgarch.Model {
	b.Helper()
	model, err := gjrgarch.New(gjrgarch.Parameters{
		Omega: 0.000021, Alpha: 0.08, Gamma: 0.14, Beta: 0.73,
	})
	if err != nil {
		b.Fatal(err)
	}
	return model
}

// BenchmarkFilter measures complete validation, causal traversal, and owned output
// allocation under T=1000 and T=10000 return windows with downside clusters.
func BenchmarkFilter(b *testing.B) {
	model := benchmarkModel(b)
	for _, size := range [...]int{1000, 10000} {
		b.Run(fmt.Sprintf("T=%d/stress_clusters", size), func(b *testing.B) {
			returns := benchmarkReturns(size)
			ctx := context.Background()
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := model.Filter(ctx, returns, 0.0003); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkStep measures the production hot path without data generation or
// forecasting allocation inside the timer. All returned states are consumed.
func BenchmarkStep(b *testing.B) {
	model := benchmarkModel(b)
	state, err := model.InitialState(-0.012, 0.0003)
	if err != nil {
		b.Fatal(err)
	}
	observed := -0.006
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		state, err = model.Step(state, observed)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkForecastVariance isolates ten future conditional variance predictions
// from one validated posterior; it excludes state construction from measurement.
func BenchmarkForecastVariance(b *testing.B) {
	model := benchmarkModel(b)
	state, err := model.InitialState(-0.028, 0.00045)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := model.ForecastVariance(state, 10); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkFit measures initial QMLE plus ONE accepted optimizer-budget step.
// It does not falsely report a short iteration budget as a converged full fit.
func BenchmarkFit(b *testing.B) {
	for _, size := range [...]int{1000, 10000} {
		b.Run(fmt.Sprintf("T=%d/one_iteration_budget", size), func(b *testing.B) {
			returns := benchmarkReturns(size)
			ctx := context.Background()
			options := gjrgarch.Options{
				Initial: gjrgarch.Parameters{
					Omega: 0.000024, Alpha: 0.11, Gamma: 0.12, Beta: 0.69,
				},
				MaxIterations: 1,
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				result, err := gjrgarch.Fit(ctx, returns, options)
				if result == nil || result.Model == nil ||
					(err != nil && !errors.Is(err, gjrgarch.ErrNotConverged)) {
					b.Fatal("one-budget fit did not return a validated candidate:", err)
				}
			}
		})
	}
}
