// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package arima_test benchmarks whole-model inference across explicit polynomial regimes.
// All fixture construction and Model validation occur before timing begins.
package arima_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/lvlath/go/arima"
)

// BenchmarkForecast measures historical-innovation reconstruction plus H=128 inference.
// N grows from one thousand to one hundred thousand while p=q in {0,1,8}
// and d in {0,2}; the high-order case still executes all declared lag slots.
// The setup constructs an immutable model and does not enter the timed loop.
func BenchmarkForecast(b *testing.B) {
	for _, n := range []int{1_000, 100_000} {
		for _, degree := range []int{0, 1, 8} {
			for _, differenceOrder := range []int{0, 2} {
				b.Run(fmt.Sprintf("N%d_PQ%d_D%d", n, degree, differenceOrder), func(b *testing.B) {
					levels := make([]float64, n)
					for index := range levels {
						levels[index] = 100 + float64(index%17) + float64(index)*0.01
					}
					ar := make([]float64, degree)
					ma := make([]float64, degree)
					if degree != 0 {
						ar[0] = 0.1
						ma[0] = 0.2
					}
					model, err := arima.New(arima.Parameters{AR: ar, MA: ma, InnovationVariance: 1,
						Order: arima.Order{AR: degree, Difference: differenceOrder, MA: degree}})
					if err != nil {
						b.Fatal(err)
					}
					ctx := context.Background()
					b.ReportAllocs()
					b.ResetTimer()
					for iteration := 0; iteration < b.N; iteration++ {
						if _, err := model.Forecast(ctx, levels, 128); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		}
	}
}

// BenchmarkFitBounded profiles analytic and limited-iteration optimization regimes.
// High-order cases deliberately use two accepted-step attempts to bound workload;
// nonconvergence is an expected measured outcome, not an ignored test failure.
// This benchmark measures complete conditional fitting, including all line searches.
func BenchmarkFitBounded(b *testing.B) {
	cases := []struct {
		name  string
		n     int
		order arima.Order
	}{
		{"white_noise_N1000", 1_000, arima.Order{}},
		{"integrated_noise_N100000", 100_000, arima.Order{Difference: 2}},
		{"ARMA1_N1000", 1_000, arima.Order{AR: 1, MA: 1}},
		{"ARMA1_D2_N100000", 100_000, arima.Order{AR: 1, MA: 1, Difference: 2}},
		{"ARMA8_N1000", 1_000, arima.Order{AR: 8, MA: 8}},
		{"ARMA8_D2_N1000", 1_000, arima.Order{AR: 8, MA: 8, Difference: 2}},
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			levels := make([]float64, tc.n)
			for index := range levels {
				t := float64(index)
				levels[index] = 100 + 0.03*t + 1.2*math.Sin(t*0.41) + 0.6*math.Cos(t*0.87)
				if tc.order.Difference == 2 {
					levels[index] += 0.0003 * t * t
				}
			}
			options := arima.Options{Order: tc.order, IncludeDrift: true, MaxIterations: 2, Tolerance: 1e-8}
			ctx := context.Background()
			b.ReportAllocs()
			b.ResetTimer()
			for iteration := 0; iteration < b.N; iteration++ {
				result, err := arima.Fit(ctx, levels, options)
				if err != nil && !errors.Is(err, arima.ErrNotConverged) {
					b.Fatal(err)
				}
				if result == nil || result.Model == nil {
					b.Fatal("fit failed to publish a valid candidate")
				}
			}
		})
	}
}
