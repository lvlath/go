// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ols_test benchmarks full-rank QR training and immutable inference.
// All matrix construction and rank verification precedes benchmark timing.
package ols_test

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/ols"
)

// benchmarkTrainingData generates a reproducible full-rank dense regression
// fixture with independent pseudorandom columns and modest prediction noise.
// Its fixed seed prevents benchmark inputs changing between toolchain runs.
func benchmarkTrainingData(b *testing.B, rows, cols int) (*matrix.Dense, []float64) {
	b.Helper()
	random := rand.New(rand.NewSource(2026))
	x, err := matrix.NewDense(rows, cols)
	if err != nil {
		b.Fatal(err)
	}
	y := make([]float64, rows)
	for row := 0; row < rows; row++ {
		y[row] = 0.75
		for col := 0; col < cols; col++ {
			value := random.Float64()*2 - 1
			if err = x.Set(row, col, value); err != nil {
				b.Fatal(err)
			}
			y[row] += value / float64(col+1)
		}
		y[row] += 0.005 * (random.Float64()*2 - 1)
	}
	return x, y
}

// BenchmarkFitQR measures full design, pivoted factorization, coefficient
// back-conversion, and training residual verification as one canonical Fit.
func BenchmarkFitQR(b *testing.B) {
	for _, shape := range []struct{ rows, cols int }{
		{100, 2}, {1000, 16}, {10000, 64},
	} {
		b.Run(fmt.Sprintf("n%d_p%d", shape.rows, shape.cols), func(b *testing.B) {
			x, y := benchmarkTrainingData(b, shape.rows, shape.cols)
			ctx := context.Background()
			opts := ols.Options{Standardize: true}
			if _, err := ols.Fit(ctx, x, y, opts); err != nil {
				b.Fatalf("invalid full-rank fixture: %v", err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for iteration := 0; iteration < b.N; iteration++ {
				if _, err := ols.Fit(ctx, x, y, opts); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkPredict measures frozen coefficients and row-major FMA inference.
// Its training setup is excluded; no shared mutable buffer is reused by calls.
func BenchmarkPredict(b *testing.B) {
	for _, shape := range []struct{ rows, cols int }{
		{100, 2}, {1000, 16}, {10000, 64},
	} {
		b.Run(fmt.Sprintf("n%d_p%d", shape.rows, shape.cols), func(b *testing.B) {
			x, y := benchmarkTrainingData(b, shape.rows, shape.cols)
			ctx := context.Background()
			fitted, err := ols.FitStandardized(ctx, x, y)
			if err != nil {
				b.Fatalf("invalid full-rank fixture: %v", err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for iteration := 0; iteration < b.N; iteration++ {
				if _, err = fitted.Model.Predict(ctx, x); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkPredict_AffineArithmetic compares fixed-size repeated prediction
// workloads; its measurements separate ordinary rows from rare exact fallbacks.
// The caller's floating-point operation distribution, not just n*p, determines
// how much extra precision is required for the published physical responses.
func BenchmarkPredict_AffineArithmetic(b *testing.B) {
	const (
		features = 16
		rows     = 64
	)
	ctx := context.Background()

	// Identity training has full numerical rank and exactly representable unit
	// slopes. Fit cost is excluded from both prediction benchmark subcases.
	training, err := matrix.NewDense(features, features)
	if err != nil {
		b.Fatal(err)
	}
	for j := 0; j < features; j++ {
		if err := training.Set(j, j, 1); err != nil {
			b.Fatal(err)
		}
	}
	responses := make([]float64, features)
	for j := range responses {
		responses[j] = 1
	}
	fit, err := ols.FitThroughOrigin(ctx, training, responses)
	if err != nil {
		b.Fatal(err)
	}

	for _, benchmark := range []struct {
		name       string
		firstThree [3]float64
	}{
		{"ordinary", [3]float64{1.5, 0.75, 0.25}},
		{"lost_small_term", [3]float64{1e16, 1, -1e16}},
		{"overflow_cancellation", [3]float64{math.MaxFloat64, 0, -math.MaxFloat64}},
	} {
		b.Run(benchmark.name, func(b *testing.B) {
			holdout, err := matrix.NewDense(rows, features)
			if err != nil {
				b.Fatal(err)
			}
			for row := 0; row < rows; row++ {
				for col, value := range benchmark.firstThree {
					if err := holdout.Set(row, col, value); err != nil {
						b.Fatal(err)
					}
				}
			}

			b.ReportAllocs()
			b.ResetTimer()
			for trial := 0; trial < b.N; trial++ {
				predicted, err := fit.Model.Predict(ctx, holdout)
				if err != nil || len(predicted) != rows {
					b.Fatalf("prediction failed: length=%d, err=%v", len(predicted), err)
				}
			}
		})
	}
}
