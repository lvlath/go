// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ols_test benchmarks full-rank QR training and immutable inference.
// All matrix construction and rank verification precedes benchmark timing.
package ols_test

import (
	"context"
	"fmt"
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
