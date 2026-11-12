// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pca_test benchmarks fitted covariance PCA under reproducible
// observation-feature regimes. Training fixtures and feasibility checks are
// excluded from measurements; this suite never times a fast error return.
package pca_test

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/pca"
)

// BenchmarkPCA_CovarianceEigen measures full training with fixed retained width.
// It exposes the effect of feature dimension on covariance accumulation, Jacobi
// max-pivot scanning, and verification without conflating those costs with
// input synthesis or matrix allocation at benchmark setup.
//
// Implementation:
//   - Stage 1: construct deterministic, nondegenerate n-by-d training matrices.
//   - Stage 2: verify Fit succeeds for each shape before timing begins.
//   - Stage 3: measure repeated canonical Fit calls and report allocations.
//
// Behavior highlights:
//   - n=100/1000 and d=10/50/100 isolate sample and feature scaling.
//   - Components=1 keeps the published output width fixed across regimes.
//
// Inputs:
//   - b: benchmark harness controlling iterations; all fixtures are synthetic.
//
// Returns:
//   - Benchmark samples of ns/op, B/op, and allocs/op, not a fitted artifact.
//
// Errors:
//   - Setup or training failures abort the benchmark; invalid inputs are not timed.
//
// Determinism:
//   - Every sample derives from fixed row/column sine and cosine phases.
//
// Complexity:
//   - Each measured fit costs O(n*d*d+rotations*d*d+d*d*d+n*d)
//     time and O(n*d+d*d) workspace; the loop itself allocates no fixture.
//
// Notes:
//   - Compare results only on the same full dependency tree, CPU and toolchain.
//   - This is a covariance solver benchmark, not a direct-SVD comparison.
//
// AI-Hints:
//   - Do not remove the untimed feasibility fit: early failures fake speedups.
//   - Do not move input synthesis inside b.N or use benchmark-only solver code.
func BenchmarkPCA_CovarianceEigen(b *testing.B) {
	for _, shape := range []struct{ n, d int }{{100, 10}, {100, 50}, {100, 100}, {1000, 10}, {1000, 50}, {1000, 100}} {
		b.Run(fmt.Sprintf("n%d_d%d", shape.n, shape.d), func(b *testing.B) {
			x, err := matrix.NewDense(shape.n, shape.d)
			if err != nil {
				b.Fatal(err)
			}
			// Deterministic full-rank design with smooth and independent channel frequencies.
			values := make([]float64, shape.n*shape.d)
			for i := 0; i < shape.n; i++ {
				for j := 0; j < shape.d; j++ {
					values[i*shape.d+j] = math.Sin(float64((i+1)*(j+1))*.0037) + math.Cos(float64((i+1)*(j+2))*.011)
				}
			}
			if err = x.Fill(values); err != nil {
				b.Fatal(err)
			}
			opts := pca.Options{Components: 1}
			if _, err = pca.Fit(context.Background(), x, opts); err != nil {
				b.Fatalf("setup feasibility: %v", err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err = pca.Fit(context.Background(), x, opts); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
