// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix_test benchmarks the full reusable-factor solve operation.
// Inputs and factorization are excluded; exact residual diagnostics remain included.
package matrix_test

import (
	"context"
	"testing"

	"github.com/lvlath/go/matrix"
)

// BenchmarkMatrix_NUM01_Solve compares typical per-column RHS loads against
// heterogeneous magnitude workloads without allocating the design in the timer.
// The benchmark measures public solve, including detached results and exact norms.
func BenchmarkMatrix_NUM01_Solve(b *testing.B) {
	const dimension = 12
	ctx := context.Background()
	input, err := matrix.NewDense(dimension, dimension)
	if err != nil {
		b.Fatal(err)
	}
	for row := 0; row < dimension; row++ {
		for column := 0; column < dimension; column++ {
			value := 0.0
			if row == column {
				value = 2
			}
			if err := input.Set(row, column, value); err != nil {
				b.Fatal(err)
			}
		}
	}
	factor, err := matrix.FactorLU(ctx, input, matrix.FactorOptions{})
	if err != nil {
		b.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		rhs  []float64
	}{
		{name: "ordinary", rhs: []float64{1, 2, 3}},
		{name: "mixed_scales", rhs: []float64{1e180, 1e-180, 1}},
	} {
		b.Run(tc.name, func(b *testing.B) {
			response, createErr := matrix.NewDense(dimension, len(tc.rhs))
			if createErr != nil {
				b.Fatal(createErr)
			}
			for row := 0; row < dimension; row++ {
				for column, level := range tc.rhs {
					if err := response.Set(row, column, level); err != nil {
						b.Fatal(err)
					}
				}
			}
			b.ReportAllocs()
			b.ResetTimer()
			for run := 0; run < b.N; run++ {
				if _, err := factor.Solve(ctx, response); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
