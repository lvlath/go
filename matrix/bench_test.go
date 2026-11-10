// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix_test provides benchmarks for core matrix package operations,
// using deterministic random fill for Dense matrices.
package matrix_test

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/lvlath/go/matrix"
)

// benchSizes are the matrix sizes to benchmark.
var benchSizes = []int{128, 256, 512}

// sinks to defeat dead-code elimination
var (
	sinkM matrix.Matrix
	sinkV []float64
	sinkB bool
	sinkF float64
)

func BenchmarkAdd(b *testing.B) {
	b.ReportAllocs()
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			A := mustDense(b, n, n)
			B := mustDense(b, n, n)
			fillDenseRand(b, A, 1337)
			fillDenseRand(b, B, 4242)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				m, err := matrix.Sum(A, B)
				if err != nil {
					b.Fatal(err)
				}
				sinkM = m
			}
		})
	}
}

func BenchmarkSub(b *testing.B) {
	b.ReportAllocs()
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			A := mustDense(b, n, n)
			B := mustDense(b, n, n)
			fillDenseRand(b, A, 11)
			fillDenseRand(b, B, 22)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				m, err := matrix.Diff(A, B)
				if err != nil {
					b.Fatal(err)
				}
				sinkM = m
			}
		})
	}
}

func BenchmarkHadamard(b *testing.B) {
	b.ReportAllocs()
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			A := mustDense(b, n, n)
			B := mustDense(b, n, n)
			fillDenseRand(b, A, 1)
			fillDenseRand(b, B, 2)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				m, err := matrix.HadamardProd(A, B)
				if err != nil {
					b.Fatal(err)
				}
				sinkM = m
			}
		})
	}
}

func BenchmarkTranspose(b *testing.B) {
	b.ReportAllocs()
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			A := mustDense(b, n, n+8) // rectangular
			fillDenseRand(b, A, 7)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				m, err := matrix.T(A)
				if err != nil {
					b.Fatal(err)
				}
				sinkM = m
			}
		})
	}
}

func BenchmarkScale(b *testing.B) {
	b.ReportAllocs()
	const alpha = 1.75
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			A := mustDense(b, n, n)
			fillDenseRand(b, A, 9)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				m, err := matrix.ScaleBy(A, alpha)
				if err != nil {
					b.Fatal(err)
				}
				sinkM = m
			}
		})
	}
}

func BenchmarkMatVec(b *testing.B) {
	b.ReportAllocs()
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			A := mustDense(b, n, n)
			fillDenseRand(b, A, 99)
			x := onesVec(n)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				y, err := matrix.MatVecMul(A, x)
				if err != nil {
					b.Fatal(err)
				}
				sinkV = y
			}
		})
	}
}

func BenchmarkMul(b *testing.B) {
	b.ReportAllocs()
	for _, n := range []int{64, 96, 128} { // limits it so that CI doesn't burn
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			A := mustDense(b, n, n)
			B := mustDense(b, n, n)
			fillDenseRand(b, A, 101)
			fillDenseRand(b, B, 202)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				C, err := matrix.Product(A, B)
				if err != nil {
					b.Fatal(err)
				}
				sinkM = C
			}
		})
	}
}

func BenchmarkRowSums(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			A := mustDense(b, n, n)
			fillDenseRand(b, A, 12)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := matrix.RowSums(A); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkColSums(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			A := mustDense(b, n, n)
			fillDenseRand(b, A, 13)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := matrix.ColSums(A); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkSymmetrize(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			A := mustDense(b, n, n)
			fillDenseRand(b, A, 14)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := matrix.Symmetrize(A); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkLU(b *testing.B) {
	b.ReportAllocs()
	for _, n := range []int{32, 64, 96} {
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			A := mustDense(b, n, n)
			fillDenseRand(b, A, 303)
			// shift the diagonal to eliminate zero pivots
			for i := 0; i < n; i++ {
				aii, _ := A.At(i, i)
				_ = A.Set(i, i, aii+float64(n)+1)
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				L, U, err := matrix.LUDecompose(A)
				if err != nil {
					b.Fatal(err)
				}
				// keep alive
				sinkM, _ = L, U
			}
		})
	}
}

func BenchmarkQR(b *testing.B) {
	b.ReportAllocs()
	for _, n := range []int{32, 64, 96} {
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			A := mustDense(b, n, n)
			fillDenseRand(b, A, 404)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				Q, R, err := matrix.QRDecompose(A)
				if err != nil {
					b.Fatal(err)
				}
				sinkM, _ = Q, R
			}
		})
	}
}

func BenchmarkInverse(b *testing.B) {
	b.ReportAllocs()
	for _, n := range []int{16, 32, 64} {
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			M := mustDense(b, n, n)
			fillDenseRand(b, M, 505)
			Mt, _ := matrix.T(M)
			PD, _ := matrix.Product(Mt, M)
			for i := 0; i < n; i++ {
				aii, _ := PD.At(i, i)
				_ = PD.Set(i, i, aii+float64(n)+2)
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				Inv, err := matrix.InverseOf(PD)
				if err != nil {
					b.Fatal(err)
				}
				sinkM = Inv
			}
		})
	}
}

func BenchmarkEigenSym(b *testing.B) {
	b.ReportAllocs()
	for _, n := range []int{16, 32, 64} {
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			// SPD via A=MᵀM
			M := mustDense(b, n, n)
			fillDenseRand(b, M, 606)
			Mt, _ := matrix.T(M)
			A, _ := matrix.Product(Mt, M)
			const tol = 1e-9
			const maxIter = 200
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				vals, Q, err := matrix.EigenSym(A, tol, maxIter)
				if err != nil {
					b.Fatal(err)
				}
				if len(vals) == 0 || Q == nil {
					b.Fatal("empty eigen result")
				}
				sinkF = vals[0]
				sinkM = Q
			}
		})
	}
}

func BenchmarkAPSP_FloydWarshall_Dense(b *testing.B) {
	b.ReportAllocs()
	for _, n := range []int{128, 512} {
		// buildDistanceMatrix constructs a distance-policy matrix:
		// diagonal=0, off-diagonal=+Inf, with a few sparse edges.
		buildDistanceMatrix := func() *matrix.Dense {
			D, err := matrix.NewZeros(n, n, matrix.WithAllowInfDistances())
			if err != nil {
				b.Fatalf("NewZeros(%d,%d): %v", n, n, err)
			}
			inf := math.Inf(1)
			for i := 0; i < n; i++ {
				for j := 0; j < n; j++ {
					if i == j {
						if err = D.Set(i, j, 0); err != nil {
							b.Fatalf("Set(diag): %v", err)
						}
						continue
					}
					if err = D.Set(i, j, inf); err != nil {
						b.Fatalf("Set(+Inf): %v", err)
					}
				}
			}
			rng := rand.New(rand.NewSource(777))
			for e := 0; e < n*3; e++ {
				u := rng.Intn(n)
				v := rng.Intn(n)
				if u == v {
					continue
				}
				w := 1 + rng.Float64()*4
				if err = D.Set(u, v, w); err != nil {
					b.Fatalf("Set(edge): %v", err)
				}
			}

			return D
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			b.StopTimer()
			D := buildDistanceMatrix()
			b.StartTimer()
			if err := matrix.APSPInPlace(D); err != nil {
				b.Fatal(err)
			}
			sinkM = D
		}
	}
}

func BenchmarkStatsAndSanitize(b *testing.B) {
	b.ReportAllocs()
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("CenterColumns_n=%d", n), func(b *testing.B) {
			X := mustDense(b, n, n)
			fillDenseRand(b, X, 808)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				Xc, _, err := matrix.CenterColumns(X)
				if err != nil {
					b.Fatal(err)
				}
				sinkM = Xc
			}
		})
		b.Run(fmt.Sprintf("NormalizeRowsL1_n=%d", n), func(b *testing.B) {
			X := mustDense(b, n, n)
			fillDenseRand(b, X, 909)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				Y, _, err := matrix.NormalizeRowsL1(X)
				if err != nil {
					b.Fatal(err)
				}
				sinkM = Y
			}
		})
		b.Run(fmt.Sprintf("NormalizeRowsL2_n=%d", n), func(b *testing.B) {
			X := mustDense(b, n, n)
			fillDenseRand(b, X, 1001)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				Y, _, err := matrix.NormalizeRowsL2(X)
				if err != nil {
					b.Fatal(err)
				}
				sinkM = Y
			}
		})
		b.Run(fmt.Sprintf("Clip_n=%d", n), func(b *testing.B) {
			X := mustDense(b, n, n)
			fillDenseRand(b, X, 1111)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				Y, err := matrix.Clip(X, -0.3, 0.7)
				if err != nil {
					b.Fatal(err)
				}
				sinkM = Y
			}
		})
		b.Run(fmt.Sprintf("ReplaceInfNaN_n=%d", n), func(b *testing.B) {
			X := mustDense(b, n, n)
			fillDenseRand(b, X, 1212)
			// let's add some NaN/Inf
			_ = X.Set(0, 0, math.NaN())
			_ = X.Set(0, 1, math.Inf(1))
			_ = X.Set(1, 0, math.Inf(-1))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				Y, err := matrix.ReplaceInfNaN(X, 0)
				if err != nil {
					b.Fatal(err)
				}
				sinkM = Y
			}
		})
		b.Run(fmt.Sprintf("AllClose_n=%d", n), func(b *testing.B) {
			X := mustDense(b, n, n)
			Y := mustDense(b, n, n)
			fillDenseRand(b, X, 1313)
			fillDenseRand(b, Y, 1313) // same values ⇒ true
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				ok, err := matrix.AllClose(X, Y, 1e-9, 1e-12)
				if err != nil {
					b.Fatal(err)
				}
				sinkB = ok
			}
		})
	}
}

// --- factorization bench tests ---

// BenchmarkFactorLU_Order100 measures reusable row-pivoted decomposition
// including its O(n^3) measured condition estimate, excluding matrix setup.
func BenchmarkFactorLU_Order100(b *testing.B) {
	input := factorBenchmarkDense(b, 100, 100)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for iteration := 0; iteration < b.N; iteration++ {
		if _, err := matrix.FactorLU(ctx, input, matrix.FactorOptions{}); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkFactorQR_Tall1000x100 measures pivoted Householder QR and measured
// kappa_1(R) on a tall regression regime, with direct versus At fallback paths.
func BenchmarkFactorQR_Tall1000x100(b *testing.B) {
	for _, backend := range []string{"Dense", "Fallback"} {
		b.Run(backend, func(b *testing.B) {
			input := factorBenchmarkDense(b, 1000, 100)
			var source matrix.Matrix = input
			if backend == "Fallback" {
				source = &statsTestFallback{Matrix: input}
			}
			ctx := context.Background()
			b.ReportAllocs()
			b.ResetTimer()
			for iteration := 0; iteration < b.N; iteration++ {
				if _, err := matrix.FactorQR(ctx, source, matrix.FactorOptions{}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkFactorCholesky_Order100 measures SPD triangular construction and
// its condition estimate on a positive definite diagonal-plus-rank-one system.
func BenchmarkFactorCholesky_Order100(b *testing.B) {
	const dimension = 100
	input, err := matrix.NewDense(dimension, dimension)
	if err != nil {
		b.Fatal(err)
	}
	for i := 0; i < dimension; i++ {
		for j := 0; j < dimension; j++ {
			value := float64((i%7)+1) * float64((j%7)+1) / 100
			if i == j {
				value += dimension
			}
			if err = input.Set(i, j, value); err != nil {
				b.Fatal(err)
			}
		}
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for iteration := 0; iteration < b.N; iteration++ {
		if _, err = matrix.FactorCholesky(ctx, input, 0, 0); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkFactorsRepeatedSolve measures independent Solve workspace and exact
// residual evaluation after factoring once, for several RHS counts.
func BenchmarkFactorsRepeatedSolve(b *testing.B) {
	for _, columns := range []int{1, 8} {
		b.Run(fmt.Sprintf("rhs%d", columns), func(b *testing.B) {
			input := factorBenchmarkDense(b, 100, 100)
			rhs := factorBenchmarkDense(b, 100, columns)
			ctx := context.Background()
			factor, err := matrix.FactorLU(ctx, input, matrix.FactorOptions{})
			if err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for iteration := 0; iteration < b.N; iteration++ {
				if _, err = factor.Solve(ctx, rhs); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// --- statistics stability bench tests ---

// BenchmarkStatisticsCovariance tracks the expected quadratic feature cost
// and any additional accessor overhead. Every iteration measures exactly
// one successful covariance computation; fixture construction is excluded.
func BenchmarkStatisticsCovariance(b *testing.B) {
	for _, shape := range []struct{ rows, cols int }{
		{100, 1}, {100, 16}, {100, 128},
		{10000, 1}, {10000, 16}, {10000, 128},
	} {
		baseName := fmt.Sprintf("n%d_d%d", shape.rows, shape.cols)
		b.Run(baseName+"/Dense", func(b *testing.B) {
			input := statsBenchFixture(b, shape.rows, shape.cols)
			b.ReportAllocs()
			b.ResetTimer()
			for iteration := 0; iteration < b.N; iteration++ {
				if _, _, err := matrix.Covariance(input); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run(baseName+"/Fallback", func(b *testing.B) {
			input := &statsTestFallback{Matrix: statsBenchFixture(b, shape.rows, shape.cols)}
			b.ReportAllocs()
			b.ResetTimer()
			for iteration := 0; iteration < b.N; iteration++ {
				if _, _, err := matrix.Covariance(input); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkStatisticsCancellationRescue isolates the predictable overhead
// of exact rational reconstruction when large cross-products almost cancel.
// It must not be compared to a Dense benchmark with different observations.
func BenchmarkStatisticsCancellationRescue(b *testing.B) {
	input := statsBenchFixture(b, 100, 2)
	// Replace the fixture with the specific cancellation regime, before the
	// timer, so that the same exact path is measured on each iteration.
	for i := 0; i < 100; i++ {
		first := 1e150
		if i%2 == 1 {
			first = -first
		}
		second := 1.0
		if i == 99 {
			first = 1
			second = -99
		}
		if err := input.Set(i, 0, first); err != nil {
			b.Fatal(err)
		}
		if err := input.Set(i, 1, second); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for iteration := 0; iteration < b.N; iteration++ {
		if _, _, err := matrix.Covariance(input); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkStatisticsConstantColumns protects the no-rescue fast path for
// nonzero but constant observations. A zero cross-moment caused by constant
// columns requires no big-integer reconstruction of the same known zero.
func BenchmarkStatisticsConstantColumns(b *testing.B) {
	input, err := matrix.NewDense(100, 16)
	if err != nil {
		b.Fatal(err)
	}
	for i := 0; i < 100; i++ {
		for j := 0; j < 16; j++ {
			if err = input.Set(i, j, float64(j+1)*1e150); err != nil {
				b.Fatal(err)
			}
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for iteration := 0; iteration < b.N; iteration++ {
		if _, _, err = matrix.Covariance(input); err != nil {
			b.Fatal(err)
		}
	}
}
