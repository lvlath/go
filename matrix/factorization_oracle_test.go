// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix_test independently validates pivoted solves against exact rational algebra.
// The oracle never reads production factor internals or calls another matrix kernel.
package matrix_test

import (
	"context"
	"errors"
	"math"
	"math/big"
	"math/rand"
	"testing"

	"github.com/lvlath/go/matrix"
)

const factorOracleSeed int64 = 2026110922

// TestFactorLURationalOracle evaluates deterministic diagonally dominant
// integer systems over multiple seeds and compares against independent exact
// fractions rather than another floating-point triangular solver.
func TestFactorLURationalOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(factorOracleSeed))
	for trial := 0; trial < 100; trial++ {
		n := 2 + rng.Intn(4)
		a := make([][]float64, n)
		b := make([][]float64, n)
		br := make([]*big.Rat, n)
		for i := 0; i < n; i++ {
			a[i] = make([]float64, n)
			b[i] = []float64{float64(rng.Intn(21) - 10)}
			br[i] = new(big.Rat).SetFloat64(b[i][0])
			for j := 0; j < n; j++ {
				a[i][j] = float64(rng.Intn(13) - 6)
			}
			a[i][i] += float64(12 * n)
		}
		want, ok := factorOracleRationalSolve(factorOracleRationalMatrix(a), br)
		if !ok {
			t.Fatalf("trial %d oracle singular", trial)
		}
		factor, err := matrix.FactorLU(context.Background(), factorFixture(t, a), matrix.FactorOptions{})
		if err != nil {
			t.Fatalf("trial %d factor: %v", trial, err)
		}
		got, err := factor.Solve(context.Background(), factorFixture(t, b))
		if err != nil {
			t.Fatalf("trial %d solve: %v", trial, err)
		}
		for i := 0; i < n; i++ {
			expected, _ := want[i].Float64()
			factorNear(t, factorCell(t, got.Solution, i, 0), expected, 1e-12, 1e-11)
		}
	}
}

// TestFactorQRRationalLeastSquaresOracle validates tall noisy least-squares
// problems using exact normal equations only as a mathematically independent
// source of expected coefficients, never as a production alternative.
func TestFactorQRRationalLeastSquaresOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(factorOracleSeed + 1))
	for trial := 0; trial < 80; trial++ {
		m, n := 7, 3
		a := make([][]float64, m)
		b := make([][]float64, m)
		bv := make([]float64, m)
		for i := 0; i < m; i++ {
			a[i] = make([]float64, n)
			for j := 0; j < n; j++ {
				a[i][j] = float64(rng.Intn(11) - 5)
			}
			bv[i] = float64(rng.Intn(13) - 6)
			b[i] = []float64{bv[i]}
		}
		want, nonsingular := factorOracleLeastSquares(a, bv)
		if !nonsingular {
			continue // A rank-deficient rational fixture is not full-rank QR input.
		}
		result, err := matrix.SolveLeastSquares(context.Background(), factorFixture(t, a), factorFixture(t, b), matrix.FactorOptions{})
		if err != nil {
			t.Fatalf("trial %d QR solve: %v", trial, err)
		}
		for j := 0; j < n; j++ {
			expected, _ := want[j].Float64()
			factorNear(t, factorCell(t, result.Solution, j, 0), expected, 1e-11, 1e-10)
		}
	}
}

// TestFactorQRExactPivotNormTies fixes the lowest-original-column rule when
// candidate trailing norms are mathematically equal and exactly representable.
func TestFactorQRExactPivotNormTies(t *testing.T) {
	a := factorFixture(t, [][]float64{{1, 0}, {0, -1}, {0, 0}})
	factor, err := matrix.FactorQR(context.Background(), a, matrix.FactorOptions{})
	if err != nil {
		t.Fatal(err)
	}
	perm := factor.Permutation()
	if len(perm) != 2 || perm[0] != 0 || perm[1] != 1 {
		t.Fatalf("equal-norm original column tie = %v, want [0 1]", perm)
	}
}

// TestFactorScaledHugeIdentity verifies factorization and solve across extreme
// finite exponents without overflowing intermediate elimination products.
func TestFactorScaledHugeIdentity(t *testing.T) {
	for _, magnitude := range []float64{1e308, 1e-300} {
		a := factorFixture(t, [][]float64{{magnitude, 0}, {0, magnitude}})
		b := factorFixture(t, [][]float64{{magnitude}, {magnitude}})
		for name, factorize := range map[string]func() (func(context.Context, matrix.Matrix) (*matrix.SolveResult, error), error){
			"LU": func() (func(context.Context, matrix.Matrix) (*matrix.SolveResult, error), error) {
				f, err := matrix.FactorLU(context.Background(), a, matrix.FactorOptions{})
				if err != nil {
					return nil, err
				}
				return f.Solve, nil
			},
			"QR": func() (func(context.Context, matrix.Matrix) (*matrix.SolveResult, error), error) {
				f, err := matrix.FactorQR(context.Background(), a, matrix.FactorOptions{})
				if err != nil {
					return nil, err
				}
				return f.Solve, nil
			},
			"Cholesky": func() (func(context.Context, matrix.Matrix) (*matrix.SolveResult, error), error) {
				f, err := matrix.FactorCholesky(context.Background(), a, 0, 0)
				if err != nil {
					return nil, err
				}
				return f.Solve, nil
			},
		} {
			solve, err := factorize()
			if err != nil {
				t.Fatalf("%s at scale %g: %v", name, magnitude, err)
			}
			result, err := solve(context.Background(), b)
			if err != nil {
				t.Fatalf("%s solve at scale %g: %v", name, magnitude, err)
			}
			for i := 0; i < 2; i++ {
				factorNear(t, factorCell(t, result.Solution, i, 0), 1, 0, 1e-13)
			}
			factorNear(t, result.ConditionEstimate, 1, 0, 1e-13)
		}
	}
}

// TestFactorQRUnrepresentableResidual rejects a genuine L2 output overflow,
// even when individual RHS observations and all solved coefficients are finite.
func TestFactorQRUnrepresentableResidual(t *testing.T) {
	a := factorFixture(t, [][]float64{{1}, {0}, {0}})
	factor, err := matrix.FactorQR(context.Background(), a, matrix.FactorOptions{})
	if err != nil {
		t.Fatal(err)
	}
	b := factorFixture(t, [][]float64{{0}, {math.MaxFloat64}, {math.MaxFloat64}})
	result, err := factor.Solve(context.Background(), b)
	if result != nil {
		t.Fatalf("overflow published a result: %+v", result)
	}
	factorErrorIs(t, err, matrix.ErrOverflow)
}

// TestFactorFallbackPreservesErrors validates the same classification on
// generic Matrix.At paths and proves that an original accessor error survives.
func TestFactorFallbackPreservesErrors(t *testing.T) {
	injected := errors.New("injected backend failure")
	fallback := &factorOracleFailingMatrix{Matrix: factorFixture(t, [][]float64{{1, 0}, {0, 1}}), fault: injected}
	for _, factorize := range []func(context.Context, matrix.Matrix) error{
		func(ctx context.Context, m matrix.Matrix) error {
			_, err := matrix.FactorLU(ctx, m, matrix.FactorOptions{})
			return err
		},
		func(ctx context.Context, m matrix.Matrix) error {
			_, err := matrix.FactorQR(ctx, m, matrix.FactorOptions{})
			return err
		},
		func(ctx context.Context, m matrix.Matrix) error {
			_, err := matrix.FactorCholesky(ctx, m, 0, 0)
			return err
		},
	} {
		factorErrorIs(t, factorize(context.Background(), fallback), injected)
	}
}
