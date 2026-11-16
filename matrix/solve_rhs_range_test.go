// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix_test verifies stable reusable-factor solves across extreme RHS ranges.
// Each oracle is derived from the original binary64 values, not stored factor state.
package matrix_test

import (
	"context"
	"errors"
	"math"
	"math/big"
	"sync"
	"testing"

	"github.com/lvlath/go/matrix"
)

// rangeDense constructs an owning numeric fixture with explicit error propagation.
// It does not hide shape or allocation failures from the test caller.
func rangeDense(t *testing.T, input [][]float64) *matrix.Dense {
	t.Helper()
	rows := len(input)
	if rows == 0 || len(input[0]) == 0 {
		t.Fatal("fixture must contain observations and columns")
	}
	columns := len(input[0])
	result, err := matrix.NewDense(rows, columns)
	if err != nil {
		t.Fatal(err)
	}
	for row, values := range input {
		if len(values) != columns {
			t.Fatalf("fixture row %d has width %d, expected %d", row, len(values), columns)
		}
		for column, value := range values {
			if err := result.Set(row, column, value); err != nil {
				t.Fatal(err)
			}
		}
	}
	return result
}

// rangeSolver is the public capability shared by the three reusable factors.
// It deliberately contains no access to private scaling or permutation fields.
type rangeSolver interface {
	Solve(context.Context, matrix.Matrix) (*matrix.SolveResult, error)
}

// rangeFactors constructs all three factors from one symmetric positive definite
// fixture, allowing the same independent numerical oracle to validate each solve.
func rangeFactors(t *testing.T, a *matrix.Dense, opts matrix.FactorOptions) map[string]rangeSolver {
	t.Helper()
	ctx := context.Background()
	lu, err := matrix.FactorLU(ctx, a, opts)
	if err != nil {
		t.Fatalf("FactorLU: %v", err)
	}
	qr, err := matrix.FactorQR(ctx, a, opts)
	if err != nil {
		t.Fatalf("FactorQR: %v", err)
	}
	chol, err := matrix.FactorCholesky(ctx, a, 0, 0)
	if err != nil {
		t.Fatalf("FactorCholesky: %v", err)
	}
	return map[string]rangeSolver{"LU": lu, "QR": qr, "Cholesky": chol}
}

// rangeRat converts a finite binary64 input to an exact rational oracle value.
func rangeRat(t *testing.T, value float64) *big.Rat {
	t.Helper()
	result := new(big.Rat).SetFloat64(value)
	if result == nil {
		t.Fatalf("nonfinite oracle input: %g", value)
	}
	return result
}

// rangeRatSolveTwoByTwo solves a fixed 2-by-2 system through independent exact
// Cramer's-rule arithmetic. Production uses pivoted LU/QR/Cholesky instead.
func rangeRatSolveTwoByTwo(t *testing.T, a [4]float64, b [2]float64) [2]float64 {
	t.Helper()
	a00, a01 := rangeRat(t, a[0]), rangeRat(t, a[1])
	a10, a11 := rangeRat(t, a[2]), rangeRat(t, a[3])
	b0, b1 := rangeRat(t, b[0]), rangeRat(t, b[1])
	determinant := new(big.Rat).Sub(new(big.Rat).Mul(a00, a11), new(big.Rat).Mul(a01, a10))
	if determinant.Sign() == 0 {
		t.Fatal("oracle determinant cannot be zero")
	}
	first := new(big.Rat).Quo(
		new(big.Rat).Sub(new(big.Rat).Mul(b0, a11), new(big.Rat).Mul(a01, b1)),
		determinant,
	)
	second := new(big.Rat).Quo(
		new(big.Rat).Sub(new(big.Rat).Mul(a00, b1), new(big.Rat).Mul(b0, a10)),
		determinant,
	)
	x0, _ := new(big.Float).SetPrec(256).SetRat(first).Float64()
	x1, _ := new(big.Float).SetPrec(256).SetRat(second).Float64()
	return [2]float64{x0, x1}
}

// rangeAssertVector enforces an independent expected solution and detached shape.
// Relative tolerance is allowed only for approximate nonzero factor solves.
func rangeAssertVector(t *testing.T, result *matrix.SolveResult, column int, want [2]float64, relative float64) {
	t.Helper()
	if result == nil || result.Solution == nil || result.Rank != 2 {
		t.Fatalf("incomplete solve result: %#v", result)
	}
	if result.Solution.Rows() != 2 || column >= result.Solution.Cols() {
		t.Fatalf("unexpected solution dimensions: %d x %d", result.Solution.Rows(), result.Solution.Cols())
	}
	for row, expected := range want {
		actual, err := result.Solution.At(row, column)
		if err != nil {
			t.Fatal(err)
		}
		if math.IsNaN(actual) || math.IsInf(actual, 0) {
			t.Fatalf("solution[%d,%d] is nonfinite: %g", row, column, actual)
		}
		if expected == 0 {
			if actual != 0 {
				t.Fatalf("solution[%d,%d]=%.17g; expected exact zero", row, column, actual)
			}
		} else if math.Abs(actual/expected-1) > relative {
			t.Fatalf("solution[%d,%d]=%.17g; oracle %.17g", row, column, actual, expected)
		}
	}
}

// TestMatrix_NUM01_RepresentableLargeSolution confirms that intermediate
// RHS division by the global A scale cannot reject a representable solution.
// The matrix has kappa_1=2, so the expected accuracy is not limited by
// extreme conditioning; all original coefficients and responses are finite.
func TestMatrix_NUM01_RepresentableLargeSolution(t *testing.T) {
	const tolerance = 3e-14
	high := math.Ldexp(1.75, 1023)
	aData := [4]float64{0.75, 0.25, 0.25, 0.75}
	a := rangeDense(t, [][]float64{{aData[0], aData[1]}, {aData[2], aData[3]}})
	b := rangeDense(t, [][]float64{{high}, {high}})
	oracle := rangeRatSolveTwoByTwo(t, aData, [2]float64{high, high})
	if oracle != [2]float64{high, high} {
		t.Fatalf("independent exact oracle unexpectedly differs: %v", oracle)
	}
	for name, factor := range rangeFactors(t, a, matrix.FactorOptions{}) {
		t.Run(name, func(t *testing.T) {
			result, err := factor.Solve(context.Background(), b)
			if err != nil {
				t.Fatal(err)
			}
			rangeAssertVector(t, result, 0, oracle, tolerance)
			if math.IsNaN(result.ConditionEstimate) || math.IsInf(result.ConditionEstimate, 0) || result.ConditionEstimate < 1 {
				t.Fatalf("invalid measured condition: %g", result.ConditionEstimate)
			}
			if name != "QR" && math.Abs(result.ConditionEstimate/2-1) > 1e-12 {
				t.Fatalf("kappa1(A)=%g, expected 2", result.ConditionEstimate)
			}
			if len(result.ResidualNorms) != 1 || math.IsInf(result.ResidualNorms[0], 0) || math.IsNaN(result.ResidualNorms[0]) {
				t.Fatalf("invalid residual norms: %v", result.ResidualNorms)
			}
		})
	}
}

// TestMatrix_NUM01_DiagonalRetainsTinyComponent protects a representable tiny
// coordinate even when the matrix and RHS scales differ by hundreds of orders.
func TestMatrix_NUM01_DiagonalRetainsTinyComponent(t *testing.T) {
	a := rangeDense(t, [][]float64{{1e308, 0}, {0, 1}})
	b := rangeDense(t, [][]float64{{0}, {1e-16}})
	opts := matrix.FactorOptions{RankTolerance: math.SmallestNonzeroFloat64}
	for name, factor := range rangeFactors(t, a, opts) {
		t.Run(name, func(t *testing.T) {
			result, err := factor.Solve(context.Background(), b)
			if err != nil {
				t.Fatal(err)
			}
			rangeAssertVector(t, result, 0, [2]float64{0, 1e-16}, 2e-13)
		})
	}
}

// TestMatrix_NUM01_PerRHSScalingAndPivotRecovery verifies four independent
// RHS scales and nontrivial LU row/QR column pivots on one positive-definite A.
func TestMatrix_NUM01_PerRHSScalingAndPivotRecovery(t *testing.T) {
	aData := [4]float64{1, 2, 2, 5}
	a := rangeDense(t, [][]float64{{1, 2}, {2, 5}})
	b := rangeDense(t, [][]float64{
		{-1, 3, -1e150, -1e-150, 0},
		{-3, 6.5, -3e150, -3e-150, 0},
	})
	for name, factor := range rangeFactors(t, a, matrix.FactorOptions{}) {
		t.Run(name, func(t *testing.T) {
			result, err := factor.Solve(context.Background(), b)
			if err != nil {
				t.Fatal(err)
			}
			expected := [][2]float64{
				rangeRatSolveTwoByTwo(t, aData, [2]float64{-1, -3}),
				rangeRatSolveTwoByTwo(t, aData, [2]float64{3, 6.5}),
				rangeRatSolveTwoByTwo(t, aData, [2]float64{-1e150, -3e150}),
				rangeRatSolveTwoByTwo(t, aData, [2]float64{-1e-150, -3e-150}),
				{0, 0},
			}
			if len(result.ResidualNorms) != len(expected) {
				t.Fatalf("residual count=%d, expected %d", len(result.ResidualNorms), len(expected))
			}
			for column, want := range expected {
				rangeAssertVector(t, result, column, want, 2e-13)
			}
		})
	}
}

// TestMatrix_NUM01_SubnormalAndHugeIdentity verifies extreme RHS values
// without a false positive overflow and without silently dropping subnormals.
func TestMatrix_NUM01_SubnormalAndHugeIdentity(t *testing.T) {
	a := rangeDense(t, [][]float64{{1, 0}, {0, 1}})
	high := math.Ldexp(1.75, 1023)
	tiny := math.SmallestNonzeroFloat64
	b := rangeDense(t, [][]float64{{high, tiny, 1e-300, 0}, {-high, -tiny, -1e-300, 0}})
	for name, factor := range rangeFactors(t, a, matrix.FactorOptions{}) {
		t.Run(name, func(t *testing.T) {
			result, err := factor.Solve(context.Background(), b)
			if err != nil {
				t.Fatal(err)
			}
			for column := 0; column < b.Cols(); column++ {
				for row := 0; row < b.Rows(); row++ {
					want, _ := b.At(row, column)
					got, readErr := result.Solution.At(row, column)
					if readErr != nil || math.Float64bits(got) != math.Float64bits(want) {
						t.Fatalf("%s identity[%d,%d] got=%g want=%g err=%v", name, row, column, got, want, readErr)
					}
				}
			}
		})
	}
}

// TestMatrix_NUM01_RejectUnrepresentableFinalSolution distinguishes an
// unrepresentable final coefficient from an overflowing intermediate ratio.
func TestMatrix_NUM01_RejectUnrepresentableFinalSolution(t *testing.T) {
	a := rangeDense(t, [][]float64{{0.25, 0}, {0, 0.25}})
	b := rangeDense(t, [][]float64{{math.MaxFloat64}, {math.MaxFloat64}})
	for name, factor := range rangeFactors(t, a, matrix.FactorOptions{}) {
		t.Run(name, func(t *testing.T) {
			result, err := factor.Solve(context.Background(), b)
			if result != nil || !errors.Is(err, matrix.ErrOverflow) {
				t.Fatalf("final solution overflow: result=%#v err=%v", result, err)
			}
		})
	}
}

// TestMatrix_NUM01_ConcurrentFactorSolve ensures that per-call normalization
// state is not shared, even when numerous goroutines reuse one immutable factor.
func TestMatrix_NUM01_ConcurrentFactorSolve(t *testing.T) {
	a := rangeDense(t, [][]float64{{0.75, 0.25}, {0.25, 0.75}})
	b := rangeDense(t, [][]float64{{1}, {-1}})
	factors := rangeFactors(t, a, matrix.FactorOptions{})
	for name, factor := range factors {
		t.Run(name, func(t *testing.T) {
			const workers = 52
			var group sync.WaitGroup
			failures := make(chan error, workers)
			for worker := 0; worker < workers; worker++ {
				group.Add(1)
				go func() {
					defer group.Done()
					result, err := factor.Solve(context.Background(), b)
					if err != nil {
						failures <- err
						return
					}
					if result == nil || result.Solution == nil {
						failures <- errors.New("missing solution")
						return
					}
					got, err := result.Solution.At(0, 0)
					if err != nil || math.Abs(got-2) > 1e-13 {
						failures <- errors.New("incorrect simultaneous solution")
					}
				}()
			}
			group.Wait()
			close(failures)
			for err := range failures {
				t.Error(err)
			}
		})
	}
}

// rangeFaultMatrix wraps one input Matrix while preserving its public shape.
// Solve must return the original backend read error, never a partial result.
type rangeFaultMatrix struct {
	matrix.Matrix
	cause error
}

// At deliberately fails at the read boundary without altering factor inputs.
func (r *rangeFaultMatrix) At(_, _ int) (float64, error) {
	return 0, r.cause
}

// TestMatrix_NUM01_CancellationAndReadFailure preserves the existing external
// error precedence: context errors first, then RHS reader failures.
func TestMatrix_NUM01_CancellationAndReadFailure(t *testing.T) {
	a := rangeDense(t, [][]float64{{1, 0}, {0, 1}})
	b := rangeDense(t, [][]float64{{1}, {2}})
	for name, factor := range rangeFactors(t, a, matrix.FactorOptions{}) {
		t.Run(name, func(t *testing.T) {
			canceled, stop := context.WithCancel(context.Background())
			stop()
			result, err := factor.Solve(canceled, b)
			if result != nil || !errors.Is(err, context.Canceled) {
				t.Fatalf("canceled context: result=%#v err=%v", result, err)
			}
			unavailable := errors.New("sensor store: read unavailable")
			result, err = factor.Solve(context.Background(), &rangeFaultMatrix{Matrix: b, cause: unavailable})
			if result != nil || !errors.Is(err, unavailable) {
				t.Fatalf("backend error: result=%#v err=%v", result, err)
			}
		})
	}
}

// TestMatrix_NUM01_ExtremeDynamicRangeInsideOneRHS checks that a huge
// observation does not erase a second, independently representable coordinate.
// The exact diagonal answer is [1,1], regardless of the enormous condition.
func TestMatrix_NUM01_ExtremeDynamicRangeInsideOneRHS(t *testing.T) {
	a := rangeDense(t, [][]float64{{1e308, 0}, {0, 1}})
	b := rangeDense(t, [][]float64{{1e308}, {1}})
	opts := matrix.FactorOptions{RankTolerance: math.SmallestNonzeroFloat64}
	for name, factor := range rangeFactors(t, a, opts) {
		t.Run(name, func(t *testing.T) {
			result, err := factor.Solve(context.Background(), b)
			if err != nil {
				t.Fatal(err)
			}
			rangeAssertVector(t, result, 0, [2]float64{1, 1}, 2e-13)
		})
	}
}
