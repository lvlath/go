// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package ols_test

import (
	"context"
	"math/big"
	"testing"

	"github.com/lvlath/go/ols"
)

// TestOLS_RationalOracle checks exact binary-rational arithmetic on a small
// full-rank tall design. Rational Gram elimination exists ONLY in this test:
// production must exclusively use the rectangular pivoted QR kernel.
func TestOLS_RationalOracle(t *testing.T) {
	x := [][]float64{{1, 0}, {0, 1}, {1, 1}, {2, -1}, {-1, 2}, {3, 1}}
	y := []float64{4, 1, 3, 9, -2, 8}
	m := mustDense(t, x)
	r, err := ols.Fit(context.Background(), m, y, ols.Options{})
	if err != nil {
		t.Fatal(err)
	}
	// Build exact augmented 3x4 normal-equation system in test-only Rat arithmetic.
	a := make([][]*big.Rat, 3)
	for j := range a {
		a[j] = make([]*big.Rat, 4)
		for k := range a[j] {
			a[j][k] = new(big.Rat)
		}
	}
	for i, row := range x {
		basis := []float64{1, row[0], row[1]}
		for j := 0; j < 3; j++ {
			for k := 0; k < 3; k++ {
				a[j][k].Add(a[j][k], new(big.Rat).Mul(new(big.Rat).SetFloat64(basis[j]), new(big.Rat).SetFloat64(basis[k])))
			}
			a[j][3].Add(a[j][3], new(big.Rat).Mul(new(big.Rat).SetFloat64(basis[j]), new(big.Rat).SetFloat64(y[i])))
		}
	}
	for k := 0; k < 3; k++ {
		if a[k][k].Sign() == 0 {
			t.Fatal("oracle singular fixture")
		}
		for i := k + 1; i < 3; i++ {
			mult := new(big.Rat).Quo(a[i][k], a[k][k])
			for j := k; j < 4; j++ {
				a[i][j].Sub(a[i][j], new(big.Rat).Mul(mult, a[k][j]))
			}
		}
	}
	exact := make([]*big.Rat, 3)
	for i := 2; i >= 0; i-- {
		v := new(big.Rat).Set(a[i][3])
		for j := i + 1; j < 3; j++ {
			v.Sub(v, new(big.Rat).Mul(a[i][j], exact[j]))
		}
		exact[i] = v.Quo(v, a[i][i])
	}
	b, intercept := r.Model.Coefficients()
	got := []float64{intercept, b[0], b[1]}
	for i, v := range exact {
		want, _ := v.Float64()
		mustNear(t, got[i], want, 2e-12)
	}
}

// rationalFitOracle independently solves the exact binary-rational least-squares
// problem for a deliberately small, well-conditioned tall matrix. Normal-equation
// assembly is confined to this test oracle and never to the production OLS kernel.
func rationalFitOracle(t *testing.T, features [][]float64, response []float64, noIntercept bool) ([]float64, float64) {
	t.Helper()
	p := len(features[0])
	q := p
	if !noIntercept {
		q++
	}
	augmented := make([][]*big.Rat, q)
	for i := range augmented {
		augmented[i] = make([]*big.Rat, q+1)
		for j := range augmented[i] {
			augmented[i][j] = new(big.Rat)
		}
	}
	for row, values := range features {
		basis := make([]*big.Rat, q)
		offset := 0
		if !noIntercept {
			basis[0] = new(big.Rat).SetInt64(1)
			offset = 1
		}
		for col, value := range values {
			basis[col+offset] = new(big.Rat).SetFloat64(value)
		}
		y := new(big.Rat).SetFloat64(response[row])
		for i := 0; i < q; i++ {
			for j := 0; j < q; j++ {
				augmented[i][j].Add(augmented[i][j],
					new(big.Rat).Mul(basis[i], basis[j]))
			}
			augmented[i][q].Add(augmented[i][q], new(big.Rat).Mul(basis[i], y))
		}
	}
	// Exact row elimination uses the first nonzero pivot; the fixture is full rank.
	for pivot := 0; pivot < q; pivot++ {
		pivotRow := pivot
		for pivotRow < q && augmented[pivotRow][pivot].Sign() == 0 {
			pivotRow++
		}
		if pivotRow == q {
			t.Fatal("rational oracle fixture is rank deficient")
		}
		augmented[pivot], augmented[pivotRow] = augmented[pivotRow], augmented[pivot]
		for row := pivot + 1; row < q; row++ {
			quotient := new(big.Rat).Quo(augmented[row][pivot], augmented[pivot][pivot])
			for col := pivot; col <= q; col++ {
				augmented[row][col].Sub(augmented[row][col],
					new(big.Rat).Mul(quotient, augmented[pivot][col]))
			}
		}
	}
	theta := make([]*big.Rat, q)
	for i := q - 1; i >= 0; i-- {
		value := new(big.Rat).Set(augmented[i][q])
		for j := i + 1; j < q; j++ {
			value.Sub(value, new(big.Rat).Mul(augmented[i][j], theta[j]))
		}
		theta[i] = value.Quo(value, augmented[i][i])
	}

	// Check SSE against exactly represented original binary float64 inputs.
	sse := new(big.Rat)
	offset := 0
	if !noIntercept {
		offset = 1
	}
	for row, values := range features {
		predicted := new(big.Rat)
		if !noIntercept {
			predicted.Set(theta[0])
		}
		for col, value := range values {
			predicted.Add(predicted, new(big.Rat).Mul(
				new(big.Rat).SetFloat64(value), theta[col+offset]))
		}
		residual := new(big.Rat).Sub(new(big.Rat).SetFloat64(response[row]), predicted)
		sse.Add(sse, new(big.Rat).Mul(residual, residual))
	}
	coefficients := make([]float64, q+1)
	for i, rational := range theta {
		coefficients[i], _ = rational.Float64()
	}
	actualSSE, _ := sse.Float64()
	return coefficients, actualSSE
}

// TestOLS_RationalOracle_IndependentTallModes compares every public transform
// regime against one independent exact-rational oracle on an overdetermined fit.
func TestOLS_RationalOracle_IndependentTallModes(t *testing.T) {
	ctx := context.Background()
	features := [][]float64{
		{-3, 2, 1}, {-2, 5, 4}, {-1, 1, 8}, {0, 7, 3},
		{1, 0, 5}, {2, 9, 2}, {3, 4, 7}, {4, 8, 9},
		{5, 2, 6}, {6, 6, 0}, {7, 3, 10}, {8, 10, 4},
	}
	response := []float64{1.5, -2.5, 6.25, 4.5, -1.75, 11.0, 9.5, 16.25, 8.25, 15.0, 16.5, 20.75}
	for _, opts := range []ols.Options{
		{},
		{Standardize: true},
		{NoIntercept: true},
		{NoIntercept: true, Standardize: true},
	} {
		result, err := ols.Fit(ctx, mustDense(t, features), response, opts)
		if err != nil {
			t.Fatal(err)
		}
		want, wantSSE := rationalFitOracle(t, features, response, opts.NoIntercept)
		beta, intercept := result.Model.Coefficients()
		offset := 0
		if !opts.NoIntercept {
			offset = 1
			mustNear(t, intercept, want[0], 3e-12)
		} else if intercept != 0 {
			t.Fatalf("unexpected origin offset: %v", intercept)
		}
		for col := range beta {
			mustNear(t, beta[col], want[col+offset], 3e-12)
		}
		mustNear(t, result.ResidualSumSquares, wantSSE, 3e-12)
		mustEqualInt(t, result.Rank, len(beta)+offset, "rational rank")
	}
}
