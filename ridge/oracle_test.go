// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ridge_test uses exact rationals rather than the production QR kernel.
package ridge_test

import (
	"math/big"
	"testing"
)

// scalarOracle solves beta = sum(x*y)/(sum(x*x)+n*lambda) with exact rational
// arithmetic for the no-intercept, one-feature mathematical test problem.
func scalarOracle(t *testing.T, x, y []int64, lambda int64) (*big.Rat, *big.Rat, *big.Rat, *big.Rat) {
	t.Helper()
	if len(x) != len(y) || len(x) == 0 {
		t.Fatal("invalid oracle shape")
	}
	sumXY, sumXX := new(big.Int), new(big.Int)
	for i := range x {
		sumXY.Add(sumXY, new(big.Int).Mul(big.NewInt(x[i]), big.NewInt(y[i])))
		sumXX.Add(sumXX, new(big.Int).Mul(big.NewInt(x[i]), big.NewInt(x[i])))
	}
	den := new(big.Int).Add(sumXX, new(big.Int).Mul(big.NewInt(int64(len(x))), big.NewInt(lambda)))
	beta := new(big.Rat).SetFrac(sumXY, den)
	sse := new(big.Rat)
	for i := range x {
		residual := new(big.Rat).Sub(new(big.Rat).SetInt64(y[i]), new(big.Rat).Mul(new(big.Rat).SetInt64(x[i]), beta))
		sse.Add(sse, new(big.Rat).Mul(residual, residual))
	}
	penalty := new(big.Rat).Mul(new(big.Rat).SetFrac64(lambda, 2), new(big.Rat).Mul(beta, beta))
	objective := new(big.Rat).Add(new(big.Rat).Quo(sse, new(big.Rat).SetInt64(int64(2*len(x)))), penalty)
	return beta, sse, penalty, objective
}

// exactRidgeOracle solves the SMALL unstandardized ridge normal equations with
// arbitrary-precision rational arithmetic. It is deliberately independent of
// the production augmented QR algorithm and never called from package ridge.
// Every matrix entry is formed from integer fixture values, not rounded decimal
// approximations; Gaussian elimination uses exact rational row operations.
func exactRidgeOracle(t *testing.T, x [][]int64, y []int64, lambdaNumerator, lambdaDenominator int64, noIntercept bool) ([]*big.Rat, *big.Rat, *big.Rat, *big.Rat) {
	t.Helper()
	if len(x) == 0 || len(y) != len(x) || len(x[0]) == 0 || lambdaNumerator <= 0 || lambdaDenominator <= 0 {
		t.Fatal("invalid exact ridge oracle fixture")
	}
	n, p := len(x), len(x[0])
	q, offset := p, 0
	if !noIntercept {
		q++
		offset = 1
	}
	lambda := new(big.Rat).SetFrac(big.NewInt(lambdaNumerator), big.NewInt(lambdaDenominator))
	augmented := make([][]*big.Rat, q)
	for j := range augmented {
		augmented[j] = make([]*big.Rat, q+1)
		for k := range augmented[j] {
			augmented[j][k] = new(big.Rat)
		}
	}

	// Exact augmented normal equations are a test oracle, not the solver design.
	// The ridge penalty adds n*Lambda only to slope diagonal entries.
	for i := 0; i < n; i++ {
		if len(x[i]) != p {
			t.Fatal("ragged exact ridge fixture")
		}
		row := make([]*big.Rat, q)
		if offset != 0 {
			row[0] = new(big.Rat).SetInt64(1)
		}
		for j, value := range x[i] {
			row[j+offset] = new(big.Rat).SetInt64(value)
		}
		for j := 0; j < q; j++ {
			for k := 0; k < q; k++ {
				augmented[j][k].Add(augmented[j][k], new(big.Rat).Mul(row[j], row[k]))
			}
			augmented[j][q].Add(augmented[j][q], new(big.Rat).Mul(row[j], new(big.Rat).SetInt64(y[i])))
		}
	}
	for j := offset; j < q; j++ {
		augmented[j][j].Add(augmented[j][j], new(big.Rat).Mul(new(big.Rat).SetInt64(int64(n)), lambda))
	}

	// Exact pivoting is only a row-selection mechanism for Gaussian elimination;
	// it does not rely on production QR or its numerical rank heuristic.
	for col := 0; col < q; col++ {
		pivot := col
		for pivot < q && augmented[pivot][col].Sign() == 0 {
			pivot++
		}
		if pivot == q {
			t.Fatal("unexpected rank loss in positive-penalty exact oracle")
		}
		augmented[pivot], augmented[col] = augmented[col], augmented[pivot]
		divisor := new(big.Rat).Set(augmented[col][col])
		for k := col; k <= q; k++ {
			augmented[col][k].Quo(augmented[col][k], divisor)
		}
		for row := 0; row < q; row++ {
			if row == col {
				continue
			}
			factor := new(big.Rat).Set(augmented[row][col])
			for k := col; k <= q; k++ {
				augmented[row][k].Sub(augmented[row][k], new(big.Rat).Mul(factor, augmented[col][k]))
			}
		}
	}
	parameters := make([]*big.Rat, q)
	for j := range parameters {
		parameters[j] = new(big.Rat).Set(augmented[j][q])
	}

	var sse, norm big.Rat
	for i := 0; i < n; i++ {
		prediction := new(big.Rat)
		if offset != 0 {
			prediction.Set(parameters[0])
		}
		for j, value := range x[i] {
			prediction.Add(prediction, new(big.Rat).Mul(new(big.Rat).SetInt64(value), parameters[j+offset]))
		}
		residual := new(big.Rat).Sub(new(big.Rat).SetInt64(y[i]), prediction)
		sse.Add(&sse, new(big.Rat).Mul(residual, residual))
	}
	for j := offset; j < q; j++ {
		norm.Add(&norm, new(big.Rat).Mul(parameters[j], parameters[j]))
	}
	penalty := new(big.Rat).Mul(lambda, new(big.Rat).Quo(&norm, new(big.Rat).SetInt64(2)))
	objective := new(big.Rat).Add(new(big.Rat).Quo(&sse, new(big.Rat).SetInt64(int64(2*n))), penalty)
	return parameters, new(big.Rat).Set(&sse), penalty, objective
}
