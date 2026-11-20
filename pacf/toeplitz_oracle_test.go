// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pacf_test contains an independent exact Yule-Walker reference solver.
// Its arbitrary-precision computations are separate from production PACF arithmetic.
package pacf_test

import (
	"context"
	"math"
	"math/big"
	"testing"

	"github.com/lvlath/go/pacf"
)

// rationalAutocov builds exact rational sample covariances from integer samples.
// It delegates to a shared exact covariance oracle that never invokes PACF code.
func rationalAutocov(x []int64, center bool) []*big.Rat {
	exact := make([]*big.Rat, len(x))
	for i, value := range x {
		exact[i] = new(big.Rat).SetInt64(value)
	}
	return rationalAutocovExact(exact, center)
}

// rationalAutocovFloat64 first represents each finite input float exactly as a
// rational; this tests numerical fidelity relative to actual binary64 inputs.
func rationalAutocovFloat64(x []float64, center bool) []*big.Rat {
	exact := make([]*big.Rat, len(x))
	for i, value := range x {
		exact[i] = new(big.Rat).SetFloat64(value)
	}
	return rationalAutocovExact(exact, center)
}

// rationalAutocovExact computes the stipulated biased n-divisor covariance
// in arbitrary-precision rational arithmetic with a single global mean.
// It never forms E[xy]-mean^2 or mixes the n and n-k denominators.
func rationalAutocovExact(x []*big.Rat, center bool) []*big.Rat {
	n := len(x)
	mean := new(big.Rat)
	if center {
		for _, value := range x {
			mean.Add(mean, value)
		}
		mean.Quo(mean, new(big.Rat).SetInt64(int64(n)))
	}
	deviations := make([]*big.Rat, n)
	for i, value := range x {
		deviations[i] = new(big.Rat).Sub(value, mean)
	}
	gamma := make([]*big.Rat, n)
	for lag := range gamma {
		sum := new(big.Rat)
		for i := lag; i < n; i++ {
			sum.Add(sum, new(big.Rat).Mul(deviations[i], deviations[i-lag]))
		}
		gamma[lag] = sum.Quo(sum, new(big.Rat).SetInt64(int64(n)))
	}
	return gamma
}

// rationalSolve independently solves the order-m Toeplitz Yule-Walker system
// using exact Gauss-Jordan elimination instead of Levinson recursion.
// It returns the last AR coefficient and the gamma[0]-dot(a,gamma[1:m]) residual.
func rationalSolve(gamma []*big.Rat, m int) (*big.Rat, *big.Rat) {
	if m == 0 {
		return new(big.Rat).SetInt64(1), new(big.Rat).Set(gamma[0])
	}
	a := make([][]*big.Rat, m)
	for i := 0; i < m; i++ {
		a[i] = make([]*big.Rat, m+1)
		for j := 0; j < m; j++ {
			d := i - j
			if d < 0 {
				d = -d
			}
			a[i][j] = new(big.Rat).Set(gamma[d])
		}
		a[i][m] = new(big.Rat).Set(gamma[i+1])
	}
	for k := 0; k < m; k++ {
		pivot := k
		for pivot < m && a[pivot][k].Sign() == 0 {
			pivot++
		}
		if pivot == m {
			return nil, nil
		}
		a[k], a[pivot] = a[pivot], a[k]
		q := new(big.Rat).Set(a[k][k])
		for j := k; j <= m; j++ {
			a[k][j].Quo(a[k][j], q)
		}
		for i := 0; i < m; i++ {
			if i == k {
				continue
			}
			f := new(big.Rat).Set(a[i][k])
			for j := k; j <= m; j++ {
				a[i][j].Sub(a[i][j], new(big.Rat).Mul(f, a[k][j]))
			}
		}
	}
	variance := new(big.Rat).Set(gamma[0])
	for j := 0; j < m; j++ {
		variance.Sub(variance, new(big.Rat).Mul(a[j][m], gamma[j+1]))
	}
	return a[m-1][m], variance
}

// TestPACF_ToeplitzLinearSystemOracle checks each coefficient and prediction
// variance against an independent exact rational Toeplitz solution.
func TestPACF_ToeplitzLinearSystemOracle(t *testing.T) {
	cases := []struct {
		x      []int64
		center bool
		lag    int
	}{
		{[]int64{1, 2, 3}, true, 2}, {[]int64{1, 1}, false, 1},
		{[]int64{5, -2, 3, 6, 1, -4, 2, 9}, true, 5},
		{[]int64{8, 3, 4, 1, 2, 5, 9, 6, 7}, false, 7},
		{[]int64{1, -1, 1, -1, 1, -1, 1, -1}, true, 6},
	}
	for ci, tc := range cases {
		x := make([]float64, len(tc.x))
		for i, v := range tc.x {
			x[i] = float64(v)
		}
		got, err := pacf.Compute(context.Background(), x, pacf.Options{MaxLag: tc.lag, Center: tc.center})
		if err != nil {
			t.Fatalf("case %d: %v", ci, err)
		}
		gamma := rationalAutocov(tc.x, tc.center)
		for m := 0; m <= tc.lag; m++ {
			k, v := rationalSolve(gamma, m)
			if k == nil {
				t.Fatalf("singular oracle case %d lag %d", ci, m)
			}
			kf, _ := k.Float64()
			vf, _ := v.Float64()
			if math.Abs(kf-got.Values[m]) > 3e-11 || !closeEnough(got.PredictionVariance[m], vf, 3e-11) {
				t.Fatalf("case=%d lag=%d k=%.16g want=%.16g v=%.16g want=%.16g", ci, m, got.Values[m], kf, got.PredictionVariance[m], vf)
			}
		}
	}
}

// TestPACF_ExhaustiveSmallIntegerToeplitzOracle independently solves every
// Yule-Walker order for all length-2 through length-6 ternary sample sequences.
// This catches mirrored-coefficient overwrite, sign mistakes, and mixed divisors.
func TestPACF_ExhaustiveSmallIntegerToeplitzOracle(t *testing.T) {
	alphabet := []int64{-1, 0, 1}
	for n := 2; n <= 6; n++ {
		count := 1
		for i := 0; i < n; i++ {
			count *= len(alphabet)
		}
		for pattern := 0; pattern < count; pattern++ {
			x := make([]int64, n)
			floating := make([]float64, n)
			encoded := pattern
			for i := range x {
				x[i] = alphabet[encoded%len(alphabet)]
				floating[i] = float64(x[i])
				encoded /= len(alphabet)
			}
			for _, center := range []bool{false, true} {
				gamma := rationalAutocov(x, center)
				got, err := pacf.Compute(context.Background(), floating, pacf.Options{MaxLag: n - 1, Center: center})
				if gamma[0].Sign() == 0 {
					mustNilState(t, got, true, "zero-energy pattern")
					mustErrorIs(t, err, pacf.ErrZeroVariance)
					continue
				}
				if err != nil {
					t.Fatalf("n=%d pattern=%d centered=%t: %v", n, pattern, center, err)
				}
				for lag := 0; lag < n; lag++ {
					coefficient, variance := rationalSolve(gamma, lag)
					if coefficient == nil || variance.Sign() <= 0 {
						t.Fatalf("oracle lost exact Toeplitz positivity: n=%d pattern=%d lag=%d", n, pattern, lag)
					}
					wantPACF, _ := coefficient.Float64()
					wantVariance, _ := variance.Float64()
					if !closeEnough(got.Values[lag], wantPACF, 3e-11) ||
						!closeEnough(got.PredictionVariance[lag], wantVariance, 3e-11) {
						t.Fatalf("n=%d pattern=%d center=%t lag=%d: PACF %.16g want %.16g, variance %.16g want %.16g",
							n, pattern, center, lag, got.Values[lag], wantPACF,
							got.PredictionVariance[lag], wantVariance)
					}
				}
			}
		}
	}
}

// TestPACF_LargeIntegerOffsetAgainstExactRationalOracle verifies that mean
// subtraction retains small representable deviations near a 2^50 offset.
func TestPACF_LargeIntegerOffsetAgainstExactRationalOracle(t *testing.T) {
	const offset int64 = 1 << 50
	samples := []int64{2, -6, 5, 12, 0, -11, 7, 3, -4, 9, 1, -8, 6, 14}
	for i := range samples {
		samples[i] += offset
	}
	floating := make([]float64, len(samples))
	for i, value := range samples {
		floating[i] = float64(value)
	}
	got, err := pacf.ComputeCentered(context.Background(), floating, 10)
	if err != nil {
		t.Fatal(err)
	}
	gamma := rationalAutocov(samples, true)
	for lag := 0; lag <= 10; lag++ {
		coefficient, variance := rationalSolve(gamma, lag)
		wantPACF, _ := coefficient.Float64()
		wantVariance, _ := variance.Float64()
		if !closeEnough(got.Values[lag], wantPACF, 3e-12) ||
			!closeEnough(got.PredictionVariance[lag], wantVariance, 3e-12) {
			t.Fatalf("large-offset lag %d: PACF=%.17g want=%.17g V=%.17g want=%.17g",
				lag, got.Values[lag], wantPACF, got.PredictionVariance[lag], wantVariance)
		}
	}
}
