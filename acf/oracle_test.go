// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package acf_test implements an independent exact-rational autocorrelation oracle.
// Reference reductions share no floating-point implementation code with acf.
package acf_test

import (
	"math"
	"math/big"
	"testing"

	"github.com/lvlath/go/acf"
)

// oracleRational computes one lag from exact fractions of binary64 input bits.
// It independently derives the global mean, the two S values, and optional
// n/(n-k) scaling without calling production normalization or pair kernels.
func oracleRational(x []float64, lag int, noCenter bool, adjusted bool) float64 {
	n := len(x)
	v := make([]*big.Rat, n)
	sum := new(big.Rat)
	for i, a := range x {
		v[i] = new(big.Rat).SetFloat64(a)
		sum.Add(sum, v[i])
	}
	mean := new(big.Rat)
	if !noCenter {
		mean.Quo(sum, big.NewRat(int64(n), 1))
	}
	for i := range v {
		v[i].Sub(v[i], mean)
	}
	dot := func(k int) *big.Rat {
		out := new(big.Rat)
		for i := k; i < n; i++ {
			out.Add(out, new(big.Rat).Mul(v[i], v[i-k]))
		}
		return out
	}
	ratio := new(big.Rat).Quo(dot(lag), dot(0))
	if adjusted {
		ratio.Mul(ratio, big.NewRat(int64(n), int64(n-lag)))
	}
	f, _ := ratio.Float64()
	return f
}

// TestCompute_RationalOracle anchors representative signed and extreme vectors
// against exact rational arithmetic with a single global mean for all lags.
func TestCompute_RationalOracle(t *testing.T) {
	fixtures := [][]float64{{1, 2, 3}, {1, -1, 1, -1}, {0, 0, 1, 0, 0}, {3, 8, -5, 2, 7, 11, -1, 6, 9, 0, 4, -3}, {math.MaxFloat64, -math.MaxFloat64, math.MaxFloat64}, {math.Nextafter(1e12, math.Inf(1)), 1e12, math.Nextafter(1e12, math.Inf(-1))}}
	for _, x := range fixtures {
		for _, nc := range []bool{false, true} {
			for _, den := range []acf.Denominator{acf.CommonDenominator, acf.LagAdjusted} {
				opts := acf.Options{NoCenter: nc, Denominator: den}
				got, err := acf.Compute(x, len(x)-1, opts)
				if err != nil {
					t.Fatal(err)
				}
				for k, v := range got.Values {
					want := oracleRational(x, k, nc, den == acf.LagAdjusted)
					mustClose(t, v, want)
				}
				mustPairCounts(t, got, len(x))
			}
		}
	}
}

// TestCompute_RationalOracleGenerated checks reproducible n<=12 rational windows.
// Both estimators and both centering policies are independently evaluated at all lags.
func TestCompute_RationalOracleGenerated(t *testing.T) {
	seed := uint64(0x9e3779b97f4a7c15)
	next := func() uint64 { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed }
	for n := 2; n <= 12; n++ {
		for trial := 0; trial < 30; trial++ {
			x := make([]float64, n)
			for i := range x {
				x[i] = float64(int64(next()%201)-100) / 8
			}
			for _, noCenter := range []bool{false, true} {
				for _, den := range []acf.Denominator{acf.CommonDenominator, acf.LagAdjusted} {
					opts := acf.Options{NoCenter: noCenter, Denominator: den}
					got, err := acf.Compute(x, n-1, opts)
					if err != nil {
						t.Fatalf("n=%d trial=%d opts=%+v: %v", n, trial, opts, err)
					}
					for lag, v := range got.Values {
						mustClose(t, v, oracleRational(x, lag, noCenter, den == acf.LagAdjusted))
					}
				}
			}
		}
	}
}
