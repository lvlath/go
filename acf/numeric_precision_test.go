// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package acf_test protects binary64 centering, underflow, and scaled-ratio laws.
// Every oracle in this file operates on exact rational representations of float64 inputs.
package acf_test

import (
	"errors"
	"math"
	"math/big"
	"testing"

	"github.com/lvlath/go/acf"
)

// exactMeanAndPositiveEnergy computes independent rational sample statistics.
// Unlike the production kernel, it never scales or rounds intermediate products.
// A positive-energy result distinguishes true constants from representable spread.
func exactMeanAndPositiveEnergy(values []float64, noCenter bool) (float64, bool) {
	meanRat := new(big.Rat)
	if !noCenter {
		for _, value := range values {
			meanRat.Add(meanRat, new(big.Rat).SetFloat64(value))
		}
		meanRat.Quo(meanRat, big.NewRat(int64(len(values)), 1))
	}
	energy := new(big.Rat)
	for _, value := range values {
		deviation := new(big.Rat).Sub(new(big.Rat).SetFloat64(value), meanRat)
		energy.Add(energy, new(big.Rat).Mul(deviation, deviation))
	}
	mean, _ := meanRat.Float64()
	return mean, energy.Sign() > 0
}

// meanWithinInputScale bounds the forward error of a finite binary64 sample
// mean relative to the magnitude of the actual operands. A mean that is nearly
// zero after cancellation cannot reliably promise a fixed number of its own
// ULPs: even a well-conditioned scaled sum has input-scale rounding error.
// The exact subnormal tie regressions below deliberately remain bit-exact.
func meanWithinInputScale(got, want float64, values []float64) bool {
	if got == want {
		return true
	}
	if math.IsNaN(got) || math.IsInf(got, 0) || math.IsNaN(want) || math.IsInf(want, 0) {
		return false
	}
	maxInput := 0.0
	for _, value := range values {
		if a := math.Abs(value); a > maxInput {
			maxInput = a
		}
	}
	// For n<=12, a 16-epsilon input-scale budget covers the normalized
	// division/centering stages and rounding of the final public Mean.
	const meanErrorFactor = 16.0
	machineEpsilon := math.Nextafter(1, 2) - 1
	bound := meanErrorFactor * machineEpsilon * maxInput
	// One-half of the smallest subnormal can still round by one whole ULP.
	if bound < math.SmallestNonzeroFloat64 {
		bound = math.SmallestNonzeroFloat64
	}
	return math.Abs(got-want) <= bound
}

// TestCompute_MeanSubnormalFMA is a named regression for premature underflow
// in the computation base+(meanShift*scale), where the parenthesized product
// rounds to zero even when the correctly rounded overall mean is nonzero.
func TestCompute_MeanSubnormalFMA(t *testing.T) {
	const smallest = math.SmallestNonzeroFloat64
	cases := []struct {
		name  string
		input []float64
	}{
		{"positive-halfway-tie", []float64{smallest, 2 * smallest}},
		{"negative-halfway-tie", []float64{-2 * smallest, -smallest}},
		{"mixed-small", []float64{0, smallest, 2 * smallest}},
		{"negative-offset", []float64{-smallest, 0, smallest}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			wantedMean, positiveEnergy := exactMeanAndPositiveEnergy(tc.input, false)
			if !positiveEnergy {
				t.Fatal("expected exact nonzero energy")
			}
			got, err := acf.ComputeDefault(tc.input, len(tc.input)-1)
			if err != nil {
				t.Fatal(err)
			}
			if got.Mean != wantedMean {
				t.Fatalf("rounded mean bits=0x%x want=0x%x", math.Float64bits(got.Mean), math.Float64bits(wantedMean))
			}
			for lag, estimate := range got.Values {
				mustClose(t, estimate, oracleRational(tc.input, lag, false, false))
			}
		})
	}
}

// TestCompute_OracleExtremeMagnitudes ties both estimator conventions and
// the fitted mean to independent exact binary64 rational calculations.
func TestCompute_OracleExtremeMagnitudes(t *testing.T) {
	maxBelow := math.Nextafter(math.MaxFloat64, 0)
	cases := [][]float64{
		{math.SmallestNonzeroFloat64, 2 * math.SmallestNonzeroFloat64},
		{math.MaxFloat64, -math.MaxFloat64, math.MaxFloat64},
		{maxBelow, math.MaxFloat64, maxBelow},
		{math.Nextafter(1e12, math.Inf(1)), 1e12, math.Nextafter(1e12, math.Inf(-1))},
		{1e154, -1e154, 1e154, -1e154},
	}
	for _, values := range cases {
		for _, noCenter := range []bool{false, true} {
			for _, estimator := range []acf.Denominator{acf.CommonDenominator, acf.LagAdjusted} {
				got, err := acf.Compute(values, len(values)-1, acf.Options{NoCenter: noCenter, Denominator: estimator})
				wantMean, positiveEnergy := exactMeanAndPositiveEnergy(values, noCenter)
				if !positiveEnergy {
					mustErrorIs(t, err, acf.ErrZeroVariance)
					if got != nil {
						t.Fatal("result on zero energy")
					}
					continue
				}
				if err != nil {
					t.Fatalf("unexpected numeric error: %v", err)
				}
				if !meanWithinInputScale(got.Mean, wantMean, values) {
					t.Fatalf("mean=%.17g rational=%.17g", got.Mean, wantMean)
				}
				for lag, estimate := range got.Values {
					mustClose(t, estimate, oracleRational(values, lag, noCenter, estimator == acf.LagAdjusted))
				}
			}
		}
	}
}

// TestCompute_OracleAdversarialBinary64 uses a fixed seed and mixed floating-
// point exponents to catch arithmetic changes that ordinary small integers miss.
// It includes subnormal data, high-offset nextafter neighbors, opposite signs,
// and true constant/all-zero windows without relying on internal kernels.
func TestCompute_OracleAdversarialBinary64(t *testing.T) {
	seed := uint64(0x9e3779b97f4a7c15)
	next := func() uint64 {
		seed ^= seed << 13
		seed ^= seed >> 7
		seed ^= seed << 17
		return seed
	}
	const trials = 280
	for trial := 0; trial < trials; trial++ {
		n := 2 + int(next()%11)
		values := make([]float64, n)
		exponent := int(next()%2098) - 1074
		anchor := math.Ldexp(1, exponent)
		if anchor == 0 || math.IsInf(anchor, 0) {
			anchor = math.SmallestNonzeroFloat64
		}
		for i := range values {
			sign := 1.0
			if next()%2 == 0 {
				sign = -1
			}
			switch trial % 6 {
			case 0:
				values[i] = math.Nextafter(anchor, math.Inf(1))
				if next()%2 == 0 {
					values[i] = math.Nextafter(anchor, math.Inf(-1))
				}
			case 1:
				values[i] = sign * anchor
			case 2:
				values[i] = sign * math.Ldexp(float64(next()%257+1), int(next()%2000)-1074)
			case 3:
				values[i] = sign * float64(next()%63) * anchor / 64
			case 4:
				values[i] = sign * math.MaxFloat64
				if next()%2 == 0 {
					values[i] = sign * math.SmallestNonzeroFloat64
				}
			default:
				values[i] = float64(int(next()%41) - 20)
			}
			if math.IsInf(values[i], 0) || math.IsNaN(values[i]) {
				values[i] = 0
			}
		}
		for _, noCenter := range []bool{false, true} {
			wantMean, positiveEnergy := exactMeanAndPositiveEnergy(values, noCenter)
			for _, estimator := range []acf.Denominator{acf.CommonDenominator, acf.LagAdjusted} {
				got, err := acf.Compute(values, n-1, acf.Options{Denominator: estimator, NoCenter: noCenter})
				if !positiveEnergy {
					if got != nil || !errors.Is(err, acf.ErrZeroVariance) {
						t.Fatalf("trial=%d zero energy: result=%v error=%v", trial, got, err)
					}
					continue
				}
				if err != nil {
					t.Fatalf("trial=%d unexpected error: %v", trial, err)
				}
				if !meanWithinInputScale(got.Mean, wantMean, values) {
					t.Fatalf("trial=%d mean=%.17g rational=%.17g values=%v", trial, got.Mean, wantMean, values)
				}
				for lag, estimate := range got.Values {
					mustClose(t, estimate, oracleRational(values, lag, noCenter, estimator == acf.LagAdjusted))
				}
				mustPairCounts(t, got, n)
			}
		}
	}
}
