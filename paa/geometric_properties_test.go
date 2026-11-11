// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
//
// Package paa external mathematical regression tests for conservation of
// fractional geometry and the aligned Euclidean Jensen lower-bound law.
// The rational oracle enumerates all overlaps independently of production.
//
// AI-Hints:
//   - The mathematical lower bound does not automatically certify a rounded
//     binary64 threshold comparison; certified pruning needs outward rounding.
//   - This is an aligned Euclidean bound, not an unrestricted DTW bound.
package paa_test

import (
	"math"
	"math/big"
	"math/rand"
	"testing"

	"github.com/lvlath/go/paa"
)

// TestTransform_FractionalJensenLowerBound protects the exact mathematical
// property required when later constructing SAX-style Euclidean reductions.
// For equal-length, aligned inputs, n/w times the squared distance between
// exact PAA averages must not exceed the original squared Euclidean distance.
func TestTransform_FractionalJensenLowerBound(t *testing.T) {
	rng := rand.New(rand.NewSource(7072026))

	// The deterministic domain deliberately includes every non-divisible n/w
	// shape up to n=18; it also contains repeated, negative and zero samples.
	for n := 2; n <= 18; n++ {
		for trial := 0; trial < 8; trial++ {
			left := make([]float64, n)
			right := make([]float64, n)
			for i := range left {
				left[i] = float64(rng.Intn(121)-60) / 8
				right[i] = float64(rng.Intn(121)-60) / 8
			}

			// Compute the original squared L2 distance as exact binary rationals
			// so large cancellation or float rounding cannot falsify the theorem.
			originalSquared := new(big.Rat)
			for i := range left {
				diff := new(big.Rat).Sub(
					new(big.Rat).SetFloat64(left[i]),
					new(big.Rat).SetFloat64(right[i]),
				)
				originalSquared.Add(originalSquared, new(big.Rat).Mul(diff, diff))
			}

			for w := 1; w <= n; w++ {
				// The oracle is intentionally quadratic only in this small test.
				oracleLeft := oraclePAA(t, left, w)
				oracleRight := oraclePAA(t, right, w)
				gotLeft, err := paa.Transform(left, w)
				if err != nil {
					t.Fatalf("left n=%d w=%d: %v", n, w, err)
				}
				gotRight, err := paa.Transform(right, w)
				if err != nil {
					t.Fatalf("right n=%d w=%d: %v", n, w, err)
				}
				assertExactSegments(t, gotLeft, n, w)
				assertExactSegments(t, gotRight, n, w)

				// Check each published mean against the exact rational oracle;
				// then derive Jensen's inequality from oracle values, not from
				// rounded results that might cross an extremely tight cutoff.
				compressedSquared := new(big.Rat)
				for j := 0; j < w; j++ {
					assertOracleExactly(t, gotLeft.Values[j], oracleLeft[j])
					assertOracleExactly(t, gotRight.Values[j], oracleRight[j])
					diff := new(big.Rat).Sub(oracleLeft[j], oracleRight[j])
					compressedSquared.Add(compressedSquared, new(big.Rat).Mul(diff, diff))
				}
				compressedSquared.Mul(compressedSquared, big.NewRat(int64(n), int64(w)))
				if compressedSquared.Cmp(originalSquared) > 0 {
					t.Fatalf("Jensen violated n=%d w=%d trial=%d: projected %s > raw %s",
						n, w, trial, compressedSquared.RatString(), originalSquared.RatString())
				}
			}
		}
	}
}

// TestTransform_FractionalExtremeConstant covers an exact constant mean even
// when n/w is fractional and every individual contributor is at finite
// binary64 maximum magnitude. It catches accidental raw value*overlap use.
func TestTransform_FractionalExtremeConstant(t *testing.T) {
	const n = 17
	const w = 6
	for _, value := range []float64{math.MaxFloat64, -math.MaxFloat64} {
		values := make([]float64, n)
		for i := range values {
			values[i] = value
		}
		result, err := paa.Transform(values, w)
		if err != nil {
			t.Fatalf("constant %g: %v", value, err)
		}
		assertExactSegments(t, result, n, w)
		for j, mean := range result.Values {
			if math.Float64bits(mean) != math.Float64bits(value) {
				t.Fatalf("constant %g, segment %d: got %.17g", value, j, mean)
			}
		}
	}
}

// TestTransform_TailSpikeCannotDisappear proves that the final non-divisible
// source cell influences its last segment rather than being truncated or
// silently replaced with a zero-valued padding sample.
func TestTransform_TailSpikeCannotDisappear(t *testing.T) {
	values := []float64{100, 99, 102, 98, 104, 101, 105, 106, 100, 102, 99, 97,
		103, 104, 106, 101, 108, 110, 111, 113, 109, 125, 950}
	const w = 7
	first, err := paa.Transform(values, w)
	if err != nil {
		t.Fatal(err)
	}
	assertExactSegments(t, first, len(values), w)
	lastIndex := w - 1
	assertOracleExactly(t, first.Values[lastIndex], oraclePAA(t, values, w)[lastIndex])

	// Changing only the last input must change the last PAA segment, and it
	// must not change any earlier segment that has no geometric intersection.
	values[len(values)-1] = 125
	second, err := paa.Transform(values, w)
	if err != nil {
		t.Fatal(err)
	}
	if first.Values[lastIndex] <= second.Values[lastIndex] {
		t.Fatalf("last source cell was discarded: with incident %.17g, without %.17g",
			first.Values[lastIndex], second.Values[lastIndex])
	}
	for j := 0; j < lastIndex; j++ {
		if math.Float64bits(first.Values[j]) != math.Float64bits(second.Values[j]) {
			t.Fatalf("isolated last-sample change affected earlier segment %d", j)
		}
	}
}
