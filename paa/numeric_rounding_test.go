// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
//
// External regression tests for exact half-way and just-above-half-way rounding.
package paa_test

import (
	"math"
	"math/big"
	"math/rand"
	"testing"

	"github.com/lvlath/go/paa"
)

// TestTransform_RoundsAcrossSubnormalNormalBoundary protects a real rounding
// boundary: the exact average lies 2/3 of one subnormal ULP above min normal.
func TestTransform_RoundsAcrossSubnormalNormalBoundary(t *testing.T) {
	minNormal := math.Float64frombits(1 << 52)
	maxSubnormal := math.Nextafter(minNormal, 0)
	nextNormal := math.Nextafter(minNormal, math.Inf(1))
	nextAfterNext := math.Nextafter(nextNormal, math.Inf(1))

	// These exact means lie either side of a half-ULP or exactly on it.
	// A normal float with an even significand wins the halfway tie.
	tests := []struct {
		name   string
		values []float64
		want   float64
	}{
		{"normalAboveHalf", []float64{minNormal, nextNormal, nextNormal}, nextNormal},
		{"normalBelowHalf", []float64{minNormal, minNormal, nextNormal}, minNormal},
		{"normalHalfEvenDown", []float64{minNormal, nextNormal}, minNormal},
		{"normalHalfEvenUp", []float64{nextNormal, nextAfterNext}, nextAfterNext},
		{"subnormalToNormal", []float64{maxSubnormal, minNormal, minNormal}, minNormal},
		{"negativeAboveHalf", []float64{-minNormal, -nextNormal, -nextNormal}, -nextNormal},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := paa.Transform(tc.values, 1)
			if err != nil {
				t.Fatal(err)
			}
			if math.Float64bits(got.Values[0]) != math.Float64bits(tc.want) {
				t.Fatalf("nearest-even: got %x, want %x", math.Float64bits(got.Values[0]), math.Float64bits(tc.want))
			}
		})
	}
}

// TestTransform_ExactlyRoundedAgainstRationalOracle checks nearest-even
// rounding against the *exact* rational mean of the original binary inputs.
// Unlike tolerance-based comparisons, it catches a single lost subnormal ULP.
func TestTransform_ExactlyRoundedAgainstRationalOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(20261111))
	for n := 1; n <= 14; n++ {
		for trial := 0; trial < 40; trial++ {
			input := make([]float64, n)
			for i := range input {
				raw := rng.Uint64()
				if raw&0x7ff0000000000000 == 0x7ff0000000000000 {
					raw &^= 0x0010000000000000 // avoid NaN/Inf
				}
				input[i] = math.Float64frombits(raw)
			}
			for w := 1; w <= n; w++ {
				res, err := paa.Transform(input, w)
				if err != nil {
					t.Fatalf("n=%d trial=%d w=%d: %v", n, trial, w, err)
				}
				oracle := oraclePAA(t, input, w)
				for j := range res.Values {
					assertOracleExactly(t, res.Values[j], oracle[j])
				}
			}
		}
	}
}

// assertOracleExactly checks the nearest binary64 representation returned by
// Rat.Float64; a zero sign is immaterial for non-identity compression.
func assertOracleExactly(t *testing.T, got float64, exact *big.Rat) {
	t.Helper()
	want, _ := exact.Float64()
	if got == 0 && want == 0 {
		return
	}
	if math.Float64bits(got) != math.Float64bits(want) {
		t.Fatalf("nearest-even mismatch: got=%g (%016x), want=%g (%016x)", got, math.Float64bits(got), want, math.Float64bits(want))
	}
}
