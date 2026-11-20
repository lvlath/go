// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pacf_test protects large-offset, subnormal, and near-overflow contracts.
// These tests enter exclusively through Compute, not private arithmetic helpers.
package pacf_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/pacf"
)

// TestNumericRegression_BinaryExponentVarianceRecovery verifies positive
// physical variances near both ends of the representable binary64 range.
func TestNumericRegression_BinaryExponentVarianceRecovery(t *testing.T) {
	cases := []struct {
		name  string
		scale float64
		lag   int
	}{
		{"small normal scale, subnormal variance", math.Ldexp(1, -525), 2},
		{"smallest positive variance", math.Ldexp(1, -537), 0},
		{"large physical variance", math.Ldexp(1, 510), 2},
		{"near upper exponent", math.Ldexp(1, 511), 2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			values := []float64{tc.scale, -tc.scale, tc.scale, -tc.scale}
			got, err := pacf.Compute(context.Background(), values, pacf.Options{MaxLag: tc.lag})
			if err != nil {
				t.Fatalf("representable variance failed: %v", err)
			}
			want := tc.scale * tc.scale
			if got.PredictionVariance[0] != want {
				t.Fatalf("gamma0 %.17g want exact power of two %.17g", got.PredictionVariance[0], want)
			}
			for lag, variance := range got.PredictionVariance {
				if variance <= 0 || math.IsNaN(variance) || math.IsInf(variance, 0) {
					t.Fatalf("invalid variance at lag %d: %.17g", lag, variance)
				}
			}
		})
	}
}

// TestNumericRegression_UnrepresentablePositiveVariance reports ErrOverflow
// instead of replacing positive but unrepresentable values with fake epsilons.
func TestNumericRegression_UnrepresentablePositiveVariance(t *testing.T) {
	cases := []struct {
		name string
		x    []float64
	}{
		{"underflow to zero", []float64{math.Ldexp(1, -600), -math.Ldexp(1, -600)}},
		{"overflow from finite operands", []float64{math.Ldexp(1, 530), -math.Ldexp(1, 530)}},
		{"overflow after extreme offset centering", []float64{-1e308, 1e308}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := pacf.Compute(context.Background(), tc.x, pacf.Options{MaxLag: 1, Center: true})
			mustNilState(t, result, true, "unrepresentable prediction variance")
			mustErrorIs(t, err, pacf.ErrOverflow)
		})
	}
}

// TestNumericRegression_TinyCenteringSpread confirms that integer-spaced
// fluctuations around a high power-of-two baseline survive mean fitting.
func TestNumericRegression_TinyCenteringSpread(t *testing.T) {
	baseline := math.Ldexp(1, 52)
	fluctuation := []float64{1, 4, 2, -3, 0, 5, 3, -2, 7, -1, 4, 1}
	x := make([]float64, len(fluctuation))
	for i, value := range fluctuation {
		x[i] = baseline + value
	}
	centered, err := pacf.ComputeCentered(context.Background(), x, 8)
	if err != nil {
		t.Fatal(err)
	}
	base, err := pacf.ComputeCentered(context.Background(), fluctuation, 8)
	if err != nil {
		t.Fatal(err)
	}
	mustSlices(t, centered.Values, base.Values, 3e-13)
	mustSlices(t, centered.PredictionVariance, base.PredictionVariance, 3e-13)
}

// TestNumericRegression_UncenteredConstantAllLags independently checks that
// an uncentered constant has gamma[k]=a^2*(n-k)/n, not pair-adjusted gamma[k].
func TestNumericRegression_UncenteredConstantAllLags(t *testing.T) {
	const observations = 32
	const constant = 6.25
	x := make([]float64, observations)
	for i := range x {
		x[i] = constant
	}
	got, err := pacf.ComputeUncentered(context.Background(), x, observations-1)
	if err != nil {
		t.Fatal(err)
	}
	if !closeEnough(got.Values[1], float64(observations-1)/observations, 1e-14) {
		t.Fatalf("lag1 %.17g does not use the common n denominator", got.Values[1])
	}
	mustEqualInt(t, got.Count, observations, "constant sample count")
	for lag := range got.PredictionVariance {
		if got.PredictionVariance[lag] <= 0 {
			t.Fatalf("lag %d: constant uncentered variance must be positive", lag)
		}
	}
}

// TestNumericRegression_WithoutCenterTranslationChangesEstimand confirms that
// adding a DC offset is not a no-op when the physical zero is fixed.
func TestNumericRegression_WithoutCenterTranslationChangesEstimand(t *testing.T) {
	base := []float64{1, -2, 4, -3, 2, 0, 3, -4, 5, -1, 2, -2}
	shifted := make([]float64, len(base))
	for i, value := range base {
		shifted[i] = value + 120
	}
	basePACF, err := pacf.ComputeUncentered(context.Background(), base, 5)
	if err != nil {
		t.Fatal(err)
	}
	shiftedPACF, err := pacf.ComputeUncentered(context.Background(), shifted, 5)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(basePACF.Values[1]-shiftedPACF.Values[1]) < 0.1 {
		t.Fatal("uncentered second moments unexpectedly invariant to large DC offset")
	}
}
