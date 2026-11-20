// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pacf_test expands sentinel precedence and deterministic mathematical invariants.
// These checks exercise public Compute semantics, not implementation-private helpers.
package pacf_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/pacf"
)

// TestContract_ErrorPrecedenceConflictingInputs verifies which sentinel wins
// when multiple public preconditions are violated simultaneously.
func TestContract_ErrorPrecedenceConflictingInputs(t *testing.T) {
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	tests := []struct {
		name string
		ctx  context.Context
		x    []float64
		opts pacf.Options
		want error
	}{
		{"nil context beats nil input", nil, nil, pacf.Options{MaxLag: -1}, pacf.ErrInvalidOption},
		{"cancel beats nil input", canceled, nil, pacf.Options{MaxLag: -1}, context.Canceled},
		{"nil input beats shape", context.Background(), nil, pacf.Options{MaxLag: -1}, pacf.ErrNilInput},
		{"empty input beats shape", context.Background(), []float64{}, pacf.Options{MaxLag: -1}, pacf.ErrEmptyInput},
		{"shape beats NaN", context.Background(), []float64{math.NaN(), 1}, pacf.Options{MaxLag: -1}, pacf.ErrInvalidOption},
		{"shape beats infinity", context.Background(), []float64{math.Inf(1), 1}, pacf.Options{MaxLag: 2}, pacf.ErrInvalidOption},
		{"nonfinite beats domain", context.Background(), []float64{0, math.NaN()}, pacf.Options{MaxLag: 1}, pacf.ErrNonFinite},
		{"zero centered sample", context.Background(), []float64{4}, pacf.Options{Center: true}, pacf.ErrZeroVariance},
		{"zero uncentered singleton", context.Background(), []float64{0}, pacf.Options{}, pacf.ErrZeroVariance},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result, err := pacf.Compute(tc.ctx, tc.x, tc.opts)
			mustNilState(t, result, true, "multi-fault input")
			mustErrorIs(t, err, tc.want)
		})
	}
}

// TestContract_AffineEquivarianceCentered checks invariance under admissible
// positive/negative scaling and an exactly representable additive translation.
func TestContract_AffineEquivarianceCentered(t *testing.T) {
	x := []float64{2, 9, -3, 4, 0, 7, -2, 8, 1, -4, 6, 5, 3, -5, 10, 2}
	original, err := pacf.ComputeCentered(context.Background(), x, 11)
	if err != nil {
		t.Fatal(err)
	}
	const factor = -8.0
	const offset = 1024.0
	transformed := make([]float64, len(x))
	for i, value := range x {
		transformed[i] = factor*value + offset
	}
	got, err := pacf.ComputeCentered(context.Background(), transformed, 11)
	if err != nil {
		t.Fatal(err)
	}
	mustSlices(t, got.Values, original.Values, 2e-12)
	for lag := range got.PredictionVariance {
		want := factor * factor * original.PredictionVariance[lag]
		if !closeEnough(got.PredictionVariance[lag], want, 2e-12) {
			t.Fatalf("lag %d: variance %.17g want %.17g", lag, got.PredictionVariance[lag], want)
		}
	}
}

// TestContract_FullLengthLagRange verifies no off-by-one at the upper boundary,
// exact lag-zero anchoring, and positivity for a representative real-valued window.
func TestContract_FullLengthLagRange(t *testing.T) {
	x := []float64{2, 5, -3, 7, -1, 4, 0, 8, -4, 6, 1, -2, 9, 3, -5, 11}
	for _, center := range []bool{false, true} {
		got, err := pacf.Compute(context.Background(), x, pacf.Options{MaxLag: len(x) - 1, Center: center})
		if err != nil {
			t.Fatal(err)
		}
		mustEqualInt(t, len(got.Values), len(x), "full maximum-lag output size")
		mustEqualInt(t, len(got.PredictionVariance), len(x), "full variance output size")
		if got.Values[0] != 1 {
			t.Fatalf("center=%t must anchor lag zero exactly", center)
		}
		for lag, coefficient := range got.Values {
			if math.IsNaN(coefficient) || math.IsInf(coefficient, 0) || math.Abs(coefficient) >= 1 && lag != 0 {
				t.Fatalf("center=%t lag=%d invalid reflection %.17g", center, lag, coefficient)
			}
			if got.PredictionVariance[lag] <= 0 || math.IsInf(got.PredictionVariance[lag], 0) || math.IsNaN(got.PredictionVariance[lag]) {
				t.Fatalf("center=%t lag=%d invalid variance", center, lag)
			}
		}
	}
}
