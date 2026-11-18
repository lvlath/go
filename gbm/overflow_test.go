// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gbm_test anchors dynamic-range and cancellation regressions.
// Extreme fixtures verify public outcomes independently of internal big.Float.
package gbm_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/gbm"
)

// TestGBM05AvoidableExpOverflow protects S0=1e-300 with large positive growth.
func TestGBM05AvoidableExpOverflow(t *testing.T) {
	params := gbm.Parameters{Initial: 1e-300, Drift: 1000, Unit: gbm.TimeYears}
	result, err := gbm.SimulateTerminal(context.Background(), params, 1, &scripted{})
	if err != nil {
		t.Fatal(err)
	}
	mustNear(t, math.Log(result.Values[1]), math.Log(params.Initial)+1000, 1e-13)
	mustEqualBool(t, result.Complete, true, "avoidable overflow must not fail")
}

// TestGBM06TrueUnderflow distinguishes mathematical failure from bad input.
func TestGBM06TrueUnderflow(t *testing.T) {
	params := gbm.Parameters{Initial: 1, Volatility: 1e200}
	result, err := gbm.SimulateTerminal(context.Background(), params, 1e-300, &scripted{})
	mustErrorIs(t, err, gbm.ErrOverflow)
	mustPrefix(t, result, 1)
}

// TestGBM_AvoidableSigmaSquaredOverflow prevents squaring sigma in float64.
func TestGBM_AvoidableSigmaSquaredOverflow(t *testing.T) {
	params := gbm.Parameters{Initial: 1, Volatility: 1e160}
	result, err := gbm.SimulateTerminal(context.Background(), params, 1e-320, &scripted{})
	if err != nil {
		t.Fatal(err)
	}
	// The oracle's float64 scale is finite even though sigma*sigma is not.
	scale := 1e160 * math.Sqrt(1e-320)
	mustNear(t, result.Values[1], math.Exp(-0.5*scale*scale), 1e-13)
}

// TestHugeDriftItoCancellation verifies big-precision sqrt before subtraction.
func TestHugeDriftItoCancellation(t *testing.T) {
	params := gbm.Parameters{Initial: 45.75, Drift: 2, Volatility: 2}
	result, err := gbm.SimulateTerminal(context.Background(), params, 1e308, &scripted{})
	if err != nil {
		t.Fatal(err)
	}
	mustNear(t, result.Values[1], params.Initial, 0)
}

// TestInitialExtremesWithZeroLogReturn avoids log(S0)->exp(log(S0)) damage.
func TestInitialExtremesWithZeroLogReturn(t *testing.T) {
	for _, initial := range []float64{math.MaxFloat64, math.SmallestNonzeroFloat64} {
		params := gbm.Parameters{Initial: initial, Drift: 0.5, Volatility: 1}
		result, err := gbm.SimulateTerminal(context.Background(), params, 1, &scripted{})
		if err != nil {
			t.Fatalf("initial=%g: %v", initial, err)
		}
		if math.Float64bits(result.Values[1]) != math.Float64bits(initial) {
			t.Fatalf("initial=%g changed into %g without log growth", initial, result.Values[1])
		}
	}
}

// TestSubnormalIsLegitimate checks that no arbitrary epsilon replaces a tiny price.
func TestSubnormalIsLegitimate(t *testing.T) {
	params := gbm.Parameters{Initial: 1, Drift: -740}
	result, err := gbm.SimulateTerminal(context.Background(), params, 1, &scripted{})
	if err != nil {
		t.Fatal(err)
	}
	if !(result.Values[1] > 0 && result.Values[1] < math.SmallestNonzeroFloat64*100) {
		t.Fatalf("expected legal positive subnormal, got %.17g", result.Values[1])
	}
}

// TestSubnormalAndMaximumInitialScaling verifies true final price rounding.
func TestSubnormalAndMaximumInitialScaling(t *testing.T) {
	cases := []struct {
		name    string
		initial float64
		drift   float64
	}{
		{"minimum subnormal gets larger", math.SmallestNonzeroFloat64, 0.5},
		{"maximum finite gets smaller", math.MaxFloat64, -0.4},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := gbm.SimulateTerminal(context.Background(),
				gbm.Parameters{Initial: tc.initial, Drift: tc.drift}, 1, &scripted{})
			if err != nil {
				t.Fatal(err)
			}
			expected := tc.initial * math.Exp(tc.drift)
			if tc.initial == math.SmallestNonzeroFloat64 {
				// The result is only a few subnormal ULPs, so its exact bits matter.
				if math.Float64bits(result.Values[1]) != math.Float64bits(expected) {
					t.Fatalf("subnormal boundary price=%.17g, expected %.17g", result.Values[1], expected)
				}
			} else {
				// math.Exp(drift) is itself rounded to binary64 before multiplication.
				// Its last ULP need not match the more accurate wide converter.
				mustNear(t, result.Values[1], expected, 2e-15)
			}
		})
	}
}

// TestUnrepresentableUpperAndLowerBoundaries protects against fake clamping.
func TestUnrepresentableUpperAndLowerBoundaries(t *testing.T) {
	for _, tc := range []gbm.Parameters{
		{Initial: math.MaxFloat64, Drift: 2},
		{Initial: math.SmallestNonzeroFloat64, Drift: -2},
	} {
		result, err := gbm.SimulateTerminal(context.Background(), tc, 1, &scripted{})
		mustErrorIs(t, err, gbm.ErrOverflow)
		mustPrefix(t, result, 1)
	}
}

// TestNumericFailureDiscardsOnlyFailedStep enforces equal-length partial prefixes.
func TestNumericFailureDiscardsOnlyFailedStep(t *testing.T) {
	source := &scripted{draws: []float64{0, 1e308}}
	result, err := gbm.Simulate(context.Background(),
		gbm.Parameters{Initial: 1, Drift: 0.5, Volatility: 1},
		gbm.TimeGrid{Values: []float64{0, 1, 2, 3}}, source)
	mustErrorIs(t, err, gbm.ErrOverflow)
	mustPrefix(t, result, 2)
	mustEqualInt(t, source.count, 2, "one Normal per attempted step")
	mustNear(t, result.Times[1], 1, 0)
	mustNear(t, result.Values[1], 1, 0)
}

// TestNonFiniteGaussianSampleIsRuntimeError preserves the valid prefix.
func TestNonFiniteGaussianSampleIsRuntimeError(t *testing.T) {
	for _, bad := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		source := &scripted{draws: []float64{0, bad}}
		result, err := gbm.Simulate(context.Background(),
			gbm.Parameters{Initial: 1, Volatility: 1},
			gbm.TimeGrid{Values: []float64{0, 1, 2}}, source)
		mustErrorIs(t, err, gbm.ErrNonFinite)
		mustPrefix(t, result, 2)
		mustEqualInt(t, source.count, 2, "Normal consumption until bad sample")
	}
}

// TestExtremeAnchorCompensatedGrowth handles exp(-1000) and exp(1400) cases
// where the exponential alone cannot be represented but S0*exp(growth) can.
func TestExtremeAnchorCompensatedGrowth(t *testing.T) {
	testCases := []struct {
		name    string
		initial float64
		growth  float64
	}{
		{"large initial and shrinking log growth", math.MaxFloat64, -1000},
		{"tiny initial and expanding log growth", math.SmallestNonzeroFloat64, 1400},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := gbm.SimulateTerminal(context.Background(),
				gbm.Parameters{Initial: tc.initial, Drift: tc.growth}, 1, &scripted{})
			if err != nil {
				t.Fatal(err)
			}
			mustEqualBool(t, result.Complete, true, "compensated representable value")
			// Log2 handles every finite subnormal input through its exact
			// binary exponent; some ordinary Log implementations lose range
			// when passed the smallest positive float64 numbers.
			expectedLog := math.Log2(tc.initial)*math.Ln2 + tc.growth
			measuredLog := math.Log2(result.Values[1]) * math.Ln2
			mustNear(t, measuredLog, expectedLog, 1e-13)
		})
	}
}

// TestExtremeDiffusionShockCancellation verifies a representable shock can
// cancel an individually unrepresentable Ito correction without false failure.
func TestExtremeDiffusionShockCancellation(t *testing.T) {
	const sigma = 1e308
	source := &scripted{draws: []float64{sigma / 2}}
	result, err := gbm.SimulateTerminal(context.Background(),
		gbm.Parameters{Initial: 3.125, Drift: 0, Volatility: sigma}, 1, source)
	if err != nil {
		t.Fatal(err)
	}
	mustEqualInt(t, source.count, 1, "one huge Gaussian sample")
	if math.Float64bits(result.Values[1]) != math.Float64bits(3.125) {
		t.Fatalf("compensated correction/shock should preserve initial: %.17g", result.Values[1])
	}
}

// TestSubnormalHalfULPDecision checks final binary64 rounding near zero.
// A correctly rounded positive price just above half the smallest subnormal
// must survive; one just below half must be treated as an underflowed zero.
func TestSubnormalHalfULPDecision(t *testing.T) {
	initial := math.SmallestNonzeroFloat64
	nearHalf := -math.Ln2
	kept, err := gbm.SimulateTerminal(context.Background(),
		gbm.Parameters{Initial: initial, Drift: nearHalf}, 1, &scripted{})
	if err != nil {
		t.Fatal(err)
	}
	if math.Float64bits(kept.Values[1]) != math.Float64bits(initial) {
		t.Fatalf("above-half subnormal must round to initial, got %.17g", kept.Values[1])
	}

	belowHalf := math.Nextafter(nearHalf, math.Inf(-1))
	rejected, err := gbm.SimulateTerminal(context.Background(),
		gbm.Parameters{Initial: initial, Drift: belowHalf}, 1, &scripted{})
	mustErrorIs(t, err, gbm.ErrOverflow)
	mustPrefix(t, rejected, 1)
}
