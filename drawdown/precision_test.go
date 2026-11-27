// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// File precision_test.go protects exact rounding and non-floating witness selection.
package drawdown_test

import (
	"math"
	"math/rand"
	"testing"
)

// TestDrawdown_ExactSmallDepth preserves represented adjacent-value losses at ordinary peaks.
func TestDrawdown_ExactSmallDepth(t *testing.T) {
	mustGolden(t, []float64{1, math.Nextafter(1, 0)}, []float64{0, 0x1p-53}, 0, 1, -1)
	mustGolden(t, []float64{3, math.Nextafter(3, 0)}, []float64{0, 0x1p-51 / 3}, 0, 1, -1)
}

// TestDrawdown_DoubleRoundingRegression rejects a known one-ULP error from a rounded intermediate difference.
func TestDrawdown_DoubleRoundingRegression(t *testing.T) {
	// Plain (peak-equity)/peak rounds one ULP too high on this exact pair.
	mustGolden(t, []float64{0x1.e1c95ed8fda5ep+1, 0x1.2fd6c8eb3bc4dp+0},
		[]float64{0, 0x1.5e8daeead819bp-1}, 0, 1, -1)
}

// TestDrawdown_RoundedWitnessCollision distinguishes exact ordering from equal published floating depths.
func TestDrawdown_RoundedWitnessCollision(t *testing.T) {
	// Both depths publish as 1, but the later residual ratio is strictly smaller.
	mustGolden(t, []float64{1, 0x1p-55, 2, 0x1p-55, 2}, []float64{0, 1, 0, 1, 0}, 2, 3, 4)
	// A real total loss must displace an earlier positive residual rounded to 1.
	mustGolden(t, []float64{1, 0x1p-55, 0}, []float64{0, 1, 1}, 0, 2, -1)
	// A larger residual at a new peak still rounds to 1 but must not win.
	mustGolden(t, []float64{1, 0x1p-56, 2, 0x1p-54, 2}, []float64{0, 1, 0, 1, 0}, 0, 1, 2)
}

// TestDrawdown_NearestEvenMidpoints checks both integer-grid midpoint parities.
func TestDrawdown_NearestEvenMidpoints(t *testing.T) {
	// Depth grid ties with both even and odd quotient parity and near 1.
	const peak = 0x1p54
	for _, equity := range []float64{1, 2, 3, 5, 7, math.Nextafter(1, 0), math.Nextafter(1, 2), math.Nextafter(3, 0), math.Nextafter(3, 4)} {
		input := []float64{peak, equity}
		mustResult(t, mustCompute(t, input), rationalOracle(input))
	}
	mustGolden(t, []float64{peak, 1, 3, 5, 7}, []float64{0, 1, 1 - 0x1p-52, 1 - 0x1p-52, 1 - 0x1p-51}, 0, 1, -1)
}

// TestDrawdown_SubnormalAndExtremeEquity protects the full finite exponent domain and gradual underflow.
func TestDrawdown_SubnormalAndExtremeEquity(t *testing.T) {
	unit := math.SmallestNonzeroFloat64
	for _, input := range [][]float64{
		{unit}, {unit, 0}, {3 * unit, unit, 2 * unit}, {5 * unit, 2 * unit, 3 * unit},
		{7 * unit, 3 * unit, 4 * unit}, {0x1.0000000000001p-1022, 0x1p-1023},
		{math.MaxFloat64, math.Nextafter(math.MaxFloat64, 0), math.MaxFloat64},
		{math.MaxFloat64, unit, 0}, {math.MaxFloat64, math.MaxFloat64 / 2, unit},
	} {
		mustResult(t, mustCompute(t, input), rationalOracle(input))
	}
}

// TestDrawdown_RoundingOraclePairs checks dense, half-peak and adjacent pairs against independent rational truth.
func TestDrawdown_RoundingOraclePairs(t *testing.T) {
	rng := rand.New(rand.NewSource(73608))
	for i := 0; i < 2000; i++ {
		peak := finitePositive(rng)
		// Dense sampling near half, deep-loss grid and peak; arbitrary exponents
		// are covered by the independent raw-bit oracle suite as well.
		for _, value := range []float64{peak * rng.Float64(), math.Nextafter(peak, 0), peak * .5} {
			input := []float64{peak, value}
			mustResult(t, mustCompute(t, input), rationalOracle(input))
		}
	}
}

// TestDrawdown_ExhaustiveSubnormalGrid covers the rounded half-peak boundary
// for every lower subnormal in a dense small grid, including odd peak integers.
// Each pair is checked against exact represented-input rational arithmetic.
func TestDrawdown_ExhaustiveSubnormalGrid(t *testing.T) {
	const maxUnits = 64
	for p := 1; p <= maxUnits; p++ {
		peak := float64(p) * math.SmallestNonzeroFloat64
		for e := 0; e <= p; e++ {
			input := []float64{peak, float64(e) * math.SmallestNonzeroFloat64}
			mustResult(t, mustCompute(t, input), rationalOracle(input))
		}
	}
}

// TestDrawdown_ExactCrossPeakNeighborhood stresses near-tie cross products.
// Adjacent representable operands can move the true order without moving either
// rounded depth. Testing different exponents avoids relying on equal denominators.
func TestDrawdown_ExactCrossPeakNeighborhood(t *testing.T) {
	for _, exponent := range []int{-1000, -500, -1, 0, 1, 500, 1000} {
		peak := math.Ldexp(1.75, exponent)
		higher := peak * 2
		for _, fraction := range []float64{0x1p-54, 0x1p-53, 0.25, 0.5, 0.75, math.Nextafter(1, 0)} {
			equity := peak * fraction
			// Values are positive in all regimes and both peak values remain finite.
			for _, later := range []float64{math.Nextafter(equity*2, 0), equity * 2, math.Nextafter(equity*2, math.Inf(1))} {
				input := []float64{peak, equity, higher, min(higher, later), higher}
				mustResult(t, mustCompute(t, input), rationalOracle(input))
			}
		}
	}
}
