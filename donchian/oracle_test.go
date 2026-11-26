// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package donchian_test compares public numerical results with direct-window rational expectations.
// This file uses only exported production operations and preserves explicit fixture ownership.
// AI-Hints: Never call production arithmetic to compute an expected midpoint.
package donchian_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/lvlath/go/donchian"
)

// TestDonchian_OracleParity compares monotonic queues with direct independent windows.
// Ascending, descending, oscillating, and all-warmup histories exercise queue lifetimes.
//
// Implementation:
//   - Stage 1: Generate oscillating, ascending, and descending histories at the selected lengths.
//   - Stage 2: Compute all declared period/mode combinations.
//   - Stage 3: Compare every result row with an independently scanned complete window.
//
// Behavior highlights:
//   - Compare every result row with an independently scanned complete window.
//
// Inputs:
//   - explicit periods, both modes, and deterministic generated histories.
//
// Returns:
//   - No value; a violated assertion fails the owning test goroutine.
//
// Errors:
//   - Unexpected public errors fail the test; expected failures are classified with errors.Is.
//
// Determinism:
//   - All fixtures and nested parameter orders are fixed and contain no external state.
//
// Complexity:
//   - O(n*P) per test oracle, unlike the O(n) production call.
//
// Notes:
//   - The checks use only public results and error identities; no private state is inspected.
//
// AI-Hints:
//   - Never reuse the production queue or midpoint to manufacture the expected result.
func TestDonchian_OracleParity(t *testing.T) {
	for _, descending := range []bool{false, true} {
		high, low := monotoneBars(257, descending)
		for _, period := range []int{1, 2, 3, 14, 64, 128, 257, 258, math.MaxInt} {
			for _, exclude := range []bool{false, true} {
				opts := donchian.Options{Period: period, ExcludeCurrent: exclude}
				t.Run(fmt.Sprintf("descending=%t/P=%d/exclude=%t", descending, period, exclude), func(t *testing.T) {
					assertResult(t, mustCompute(t, high, low, opts), exactOracle(high, low, opts))
				})
			}
		}
	}
	high, low := syntheticBars(513)
	for period := 1; period <= 33; period++ {
		for _, exclude := range []bool{false, true} {
			opts := donchian.Options{Period: period, ExcludeCurrent: exclude}
			assertResult(t, mustCompute(t, high, low, opts), exactOracle(high, low, opts))
		}
	}
}

// TestDonchian_MidpointRoundingOracle checks numerical branch boundaries and reproducible bit patterns.
// Exact rational means determine expected binary64 bits, including subnormal zero signs.
//
// Implementation:
//   - Stage 1: Enumerate exponent/subnormal edge values and reproducible seeded bit-pattern pairs.
//   - Stage 2: Order each finite pair into a valid Period=1 candle and call Compute.
//   - Stage 3: Compare Middle bit-for-bit with exact rational addition, division, and one rounding.
//
// Behavior highlights:
//   - Compare Middle bit-for-bit with exact rational addition, division, and one rounding.
//
// Inputs:
//   - boundary magnitudes and a local fixed-seed sequence spanning all exponents.
//
// Returns:
//   - No value; a violated assertion fails the owning test goroutine.
//
// Errors:
//   - Unexpected public errors fail the test; expected failures are classified with errors.Is.
//
// Determinism:
//   - The edge list is fixed and the local pseudo-random generator has explicit seed 34.
//
// Complexity:
//   - O(k) pairs with bounded binary64 rational arithmetic.
//
// Notes:
//   - The checks use only public results and error identities; no private state is inspected.
//
// AI-Hints:
//   - Do not enlarge an epsilon after a mismatch; this contract requires exact bits.
func TestDonchian_MidpointRoundingOracle(t *testing.T) {
	values := []float64{
		0, math.Copysign(0, -1),
		math.SmallestNonzeroFloat64, 2 * math.SmallestNonzeroFloat64,
		3 * math.SmallestNonzeroFloat64, 5 * math.SmallestNonzeroFloat64,
		-math.SmallestNonzeroFloat64, -3 * math.SmallestNonzeroFloat64,
		math.Nextafter(0x1p-1022, 0), 0x1p-1022,
		math.Nextafter(0x1p-1022, 1), math.Nextafter(0x1p-1021, 0),
		0x1p-1021, math.Nextafter(0x1p-1021, 1),
		-0x1p-1022, -math.Nextafter(0x1p-1022, 1),
		1, math.Nextafter(1, 2), math.Nextafter(math.Nextafter(1, 2), 2),
		-1, math.Nextafter(-1, -2),
		math.MaxFloat64 / 2, math.Nextafter(math.MaxFloat64/2, math.Inf(1)),
		math.MaxFloat64, math.Nextafter(math.MaxFloat64, 0),
		-math.MaxFloat64, -math.MaxFloat64 / 2,
	}
	for _, a := range values {
		for _, b := range values {
			high, low := a, b
			if high < low {
				high, low = low, high
			}
			result := mustCompute(t, []float64{high}, []float64{low}, donchian.Options{Period: 1})
			assertFloats(t, result.Middle, []float64{exactMidpoint(high, low)})
		}
	}
	// Procedural bit patterns explore the complete exponent range, not just prices
	// near one. The RNG is local with an explicit, reproducible seed.
	rng := rand.New(rand.NewSource(34))
	for i := 0; i < 8192; i++ {
		a, b := math.Float64frombits(rng.Uint64()), math.Float64frombits(rng.Uint64())
		if math.IsNaN(a) || math.IsInf(a, 0) || math.IsNaN(b) || math.IsInf(b, 0) {
			continue
		}
		if a < b {
			a, b = b, a
		}
		result := mustCompute(t, []float64{a}, []float64{b}, donchian.Options{Period: 1})
		assertFloats(t, result.Middle, []float64{exactMidpoint(a, b)})
	}
}

// TestDonchian_ExactAffineFixtures checks carefully representable scaling and translation.
// The direct oracle remains authoritative after transforming admitted boundary values.
//
// Implementation:
//   - Stage 1: Build quarter-unit candles and select exactly representable scale/shift transformations.
//   - Stage 2: Compute original and transformed histories in both modes.
//   - Stage 3: Check oracle parity and the exact transformed boundary triples on ready rows.
//
// Behavior highlights:
//   - Check oracle parity and the exact transformed boundary triples on ready rows.
//
// Inputs:
//   - binary-exact quarter-unit data and exactly representable affine transformations.
//
// Returns:
//   - No value; a violated assertion fails the owning test goroutine.
//
// Errors:
//   - Unexpected public errors fail the test; expected failures are classified with errors.Is.
//
// Determinism:
//   - Transformations are visited in literal order and do not depend on observed outputs.
//
// Complexity:
//   - O(n*P) oracle work per transformation.
//
// Notes:
//   - The checks use only public results and error identities; no private state is inspected.
//
// AI-Hints:
//   - General floating-point translations need not commute with rounded midpoint evaluation.
func TestDonchian_ExactAffineFixtures(t *testing.T) {
	high, low := syntheticBars(129)
	for _, exclude := range []bool{false, true} {
		opts := donchian.Options{Period: 14, ExcludeCurrent: exclude}
		original := mustCompute(t, high, low, opts)
		for _, transform := range []struct{ scale, shift float64 }{{8, 0}, {1, 0x1p40}, {4, -128}} {
			h, l := make([]float64, len(high)), make([]float64, len(low))
			for i := range h {
				h[i], l[i] = high[i]*transform.scale+transform.shift, low[i]*transform.scale+transform.shift
			}
			got := mustCompute(t, h, l, opts)
			assertResult(t, got, exactOracle(h, l, opts))
			for i, valid := range original.Valid {
				if valid {
					assertFloats(t, []float64{got.Upper[i], got.Middle[i], got.Lower[i]}, []float64{
						original.Upper[i]*transform.scale + transform.shift,
						original.Middle[i]*transform.scale + transform.shift,
						original.Lower[i]*transform.scale + transform.shift,
					})
				}
			}
		}
	}
}
