// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw_test checks squared Euclidean range boundaries with exact binary rationals.
// The oracle recomputes all coordinate differences independently of scaled accumulation.
package dtw_test

import (
	"math"
	"math/big"
	"testing"

	"github.com/lvlath/go/dtw"
)

// rationalSquaredL2 computes an exact-real squared distance from original float64 vectors.
// Each finite binary64 coordinate is converted to an exact rational before
// subtraction and multiplication, so no production rounding is reproduced.
//
// Implementation:
//   - Stage 1: create an exact rational accumulator.
//   - Stage 2: form and square each exact coordinate difference.
//   - Stage 3: return the complete rational squared distance.
//
// Behavior highlights:
//   - It measures mathematical range separately from approximate binary64 precision.
//
// Inputs:
//   - x, y: finite equal-length test coordinate vectors.
//
// Returns:
//   - *big.Rat: independently owned exact non-negative squared distance.
//
// Errors:
//   - None; all test fixtures satisfy shape and finiteness preconditions.
//
// Determinism:
//   - Exact feature-index order and rational arithmetic.
//
// Complexity:
//   - O(d) high-precision arithmetic operations with unbounded intermediate precision.
//
// Notes:
//   - Intentionally too expensive for the production per-cell hot loop.
//
// AI-Hints:
//   - Never compare printed decimal float strings as exact-rational input values.
//   - Do not reuse squaredSumOfDifferences or other production routines here.
func rationalSquaredL2(x, y []float64) *big.Rat {
	result := new(big.Rat)
	for index := range x {
		left := new(big.Rat).SetFloat64(x[index])
		right := new(big.Rat).SetFloat64(y[index])
		diff := new(big.Rat).Sub(left, right)
		result.Add(result, new(big.Rat).Mul(diff, diff))
	}
	return result
}

// TestSquaredL2_BoundaryRationalOracle checks finite and overflowing squared norms.
// It distinguishes an exact-real squared value beyond MaxFloat64 from a finite
// rounded squared distance, including the subnormal end of the range.
//
// Implementation:
//   - Stage 1: establish fixed vectors near the largest finite squared-distance boundary.
//   - Stage 2: classify each pair with independent rational squared differences.
//   - Stage 3: compare production range classification and finite rounding within two ULPs.
//
// Behavior highlights:
//   - Large equal coordinates must yield zero without intermediate norm overflow.
//
// Inputs:
//   - Finite, fixed coordinates around powers of two and the subnormal range.
//
// Returns:
//   - Test failure on false overflow, missing overflow, or material finite drift.
//
// Errors:
//   - The numeric helper uses +Inf as a non-finite overflow signal.
//
// Determinism:
//   - Exact-rational oracle and fixed enumeration order.
//
// Complexity:
//   - O(d) multiprecision arithmetic plus O(d) production operations per fixture.
//
// Notes:
//   - Matching an exact-real number bit-for-bit is not required after binary64 rounding.
//
// AI-Hints:
//   - Do not clamp positive subnormal squared costs to an absolute epsilon.
//   - The oracle must remain independent of the production recurrence and scaling.
func TestSquaredL2_BoundaryRationalOracle(t *testing.T) {
	power512 := math.Ldexp(1, 512)
	for _, tc := range []struct {
		name string
		x, y []float64
	}{
		{"just-below-overflow", []float64{math.Nextafter(power512, 0)}, []float64{0}},
		{"exact-overflow", []float64{power512}, []float64{0}},
		{"two-large-finite", []float64{math.Ldexp(1, 511), math.Ldexp(1, 511)}, []float64{0, 0}},
		{"three-large-overflow", []float64{power512, math.Ldexp(1, 511)}, []float64{0, 0}},
		{"equal-huge", []float64{1e200, 1e200}, []float64{1e200, 1e200}},
		{"exact-subnormal", []float64{math.Ldexp(1, -537)}, []float64{0}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			exact := rationalSquaredL2(tc.x, tc.y)
			boundary := new(big.Rat).SetFloat64(math.MaxFloat64)
			got := dtw.SquaredL2OfDifferences(tc.x, tc.y)
			if exact.Cmp(boundary) > 0 {
				if !math.IsInf(got, 1) {
					t.Fatalf("exact squared distance exceeds MaxFloat64 but got %g", got)
				}
				return
			}

			want, _ := exact.Float64()
			if math.IsInf(got, 1) || math.IsNaN(got) {
				t.Fatalf("exact finite squared distance %g classified as %g", want, got)
			}
			if want == got {
				return
			}
			// The allowed error is local-scale ULPs, not a global epsilon that
			// would silently erase genuinely representable tiny squared costs.
			unit := math.Abs(math.Nextafter(want, math.Inf(1)) - want)
			if math.Abs(got-want) > 2*unit {
				t.Fatalf("got %g want %g; difference exceeds two ULPs", got, want)
			}
		})
	}
}
