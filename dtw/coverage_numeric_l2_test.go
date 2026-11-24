// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw_test verifies the exceptional numerical domain of the public squared-L2 helper.
// These cases distinguish dirty coordinates, overflow during subtraction, and finite accumulation.
package dtw_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/dtw"
)

// TestCoverageSquaredL2OfDifferences_NonFiniteAndDifferenceOverflow tests classified local arithmetic.
// Subtraction may overflow even when both coordinates are finite, while a dirty coordinate
// invalidates the complete vector rather than being ignored after an overflowing feature.
// Implementation:
//   - Stage 1: construct finite opposite-sign extremes and nonfinite coordinate pairs.
//   - Stage 2: invoke the public helper in both operand orientations.
//   - Stage 3: verify NaN versus positive infinity without numeric tolerance.
//
// Determinism: fixed vectors, including a dirty coordinate after an overflowing feature.
// Complexity: Time O(d), Space O(1) for each vector pair.
// AI-Hints: Do not convert NaN or true overflow into a zero distance or a large finite match.
func TestCoverageSquaredL2OfDifferences_NonFiniteAndDifferenceOverflow(t *testing.T) {
	for _, tc := range []struct {
		name    string
		x, y    []float64
		invalid bool
	}{
		{"nan", []float64{math.NaN()}, []float64{0}, true},
		{"positive-infinity", []float64{math.Inf(1)}, []float64{0}, true},
		{"negative-infinity", []float64{math.Inf(-1)}, []float64{0}, true},
		{"opposite-finite-extremes", []float64{math.MaxFloat64}, []float64{-math.MaxFloat64}, false},
		{"invalid-after-overflow", []float64{math.MaxFloat64, math.NaN()}, []float64{-math.MaxFloat64, 0}, true},
		{"overflow-after-invalid", []float64{math.NaN(), math.MaxFloat64}, []float64{0, -math.MaxFloat64}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Symmetric squared distance must preserve the same numeric classification.
			for _, got := range []float64{dtw.SquaredL2OfDifferences(tc.x, tc.y), dtw.SquaredL2OfDifferences(tc.y, tc.x)} {
				if tc.invalid {
					if !math.IsNaN(got) {
						t.Fatalf("dirty coordinate vector returned %v; want NaN", got)
					}
				} else {
					mustInf(t, got, "finite operands with overflowing subtraction")
				}
			}
		})
	}
	// The matrix facade converts the overflow signal into both public numeric sentinels.
	x := regressionDense(t, 1, 1, math.MaxFloat64)
	y := regressionDense(t, 1, 1, -math.MaxFloat64)
	got, err := dtw.AlignMatrix(x, y)
	mustErrorIs(t, err, dtw.ErrOverflow, "matrix difference overflow")
	mustErrorIs(t, err, dtw.ErrNaNInf, "matrix numeric family")
	if got != nil {
		t.Fatalf("matrix local overflow published %+v", got)
	}
}

// TestCoverageSquaredL2OfDifferences_ExactPermutation verifies both exponent-update directions.
// The integer 3-4-5 fixture has exactly representable differences, squares and sum.
// Implementation: reverse feature order and require the same exact squared distance in both cases.
// Complexity: Time O(d), Space O(1); no tolerance is required for these exact values.
func TestCoverageSquaredL2OfDifferences_ExactPermutation(t *testing.T) {
	for _, vector := range [][]float64{{3, 4}, {4, 3}} {
		got := dtw.SquaredL2OfDifferences(vector, []float64{0, 0})
		mustFloatEqual(t, got, 25, 0, "exact squared distance under feature permutation")
	}
}
