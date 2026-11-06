// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw_test verifies the exported numeric helper through the public API.
// Independent assertions protect finite rounding, invalid shape and range classification.
package dtw_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/dtw"
)

// TestSquaredL2OfDifferences verifies stable squared local costs and explicit non-finite signals.
// It exercises close large coordinates, equal huge coordinates, mixed scales,
// a representable subnormal square, and truly unrepresentable squared distances.
//
// Implementation:
//   - Stage 1: define fixed coordinate-vector pairs with independently known outcomes.
//   - Stage 2: call the public squared-L2 helper on each read-only pair.
//   - Stage 3: compare finite values exactly and classify +Inf or NaN explicitly.
//
// Behavior highlights:
//   - Exact equal large coordinates must not produce cancellation or overflow.
//   - A real unrepresentable squared distance must not be silently clamped.
//
// Inputs:
//   - Fixed finite vectors of equal shape, plus one invalid-shape fixture.
//
// Returns:
//   - Test failure on incorrect numeric or invalid-domain classification.
//
// Errors:
//   - This helper has no error return; special float64 values are verified directly.
//
// Determinism:
//   - Feature coordinates are visited in ascending order.
//
// Complexity:
//   - Time O(d) and O(1) auxiliary space per fixture.
//
// Notes:
//   - AlignMatrix converts these non-finite signals to package-owned sentinels.
//
// AI-Hints:
//   - Do not replace coordinate differences with squared norm identities.
//   - Do not apply an absolute threshold that erases representable subnormals.
func TestSquaredL2OfDifferences(t *testing.T) {
	for _, tc := range []struct {
		name string
		x, y []float64
		want float64
	}{
		{"near-large", []float64{1e9}, []float64{1e9 + 1}, 1},
		{"equal-huge", []float64{1e200}, []float64{1e200}, 0},
		{"mixed-scale", []float64{1e100, 1e-100}, []float64{0, 0}, 1e200},
		{"subnormal-square", []float64{math.Ldexp(1, -537)}, []float64{0}, math.SmallestNonzeroFloat64},
		{"overflow", []float64{1e200}, []float64{0}, math.Inf(1)},
		{"invalid-shape", []float64{1, 2}, []float64{1}, math.NaN()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := dtw.SquaredL2OfDifferences(tc.x, tc.y)
			if math.IsNaN(tc.want) {
				if !math.IsNaN(got) {
					t.Fatalf("got %g; want NaN on invalid shape", got)
				}
				return
			}
			if got != tc.want {
				t.Fatalf("got %g; want %g", got, tc.want)
			}
		})
	}
}
