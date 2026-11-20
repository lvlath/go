// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix_test fuzzes the SVD boundary against analytic diagonal oracles.
// It does not rely on the implementation's own decomposition for expected values.
package matrix_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/matrix"
)

// FuzzSVD_IndependentColumnScales validates independently representable diagonal
// singular modes across the complete normal/subnormal binary64 exponent range.
//
// Inputs are bounded before numeric construction, so any failure is a real
// decomposition/regression failure rather than a test generator overflow.
func FuzzSVD_IndependentColumnScales(f *testing.F) {
	f.Add(int16(-1073), int16(1024), false)
	f.Add(int16(0), int16(-900), true)
	f.Add(int16(-800), int16(-801), false)
	f.Add(int16(1000), int16(999), true)
	f.Fuzz(func(t *testing.T, originalA, originalB int16, transpose bool) {
		const minimumExponent, maximumExponent = -1073, 1024
		const exponentSpan = maximumExponent - minimumExponent + 1
		// Modulo after conversion to int gives a bounded shift; negative values
		// are translated to the same deterministic nonnegative residue range.
		exponentA := (int(originalA)%exponentSpan+exponentSpan)%exponentSpan + minimumExponent
		exponentB := (int(originalB)%exponentSpan+exponentSpan)%exponentSpan + minimumExponent
		first := math.Ldexp(0.5, exponentA)
		second := math.Ldexp(0.5, exponentB)
		input := []float64{first, 0, 0, second}
		if transpose {
			// Use a 2-by-3 wide diagonal embedding to exercise transpose duality.
			input = []float64{first, 0, 0, 0, second, 0}
		}
		rows, cols := 2, 2
		if transpose {
			cols = 3
		}
		result, err := matrix.SVDDefault(context.Background(), svdFixture(t, rows, cols, input))
		if err != nil {
			t.Fatalf("analytic diagonal input: %v", err)
		}
		if len(result.SingularValues) != 2 || result.SingularValues[0] != math.Max(first, second) ||
			result.SingularValues[1] != math.Min(first, second) || result.Rank < 1 || !result.Converged {
			t.Fatalf("exponents (%d,%d), shape %dx%d: values %v rank %d",
				exponentA, exponentB, rows, cols, result.SingularValues, result.Rank)
		}
		svdAssertOrthonormal(t, result, rows, cols)
	})
}
