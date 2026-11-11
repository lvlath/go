// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
//
// External tests for safe subrange aggregation and value-only result projection.
// AI-Hints: A subrange uses local segment coordinates; never shift its numerators
// into the parent coordinate space without accounting for the denominator.
package paa_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/lvlath/go/paa"
)

// TestTransformRange_FractionalGeometry proves that the projected window keeps
// all of its samples, validates only the selected values and uses local geometry.
func TestTransformRange_FractionalGeometry(t *testing.T) {
	values := []float64{math.NaN(), 0, 2, 4, 6, 8, math.Inf(1)}
	got, err := paa.TransformRange(values, 1, 6, 2)
	if err != nil {
		t.Fatal(err)
	}
	assertExactSegments(t, got, 5, 2)
	assertOracleExactly(t, got.Values[0], oraclePAA(t, values[1:6], 2)[0])
	assertOracleExactly(t, got.Values[1], oraclePAA(t, values[1:6], 2)[1])
	if got.Values[0] != 1.6 || got.Values[1] != 6.4 {
		t.Fatalf("wrong fractional means: %v", got.Values)
	}
}

// TestTransformRange_EquivalentToExplicitSlice compares the wrapper against
// the canonical facade for numerous valid intervals and output dimensions.
func TestTransformRange_EquivalentToExplicitSlice(t *testing.T) {
	rng := rand.New(rand.NewSource(731))
	values := make([]float64, 70)
	for i := range values {
		values[i] = float64(rng.Intn(2001)-1000) / 32
	}
	for start := 0; start < len(values); start += 7 {
		for end := start + 1; end <= len(values); end += 11 {
			for _, w := range []int{1, (end - start + 1) / 2, end - start} {
				want, err := paa.Transform(values[start:end], w)
				if err != nil {
					t.Fatal(err)
				}
				got, err := paa.TransformRange(values, start, end, w)
				if err != nil {
					t.Fatal(err)
				}
				assertExactSegments(t, got, end-start, w)
				for j := range want.Values {
					if math.Float64bits(got.Values[j]) != math.Float64bits(want.Values[j]) || got.Segments[j] != want.Segments[j] {
						t.Fatalf("range [%d:%d], w=%d, j=%d disagrees with Transform", start, end, w, j)
					}
				}
			}
		}
	}
}

// TestTransformRange_ErrorPrecedence locks the canonical count, nil, empty,
// bounds and numeric order before any potentially panicking slice expression.
func TestTransformRange_ErrorPrecedence(t *testing.T) {
	tests := []struct {
		name   string
		values []float64
		start  int
		end    int
		w      int
		want   error
	}{
		{"countBeforeNil", nil, 0, 1, 0, paa.ErrInvalidOption},
		{"negativeStartAfterNil", nil, -1, 1, 1, paa.ErrNilInput},
		{"reversedBoundsAfterNil", nil, 3, 2, 1, paa.ErrNilInput},
		{"emptyRangeAfterNil", nil, 0, 0, 1, paa.ErrNilInput},
		{"nil", nil, 0, 1, 1, paa.ErrNilInput},
		{"emptySlice", []float64{}, 0, 1, 1, paa.ErrEmptyInput},
		{"emptySourceBeforeInvalidBounds", []float64{}, -1, 0, 1, paa.ErrEmptyInput},
		{"negativeStart", []float64{1, 2}, -1, 1, 1, paa.ErrInvalidOption},
		{"reversedBounds", []float64{1, 2}, 1, 0, 1, paa.ErrInvalidOption},
		{"emptyWindow", []float64{1, 2}, 1, 1, 1, paa.ErrInvalidOption},
		{"endBeyondLength", []float64{1, 2}, 0, 3, 1, paa.ErrInvalidOption},
		{"countBeyondWindowBeforeNonfinite", []float64{math.NaN(), 2}, 0, 1, 2, paa.ErrInvalidOption},
		{"nonfiniteWithinWindow", []float64{1, math.Inf(-1), 3}, 1, 3, 1, paa.ErrNonFinite},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := paa.TransformRange(tt.values, tt.start, tt.end, tt.w)
			mustErrorIs(t, err, tt.want)
			if got != nil {
				t.Fatalf("partial result on error: %+v", got)
			}
		})
	}
}

// TestMeans_ProjectsCanonicalTransform verifies that the convenience wrapper
// drops only geometry, never alters arithmetic or exposes input storage.
func TestMeans_ProjectsCanonicalTransform(t *testing.T) {
	inputs := [][]float64{
		{0, 2, 4, 6, 8},
		{math.MaxFloat64, -math.MaxFloat64, math.SmallestNonzeroFloat64},
		{-0.0, 2, -4, 6},
	}
	for _, input := range inputs {
		for w := 1; w <= len(input); w++ {
			whole, err := paa.Transform(input, w)
			if err != nil {
				t.Fatal(err)
			}
			means, err := paa.Means(input, w)
			if err != nil {
				t.Fatal(err)
			}
			mustEqualInt(t, len(means), w, "means length")
			for j := range means {
				if math.Float64bits(means[j]) != math.Float64bits(whole.Values[j]) {
					t.Fatalf("w=%d j=%d: Means differs from Transform", w, j)
				}
			}
			means[0] = 919
			if input[0] == 919 || whole.Values[0] == 919 {
				t.Fatal("Means result aliases another slice")
			}
		}
	}
}

// TestMeans_ErrorIdentity ensures the projection leaves canonical error
// precedence and nil-output policy unchanged.
func TestMeans_ErrorIdentity(t *testing.T) {
	cases := []struct {
		values []float64
		w      int
		want   error
	}{
		{nil, -1, paa.ErrInvalidOption},
		{nil, 1, paa.ErrNilInput},
		{[]float64{}, 1, paa.ErrEmptyInput},
		{[]float64{math.NaN()}, 1, paa.ErrNonFinite},
		{[]float64{1}, 2, paa.ErrInvalidOption},
	}
	for _, tc := range cases {
		got, err := paa.Means(tc.values, tc.w)
		mustErrorIs(t, err, tc.want)
		if got != nil {
			t.Fatal("Means returned a slice on error")
		}
	}
}

// TestTransform_IdentityPreservesSignedZero verifies a detached identity
// preserves the original binary64 representation as well as its numeric value.
func TestTransform_IdentityPreservesSignedZero(t *testing.T) {
	values := []float64{math.Copysign(0, -1), math.SmallestNonzeroFloat64, -math.MaxFloat64}
	got, err := paa.Transform(values, len(values))
	if err != nil {
		t.Fatal(err)
	}
	for j, v := range values {
		if math.Float64bits(v) != math.Float64bits(got.Values[j]) {
			t.Fatalf("detached identity bit mismatch j=%d: %x != %x", j, math.Float64bits(v), math.Float64bits(got.Values[j]))
		}
	}
	got.Values[0] = 3
	if math.Float64bits(values[0]) != math.Float64bits(math.Copysign(0, -1)) {
		t.Fatal("identity output aliases input")
	}
}

// TestTransformRange_DetachedOwnership verifies that the checked window facade
// does not capture the full source, a view of its values, or shared output state.
func TestTransformRange_DetachedOwnership(t *testing.T) {
	data := []float64{77, 0, 2, 4, 6, 8, 88}
	original := append([]float64(nil), data...)
	first, err := paa.TransformRange(data, 1, 6, 2)
	if err != nil {
		t.Fatal(err)
	}
	first.Values[0] = 909
	first.Segments[0].RightNumerator = 909

	// Mutating the output must not change any source observation.
	for j := range data {
		if math.Float64bits(data[j]) != math.Float64bits(original[j]) {
			t.Fatalf("source mutated at sample %d", j)
		}
	}

	// A second call must have the canonical numeric and geometric result,
	// independent of an earlier caller-owned result's mutation.
	second, err := paa.TransformRange(data, 1, 6, 2)
	if err != nil {
		t.Fatal(err)
	}
	assertExactSegments(t, second, 5, 2)
	if second.Values[0] != 1.6 || second.Values[1] != 6.4 {
		t.Fatalf("later range call corrupted: %v", second.Values)
	}

	// Finally, changing source storage after return must not alter either
	// already published output slice.
	data[1] = 999
	if second.Values[0] != 1.6 || second.Segments[0].RightNumerator != 5 {
		t.Fatal("range result retains an input alias")
	}
}
