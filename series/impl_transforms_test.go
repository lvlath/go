// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// External transform tests validate finite arithmetic, masks, and causality.
// They use public facades without accessing private kernels or state.

package series_test

import (
	"errors"
	"math"
	"math/big"
	"slices"
	"testing"

	"github.com/lvlath/go/series"
)

// TestTransforms_Fixtures checks exact lag, difference, and simple-return values.
// It also verifies all-invalid masks when the lag exceeds the input length.
func TestTransforms_Fixtures(t *testing.T) {
	lag, err := series.Lag([]float64{2, 4, 8}, 1)
	if err != nil {
		t.Fatal(err)
	}
	assertTransform(t, lag, []float64{0, 2, 4}, []bool{false, true, true})
	diff, err := series.Difference([]float64{2, 4, 8}, 2)
	if err != nil {
		t.Fatal(err)
	}
	assertTransform(t, diff, []float64{0, 0, 6}, []bool{false, false, true})
	ret, err := series.SimpleReturns([]float64{2, 4, 0})
	if err != nil {
		t.Fatal(err)
	}
	assertTransform(t, ret, []float64{0, 1, -1}, []bool{false, true, true})
	for _, fn := range []func([]float64, int) (*series.TransformResult, error){series.Lag, series.Difference} {
		got, err := fn([]float64{1, 2}, 10)
		if err != nil {
			t.Fatal(err)
		}
		assertTransform(t, got, []float64{0, 0}, []bool{false, false})
	}
}

// TestTransforms_PrefixInvariant checks causality under future observations.
// Both numeric values and masks of the existing prefix must remain unchanged.
func TestTransforms_PrefixInvariant(t *testing.T) {
	for _, tc := range []struct {
		name string
		fn   func([]float64) (*series.TransformResult, error)
	}{
		{"lag", func(x []float64) (*series.TransformResult, error) { return series.Lag(x, 2) }},
		{"difference", func(x []float64) (*series.TransformResult, error) { return series.Difference(x, 2) }},
		{"simple", series.SimpleReturns},
		{"log", series.LogReturns},
	} {
		t.Run(tc.name, func(t *testing.T) {
			prefix := []float64{2, 4, 8, 3, 6}
			before, err := tc.fn(prefix)
			if err != nil {
				t.Fatal(err)
			}
			after, err := tc.fn(append(append([]float64(nil), prefix...), 12, 24))
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(before.Values, after.Values[:len(prefix)]) || !slices.Equal(before.Valid, after.Valid[:len(prefix)]) {
				t.Fatalf("future changed prefix: before=%+v after=%+v", before, after)
			}
		})
	}
}

// TestLogReturns_ExtremePositiveValues checks finite logarithmic return magnitudes.
// The counterexample proves direct ratio formation would overflow float64.
func TestLogReturns_ExtremePositiveValues(t *testing.T) {
	result, err := series.LogReturns([]float64{1e-300, 1e300})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Valid[1] || math.Abs(result.Values[1]-600*math.Ln10) > 1e-12 {
		t.Fatal(result)
	}
	// The independently evaluated ratio 1e300/1e-300 is not representable.
	a, b := 1e300, 1e-300
	if !math.IsInf(a/b, 1) {
		t.Fatal("counterexample invalid")
	}
}

// TestTransformResult_Detached proves input/output memory isolation.
// Returned values and masks are caller-owned and must not alias source data.
func TestTransformResult_Detached(t *testing.T) {
	for _, tc := range []struct {
		name string
		fn   func([]float64) (*series.TransformResult, error)
	}{
		{"identity", func(v []float64) (*series.TransformResult, error) { return series.Lag(v, 0) }},
		{"lag", func(v []float64) (*series.TransformResult, error) { return series.Lag(v, 1) }},
		{"difference", func(v []float64) (*series.TransformResult, error) { return series.Difference(v, 1) }},
		{"simple", series.SimpleReturns}, {"log", series.LogReturns},
	} {
		t.Run(tc.name, func(t *testing.T) {
			input := []float64{2, 3, 5}
			got, err := tc.fn(input)
			if err != nil {
				t.Fatal(err)
			}
			valuesCopy := append([]float64(nil), got.Values...)
			input[0] = 100
			if !slices.Equal(got.Values, valuesCopy) {
				t.Fatal("result aliases input")
			}
			got.Values[0], got.Valid[0] = 300, true
			if input[0] != 100 {
				t.Fatal("output aliases input")
			}
		})
	}
}

// TestTransforms_InvalidAndOverflow protects options, domain, and numeric errors.
// Every failed call must return nil and preserve its package-owned sentinel.
func TestTransforms_InvalidAndOverflow(t *testing.T) {
	checks := []struct {
		name string
		fn   func() (*series.TransformResult, error)
		want error
	}{
		{"lag option before nil", func() (*series.TransformResult, error) { return series.Lag(nil, -1) }, series.ErrInvalidOption},
		{"difference zero lag", func() (*series.TransformResult, error) { return series.Difference(nil, 0) }, series.ErrInvalidOption},
		{"nil", func() (*series.TransformResult, error) { return series.Lag(nil, 1) }, series.ErrNilInput},
		{"empty", func() (*series.TransformResult, error) { return series.Lag([]float64{}, 1) }, series.ErrEmptyInput},
		{"warmup nonfinite", func() (*series.TransformResult, error) { return series.Lag([]float64{math.NaN()}, 3) }, series.ErrNonFinite},
		{"difference overflow", func() (*series.TransformResult, error) {
			return series.Difference([]float64{-math.MaxFloat64, math.MaxFloat64}, 1)
		}, series.ErrOverflow},
		{"simple zero", func() (*series.TransformResult, error) { return series.SimpleReturns([]float64{2, 0, 4}) }, series.ErrDomain},
		{"simple overflow", func() (*series.TransformResult, error) {
			return series.SimpleReturns([]float64{math.SmallestNonzeroFloat64, math.MaxFloat64})
		}, series.ErrOverflow},
		{"log singleton zero", func() (*series.TransformResult, error) { return series.LogReturns([]float64{0}) }, series.ErrDomain},
		{"log negative", func() (*series.TransformResult, error) { return series.LogReturns([]float64{2, -1}) }, series.ErrDomain},
		{"log nonfinite before domain", func() (*series.TransformResult, error) { return series.LogReturns([]float64{-2, math.Inf(1)}) }, series.ErrNonFinite},
	}
	for _, tc := range checks {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tc.fn()
			if got != nil || !errors.Is(err, tc.want) {
				t.Fatalf("got=%v err=%v want=%v", got, err, tc.want)
			}
		})
	}
}

// TestDifference_IndependentRationalOracle uses exact rational float inputs.
// The independent oracle checks subtraction results within scale-aware tolerance.
func TestDifference_IndependentRationalOracle(t *testing.T) {
	inputs := [][]float64{{0.1, 0.3, 0.8, 1.2}, {math.MaxFloat64, math.MaxFloat64 / 2, 0}, {-5, 7, -11}}
	for _, input := range inputs {
		got, err := series.Difference(input, 1)
		if err != nil {
			t.Fatal(err)
		}
		for i := 1; i < len(input); i++ {
			a := new(big.Rat).SetFloat64(input[i])
			b := new(big.Rat).SetFloat64(input[i-1])
			a.Sub(a, b)
			oracle, _ := a.Float64()
			if math.Abs(got.Values[i]-oracle) > math.Max(1e-15, math.Abs(oracle)*2e-15) {
				t.Fatalf("i=%d got=%g oracle=%g", i, got.Values[i], oracle)
			}
		}
	}
}

// TestTransforms_Singletons anchors the one-element warmup contract.
// A positive singleton is valid input despite having no adjacent return pair.
func TestTransforms_Singletons(t *testing.T) {
	for _, fn := range []func([]float64) (*series.TransformResult, error){series.SimpleReturns, series.LogReturns} {
		got, err := fn([]float64{2})
		if err != nil {
			t.Fatal(err)
		}
		assertTransform(t, got, []float64{0}, []bool{false})
	}
}
