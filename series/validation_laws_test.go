// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// External validation tests apply common input gates across every unary facade.
// They prove the same nil/empty/finite law without any internal test hooks.
package series_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/series"
)

// TestTransforms_SharedInputValidation checks all unary entry-point error laws.
// Every transform must reject nil, allocated-empty, and nonfinite observations
// before a valid result can be published, regardless of warmup length.
func TestTransforms_SharedInputValidation(t *testing.T) {
	facades := []struct {
		name string
		call func([]float64) (*series.TransformResult, error)
	}{
		{name: "Lag", call: func(v []float64) (*series.TransformResult, error) { return series.Lag(v, 100) }},
		{name: "Difference", call: func(v []float64) (*series.TransformResult, error) { return series.Difference(v, 100) }},
		{name: "SimpleReturns", call: series.SimpleReturns},
		{name: "LogReturns", call: series.LogReturns},
	}

	for _, facade := range facades {
		t.Run(facade.name, func(t *testing.T) {
			for _, tc := range []struct {
				name   string
				values []float64
				want   error
			}{
				{name: "nil", values: nil, want: series.ErrNilInput},
				{name: "empty", values: []float64{}, want: series.ErrEmptyInput},
				{name: "nan", values: []float64{math.NaN()}, want: series.ErrNonFinite},
				{name: "positive infinity", values: []float64{math.Inf(1)}, want: series.ErrNonFinite},
				{name: "negative infinity", values: []float64{math.Inf(-1)}, want: series.ErrNonFinite},
			} {
				t.Run(tc.name, func(t *testing.T) {
					got, err := facade.call(tc.values)
					mustErrorIs(t, err, tc.want)
					if got != nil {
						t.Fatalf("failed input published a result %+v", got)
					}
				})
			}
		})
	}
}
