// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package montecarlo_test fuzzes the public sample-validation and moments boundary.
// Every generated input is evaluated through montecarlo.Run, without reflection.
package montecarlo_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/montecarlo"
	"github.com/lvlath/go/rng"
)

// FuzzRunScalarDomain checks finite versus nonfinite sample classification.
// Constant finite vectors must never overflow their own mean or variance.
func FuzzRunScalarDomain(f *testing.F) {
	for _, bits := range []uint64{
		0, 1, 0x8000000000000000, math.Float64bits(math.MaxFloat64),
		math.Float64bits(-math.MaxFloat64), math.Float64bits(math.NaN()),
		math.Float64bits(math.Inf(1)), math.Float64bits(math.Inf(-1)),
	} {
		f.Add(bits)
	}
	f.Fuzz(func(t *testing.T, bits uint64) {
		value := math.Float64frombits(bits)
		sampler := &funcSampler{1, func(_ context.Context, _ uint64, _ rng.RNG) ([]float64, error) {
			return []float64{value}, nil
		}}
		result, err := montecarlo.Run(context.Background(), montecarlo.Plan{Paths: 2}, sampler)
		if math.IsInf(value, 0) || math.IsNaN(value) {
			mustErrorIs(t, err, montecarlo.ErrNonFinite)
			if result == nil || result.Completed != 0 || result.Mean != nil || result.Complete {
				t.Fatal("nonfinite scalar escaped into a successful prefix")
			}
			return
		}
		if err != nil {
			t.Fatalf("constant finite input unexpectedly failed: %v", err)
		}
		if result == nil || !result.Complete || result.Completed != 2 ||
			result.SampleVariance[0] != 0 || result.MeanStandardError[0] != 0 {
			t.Fatal("constant finite input did not preserve exact zero variance")
		}
	})
}
