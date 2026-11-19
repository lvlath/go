// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gjrgarch_test fuzzes the exported posterior-state boundary.
// Arbitrary binary64 observations must never bypass validation or break causality.
package gjrgarch_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/gjrgarch"
)

// FuzzGJR_StepCausalState verifies the previous-shock variance invariant under
// arbitrary binary64 observations, including NaN, infinities, and extreme values.
// A valid new observation can change only the new residual, not the next variance.
//
// Implementation:
//   - Stage 1: create one immutable stationary model and its fixed posterior.
//   - Stage 2: compare the known-shock forecast with Step for arbitrary new returns.
//   - Stage 3: classify invalid observations or verify a fully detached valid state.
//
// Determinism: Independent inputs are mapped from their original IEEE-754 bits.
// Complexity: O(1) time and memory per fuzz case; no mutable package globals.
// AI-Hints: A Step that uses its incoming observation in h[t+1] will fail this test.
func FuzzGJR_StepCausalState(f *testing.F) {
	for _, bits := range []uint64{
		0, 0x8000000000000000, 0x3f847ae147ae147b,
		0xbff0000000000000, 0x7fefffffffffffff,
		0xffefffffffffffff, 0x7ff0000000000000,
		0xfff0000000000000, 0x7ff8000000000001,
	} {
		f.Add(bits)
	}
	f.Fuzz(func(t *testing.T, bits uint64) {
		model, err := gjrgarch.New(gjrgarch.Parameters{
			Mean: 0.0003, Omega: 0.00002, Alpha: 0.09, Gamma: 0.15, Beta: 0.7,
		})
		if err != nil {
			t.Fatal(err)
		}
		previous, err := model.InitialState(-0.014, 0.00034)
		if err != nil {
			t.Fatal(err)
		}
		forecast, err := model.ForecastNextVariance(previous)
		if err != nil {
			t.Fatal(err)
		}
		observed := math.Float64frombits(bits)
		original := previous
		next, stepErr := model.Step(previous, observed)
		if previous != original {
			t.Fatal("Step mutated caller-owned previous state")
		}
		if math.IsNaN(observed) || math.IsInf(observed, 0) {
			if !errors.Is(stepErr, gjrgarch.ErrNonFinite) || next.Steps != 0 {
				t.Fatalf("nonfinite observation must be rejected atomically: %v", stepErr)
			}
			return
		}
		if stepErr != nil {
			if !errors.Is(stepErr, gjrgarch.ErrOverflow) || next.Steps != 0 {
				t.Fatalf("unexpected finite-observation failure: %v", stepErr)
			}
			return
		}
		if next.LastVariance != forecast || next.Steps != 2 || next.ModelID != model.ID() ||
			next.LastResidual != observed-0.0003 {
			t.Fatalf("look-ahead variance or posterior drift: %+v expected variance %.17g", next, forecast)
		}
	})
}
