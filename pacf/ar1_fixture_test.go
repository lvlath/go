// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pacf_test tests finite-sample estimation on a seeded AR(1) process.
// Unlike a symbolic Yule-Walker identity, this fixture exercises pacf.Compute.
package pacf_test

import (
	"context"
	"math"
	"math/rand"
	"testing"

	"github.com/lvlath/go/pacf"
)

// TestPACF05_AR1FiniteSampleCutoff verifies convergence toward the theoretical
// AR(1) PACF shape without demanding exact zeros from a finite sample.
// The fixed seed makes this a deterministic statistical regression fixture.
func TestPACF05_AR1FiniteSampleCutoff(t *testing.T) {
	const n = 16384
	const coefficient = 0.55
	const maxLag = 10
	const firstLagTolerance = 0.035
	const residualLagTolerance = 0.035

	rng := rand.New(rand.NewSource(2705))
	process := make([]float64, n)
	for i := 1; i < n; i++ {
		process[i] = coefficient*process[i-1] + rng.NormFloat64()
	}

	got, err := pacf.ComputeCentered(context.Background(), process, maxLag)
	if err != nil {
		t.Fatalf("AR(1) PACF failed: %v", err)
	}
	if math.Abs(got.Values[1]-coefficient) > firstLagTolerance {
		t.Fatalf("lag-1 PACF %.10f differs from AR coefficient %.2f", got.Values[1], coefficient)
	}
	for lag := 2; lag <= maxLag; lag++ {
		if math.Abs(got.Values[lag]) > residualLagTolerance {
			t.Fatalf("lag %d: PACF %.10f exceeds fixed-seed sampling tolerance", lag, got.Values[lag])
		}
	}
	if got.PredictionVariance[1] >= got.PredictionVariance[0] {
		t.Fatal("AR(1) first predictor order should reduce in-window error variance")
	}
}

// TestPACF05_AR1TheoryIsNotExactSample asserts that the estimated higher-lag
// reflection coefficient is not spuriously overwritten with a theoretical zero.
// The same output must agree with an independent Toeplitz numerical oracle.
func TestPACF05_AR1TheoryIsNotExactSample(t *testing.T) {
	const n = 32
	rng := rand.New(rand.NewSource(2715))
	x := make([]float64, n)
	for i := 1; i < n; i++ {
		x[i] = 0.5*x[i-1] + rng.NormFloat64()
	}
	got, err := pacf.ComputeCentered(context.Background(), x, 6)
	if err != nil {
		t.Fatal(err)
	}
	if got.Values[2] == 0 && got.Values[3] == 0 {
		t.Fatal("finite-sample PACF must not be coerced to a theoretical AR(1) cutoff")
	}
	gamma := rationalAutocovFloat64(x, true)
	for lag := range got.Values {
		wantCoefficient, wantVariance := rationalSolve(gamma, lag)
		if wantCoefficient == nil {
			t.Fatalf("unexpected exact Toeplitz singularity at lag %d", lag)
		}
		k, _ := wantCoefficient.Float64()
		v, _ := wantVariance.Float64()
		if math.IsNaN(got.Values[lag]) || math.IsInf(got.Values[lag], 0) ||
			!closeEnough(got.Values[lag], k, 3e-11) ||
			!closeEnough(got.PredictionVariance[lag], v, 3e-11) {
			t.Fatalf("lag %d: got k=%.17g v=%.17g, oracle k=%.17g v=%.17g",
				lag, got.Values[lag], got.PredictionVariance[lag], k, v)
		}
	}
}
