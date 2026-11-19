// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gjrgarch_test checks exported analytic QMLE derivatives against an
// independent central-difference oracle; no production internals are accessed.
package gjrgarch_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/gjrgarch"
)

// gradientFixture represents stationary, stylized index daily returns of both signs.
// Variation in magnitude and sign exercises the leverage indicator and derivative chain.
func gradientFixture() []float64 {
	returns := make([]float64, 72)
	for i := range returns {
		returns[i] = 0.43*math.Sin(0.47*float64(i)+0.3) +
			0.17*math.Cos(0.19*float64(i)+0.8) - 0.04
	}
	return returns
}

// setGradientCoordinate changes one model coefficient without changing the others.
// It is confined to the independent finite-difference oracle in the external tests.
func setGradientCoordinate(p gjrgarch.Parameters, coordinate int, value float64) gjrgarch.Parameters {
	switch coordinate {
	case 0:
		p.Mean = value
	case 1:
		p.Omega = value
	case 2:
		p.Alpha = value
	case 3:
		p.Gamma = value
	case 4:
		p.Beta = value
	}
	return p
}

// gradientCoordinates returns the caller-visible parameter ordering of QMLEEvaluation.Gradient.
func gradientCoordinates(p gjrgarch.Parameters) [5]float64 {
	return [5]float64{p.Mean, p.Omega, p.Alpha, p.Gamma, p.Beta}
}

// TestGJR_GradientFiniteDifference checks every analytic derivative in THREE
// distinct interior regimes, including negative Gamma and the h[0] dependency.
func TestGJR_GradientFiniteDifference(t *testing.T) {
	cases := []struct {
		name       string
		returns    []float64
		parameters gjrgarch.Parameters
	}{
		{
			name:    "positive leverage with 72 observations",
			returns: gradientFixture(),
			parameters: gjrgarch.Parameters{
				Mean: -0.03, Omega: 0.12, Alpha: 0.11, Gamma: 0.16, Beta: 0.48,
			},
		},
		{
			name:    "negative leverage with nonnegative downside coefficient",
			returns: gradientFixture(),
			parameters: gjrgarch.Parameters{
				Mean: 0.04, Omega: 0.08, Alpha: 0.22, Gamma: -0.09, Beta: 0.38,
			},
		},
		{
			name:    "single observation isolates stationary h0 derivative",
			returns: []float64{0.27},
			parameters: gjrgarch.Parameters{
				Mean: -0.03, Omega: 0.19, Alpha: 0.09, Gamma: 0.07, Beta: 0.35,
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			base, err := gjrgarch.EvaluateGaussianQMLE(ctx, tc.returns, tc.parameters, 0)
			if err != nil {
				t.Fatal(err)
			}
			coordinates := gradientCoordinates(tc.parameters)
			analytic := gradientCoordinates(base.Gradient)
			for j, original := range coordinates {
				// h is relative to the tested coordinate and stays inside the feasible interior.
				h := 1e-6 * math.Max(1, math.Abs(original))
				left := setGradientCoordinate(tc.parameters, j, original-h)
				right := setGradientCoordinate(tc.parameters, j, original+h)
				minus, leftErr := gjrgarch.EvaluateGaussianQMLE(ctx, tc.returns, left, 0)
				plus, rightErr := gjrgarch.EvaluateGaussianQMLE(ctx, tc.returns, right, 0)
				if leftErr != nil || rightErr != nil {
					t.Fatalf("coordinate %d: left=%v right=%v", j, leftErr, rightErr)
				}
				finiteDifference := (plus.Objective - minus.Objective) / (2 * h)
				scale := math.Max(1, math.Max(math.Abs(finiteDifference), math.Abs(analytic[j])))
				if got := math.Abs(analytic[j]-finiteDifference) / scale; got > 1e-5 {
					t.Fatalf("coordinate %d: analytic=%.12g finiteDifference=%.12g relative=%.6g", j, analytic[j], finiteDifference, got)
				}
			}
			if base.Observations != len(tc.returns) {
				t.Fatal("evaluation lost sample count")
			}
		})
	}
}

// TestGJR_GradientOneSidedAtBoundary checks feasible directional derivatives on
// the active Alpha=0 face and the distinct Alpha+Gamma=0 face.
func TestGJR_GradientOneSidedAtBoundary(t *testing.T) {
	returns := gradientFixture()
	cases := []struct {
		name       string
		params     gjrgarch.Parameters
		coordinate int
	}{
		{"alpha zero", gjrgarch.Parameters{Omega: 0.15, Alpha: 0, Gamma: 0.11, Beta: 0.4}, 2},
		{"negative-shock coefficient zero", gjrgarch.Parameters{Omega: 0.15, Alpha: 0.18, Gamma: -0.18, Beta: 0.4}, 3},
	}
	const h = 1e-6
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			baseline, err := gjrgarch.EvaluateGaussianQMLE(context.Background(), returns, tc.params, 0)
			if err != nil {
				t.Fatal(err)
			}
			coords := gradientCoordinates(tc.params)
			stepped := setGradientCoordinate(tc.params, tc.coordinate, coords[tc.coordinate]+h)
			forward, err := gjrgarch.EvaluateGaussianQMLE(context.Background(), returns, stepped, 0)
			if err != nil {
				t.Fatal(err)
			}
			directional := (forward.Objective - baseline.Objective) / h
			analytic := gradientCoordinates(baseline.Gradient)[tc.coordinate]
			if diff := math.Abs(directional-analytic) / math.Max(1, math.Abs(analytic)); diff > 1e-5 {
				t.Fatalf("boundary gradient: analytic=%g one-sided=%g discrepancy=%g", analytic, directional, diff)
			}
		})
	}
}
