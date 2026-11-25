// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gjrgarch_test checks the geometry of the first projected BFGS step.
// Analytic single-observation cases use known face normals rather than reusing
// the production active-set enumerator or matrix solver as a reference.
package gjrgarch_test

import (
	"context"
	"errors"
	"testing"

	"github.com/lvlath/go/gjrgarch"
)

// TestGJR_FirstProjectedStepAnalyticFaces verifies the actual nearest-face solution.
// With one observation and scale=1, the loss has closed-form derivatives. Each
// selected fixture violates one known face (or two orthogonal faces), so the
// Euclidean projector has a simple independent analytic expression.
//
// Implementation:
//   - Stage 1: Derive the gradient of the stationary one-observation density.
//   - Stage 2: Project against the known active normals, without calling matrix.
//   - Stage 3: Check the accepted full/half Armijo step and its original-unit loss.
//
// Complexity: O(1) oracle arithmetic and one bounded Fit update per fixture.
// AI-Hints: Checking only feasible output is insufficient; a componentwise clip
// can be feasible yet not be the nearest Euclidean point. The negative-shock
// face must move both Alpha and Gamma when the unconstrained sum is negative.
func TestGJR_FirstProjectedStepAnalyticFaces(t *testing.T) {
	const (
		observation      = 1.0
		floor            = 0.01
		persistenceLimit = 1 - 64*0x1p-52
	)
	cases := []struct {
		name    string
		initial gjrgarch.Parameters
		// The fixture declares its analytic active faces and expected accepted step.
		alphaZero, betaZero, shockZero, stationaryCap bool
		step                                          float64
	}{
		{"alpha_face", gjrgarch.Parameters{Mean: 0.125, Omega: 0.5, Alpha: 0.03125, Gamma: 0.25, Beta: 0.25}, true, false, false, false, 1},
		{"beta_face", gjrgarch.Parameters{Mean: 0.125, Omega: 0.5, Alpha: 0.25, Gamma: 0.25, Beta: 0.03125}, false, true, false, false, 1},
		{"negative_shock_face", gjrgarch.Parameters{Mean: 0.125, Omega: 0.5, Alpha: 0.25, Gamma: -0.1875, Beta: 0.25}, false, false, true, false, 1},
		{"orthogonal_faces", gjrgarch.Parameters{Mean: 0.25, Omega: 0.5, Alpha: 0.25, Gamma: -0.1875, Beta: 0.03125}, false, true, true, false, 1},
		{"stationarity_cap_backtracking", gjrgarch.Parameters{Omega: 0.375, Alpha: 0.125, Gamma: 0.125, Beta: 0.125}, false, false, false, true, 0.5},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := tc.initial
			// Differentiate the scalar closed form h0=Omega/(1-rho) directly.
			rho := p.Alpha + p.Beta + p.Gamma/2
			h := p.Omega / (1 - rho)
			residual := observation - p.Mean
			z2 := residual * residual / h
			k := (1 - z2) / (2 * (1 - rho))
			trial := gjrgarch.Parameters{
				Mean:  p.Mean + residual/h,
				Omega: p.Omega - (1-z2)/(2*p.Omega),
				Alpha: p.Alpha - k, Gamma: p.Gamma - k/2, Beta: p.Beta - k,
			}
			// The normal of Alpha=0 is the Alpha axis.
			if tc.alphaZero {
				if trial.Alpha >= 0 {
					t.Fatal("fixture does not cross Alpha=0")
				}
				trial.Alpha = 0
			}
			// Beta=0 is orthogonal to the negative-shock face.
			if tc.betaZero {
				if trial.Beta >= 0 {
					t.Fatal("fixture does not cross Beta=0")
				}
				trial.Beta = 0
			}
			if tc.shockZero {
				// The normal (1,1,0) has squared norm 2.
				sum := trial.Alpha + trial.Gamma
				if sum >= 0 {
					t.Fatal("fixture does not cross the negative-shock face")
				}
				trial.Alpha -= sum / 2
				trial.Gamma = -trial.Alpha
			}
			if tc.stationaryCap {
				// The persistence normal (1,1/2,1) has squared norm 9/4.
				excess := trial.Alpha + trial.Beta + trial.Gamma/2 - persistenceLimit
				if excess <= 0 {
					t.Fatal("fixture does not cross the stationarity cap")
				}
				distance := excess / (9.0 / 4.0)
				trial.Alpha -= distance
				trial.Gamma -= distance / 2
				trial.Beta -= distance
			}
			// Interpolate the analytically known accepted step, not the final
			// optimizer output. The capped case needs exactly one halving.
			want := gjrgarch.Parameters{
				Mean:  p.Mean + tc.step*(trial.Mean-p.Mean),
				Omega: p.Omega + tc.step*(trial.Omega-p.Omega),
				Alpha: p.Alpha + tc.step*(trial.Alpha-p.Alpha),
				Gamma: p.Gamma + tc.step*(trial.Gamma-p.Gamma),
				Beta:  p.Beta + tc.step*(trial.Beta-p.Beta),
			}
			result, err := gjrgarch.Fit(context.Background(), []float64{observation}, gjrgarch.Options{
				Initial: p, VarianceFloor: floor, MaxIterations: 1, Tolerance: 1e-12,
			})
			if err != nil && !errors.Is(err, gjrgarch.ErrNotConverged) {
				t.Fatal(err)
			}
			if result == nil || result.Model == nil {
				t.Fatal("projected update was not published")
			}
			mustEqualInt(t, result.Diagnostics.Iterations, 1, "accepted projected step")
			mustEqualInt(t, len(result.Diagnostics.Objective), 2, "accepted history")
			got := result.Model.Parameters()
			// Compare all five coordinates; feasibility alone is not an oracle.
			mustNearRelative(t, got.Mean, want.Mean, 2e-12, "Mean")
			mustNearRelative(t, got.Omega, want.Omega, 2e-12, "Omega")
			mustNearRelative(t, got.Alpha, want.Alpha, 2e-12, "Alpha")
			mustNearRelative(t, got.Gamma, want.Gamma, 2e-12, "Gamma")
			mustNearRelative(t, got.Beta, want.Beta, 2e-12, "Beta")
			// Exact active faces are part of the feasible model domain.
			if tc.alphaZero && got.Alpha != 0 {
				t.Fatal("Alpha face drift")
			}
			if tc.betaZero && got.Beta != 0 {
				t.Fatal("Beta face drift")
			}
			if tc.shockZero && got.Alpha+got.Gamma != 0 {
				t.Fatal("negative-shock face drift")
			}
			if got.Omega < floor || got.Alpha < 0 || got.Beta < 0 || got.Alpha+got.Gamma < 0 || got.Beta+got.Alpha/2+(got.Alpha+got.Gamma)/2 > persistenceLimit {
				t.Fatal("infeasible projected candidate")
			}
			before, after := referenceSingletonLoss(p, observation), referenceSingletonLoss(want, observation)
			if !(after < before) {
				t.Fatal("oracle step must improve the objective")
			}
			mustNearRelative(t, result.Diagnostics.Objective[0], before, 2e-12, "initial loss")
			mustNearRelative(t, result.Diagnostics.Objective[1], after, 2e-12, "accepted loss")
			mustNearRelative(t, result.LogLikelihood, -after, 2e-12, "singleton total likelihood")
		})
	}
}
