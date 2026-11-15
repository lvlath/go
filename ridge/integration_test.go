// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ridge_test verifies that changes to the matrix QR right-hand-side
// arithmetic preserve the regression objective and its physical-unit outputs.
package ridge_test

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/lvlath/go/ridge"
)

// ridgeScaledClose compares quantities in the same physical units without
// subtracting potentially enormous or subnormal unscaled operands.
//
// Implementation:
//   - Stage 1: require both the observed and expected quantities to be finite.
//   - Stage 2: divide by the strictly positive, finite reference magnitude.
//   - Stage 3: compare the dimensionless relative discrepancy against tolerance.
//
// Behavior highlights:
//   - The oracle explicitly rejects an incorrectly rounded zero result.
//
// Inputs:
//   - t: owning test; name: diagnostic field; got, want: finite nonzero values.
//   - tolerance: permissible dimensionless relative discrepancy.
//
// Returns:
//   - None; reports an assertion failure to the owning testing goroutine.
//
// Errors:
//   - Calls t.Errorf on invalid or insufficiently accurate output.
//
// Determinism:
//   - Pure comparison in a fixed call order, without map or goroutine iteration.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Use only when the reference value is nonzero and ratios are representable.
//
// AI-Hints:
//   - A raw absolute tolerance would incorrectly accept a lost subnormal slope.
//   - Do not replace the independent analytic reference with the fitted result.
func ridgeScaledClose(t *testing.T, name string, got, want, tolerance float64) {
	t.Helper()
	if math.IsNaN(got) || math.IsInf(got, 0) || math.IsNaN(want) || math.IsInf(want, 0) || want == 0 {
		t.Errorf("%s: invalid numeric comparison: got=%g want=%g", name, got, want)
		return
	}
	if discrepancy := math.Abs(got/want - 1); discrepancy > tolerance {
		t.Errorf("%s: got=%.17g want=%.17g (relative discrepancy %.3g > %.3g)",
			name, got, want, discrepancy, tolerance)
	}
}

// TestRidge_NUM01AffineResponseScaleOracle protects the complete normalized
// objective against a change to the dependent QR factor's RHS scaling policy.
//
// For x=[-1,+1], y=s*[2,4], Lambda=1, the exact optimum is
// beta=s/2, intercept=3s, SSE=s^2/2, Penalty=s^2/8, Objective=s^2/4.
// No solver factorization or fitted coefficient appears in this oracle.
//
// Implementation:
//   - Stage 1: construct one fixed physical design with two observations.
//   - Stage 2: solve three otherwise identical tasks at binary response scales.
//   - Stage 3: compare every public scalar to its exact closed-form value.
//
// Behavior highlights:
//   - The intercept stays free and therefore scales by s, not by a penalty.
//   - Extremely small and large RHS values exercise upstream QR scaling without
//     relying on cancellation of unrepresentably large objective diagnostics.
//
// Inputs:
//   - Mathematically fixed n=2, p=1, Lambda=1; binary s=2^(-400,0,400).
//
// Returns:
//   - No value; successful fits must satisfy the independent scalar equations.
//
// Errors:
//   - Reports unexpected Fit failures or inaccurate coefficients/diagnostics.
//
// Determinism:
//   - Subcases and design cells are visited in fixed ascending order.
//
// Complexity:
//   - Three O((n+p)*(p+1)^2) QR fits plus O(1) assertion work per fit.
//
// Notes:
//   - A factor backend may have a genuine numerical limitation outside these
//     representable, well-conditioned examples; this test does not relax errors.
//
// AI-Hints:
//   - RHS normalization is an internal solver choice, not permission to rescale
//     Lambda or to compute Penalty from QR's equilibrated gamma coefficients.
//   - A very small absolute error must not hide a coefficient rounded to zero.
func TestRidge_NUM01AffineResponseScaleOracle(t *testing.T) {
	const lambda = 1.0
	const tolerance = 1e-9
	const baseExponent = 0
	input := dense(t, [][]float64{{-1}, {1}})

	for _, exponent := range []int{-400, baseExponent, 400} {
		t.Run(fmt.Sprintf("scale_2^%d", exponent), func(t *testing.T) {
			scale := math.Ldexp(1, exponent)
			scaleSquared := math.Ldexp(1, 2*exponent)
			fit, err := ridge.FitWithLambda(context.Background(), input,
				[]float64{2 * scale, 4 * scale}, lambda)
			if err != nil {
				t.Fatalf("affine fit failed at binary exponent %d: %v", exponent, err)
			}
			if fit == nil || fit.Model == nil || fit.IsNil() {
				t.Fatal("successful fit must publish one present model and result")
			}
			beta, intercept := fit.Model.Coefficients()
			if len(beta) != 1 {
				t.Fatalf("coefficient width=%d want 1", len(beta))
			}
			ridgeScaledClose(t, "beta", beta[0], scale/2, tolerance)
			ridgeScaledClose(t, "unpenalized intercept", intercept, 3*scale, tolerance)
			ridgeScaledClose(t, "SSE", fit.ResidualSumSquares, scaleSquared/2, tolerance)
			ridgeScaledClose(t, "Penalty", fit.Penalty, scaleSquared/8, tolerance)
			ridgeScaledClose(t, "Objective", fit.Objective, scaleSquared/4, tolerance)
		})
	}
}

// TestRidge_NUM01WideStandardizedScaleEquivariance verifies that response-unit
// conversion changes fitted coefficients linearly and both objective components
// quadratically, while the feature scaling and penalty basis remain frozen.
//
// Implementation:
//   - Stage 1: fit a duplicated-column, p>n design in standardized coordinates.
//   - Stage 2: multiply the same responses by 2^(-180) and 2^(180) independently.
//   - Stage 3: compare the original-unit model and diagnostics after undoing only
//     the response-unit conversion, never re-standardizing the source features.
//
// Behavior highlights:
//   - The exact homogeneity J(s*y,s*theta,s*b)=s^2*J(y,theta,b) holds for s>0.
//   - The penalty on optimization slopes must obey the same scaling law as SSE.
//   - Duplicated features and p>n remain valid with strictly positive Lambda.
//
// Inputs:
//   - Fixed three-sample, five-feature design, Lambda=0.75 and positive dyadic s.
//
// Returns:
//   - No value; public coefficients and diagnostics remain scale-equivalent.
//
// Errors:
//   - Reports Fit errors, shape mismatches, or violated homogeneity.
//
// Determinism:
//   - Exponents ascend; all coefficient indices are checked in original order.
//
// Complexity:
//   - Three O((n+p)*(p+1)^2) QR fits and O(p) verification.
//
// Notes:
//   - Homogeneity is an exact mathematical invariant; the finite tolerance
//     admits ordinary QR rounding, not a different estimator or objective.
//
// AI-Hints:
//   - Do not change Lambda when converting the response units alone; its
//     dimensionful coefficient must retain the optimization convention.
//   - Do not compare physical beta against standardized theta for Penalty.
func TestRidge_NUM01WideStandardizedScaleEquivariance(t *testing.T) {
	const lambda = 0.75
	const tolerance = 1e-8
	input := dense(t, [][]float64{
		{1, 2, 4, 1, 4},
		{2, 3, 5, 2, 5},
		{4, 6, 7, 4, 7},
	})
	responses := []float64{1, 3, 2}
	base, err := ridge.FitStandardized(context.Background(), input, responses, lambda)
	if err != nil {
		t.Fatalf("reference wide fit: %v", err)
	}
	baseBeta, baseIntercept := base.Model.Coefficients()
	if base.Penalty <= 0 || base.ResidualSumSquares <= 0 || base.Objective <= 0 {
		t.Fatalf("fixture needs positive diagnostics: SSE=%g penalty=%g objective=%g",
			base.ResidualSumSquares, base.Penalty, base.Objective)
	}

	for _, exponent := range []int{-180, 180} {
		t.Run(fmt.Sprintf("scale_2^%d", exponent), func(t *testing.T) {
			scale := math.Ldexp(1, exponent)
			scaleSquared := math.Ldexp(1, 2*exponent)
			scaledY := make([]float64, len(responses))
			for i, value := range responses {
				scaledY[i] = value * scale
			}
			fit, err := ridge.FitStandardized(context.Background(), input, scaledY, lambda)
			if err != nil {
				t.Fatalf("rescaled wide fit: %v", err)
			}
			beta, intercept := fit.Model.Coefficients()
			if len(beta) != len(baseBeta) {
				t.Fatalf("feature width changed: got=%d want=%d", len(beta), len(baseBeta))
			}
			for j, value := range beta {
				// Relative to the *base* coefficient after removing response units.
				if math.Abs(baseBeta[j]) < 1e-14 {
					if math.Abs(value/scale-baseBeta[j]) > tolerance {
						t.Errorf("beta[%d] near zero: scaled=%g base=%g", j, value/scale, baseBeta[j])
					}
					continue
				}
				ridgeScaledClose(t, "physical beta", value/scale, baseBeta[j], tolerance)
			}
			if math.Abs(baseIntercept) < 1e-14 {
				if math.Abs(intercept/scale-baseIntercept) > tolerance {
					t.Errorf("intercept: scaled=%g base=%g", intercept/scale, baseIntercept)
				}
			} else {
				ridgeScaledClose(t, "physical intercept", intercept/scale, baseIntercept, tolerance)
			}
			ridgeScaledClose(t, "SSE", fit.ResidualSumSquares/scaleSquared, base.ResidualSumSquares, tolerance)
			ridgeScaledClose(t, "Penalty", fit.Penalty/scaleSquared, base.Penalty, tolerance)
			ridgeScaledClose(t, "Objective", fit.Objective/scaleSquared, base.Objective, tolerance)
		})
	}
}
