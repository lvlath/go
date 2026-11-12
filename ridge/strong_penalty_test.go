// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ridge_test protects strongly regularized, still nonzero solutions.
// The independent analytic oracles do not depend on QR column-pivot decisions.
package ridge_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/ridge"
)

// TestRidge_StrongLambdaInterceptOracle guards two distinct aspects of numerical
// rank: the intercept must remain unpenalized, while the small slope must not
// disappear when the shrinkage is much stronger than the observed feature norm.
func TestRidge_StrongLambdaInterceptOracle(t *testing.T) {
	const lambda = 1e100
	fit, err := ridge.FitWithLambda(context.Background(), dense(t, [][]float64{{-1}, {1}}), []float64{2, 4}, lambda)
	if err != nil {
		t.Fatal(err)
	}
	beta, intercept := fit.Model.Coefficients()
	wantSlope := 1 / (1 + lambda)
	if len(beta) != 1 || math.Abs(beta[0]/wantSlope-1) > 1e-8 {
		t.Fatalf("strong-penalty slope: got %v, want %.17g", beta, wantSlope)
	}
	if math.Abs(intercept-3) > 1e-12 {
		t.Fatalf("unpenalized intercept: %.17g want 3", intercept)
	}
	wantPenalty := (lambda / 2) * wantSlope * wantSlope
	if math.Abs(fit.Penalty/wantPenalty-1) > 1e-8 {
		t.Fatalf("strong-penalty Penalty: %.17g want %.17g", fit.Penalty, wantPenalty)
	}
}

// TestRidge_StrongLambdaDuplicateColumnsOracle proves that equal duplicated
// predictors split their common shrinkage symmetrically, even at huge Lambda.
// The exact normal-equation oracle is only used to derive the expected scalar.
func TestRidge_StrongLambdaDuplicateColumnsOracle(t *testing.T) {
	const lambda = 1e100
	fit, err := ridge.FitThroughOrigin(context.Background(), dense(t, [][]float64{
		{1, 1}, {2, 2}, {3, 3}, {4, 4},
	}), []float64{1, 2, 3, 4}, lambda)
	if err != nil {
		t.Fatal(err)
	}
	beta, intercept := fit.Model.Coefficients()
	want := 30 / (60 + 4*lambda)
	if intercept != 0 || len(beta) != 2 {
		t.Fatalf("invalid model coordinates beta=%v intercept=%g", beta, intercept)
	}
	for j, slope := range beta {
		if math.Abs(slope/want-1) > 1e-8 {
			t.Fatalf("slope[%d]=%.17g want %.17g", j, slope, want)
		}
	}
}

// TestRidge_MaxFloatLambdaNonzero verifies that sqrt(n*Lambda) and the small
// output slope are both representable even when n*Lambda overflows float64.
func TestRidge_MaxFloatLambdaNonzero(t *testing.T) {
	lambda := math.MaxFloat64
	fit, err := ridge.FitThroughOrigin(context.Background(), dense(t, [][]float64{{1}, {2}}), []float64{1, 2}, lambda)
	if err != nil {
		t.Fatal(err)
	}
	beta, intercept := fit.Model.Coefficients()
	// Rearranged to avoid overflowing the oracle's denominator.
	want := (5 / lambda) / (2 + 5/lambda)
	if intercept != 0 || len(beta) != 1 || beta[0] == 0 || math.Abs(beta[0]/want-1) > 1e-8 {
		t.Fatalf("slope %.17g want %.17g", beta[0], want)
	}
}
