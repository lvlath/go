// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ols_test protects representation invariance of full-rank OLS.
// The tests invoke only the public ols API and use numerical comparisons for
// floating-point recovery while comparing discrete error/shape laws exactly.
package ols_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/ols"
)

// TestOLS_InterceptFeatureTranslationEquivariance proves a change of feature
// origin is a basis translation, not a change of the affine prediction law.
// The original slopes must remain approximately equal and the intercept must
// absorb the translation, for both unscaled and population-scaled fits.
func TestOLS_InterceptFeatureTranslationEquivariance(t *testing.T) {
	ctx := context.Background()
	originalRows := [][]float64{
		{10, 2}, {13, 5}, {18, 3}, {22, 8}, {27, 4}, {33, 9},
		{39, 6}, {42, 11}, {48, 5}, {54, 10}, {59, 7}, {64, 13},
	}
	response := []float64{31, 43, 43, 59, 57, 77, 78, 94, 91, 110, 111, 130}
	const shiftPressure = 1000.0
	const shiftDuty = -200.0
	translatedRows := make([][]float64, len(originalRows))
	for i, row := range originalRows {
		translatedRows[i] = []float64{row[0] + shiftPressure, row[1] + shiftDuty}
	}

	for _, standardized := range []bool{false, true} {
		name := "unscaled"
		if standardized {
			name = "population_scaled"
		}
		t.Run(name, func(t *testing.T) {
			base, err := ols.Fit(ctx, mustDense(t, originalRows), response,
				ols.Options{Standardize: standardized})
			if err != nil {
				t.Fatal(err)
			}
			translated, err := ols.Fit(ctx, mustDense(t, translatedRows), response,
				ols.Options{Standardize: standardized})
			if err != nil {
				t.Fatal(err)
			}
			mustEqualInt(t, base.Rank, 3, "original full rank")
			mustEqualInt(t, translated.Rank, 3, "translated full rank")
			beta, intercept := base.Model.Coefficients()
			translatedBeta, translatedIntercept := translated.Model.Coefficients()
			for j := range beta {
				mustNear(t, translatedBeta[j], beta[j], 1e-10)
			}
			wantIntercept := math.FMA(-shiftDuty, beta[1],
				math.FMA(-shiftPressure, beta[0], intercept))
			mustNear(t, translatedIntercept, wantIntercept, 1e-10)

			// A heldout row and its translated counterpart must represent the
			// same physical observation under the two coordinate systems.
			basePrediction, err := base.Model.Predict(ctx, mustDense(t, [][]float64{{71, 15}}))
			if err != nil {
				t.Fatal(err)
			}
			translatedPrediction, err := translated.Model.Predict(ctx,
				mustDense(t, [][]float64{{71 + shiftPressure, 15 + shiftDuty}}))
			if err != nil {
				t.Fatal(err)
			}
			mustNear(t, translatedPrediction[0], basePrediction[0], 1e-10)
			mustNear(t, translated.ResidualSumSquares, base.ResidualSumSquares, 1e-9)
		})
	}
}

// TestOLS_ThroughOriginUnitChangeEquivariance checks that changing feature
// units rescales slopes without creating a hidden offset. A through-origin
// fit must preserve predictions when one predictor is converted to new units.
func TestOLS_ThroughOriginUnitChangeEquivariance(t *testing.T) {
	ctx := context.Background()
	originalRows := [][]float64{
		{1, 4}, {2, 1}, {3, 5}, {4, 2}, {5, 7}, {6, 3},
		{7, 8}, {8, 5}, {9, 9}, {10, 6}, {11, 10}, {12, 7},
	}
	response := []float64{15, 20, 32, 40, 57, 60, 77, 83, 97, 104, 120, 125}
	const convertedUnits = 1000.0
	convertedRows := make([][]float64, len(originalRows))
	for i, row := range originalRows {
		convertedRows[i] = []float64{row[0] * convertedUnits, row[1]}
	}

	for _, standardized := range []bool{false, true} {
		name := "raw_units"
		if standardized {
			name = "scaled_units"
		}
		t.Run(name, func(t *testing.T) {
			oldUnits, err := ols.Fit(ctx, mustDense(t, originalRows), response,
				ols.Options{NoIntercept: true, Standardize: standardized})
			if err != nil {
				t.Fatal(err)
			}
			newUnits, err := ols.Fit(ctx, mustDense(t, convertedRows), response,
				ols.Options{NoIntercept: true, Standardize: standardized})
			if err != nil {
				t.Fatal(err)
			}
			oldBeta, oldIntercept := oldUnits.Model.Coefficients()
			newBeta, newIntercept := newUnits.Model.Coefficients()
			if oldIntercept != 0 || math.Signbit(oldIntercept) ||
				newIntercept != 0 || math.Signbit(newIntercept) {
				t.Fatalf("origin constraint lost: intercepts %g and %g", oldIntercept, newIntercept)
			}
			mustNear(t, convertedUnits*newBeta[0], oldBeta[0], 1e-10)
			mustNear(t, newBeta[1], oldBeta[1], 1e-10)
			predA, err := oldUnits.Model.Predict(ctx, mustDense(t, [][]float64{{13, 11}}))
			if err != nil {
				t.Fatal(err)
			}
			predB, err := newUnits.Model.Predict(ctx,
				mustDense(t, [][]float64{{13 * convertedUnits, 11}}))
			if err != nil {
				t.Fatal(err)
			}
			mustNear(t, predA[0], predB[0], 1e-10)
			mustNear(t, oldUnits.ResidualSumSquares, newUnits.ResidualSumSquares, 1e-9)
		})
	}
}
