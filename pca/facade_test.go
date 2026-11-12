// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// The public PCA helpers must delegate to Fit without changing numerical policy.
// All assertions use the independent pca_test package and observable model APIs.
package pca_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/pca"
	"github.com/lvlath/go/stats"
)

// assertSameFittedModel checks two independent fits using the same policy.
// A whole repeated eigenspace may rotate between kernels; these assertions
// compare only repeated calls of this exact deterministic kernel and options.
func assertSameFittedModel(t *testing.T, got, expected *pca.FitResult) {
	t.Helper()
	if got == nil || expected == nil || got.Model == nil || expected.Model == nil {
		t.Fatal("models must both be fitted")
	}
	if got.Rank != expected.Rank || len(got.Eigenvalues) != len(expected.Eigenvalues) {
		t.Fatal("wrapper changed the rank or retained dimension")
	}
	for i := range got.Eigenvalues {
		mustNear(t, got.Eigenvalues[i], expected.Eigenvalues[i], 1e-12)
		mustNear(t, got.ExplainedVarianceRatio[i], expected.ExplainedVarianceRatio[i], 1e-12)
	}
	mustNear(t, got.ReconstructionResidual, expected.ReconstructionResidual, 1e-12)
	a, b := got.Model.Components(), expected.Model.Components()
	if a.Rows() != b.Rows() || a.Cols() != b.Cols() {
		t.Fatal("wrapper changed the fitted basis shape")
	}
	for row := 0; row < a.Rows(); row++ {
		for col := 0; col < a.Cols(); col++ {
			mustNear(t, mustCell(t, a, row, col), mustCell(t, b, row, col), 1e-12)
		}
	}
}

func TestPCA_FitFacadeWrappersHaveOneKernel(t *testing.T) {
	ctx := context.Background()
	// Independent predictor scales keep standardized vs raw PCA genuinely distinct.
	x := fixture(t, [][]float64{
		{5, 75, 2}, {7, 80, 3}, {9, 70, 8}, {11, 100, 6},
		{4, 85, 9}, {10, 90, 1}, {12, 110, 7}, {13, 65, 5},
	})
	cases := []struct {
		name string
		opts pca.Options
		call func() (*pca.FitResult, error)
	}{
		{"default", pca.Options{}, func() (*pca.FitResult, error) { return pca.FitDefault(ctx, x) }},
		{"components", pca.Options{Components: 2}, func() (*pca.FitResult, error) { return pca.FitWithComponents(ctx, x, 2) }},
		{"standardize", pca.Options{Standardize: true}, func() (*pca.FitResult, error) { return pca.FitStandardized(ctx, x) }},
		{"standardize components", pca.Options{Components: 2, Standardize: true}, func() (*pca.FitResult, error) {
			return pca.FitStandardizedWithComponents(ctx, x, 2)
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tc.call()
			if err != nil {
				t.Fatal(err)
			}
			want, err := pca.Fit(ctx, x, tc.opts)
			if err != nil {
				t.Fatal(err)
			}
			assertSameFittedModel(t, got, want)
		})
	}
}

func TestPCA_WrapperErrorIdentityAndPrecedence(t *testing.T) {
	ctx := context.Background()
	x := fixture(t, [][]float64{{-1, 1}, {0, 0}, {1, -1}})
	xConstant := fixture(t, [][]float64{{-1, 0}, {0, 0}, {1, 0}})
	cases := []struct {
		name string
		call func() (*pca.FitResult, error)
		want error
	}{
		{"nil context", func() (*pca.FitResult, error) { return pca.FitDefault(nil, nil) }, pca.ErrInvalidOption},
		{"negative components", func() (*pca.FitResult, error) { return pca.FitWithComponents(ctx, nil, -1) }, pca.ErrInvalidOption},
		{"missing input", func() (*pca.FitResult, error) { return pca.FitWithComponents(ctx, nil, 1) }, pca.ErrNilInput},
		{"rank deficient", func() (*pca.FitResult, error) { return pca.FitWithComponents(ctx, x, 2) }, pca.ErrRankDeficient},
		{"standardized constant", func() (*pca.FitResult, error) {
			return pca.FitStandardizedWithComponents(ctx, xConstant, 1)
		}, pca.ErrZeroVariance},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := tc.call()
			mustErrorIs(t, err, tc.want)
			if result != nil {
				t.Fatal("failed wrapper published a model")
			}
			if tc.name == "standardized constant" && !errors.Is(err, stats.ErrZeroVariance) {
				t.Fatalf("stats root cause was discarded: %v", err)
			}
		})
	}
}

func TestPCA_ReconstructEquivalentDetachedAndZeroRows(t *testing.T) {
	ctx := context.Background()
	train := fixture(t, [][]float64{
		{2, 10, 3}, {4, 20, 2}, {6, 30, 8}, {8, 40, 1},
		{10, 50, 9}, {12, 60, 4}, {14, 70, 7}, {16, 80, 5},
	})
	fit, err := pca.FitWithComponents(ctx, train, 2)
	if err != nil {
		t.Fatal(err)
	}
	future := fixture(t, [][]float64{{3, 15, 4}, {7, 35, 6}, {12, 60, 7}})
	// Compare direct public composition with its new convenience wrapper.
	scores, err := fit.Model.Transform(ctx, future)
	if err != nil {
		t.Fatal(err)
	}
	direct, err := fit.Model.InverseTransform(ctx, scores)
	if err != nil {
		t.Fatal(err)
	}
	wrapped, err := fit.Model.Reconstruct(ctx, future)
	if err != nil {
		t.Fatal(err)
	}
	for row := 0; row < future.Rows(); row++ {
		for col := 0; col < future.Cols(); col++ {
			mustNear(t, mustCell(t, wrapped, row, col), mustCell(t, direct, row, col), 1e-12)
		}
	}
	if err = wrapped.Set(0, 0, -6789); err != nil {
		t.Fatal(err)
	}
	fresh, err := fit.Model.Reconstruct(ctx, future)
	if err != nil {
		t.Fatal(err)
	}
	mustNear(t, mustCell(t, fresh, 0, 0), mustCell(t, direct, 0, 0), 1e-12)
	mustNear(t, mustCell(t, future, 0, 0), 3, 0)
	zero, err := matrix.NewDense(0, train.Cols())
	if err != nil {
		t.Fatal(err)
	}
	empty, err := fit.Model.Reconstruct(ctx, zero)
	if err != nil {
		t.Fatal(err)
	}
	if empty.Rows() != 0 || empty.Cols() != train.Cols() {
		t.Fatalf("zero-row reconstruction shape: %d x %d", empty.Rows(), empty.Cols())
	}
}

func TestPCA_ReconstructErrorsNeverPublishPartialOutput(t *testing.T) {
	ctx := context.Background()
	train := fixture(t, [][]float64{{-1, -1}, {0, 0}, {1, 1}})
	fit, err := pca.FitDefault(ctx, train)
	if err != nil {
		t.Fatal(err)
	}
	bad := fixture(t, [][]float64{{1, 2, 3}})
	for _, tc := range []struct {
		name string
		m    *pca.Model
		ctx  context.Context
		x    matrix.Matrix
		want error
	}{
		{"nil model", nil, ctx, train, pca.ErrNilInput},
		{"unfitted model", &pca.Model{}, ctx, train, pca.ErrNilInput},
		{"nil context", fit.Model, nil, train, pca.ErrInvalidOption},
		{"wrong shape", fit.Model, ctx, bad, pca.ErrShape},
		{"nil input", fit.Model, ctx, nil, pca.ErrNilInput},
	} {
		t.Run(tc.name, func(t *testing.T) {
			result, failure := tc.m.Reconstruct(tc.ctx, tc.x)
			mustErrorIs(t, failure, tc.want)
			if result != nil {
				t.Fatal("partial reconstruction was published")
			}
		})
	}
}

func TestPCA_StandardizedHoldoutUsesFrozenPopulationScales(t *testing.T) {
	ctx := context.Background()
	train := fixture(t, [][]float64{{8, 100}, {10, 200}, {12, 300}, {14, 400}})
	fit, err := pca.FitStandardizedWithComponents(ctx, train, 1)
	if err != nil {
		t.Fatal(err)
	}
	// Both centered, population-standardized coordinates are 7/sqrt(5).
	// The canonical all-positive 45-degree eigenvector maps them to one score.
	holdout := fixture(t, [][]float64{{18, 600}})
	scores, err := fit.Model.Transform(ctx, holdout)
	if err != nil {
		t.Fatal(err)
	}
	mustNear(t, mustCell(t, scores, 0, 0), 7*math.Sqrt(2.0/5.0), 1e-11)
	originalUnits, err := fit.Model.Reconstruct(ctx, holdout)
	if err != nil {
		t.Fatal(err)
	}
	mustNear(t, mustCell(t, originalUnits, 0, 0), 18, 1e-10)
	mustNear(t, mustCell(t, originalUnits, 0, 1), 600, 1e-9)
}
