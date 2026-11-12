// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// PCA contract tests exercise the public facade from an independent package.
package pca_test

import (
	"context"
	"errors"
	"math"
	"sync"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/pca"
)

func TestPCA_LowRankReconstruction(t *testing.T) {
	x := fixture(t, [][]float64{{-1, 0}, {0, 0}, {1, 0}})
	got, err := pca.Fit(context.Background(), x, pca.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if got.Rank != 1 || len(got.Eigenvalues) != 1 || got.Model == nil {
		t.Fatalf("invalid low-rank artifact: %+v", got)
	}
	mustNear(t, got.Eigenvalues[0], 1, 1e-12)
	mustNear(t, got.ExplainedVarianceRatio[0], 1, 1e-12)
	mustNear(t, got.ReconstructionResidual, 0, 1e-12)
	mustNear(t, mustCell(t, got.Model.Components(), 0, 0), 1, 1e-12)
	scores, err := got.Model.Transform(context.Background(), x)
	if err != nil {
		t.Fatal(err)
	}
	rebuilt, err := got.Model.InverseTransform(context.Background(), scores)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < x.Rows(); i++ {
		for j := 0; j < x.Cols(); j++ {
			mustNear(t, mustCell(t, rebuilt, i, j), mustCell(t, x, i, j), 1e-12)
		}
	}
}
func TestPCA_RankDeficientRequest(t *testing.T) {
	x := fixture(t, [][]float64{{-1, 0}, {0, 0}, {1, 0}})
	got, err := pca.Fit(context.Background(), x, pca.Options{Components: 2})
	mustErrorIs(t, err, pca.ErrRankDeficient)
	if got != nil {
		t.Fatalf("partial model published: %+v", got)
	}
}
func TestPCA_HeldoutCentering(t *testing.T) {
	train := fixture(t, [][]float64{{10}, {12}, {14}, {16}})
	result, err := pca.Fit(context.Background(), train, pca.Options{})
	if err != nil {
		t.Fatal(err)
	}
	heldout := fixture(t, [][]float64{{20}, {22}})
	projected, err := result.Model.Transform(context.Background(), heldout)
	if err != nil {
		t.Fatal(err)
	}
	// Training mean 13, so positive canonical scores 7 and 9, not -1 and +1.
	mustNear(t, mustCell(t, projected, 0, 0), 7, 1e-12)
	mustNear(t, mustCell(t, projected, 1, 0), 9, 1e-12)
}
func TestPCA_ZeroTotalVariance(t *testing.T) {
	x := fixture(t, [][]float64{{4, 7}, {4, 7}, {4, 7}})
	got, err := pca.Fit(context.Background(), x, pca.Options{})
	mustErrorIs(t, err, pca.ErrZeroVariance)
	if got != nil {
		t.Fatal("partial model")
	}
}
func TestPCA_StandardizeRejectConstant(t *testing.T) {
	x := fixture(t, [][]float64{{-1, 0}, {0, 0}, {1, 0}})
	got, err := pca.Fit(context.Background(), x, pca.Options{Standardize: true})
	mustErrorIs(t, err, pca.ErrZeroVariance)
	if got != nil {
		t.Fatal("partial model")
	}
}
func TestPCA_MaxRotationsFailure(t *testing.T) {
	// Non-diagonal 3x3 covariance needs more than one maximal-pivot rotation.
	x := fixture(t, [][]float64{{1, 2, 4}, {2, 4, 1}, {4, 1, 3}, {8, 3, 2}, {3, 7, 5}})
	got, err := pca.Fit(context.Background(), x, pca.Options{MaxRotations: 1})
	mustErrorIs(t, err, pca.ErrNotConverged)
	mustErrorIs(t, err, matrix.ErrMatrixEigenFailed)
	if got != nil {
		t.Fatal("partially fitted result")
	}
}
func TestPCA_ComponentsDetached(t *testing.T) {
	x := fixture(t, [][]float64{{-1, 0}, {0, 0}, {1, 0}})
	r, err := pca.Fit(context.Background(), x, pca.Options{})
	if err != nil {
		t.Fatal(err)
	}
	c := r.Model.Components()
	if err = c.Set(0, 0, -123); err != nil {
		t.Fatal(err)
	}
	scores, err := r.Model.Transform(context.Background(), x)
	if err != nil {
		t.Fatal(err)
	}
	mustNear(t, mustCell(t, scores, 2, 0), 1, 1e-12)
}
func TestPCA_FitResult_IsNilInterface(t *testing.T) {
	var absent *pca.FitResult
	var nilable core.Nilable = absent
	if !nilable.IsNil() {
		t.Fatal("typed nil must be nil")
	}
	nilable = &pca.FitResult{}
	if nilable.IsNil() {
		t.Fatal("allocated result is present")
	}
}
func TestPCA_ConcurrentTransformRace(t *testing.T) {
	train := fixture(t, [][]float64{{1, 4, 3}, {2, 7, 5}, {4, 2, 6}, {7, 3, 8}, {9, 4, 2}, {5, 9, 1}})
	r, err := pca.Fit(context.Background(), train, pca.Options{Components: 2})
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	errs := make(chan error, 64)
	for i := 0; i < 64; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); _, e := r.Model.Transform(context.Background(), train); errs <- e }()
	}
	wg.Wait()
	close(errs)
	for e := range errs {
		if e != nil {
			t.Fatalf("concurrent inference: %v", e)
		}
	}
}
func TestPCA_ThinSVD_ReturnsUnsupported(t *testing.T) {
	got, err := pca.Fit(context.Background(), nil, pca.Options{Solver: pca.ThinSVD})
	mustErrorIs(t, err, pca.ErrUnsupportedSolver)
	if got != nil {
		t.Fatal("partial model")
	}
}
func TestPCA_ContextAndOptionPrecedence(t *testing.T) {
	_, err := pca.Fit(nil, nil, pca.Options{Solver: pca.ThinSVD})
	mustErrorIs(t, err, pca.ErrInvalidOption)
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = pca.Fit(canceled, nil, pca.Options{Solver: pca.ThinSVD})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel precedence: %v", err)
	}
	_, err = pca.Fit(context.Background(), nil, pca.Options{EigenTolerance: math.NaN()})
	mustErrorIs(t, err, pca.ErrInvalidOption)
}
func TestPCA_ZeroRowInference(t *testing.T) {
	train := fixture(t, [][]float64{{1, 2}, {2, 4}, {3, 6}})
	r, err := pca.Fit(context.Background(), train, pca.Options{})
	if err != nil {
		t.Fatal(err)
	}
	zero, err := matrix.NewDense(0, 2)
	if err != nil {
		t.Fatal(err)
	}
	scores, err := r.Model.Transform(context.Background(), zero)
	if err != nil {
		t.Fatal(err)
	}
	if scores.Rows() != 0 || scores.Cols() != 1 {
		t.Fatalf("bad zero-row score shape: %dx%d", scores.Rows(), scores.Cols())
	}
	restored, err := r.Model.InverseTransform(context.Background(), scores)
	if err != nil {
		t.Fatal(err)
	}
	if restored.Rows() != 0 || restored.Cols() != 2 {
		t.Fatal("bad zero-row reconstruction shape")
	}
}
