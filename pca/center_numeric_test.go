// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pca_test protects the frozen-mean and scaled-spectrum invariants.
// Independent binary64 oracles prevent a numerically incorrect model from
// passing merely because Fit and Transform share the same broken arithmetic.
package pca_test

import (
	"context"
	"errors"
	"math"
	"math/big"
	"sync"
	"testing"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/pca"
)

// TestPCATrainingCenterConsistency ensures fitted scores share the center
// used to construct sample covariance, including a mean between float64 values.
func TestPCATrainingCenterConsistency(t *testing.T) {
	base := math.Ldexp(1, 450)
	next := math.Nextafter(base, math.Inf(1))
	gap := next - base
	train, err := matrix.NewDense(2, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err = train.Fill([]float64{base, next}); err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	fit, err := pca.FitDefault(ctx, train)
	if err != nil {
		t.Fatalf("Fit failed for representable covariance: %v", err)
	}
	if fit.Rank != 1 {
		t.Fatalf("rank: got %d, want 1", fit.Rank)
	}
	if relativeDiff(fit.Eigenvalues[0], gap*gap/2) > 2e-12 {
		t.Fatalf("sample eigenvalue %.17g, exact %.17g", fit.Eigenvalues[0], gap*gap/2)
	}
	scores, err := fit.Model.Transform(ctx, train)
	if err != nil {
		t.Fatal(err)
	}
	left, err := scores.At(0, 0)
	if err != nil {
		t.Fatal(err)
	}
	right, err := scores.At(1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(left/gap+0.5) > 5e-15 || math.Abs(right/gap-0.5) > 5e-15 {
		t.Fatalf("training scores must be centered around the exact midpoint: [%g, %g] / gap=%g", left, right, gap)
	}
	if math.Abs((left+right)/gap) > 5e-15 {
		t.Fatalf("training mean score drift: %.17g", (left+right)/gap)
	}
	recovered, err := fit.Model.InverseTransform(ctx, scores)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []float64{base, next} {
		got, readErr := recovered.At(i, 0)
		if readErr != nil {
			t.Fatal(readErr)
		}
		if got != want {
			t.Fatalf("inverse row %d: got %.17g, want %.17g", i, got, want)
		}
	}
	if fit.ReconstructionResidual > 1e-10*gap {
		t.Fatalf("full-rank reconstruction residual %.17g exceeds gap-relative limit", fit.ReconstructionResidual)
	}
}

// TestPCACenteredTruncatedResidual checks a centered 2D training residual
// against a separate discarded-variance oracle with a very large affine offset.
func TestPCACenteredTruncatedResidual(t *testing.T) {
	base := math.Ldexp(1, 450)
	gap := math.Nextafter(base, math.Inf(1)) - base
	train, err := matrix.NewDense(4, 2)
	if err != nil {
		t.Fatal(err)
	}
	if err = train.Fill([]float64{
		base, 0,
		base + gap, gap,
		base + 2*gap, gap,
		base + 3*gap, 0,
	}); err != nil {
		t.Fatal(err)
	}
	fit, err := pca.FitWithComponents(context.Background(), train, 1)
	if err != nil {
		t.Fatal(err)
	}
	if fit.Rank != 2 {
		t.Fatalf("rank: %d", fit.Rank)
	}
	if math.Abs(fit.ReconstructionResidual/gap-1) > 1e-10 {
		t.Fatalf("residual / gap = %.17g, exact = 1", fit.ReconstructionResidual/gap)
	}
	if fit.ExplainedVarianceRatio[0] <= 0 || fit.ExplainedVarianceRatio[0] >= 1 {
		t.Fatalf("truncation ratio: %.17g", fit.ExplainedVarianceRatio[0])
	}

	// Projected training variance is an independent matrix covariance oracle:
	// it must equal the accepted leading eigenvalue, not the covariance of
	// features centered with a separately rounded one-float mean.
	scores, err := fit.Model.Transform(context.Background(), train)
	if err != nil {
		t.Fatal(err)
	}
	scoreCovariance, _, err := matrix.Covariance(scores)
	if err != nil {
		t.Fatal(err)
	}
	projectedVariance, err := scoreCovariance.At(0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if relativeDiff(projectedVariance, fit.Eigenvalues[0]) > 2e-11 {
		t.Fatalf("projected variance %.17g != fitted eigenvalue %.17g", projectedVariance, fit.Eigenvalues[0])
	}
}

// TestPCAStandardizedCenterConsistency proves that population scaling
// and covariance centering remain two frozen but distinct affine stages.
func TestPCAStandardizedCenterConsistency(t *testing.T) {
	base := math.Ldexp(1, 450)
	next := math.Nextafter(base, math.Inf(1))
	train, err := matrix.NewDense(2, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err = train.Fill([]float64{base, next}); err != nil {
		t.Fatal(err)
	}
	fit, err := pca.FitStandardized(context.Background(), train)
	if err != nil {
		t.Fatal(err)
	}
	scores, err := fit.Model.Transform(context.Background(), train)
	if err != nil {
		t.Fatal(err)
	}
	first, err := scores.At(0, 0)
	if err != nil {
		t.Fatal(err)
	}
	second, err := scores.At(1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(first+second) > 1e-12 || math.Abs(second-first) < 0.5 {
		t.Fatalf("standardized score center drift: [%g,%g]", first, second)
	}
	inverse, err := fit.Model.InverseTransform(context.Background(), scores)
	if err != nil {
		t.Fatal(err)
	}
	for row, expected := range []float64{base, next} {
		got, err := inverse.At(row, 0)
		if err != nil {
			t.Fatal(err)
		}
		if got != expected {
			t.Fatalf("inverse standardized row %d: got %g, want %g", row, got, expected)
		}
	}
}

// TestPCARepresentableSpectrum protects the rank-one covariance whose
// infinity norm overflows, although the leading eigenvalue remains finite.
func TestPCARepresentableSpectrum(t *testing.T) {
	u := math.Sqrt(1.2e308)
	v := 0.7e308 / u
	x, err := matrix.NewDense(3, 2)
	if err != nil {
		t.Fatal(err)
	}
	if err = x.Fill([]float64{-u, -v, 0, 0, u, v}); err != nil {
		t.Fatal(err)
	}
	var exactU, exactV, exact big.Rat
	exactU.SetFloat64(u)
	exactV.SetFloat64(v)
	exact.Mul(&exactU, &exactU)
	var squareV big.Rat
	squareV.Mul(&exactV, &exactV)
	exact.Add(&exact, &squareV)
	want, _ := exact.Float64()
	if !isFiniteTest(want) {
		t.Fatal("oracle must be representable")
	}
	fit, err := pca.FitDefault(context.Background(), x)
	if err != nil {
		t.Fatalf("representable spectrum rejected: %v", err)
	}
	if fit.Rank != 1 || len(fit.Eigenvalues) != 1 {
		t.Fatalf("expected rank 1, got rank=%d values=%v", fit.Rank, fit.Eigenvalues)
	}
	if relativeDiff(fit.Eigenvalues[0], want) > 2e-12 {
		t.Fatalf("eigenvalue got %.17g, exact %.17g", fit.Eigenvalues[0], want)
	}
	if math.Abs(fit.ExplainedVarianceRatio[0]-1) > 2e-12 {
		t.Fatalf("rank-one variance fraction %.17g, want 1", fit.ExplainedVarianceRatio[0])
	}
	if fit.ReconstructionResidual > 1e-11*math.Max(math.Abs(u), math.Abs(v)) {
		t.Fatalf("rank-one reconstruction error %.17g", fit.ReconstructionResidual)
	}
}

// TestPCAUnrepresentableEigenvalue verifies that scaled Jacobi does not
// turn a genuinely overflowing required eigenvalue into a successful fit.
func TestPCAUnrepresentableEigenvalue(t *testing.T) {
	u := math.Sqrt(1.7e308)
	v := math.Sqrt(0.4e308)
	x, err := matrix.NewDense(3, 2)
	if err != nil {
		t.Fatal(err)
	}
	if err = x.Fill([]float64{-u, -v, 0, 0, u, v}); err != nil {
		t.Fatal(err)
	}
	result, err := pca.FitDefault(context.Background(), x)
	if result != nil || !errors.Is(err, pca.ErrOverflow) {
		t.Fatalf("unrepresentable eigenvalue: result=%v, err=%v; want nil ErrOverflow", result, err)
	}
}

// TestPCASmallBinaryScale checks scale-aware covariance analysis near
// subnormal binary exponents without changing eigenvector canonicalization.
func TestPCASmallBinaryScale(t *testing.T) {
	u := 1e-150
	v := 0.7e-150
	x, err := matrix.NewDense(3, 2)
	if err != nil {
		t.Fatal(err)
	}
	if err = x.Fill([]float64{-u, -v, 0, 0, u, v}); err != nil {
		t.Fatal(err)
	}
	fit, err := pca.FitDefault(context.Background(), x)
	if err != nil {
		t.Fatal(err)
	}
	if fit.Rank != 1 {
		t.Fatalf("rank = %d, want 1", fit.Rank)
	}
	if relativeDiff(fit.Eigenvalues[0], u*u+v*v) > 1e-11 {
		t.Fatalf("tiny eigenvalue %g, oracle %g", fit.Eigenvalues[0], u*u+v*v)
	}
}

// relativeDiff measures a finite positive reference-relative deviation.
//
//	inputs guarantee a nonzero, representable oracle denominator.
func relativeDiff(a, b float64) float64 { return math.Abs(a/b - 1) }
func isFiniteTest(v float64) bool       { return !math.IsNaN(v) && !math.IsInf(v, 0) }

// TestPCAPopulationScalingDoesNotOverflowBeforeDivision proves that
// original-unit heldout subtraction may overflow even when the fitted
// population-standardized coordinate is small, finite and meaningful.
func TestPCAPopulationScalingDoesNotOverflowBeforeDivision(t *testing.T) {
	train, err := matrix.NewDense(2, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err = train.Fill([]float64{-1.5e308, -0.5e308}); err != nil {
		t.Fatal(err)
	}
	fit, err := pca.FitStandardized(context.Background(), train)
	if err != nil {
		t.Fatalf("finite scales must not require finite squared variance: %v", err)
	}
	future, err := matrix.NewDense(1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err = future.Set(0, 0, 1e308); err != nil {
		t.Fatal(err)
	}
	scores, err := fit.Model.Transform(context.Background(), future)
	if err != nil {
		t.Fatalf("finite scaled delta must not overflow: %v", err)
	}
	got, err := scores.At(0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got-4) > 1e-12 {
		t.Fatalf("holdout score: got %.17g, expected approximately 4", got)
	}
	inverse, err := fit.Model.InverseTransform(context.Background(), scores)
	if err != nil {
		t.Fatal(err)
	}
	restored, err := inverse.At(0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if relativeDiff(restored, 1e308) > 1e-15 {
		t.Fatalf("restoration got %.17g, expected 1e308", restored)
	}
}

// TestPCAFrozenCenterConcurrentHoldout protects both the exact center
// and per-call workspace when many goroutines transform identical heldout data.
// Splitting a batch must not change the fitted coordinate origin or scores.
func TestPCAFrozenCenterConcurrentHoldout(t *testing.T) {
	const workers = 64
	base := math.Ldexp(1, 450)
	gap := math.Nextafter(base, math.Inf(1)) - base
	training, err := matrix.NewDense(2, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err = training.Fill([]float64{base, base + gap}); err != nil {
		t.Fatal(err)
	}
	fit, err := pca.FitDefault(context.Background(), training)
	if err != nil {
		t.Fatal(err)
	}
	future, err := matrix.NewDense(1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err = future.Set(0, 0, base+2*gap); err != nil {
		t.Fatal(err)
	}

	failures := make(chan error, workers)
	var group sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		group.Add(1)
		go func() {
			defer group.Done()
			z, err := fit.Model.Transform(context.Background(), future)
			if err != nil {
				failures <- err
				return
			}
			got, err := z.At(0, 0)
			if err != nil {
				failures <- err
				return
			}
			if math.Abs(got/gap-1.5) > 1e-13 {
				failures <- errors.New("heldout center drift under concurrency")
				return
			}
			_, err = fit.Model.InverseTransform(context.Background(), z)
			if err != nil {
				failures <- err
			}
		}()
	}
	group.Wait()
	close(failures)
	for err = range failures {
		t.Errorf("parallel frozen model inference: %v", err)
	}
}
