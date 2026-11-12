// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Extreme-magnitude acceptance probes target the covariance dependency gate.
package pca_test

import (
	"context"
	"testing"

	"github.com/lvlath/go/pca"
)

func TestPCA_ExtremeConstantCovariance(t *testing.T) {
	x := fixture(t, [][]float64{{1e308}, {1e308}})
	result, err := pca.Fit(context.Background(), x, pca.Options{})
	mustErrorIs(t, err, pca.ErrZeroVariance)
	if result != nil {
		t.Fatal("constant extreme input produced a model")
	}
}
func TestPCA_ExtremeRepresentableCovariance(t *testing.T) {
	// Sum of two squared observations is 2e308 (> MaxFloat64), but sample
	// variance after division by n-1 is exactly 1e308 (representable).
	x := fixture(t, [][]float64{{-1e154}, {0}, {1e154}})
	result, err := pca.Fit(context.Background(), x, pca.Options{})
	if err != nil {
		t.Fatal(err)
	}
	mustNear(t, result.Eigenvalues[0]/1e308, 1, 1e-12)
	if result.Rank != 1 {
		t.Fatal("unexpected extreme covariance rank")
	}
	if result.ReconstructionResidual > 1e140 {
		t.Fatalf("extreme reconstruction residual: %g", result.ReconstructionResidual)
	}
}
