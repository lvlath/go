// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Orthogonal-projector tests avoid relying on arbitrary basis rotations.
package pca_test

import (
	"context"
	"testing"

	"github.com/lvlath/go/pca"
)

func TestPCA_RepeatedEigenspaceProjector(t *testing.T) {
	x := fixture(t, [][]float64{{-1, -1}, {-1, 1}, {1, -1}, {1, 1}})
	result, err := pca.Fit(context.Background(), x, pca.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Rank != 2 {
		t.Fatalf("isotropic spectrum rank: %d", result.Rank)
	}
	components := result.Model.Components()
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			var projector float64
			for h := 0; h < 2; h++ {
				projector += mustCell(t, components, i, h) * mustCell(t, components, j, h)
			}
			want := 0.0
			if i == j {
				want = 1
			}
			mustNear(t, projector, want, 1e-11)
		}
	}
}
func TestPCA_DiscardedVarianceResidual(t *testing.T) {
	x := fixture(t, [][]float64{{-2, -1}, {-2, 1}, {2, -1}, {2, 1}})
	result, err := pca.Fit(context.Background(), x, pca.Options{Components: 1})
	if err != nil {
		t.Fatal(err)
	}
	if result.Rank != 2 || len(result.ExplainedVarianceRatio) != 1 {
		t.Fatal("rank/truncation")
	}
	mustNear(t, result.ExplainedVarianceRatio[0], 0.8, 1e-12)
	mustNear(t, result.ReconstructionResidual*result.ReconstructionResidual, 4, 1e-11)
}
