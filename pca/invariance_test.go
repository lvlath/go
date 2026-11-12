// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// These external invariance tests protect training order independence and
// inference batch partition invariance without inspecting PCA private state.
// They compare mathematical projectors rather than asserting a particular
// eigenvector basis in cases where orthogonal rotations would be legitimate.
package pca_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/pca"
)

// TestPCA_TrainingRowPermutationPreservesProjector compares two fits built
// from the same multiset of observations in different chronological orders.
// Stable covariance arithmetic may round differently across reductions;
// the correct oracle is the retained projector and spectrum, not byte equality
// of intermediate floating-point operations or arbitrary eigenvector signs.
func TestPCA_TrainingRowPermutationPreservesProjector(t *testing.T) {
	observations := [][]float64{
		{1.2, 4.3, -2.1}, {2.1, 3.7, -1.4}, {3.8, 6.4, 1.2},
		{4.7, 2.3, 0.8}, {5.3, 4.8, 2.5}, {6.6, 3.1, 1.1},
		{7.4, 6.9, -0.3}, {8.2, 5.2, 3.6}, {9.1, 3.9, 0.2},
	}
	permuted := make([][]float64, len(observations))
	for i := range observations {
		// Reverse the sample order without changing feature columns or values.
		permuted[i] = append([]float64(nil), observations[len(observations)-1-i]...)
	}

	ctx := context.Background()
	original, err := pca.FitWithComponents(ctx, fixture(t, observations), 2)
	if err != nil {
		t.Fatal(err)
	}
	reordered, err := pca.FitWithComponents(ctx, fixture(t, permuted), 2)
	if err != nil {
		t.Fatal(err)
	}
	mustEqualInt(t, original.Rank, reordered.Rank, "numerical rank after row permutation")
	mustRelativeNear(t, original.ReconstructionResidual, reordered.ReconstructionResidual, 1e-10, 1e-12)

	// A projected subspace is invariant to orthogonal changes of its basis.
	// Compare P=Vk*Vk^T, not the signs or individual coordinates of Vk.
	left := original.Model.Components()
	right := reordered.Model.Components()
	for feature := 0; feature < left.Rows(); feature++ {
		for other := 0; other < left.Rows(); other++ {
			var leftEntry, rightEntry float64
			for component := 0; component < left.Cols(); component++ {
				leftEntry = math.FMA(mustCell(t, left, feature, component), mustCell(t, left, other, component), leftEntry)
				rightEntry = math.FMA(mustCell(t, right, feature, component), mustCell(t, right, other, component), rightEntry)
			}
			mustRelativeNear(t, leftEntry, rightEntry, 1e-10, 1e-12)
		}
	}
	for component := range original.Eigenvalues {
		mustRelativeNear(t, original.Eigenvalues[component], reordered.Eigenvalues[component], 1e-10, 1e-12)
		mustRelativeNear(t, original.ExplainedVarianceRatio[component], reordered.ExplainedVarianceRatio[component], 1e-10, 1e-12)
	}
}

// TestPCA_TransformChunkingCannotRecenterHoldout exercises realistic streaming
// inference: projecting a whole holdout ledger or its individual rows must give
// bit-identical scores because the fitted origin never depends on batch size.
// This catches changes that accidentally re-fit means in Transform.
func TestPCA_TransformChunkingCannotRecenterHoldout(t *testing.T) {
	ctx := context.Background()
	training := fixture(t, [][]float64{
		{100, 2, 7}, {105, 3, 5}, {97, 5, 9}, {110, 8, 4},
		{99, 9, 3}, {103, 11, 6}, {111, 7, 12}, {94, 4, 1},
	})
	fitted, err := pca.FitWithComponents(ctx, training, 2)
	if err != nil {
		t.Fatal(err)
	}
	future := [][]float64{{120, 1, 5}, {95, 14, 3}, {130, 7, 11}, {103, 6, 8}}
	batched, err := fitted.Model.Transform(ctx, fixture(t, future))
	if err != nil {
		t.Fatal(err)
	}

	for row, observation := range future {
		// One-observation batches have different batch means from the full
		// holdout. Their scores must nevertheless be exactly identical.
		single, transformErr := fitted.Model.Transform(ctx, fixture(t, [][]float64{observation}))
		if transformErr != nil {
			t.Fatal(transformErr)
		}
		for component := 0; component < batched.Cols(); component++ {
			wholeScore := mustCell(t, batched, row, component)
			singleScore := mustCell(t, single, 0, component)
			if math.Float64bits(wholeScore) != math.Float64bits(singleScore) {
				t.Fatalf("row=%d component=%d: chunking changed frozen score %.17g to %.17g", row, component, wholeScore, singleScore)
			}
		}
	}
}
