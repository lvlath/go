// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Independent rational covariance oracle: does not call production matrix.Covariance.
package pca_test

import (
	"context"
	"math"
	"math/big"
	"testing"

	"github.com/lvlath/go/pca"
)

// exactCovariance2D treats each input float64 as its exact binary rational,
// accumulates the column means with rational arithmetic, and divides by n-1.
// It intentionally does not depend on the matrix or PCA covariance kernels.
func exactCovariance2D(rows [][2]float64) [3]float64 {
	n := int64(len(rows))
	var means [2]*big.Rat
	for j := 0; j < 2; j++ {
		means[j] = new(big.Rat)
		for _, row := range rows {
			means[j].Add(means[j], new(big.Rat).SetFloat64(row[j]))
		}
		means[j].Quo(means[j], big.NewRat(n, 1))
	}
	var c [3]float64
	for pair, cols := range [][2]int{{0, 0}, {0, 1}, {1, 1}} {
		sum := new(big.Rat)
		for _, row := range rows {
			a := new(big.Rat).Sub(new(big.Rat).SetFloat64(row[cols[0]]), means[cols[0]])
			b := new(big.Rat).Sub(new(big.Rat).SetFloat64(row[cols[1]]), means[cols[1]])
			sum.Add(sum, new(big.Rat).Mul(a, b))
		}
		sum.Quo(sum, big.NewRat(n-1, 1))
		c[pair], _ = sum.Float64()
	}
	return c
}

func TestPCA_TwoByTwoExactCovarianceSpectralOracle(t *testing.T) {
	rows := [][2]float64{{1, 0}, {2, 5}, {6, 3}, {8, 9}, {13, 4}, {21, 14}, {34, 7}}
	xRows := make([][]float64, len(rows))
	for i, row := range rows {
		xRows[i] = []float64{row[0], row[1]}
	}
	fit, err := pca.Fit(context.Background(), fixture(t, xRows), pca.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if fit.Rank != 2 || len(fit.Eigenvalues) != 2 {
		t.Fatal("expected rank two")
	}
	c := exactCovariance2D(rows)
	trace := c[0] + c[2]
	gap := math.Hypot(c[0]-c[2], 2*c[1])
	wantHigh, wantLow := (trace+gap)/2, (trace-gap)/2
	mustNear(t, fit.Eigenvalues[0], wantHigh, 1e-10)
	mustNear(t, fit.Eigenvalues[1], wantLow, 1e-10)
	mustNear(t, fit.ExplainedVarianceRatio[0], wantHigh/trace, 1e-12)
	mustNear(t, fit.ReconstructionResidual, 0, 1e-10)
}
func TestPCA_FeaturePermutationProjector(t *testing.T) {
	a := fixture(t, [][]float64{{0, 5, 1}, {2, 1, 3}, {4, 8, 4}, {6, 2, 2}, {8, 9, 7}, {10, 3, 6}, {12, 4, 2}})
	b := fixture(t, [][]float64{{1, 0, 5}, {3, 2, 1}, {4, 4, 8}, {2, 6, 2}, {7, 8, 9}, {6, 10, 3}, {2, 12, 4}})
	fitA, err := pca.Fit(context.Background(), a, pca.Options{Components: 2})
	if err != nil {
		t.Fatal(err)
	}
	fitB, err := pca.Fit(context.Background(), b, pca.Options{Components: 2})
	if err != nil {
		t.Fatal(err)
	}
	ca, cb := fitA.Model.Components(), fitB.Model.Components()
	permutation := []int{1, 2, 0}
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			var pa, pb float64
			for k := 0; k < 2; k++ {
				pa += mustCell(t, ca, i, k) * mustCell(t, ca, j, k)
				pb += mustCell(t, cb, permutation[i], k) * mustCell(t, cb, permutation[j], k)
			}
			mustNear(t, pa, pb, 1e-10)
		}
	}
	mustNear(t, fitA.ReconstructionResidual, fitB.ReconstructionResidual, 1e-10)
}
func TestPCA_RankToleranceExplicit(t *testing.T) {
	x := fixture(t, [][]float64{{-2, -1}, {-2, 1}, {2, -1}, {2, 1}})
	fit, err := pca.Fit(context.Background(), x, pca.Options{RankTolerance: .30})
	if err != nil {
		t.Fatal(err)
	}
	if fit.Rank != 1 {
		t.Fatalf("expected spectrum rank 1 under 30%% relative cutoff, got %d", fit.Rank)
	}
	_, err = pca.Fit(context.Background(), x, pca.Options{Components: 2, RankTolerance: .30})
	mustErrorIs(t, err, pca.ErrRankDeficient)
	_, err = pca.Fit(context.Background(), x, pca.Options{RankTolerance: 1})
	mustErrorIs(t, err, pca.ErrRankDeficient)
}
