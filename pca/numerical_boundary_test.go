// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pca_test defends finite-coordinate and spectral publication boundaries.
// Fixtures derive their expected behavior from exact binary64 geometry.
package pca_test

import (
	"context"
	"math"
	"math/big"
	"testing"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/pca"
)

// TestPCA_SmallHeldoutCoordinateSurvivesAnchorCancellation guards frozen centering.
// A large anchor and the opposite displacement must not erase a small observation
// merely because their exact sum cancels in the fitted center.
// Implementation:
//   - Stage 1: fit the symmetric one-feature population with exact center zero.
//   - Stage 2: transform heldout values across widely separated binary exponents.
//   - Stage 3: compare to the exact signed coordinates and reconstruct them.
//
// Behavior highlights: Both anchor signs and multiple magnitudes are exercised.
// Errors: Any loss of a representable coordinate or unexpected fit error fails.
// Determinism: One-dimensional canonical loading is positive one.
// Complexity: O(n+nnew) per small case, without timing assumptions.
// AI-Hints: A correct training residual alone cannot validate small holdout scores.
func TestPCA_SmallHeldoutCoordinateSurvivesAnchorCancellation(t *testing.T) {
	ctx := context.Background()
	// The sample variance 2*a*a is finite; the origin is exactly zero.
	a := math.Ldexp(1, 450)
	for _, tc := range []struct {
		name   string
		anchor float64
	}{
		{"positive_anchor", a},
		{"negative_anchor", -a},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fit, err := pca.FitDefault(ctx, fixture(t, [][]float64{{tc.anchor}, {-tc.anchor}}))
			if err != nil {
				t.Fatal(err)
			}
			mustEqualInt(t, fit.Rank, 1, "single varying feature")
			mustNear(t, mustCell(t, fit.Model.Components(), 0, 0), 1, 0)
			// Every value is a binary64 value exactly, and all are tiny compared
			// with the fitted anchor. The correct centered result is itself.
			// This last value retains a low bit that is lost in value-anchor,
			// but large enough that the recovered correction is well conditioned.
			compensated := math.Ldexp(1, 430) + math.Ldexp(1, 378)
			heldout := fixture(t, [][]float64{{-3}, {0}, {1}, {math.Ldexp(1, -100)}, {math.Ldexp(1, 200)}, {compensated}})
			scores, err := fit.Model.Transform(ctx, heldout)
			if err != nil {
				t.Fatal(err)
			}
			restored, err := fit.Model.InverseTransform(ctx, scores)
			if err != nil {
				t.Fatal(err)
			}
			for row := 0; row < heldout.Rows(); row++ {
				want := mustCell(t, heldout, row, 0)
				mustNear(t, mustCell(t, scores, row, 0), want, 0)
				mustNear(t, mustCell(t, restored, row, 0), want, 0)
			}
		})
	}
}

// TestPCA_SmallHeldoutCoordinateSurvivesNonzeroCompositeCenter verifies low terms.
// The exact mean of [a,3,-a] is one, even though a rounded anchor displacement
// loses that unit unless its low component participates in the same affine sum.
// Implementation: Fit once and compare exact holdout scores and inverse values.
// Behavior highlights: The nonzero low center term must not be discarded.
// Errors: All expected values are finite and exactly representable.
// Determinism: One feature, canonical loading +1, fixed binary anchor.
// Complexity: O(n+nnew) for the one-dimensional case.
// AI-Hints: Do not compare the output only against the implementation's own mean.
func TestPCA_SmallHeldoutCoordinateSurvivesNonzeroCompositeCenter(t *testing.T) {
	ctx := context.Background()
	a := math.Ldexp(1, 450)
	fit, err := pca.FitDefault(ctx, fixture(t, [][]float64{{a}, {3}, {-a}}))
	if err != nil {
		t.Fatal(err)
	}
	heldout := fixture(t, [][]float64{{0}, {1}, {2}})
	scores, err := fit.Model.Transform(ctx, heldout)
	if err != nil {
		t.Fatal(err)
	}
	restored, err := fit.Model.InverseTransform(ctx, scores)
	if err != nil {
		t.Fatal(err)
	}
	for row, want := range []float64{-1, 0, 1} {
		mustNear(t, mustCell(t, scores, row, 0), want, 0)
		mustNear(t, mustCell(t, restored, row, 0), want+1, 0)
	}
}

// TestPCA_StandardizedTinyHoldoutSurvivesPreparedCenterCancellation checks scaling.
// With training [-1,+1], the population scale is exactly one and both fitted
// centers are zero; a sub-ULP heldout observation therefore must survive.
// Implementation: Compare the positive and negative tiny heldout values directly.
// Behavior highlights: Covers the two-stage standardized PCA coordinate system.
// Errors: A successful score of zero is incorrect for a nonzero observation.
// Determinism: Exact powers of two and one positive canonical loading.
// Complexity: O(n+nnew) for the single feature.
// AI-Hints: Standardization does not eliminate the need for stable PCA centering.
func TestPCA_StandardizedTinyHoldoutSurvivesPreparedCenterCancellation(t *testing.T) {
	ctx := context.Background()
	fit, err := pca.FitStandardized(ctx, fixture(t, [][]float64{{-1}, {1}}))
	if err != nil {
		t.Fatal(err)
	}
	tiny := math.Ldexp(1, -100)
	heldout := fixture(t, [][]float64{{tiny}, {-tiny}})
	scores, err := fit.Model.Transform(ctx, heldout)
	if err != nil {
		t.Fatal(err)
	}
	for row, want := range []float64{tiny, -tiny} {
		mustNear(t, mustCell(t, scores, row, 0), want, 0)
	}
	inverse, err := fit.Model.InverseTransform(ctx, scores)
	if err != nil {
		t.Fatal(err)
	}
	for row, want := range []float64{tiny, -tiny} {
		mustNear(t, mustCell(t, inverse, row, 0), want, 0)
	}
}

// TestPCA_InverseRejectsNonfinitePreparedProjection guards rational restoration.
// Finite scores along two retained directions can have an unrepresentable
// physical projection even before the frozen center is restored.
// Implementation:
//   - Stage 1: fit a two-dimensional covariance with rotated eigenvectors.
//   - Stage 2: align finite maximum scores to add in the first loading row.
//   - Stage 3: prove the projection overflows using an independent exact dot
//     product and require nil output with ErrOverflow.
//
// Behavior highlights: No NaN or infinity is supplied by the caller.
// Errors: ErrOverflow, never a successful value fabricated by rational conversion.
// Determinism: Scores use the actual canonical loading signs; no assumed sign tie.
// Complexity: O(d*k) for the projection oracle on this fixed 2x2 basis.
// AI-Hints: big.Rat.SetFloat64 cannot represent Inf; ignoring its nil return
// silently changes the numerical expression rather than rescuing it.
func TestPCA_InverseRejectsNonfinitePreparedProjection(t *testing.T) {
	ctx := context.Background()
	fit, err := pca.FitDefault(ctx, fixture(t, [][]float64{{2, 2}, {-2, -2}, {1, -1}, {-1, 1}}))
	if err != nil {
		t.Fatal(err)
	}
	mustEqualInt(t, fit.Rank, 2, "rotated full-rank training basis")
	loadings := fit.Model.Components()
	// The first feature has two loadings of magnitude approximately 1/sqrt(2).
	// Align the scores without depending on sign choices in the second eigenvector.
	row := []float64{
		math.Copysign(math.MaxFloat64, mustCell(t, loadings, 0, 0)),
		math.Copysign(math.MaxFloat64, mustCell(t, loadings, 0, 1)),
	}
	// Prove overflow from the actual finite operands before calling InverseTransform.
	// The exact sum is approximately sqrt(2)*MaxFloat64, not a canceled product.
	var sum, product, score, loading big.Rat
	for component, value := range row {
		score.SetFloat64(value)
		loading.SetFloat64(mustCell(t, loadings, 0, component))
		product.Mul(&score, &loading)
		sum.Add(&sum, &product)
	}
	projected, _ := sum.Float64()
	if !math.IsInf(projected, 1) {
		t.Fatalf("invalid overflow oracle: projection=%g", projected)
	}
	out, err := fit.Model.InverseTransform(ctx, fixture(t, [][]float64{row}))
	if out != nil {
		t.Fatal("unrepresentable inverse projection produced a matrix")
	}
	mustErrorIs(t, err, pca.ErrOverflow)
}

// TestPCA_HeldoutPreprocessingOverflowClassification checks two distinct stages.
// A valid fitted covariance does not guarantee all finite future coordinates
// have representable centered or population-scaled values.
// Implementation: Fit finite models, then supply finite out-of-range observations.
// Errors: ErrOverflow rather than ErrNonFinite, with no score matrix published.
// Determinism: Fixed finite extreme values; no allocation-pressure tests.
// Complexity: O(n*d*k) for the small cases.
// AI-Hints: Distinguish a failed centered subtraction from a failed scaled ratio.
func TestPCA_HeldoutPreprocessingOverflowClassification(t *testing.T) {
	ctx := context.Background()
	t.Run("constant_feature_opposite_extreme", func(t *testing.T) {
		// A large constant channel has zero covariance; another feature varies.
		// Fit remains representable, but the new constant-channel delta is -2*Max.
		maxValue := math.MaxFloat64
		train := fixture(t, [][]float64{{maxValue, -1}, {maxValue, 0}, {maxValue, 1}})
		fit, err := pca.FitDefault(ctx, train)
		if err != nil {
			t.Fatal(err)
		}
		out, err := fit.Model.Transform(ctx, fixture(t, [][]float64{{-maxValue, 0}}))
		if out != nil {
			t.Fatal("overflowed prepared coordinate produced a matrix")
		}
		mustErrorIs(t, err, pca.ErrOverflow)
	})
	t.Run("population_scale_quotient", func(t *testing.T) {
		// The training scale is small but nonzero; the standardized holdout
		// quotient, not its finite original input, is unrepresentable.
		a := math.Ldexp(1, -300)
		fit, err := pca.FitStandardized(ctx, fixture(t, [][]float64{{-a}, {a}}))
		if err != nil {
			t.Fatal(err)
		}
		out, err := fit.Model.Transform(ctx, fixture(t, [][]float64{{math.MaxFloat64}}))
		if out != nil {
			t.Fatal("overflowed scale quotient produced a matrix")
		}
		mustErrorIs(t, err, pca.ErrOverflow)
	})
}

// TestPCA_EigenToleranceExtremeProductAndUnderflow exercises scaled tolerances.
// The stop threshold must be representable without silently accepting bad spectra.
// Implementation: Use exact diagonal and correlated covariance fixtures.
// Behavior highlights: A rounded-to-zero tolerance is floored; an overflowing
// product is rejected rather than used as a vacuous convergence criterion.
// Errors: ErrOverflow only for the unrepresentable scaled tolerance product.
// Determinism: Both covariance matrices have exact binary64 entries.
// Complexity: Fixed one- and three-feature fits.
// AI-Hints: The tolerance is relative to the scaled covariance infinity norm.
func TestPCA_EigenToleranceExtremeProductAndUnderflow(t *testing.T) {
	ctx := context.Background()
	t.Run("underflow_to_zero", func(t *testing.T) {
		// Sample covariance is 1; normalization gives 0.5. Multiplying by
		// the smallest positive float rounds to zero (nearest-even tie).
		x := fixture(t, [][]float64{{-1}, {0}, {1}})
		fit, err := pca.Fit(ctx, x, pca.Options{EigenTolerance: math.SmallestNonzeroFloat64})
		if err != nil {
			t.Fatal(err)
		}
		mustEqualInt(t, fit.Rank, 1, "diagonal rank")
		mustNear(t, fit.Eigenvalues[0], 1, 0)
	})
	t.Run("overflow_product", func(t *testing.T) {
		// The scaled covariance has every entry 0.5 and row sum 1.5.
		// MaxFloat64*1.5 cannot be materialized as a finite stopping threshold.
		x := fixture(t, [][]float64{{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}})
		fit, err := pca.Fit(ctx, x, pca.Options{EigenTolerance: math.MaxFloat64})
		if fit != nil {
			t.Fatal("unrepresentable tolerance admitted a fitted model")
		}
		mustErrorIs(t, err, pca.ErrOverflow)
	})
}

// TestPCA_SubnormalCovariancePublicationGuards checks quantized sample covariance.
// Rounding each covariance cell independently can destroy PSD, inflate rank,
// or leave a positive normalized eigenvalue that underflows during publication.
// Implementation:
//   - Stage 1: create power-of-two observations with analytically known covariance.
//   - Stage 2: verify the dependency's published cells against an exact binary oracle.
//   - Stage 3: require the applicable PCA rejection, never a repaired fake model.
//
// Behavior highlights: Real finite input data, no forged covariance or solver.
// Errors: ErrNumerical for PSD/rank violations; ErrOverflow for a retained
// positive mode whose restored eigenvalue rounds to zero.
// Determinism: Exact rational coefficients separated from rounding boundaries.
// Complexity: Constant-size covariance and eigensolver fixtures.
// AI-Hints: A covariance matrix is PSD in real arithmetic, but its rounded
// float64 representation is not automatically PSD or rank preserving.
func TestPCA_SubnormalCovariancePublicationGuards(t *testing.T) {
	ctx := context.Background()
	// q*q equals the smallest positive float64 in exact arithmetic.
	q := math.Ldexp(1, -537)
	m := math.SmallestNonzeroFloat64
	cases := []struct {
		name string
		rows [][]float64
		want error
		cov  [4]float64
	}{
		{
			// Exact covariance entries in units of m: 1/4, 5/8, 25/16.
			// Independently rounded entries become [0,1;1,2]; det < 0.
			name: "material_negative_eigenvalue",
			rows: [][]float64{{-q / 2, -5 * q / 4}, {0, 0}, {q / 2, 5 * q / 4}},
			want: pca.ErrNumerical,
			cov:  [4]float64{0, m, m, 2 * m},
		},
		{
			// Exact two-row covariance entries in units of m: 2, 9/4, 81/32.
			// Rounded [2,2;2,3] has determinant 2 and rank 2, exceeding n-1=1.
			name: "rank_exceeds_centered_sample_bound",
			rows: [][]float64{{-q, -9 * q / 8}, {q, 9 * q / 8}},
			want: pca.ErrNumerical,
			cov:  [4]float64{2 * m, 2 * m, 2 * m, 3 * m},
		},
		{
			// Four symmetric rows give [25/24,25/24;25/24,25/12]*m.
			// Rounded [1,1;1,2] has lambdaMin=(3-sqrt(5))/2*m < m/2.
			// Both modes pass the default relative rank threshold, so publishing
			// a zero retained eigenvalue would violate the fitted rank contract.
			name: "retained_eigenvalue_underflows_on_restore",
			rows: [][]float64{{5 * q / 4, 5 * q / 4}, {-5 * q / 4, -5 * q / 4}, {0, 5 * q / 4}, {0, -5 * q / 4}},
			want: pca.ErrOverflow,
			cov:  [4]float64{m, m, m, 2 * m},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			x := fixture(t, tc.rows)
			// Assert the dependency boundary separately. If covariance changes,
			// the test must not accidentally pass for an unrelated PCA error.
			covariance, _, err := matrix.Covariance(x)
			if err != nil {
				t.Fatal(err)
			}
			for row := 0; row < 2; row++ {
				for col := 0; col < 2; col++ {
					mustNear(t, mustCell(t, covariance, row, col), tc.cov[row*2+col], 0)
				}
			}
			fit, err := pca.FitDefault(ctx, x)
			if fit != nil {
				t.Fatal("invalid or unpublishable spectrum produced a model")
			}
			mustErrorIs(t, err, tc.want)
		})
	}
}
