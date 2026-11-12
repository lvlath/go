// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// These external tests anchor dependency error identities without substituting
// another covariance or eigenvalue implementation inside the production package.
package pca_test

import (
	"context"
	"errors"
	"math"
	"sync"
	"testing"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/pca"
	"github.com/lvlath/go/stats"
)

// --- inference ---

func TestPCA_InverseBuffersOneScoreRow(t *testing.T) {
	ctx := context.Background()
	train := fixture(t, [][]float64{
		{1, 2, 3, 4}, {2, 3, 2, 1}, {3, 5, 7, 2}, {7, 4, 3, 9},
		{8, 2, 6, 5}, {4, 9, 5, 3}, {9, 6, 8, 7},
	})
	fit, err := pca.FitWithComponents(ctx, train, 2)
	if err != nil {
		t.Fatal(err)
	}
	scores := fixture(t, [][]float64{{0.2, -0.4}, {1.0, 2.0}, {-1.2, 0.5}})
	backend := &countReadsMatrix{Matrix: scores}
	reconstructed, err := fit.Model.InverseTransform(ctx, backend)
	if err != nil {
		t.Fatal(err)
	}
	if reconstructed.Rows() != scores.Rows() || reconstructed.Cols() != train.Cols() {
		t.Fatal("incorrect inverse shape")
	}
	// The public contract performs one complete nonfinite scan and one buffered
	// read per score row. The old implementation read every score again for
	// every original feature, scaling At dispatches with d*k.
	mustEqualInt(t, backend.reads, 2*scores.Rows()*scores.Cols(), "score read count")
	mustNear(t, mustCell(t, scores, 0, 0), 0.2, 0)
}

func TestPCA_HeldoutInputsAndFittedWorkspaceRemainDetached(t *testing.T) {
	ctx := context.Background()
	training := fixture(t, [][]float64{{3, 10, 7}, {5, 11, 9}, {9, 15, 1}, {4, 17, 6}, {8, 8, 3}, {11, 12, 2}})
	inputBackup := training.Clone()
	fit, err := pca.FitDefault(ctx, training)
	if err != nil {
		t.Fatal(err)
	}
	heldout := fixture(t, [][]float64{{12, 20, 8}, {6, 9, 2}})
	heldoutBackup := heldout.Clone()
	first, err := fit.Model.Transform(ctx, heldout)
	if err != nil {
		t.Fatal(err)
	}
	if err = first.Set(0, 0, -999); err != nil {
		t.Fatal(err)
	}
	second, err := fit.Model.Transform(ctx, heldout)
	if err != nil {
		t.Fatal(err)
	}
	if mustCell(t, second, 0, 0) == -999 {
		t.Fatal("Model shares a buffer with an earlier Transform result")
	}
	for row := 0; row < training.Rows(); row++ {
		for col := 0; col < training.Cols(); col++ {
			mustNear(t, mustCell(t, training, row, col), mustCell(t, inputBackup, row, col), 0)
		}
	}
	for row := 0; row < heldout.Rows(); row++ {
		for col := 0; col < heldout.Cols(); col++ {
			mustNear(t, mustCell(t, heldout, row, col), mustCell(t, heldoutBackup, row, col), 0)
		}
	}
}

func TestPCA_ConcurrentReconstructRace(t *testing.T) {
	ctx := context.Background()
	training := fixture(t, [][]float64{{1, 4, 2}, {3, 2, 1}, {4, 1, 9}, {9, 5, 3}, {8, 6, 7}, {6, 2, 8}, {4, 9, 1}})
	fit, err := pca.FitWithComponents(ctx, training, 2)
	if err != nil {
		t.Fatal(err)
	}
	heldout := fixture(t, [][]float64{{2, 3, 4}, {4, 5, 6}, {3, 8, 1}})
	reference, err := fit.Model.Reconstruct(ctx, heldout)
	if err != nil {
		t.Fatal(err)
	}
	const workers = 96
	results := make(chan error, workers)
	var wg sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			reconstructed, callErr := fit.Model.Reconstruct(ctx, heldout)
			if callErr == nil {
				for row := 0; row < heldout.Rows(); row++ {
					for col := 0; col < heldout.Cols(); col++ {
						got, err1 := reconstructed.At(row, col)
						want, err2 := reference.At(row, col)
						if err1 != nil || err2 != nil {
							callErr = errors.Join(err1, err2)
							break
						}
						if got != want {
							callErr = errors.New("non-deterministic concurrent result")
							break
						}
					}
				}
			}
			results <- callErr
		}()
	}
	wg.Wait()
	close(results)
	for err = range results {
		if err != nil {
			t.Fatal(err) // The owner test goroutine alone calls Fatal.
		}
	}
}

func TestPCA_TransformRejectsRepresentableInputsWithOverflowingScore(t *testing.T) {
	ctx := context.Background()
	training := fixture(t, [][]float64{{-1, -1}, {0, 0}, {1, 1}})
	fit, err := pca.FitDefault(ctx, training)
	if err != nil {
		t.Fatal(err)
	}
	// Both values are finite binary64 but their positively correlated
	// normalized projection exceeds MaxFloat64. No Inf score may escape.
	future := fixture(t, [][]float64{{1.6e308, 1.6e308}})
	result, err := fit.Model.Transform(ctx, future)
	mustErrorIs(t, err, pca.ErrOverflow)
	if result != nil {
		t.Fatal("partial score matrix")
	}
}

func TestPCA_InverseRejectsUnrepresentablePhysicalUnits(t *testing.T) {
	ctx := context.Background()
	train := fixture(t, [][]float64{{-2, -1}, {0, 0}, {2, 1}})
	fit, err := pca.FitStandardizedWithComponents(ctx, train, 1)
	if err != nil {
		t.Fatal(err)
	}
	// A finite score is not a guarantee that inverse scaling can be represented.
	hugeScore := fixture(t, [][]float64{{math.MaxFloat64}})
	inverse, err := fit.Model.InverseTransform(ctx, hugeScore)
	mustErrorIs(t, err, pca.ErrOverflow)
	if inverse != nil {
		t.Fatal("partial inverse after numeric overflow")
	}
}

func TestPCA_CancellationAndBackendFailureDoNotPublishInference(t *testing.T) {
	ctx := context.Background()
	train := fixture(t, [][]float64{{-2, 5}, {0, 0}, {2, 3}, {7, 4}})
	fit, err := pca.FitWithComponents(ctx, train, 1)
	if err != nil {
		t.Fatal(err)
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if got, failure := fit.Model.Transform(canceled, train); got != nil || !errors.Is(failure, context.Canceled) {
		t.Fatalf("cancellation must return nil and context error: result=%v error=%v", got, failure)
	}
	if got, failure := fit.Model.Reconstruct(canceled, train); got != nil || !errors.Is(failure, context.Canceled) {
		t.Fatalf("canceled reconstruct must not publish: result=%v error=%v", got, failure)
	}
	// A valid backend can fail on one read (for instance, a storage I/O error).
	// Its original sentinel identity must survive the public facade unchanged.
	backendFailure := errors.New("instrument buffer read failure")
	brokenInput := &adversarialMatrix{Matrix: train, badRow: 1, badCol: 0, err: backendFailure}
	result, err := fit.Model.Transform(ctx, brokenInput)
	if result != nil || !errors.Is(err, backendFailure) {
		t.Fatalf("Transform backend error lost: %v", err)
	}
	score := fixture(t, [][]float64{{1}, {2}})
	brokenScores := &adversarialMatrix{Matrix: score, badRow: 1, badCol: 0, err: backendFailure}
	result, err = fit.Model.InverseTransform(ctx, brokenScores)
	if result != nil || !errors.Is(err, backendFailure) {
		t.Fatalf("InverseTransform backend error lost: %v", err)
	}
}

// --- dependency ---

func TestPCA_UnrepresentableCovariancePreservesMatrixCause(t *testing.T) {
	ctx := context.Background()
	// The two supplied observations are finite; their sample variance is
	// 2*MaxFloat64^2, which cannot be materialized as binary64 covariance.
	// A robust dependency must report arithmetic overflow, not input NaN/Inf.
	x := fixture(t, [][]float64{{-math.MaxFloat64}, {math.MaxFloat64}})
	fit, err := pca.FitDefault(ctx, x)
	mustErrorIs(t, err, pca.ErrOverflow)
	mustErrorIs(t, err, matrix.ErrOverflow)
	if fit != nil {
		t.Fatal("unrepresentable covariance produced a fitted result")
	}
}

func TestPCA_StandardizeRejectsConstantAndRetainsStatsCause(t *testing.T) {
	ctx := context.Background()
	// Nonstandardized PCA would accept the second constant channel when the
	// first varies; population standardization must refuse its exact zero M2.
	x := fixture(t, [][]float64{{-4, 9}, {-1, 9}, {2, 9}, {5, 9}})
	unscaled, err := pca.FitDefault(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	mustEqualInt(t, unscaled.Rank, 1, "unscaled constant channel allowed")
	standardized, err := pca.FitStandardized(ctx, x)
	mustErrorIs(t, err, pca.ErrZeroVariance)
	mustErrorIs(t, err, stats.ErrZeroVariance)
	if standardized != nil {
		t.Fatal("standardization published model for constant feature")
	}
}

func TestPCA_BackendErrorPreservesOriginalIdentity(t *testing.T) {
	ctx := context.Background()
	x := fixture(t, [][]float64{{1, 4}, {2, 5}, {3, 7}})
	root := errors.New("sensor archive unavailable")
	broken := &adversarialMatrix{Matrix: x, badRow: 2, badCol: 1, err: root}
	fit, err := pca.FitDefault(ctx, broken)
	if fit != nil || !errors.Is(err, root) {
		t.Fatalf("backend failure identity lost: result=%v error=%v", fit, err)
	}
}

// --- numeric ---

func TestPCA_RelativeEigenToleranceIsScaleEquivariant(t *testing.T) {
	ctx := context.Background()
	base := [][]float64{
		{-3, 2, 1}, {-1, 5, 2}, {0, 4, 6}, {2, 1, 5},
		{5, 8, 2}, {4, 7, 9}, {-2, -1, 3}, {7, 0, 4},
	}
	const tolerance = 1e-10
	options := pca.Options{Components: 2, EigenTolerance: tolerance}
	reference, err := pca.Fit(ctx, fixture(t, base), options)
	if err != nil {
		t.Fatal(err)
	}
	referenceComponents := reference.Model.Components()
	for _, scale := range []float64{1e-7, 1e-4, 1e4, 1e7} {
		t.Run(fmtScaleForTest(scale), func(t *testing.T) {
			scaled := make([][]float64, len(base))
			for row, values := range base {
				scaled[row] = make([]float64, len(values))
				for col, value := range values {
					scaled[row][col] = value * scale
				}
			}
			fit, failure := pca.Fit(ctx, fixture(t, scaled), options)
			if failure != nil {
				t.Fatal(failure)
			}
			mustEqualInt(t, fit.Rank, reference.Rank, "scale-invariant eigen rank")
			for component := range reference.Eigenvalues {
				mustRelativeNear(t, fit.Eigenvalues[component]/(scale*scale), reference.Eigenvalues[component], 2e-8, 1e-11)
				mustRelativeNear(t, fit.ExplainedVarianceRatio[component], reference.ExplainedVarianceRatio[component], 2e-8, 1e-11)
			}
			// Different but valid sign/basis orientations must agree on the
			// retained two-dimensional projector whenever the eigengap is clear.
			currentComponents := fit.Model.Components()
			for i := 0; i < referenceComponents.Rows(); i++ {
				for j := 0; j < referenceComponents.Rows(); j++ {
					var expected, actual float64
					for axis := 0; axis < referenceComponents.Cols(); axis++ {
						expected = math.FMA(mustCell(t, referenceComponents, i, axis), mustCell(t, referenceComponents, j, axis), expected)
						actual = math.FMA(mustCell(t, currentComponents, i, axis), mustCell(t, currentComponents, j, axis), actual)
					}
					mustRelativeNear(t, actual, expected, 2e-8, 1e-10)
				}
			}
			mustRelativeNear(t, fit.ReconstructionResidual/scale, reference.ReconstructionResidual, 2e-8, 1e-11)
		})
	}
}

// fmtScaleForTest names orders of magnitude without relying on log10 roundoff.
func fmtScaleForTest(scale float64) string {
	switch scale {
	case 1e-7:
		return "small_1e-7"
	case 1e-4:
		return "small_1e-4"
	case 1e4:
		return "large_1e4"
	default:
		return "large_1e7"
	}
}

func TestPCA_ProjectorIsSymmetricAndIdempotent(t *testing.T) {
	ctx := context.Background()
	observations := fixture(t, [][]float64{
		{1, 9, 2, 3}, {2, 7, 5, 4}, {7, 5, 4, 1}, {9, 3, 2, 8},
		{5, 6, 8, 3}, {3, 2, 1, 7}, {8, 4, 7, 5}, {4, 1, 3, 9},
	})
	fit, err := pca.FitWithComponents(ctx, observations, 2)
	if err != nil {
		t.Fatal(err)
	}
	basis := fit.Model.Components()
	const features = 4
	var projector [features][features]float64
	for row := 0; row < features; row++ {
		for col := 0; col < features; col++ {
			for axis := 0; axis < basis.Cols(); axis++ {
				projector[row][col] = math.FMA(mustCell(t, basis, row, axis), mustCell(t, basis, col, axis), projector[row][col])
			}
		}
	}
	for row := 0; row < features; row++ {
		for col := 0; col < features; col++ {
			mustNear(t, projector[row][col], projector[col][row], 1e-12)
			var square float64
			for coordinate := 0; coordinate < features; coordinate++ {
				square = math.FMA(projector[row][coordinate], projector[coordinate][col], square)
			}
			mustNear(t, square, projector[row][col], 1e-10)
		}
	}
}

func TestPCA_StandardizedResidualMatchesIndependentTailEnergy(t *testing.T) {
	ctx := context.Background()
	// Both features have population variance one after scaling, but their
	// physical units differ by two orders of magnitude and are independent.
	x := fixture(t, [][]float64{{-2, -300}, {-2, 300}, {2, -300}, {2, 300}})
	fit, err := pca.FitStandardizedWithComponents(ctx, x, 1)
	if err != nil {
		t.Fatal(err)
	}
	if fit.Rank != 2 {
		t.Fatalf("expected full standardized rank 2, got %d", fit.Rank)
	}
	mustNear(t, fit.ExplainedVarianceRatio[0], .5, 1e-12)
	// Total centered standardized energy is 8. Exactly one of two equal
	// orthogonal components is retained, so the optimization-space loss is 4.
	mustNear(t, fit.ReconstructionResidual*fit.ReconstructionResidual, 4, 1e-10)
	// A physical-unit reconstruction error is generally much larger; it must
	// not be accidentally stored in the standardized FitResult diagnostic.
	restored, err := fit.Model.Reconstruct(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	var physicalSSE float64
	for row := 0; row < x.Rows(); row++ {
		for col := 0; col < x.Cols(); col++ {
			delta := mustCell(t, x, row, col) - mustCell(t, restored, row, col)
			physicalSSE = math.FMA(delta, delta, physicalSSE)
		}
	}
	if physicalSSE <= 100 {
		t.Fatalf("expected physical-unit residual to differ from scaled one: %.9g", physicalSSE)
	}
}

func TestPCA_CanonicalLoadingSignsOnRepeatedDeterministicFits(t *testing.T) {
	ctx := context.Background()
	x := fixture(t, [][]float64{
		{2, 5, 1}, {3, 1, 7}, {8, 2, 6}, {1, 9, 3}, {7, 4, 5}, {9, 3, 2}, {4, 7, 8},
	})
	first, err := pca.FitDefault(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	initial := first.Model.Components()
	for iteration := 0; iteration < 12; iteration++ {
		next, failure := pca.FitDefault(ctx, x)
		if failure != nil {
			t.Fatal(failure)
		}
		loading := next.Model.Components()
		for axis := 0; axis < loading.Cols(); axis++ {
			pivot, maximum := 0, 0.0
			for row := 0; row < loading.Rows(); row++ {
				value := mustCell(t, loading, row, axis)
				if absolute := math.Abs(value); absolute > maximum {
					maximum, pivot = absolute, row // Exact ties keep the lower row index.
				}
				mustNear(t, value, mustCell(t, initial, row, axis), 0)
			}
			if maximum <= 0 || mustCell(t, loading, pivot, axis) <= 0 {
				t.Fatalf("axis %d violates canonical positive max-abs coordinate", axis)
			}
		}
	}
}

func TestPCA_RankDeficiencyOfExactlyDuplicateFeatures(t *testing.T) {
	ctx := context.Background()
	// An exact duplicate channel contributes no independent centered direction.
	// This check relies on a fixed mathematical rank, not on a tuned epsilon.
	x := fixture(t, [][]float64{{-3, -3, 2}, {-1, -1, 1}, {0, 0, -2}, {1, 1, 0}, {3, 3, 4}})
	fit, err := pca.FitDefault(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	mustEqualInt(t, fit.Rank, 2, "duplicate column does not increase rank")
	if request, failure := pca.FitWithComponents(ctx, x, 3); request != nil || failure == nil {
		t.Fatalf("unattainable rank 3 must fail: result=%v err=%v", request, failure)
	} else {
		mustErrorIs(t, failure, pca.ErrRankDeficient)
	}
}

// TestPCA_ExtremeEigenToleranceCannotDisableEigenpairChecks protects the
// numerical verification gate when a finite Jacobi stopping threshold is so
// large that blindly reusing it would make the check infinite and vacuous.
// An independent accuracy cap must reject the unrotated false eigenpairs.
func TestPCA_ExtremeEigenToleranceCannotDisableEigenpairChecks(t *testing.T) {
	ctx := context.Background()
	// With this sample covariance C=[[1,1],[1,1]], ||C||_infinity=2.
	// EigenTolerance*||C|| is finite (1e308), while the verification
	// multiplier and sqrt(d) would overflow binary64 if the Jacobi stopping
	// tolerance were incorrectly reused as the spectral verification allowance.
	x := fixture(t, [][]float64{{-1, -1}, {0, 0}, {1, 1}})
	result, err := pca.Fit(ctx, x, pca.Options{EigenTolerance: 5e307})
	mustErrorIs(t, err, pca.ErrNumerical)
	if result != nil {
		t.Fatal("an extreme tolerance disabled independent eigenpair checks")
	}
}

// TestPCA_LooseEigenToleranceDoesNotCertifyIncorrectEigenpairs ensures that a
// permissive Jacobi stopping option cannot disable the independent eigenpair
// equation check, which is necessary to keep numerical rank reproducible.
func TestPCA_LooseEigenToleranceDoesNotCertifyIncorrectEigenpairs(t *testing.T) {
	ctx := context.Background()
	// Sample covariance is [[1,1],[1,1]], with true spectrum {2,0}.
	// An absolute pivot threshold above 1 makes bare Jacobi stop at the identity;
	// its diagonal entries {1,1} are not eigenvalues of this covariance.
	x := fixture(t, [][]float64{{-1, -1}, {0, 0}, {1, 1}})
	result, err := pca.Fit(ctx, x, pca.Options{EigenTolerance: 2})
	mustErrorIs(t, err, pca.ErrNumerical)
	if result != nil {
		t.Fatal("a loose stopping tolerance published false eigenvalues and rank")
	}
}
