// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pca_test verifies public PCA error atomicity and dependency identity.
// The probes exercise real kernels through Matrix and Context interfaces.
package pca_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/pca"
	"github.com/lvlath/go/stats"
)

// TestPCA_FitCancellationAtEveryCheckpoint defends cancellation during fitting.
// Both preprocessing policies must publish no result at any observed checkpoint.
// Implementation: Calibrate the successful path and cancel at each Err call.
// Behavior highlights: Includes covariance, center fitting, eigenspectrum checks,
// and residual construction, without requiring interruption inside EigenSym.
// Determinism: Real contexts, no timeouts or sleeps; all fixtures remain stable.
// Complexity: O(q*Tfit) for q observed checks on a fixed small training matrix.
// AI-Hints: Do not hard-code context check numbers or weaken late cancellation.
func TestPCA_FitCancellationAtEveryCheckpoint(t *testing.T) {
	// The six observations have unequal, nonconstant feature variances and
	// nontrivial cross-covariance, so the eigensolver cannot take a trivial path.
	train := fixture(t, [][]float64{
		{1, 4, 2}, {3, 1, 7}, {5, 6, 3},
		{2, 8, 9}, {8, 2, 5}, {6, 9, 1},
	})
	for _, tc := range []struct {
		name string
		opts pca.Options
	}{
		{"centered", pca.Options{Components: 2}},
		{"standardized", pca.Options{Components: 2, Standardize: true}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			verifyCancellationSweep(t, func(ctx context.Context) (bool, error) {
				result, err := pca.Fit(ctx, train, tc.opts)
				return result != nil, err
			})
		})
	}
}

// TestPCA_FitBackendFailureAtEveryRead protects all training passes.
// A backend can fail after input validation without mutating any data value.
// Implementation: Observe the actual read count and fail one access per run.
// Behavior highlights: Exercises generic covariance/statistics access, fitted
// center reads and training residual reads through the public Fit facade.
// Errors: The original backend cause must survive and the result must be nil.
// Determinism: Fresh single-call wrappers over the same immutable matrix.
// Complexity: O(q*Tfit), with bounded q for the small fixture.
// AI-Hints: A passing preflight is not permission to swallow a later At error.
func TestPCA_FitBackendFailureAtEveryRead(t *testing.T) {
	train := fixture(t, [][]float64{
		{1, 4, 2}, {3, 1, 7}, {5, 6, 3},
		{2, 8, 9}, {8, 2, 5}, {6, 9, 1},
	})
	for _, tc := range []struct {
		name string
		opts pca.Options
	}{
		{"centered", pca.Options{Components: 2}},
		{"standardized", pca.Options{Components: 2, Standardize: true}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			verifyReadFailureSweep(t, train, func(input matrix.Matrix) (bool, error) {
				result, err := pca.Fit(context.Background(), input, tc.opts)
				return result != nil, err
			})
		})
	}
}

// TestPCA_InferenceCancellationAtEveryCheckpoint checks both projection directions.
// Reconstruct must also remain atomic when cancellation occurs during its inverse leg.
// Implementation: Fit once; prepare stable input and scores; sweep public calls.
// Behavior highlights: The fitted model remains reusable after every rejected call.
// Determinism: Fixed data and a genuine cancelable context per invocation.
// Complexity: O(q*n*d*k) for each small inference fixture.
// AI-Hints: Cancellation is tested after admission, not only at method entry.
func TestPCA_InferenceCancellationAtEveryCheckpoint(t *testing.T) {
	ctx := context.Background()
	train := fixture(t, [][]float64{{1, 4, 2}, {3, 1, 7}, {5, 6, 3}, {2, 8, 9}})
	fit, err := pca.FitStandardizedWithComponents(ctx, train, 2)
	if err != nil {
		t.Fatal(err)
	}
	// Freeze an independent score batch for the inverse cancellation sweep.
	scores, err := fit.Model.Transform(ctx, train)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name  string
		input matrix.Matrix
		call  func(context.Context, matrix.Matrix) (matrix.Matrix, error)
	}{
		{"transform", train, fit.Model.Transform},
		{"inverse", scores, fit.Model.InverseTransform},
		{"reconstruct", train, fit.Model.Reconstruct},
	} {
		t.Run(tc.name, func(t *testing.T) {
			verifyCancellationSweep(t, func(callCtx context.Context) (bool, error) {
				result, callErr := tc.call(callCtx, tc.input)
				return result != nil, callErr
			})
			// A canceled call cannot damage the privately owned fitted state.
			result, callErr := tc.call(ctx, tc.input)
			if callErr != nil || result == nil {
				t.Fatalf("model reuse after cancellation: result=%v err=%v", result, callErr)
			}
		})
	}
}

// TestPCA_InferenceBackendFailureAtEveryRead checks late caller-data failures.
// Each operation must reject both prevalidation and computation-pass access errors.
// Implementation: Fit once and sweep all observed reads for both directions.
// Errors: Preserve the exact upstream cause and return no partially filled matrix.
// Determinism: No changes to the input values or fitted parameters between calls.
// Complexity: O(q*n*d*k) for q reads on the fixed fixture.
// AI-Hints: Buffering a row does not permit ignoring a failed Matrix.At call.
func TestPCA_InferenceBackendFailureAtEveryRead(t *testing.T) {
	ctx := context.Background()
	train := fixture(t, [][]float64{{1, 4, 2}, {3, 1, 7}, {5, 6, 3}, {2, 8, 9}})
	fit, err := pca.FitWithComponents(ctx, train, 2)
	if err != nil {
		t.Fatal(err)
	}
	scores, err := fit.Model.Transform(ctx, train)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name  string
		input matrix.Matrix
		call  func(context.Context, matrix.Matrix) (matrix.Matrix, error)
	}{
		{"transform", train, fit.Model.Transform},
		{"inverse", scores, fit.Model.InverseTransform},
		{"reconstruct", train, fit.Model.Reconstruct},
	} {
		t.Run(tc.name, func(t *testing.T) {
			verifyReadFailureSweep(t, tc.input, func(input matrix.Matrix) (bool, error) {
				result, callErr := tc.call(ctx, input)
				return result != nil, callErr
			})
		})
	}
}

// TestPCA_DependencyErrorIdentityAfterPreflight verifies the adapter protocol.
// A lazy Matrix backend can return a wrapped upstream matrix or stats sentinel.
// Implementation: Allow the entire finite scan, then fail the next backend read
// while Fit is delegated to the real covariance or standardizer implementation.
// Behavior highlights: Known causes acquire their PCA classification; opaque
// causes remain discoverable without inventing a replacement error identity.
// Errors: The wrapped upstream cause and applicable PCA sentinel match errors.Is.
// Determinism: Stable finite inputs, no mutation, one fault per invocation.
// Complexity: O(n*d) per early failure; no spectral computation is required.
// AI-Hints: This is fault propagation, not a claim that valid training data can
// make stats emit every sentinel under the fixed population policy.
func TestPCA_DependencyErrorIdentityAfterPreflight(t *testing.T) {
	train := fixture(t, [][]float64{{1, 4}, {3, 1}, {5, 6}})
	// Fit documents a complete finite scan before delegation. The first
	// subsequent read is allowed to fail in the external data source.
	failAt := train.Rows()*train.Cols() + 1
	opaque := errors.New("lazy feature source: unavailable observation")
	cases := []struct {
		name        string
		standardize bool
		cause       error
		want        error
	}{
		{"stats/zero_variance", true, stats.ErrZeroVariance, pca.ErrZeroVariance},
		{"stats/nonfinite", true, stats.ErrNonFinite, pca.ErrNonFinite},
		{"stats/overflow", true, stats.ErrOverflow, pca.ErrOverflow},
		{"stats/option", true, stats.ErrInvalidOption, pca.ErrInvalidOption},
		{"stats/shape", true, stats.ErrShape, pca.ErrShape},
		{"stats/insufficient", true, stats.ErrInsufficientSamples, pca.ErrShape},
		{"stats/nil", true, stats.ErrNilInput, pca.ErrNilInput},
		{"stats/empty", true, stats.ErrEmptyInput, pca.ErrEmptyInput},
		{"stats/opaque", true, opaque, nil},
		{"matrix/overflow", false, matrix.ErrOverflow, pca.ErrOverflow},
		{"matrix/nonfinite", false, matrix.ErrNaNInf, pca.ErrNonFinite},
		{"matrix/nil", false, matrix.ErrNilMatrix, pca.ErrNilInput},
		{"matrix/dimensions", false, matrix.ErrInvalidDimensions, pca.ErrShape},
		{"matrix/dimension_mismatch", false, matrix.ErrDimensionMismatch, pca.ErrShape},
		{"matrix/asymmetry", false, matrix.ErrAsymmetry, pca.ErrNumerical},
		{"matrix/opaque", false, opaque, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Wrap the upstream sentinel so an adapter cannot rely on == or text.
			cause := fmt.Errorf("lazy source read: %w", tc.cause)
			input := &readFaultMatrix{Matrix: train, failAt: failAt, cause: cause}
			result, err := pca.Fit(context.Background(), input, pca.Options{Standardize: tc.standardize})
			if result != nil {
				t.Fatal("failed dependency read published a partial FitResult")
			}
			mustErrorIs(t, err, cause)
			mustErrorIs(t, err, tc.cause)
			if tc.want != nil {
				mustErrorIs(t, err, tc.want)
			}
		})
	}
}

// TestPCA_StandardizerUnrepresentablePositiveScale exercises a real stats failure.
// The feature is not constant, but its positive population deviation rounds to zero.
// Implementation: Use one smallest-subnormal sample among four observations.
// Errors: Both stats.ErrOverflow and pca.ErrOverflow must be discoverable.
// Determinism: The exact deviation sqrt(3)/4 times the smallest subnormal is
// strictly below half of that subnormal, avoiding a rounding-tie assumption.
// Complexity: O(n) training preprocessing with n=4; no eigensolver work.
// AI-Hints: A nonzero mathematical variance must not be relabeled as a constant.
func TestPCA_StandardizerUnrepresentablePositiveScale(t *testing.T) {
	x := fixture(t, [][]float64{{0}, {0}, {math.SmallestNonzeroFloat64}, {0}})
	result, err := pca.FitStandardized(context.Background(), x)
	if result != nil {
		t.Fatal("unrepresentable positive scale published a model")
	}
	mustErrorIs(t, err, pca.ErrOverflow)
	mustErrorIs(t, err, stats.ErrOverflow)
}

// TestPCA_TypedNilViewAndInvalidDimensions checks public admission boundaries.
// Nil View and unrepresentable element counts must fail before cell access.
// Implementation: Fit a small rank-one model, then use typed nil and shape probes.
// Errors: ErrNilInput, ErrShape, or ErrOverflow; no partial result.
// Determinism: Architecture-relative math.MaxInt bounds; no giant allocation.
// Complexity: O(1) for each rejected input after fitting the small fixture.
// AI-Hints: Do not use reflect or construct a giant Dense to test checked sizes.
func TestPCA_TypedNilViewAndInvalidDimensions(t *testing.T) {
	ctx := context.Background()
	fit, err := pca.FitWithComponents(ctx, fixture(t, [][]float64{{-1, -1}, {0, 0}, {1, 1}}), 1)
	if err != nil {
		t.Fatal(err)
	}
	var nilView *matrix.View
	cases := []struct {
		name string
		x    matrix.Matrix
		want error
	}{
		{"nil_view", nilView, pca.ErrNilInput},
		{"negative_rows", phantomMatrix{rows: -1, cols: 2}, pca.ErrShape},
		{"negative_columns", phantomMatrix{rows: 2, cols: -1}, pca.ErrShape},
		{"element_count_overflow", phantomMatrix{rows: math.MaxInt/2 + 1, cols: 2}, pca.ErrOverflow},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result, fitErr := pca.FitDefault(ctx, tc.x)
			if result != nil {
				t.Fatal("invalid shape published a fit")
			}
			mustErrorIs(t, fitErr, tc.want)
			out, transformErr := fit.Model.Transform(ctx, tc.x)
			if out != nil {
				t.Fatal("invalid shape published a transform")
			}
			mustErrorIs(t, transformErr, tc.want)
		})
	}
	out, err := fit.Model.InverseTransform(ctx, nilView)
	if out != nil {
		t.Fatal("typed-nil View published an inverse")
	}
	mustErrorIs(t, err, pca.ErrNilInput)
}

// TestPCA_InverseOutputSizePrecedesScoreReads protects the widened result shape.
// A one-column score batch may be representable in bytes while its two-column
// reconstruction is not, so output size admission must happen before scanning.
// Implementation: Fit d=2,k=1; use a shape-only input with no readable cells.
// Errors: ErrOverflow, not an access error or an attempted huge allocation.
// Determinism: Uses math.MaxInt for the target architecture.
// Complexity: O(1) rejection after the fixed-size fit.
// AI-Hints: Check rows*d independently of the already admitted rows*k.
func TestPCA_InverseOutputSizePrecedesScoreReads(t *testing.T) {
	ctx := context.Background()
	fit, err := pca.FitWithComponents(ctx, fixture(t, [][]float64{{-1, -1}, {0, 0}, {1, 1}}), 1)
	if err != nil {
		t.Fatal(err)
	}
	// Input float storage fits int; doubling the feature count does not.
	scores := phantomMatrix{rows: math.MaxInt / 8, cols: 1}
	out, err := fit.Model.InverseTransform(ctx, scores)
	if out != nil {
		t.Fatal("unrepresentable inverse shape produced output")
	}
	mustErrorIs(t, err, pca.ErrOverflow)
}
