// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ridge_test checks cancellation and read-failure atomicity through
// the public training and inference facades, including exact affine fallback.
package ridge_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/ridge"
)

// TestRidge_FitCancellationEveryCheckpoint exercises every observed cancellation
// boundary for each intercept/scaling policy, from validation through publication.
// Implementation:
//   - Stage 1: create an immutable, nonconstant two-feature training fixture.
//   - Stage 2: discover the successful operation's poll sequence for each policy.
//   - Stage 3: interrupt each observed poll and require an atomic failure.
//
// Behavior highlights: cancellation remains sticky in the underlying Go context.
// Inputs: four explicit model policies and a positive finite Lambda.
// Returns: no value; failures identify the policy and cancellation checkpoint.
// Errors: context.Canceled and a nil FitResult are mandatory on interruption.
// Determinism: fixed input, policy order and poll order; no timers or goroutines.
// Complexity: O(k*Tfit) per policy for k observed polls; fixture dimensions are fixed.
// Notes: this also exercises the dependency solver's cooperative cancellation.
// AI-Hints: do not depend on the exact number of polls inside matrix or stats.
func TestRidge_FitCancellationEveryCheckpoint(t *testing.T) {
	// Independent columns avoid a rank-policy failure in the baseline execution.
	x := dense(t, [][]float64{{1, 2}, {3, 1}, {2, 4}})
	y := []float64{3, 5, 7}
	const lambda = 0.75
	policies := []struct {
		name string
		opts ridge.Options
	}{
		{"affine", ridge.Options{Lambda: lambda}},
		{"affine_standardized", ridge.Options{Lambda: lambda, Standardize: true}},
		{"origin", ridge.Options{Lambda: lambda, NoIntercept: true}},
		{"origin_standardized", ridge.Options{Lambda: lambda, NoIntercept: true, Standardize: true}},
	}
	for _, policy := range policies {
		t.Run(policy.name, func(t *testing.T) {
			exerciseRidgeCancellation(t, func(ctx context.Context) (bool, error) {
				fit, err := ridge.Fit(ctx, x, y, policy.opts)
				return fit != nil, err
			})
		})
	}
}

// TestRidge_PredictCancellationEveryCheckpoint verifies atomic inference for
// both ordinary compensated summation and the exact overflow-recovery path.
// Implementation:
//   - Stage 1: train opposite slopes whose magnitudes exceed one.
//   - Stage 2: construct ordinary rows and a finite exact-cancellation row.
//   - Stage 3: interrupt every observed validation/computation/publication poll.
//
// Behavior highlights: even a completed earlier row must not be published on error.
// Inputs: immutable fitted model and two detached inference batches.
// Returns: no value; each canceled invocation must return a nil slice.
// Errors: context.Canceled is preserved, never replaced with ErrOverflow.
// Determinism: fixed rows and synchronous context cancellation.
// Complexity: O(k*Tpredict) for each of the two fixed-size batches.
// Notes: only the exceptional row needs arbitrary-precision arithmetic.
// AI-Hints: test a successful exact fallback before injecting cancellation into it.
func TestRidge_PredictCancellationEveryCheckpoint(t *testing.T) {
	ctx := context.Background()
	train := dense(t, [][]float64{{1, 0}, {0, 1}})
	fit, err := ridge.FitThroughOrigin(ctx, train, []float64{6, -6}, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	beta, _ := fit.Model.Coefficients()
	if len(beta) != 2 || beta[0] <= 1 || beta[1] >= -1 {
		t.Fatalf("invalid exceptional-path fixture: slopes=%v", beta)
	}
	batches := []struct {
		name string
		x    matrix.Matrix
	}{
		{"ordinary", dense(t, [][]float64{{0, 0}, {1, 2}})},
		{"exact_fallback", dense(t, [][]float64{{0, 0}, {math.MaxFloat64, math.MaxFloat64}})},
	}
	for _, batch := range batches {
		t.Run(batch.name, func(t *testing.T) {
			exerciseRidgeCancellation(t, func(ctx context.Context) (bool, error) {
				values, err := fit.Model.Predict(ctx, batch.x)
				return values != nil, err
			})
		})
	}
}

// TestRidge_PredictLateBackendFailure verifies errors from an external Matrix
// that becomes unreadable only after complete preflight validation succeeds.
// Implementation:
//   - Stage 1: fit a valid model and construct a two-row reader.
//   - Stage 2: fail a selected cell on the ordinary or exact arithmetic reread.
//     Also exercise the defensive rejection of invalid data returned on reread.
//   - Stage 3: require the original sentinel and no partially completed batch.
//
// Behavior highlights: values remain stable until a simulated backend fault.
// Inputs: deterministic custom Matrix, ordinary and overflowing-product features.
// Returns: no value; both inference paths must fail atomically.
// Errors: the reader's unique sentinel must survive errors.Is; invalid data must
// return ridge.ErrNonFinite.
// Determinism: count visits to one selected cell, not all internal function calls.
// Complexity: O(1) for the fixed two-row fixtures.
// Notes: nonfinite rereads model a defective external reader, not a supported
// mutable input contract. All calls are synchronous and deterministic.
// AI-Hints: an already-computed first row is not a valid partial Predict result.
func TestRidge_PredictLateBackendFailure(t *testing.T) {
	ctx := context.Background()
	fit, err := ridge.FitThroughOrigin(ctx, dense(t, [][]float64{{1, 0}, {0, 1}}), []float64{6, -6}, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	beta, _ := fit.Model.Coefficients()
	if len(beta) != 2 || beta[0] <= 1 || beta[1] >= -1 {
		t.Fatalf("invalid fallback fixture: slopes=%v", beta)
	}
	marker := errors.New("prediction source read failed")
	cases := []struct {
		name      string
		value     float64
		failVisit int
		nonfinite bool
	}{
		// First visit is the preflight, second is compensated inference.
		{"ordinary_read_error", 2, 2, false},
		{"ordinary_nonfinite", 2, 2, true},
		// Third visit is the exact reread after the overflowing first product.
		{"exact_read_error", math.MaxFloat64, 3, false},
		{"exact_nonfinite", math.MaxFloat64, 3, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			visits := 0
			input := &syntheticMatrix{rows: 2, cols: 2, at: func(row, col int) (float64, error) {
				if row == 0 {
					return 0, nil
				}
				if col == 0 {
					visits++
					if visits == tc.failVisit {
						if tc.nonfinite {
							return math.NaN(), nil
						}
						return 0, marker
					}
				}
				return tc.value, nil
			}}
			values, err := fit.Model.Predict(ctx, input)
			wantErr := marker
			if tc.nonfinite {
				wantErr = ridge.ErrNonFinite
			}
			if values != nil || !errors.Is(err, wantErr) || visits != tc.failVisit {
				t.Fatalf("visits=%d/%d output=%v err=%v", visits, tc.failVisit, values, err)
			}
		})
	}
}

// TestRidge_PredictRejectsAbsentInputOnFittedModel distinguishes invalid model
// receivers from absent inference data, including the core.Nilable convention.
// Implementation: fit once, then exercise nil interface and typed-nil backends.
// Behavior highlights: a valid model must not dereference either missing input.
// Inputs: trained model, nil Matrix, typed-nil *matrix.Dense and custom Matrix.
// Returns: no value; every call must produce a nil output slice.
// Errors: ErrNilInput; typed nil also preserves matrix.ErrNilMatrix.
// Determinism: fixed three-case order.
// Complexity: O(1) for these fixed-size fixtures.
// Notes: zero-row but present matrices are a separate successful contract.
// AI-Hints: testing an unfitted receiver does not cover the input presence gate.
func TestRidge_PredictRejectsAbsentInputOnFittedModel(t *testing.T) {
	ctx := context.Background()
	fit, err := ridge.FitThroughOrigin(ctx, dense(t, [][]float64{{1}, {2}}), []float64{1, 2}, 1)
	if err != nil {
		t.Fatal(err)
	}
	var missingDense *matrix.Dense
	var missingCustom *syntheticMatrix
	cases := []struct {
		name  string
		input matrix.Matrix
		typed bool
	}{
		{"nil_interface", nil, false},
		{"typed_nil_dense", missingDense, true},
		{"typed_nil_custom", missingCustom, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			values, err := fit.Model.Predict(ctx, tc.input)
			if values != nil {
				t.Fatalf("unexpected publication: %v", values)
			}
			mustErrorIs(t, err, ridge.ErrNilInput)
			if tc.typed {
				mustErrorIs(t, err, matrix.ErrNilMatrix)
			}
		})
	}
}
