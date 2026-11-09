// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package stats_test verifies training-only fit parameters and immutable inference.
// The tests use the shared Matrix fixture helpers and public error/result APIs.
package stats_test

import (
	"fmt"
	"sync"
	"testing"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/stats"
)

// TestStandardizer_HeldoutDoesNotRefit protects the train/inference boundary.
// A holdout batch with a different mean must use the original training scale
// and mean, not its own newly estimated normalization parameters.
//
// Implementation:
//   - Stage 1: fit a one-feature model on independent training observations.
//   - Stage 2: standardize a shifted holdout batch and check both coordinates.
//   - Stage 3: invert the output and verify recovery of the original values.
//
// Behavior highlights:
//   - Transform and InverseTransform never modify fitted model parameters.
//
// Inputs:
//   - t: owning test; matrices are detached fixtures with finite values.
//
// Returns:
//   - No result; fails on holdout leakage or incorrect inverse coordinates.
//
// Errors:
//   - No error is expected for compatible finite matrices and a fitted model.
//
// Determinism:
//   - Fixed two-element holdout order, exact expected floating results.
//
// Complexity:
//   - O(rows*cols) fitting and independent matrix transformation.
//
// Notes:
//   - Holdout model parameters do not depend on the holdout batch shape or mean.
//
// AI-Hints:
//   - Never call FitStandardizer from Transform, even to handle unusual data.
//   - InverseTransform must use the same model, not a fresh fit of z.
func TestStandardizer_HeldoutDoesNotRefit(t *testing.T) {
	// Stage 1: population mean=2 and standard deviation=1.
	fit, err := stats.FitStandardizer(mustDense(t, [][]float64{{1}, {3}}), stats.ScaleOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// Stage 2: a refit of {5,7} would yield {-1,+1}, not {3,5}.
	holdout := mustDense(t, [][]float64{{5}, {7}})
	standardized, err := fit.Model.Transform(holdout)
	if err != nil {
		t.Fatal(err)
	}
	if mustCell(t, standardized, 0, 0) != 3 || mustCell(t, standardized, 1, 0) != 5 {
		t.Fatal("holdout data was normalized with different parameters")
	}

	// Stage 3: the inverse reuses the same training coordinates.
	reconstructed, err := fit.Model.InverseTransform(standardized)
	if err != nil {
		t.Fatal(err)
	}
	if mustCell(t, reconstructed, 0, 0) != 5 || mustCell(t, reconstructed, 1, 0) != 7 {
		t.Fatal("inverse transform did not restore holdout observations")
	}
}

// TestStandardizer_ConstantPolicyPreservesNovelValue checks the exact zero law.
// RejectConstant must refuse fitting a genuinely constant training feature;
// UnitScaleConstant must retain the mean and preserve a novel input's delta.
//
// Implementation:
//   - Stage 1: assert rejection under the default constant-feature policy.
//   - Stage 2: refit with an explicit unit scale and inspect detached metadata.
//   - Stage 3: transform a novel point and verify it is not forced to zero.
//
// Behavior highlights:
//   - Constant means M2==0 in the arithmetic policy, not variance<epsilon.
//
// Inputs:
//   - t: owning test; constant train values and a distinct holdout observation.
//
// Returns:
//   - No result; fails on a wrong sentinel or transformed coordinate.
//
// Errors:
//   - ErrZeroVariance is required under RejectConstant.
//
// Determinism:
//   - Exactly one finite training column, no iteration-order ambiguity.
//
// Complexity:
//   - O(rows*cols) fitting and detached inference allocation.
//
// Notes:
//   - Novel observations differ from constant training mean by ordinary delta.
//
// AI-Hints:
//   - Never replace unit scaling with forced output zero for unseen values.
//   - An epsilon threshold would corrupt legitimate small training variance.
func TestStandardizer_ConstantPolicyPreservesNovelValue(t *testing.T) {
	training := mustDense(t, [][]float64{{5}, {5}})
	// Stage 1: the default rejects constant columns after the fit scan.
	rejected, err := stats.FitStandardizer(training, stats.ScaleOptions{})
	mustErrorIs(t, err, stats.ErrZeroVariance)
	mustNilState(t, rejected, true, "rejected_constant_fit")

	// Stage 2: the explicit policy leaves both fitted mean and scale finite.
	fit, err := stats.FitStandardizer(training, stats.ScaleOptions{ConstantPolicy: stats.UnitScaleConstant})
	if err != nil {
		t.Fatal(err)
	}
	if fit.Means[0] != 5 || fit.Scales[0] != 1 || !fit.Constant[0] {
		t.Fatalf("unit-scale fit metadata incorrect: mean=%g scale=%g constant=%t", fit.Means[0], fit.Scales[0], fit.Constant[0])
	}

	// Stage 3: a holdout value of 7 becomes (7-5)/1=2, not zero.
	standardized, err := fit.Model.Transform(mustDense(t, [][]float64{{7}}))
	if err != nil {
		t.Fatal(err)
	}
	if mustCell(t, standardized, 0, 0) != 2 {
		t.Fatal("the unit-scale policy incorrectly removed the novel value's delta")
	}
}

// TestStandardizer_InputMutationIsolation defends detached model ownership.
// Mutating exported fit metadata, source training data, or previous inference
// outputs must not modify private parameters or subsequent inference results.
//
// Implementation:
//   - Stage 1: fit a model and mutate all exported result parameter slices.
//   - Stage 2: mutate the original training matrix and transform new data.
//   - Stage 3: mutate a returned matrix and compare a second transform result.
//
// Behavior highlights:
//   - The model owns copies of its parameters and allocates independent outputs.
//
// Inputs:
//   - t: owning test; all mutable inputs are caller-owned Matrix/Dense values.
//
// Returns:
//   - No result; fails when any caller mutation leaks into the model.
//
// Errors:
//   - No error expected; every test fixture Set must succeed explicitly.
//
// Determinism:
//   - Fixed mutation order and repeated inference using identical holdout values.
//
// Complexity:
//   - O(rows*cols) for each fit/transform; constant-size ownership fixtures.
//
// Notes:
//   - The exported metadata is deliberately mutable, but does not drive inference.
//
// AI-Hints:
//   - Do not assign result.Means directly to Standardizer's private means slice.
//   - Never return a matrix backed by a shared model-owned inference workspace.
func TestStandardizer_InputMutationIsolation(t *testing.T) {
	train := mustDense(t, [][]float64{{1}, {3}})
	fit, err := stats.FitStandardizer(train, stats.ScaleOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// Stage 1: all three public arrays are caller-owned and writable.
	fit.Means[0] = 200
	fit.Scales[0] = 300
	fit.Constant[0] = true

	// Stage 2: a source modification must not affect already fitted parameters.
	if err := train.Set(0, 0, 999); err != nil {
		t.Fatal(err)
	}
	holdout := mustDense(t, [][]float64{{5}})
	first, err := fit.Model.Transform(holdout)
	if err != nil {
		t.Fatal(err)
	}
	if mustCell(t, first, 0, 0) != 3 {
		t.Fatal("training input or exported metadata leaked into immutable model")
	}

	// Stage 3: result storage is per-call and cannot affect the next result.
	if err := first.Set(0, 0, 19); err != nil {
		t.Fatal(err)
	}
	second, err := fit.Model.Transform(holdout)
	if err != nil {
		t.Fatal(err)
	}
	if mustCell(t, second, 0, 0) != 3 {
		t.Fatal("output matrix aliased model parameters or a shared result buffer")
	}
}

// TestStandardizer_ConcurrentInference exercises a shared immutable model.
// Independent goroutines transform distinct inputs while the model remains
// quiescent; the owning test aggregates errors in deterministic index order.
//
// Implementation:
//   - Stage 1: fit a model once on deterministic, nonconstant observations.
//   - Stage 2: give each goroutine an independent input and output workspace.
//   - Stage 3: wait for completion, then report errors from the owning goroutine.
//
// Behavior highlights:
//   - No goroutine calls t.Fatal or t.FailNow; every result is locally owned.
//
// Inputs:
//   - t: owning test; workers share only the read-only Standardizer pointer.
//
// Returns:
//   - No result; fails on a race-visible error or an incorrect coordinate.
//
// Errors:
//   - Every constructor, setter, and inference error is captured per worker.
//
// Determinism:
//   - Each worker writes a unique error slot; the owner scans slots in index order.
//
// Complexity:
//   - O(workers*rows*cols) total work; O(workers) independent error slots.
//
// Notes:
//   - Race safety does not extend to simultaneous caller mutation of input matrices.
//
// AI-Hints:
//   - Do not cache a mutable output matrix inside Standardizer for reuse.
//   - Collect worker errors in distinct slots rather than fataling from a goroutine.
func TestStandardizer_ConcurrentInference(t *testing.T) {
	const workerCount = 32
	fit, err := stats.FitStandardizer(mustDense(t, [][]float64{{1}, {3}}), stats.ScaleOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// Stage 2: each index owns both a unique input and a dedicated error slot.
	workerErrors := make([]error, workerCount)
	var workers sync.WaitGroup
	for index := 0; index < workerCount; index++ {
		workers.Add(1)
		go func(workerIndex int) {
			defer workers.Done()

			input, buildErr := matrix.NewDense(1, 1)
			if buildErr != nil {
				workerErrors[workerIndex] = buildErr
				return
			}
			value := float64(5 + 2*workerIndex)
			if buildErr = input.Set(0, 0, value); buildErr != nil {
				workerErrors[workerIndex] = buildErr
				return
			}
			output, transformErr := fit.Model.Transform(input)
			if transformErr != nil {
				workerErrors[workerIndex] = transformErr
				return
			}
			actual, readErr := output.At(0, 0)
			if readErr != nil {
				workerErrors[workerIndex] = readErr
				return
			}
			if expected := value - 2; actual != expected {
				workerErrors[workerIndex] = fmt.Errorf("input=%g: output=%g, expected=%g", value, actual, expected)
			}
		}(index)
	}

	// Stage 3: only the owning goroutine fails the test, in stable order.
	workers.Wait()
	for index, workerErr := range workerErrors {
		if workerErr != nil {
			t.Fatalf("inference worker %d: %v", index, workerErr)
		}
	}
}
