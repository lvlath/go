// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package stats_test checks inference failure precedence and nil-result laws.
// Each scenario uses an immutable fitted model and a distinct error boundary.
package stats_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/stats"
)

// TestStandardizer_InferenceErrorBranches protects failures after fitting.
// It distinguishes model presence, input shape, nonfinite data, forward and
// inverse overflow, and matrix.At read failure after output allocation.
//
// Implementation:
//   - Stage 1: create a valid fitted model and check nil/typed-nil cases.
//   - Stage 2: exercise shape and nonfinite precedence before output allocation.
//   - Stage 3: assert both overflow directions and no partial result on late errors.
//
// Behavior highlights:
//   - A late nonfinite observation outranks an earlier potential overflow.
//   - A read failure on the second pass never publishes a partial output.
//
// Inputs:
//   - t: owning test; Matrix fixtures are exclusively read by inference.
//
// Returns:
//   - No value; errors are compared by sentinel identity via errors.Is.
//
// Errors:
//   - ErrNilInput, ErrShape, ErrNonFinite, ErrOverflow, and external At errors.
//
// Determinism:
//   - Fixed validation pass followed by a fixed computation pass, row-major.
//
// Complexity:
//   - O(rows*cols) per inference fixture, constant-sized in this test.
//
// Notes:
//   - Input mutation by another goroutine remains outside Matrix's concurrency law.
//
// AI-Hints:
//   - Do not return output + error when one inferred cell cannot be computed.
//   - Do not classify a late NaN as an earlier arithmetic overflow.
func TestStandardizer_InferenceErrorBranches(t *testing.T) {
	fit, err := stats.FitStandardizer(mustDense(t, [][]float64{{-2}, {2}}), stats.ScaleOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// Stage 1: both absent input forms are rejected by the canonical
	// Matrix/core.Nilable mechanism, not by dereferencing their pointers.
	var typedNil *matrix.Dense
	for name, input := range map[string]matrix.Matrix{"interface_nil": nil, "typed_nil": typedNil} {
		actual, callErr := fit.Model.Transform(input)
		mustErrorIs(t, callErr, stats.ErrNilInput)
		mustNilState(t, actual, true, name)
	}

	// Stage 2: incorrect feature count is rejected before reading any cells.
	wrongShape, err := matrix.NewDense(1, 2)
	if err != nil {
		t.Fatal(err)
	}
	actual, callErr := fit.Model.Transform(wrongShape)
	mustErrorIs(t, callErr, stats.ErrShape)
	mustNilState(t, actual, true, "wrong_feature_count")

	// First cell could overflow if processed early, but a NaN in a later
	// row is encountered during validation before any numerical computation.
	invalid := &scriptedMatrix{rows: 2, cols: 1, cells: [][]float64{{math.MaxFloat64}, {math.NaN()}}}
	actual, callErr = fit.Model.Transform(invalid)
	mustErrorIs(t, callErr, stats.ErrNonFinite)
	mustNilState(t, actual, true, "late_nonfinite")

	// Stage 3a: both directions reject genuinely unrepresentable results.
	// A scale of two yields an overflowing inverse product for MaxFloat64.
	actual, callErr = fit.Model.InverseTransform(mustDense(t, [][]float64{{math.MaxFloat64}}))
	mustErrorIs(t, callErr, stats.ErrOverflow)
	mustNilState(t, actual, true, "inverse_overflow")

	// A tiny but representable positive scale makes a large finite input
	// require a quotient beyond MaxFloat64.
	minSubnormal := math.SmallestNonzeroFloat64
	tinyFit, fitErr := stats.FitStandardizer(
		mustDense(t, [][]float64{{0}, {4 * minSubnormal}}),
		stats.ScaleOptions{},
	)
	if fitErr != nil {
		t.Fatal(fitErr)
	}
	actual, callErr = tinyFit.Model.Transform(mustDense(t, [][]float64{{math.MaxFloat64}}))
	mustErrorIs(t, callErr, stats.ErrOverflow)
	mustNilState(t, actual, true, "forward_overflow")

	// Stage 3b: a Matrix.At implementation may fail during the calculation
	// pass, after an earlier valid cell has already been written privately.
	readCounter := 0
	readFailure := errors.New("scripted later read failure")
	lateReader := &scriptedMatrix{
		rows: 2, cols: 1, value: 1,
		readErr: readFailure, readErrAt: 4, readCount: &readCounter,
	}
	actual, callErr = fit.Model.Transform(lateReader)
	if !errors.Is(callErr, readFailure) {
		t.Fatalf("late matrix.At error identity lost: %v", callErr)
	}
	mustNilState(t, actual, true, "late_matrix_error")
	if readCounter != 4 {
		t.Fatalf("unexpected second-pass read position: %d", readCounter)
	}
}

// TestStandardizer_ScaleUnderflowIsNotConstant separates two failure classes.
// A small positive M2 remains mathematically nonconstant, but its fitted
// positive scale may round to zero and cannot satisfy Standardizer's contract.
//
// Implementation:
//   - Stage 1: construct two distinct subnormal observations.
//   - Stage 2: confirm the exact rational scale rounds to zero in float64.
//   - Stage 3: require ErrOverflow, not ErrZeroVariance or an invalid model.
//
// Behavior highlights:
//   - The constant-feature policy cannot reinterpret representational underflow.
//
// Inputs:
//   - t: owning test; both observations are finite and distinct.
//
// Returns:
//   - No value; confirms nil fit result and correct error classification.
//
// Errors:
//   - ErrOverflow is required because no strictly positive scale can be stored.
//
// Determinism:
//   - Two fixed IEEE subnormal inputs, population ddof=0.
//
// Complexity:
//   - O(1) for the two-row fit; separate oracle uses bounded rational work.
//
// Notes:
//   - Summary variance may round to zero, while fitting cannot use zero scale.
//
// AI-Hints:
//   - Never replace underflow with UnitScaleConstant unless M2 is truly zero.
//   - Do not decide constancy using the rounded public Summary.Variance.
func TestStandardizer_ScaleUnderflowIsNotConstant(t *testing.T) {
	values := []float64{0, math.SmallestNonzeroFloat64}
	if oracleRationalScale(values, 0) != 0 {
		t.Fatal("oracle setup must produce a positive true scale that rounds to zero")
	}
	fit, err := stats.FitStandardizer(mustDense(t, [][]float64{{values[0]}, {values[1]}}), stats.ScaleOptions{ConstantPolicy: stats.UnitScaleConstant})
	mustErrorIs(t, err, stats.ErrOverflow)
	mustNilState(t, fit, true, "scale_underflow")
}

// TestStandardizer_SampleWrapperDelegates verifies a unique fitting kernel.
// A convenience wrapper may choose ddof but must preserve column order,
// strict constant policy, and all results from the canonical FitStandardizer.
//
// Implementation:
//   - Stage 1: construct a nonconstant two-feature training matrix.
//   - Stage 2: fit through both sample-wrapper and canonical configurations.
//   - Stage 3: compare every metadata field and holdout inference coordinate.
//
// Behavior highlights:
//   - The wrapper does not introduce alternative central-moment mathematics.
//
// Inputs:
//   - t: owning test; both models receive the same read-only training input.
//
// Returns:
//   - No result; fails if wrapper and canonical public results differ.
//
// Errors:
//   - ErrInvalidOption is preserved for an invalid enum before input checks.
//
// Determinism:
//   - Ascending feature order and fixed row-major input traversal.
//
// Complexity:
//   - Two independent O(rows*cols) fits and one O(cols) inference comparison.
//
// Notes:
//   - Public metadata slices are detached, not shared between results.
//
// AI-Hints:
//   - Never move a second variance calculation into a convenience facade.
//   - Keep sample ddof separate from the population default.
func TestStandardizer_SampleWrapperDelegates(t *testing.T) {
	train := mustDense(t, [][]float64{{1, 10}, {3, 12}, {5, 14}})
	canonical, err := stats.FitStandardizer(train, stats.ScaleOptions{DDOF: 1})
	if err != nil {
		t.Fatal(err)
	}
	wrapped, err := stats.FitSampleStandardizer(train, stats.RejectConstant)
	if err != nil {
		t.Fatal(err)
	}
	for column := 0; column < train.Cols(); column++ {
		if canonical.Means[column] != wrapped.Means[column] || canonical.Scales[column] != wrapped.Scales[column] || canonical.Constant[column] != wrapped.Constant[column] {
			t.Fatalf("sample wrapper changed feature %d metadata", column)
		}
	}

	heldout := mustDense(t, [][]float64{{7, 16}})
	fromCanonical, err := canonical.Model.Transform(heldout)
	if err != nil {
		t.Fatal(err)
	}
	fromWrapper, err := wrapped.Model.Transform(heldout)
	if err != nil {
		t.Fatal(err)
	}
	for column := 0; column < heldout.Cols(); column++ {
		if mustCell(t, fromCanonical, 0, column) != mustCell(t, fromWrapper, 0, column) {
			t.Fatalf("sample wrapper changed feature %d inference", column)
		}
	}

	invalid, err := stats.FitSampleStandardizer(nil, stats.ConstantPolicy(255))
	mustErrorIs(t, err, stats.ErrInvalidOption)
	mustNilState(t, invalid, true, "invalid_sample_policy")
}
