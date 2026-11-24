// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw_test verifies public Matrix adapter boundaries and error propagation.
// These fixtures exercise invalid shapes and backend faults without allocating enormous matrices.
// AI-Hints:
//   - Do not simulate a read error by mutating a validated matrix during alignment.
//   - Preserve both the DTW sentinel and the original backend error.
package dtw_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/dtw"
	"github.com/lvlath/go/matrix"
)

// TestCoverageAlignMatrix_ValidationBoundaries checks both source positions independently.
// Nil, zero-row and malformed-shape inputs must fail before local-cost publication.
// Implementation:
//   - Stage 1: construct one valid Dense and boundary-specific Matrix values.
//   - Stage 2: invoke the public multivariate facade for each invalid operand.
//   - Stage 3: require nil result and all relevant sentinel identities.
//
// Complexity: Time O(1), Space O(1) per fixture; no oversized matrix is allocated.
// AI-Hints: Do not pass a typed-nil Matrix to its methods before validating nil state.
func TestCoverageAlignMatrix_ValidationBoundaries(t *testing.T) {
	valid := regressionDense(t, 1, 1, 2)
	empty := regressionDense(t, 0, 1)
	var typedNil *matrix.Dense
	// Negative metadata is invalid even when the backing object itself is a valid Dense.
	negative := coverageShapeMatrix{Matrix: valid, rows: -1, cols: 1}
	for _, tc := range []struct {
		name string
		x, y matrix.Matrix
		want []error
	}{
		{"nil-left", nil, valid, []error{dtw.ErrBadInput, matrix.ErrNilMatrix}},
		{"typed-nil-right", valid, typedNil, []error{dtw.ErrBadInput, matrix.ErrNilMatrix}},
		{"empty-left", empty, valid, []error{dtw.ErrEmptyInput}},
		{"empty-right", valid, empty, []error{dtw.ErrEmptyInput}},
		{"negative-output-shape", negative, valid, []error{dtw.ErrBadInput, matrix.ErrInvalidDimensions}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := dtw.AlignMatrix(tc.x, tc.y)
			if got != nil {
				t.Fatalf("invalid matrix input published a result: %+v", got)
			}
			for _, target := range tc.want {
				mustErrorIs(t, err, target, tc.name)
			}
		})
	}
}

// TestCoverageAlignMatrix_RightNonFinite checks the second input's finite-validation boundary.
// A raw Dense may hold dirty values, but multivariate DTW must reject them before DP.
// Implementation: construct raw data, align, then check both package error identities.
// Complexity: Time O(1), Space O(1) per one-cell fixture.
func TestCoverageAlignMatrix_RightNonFinite(t *testing.T) {
	valid := regressionDense(t, 1, 1, 1)
	for _, value := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		dirty, err := matrix.NewPreparedDense(1, 1, matrix.WithNoValidateNaNInf())
		mustNoError(t, err, "create raw right matrix")
		mustNoError(t, dirty.Set(0, 0, value), "store nonfinite test value")
		got, err := dtw.AlignMatrix(valid, dirty)
		if got != nil {
			t.Fatalf("nonfinite right operand %v published %+v", value, got)
		}
		mustErrorIs(t, err, dtw.ErrNaNInf, "right input finite validation")
		mustErrorIs(t, err, matrix.ErrNaNInf, "preserve Matrix finite validation")
	}
}

// TestCoverageAlignMatrix_BackendReadFailure checks errors before and after finite validation.
// A Matrix backend may fail to read a valid coordinate without altering its source data.
// Implementation:
//   - Stage 1: construct a stable one-cell source and a local read-failure sentinel.
//   - Stage 2: inject a failure on the first or second read on each operand.
//   - Stage 3: verify the validation/compute error family and the underlying cause.
//
// Determinism: one source coordinate eliminates any reliance on multi-cell access order.
// Complexity: Time O(1), Space O(1) per fixture.
// AI-Hints: A post-validation read error is not evidence of a nonfinite squared distance.
func TestCoverageAlignMatrix_BackendReadFailure(t *testing.T) {
	base := regressionDense(t, 1, 1, 3)
	readErr := errors.New("matrix backend read failure")
	for _, tc := range []struct {
		name    string
		right   bool
		allowed int
		want    error
	}{
		{"left-validation", false, 0, dtw.ErrNaNInf},
		{"right-validation", true, 0, dtw.ErrNaNInf},
		{"left-compute", false, 1, dtw.ErrBadInput},
		{"right-compute", true, 1, dtw.ErrBadInput},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fault := &coverageReadFaultMatrix{Matrix: base, remainingReads: tc.allowed, readErr: readErr}
			var x, y matrix.Matrix = fault, base
			if tc.right {
				x, y = base, fault
			}
			got, err := dtw.AlignMatrix(x, y, dtw.WithReturnLocalCost(true))
			if got != nil {
				t.Fatalf("backend failure published a result: %+v", got)
			}
			mustErrorIs(t, err, tc.want, tc.name)
			mustErrorIs(t, err, readErr, "preserve read failure")
			mustCoverageMatrixCell(t, base, 0, 0, 3, "unchanged source")
		})
	}
}

// TestCoverageAlignCostMatrix_RejectsInvalidStorageDomain checks safe materialization.
// Unrepresentable dimensions must be rejected before any allocation or coordinate read.
// Implementation: use a real one-cell backend with malformed shape metadata and check sentinels.
// Complexity: Time O(1), Space O(1) per fixture regardless of reported dimensions.
// AI-Hints: Never allocate MaxInt-sized test data to exercise a checked product.
func TestCoverageAlignCostMatrix_RejectsInvalidStorageDomain(t *testing.T) {
	base := regressionDense(t, 1, 1, 1)
	var typedNil *matrix.Dense
	for _, tc := range []struct {
		name  string
		local matrix.Matrix
		want  []error
	}{
		{"nil", nil, []error{dtw.ErrBadInput, matrix.ErrNilMatrix}},
		{"typed-nil", typedNil, []error{dtw.ErrBadInput, matrix.ErrNilMatrix}},
		{"negative-rows", coverageShapeMatrix{Matrix: base, rows: -1, cols: 1}, []error{dtw.ErrBadInput}},
		{"negative-cols", coverageShapeMatrix{Matrix: base, rows: 1, cols: -1}, []error{dtw.ErrBadInput}},
		{"product-overflow", coverageShapeMatrix{Matrix: base, rows: math.MaxInt, cols: 2}, []error{dtw.ErrBadInput}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := dtw.AlignCostMatrix(tc.local)
			if got != nil {
				t.Fatalf("invalid cost matrix published %+v", got)
			}
			for _, target := range tc.want {
				mustErrorIs(t, err, target, tc.name)
			}
		})
	}
}

// TestCoverageAlignCostMatrix_BackendReadAndArtifactFailure separates two failure stages.
// A failed initial materialization cannot publish an objective; a failed optional copy
// must return its original error without advertising a complete LocalCost artifact.
// Implementation:
//   - Stage 1: inject a first-read failure and require a nil result.
//   - Stage 2: permit materialization but fail a later artifact read.
//   - Stage 3: verify sentinel preservation and absence of an incomplete artifact.
//
// Complexity: Time O(1), Space O(1) for each one-cell alignment.
// Notes: The existing adapter currently returns a computed Result together with the copy error.
// AI-Hints: A nonnil Result accompanied by an error is not a successful artifact publication.
func TestCoverageAlignCostMatrix_BackendReadAndArtifactFailure(t *testing.T) {
	base := regressionDense(t, 1, 1, 4)
	readErr := errors.New("cost matrix backend read failure")
	// A failure before materialization cannot produce a DP result.
	fault := &coverageReadFaultMatrix{Matrix: base, readErr: readErr}
	got, err := dtw.AlignCostMatrix(fault)
	if got != nil {
		t.Fatalf("materialization failure published %+v", got)
	}
	mustErrorIs(t, err, dtw.ErrBadInput, "materialization failure")
	mustErrorIs(t, err, readErr, "materialization backend cause")

	// The current optional-copy path rereads the unchanged source after scoring.
	fault = &coverageReadFaultMatrix{Matrix: base, remainingReads: 1, readErr: readErr}
	got, err = dtw.AlignCostMatrix(fault, dtw.WithReturnLocalCost(true))
	mustErrorIs(t, err, dtw.ErrBadInput, "artifact copy failure")
	mustErrorIs(t, err, readErr, "artifact backend cause")
	if got != nil && got.LocalCost != nil {
		t.Fatalf("failed copy published a local-cost artifact: %+v", got.LocalCost)
	}
	// A healthy backend must publish the whole cost surface, detached from its source.
	fault = &coverageReadFaultMatrix{Matrix: base, remainingReads: 2, readErr: readErr}
	got, err = dtw.AlignCostMatrix(fault, dtw.WithReturnLocalCost(true))
	mustNoError(t, err, "successful artifact publication")
	if got == nil || got.LocalCost == nil || !got.Reachable || got.Distance != 4 {
		t.Fatalf("unexpected healthy backend result: %+v", got)
	}
	mustCoverageMatrixCell(t, got.LocalCost, 0, 0, 4, "published local cost")
	mustNoError(t, got.LocalCost.Set(0, 0, 7), "mutate detached local cost")
	mustCoverageMatrixCell(t, base, 0, 0, 4, "unchanged cost matrix")
}

// TestCoverageAlignMatrix_ZeroFeatureRows verifies the empty-feature sum convention.
// Nonempty time axes with zero features have squared row distance zero, not an empty time domain.
// Implementation: align two zero-feature matrices, request artifacts, and inspect the full surface.
// Determinism: radius zero admits only the unique diagonal witness.
// Complexity: Time O(n*m), Space O(n*m) for the explicitly requested artifacts.
// AI-Hints: Distinguish zero rows (invalid time axis) from zero columns (empty feature sum).
func TestCoverageAlignMatrix_ZeroFeatureRows(t *testing.T) {
	x := regressionDense(t, 2, 0)
	y := regressionDense(t, 2, 0)
	got, err := dtw.AlignMatrix(x, y, dtw.WithWindow(0), dtw.WithReturnPath(true), dtw.WithReturnLocalCost(true))
	mustNoError(t, err, "zero-feature alignment")
	if got == nil || !got.Reachable || got.LocalCost == nil || got.Distance != 0 {
		t.Fatalf("zero-feature squared sum must be zero: %+v", got)
	}
	mustPathEqual(t, got.Path, dtw.Path{{I: 0, J: 0}, {I: 1, J: 1}}, "zero-feature witness")
	mustMatrixShape(t, got.LocalCost, 2, 2, "full zero-feature artifact")
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			mustCoverageMatrixCell(t, got.LocalCost, i, j, 0, "empty sum local cost")
		}
	}
}
