// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw_test safeguards existing error precedence, ownership, and metric defaults.
// These tests use only the published facades, options, and Result fields.
package dtw_test

import (
	"errors"
	"testing"

	"github.com/lvlath/go/dtw"
	"github.com/lvlath/go/matrix"
)

// TestAlign_NilOptionBeforeInvalidInput preserves the legacy validation precedence.
// A nil functional option is rejected during public facade assembly before
// sequence or local-matrix validation, including simultaneously nil inputs.
//
// Implementation:
//   - Stage 1: call scalar Align with both nil input and nil Option.
//   - Stage 2: call flat-matrix AlignCostMatrix with nil input and nil Option.
//   - Stage 3: verify ErrNilOption and nil results for both public entry points.
//
// Behavior highlights:
//   - New numerical kernels do not change the established facade precedence.
//
// Inputs:
//   - Nil slices/interfaces and an explicit nil Option value.
//
// Returns:
//   - Test failure on a different sentinel or non-nil Result.
//
// Errors:
//   - ErrNilOption must be observable via errors.Is.
//
// Determinism:
//   - Fixed legacy options-before-input validation order.
//
// Complexity:
//   - Time O(1), Space O(1), no allocation-heavy kernel is entered.
//
// Notes:
//   - This tests historical precedence, not the validation order for future APIs.
//
// AI-Hints:
//   - Do not normalize existing facades to future context-first APIs.
//   - Do not move input validation before applyOptions without a release decision.
func TestAlign_NilOptionBeforeInvalidInput(t *testing.T) {
	scalar, scalarErr := dtw.Align(nil, nil, nil)
	if scalar != nil || !errors.Is(scalarErr, dtw.ErrNilOption) {
		t.Fatalf("scalar result=%+v err=%v, want ErrNilOption", scalar, scalarErr)
	}
	matrixResult, matrixErr := dtw.AlignCostMatrix(nil, nil)
	if matrixResult != nil || !errors.Is(matrixErr, dtw.ErrNilOption) {
		t.Fatalf("matrix result=%+v err=%v, want ErrNilOption", matrixResult, matrixErr)
	}
}

// TestAlign_ReturnedBuffersDetachedAcrossCalls protects caller ownership of artifacts.
// Mutating a returned Result.Path and Result.LocalCost must not change inputs
// or subsequent independent calls over those same input sequences.
//
// Implementation:
//   - Stage 1: request a path and full local surface from an immutable scalar pair.
//   - Stage 2: mutate both returned artifacts without touching the inputs.
//   - Stage 3: repeat the alignment and compare the unmodified expected result.
//
// Behavior highlights:
//   - Both the path slice and local Dense are caller-owned, detached results.
//
// Inputs:
//   - Two fixed, equal three-element scalar sequences.
//
// Returns:
//   - Test failure if mutating one Result corrupts another or the source slices.
//
// Errors:
//   - Unexpected artifact Set or alignment error fails the test.
//
// Determinism:
//   - Repeat calls produce equal distance and a diagonal representative.
//
// Complexity:
//   - Time O(9) twice, Space O(9) per result artifact.
//
// Notes:
//   - Concurrent mutation *during* a call is explicitly not supported.
//
// AI-Hints:
//   - Never publish references to private rolling DP or cached local buffers.
//   - Do not claim input-snapshot isolation while the caller is concurrently writing.
func TestAlign_ReturnedBuffersDetachedAcrossCalls(t *testing.T) {
	first, second := []float64{0, 1, 2}, []float64{0, 1, 2}
	opts := []dtw.Option{dtw.WithReturnPath(true), dtw.WithReturnLocalCost(true)}
	one, err := dtw.Align(first, second, opts...)
	if err != nil || one == nil {
		t.Fatalf("first Align result=%+v err=%v", one, err)
	}
	one.Path[0] = dtw.Coord{I: 99, J: 99}
	if err := one.LocalCost.Set(0, 0, 99); err != nil {
		t.Fatalf("mutate local artifact: %v", err)
	}

	two, err := dtw.Align(first, second, opts...)
	if err != nil || two == nil || two.Distance != 0 {
		t.Fatalf("second Align result=%+v err=%v", two, err)
	}
	mustPathEqual(t, two.Path, dtw.Path{{I: 0, J: 0}, {I: 1, J: 1}, {I: 2, J: 2}}, "detached path")
	value, readErr := two.LocalCost.At(0, 0)
	if readErr != nil || value != 0 || first[0] != 0 || second[0] != 0 {
		t.Fatalf("new local=%g err=%v input-first=%v input-second=%v", value, readErr, first, second)
	}
}

// TestAlign_ScalarSquaredAndMatrixMetricsAgree checks parity only under matching metrics.
// The scalar default is absolute distance while AlignMatrix uses squared L2;
// explicitly selecting WithSquaredCost aligns their numeric contracts for 1D rows.
//
// Implementation:
//   - Stage 1: create paired scalar sequences and their equivalent one-feature matrices.
//   - Stage 2: run scalar Align with WithSquaredCost and matrix AlignMatrix.
//   - Stage 3: compare distance and path exactly for equal window/penalty options.
//
// Behavior highlights:
//   - A metric mismatch is not a valid backend-parity regression test.
//
// Inputs:
//   - Fixed three-step and two-step sequences with one feature per row.
//
// Returns:
//   - Test failure if numeric objective or representative path differs.
//
// Errors:
//   - Unexpected validation or DP error fails the test.
//
// Determinism:
//   - Fixed inputs, common local metric, strict diagonal/up/left order.
//
// Complexity:
//   - Both adapters run O(n*m); matrix feature dimension is one.
//
// Notes:
//   - This does not change the scalar AbsoluteCost default.
//
// AI-Hints:
//   - Explicitly set WithSquaredCost before asserting scalar/matrix parity.
//   - Do not alter AlignMatrix's default squared-L2 contract for convenience.
func TestAlign_ScalarSquaredAndMatrixMetricsAgree(t *testing.T) {
	first := []float64{0, 1, 2}
	second := []float64{0, 2}
	x, err := matrix.NewPreparedDense(len(first), 1)
	if err != nil {
		t.Fatalf("NewPreparedDense x: %v", err)
	}
	if err = x.Fill(first); err != nil {
		t.Fatalf("Fill x: %v", err)
	}
	y, err := matrix.NewPreparedDense(len(second), 1)
	if err != nil {
		t.Fatalf("NewPreparedDense y: %v", err)
	}
	if err = y.Fill(second); err != nil {
		t.Fatalf("Fill y: %v", err)
	}

	matrixResult, matrixErr := dtw.AlignMatrix(x, y, dtw.WithReturnPath(true))
	scalarResult, scalarErr := dtw.Align(first, second, dtw.WithSquaredCost(), dtw.WithReturnPath(true))
	if matrixErr != nil || scalarErr != nil || matrixResult == nil || scalarResult == nil {
		t.Fatalf("matrix=%+v err=%v scalar=%+v err=%v", matrixResult, matrixErr, scalarResult, scalarErr)
	}
	if matrixResult.Distance != scalarResult.Distance {
		t.Fatalf("matrix distance=%g scalar distance=%g", matrixResult.Distance, scalarResult.Distance)
	}
	mustPathEqual(t, matrixResult.Path, scalarResult.Path, "squared scalar/matrix path")
}
