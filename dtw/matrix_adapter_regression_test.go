// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw_test checks public matrix adapters under adversarial backend and
// full-surface numeric failures, independently of the selected warping band.
package dtw_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/dtw"
	"github.com/lvlath/go/matrix"
)

// TestMatrixAdapter_RejectsPostValidationNumericCorruption verifies a defensive guard.
// A faulty backend can deliver a nonfinite payload after a successful prior read;
// the resulting squared local cost must never escape into the DP recurrence.
//
// Implementation:
//   - Stage 1: construct a stable one-cell Dense source and a corrupt-read wrapper.
//   - Stage 2: allow initial finite validation, then inject a nonfinite read payload.
//   - Stage 3: verify nil result, numeric classification, and source integrity.
//
// Behavior highlights:
//   - Corrupted nonfinite input is not a finite-input arithmetic overflow.
//
// Inputs:
//   - t: owning test; each wrapper is fresh and faults after one successful read.
//
// Returns:
//   - None; publication or misclassification of corrupt data fails the test.
//
// Errors:
//   - ErrNaNInf; not ErrOverflow for a corrupted nonfinite input coordinate.
//
// Determinism:
//   - One-cell sources isolate the two read phases without timing or races.
//
// Complexity:
//   - O(1) time and space per scenario.
//
// Notes:
//   - This fixture deliberately violates stable numeric reads; it does not model
//     supported concurrent mutation or a normally conforming Matrix backend.
//
// AI-Hints:
//   - Do not use a stateful backend as evidence of normal callback-order guarantees.
func TestMatrixAdapter_RejectsPostValidationNumericCorruption(t *testing.T) {
	base := regressionDense(t, 1, 1, 4)
	for _, tc := range []struct {
		name  string
		value float64
	}{
		{"nan", math.NaN()}, {"positive-infinity", math.Inf(1)}, {"negative-infinity", math.Inf(-1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, right := range []bool{false, true} {
				fault := &coverageCorruptReadMatrix{Matrix: base, remainingReads: 1, corruptValue: tc.value}
				var x, y matrix.Matrix = fault, base
				if right {
					x, y = base, fault // Exercise each source operand independently.
				}
				got, err := dtw.AlignMatrix(x, y, dtw.WithReturnLocalCost(true))
				mustMatrixKernelFailure(t, got, err, dtw.ErrNaNInf)
				if errors.Is(err, dtw.ErrOverflow) {
					t.Fatalf("corrupt input incorrectly classified as finite arithmetic overflow: %v", err)
				}
				mustCoverageMatrixCell(t, base, 0, 0, 4, "stable source after injected corruption")
			}
		})
	}
}

// TestMatrixAdapter_OffBandFailureCannotHideBehindFiniteDiagonal checks complete local validation.
// The diagonal itself has zero cost, but a failing or unrepresentable off-band cell
// must still invalidate a precomputed or multivariate local surface.
//
// Implementation:
//   - Stage 1: use two equal two-step series with an unrepresentable cross-distance.
//   - Stage 2: verify both local-artifact policies reject the off-band overflow.
//   - Stage 3: inject an off-band precomputed read failure after the first valid cell.
//
// Behavior highlights:
//   - Full local-surface validation is distinct from admitted-cell DP evaluation.
//
// Inputs:
//   - t: owning test; finite coordinates and small backend fixtures.
//
// Returns:
//   - None; a hidden invalid local cost fails the test.
//
// Errors:
//   - ErrOverflow joined with ErrNaNInf; ErrBadInput joined with the backend cause.
//
// Determinism:
//   - Strict diagonal admission and row-major source reads.
//
// Complexity:
//   - O(1) time and space for the fixed two-step fixtures.
//
// Notes:
//   - Unlike scalar Align without artifacts, AlignMatrix constructs the entire local surface.
//
// AI-Hints:
//   - Do not confuse a discarded accumulated overflow with an invalid local-cost artifact.
func TestMatrixAdapter_OffBandFailureCannotHideBehindFiniteDiagonal(t *testing.T) {
	large := math.Ldexp(1, 512) // Its squared distance from zero exceeds binary64 range.
	source := regressionDense(t, 2, 1, 0, large)
	base := regressionDense(t, 2, 2, 0, 1, 1, 0)
	readErr := errors.New("off-band cost read failure")
	for _, artifact := range []bool{false, true} {
		got, err := dtw.AlignMatrix(source, source, dtw.WithWindow(0), dtw.WithReturnLocalCost(artifact))
		mustMatrixKernelFailure(t, got, err, dtw.ErrOverflow, dtw.ErrNaNInf)
		if errors.Is(err, dtw.ErrNoPath) {
			t.Fatalf("invalid off-band local surface reported structural absence: %v", err)
		}
		fault := &coverageReadFaultMatrix{Matrix: base, remainingReads: 1, readErr: readErr}
		got, err = dtw.AlignCostMatrix(fault, dtw.WithWindow(0), dtw.WithReturnLocalCost(artifact))
		mustMatrixKernelFailure(t, got, err, dtw.ErrBadInput, readErr)
	}
	mustCoverageMatrixCell(t, source, 1, 0, large, "unchanged large coordinate")
	mustCoverageMatrixCell(t, base, 0, 1, 1, "unchanged off-band cost")
}
