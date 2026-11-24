// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix_test verifies structural and numeric validators independently of kernels.
// Each case checks the public distinction between invalid shape, invalid input and backend failure.
package matrix_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/matrix"
)

// TestCoverageNonEmptyShapeAndBinaryValidation preserves the zero-shape structural law.
// Operations requiring a nonempty domain may reject valid empty matrices without calling them nil.
// Implementation: Exercise nil, typed nil, all three zero shapes and a nonempty rectangular input.
// Behavior highlights: IsZeroShape(nil) is false; nil and zero dimensions are different failure classes.
// Inputs: Valid Dense instances and absent Matrix interfaces.
// Returns: Test assertions only.
// Errors: ErrNilMatrix, ErrInvalidDimensions and ErrDimensionMismatch.
// Determinism: Fixed shape and operand order.
// Complexity: O(1) for every validation; fixtures have constant size.
// Notes: Zero-shape construction itself remains legal.
// AI-Hints: Do not replace ValidateNonEmptyShape with a blanket prohibition on zero-sized matrices.
func TestCoverageNonEmptyShapeAndBinaryValidation(t *testing.T) {
	if matrix.IsZeroShape(nil) {
		t.Fatal("nil is not a zero-shaped matrix")
	}
	var absent *matrix.Dense
	AssertErrorIs(t, matrix.ValidateNonEmptyShape(nil), matrix.ErrNilMatrix)
	AssertErrorIs(t, matrix.ValidateNonEmptyShape(absent), matrix.ErrNilMatrix)
	for _, shape := range [][2]int{{0, 0}, {0, 3}, {3, 0}} {
		d := MustDense(t, shape[0], shape[1])
		if !matrix.IsZeroShape(d) {
			t.Fatalf("shape %v lost structural zero domain", shape)
		}
		AssertErrorIs(t, matrix.ValidateNonEmptyShape(d), matrix.ErrInvalidDimensions)
	}
	valid := MustDense(t, 2, 3)
	if matrix.IsZeroShape(valid) || matrix.ValidateNonEmptyShape(valid) != nil {
		t.Fatal("nonempty matrix rejected")
	}
	AssertErrorIs(t, matrix.ValidateBinarySameShape(nil, valid), matrix.ErrNilMatrix)
	AssertErrorIs(t, matrix.ValidateBinarySameShape(valid, absent), matrix.ErrNilMatrix)
	AssertErrorIs(t, matrix.ValidateBinarySameShape(valid, MustDense(t, 3, 2)), matrix.ErrDimensionMismatch)
}

// TestCoverageSymmetryAndFiniteBackendValidation distinguishes access failure from asymmetry.
// Symmetry examines both triangular entries and finite diagonal values for order at least two.
// Implementation: Validate exact and tolerance-admitted symmetry, reject invalid cells and
// inject read errors at diagonal, upper-triangle and lower-triangle coordinates.
// Behavior highlights: A negative finite tolerance is normalized by the legacy validator.
// Inputs: Finite symmetric matrices, explicitly raw invalid matrices and generic fault adapters.
// Returns: Test assertions only.
// Errors: ErrNaNInf, ErrAsymmetry and errCoverageBackend.
// Determinism: Diagonal scan followed by upper/lower coordinate pairs.
// Complexity: O(n^2) time, O(n^2) fixture space.
// Notes: The scalar shape fast path is tested only with finite values.
// AI-Hints: ValidateSymmetric is not a substitute for a full finite-domain validation in new solvers.
func TestCoverageSymmetryAndFiniteBackendValidation(t *testing.T) {
	if err := matrix.ValidateSymmetric(MustDense(t, 1, 1), 0); err != nil {
		t.Fatal(err)
	}
	near := NewFilledDense(t, 2, 2, []float64{3, 1, 1.125, 4})
	if err := matrix.ValidateSymmetric(near, -0.125); err != nil {
		t.Fatal(err)
	}
	AssertErrorIs(t, matrix.ValidateSymmetric(near, 0.0625), matrix.ErrAsymmetry)
	for _, coordinate := range [][2]int{{0, 0}, {0, 1}, {1, 0}} {
		backend := &coverageMatrix{Matrix: near, read: coverageFaultAt(coordinate[0], coordinate[1], 0)}
		AssertErrorIs(t, matrix.ValidateSymmetric(backend, 0.125), errCoverageBackend)
	}
	for _, values := range [][]float64{
		{math.NaN(), 1, 1, 4}, {3, math.Inf(1), 1, 4}, {3, 1, math.Inf(-1), 4},
	} {
		AssertErrorIs(t, matrix.ValidateSymmetric(coverageRawDense(t, 2, 2, values), 1), matrix.ErrNaNInf)
	}
	AssertErrorIs(t, matrix.ValidateAllFinite(&coverageMatrix{Matrix: near, read: coverageFaultAt(1, 0, 0)}), errCoverageBackend)
}

// TestCoverageZeroOffDiagonalContract verifies the diagonal-independent structural predicate.
// A nonzero diagonal does not invalidate a diagonal operator, while any large off-diagonal entry does.
// Implementation: Exercise shape/tolerance admission, vacuous shapes, a signed tolerance boundary,
// a nonfinite off-diagonal and a generic backend failure.
// Behavior highlights: Both triangles participate; equality with the tolerance is accepted.
// Inputs: Fixed empty, scalar and 2x2 matrices.
// Returns: Test assertions only.
// Errors: ErrNilMatrix, ErrNonSquare, ErrNaNInf and errCoverageBackend.
// Determinism: Row-major scan skipping diagonal coordinates.
// Complexity: O(n^2) time, O(n^2) fixture space.
// Notes: This predicate does not certify the finiteness of diagonal values.
// AI-Hints: For spectral input, use the appropriate symmetry and finiteness checks as well.
func TestCoverageZeroOffDiagonalContract(t *testing.T) {
	base := NewFilledDense(t, 2, 2, []float64{3, 0.125, -0.125, 4})
	for _, tc := range []struct {
		name string
		m    matrix.Matrix
		tol  float64
		want bool
		err  error
	}{
		{"nil", nil, 0, false, matrix.ErrNilMatrix},
		{"rectangular", MustDense(t, 2, 3), 0, false, matrix.ErrNonSquare},
		{"invalid-tolerance", base, math.Inf(1), false, matrix.ErrNaNInf},
		{"empty", MustDense(t, 0, 0), 0, true, nil},
		{"scalar", NewFilledDense(t, 1, 1, []float64{9}), 0, true, nil},
		{"boundary", base, -0.125, true, nil},
		{"nonzero", base, 0.0625, false, nil},
		{"backend", &coverageMatrix{Matrix: base, read: coverageFaultAt(1, 0, 0)}, 0.125, false, errCoverageBackend},
		{"nonfinite", coverageRawDense(t, 2, 2, []float64{1, 0, math.NaN(), 2}), 0, false, matrix.ErrNaNInf},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := matrix.IsZeroOffDiagonal(tc.m, tc.tol)
			if tc.err != nil {
				AssertErrorIs(t, err, tc.err)
			} else if err != nil {
				t.Fatal(err)
			}
			if got != tc.want {
				t.Fatalf("zero off diagonal=%t want=%t", got, tc.want)
			}
		})
	}
}

// TestCoverageDistanceValidatorAdmission verifies policy precedence and finite diagonal constraints.
// An invalid option, invalid external shape, backend access failure and negative diagonal are distinct.
// Implementation: Apply the distance validator to otherwise valid or explicitly malformed descriptors.
// Behavior highlights: Invalid options precede data access; negative diagonal signals a cycle.
// Inputs: A valid two-node distance table, malformed metadata and fixed invalid diagonal values.
// Returns: Test assertions only.
// Errors: ErrNilOption, ErrInvalidDimensions, errCoverageBackend and ErrNegativeCycle.
// Determinism: Fixed option and row-major cell order.
// Complexity: O(n^2) time and fixture storage; malformed descriptors allocate no matrix.
// Notes: Negative dimensions are used only to test validation of external metadata.
// AI-Hints: Do not allocate an invalid descriptor or bypass public shape admission.
func TestCoverageDistanceValidatorAdmission(t *testing.T) {
	base := NewFilledDense(t, 2, 2, []float64{0, 2, 3, 0})
	AssertErrorIs(t, matrix.ValidateDistanceMatrix(nil, nil), matrix.ErrNilOption)
	AssertErrorIs(t, matrix.ValidateDistanceMatrix(statsTestShapeOnly{rows: -1, cols: -1}), matrix.ErrInvalidDimensions)
	AssertErrorIs(t, matrix.ValidateDistanceMatrix(&coverageMatrix{Matrix: base, read: coverageFaultAt(0, 1, 0)}), errCoverageBackend)
	AssertErrorIs(t, matrix.ValidateDistanceMatrix(NewFilledDense(t, 2, 2, []float64{0, 2, 3, -1})), matrix.ErrNegativeCycle)
}
