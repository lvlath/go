// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix_test verifies the public legacy eigensolver and unpivoted LU contracts.
// Independent eigenpair and reconstruction checks avoid using one solver as another's oracle.
package matrix_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/matrix"
)

// TestCoverageEigenGenericEigenpairs checks a true interface-backed Jacobi computation.
// Its clone remains generic, so both rotation and final spectrum scans are exercised.
// Implementation: Decompose a symmetric three-channel coupling matrix, compare its known
// spectrum, and independently verify every eigenpair and column inner product.
// Behavior highlights: Eigenvalues need not be ordered; repeated bases need not match componentwise.
// Inputs: A=2I+v*v^T with v=(1,2,2); exact eigenvalues are 2,2,11.
// Returns: Test assertions only.
// Errors: Every input access, eigensolver result and output coordinate is checked.
// Determinism: Fixed source and pairwise reductions; spectrum membership is order-independent.
// Complexity: O(maxRotations*n^2) Jacobi work and O(n^3) oracle checks; O(n^2) fixture space.
// Notes: The spectrum oracle uses the rank-one analytic identity, not a covariance solve.
// AI-Hints: Do not compare individual eigenvectors in the repeated-eigenvalue subspace.
func TestCoverageEigenGenericEigenpairs(t *testing.T) {
	values := []float64{3, 2, 2, 2, 6, 4, 2, 4, 6}
	for _, generic := range []bool{false, true} {
		d := NewFilledDense(t, 3, 3, values)
		var input matrix.Matrix = d
		if generic {
			input = &coverageMatrix{Matrix: d}
		}
		eigenvalues, q, err := matrix.EigenSym(input, coverageEigenTolerance, coverageEigenRotations)
		if err != nil || len(eigenvalues) != 3 || q == nil || q.Rows() != 3 || q.Cols() != 3 {
			t.Fatalf("eigenvalues=%v q=%v err=%v", eigenvalues, q, err)
		}
		low, high := 0, 0
		for col, lambda := range eigenvalues {
			if InDelta(t, lambda, 2, coverageRelativeTolerance) {
				low++
			} else if InDelta(t, lambda, 11, coverageRelativeTolerance) {
				high++
			} else {
				t.Fatalf("unexpected eigenvalue %.17g", lambda)
			}
			for row := 0; row < 3; row++ {
				// Evaluate A*q directly from the original finite coefficients.
				var aq float64
				for j := 0; j < 3; j++ {
					aq += values[row*3+j] * MustAt(t, q, j, col)
				}
				factorNear(t, aq, lambda*MustAt(t, q, row, col), coverageRelativeTolerance, coverageRelativeTolerance)
			}
			for other := 0; other < 3; other++ {
				var dot float64
				for row := 0; row < 3; row++ {
					dot += MustAt(t, q, row, col) * MustAt(t, q, row, other)
				}
				want := 0.0
				if col == other {
					want = 1
				}
				factorNear(t, dot, want, coverageRelativeTolerance, 0)
			}
		}
		if low != 2 || high != 1 {
			t.Fatalf("incorrect spectrum multiplicity: %v", eigenvalues)
		}
		CompareExact(t, [][]float64{{3, 2, 2}, {2, 6, 4}, {2, 4, 6}}, d)
	}
}

// TestCoverageEigenGenericCloneFailures verifies distinct read and write failures after admission.
// A valid original matrix is never modified; the fault is installed only on its detached clone.
// Implementation: Force the unique initial pivot (0,1), reject one read or write in each
// rotation stage, and separately reject the final scan or diagonal extraction.
// Behavior highlights: A backend failure cannot publish eigenvalues or a partial basis.
// Inputs: Finite symmetric 3x3 matrix, diagonal 2x2 matrix, and independent clone policies.
// Returns: Test assertions only.
// Errors: errCoverageBackend through the public EigenSym facade.
// Determinism: Unique pivot, fixed coordinate order and independent clone counters.
// Complexity: Bounded small-matrix Jacobi work for each fault.
// Notes: These are genuine Matrix interface error paths, not synthetic failures of private Dense.
// AI-Hints: Use a backend-preserving Clone to exercise the generic branch after validation.
func TestCoverageEigenGenericCloneFailures(t *testing.T) {
	for _, tc := range []struct {
		name  string
		read  coverageAccessFault
		write coverageAccessFault
	}{
		{"pivot-scan", coverageFaultAt(0, 1, 0), coverageAccessFault{}},
		{"pivot-diagonal-p", coverageFaultAt(0, 0, 0), coverageAccessFault{}},
		{"pivot-diagonal-q", coverageFaultAt(1, 1, 0), coverageAccessFault{}},
		{"pivot-cross", coverageFaultAt(0, 1, 1), coverageAccessFault{}},
		{"rotation-read-p", coverageFaultAt(2, 0, 0), coverageAccessFault{}},
		{"rotation-read-q", coverageFaultAt(2, 1, 0), coverageAccessFault{}},
		{"rotation-write-ip", coverageAccessFault{}, coverageFaultAt(2, 0, 0)},
		{"rotation-write-pi", coverageAccessFault{}, coverageFaultAt(0, 2, 0)},
		{"rotation-write-iq", coverageAccessFault{}, coverageFaultAt(2, 1, 0)},
		{"rotation-write-qi", coverageAccessFault{}, coverageFaultAt(1, 2, 0)},
		{"rotation-write-pp", coverageAccessFault{}, coverageFaultAt(0, 0, 0)},
		{"rotation-write-qq", coverageAccessFault{}, coverageFaultAt(1, 1, 0)},
		{"rotation-write-pq", coverageAccessFault{}, coverageFaultAt(0, 1, 0)},
		{"rotation-write-qp", coverageAccessFault{}, coverageFaultAt(1, 0, 0)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			base := NewFilledDense(t, 3, 3, []float64{4, 2, 1, 2, 3, 0.5, 1, 0.5, 5})
			backend := &coverageMatrix{Matrix: base, cloneRead: tc.read, cloneWrite: tc.write}
			values, vectors, err := matrix.EigenSym(backend, coverageEigenTolerance, coverageEigenRotations)
			coverageAssertMatrixError(t, vectors, err, errCoverageBackend)
			if values != nil {
				t.Fatal("eigenvalues escaped a failed rotation")
			}
			CompareExact(t, [][]float64{{4, 2, 1}, {2, 3, 0.5}, {1, 0.5, 5}}, base)
		})
	}
	// A diagonal matrix stops after one scan; its final scan and extraction are separate reads.
	for _, tc := range []struct {
		name  string
		fault coverageAccessFault
	}{{"final-scan", coverageFaultAt(0, 1, 1)}, {"extract-diagonal", coverageFaultAt(0, 0, 0)}} {
		t.Run(tc.name, func(t *testing.T) {
			base := NewFilledDense(t, 2, 2, []float64{3, 0, 0, 4})
			values, vectors, err := matrix.EigenSym(&coverageMatrix{Matrix: base, cloneRead: tc.fault}, coverageEigenTolerance, 1)
			coverageAssertMatrixError(t, vectors, err, errCoverageBackend)
			if values != nil {
				t.Fatal("failed final read published eigenvalues")
			}
		})
	}
}

// TestCoverageEigenGenericBudgetRejectsUnfinishedBasis verifies honest iteration exhaustion.
// A coupled third coordinate remains after one Jacobi rotation, so the final scan must reject it.
// Implementation: Run one rotation on a coupled symmetric matrix through the generic clone.
// Behavior highlights: No partial eigenvalue or eigenvector artifact is published.
// Inputs: A finite 3x3 symmetric matrix with a unique maximal off-diagonal pivot.
// Returns: Test assertions only.
// Errors: ErrMatrixEigenFailed.
// Determinism: The initial pivot and one-rotation budget are fixed.
// Complexity: O(n^2) scan/rotation work and O(n^2) storage.
// Notes: This is the legacy EigenSym error law, not SVD's diagnostic partial-result law.
// AI-Hints: Never consume a budget-exhausted eigensolver result as a certified spectrum.
func TestCoverageEigenGenericBudgetRejectsUnfinishedBasis(t *testing.T) {
	base := NewFilledDense(t, 3, 3, []float64{4, 2, 1, 2, 3, 0.5, 1, 0.5, 5})
	values, vectors, err := matrix.EigenSym(&coverageMatrix{Matrix: base}, coverageEigenTolerance, 1)
	coverageAssertMatrixError(t, vectors, err, matrix.ErrMatrixEigenFailed)
	if values != nil {
		t.Fatal("nonconverged eigensolver published spectrum")
	}
}

// TestCoverageLegacyLUGenericRefusal checks genuine source and finite-output failures.
// It does not manufacture impossible access errors in the private Dense L and U buffers.
// Implementation: Refuse upper/lower source cells, reject an exact zero pivot, and exercise
// nonfinite upper and genuinely unrepresentable lower factor entries on the generic path.
// Behavior highlights: LU never publishes a partial triangular decomposition on failure.
// Inputs: Finite 2x2 matrices and one explicitly raw nonfinite matrix.
// Returns: Test assertions only.
// Errors: errCoverageBackend, ErrSingular and historical ErrNaNInf.
// Determinism: Fixed unpivoted Doolittle traversal.
// Complexity: O(n^3) time and O(n^2) space for LU; fixtures are small.
// Notes: Pivoted FactorLU is the recommended solver for zero-leading-pivot systems.
// AI-Hints: Inverse always consumes Dense LU factors; wrapping only the input cannot reach its dead generic branch.
func TestCoverageLegacyLUGenericRefusal(t *testing.T) {
	base := NewFilledDense(t, 2, 2, []float64{4, 2, 1, 3})
	for _, tc := range []struct {
		name  string
		input matrix.Matrix
		want  error
	}{
		{"upper-source", &coverageMatrix{Matrix: base, read: coverageFaultAt(0, 1, 0)}, errCoverageBackend},
		{"lower-source", &coverageMatrix{Matrix: base, read: coverageFaultAt(1, 0, 0)}, errCoverageBackend},
		{"zero-pivot", &coverageMatrix{Matrix: NewFilledDense(t, 2, 2, []float64{0, 1, 1, 2})}, matrix.ErrSingular},
		{"upper-nonfinite", &coverageMatrix{Matrix: coverageRawDense(t, 1, 1, []float64{math.NaN()})}, matrix.ErrNaNInf},
		{"lower-unrepresentable", &coverageMatrix{Matrix: NewFilledDense(t, 2, 2, []float64{math.SmallestNonzeroFloat64, 0, 1, 1})}, matrix.ErrNaNInf},
		{"invalid-allocation", statsTestShapeOnly{rows: int(^uint(0) >> 1), cols: int(^uint(0) >> 1)}, matrix.ErrInvalidDimensions},
	} {
		t.Run(tc.name, func(t *testing.T) {
			lower, upper, err := matrix.LUDecompose(tc.input)
			coverageAssertMatrixError(t, lower, err, tc.want)
			if upper != nil {
				t.Fatal("failed LU published upper factor")
			}
		})
	}
	// Inverse delegates to LU and preserves its source error without publishing an inverse.
	inverse, err := matrix.InverseOf(&coverageMatrix{Matrix: base, read: coverageFaultAt(1, 0, 0)})
	coverageAssertMatrixError(t, inverse, err, errCoverageBackend)
}
