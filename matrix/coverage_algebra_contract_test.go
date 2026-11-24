// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix_test checks matrix allocation, generic backend errors and algebraic invariants.
// Dense and interface-backed matrices must preserve the published contract without hidden aliasing.
package matrix_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/matrix"
)

// TestCoverageFacadeAllocationValidation verifies admission without large allocations.
// The shape-only fixture represents an external descriptor whose requested result cannot fit int.
// Implementation: Exercise constructors, facade nil checks and oversized output shapes.
// Behavior highlights: All allocating operations return no result for invalid dimensions.
// Inputs: Nil inputs, a non-square matrix, and positive dimensions whose product overflows int.
// Returns: Test assertions only.
// Errors: ErrInvalidDimensions, ErrNilMatrix and ErrNonSquare.
// Determinism: Ordered cases with no real oversized allocations.
// Complexity: O(1) per rejected descriptor; small constant fixture storage.
// Notes: This does not attempt to simulate an unrecoverable runtime allocation failure.
// AI-Hints: Distinguish checked integer shape overflow from running out of physical RAM.
func TestCoverageFacadeAllocationValidation(t *testing.T) {
	if d, err := matrix.NewZeros(-1, 2); d != nil || err == nil {
		t.Fatalf("NewZeros admitted negative shape: %v %v", d, err)
	} else {
		AssertErrorIs(t, err, matrix.ErrInvalidDimensions)
	}
	if d, err := matrix.NewIdentity(-1); d != nil || err == nil {
		t.Fatalf("NewIdentity admitted negative order: %v %v", d, err)
	} else {
		AssertErrorIs(t, err, matrix.ErrInvalidDimensions)
	}
	if matrix.CloneMatrix(nil) != nil {
		t.Fatal("cloning nil produced a matrix")
	}
	if d, err := matrix.ZerosLike(nil); d != nil || err == nil {
		t.Fatalf("ZerosLike(nil)=%v %v", d, err)
	} else {
		AssertErrorIs(t, err, matrix.ErrNilMatrix)
	}
	if d, err := matrix.IdentityLike(MustDense(t, 1, 2)); d != nil || err == nil {
		t.Fatalf("IdentityLike rectangular=%v %v", d, err)
	} else {
		AssertErrorIs(t, err, matrix.ErrNonSquare)
	}
	// The output shape overflows int, not just the available machine memory.
	maxInt := int(^uint(0) >> 1)
	oversized := statsTestShapeOnly{rows: maxInt, cols: 2}
	if d, err := matrix.ZerosLike(oversized); d != nil || err == nil {
		t.Fatalf("ZerosLike oversized=%v %v", d, err)
	} else {
		AssertErrorIs(t, err, matrix.ErrInvalidDimensions)
	}
	for _, tc := range []struct {
		name string
		call func() (matrix.Matrix, error)
	}{
		{"sum", func() (matrix.Matrix, error) { return matrix.Sum(oversized, oversized) }},
		{"difference", func() (matrix.Matrix, error) { return matrix.Diff(oversized, oversized) }},
		{"hadamard", func() (matrix.Matrix, error) { return matrix.HadamardProd(oversized, oversized) }},
		{"transpose", func() (matrix.Matrix, error) { return matrix.T(oversized) }},
		{"scale", func() (matrix.Matrix, error) { return matrix.ScaleBy(oversized, 2) }},
		{"product", func() (matrix.Matrix, error) {
			return matrix.Product(statsTestShapeOnly{rows: maxInt, cols: 1}, statsTestShapeOnly{rows: 1, cols: 2})
		}},
		{"sanitize", func() (matrix.Matrix, error) { return matrix.ReplaceInfNaN(oversized, 0) }},
		{"clip", func() (matrix.Matrix, error) { return matrix.Clip(oversized, -1, 1) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tc.call()
			coverageAssertMatrixError(t, got, err, matrix.ErrInvalidDimensions)
		})
	}
}

// TestCoverageAlgebraBackendReadErrors checks both operands of every generic binary kernel.
// A valid finite source becomes unavailable only at a requested cell.
// Implementation: Call public facade functions with the faulty backend on each operand in turn.
// Behavior highlights: Backend errors survive operation-specific wrapping and no result escapes.
// Inputs: Two compatible finite 2x2 matrices and a fault at (0,0).
// Returns: Test assertions only.
// Errors: errCoverageBackend, independent of matrix-specific sentinels.
// Determinism: Table order and coordinate-based refusal.
// Complexity: O(n^3) worst-case for Product; O(n^2) storage for all fixtures.
// Notes: A nonzero left operand ensures Mul cannot skip the requested right-hand read.
// AI-Hints: A nil matrix tests admission, not a failure during generic computation.
func TestCoverageAlgebraBackendReadErrors(t *testing.T) {
	a := NewFilledDense(t, 2, 2, []float64{1, 2, 3, 4})
	b := NewFilledDense(t, 2, 2, []float64{2, 3, 4, 5})
	for _, tc := range []struct {
		name string
		call func(matrix.Matrix, matrix.Matrix) (matrix.Matrix, error)
	}{{"add", matrix.Sum}, {"sub", matrix.Diff}, {"hadamard", matrix.HadamardProd}, {"mul", matrix.Product}} {
		t.Run(tc.name, func(t *testing.T) {
			for _, left := range []bool{true, false} {
				var x, y matrix.Matrix = a, b
				if left {
					x = &coverageMatrix{Matrix: a, read: coverageFaultAt(0, 0, 0)}
				} else {
					y = &coverageMatrix{Matrix: b, read: coverageFaultAt(0, 0, 0)}
				}
				got, err := tc.call(x, y)
				coverageAssertMatrixError(t, got, err, errCoverageBackend)
			}
		})
	}
	for _, tc := range []struct {
		name string
		call func(matrix.Matrix) (matrix.Matrix, error)
	}{
		{"transpose", matrix.T},
		{"scale", func(input matrix.Matrix) (matrix.Matrix, error) { return matrix.ScaleBy(input, 2) }},
		{"sanitize", func(input matrix.Matrix) (matrix.Matrix, error) { return matrix.ReplaceInfNaN(input, 0) }},
		{"clip", func(input matrix.Matrix) (matrix.Matrix, error) { return matrix.Clip(input, -1, 1) }},
		{"symmetrize", matrix.Symmetrize},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tc.call(&coverageMatrix{Matrix: a, read: coverageFaultAt(0, 0, 0)})
			coverageAssertMatrixError(t, got, err, errCoverageBackend)
		})
	}
	// Symmetrize reads the input once in Transpose and again when forming the sum.
	got, err := matrix.Symmetrize(&coverageMatrix{Matrix: a, read: coverageFaultAt(0, 0, 1)})
	coverageAssertMatrixError(t, got, err, errCoverageBackend)
	got, err = matrix.Symmetrize(MustDense(t, 1, 2))
	coverageAssertMatrixError(t, got, err, matrix.ErrNonSquare)
	got, err = matrix.T(nil)
	coverageAssertMatrixError(t, got, err, matrix.ErrNilMatrix)
	got, err = matrix.ScaleBy(nil, 1)
	coverageAssertMatrixError(t, got, err, matrix.ErrNilMatrix)
	got, err = matrix.Clip(nil, 0, 1)
	coverageAssertMatrixError(t, got, err, matrix.ErrNilMatrix)
}

// TestCoverageGenericAlgebraOutputPolicy protects finite-only output storage on the generic path.
// The tested mathematical results are genuinely unrepresentable or contain invalid observations.
// Implementation: Use finite overflow witnesses and raw invalid ingress, then require classified errors.
// Behavior highlights: No successful generic result may contain NaN or infinity in its owning Dense.
// Inputs: 1x1 finite maximum magnitudes and an explicitly permissive NaN fixture.
// Returns: Test assertions only.
// Errors: Historical ErrNaNInf from generic Dense output writes.
// Determinism: Single-cell operations, no rounding-sensitive cancellation.
// Complexity: O(1) per case.
// Notes: This test documents the generic output policy; it does not endorse unchecked Dense fast paths.
// AI-Hints: Never equate a raw ingestion policy with permission to publish nonfinite algebra results.
func TestCoverageGenericAlgebraOutputPolicy(t *testing.T) {
	huge := &coverageMatrix{Matrix: NewFilledDense(t, 1, 1, []float64{math.MaxFloat64})}
	two := NewFilledDense(t, 1, 1, []float64{2})
	dirty := &coverageMatrix{Matrix: coverageRawDense(t, 1, 1, []float64{math.NaN()})}
	for _, tc := range []struct {
		name string
		call func() (matrix.Matrix, error)
	}{
		{"add", func() (matrix.Matrix, error) { return matrix.Sum(huge, huge) }},
		{"product", func() (matrix.Matrix, error) { return matrix.Product(huge, two) }},
		{"hadamard", func() (matrix.Matrix, error) { return matrix.HadamardProd(huge, two) }},
		{"scale", func() (matrix.Matrix, error) { return matrix.ScaleBy(huge, 2) }},
		{"transpose", func() (matrix.Matrix, error) { return matrix.T(dirty) }},
		{"clip-nan", func() (matrix.Matrix, error) { return matrix.Clip(dirty, -1, 1) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tc.call()
			coverageAssertMatrixError(t, got, err, matrix.ErrNaNInf)
		})
	}
}

// TestCoverageReductionFacadeErrors verifies vector reduction publication and error propagation.
// RowSums uses MatVec directly, while ColSums first materializes a transpose.
// Implementation: Check both finite vector oracles and backend failure before publication.
// Behavior highlights: An error never publishes a partially computed vector.
// Inputs: A finite rectangular matrix with signed coordinates.
// Returns: Test assertions only.
// Errors: ErrNilMatrix and errCoverageBackend.
// Determinism: Fixed row-major read order.
// Complexity: O(rows*cols) time; ColSums additionally allocates its transpose.
// Notes: Input storage remains caller-owned and unchanged.
// AI-Hints: Do not substitute AllClose for an independent expected reduction vector.
func TestCoverageReductionFacadeErrors(t *testing.T) {
	d := NewFilledDense(t, 2, 3, []float64{2, -3, 4, 1, 0, -2})
	for _, tc := range []struct {
		name string
		call func(matrix.Matrix) ([]float64, error)
		want []float64
	}{
		{"row-sums", matrix.RowSums, []float64{3, -1}},
		{"col-sums", matrix.ColSums, []float64{3, -3, 2}},
		{"mat-vec", func(m matrix.Matrix) ([]float64, error) { return matrix.MatVecMul(m, []float64{1, 2, -1}) }, []float64{-8, 3}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tc.call(&coverageMatrix{Matrix: d})
			if err != nil || !AlmostEqualSlice(got, tc.want, 0) {
				t.Fatalf("result=%v want=%v err=%v", got, tc.want, err)
			}
			got, err = tc.call(&coverageMatrix{Matrix: d, read: coverageFaultAt(1, 2, 0)})
			AssertErrorIs(t, err, errCoverageBackend)
			if got != nil {
				t.Fatal("failed reduction published vector")
			}
			got, err = tc.call(nil)
			AssertErrorIs(t, err, matrix.ErrNilMatrix)
			if got != nil {
				t.Fatal("nil input published vector")
			}
		})
	}
}

// TestCoverageGenericSanitizationAndAllClose checks explicit nonfinite comparison and sanitation.
// Same-sign infinities compare equal, opposite signs and every NaN do not.
// Implementation: Compare scalar fixtures through both dispatch modes and verify generic copies.
// Behavior highlights: Cleaning replaces all three nonfinite categories without altering input.
// Inputs: Finite values, NaN, +Inf and -Inf under explicit raw storage policy.
// Returns: Test assertions only.
// Errors: Backend errors are distinct from a legitimate false comparison.
// Determinism: Fixed cases and exact scalar comparisons.
// Complexity: O(rows*cols) time for sanitation and comparison; O(rows*cols) output storage.
// Notes: NaN is never considered close, even to itself.
// AI-Hints: Test output with explicit expected values rather than reusing the sanitizer as oracle.
func TestCoverageGenericSanitizationAndAllClose(t *testing.T) {
	for _, tc := range []struct {
		name string
		a, b float64
		want bool
	}{
		{"positive-infinity", math.Inf(1), math.Inf(1), true},
		{"negative-infinity", math.Inf(-1), math.Inf(-1), true},
		{"opposite-infinity", math.Inf(1), math.Inf(-1), false},
		{"finite-infinity", 1, math.Inf(1), false},
		{"nan-self", math.NaN(), math.NaN(), false},
		{"nan-finite", 1, math.NaN(), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var a, b matrix.Matrix = coverageRawDense(t, 1, 1, []float64{tc.a}), coverageRawDense(t, 1, 1, []float64{tc.b})
			for _, generic := range []bool{false, true} {
				if generic {
					a, b = &coverageMatrix{Matrix: a}, &coverageMatrix{Matrix: b}
				}
				got, err := matrix.AllClose(a, b, 0, 0)
				if err != nil || got != tc.want {
					t.Fatalf("generic=%t: close=%t want=%t err=%v", generic, got, tc.want, err)
				}
			}
		})
	}
	input := coverageRawDense(t, 2, 3, []float64{math.NaN(), math.Inf(1), math.Inf(-1), -3, 0.5, 4})
	clean, err := matrix.ReplaceInfNaN(&coverageMatrix{Matrix: input}, 0)
	if err != nil {
		t.Fatal(err)
	}
	CompareExact(t, [][]float64{{0, 0, 0}, {-3, 0.5, 4}}, clean)
	clipped, err := matrix.Clip(&coverageMatrix{Matrix: clean}, -1, 1)
	if err != nil {
		t.Fatal(err)
	}
	CompareExact(t, [][]float64{{0, 0, 0}, {-1, 0.5, 1}}, clipped)
	if !math.IsNaN(MustAt(t, input, 0, 0)) || !math.IsInf(MustAt(t, input, 0, 1), 1) {
		t.Fatal("sanitization mutated the raw input")
	}
	for _, left := range []bool{true, false} {
		var a, b matrix.Matrix = clean, clean
		if left {
			a = &coverageMatrix{Matrix: clean, read: coverageFaultAt(1, 2, 0)}
		} else {
			b = &coverageMatrix{Matrix: clean, read: coverageFaultAt(1, 2, 0)}
		}
		ok, err := matrix.AllClose(a, b, 0, 0)
		AssertErrorIs(t, err, errCoverageBackend)
		if ok {
			t.Fatal("failed comparison returned true")
		}
	}
}

// TestCoverageRowTransformsSecondPass verifies the broadcast and scaling stages independently of reductions.
// A backend failure after the first successful row read must still abort the full transform.
// Implementation: Check analytic row means and L1 norms, then inject a second-pass read refusal.
// Behavior highlights: A failed transform returns neither its matrix nor partial metadata.
// Inputs: Two finite rows, one containing signed values and one exactly zero.
// Returns: Test assertions only.
// Errors: errCoverageBackend from the application pass.
// Determinism: Each cell is observed once in the reduction and once in the application pass.
// Complexity: O(rows*cols) time and output storage.
// Notes: These compatibility transforms are not the stable L2 or column-statistics kernels.
// AI-Hints: Exercise the broadcast through CenterRows, not a newly exported private helper.
func TestCoverageRowTransformsSecondPass(t *testing.T) {
	d := NewFilledDense(t, 2, 2, []float64{-2, 6, 0, 0})
	for _, tc := range []struct {
		name string
		call func(matrix.Matrix) (matrix.Matrix, []float64, error)
		want [][]float64
		meta []float64
	}{
		{"center", matrix.CenterRows, [][]float64{{-4, 4}, {0, 0}}, []float64{2, 0}},
		{"normalize-l1", matrix.NormalizeRowsL1, [][]float64{{-0.25, 0.75}, {0, 0}}, []float64{8, 0}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, metadata, err := tc.call(&coverageMatrix{Matrix: d})
			if err != nil || !AlmostEqualSlice(metadata, tc.meta, 0) {
				t.Fatalf("metadata=%v want=%v err=%v", metadata, tc.meta, err)
			}
			CompareExact(t, tc.want, got)
			got, metadata, err = tc.call(&coverageMatrix{Matrix: d, read: coverageFaultAt(0, 1, 1)})
			coverageAssertMatrixError(t, got, err, errCoverageBackend)
			if metadata != nil {
				t.Fatal("failed transform published metadata")
			}
		})
	}
	CompareExact(t, [][]float64{{-2, 6}, {0, 0}}, d)
	// A nonfinite centered coordinate cannot be written into a finite-only generic output.
	got, metadata, err := matrix.CenterRows(&coverageMatrix{Matrix: coverageRawDense(t, 1, 2, []float64{math.NaN(), 1})})
	coverageAssertMatrixError(t, got, err, matrix.ErrNaNInf)
	if metadata != nil {
		t.Fatal("failed centering published metadata")
	}
}
