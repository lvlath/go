// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package matrix_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/matrix"
)

// --- ewBroadcastSubCols -------------------------------------------------------

func TestEwBroadcastSubCols_FastAndFallback_Match(t *testing.T) {
	t.Parallel()

	X := NewFilledDense(t, 2, 3, []float64{1, 2, 3, 10, 20, 30})
	colMeans := []float64{4, 5, 6}

	gotFast, err := matrix.EwBroadcastSubColsTestOnly(X, colMeans)
	if err != nil {
		t.Fatalf("fast: %v", err)
	}
	gotSlow, err := matrix.EwBroadcastSubColsTestOnly(hide{X}, colMeans)
	if err != nil {
		t.Fatalf("slow: %v", err)
	}

	exp := [][]float64{
		{-3, -3, -3},
		{6, 15, 24},
	}
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			a := MustAt(t, gotFast, i, j)
			b := MustAt(t, gotSlow, i, j)
			if a != exp[i][j] || b != exp[i][j] {
				t.Fatalf("subCols[%d,%d]: fast=%v slow=%v want=%v", i, j, a, b, exp[i][j])
			}
		}
	}
}

func TestEwBroadcastSubCols_DimMismatch_Err(t *testing.T) {
	t.Parallel()
	X := NewFilledDense(t, 2, 3, []float64{1, 2, 3, 4, 5, 6})
	_, err := matrix.EwBroadcastSubColsTestOnly(X, []float64{0, 0})
	if !errors.Is(err, matrix.ErrDimensionMismatch) {
		t.Fatalf("want ErrDimensionMismatch, got %v", err)
	}
}

// --- ewBroadcastSubRows -------------------------------------------------------

func TestEwBroadcastSubRows_FastAndFallback_Match(t *testing.T) {
	t.Parallel()

	X := NewFilledDense(t, 2, 3, []float64{1, 2, 3, 10, 20, 30})
	rowMeans := []float64{2, 20}

	gotFast, err := matrix.EwBroadcastSubRowsTestOnly(X, rowMeans)
	if err != nil {
		t.Fatalf("fast: %v", err)
	}
	gotSlow, err := matrix.EwBroadcastSubRowsTestOnly(hide{X}, rowMeans)
	if err != nil {
		t.Fatalf("slow: %v", err)
	}

	exp := [][]float64{
		{-1, 0, 1},
		{-10, 0, 10},
	}
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			a := MustAt(t, gotFast, i, j)
			b := MustAt(t, gotSlow, i, j)
			if a != exp[i][j] || b != exp[i][j] {
				t.Fatalf("subRows[%d,%d]: fast=%v slow=%v want=%v", i, j, a, b, exp[i][j])
			}
		}
	}
}

func TestEwBroadcastSubRows_DimMismatch_Err(t *testing.T) {
	t.Parallel()
	X := NewFilledDense(t, 2, 3, []float64{1, 2, 3, 4, 5, 6})
	_, err := matrix.EwBroadcastSubRowsTestOnly(X, []float64{0})
	if !errors.Is(err, matrix.ErrDimensionMismatch) {
		t.Fatalf("want ErrDimensionMismatch, got %v", err)
	}
}

// --- ewScaleCols --------------------------------------------------------------

func TestEwScaleCols_FastAndFallback_Match(t *testing.T) {
	t.Parallel()

	X := NewFilledDense(t, 2, 3, []float64{1, 2, 3, -1, -2, -3})
	scale := []float64{10, 0.5, -2}

	gotFast, err := matrix.EwScaleColsTestOnly(X, scale)
	if err != nil {
		t.Fatalf("fast: %v", err)
	}
	gotSlow, err := matrix.EwScaleColsTestOnly(hide{X}, scale)
	if err != nil {
		t.Fatalf("slow: %v", err)
	}

	exp := [][]float64{
		{10, 1, -6},
		{-10, -1, 6},
	}
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			a := MustAt(t, gotFast, i, j)
			b := MustAt(t, gotSlow, i, j)
			if a != exp[i][j] || b != exp[i][j] {
				t.Fatalf("scaleCols[%d,%d]: fast=%v slow=%v want=%v", i, j, a, b, exp[i][j])
			}
		}
	}
}

func TestEwScaleCols_DimMismatch_Err(t *testing.T) {
	t.Parallel()
	X := NewFilledDense(t, 2, 3, []float64{1, 2, 3, 4, 5, 6})
	_, err := matrix.EwScaleColsTestOnly(X, []float64{1, 2})
	if !errors.Is(err, matrix.ErrDimensionMismatch) {
		t.Fatalf("want ErrDimensionMismatch, got %v", err)
	}
}

// --- ewScaleRows --------------------------------------------------------------

func TestEwScaleRows_FastAndFallback_Match(t *testing.T) {
	t.Parallel()

	X := NewFilledDense(t, 2, 3, []float64{1, 2, 3, -1, -2, -3})
	scale := []float64{3, -0.5}

	gotFast, err := matrix.EwScaleRowsTestOnly(X, scale)
	if err != nil {
		t.Fatalf("fast: %v", err)
	}
	gotSlow, err := matrix.EwScaleRowsTestOnly(hide{X}, scale)
	if err != nil {
		t.Fatalf("slow: %v", err)
	}

	exp := [][]float64{
		{3, 6, 9},
		{0.5, 1, 1.5}, // -0.5 * [-1,-2,-3] = [0.5,1,1.5]
	}
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			a := MustAt(t, gotFast, i, j)
			b := MustAt(t, gotSlow, i, j)
			if a != exp[i][j] || b != exp[i][j] {
				t.Fatalf("scaleRows[%d,%d]: fast=%v slow=%v want=%v", i, j, a, b, exp[i][j])
			}
		}
	}
}

func TestEwScaleRows_DimMismatch_Err(t *testing.T) {
	t.Parallel()
	X := NewFilledDense(t, 2, 3, []float64{1, 2, 3, 4, 5, 6})
	_, err := matrix.EwScaleRowsTestOnly(X, []float64{1})
	if !errors.Is(err, matrix.ErrDimensionMismatch) {
		t.Fatalf("want ErrDimensionMismatch, got %v", err)
	}
}

// --- ewReplaceInfNaN ----------------------------------------------------------

func TestEwReplaceInfNaN_ReplacesNonFinite(t *testing.T) {
	t.Parallel()

	// Build a dirty matrix via raw ingest (Fill), not via Set().
	const repl = 7
	bad := []float64{0, math.Inf(1), math.Inf(-1), math.NaN(), 1.5, -2.0}
	X, _ := matrix.NewPreparedDense(2, 3, matrix.WithNoValidateNaNInf())
	MustFillRowMajor(t, X, bad)
	got, err := matrix.EwReplaceInfNaNTestOnly(X, repl)
	if err != nil {
		t.Fatalf("ReplaceInfNaN: %v", err)
	}

	exp := []float64{0, repl, repl, repl, 1.5, -2}
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			v := MustAt(t, got, i, j)
			if v != exp[i*3+j] {
				t.Fatalf("ReplaceInfNaN[%d,%d]: got=%v want=%v", i, j, v, exp[i*3+j])
			}
		}
	}
}

func TestEwReplaceInfNaN_InvalidReplacement_Err(t *testing.T) {
	t.Parallel()

	X := NewFilledDense(t, 1, 1, []float64{0})
	if _, err := matrix.EwReplaceInfNaNTestOnly(X, math.NaN()); !errors.Is(err, matrix.ErrNaNInf) {
		t.Fatalf("want ErrNaNInf, got %v", err)
	}
	if _, err := matrix.EwReplaceInfNaNTestOnly(X, math.Inf(1)); !errors.Is(err, matrix.ErrNaNInf) {
		t.Fatalf("want ErrNaNInf, got %v", err)
	}
}

// --- ewClipRange --------------------------------------------------------------

func TestEwClipRange_ClampAndSwap(t *testing.T) {
	t.Parallel()

	X := NewFilledDense(t, 2, 3, []float64{-10, -1, 0, 1, 5, 10})

	gotSwap, err := matrix.EwClipRangeTestOnly(X, 5, -1) // swap to [-1,5]
	if err != nil {
		t.Fatalf("swap: %v", err)
	}
	gotNorm, err := matrix.EwClipRangeTestOnly(X, -1, 5)
	if err != nil {
		t.Fatalf("norm: %v", err)
	}

	exp := []float64{-1, -1, 0, 1, 5, 5}
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			a := MustAt(t, gotSwap, i, j)
			b := MustAt(t, gotNorm, i, j)
			if a != exp[i*3+j] || b != exp[i*3+j] {
				t.Fatalf("clip[%d,%d]: swap=%v norm=%v want=%v", i, j, a, b, exp[i*3+j])
			}
		}
	}
}

func TestEwClipRange_InvalidBounds_Err(t *testing.T) {
	t.Parallel()

	X := NewFilledDense(t, 1, 1, []float64{0})
	if _, err := matrix.EwClipRangeTestOnly(X, math.Inf(1), 0); !errors.Is(err, matrix.ErrNaNInf) {
		t.Fatalf("want ErrNaNInf, got %v", err)
	}
	if _, err := matrix.EwClipRangeTestOnly(X, -1, math.NaN()); !errors.Is(err, matrix.ErrNaNInf) {
		t.Fatalf("want ErrNaNInf, got %v", err)
	}
}

// --- ewAllClose ---------------------------------------------------------------

func TestEwAllClose_BasicTruthTable(t *testing.T) {
	t.Parallel()

	a := NewFilledDense(t, 2, 2, []float64{0, 0, 0, 0})
	b := NewFilledDense(t, 2, 2, []float64{0, 0, 0, 0})

	ok, err := matrix.EwAllCloseTestOnly(a, b, 1e-8, 1e-8)
	if err != nil || !ok {
		t.Fatalf("identical: ok=%v err=%v", ok, err)
	}

	_ = b.Set(1, 1, 1e-10)
	ok, err = matrix.EwAllCloseTestOnly(a, b, 1e-8, 1e-8)
	if err != nil || !ok {
		t.Fatalf("within tol: ok=%v err=%v", ok, err)
	}

	_ = b.Set(0, 0, 1e-6)
	ok, err = matrix.EwAllCloseTestOnly(a, b, 0, 1e-8)
	if err != nil {
		t.Fatalf("outside err: %v", err)
	}
	if ok {
		t.Fatalf("outside: expected false, got true")
	}
}

func TestEwAllClose_ErrorsAndNormalization(t *testing.T) {
	t.Parallel()

	a := NewFilledDense(t, 2, 2, []float64{0, 0, 0, 0})
	b := NewFilledDense(t, 2, 3, []float64{0, 0, 0, 0, 0, 0})
	if _, err := matrix.EwAllCloseTestOnly(a, b, 1e-6, 1e-6); !errors.Is(err, matrix.ErrDimensionMismatch) {
		t.Fatalf("dim mismatch: %v", err)
	}
	if _, err := matrix.EwAllCloseTestOnly(nil, a, 1e-6, 1e-6); !errors.Is(err, matrix.ErrNilMatrix) {
		t.Fatalf("nil a: %v", err)
	}
	if _, err := matrix.EwAllCloseTestOnly(a, nil, 1e-6, 1e-6); !errors.Is(err, matrix.ErrNilMatrix) {
		t.Fatalf("nil b: %v", err)
	}
	if _, err := matrix.EwAllCloseTestOnly(a, a, math.NaN(), 1e-6); !errors.Is(err, matrix.ErrNaNInf) {
		t.Fatalf("rtol NaN: %v", err)
	}
	if _, err := matrix.EwAllCloseTestOnly(a, a, 1e-6, math.Inf(-1)); !errors.Is(err, matrix.ErrNaNInf) {
		t.Fatalf("atol Inf: %v", err)
	}

	c := NewFilledDense(t, 1, 1, []float64{5e-6})
	ok, err := matrix.EwAllCloseTestOnly(NewFilledDense(t, 1, 1, []float64{0}), c, -1e-5, 1e-5) // negatives abs-ed
	if err != nil {
		t.Fatalf("neg tol err: %v", err)
	}
	if !ok {
		t.Fatalf("neg tol expected true")
	}
}

// -------------------------
// --- additional-checks ---
// -------------------------

// --- Clip ---------------------------------------------------------------------

// Clip should clamp to [lo,hi]; when lo>hi it must swap bounds; non-finite bounds must error.
func TestClip_SwapsBounds_And_Clamps(t *testing.T) {
	t.Parallel()

	// Build 2×3 with a range of finite values to exercise both ends.
	M := NewFilledDense(t, 2, 3, []float64{-10, -1, 0, 1, 5, 10})

	// Case A: lo > hi, must normalize to [-1,5].
	gotA, err := matrix.Clip(M, 5, -1)
	if err != nil {
		t.Fatalf("Clip swap: %v", err)
	}
	// Case B: explicit normalized bounds.
	gotB, err := matrix.Clip(M, -1, 5)
	if err != nil {
		t.Fatalf("Clip normalized: %v", err)
	}

	// Check clamped values and that swap path equals normalized path.
	exp := []float64{-1, -1, 0, 1, 5, 5}
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			a := MustAt(t, gotA, i, j)
			b := MustAt(t, gotB, i, j)
			e := exp[i*3+j]
			if a != e || b != e {
				t.Fatalf("Clip[%d,%d]: gotA=%v gotB=%v want=%v", i, j, a, b, e)
			}
		}
	}
}

func TestClip_InvalidBounds_Error(t *testing.T) {
	t.Parallel()

	M := NewFilledDense(t, 1, 1, []float64{0})

	// lo=+Inf → ErrNaNInf
	if _, err := matrix.Clip(M, math.Inf(1), 0); !errors.Is(err, matrix.ErrNaNInf) {
		t.Fatalf("Clip(+Inf,0): want ErrNaNInf, got %v", err)
	}
	// hi=NaN → ErrNaNInf
	if _, err := matrix.Clip(M, -1, math.NaN()); !errors.Is(err, matrix.ErrNaNInf) {
		t.Fatalf("Clip(-1,NaN): want ErrNaNInf, got %v", err)
	}
}

// --- ReplaceInfNaN ------------------------------------------------------------

// ReplaceInfNaN must replace all {NaN, ±Inf} with a finite value; reject non-finite replacement.
func TestReplaceInfNaN_BasicAndInvalidVal(t *testing.T) {
	t.Parallel()

	// Input contains 0, +Inf, -Inf, NaN, and two finite normals.
	// Build a dirty matrix via raw ingest (Fill), not via Set().
	bad := []float64{0, math.Inf(1), math.Inf(-1), math.NaN(), 1.5, -2.0}
	M, _ := matrix.NewPreparedDense(2, 3, matrix.WithNoValidateNaNInf())

	MustFillRowMajor(t, M, bad)
	// Valid replacement value (finite).
	const repl = 42.0
	out, err := matrix.ReplaceInfNaN(M, repl)
	if err != nil {
		t.Fatalf("ReplaceInfNaN valid: %v", err)
	}

	// Validate every entry is finite and Inf/NaN are replaced.
	want := []float64{0, repl, repl, repl, 1.5, -2.0}
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			v := MustAt(t, out, i, j)
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("ReplaceInfNaN produced non-finite at [%d,%d]: %v", i, j, v)
			}
			if v != want[i*3+j] {
				t.Fatalf("ReplaceInfNaN[%d,%d]: got %v, want %v", i, j, v, want[i*3+j])
			}
		}
	}

	// Invalid replacement values must be rejected.
	if _, err := matrix.ReplaceInfNaN(M, math.Inf(1)); !errors.Is(err, matrix.ErrNaNInf) {
		t.Fatalf("ReplaceInfNaN val=+Inf: want ErrNaNInf, got %v", err)
	}
	if _, err := matrix.ReplaceInfNaN(M, math.NaN()); !errors.Is(err, matrix.ErrNaNInf) {
		t.Fatalf("ReplaceInfNaN val=NaN: want ErrNaNInf, got %v", err)
	}
}

func TestReplaceInfNaN_ReplacesAllNonFinite_WithFiniteValue(t *testing.T) {
	t.Parallel()

	//A := MustDense(t, 2, 3)
	A, _ := matrix.NewPreparedDense(2, 3, matrix.WithNoValidateNaNInf())
	bad := []float64{1, math.Inf(+1), -3.5, math.NaN(), 0, math.Inf(-1)}
	// Raw-ingest to avoid Set() numeric-policy on NaN/Inf.
	MustFillRowMajor(t, A, bad)
	const rep = 7.0

	B, err := matrix.ReplaceInfNaN(A, rep)
	if err != nil {
		t.Fatalf("ReplaceInfNaN: %v", err)
	}

	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			v := MustAt(t, B, i, j)
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("non-finite survived at [%d,%d]: %v", i, j, v)
			}
			orig := bad[i*3+j]
			if math.IsNaN(orig) || math.IsInf(orig, 0) {
				if v != rep {
					t.Fatalf("expected replacement at [%d,%d]: got %v, want %v", i, j, v, rep)
				}
			} else if v != orig {
				t.Fatalf("finite changed unexpectedly at [%d,%d]: got %v, want %v", i, j, v, orig)
			}
		}
	}
}

func TestReplaceInfNaN_InvalidReplacement_ReturnsErrNaNInf(t *testing.T) {
	t.Parallel()

	A := MustDense(t, 1, 1)
	_ = A.Set(0, 0, 0)

	if _, err := matrix.ReplaceInfNaN(A, math.NaN()); !errors.Is(err, matrix.ErrNaNInf) {
		t.Fatalf("want ErrNaNInf for NaN replacement, got %v", err)
	}
	if _, err := matrix.ReplaceInfNaN(A, math.Inf(-1)); !errors.Is(err, matrix.ErrNaNInf) {
		t.Fatalf("want ErrNaNInf for Inf replacement, got %v", err)
	}
	// nil matrix => ErrNilMatrix
	var nilM matrix.Matrix
	if _, err := matrix.ReplaceInfNaN(nilM, 0); !errors.Is(err, matrix.ErrNilMatrix) {
		t.Fatalf("want ErrNilMatrix, got %v", err)
	}
}

// --- AllClose -----------------------------------------------------------------

// AllClose basic truth table: identical → true; within tolerance → true; outside → false.
func TestAllClose_Basics(t *testing.T) {
	t.Parallel()

	a := NewFilledDense(t, 2, 2, []float64{0, 0, 0, 0})
	b := NewFilledDense(t, 2, 2, []float64{0, 0, 0, 0})

	// Identical.
	ok, err := matrix.AllClose(a, b, 1e-8, 1e-8)
	if err != nil || !ok {
		t.Fatalf("AllClose identical: ok=%v err=%v", ok, err)
	}

	// Slightly different but within tolerance.
	MustSet(t, b, 1, 1, 1e-10)
	ok, err = matrix.AllClose(a, b, 1e-8, 1e-8)
	if err != nil || !ok {
		t.Fatalf("AllClose within tol: ok=%v err=%v", ok, err)
	}

	// Outside tolerance (tight atol).
	MustSet(t, b, 0, 0, 1e-6)
	ok, err = matrix.AllClose(a, b, 0, 1e-8) // pure absolute tolerance
	if err != nil {
		t.Fatalf("AllClose outside tol err: %v", err)
	}
	if ok {
		t.Fatalf("AllClose outside tol: expected false, got true")
	}
}

// AllClose errors: shape mismatch, nil matrices, bad tolerances.
func TestAllClose_Errors(t *testing.T) {
	t.Parallel()

	a := NewFilledDense(t, 2, 2, []float64{0, 0, 0, 0})
	// Shape mismatch: 2×3
	b3x := NewFilledDense(t, 2, 3, []float64{0, 0, 0, 0, 0, 0})

	// Dimension mismatch.
	if _, err := matrix.AllClose(a, b3x, 1e-6, 1e-6); !errors.Is(err, matrix.ErrDimensionMismatch) {
		t.Fatalf("AllClose dim mismatch: want ErrDimensionMismatch, got %v", err)
	}

	// Nil matrices.
	if _, err := matrix.AllClose(nil, a, 1e-6, 1e-6); !errors.Is(err, matrix.ErrNilMatrix) {
		t.Fatalf("AllClose nil a: want ErrNilMatrix, got %v", err)
	}
	if _, err := matrix.AllClose(a, nil, 1e-6, 1e-6); !errors.Is(err, matrix.ErrNilMatrix) {
		t.Fatalf("AllClose nil b: want ErrNilMatrix, got %v", err)
	}

	// Bad tolerances (NaN/Inf).
	if _, err := matrix.AllClose(a, a, math.NaN(), 1e-6); !errors.Is(err, matrix.ErrNaNInf) {
		t.Fatalf("AllClose rtol NaN: want ErrNaNInf, got %v", err)
	}
	if _, err := matrix.AllClose(a, a, 1e-6, math.Inf(-1)); !errors.Is(err, matrix.ErrNaNInf) {
		t.Fatalf("AllClose atol Inf: want ErrNaNInf, got %v", err)
	}
}

func TestAllClose_ZeroShapeVacuouslyTrue(t *testing.T) {
	t.Parallel()

	a, err := matrix.NewDense(0, 3)
	if err != nil {
		t.Fatalf("NewDense a: %v", err)
	}
	b, err := matrix.NewDense(0, 3)
	if err != nil {
		t.Fatalf("NewDense b: %v", err)
	}

	ok, err := matrix.AllClose(a, b, 1e-9, 1e-9)
	if err != nil {
		t.Fatalf("AllClose: %v", err)
	}
	if !ok {
		t.Fatal("zero-shape matrices with identical shape must be close")
	}
}

// Negative tolerances must be accepted and treated as absolute values.
func TestAllClose_NegativeTolerances_AreNormalized(t *testing.T) {
	t.Parallel()

	a := NewFilledDense(t, 1, 1, []float64{0})
	b := NewFilledDense(t, 1, 1, []float64{5e-6})

	ok, err := matrix.AllClose(a, b, -1e-5, 1e-5) // normalized to 1e-5 and 1e-9
	if err != nil {
		t.Fatalf("AllClose negative tol err: %v", err)
	}
	if !ok {
		t.Fatalf("AllClose negative tol: expected true, got false")
	}
}

// Fast path (*Dense) and fallback (non-*Dense) must agree on the boolean result.
func TestAllClose_FallbackMatchesFast(t *testing.T) {
	t.Parallel()

	a := NewFilledDense(t, 2, 2, []float64{0, 0, 0, 0})
	b := NewFilledDense(t, 2, 2, []float64{0, 1e-9, 0, 0})

	// Fast path: both *Dense.
	okFast, err := matrix.AllClose(a, b, 1e-8, 1e-8)
	if err != nil {
		t.Fatalf("AllClose fast err: %v", err)
	}

	// Fallback: hide both behind interface wrapper.
	okSlow, err := matrix.AllClose(hide{a}, hide{b}, 1e-8, 1e-8)
	if err != nil {
		t.Fatalf("AllClose slow err: %v", err)
	}

	if okFast != okSlow {
		t.Fatalf("AllClose mismatch fast=%v slow=%v", okFast, okSlow)
	}
}
