// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix_test validates public elementwise operations and verifies
// observable Dense/fallback equivalence without relying on private hooks.
package matrix_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/matrix"
)

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
	if _, err = matrix.ReplaceInfNaN(M, math.Inf(1)); !errors.Is(err, matrix.ErrNaNInf) {
		t.Fatalf("ReplaceInfNaN val=+Inf: want ErrNaNInf, got %v", err)
	}
	if _, err = matrix.ReplaceInfNaN(M, math.NaN()); !errors.Is(err, matrix.ErrNaNInf) {
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
