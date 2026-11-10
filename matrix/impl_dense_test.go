// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix_test contains unit tests for the Dense implementation
// of the Matrix interface in the matrix package.
package matrix_test

import (
	"errors"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/lvlath/go/matrix"
)

// TestNewDenseInvalidDimensions ensures that NewDense rejects negative dimensions
// but accepts zero-sized shapes.
func TestNewDenseInvalidDimensions(t *testing.T) {
	// Negative rows.
	if _, err := matrix.NewDense(-1, 5); !errors.Is(err, matrix.ErrInvalidDimensions) {
		t.Fatalf("rows=-1 must fail with ErrInvalidDimensions; got %v", err)
	}
	// Negative cols.
	if _, err := matrix.NewDense(5, -1); !errors.Is(err, matrix.ErrInvalidDimensions) {
		t.Fatalf("cols=-1 must fail with ErrInvalidDimensions; got %v", err)
	}

	// Zero-sized shapes are legal and must not fail.
	if _, err := matrix.NewDense(0, 5); err != nil {
		t.Fatalf("rows=0 must be allowed; got %v", err)
	}
	if _, err := matrix.NewDense(5, 0); err != nil {
		t.Fatalf("cols=0 must be allowed; got %v", err)
	}
	if _, err := matrix.NewDense(0, 0); err != nil {
		t.Fatalf("0x0 must be allowed; got %v", err)
	}
}

func TestNewDenseDefaultZero(t *testing.T) {
	t.Run("NonZeroShapes", func(t *testing.T) {
		m, err := matrix.NewDense(3, 4)
		if err != nil {
			t.Fatalf("NewDense(3,4): %v", err)
		}
		// immediately after creation all elements should be 0
		var i, j int // loop iterators
		var v float64
		for i = 0; i < m.Rows(); i++ {
			for j = 0; j < m.Cols(); j++ {
				v, err = m.At(i, j)
				if err != nil {
					t.Fatalf("At(%d,%d): %v", i, j, err)
				}
				if v != 0.0 {
					t.Fatalf("default value at (%d,%d) must be 0; got %v", i, j, v)
				}
			}
		}
	})

	t.Run("ZeroSizedShapes", func(t *testing.T) {
		if m, err := matrix.NewDense(0, 5); err != nil || m.Rows() != 0 || m.Cols() != 5 {
			t.Fatalf("0x5: got (%d,%d), err=%v", m.Rows(), m.Cols(), err)
		}
		if m, err := matrix.NewDense(5, 0); err != nil || m.Rows() != 5 || m.Cols() != 0 {
			t.Fatalf("5x0: got (%d,%d), err=%v", m.Rows(), m.Cols(), err)
		}
		if m, err := matrix.NewDense(0, 0); err != nil || m.Rows() != 0 || m.Cols() != 0 {
			t.Fatalf("0x0: got (%d,%d), err=%v", m.Rows(), m.Cols(), err)
		}
	})
}

func TestNewDense_ZeroShapesAreValid(t *testing.T) {
	t.Parallel()

	cases := []struct {
		rows int
		cols int
	}{
		{0, 0},
		{0, 3},
		{3, 0},
	}

	for _, tc := range cases {
		t.Run(fmt.Sprintf("%dx%d", tc.rows, tc.cols), func(t *testing.T) {
			t.Parallel()

			m, err := matrix.NewDense(tc.rows, tc.cols)
			if err != nil {
				t.Fatalf("NewDense(%d,%d): %v", tc.rows, tc.cols, err)
			}
			if m.Rows() != tc.rows || m.Cols() != tc.cols {
				t.Fatalf("shape: got %dx%d, want %dx%d",
					m.Rows(), m.Cols(), tc.rows, tc.cols)
			}
			if _, err = m.At(0, 0); !errors.Is(err, matrix.ErrOutOfRange) {
				t.Fatalf("At(0,0): got %v, want ErrOutOfRange", err)
			}
		})
	}
}

// TestRowsColsShape verifies that Rows/Cols/Shape return correct dimensions.
func TestRowsColsShape(t *testing.T) {
	m, err := matrix.NewDense(3, 4)
	if err != nil {
		t.Fatalf("NewDense: %v", err)
	}
	if got := m.Rows(); got != 3 {
		t.Fatalf("Rows: got %d, want 3", got)
	}
	if got := m.Cols(); got != 4 {
		t.Fatalf("Cols: got %d, want 4", got)
	}
	r, c := m.Shape()
	if r != 3 || c != 4 {
		t.Fatalf("Shape: got (%d,%d), want (3,4)", r, c)
	}
}

// TestValidateSquare_ErrNonSquare ensures non-square matrices fail with ErrNonSquare.
func TestValidateSquare_ErrNonSquare(t *testing.T) {
	m, err := matrix.NewDense(2, 3)
	if err != nil {
		t.Fatalf("NewDense: %v", err)
	}
	if err = matrix.ValidateSquare(m); !errors.Is(err, matrix.ErrNonSquare) {
		t.Fatalf("ValidateSquare(2x3): want ErrNonSquare, got %v", err)
	}
}

// TestAtSetOutOfRange exercises ErrOutOfRange on invalid indices (no panics).
func TestAtSetOutOfRange(t *testing.T) {
	m, err := matrix.NewDense(2, 2)
	if err != nil {
		t.Fatalf("NewDense: %v", err)
	}

	// At out of bounds
	if _, err = m.At(-1, 0); !errors.Is(err, matrix.ErrOutOfRange) {
		t.Fatalf("At(-1,0) expected ErrOutOfRange, got %v", err)
	}
	if _, err = m.At(0, 2); !errors.Is(err, matrix.ErrOutOfRange) {
		t.Fatalf("At(0,2) expected ErrOutOfRange, got %v", err)
	}

	// Set out of bounds
	if err = m.Set(2, 0, 1.0); !errors.Is(err, matrix.ErrOutOfRange) {
		t.Fatalf("Set(2,0) expected ErrOutOfRange, got %v", err)
	}
	if err = m.Set(0, -1, 1.0); !errors.Is(err, matrix.ErrOutOfRange) {
		t.Fatalf("Set(0,-1) expected ErrOutOfRange, got %v", err)
	}
}

// TestSetGet validates Set followed by At on valid indices.
func TestSetGet(t *testing.T) {
	m, err := matrix.NewDense(2, 3)
	if err != nil {
		t.Fatalf("NewDense: %v", err)
	}
	if err = m.Set(1, 2, 7.89); err != nil {
		t.Fatalf("Set: %v", err)
	}
	val, err := m.At(1, 2)
	if err != nil {
		t.Fatalf("At: %v", err)
	}
	if val != 7.89 {
		t.Fatalf("unexpected value: got %v, want 7.89", val)
	}
}

// TestNaNInfPolicy ensures default ValidateNaNInf rejects NaN and ±Inf.
func TestNaNInfPolicy(t *testing.T) {
	m, err := matrix.NewDense(1, 1)
	if err != nil {
		t.Fatalf("NewDense: %v", err)
	}
	if err = m.Set(0, 0, math.NaN()); !errors.Is(err, matrix.ErrNaNInf) {
		t.Fatalf("Set NaN: expected ErrNaNInf, got %v", err)
	}
	if err = m.Set(0, 0, math.Inf(1)); !errors.Is(err, matrix.ErrNaNInf) {
		t.Fatalf("Set +Inf: expected ErrNaNInf, got %v", err)
	}
	if err = m.Set(0, 0, math.Inf(-1)); !errors.Is(err, matrix.ErrNaNInf) {
		t.Fatalf("Set -Inf: expected ErrNaNInf, got %v", err)
	}
}

// TestFill_Basic
func TestFill_Basic(t *testing.T) {
	m, err := matrix.NewDense(2, 3)
	if err != nil {
		t.Fatalf("NewDense: %v", err)
	}
	data := []float64{1, 2, 3, 4, 5, 6}
	if err = m.Fill(data); err != nil {
		t.Fatalf("Fill: %v", err)
	}
	want := [][]float64{{1, 2, 3}, {4, 5, 6}}
	var i, j int // loop iterators
	var got float64
	for i = 0; i < m.Rows(); i++ {
		for j = 0; j < m.Cols(); j++ {
			got, err = m.At(i, j)
			if err != nil {
				t.Fatalf("At(%d,%d): %v", i, j, err)
			}
			if got != want[i][j] {
				t.Fatalf("Fill mismatch at (%d,%d): got %v, want %v", i, j, got, want[i][j])
			}
		}
	}
}

// TestFill_ZeroSize
func TestFill_ZeroSize(t *testing.T) {
	// 0x0
	m0, err := matrix.NewDense(0, 0)
	if err != nil {
		t.Fatalf("NewDense(0,0): %v", err)
	}
	if err = m0.Fill([]float64{}); err != nil {
		t.Fatalf("Fill(0x0): %v", err)
	}

	// 0x3
	m1, err := matrix.NewDense(0, 3)
	if err != nil {
		t.Fatalf("NewDense(0,3): %v", err)
	}
	if err = m1.Fill([]float64{}); err != nil {
		t.Fatalf("Fill(0x3): %v", err)
	}

	// 3x0
	m2, err := matrix.NewDense(3, 0)
	if err != nil {
		t.Fatalf("NewDense(3,0): %v", err)
	}
	if err = m2.Fill([]float64{}); err != nil {
		t.Fatalf("Fill(3x0): %v", err)
	}
}

// TestFill_InvalidLength
func TestFill_InvalidLength(t *testing.T) {
	m, err := matrix.NewDense(2, 2)
	if err != nil {
		t.Fatalf("NewDense: %v", err)
	}
	// Expect ErrInvalidDimensions for mismatched length.
	if err = matrix.ExportedDenseFill(m, []float64{1, 2, 3}); !errors.Is(err, matrix.ErrInvalidDimensions) {
		t.Fatalf("Fill invalid length: expected ErrInvalidDimensions, got %v", err)
	}
}

// TestFill_PolicyEnforced ensures Fill enforces the same numeric policy as Set:
//   - NaN and -Inf are always rejected under validation.
//   - +Inf is rejected by default, but allowed when allowInfDistances=true.
func TestFill_PolicyEnforced(t *testing.T) {
	t.Run("DefaultPolicyRejectsInf", func(t *testing.T) {
		m, err := matrix.NewDense(1, 2)
		if err != nil {
			t.Fatalf("NewDense: %v", err)
		}
		// +Inf must be rejected by default policy.
		if err = m.Fill([]float64{math.Inf(1), 0}); !errors.Is(err, matrix.ErrNaNInf) {
			t.Fatalf("Fill +Inf (default policy): expected ErrNaNInf, got %v", err)
		}
		// NaN must be rejected by default policy.
		if err = m.Fill([]float64{math.NaN(), 0}); !errors.Is(err, matrix.ErrNaNInf) {
			t.Fatalf("Fill NaN (default policy): expected ErrNaNInf, got %v", err)
		}
		// -Inf must be rejected by default policy.
		if err = m.Fill([]float64{math.Inf(-1), 0}); !errors.Is(err, matrix.ErrNaNInf) {
			t.Fatalf("Fill -Inf (default policy): expected ErrNaNInf, got %v", err)
		}
	})

	t.Run("AllowInfDistancesAcceptsPosInf", func(t *testing.T) {
		// Allocate with explicit distance policy to allow +Inf.
		m, err := matrix.NewPreparedDense(1, 2, matrix.WithAllowInfDistances())
		if err != nil {
			t.Fatalf("NewPreparedDense: %v", err)
		}
		// +Inf must be accepted under allowInfDistances.
		if err = m.Fill([]float64{math.Inf(1), 0}); err != nil {
			t.Fatalf("Fill +Inf (allowInfDistances): %v", err)
		}
		// Read back and confirm the sentinel is preserved.
		v, err := m.At(0, 0)
		if err != nil {
			t.Fatalf("At: %v", err)
		}
		if !math.IsInf(v, 1) {
			t.Fatalf("expected +Inf stored at (0,0), got %v", v)
		}
	})
}

// TestCloneIndependence ensures Clone produces an independent deep copy
// and preserves numeric policy (strict by default).
func TestCloneIndependence(t *testing.T) {
	m, err := matrix.NewDense(2, 2)
	if err != nil {
		t.Fatalf("NewDense: %v", err)
	}
	_ = m.Set(0, 0, 1.0)
	_ = m.Set(1, 1, 2.0)

	clone := m.Clone()
	if err = clone.Set(0, 0, 3.0); err != nil {
		t.Fatalf("clone.Set: %v", err)
	}
	orig, err := m.At(0, 0)
	if err != nil {
		t.Fatalf("m.At: %v", err)
	}
	clv, err := clone.At(0, 0)
	if err != nil {
		t.Fatalf("clone.At: %v", err)
	}
	if orig != 1.0 || clv != 3.0 {
		t.Fatalf("clone independence mismatch: orig=%v clone=%v", orig, clv)
	}

	// Policy preservation: default is strict, so ErrNaNInf on clone too.
	if err = clone.Set(0, 1, math.NaN()); !errors.Is(err, matrix.ErrNaNInf) {
		t.Fatalf("clone NaN policy: expected ErrNaNInf, got %v", err)
	}
}

// TestClone_PreservesAllowInfDistances ensures distance-policy survives cloning.
func TestClone_PreservesAllowInfDistances(t *testing.T) {
	m, err := matrix.NewPreparedDense(1, 2, matrix.WithAllowInfDistances())
	if err != nil {
		t.Fatalf("NewPreparedDense: %v", err)
	}
	// Store +Inf (must succeed under distance policy).
	if err = m.Set(0, 0, math.Inf(1)); err != nil {
		t.Fatalf("Set +Inf: %v", err)
	}
	clone := m.Clone()
	// Clone must still allow +Inf writes (policy preservation).
	if err = clone.Set(0, 1, math.Inf(1)); err != nil {
		t.Fatalf("clone.Set +Inf: %v", err)
	}
}

// TestStringFormat checks that String formats the matrix as expected.
func TestStringFormat(t *testing.T) {
	m, err := matrix.NewDense(2, 2)
	if err != nil {
		t.Fatalf("NewDense: %v", err)
	}
	_ = m.Set(0, 0, 1)
	_ = m.Set(0, 1, 2)
	_ = m.Set(1, 0, 3)
	_ = m.Set(1, 1, 4)

	got := m.String()
	want := "[1, 2]\n[3, 4]\n"
	if got != want {
		t.Fatalf("String mismatch:\n got: %q\nwant: %q", got, want)
	}
	if strings.Count(got, "\n") != 2 {
		t.Fatalf("expected two newline-terminated rows, got: %q", got)
	}
}

// TestViewBoundsAndMutation validates view bounds and write-through semantics.
func TestViewBoundsAndMutation(t *testing.T) {
	m, err := matrix.NewDense(3, 3)
	if err != nil {
		t.Fatalf("NewDense: %v", err)
	}
	// Fill base deterministically
	v := 1.0
	var i, j int
	for i = 0; i < m.Rows(); i++ {
		for j = 0; j < m.Cols(); j++ {
			if err = m.Set(i, j, v); err != nil {
				t.Fatalf("fill: %v", err)
			}
			v++
		}
	}

	// Invalid: window exceeds bounds
	if _, err = m.View(1, 1, 3, 3); !errors.Is(err, matrix.ErrOutOfRange) {
		t.Fatalf("View out of bounds must return ErrOutOfRange; got %v", err)
	}
	// Valid 2x2 view from (1,1)
	view, err := m.View(1, 1, 2, 2)
	if err != nil {
		t.Fatalf("View: %v", err)
	}
	// Read via view
	val, err := view.At(0, 0)
	if err != nil {
		t.Fatalf("view.At: %v", err)
	}
	if base, _ := m.At(1, 1); val != base {
		t.Fatalf("view read mismatch: view=%v base=%v", val, base)
	}
	// Write via view and observe in base
	if err = view.Set(1, 1, 123.0); err != nil {
		t.Fatalf("view.Set: %v", err)
	}
	bv, _ := m.At(2, 2)
	if bv != 123.0 {
		t.Fatalf("write-through failed: base(2,2)=%v", bv)
	}
	// View bounds check
	if _, err = view.At(2, 0); !errors.Is(err, matrix.ErrOutOfRange) {
		t.Fatalf("view.At out-of-range must be ErrOutOfRange; got %v", err)
	}
}

// TestInducedCoversZeroSizedAndBounds validates Induced edge cases and bounds.
func TestInducedCoversZeroSizedAndBounds(t *testing.T) {
	m, err := matrix.NewDense(2, 3)
	if err != nil {
		t.Fatalf("NewDense: %v", err)
	}
	// set distinct values
	x := 1.0
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			_ = m.Set(i, j, x)
			x++
		}
	}

	// Zero-sized results are legal
	z1, err := m.Induced([]int{}, []int{0, 1})
	if err != nil || z1.Rows() != 0 || z1.Cols() != 2 {
		t.Fatalf("Induced zero×2 failed: z1=(%d,%d), err=%v", z1.Rows(), z1.Cols(), err)
	}
	z2, err := m.Induced([]int{0, 1}, []int{})
	if err != nil || z2.Rows() != 2 || z2.Cols() != 0 {
		t.Fatalf("Induced 2×zero failed: z2=(%d,%d), err=%v", z2.Rows(), z2.Cols(), err)
	}

	// Bounds checking on indices
	if _, err = m.Induced([]int{-1}, []int{0}); !errors.Is(err, matrix.ErrOutOfRange) {
		t.Fatalf("Induced with bad row must be ErrOutOfRange; got %v", err)
	}
	if _, err = m.Induced([]int{0}, []int{3}); !errors.Is(err, matrix.ErrOutOfRange) {
		t.Fatalf("Induced with bad col must be ErrOutOfRange; got %v", err)
	}

	// Correct copy with duplicates
	sub, err := m.Induced([]int{1, 1}, []int{2, 0})
	if err != nil {
		t.Fatalf("Induced: %v", err)
	}
	v00, _ := sub.At(0, 0) // from base(1,2)
	v01, _ := sub.At(0, 1) // from base(1,0)
	v10, _ := sub.At(1, 0) // duplicate row -> same as v00
	if v00 != 6 || v01 != 4 || v10 != 6 {
		t.Fatalf("Induced copy mismatch: got [%v %v; %v ...], want [6 4; 6 ...]", v00, v01, v10)
	}
}

// TestView_PolicyEnforced ensures View.Set respects the base allowInfDistances policy.
func TestView_PolicyEnforced(t *testing.T) {
	t.Run("DefaultPolicyRejectsInf", func(t *testing.T) {
		m, err := matrix.NewDense(2, 2)
		if err != nil {
			t.Fatalf("NewDense: %v", err)
		}
		vw, err := m.View(0, 0, 2, 2)
		if err != nil {
			t.Fatalf("View: %v", err)
		}
		if err = vw.Set(0, 0, math.Inf(1)); !errors.Is(err, matrix.ErrNaNInf) {
			t.Fatalf("view.Set +Inf (default): expected ErrNaNInf, got %v", err)
		}
	})
	t.Run("AllowInfDistancesAcceptsPosInf", func(t *testing.T) {
		m, err := matrix.NewPreparedDense(2, 2, matrix.WithAllowInfDistances())
		if err != nil {
			t.Fatalf("NewPreparedDense: %v", err)
		}
		vw, err := m.View(0, 0, 2, 2)
		if err != nil {
			t.Fatalf("View: %v", err)
		}
		if err = vw.Set(0, 0, math.Inf(1)); err != nil {
			t.Fatalf("view.Set +Inf (allowInfDistances): %v", err)
		}
		got, err := m.At(0, 0)
		if err != nil {
			t.Fatalf("At: %v", err)
		}
		if !math.IsInf(got, 1) {
			t.Fatalf("expected base(0,0)=+Inf, got %v", got)
		}
	})
}

// TestApplyAndDo ensures Apply enforces numeric policy and Do iterates deterministically.
func TestApplyAndDo(t *testing.T) {
	m, err := matrix.NewDense(2, 2)
	if err != nil {
		t.Fatalf("NewDense: %v", err)
	}
	// Fill with 1..4
	val := 1.0
	var i, j int
	for i = 0; i < 2; i++ {
		for j = 0; j < 2; j++ {
			if err = m.Set(i, j, val); err != nil {
				t.Fatalf("fill: %v", err)
			}
			val++
		}
	}
	// Apply a safe transform
	if err = m.Apply(func(_, _ int, v float64) float64 { return v * 2 }); err != nil {
		t.Fatalf("Apply safe: %v", err)
	}
	// Apply that yields NaN at a certain position
	err = m.Apply(func(i, j int, v float64) float64 {
		if i == 1 && j == 1 {
			return math.NaN()
		}
		return v
	})
	if !errors.Is(err, matrix.ErrNaNInf) {
		t.Fatalf("Apply NaN must fail with ErrNaNInf; got %v", err)
	}

	// Do walks in row-major order and can stop early.
	seen := 0
	m.Do(func(_, _ int, _ float64) bool {
		seen++
		return seen < 2 // stop after visiting 2 elements
	})
	if seen != 2 {
		t.Fatalf("Do early-stop: visited %d elements, want 2", seen)
	}
}

// TestApply_AllowsPosInfWhenConfigured ensures Apply does not reject +Inf
// when allowInfDistances is enabled (distance-policy matrices).
func TestApply_AllowsPosInfWhenConfigured(t *testing.T) {
	m, err := matrix.NewPreparedDense(1, 2, matrix.WithAllowInfDistances())
	if err != nil {
		t.Fatalf("NewPreparedDense: %v", err)
	}
	// Seed with +Inf and finite.
	if err = m.Set(0, 0, math.Inf(1)); err != nil {
		t.Fatalf("Set +Inf: %v", err)
	}
	if err = m.Set(0, 1, 1.0); err != nil {
		t.Fatalf("Set finite: %v", err)
	}
	// Identity Apply must not fail on +Inf under distance policy.
	if err = m.Apply(func(_, _ int, v float64) float64 { return v }); err != nil {
		t.Fatalf("Apply identity (allowInfDistances): %v", err)
	}
}

// TestNewDenseCheckedElementCount proves constructors reject overflowing
// positive shape products through ErrInvalidDimensions, before allocation.
// Zero-element shapes remain legal even when one dimension is very large.
func TestNewDenseCheckedElementCount(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	if got, err := matrix.NewDense(maxInt, 2); got != nil || !errors.Is(err, matrix.ErrInvalidDimensions) {
		t.Fatalf("NewDense overflowing shape: got=%v err=%v", got, err)
	}
	if got, err := matrix.NewPreparedDense(maxInt, 2); got != nil || !errors.Is(err, matrix.ErrInvalidDimensions) {
		t.Fatalf("NewPreparedDense overflowing shape: got=%v err=%v", got, err)
	}
	if got, err := matrix.NewDense(0, maxInt); err != nil || got == nil || got.Rows() != 0 || got.Cols() != maxInt {
		t.Fatalf("NewDense zero-domain shape: got=%v err=%v", got, err)
	}
}
