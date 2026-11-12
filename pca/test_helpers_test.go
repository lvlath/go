// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// External test helpers centralize error identity, nilability, and finite oracles.
// They deliberately do not import reflect or an assertion framework.
package pca_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/matrix"
)

// countReadsMatrix counts public Matrix.At calls without changing cell values.
// It is used only in a single-threaded contract test of reconstruction work.
type countReadsMatrix struct {
	matrix.Matrix
	reads int
}

func (m *countReadsMatrix) At(row, col int) (float64, error) {
	m.reads++
	return m.Matrix.At(row, col)
}

// fixture builds a fully owned finite matrix from a rectangular test ledger.
// An empty/ragged fixture is a test setup error, not an algorithmic observation.
func fixture(t *testing.T, rows [][]float64) *matrix.Dense {
	t.Helper()
	if len(rows) == 0 || len(rows[0]) == 0 {
		t.Fatal("empty test matrix")
	}
	m, err := matrix.NewDense(len(rows), len(rows[0]))
	if err != nil {
		t.Fatal(err)
	}
	for i, row := range rows {
		if len(row) != len(rows[0]) {
			t.Fatal("ragged fixture")
		}
		for j, v := range row {
			if err = m.Set(i, j, v); err != nil {
				t.Fatal(err)
			}
		}
	}
	return m
}

// mustErrorIs checks the stable sentinel hierarchy, including joined causes.
func mustErrorIs(t *testing.T, err, target error) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Fatalf("expected errors.Is(%v,%v)", err, target)
	}
}

// mustNear refuses NaN/Inf, even when the subtraction would otherwise hide them.
// The absolute threshold is intended for small well-scaled exact fixtures.
func mustNear(t *testing.T, got, want, tol float64) {
	t.Helper()
	if math.IsNaN(got) || math.IsInf(got, 0) || math.IsNaN(want) || math.IsInf(want, 0) ||
		math.IsNaN(tol) || math.IsInf(tol, 0) || tol < 0 || math.Abs(got-want) > tol {
		t.Fatalf("got %.17g want %.17g absolute tolerance %.2g", got, want, tol)
	}
}

// mustRelativeNear uses both absolute and scale-relative budgets for a finite oracle.
// It does not turn an infinite expected result into a successful numeric comparison.
func mustRelativeNear(t *testing.T, got, want, relative, absolute float64) {
	t.Helper()
	if math.IsNaN(got) || math.IsInf(got, 0) || math.IsNaN(want) || math.IsInf(want, 0) ||
		math.IsNaN(relative) || math.IsInf(relative, 0) || relative < 0 ||
		math.IsNaN(absolute) || math.IsInf(absolute, 0) || absolute < 0 {
		t.Fatalf("invalid finite comparison: got=%g want=%g rtol=%g atol=%g", got, want, relative, absolute)
	}
	scale := math.Max(math.Abs(got), math.Abs(want))
	bound := absolute + relative*scale
	if math.Abs(got-want) > bound {
		t.Fatalf("got %.17g want %.17g bound %.4g", got, want, bound)
	}
}

// mustCell reads one coordinate and fails only from the owner test goroutine.
func mustCell(t *testing.T, m matrix.Matrix, i, j int) float64 {
	t.Helper()
	value, err := m.At(i, j)
	if err != nil {
		t.Fatal(err)
	}
	return value
}

// isNilLike checks only explicit lvlath nilability, never reflection-based internals.
// Unknown interface values are present unless the static interface is nil.
func isNilLike(value any) bool {
	if value == nil {
		return true
	}
	if nilable, ok := value.(core.Nilable); ok {
		return nilable.IsNil()
	}
	return false
}

// mustNilState asserts the explicit pointer-presence contract of core.Nilable.
func mustNilState(t *testing.T, value any, wantNil bool, operation string) {
	t.Helper()
	if gotNil := isNilLike(value); gotNil != wantNil {
		t.Fatalf("%s: got nil=%t, want nil=%t", operation, gotNil, wantNil)
	}
}

// mustEqualBool requires an exact branch outcome and reports its operating context.
func mustEqualBool(t *testing.T, got, want bool, operation string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %t want %t", operation, got, want)
	}
}

// mustEqualInt requires an exact shape/rank/length for a discrete invariant.
func mustEqualInt(t *testing.T, got, want int, operation string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %d want %d", operation, got, want)
	}
}

// mustEqualString compares exact deterministic labels without reflection.
func mustEqualString(t *testing.T, got, want, operation string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %q want %q", operation, got, want)
	}
}
