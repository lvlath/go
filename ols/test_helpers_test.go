// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ols_test defines external-only fixture builders and contract assertions.
// Helpers in this file are used by regression, oracle, and error-precedence tests.
package ols_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/matrix"
)

// mustDense constructs an owning finite n-by-p fixture through the public API.
// It reports ragged rows or rejected data without silently altering dimensions.
func mustDense(t *testing.T, data [][]float64) *matrix.Dense {
	t.Helper()
	rows := len(data)
	cols := 0
	if rows > 0 {
		cols = len(data[0])
	}
	m, err := matrix.NewDense(rows, cols)
	if err != nil {
		t.Fatal(err)
	}
	for row, values := range data {
		if len(values) != cols {
			t.Fatalf("ragged fixture at row %d: got %d columns, want %d", row, len(values), cols)
		}
		for col, value := range values {
			if err := m.Set(row, col, value); err != nil {
				t.Fatalf("fixture Set(%d,%d): %v", row, col, err)
			}
		}
	}
	return m
}

// mustNear compares numerical results using a mixed absolute/relative budget.
// A missing or nonfinite computed result cannot accidentally pass NaN comparisons.
func mustNear(t *testing.T, got, want, tolerance float64) {
	t.Helper()
	if !finiteForTest(got) || !finiteForTest(want) || math.Abs(got-want) > tolerance*math.Max(1, math.Abs(want)) {
		t.Fatalf("got %.17g, want %.17g (tolerance %.3g)", got, want, tolerance)
	}
}

// finiteForTest classifies float64 test observations without reflection.
// It prevents a NaN difference from silently satisfying a numeric assertion.
func finiteForTest(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

// mustErrorIs verifies semantic error identity rather than fragile error strings.
func mustErrorIs(t *testing.T, err, target error) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Fatalf("got error %v; want errors.Is(..., %v)", err, target)
	}
}

// mustEqualInt anchors exact dimensional or rank invariants.
func mustEqualInt(t *testing.T, got, want int, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s = %d; want %d", op, got, want)
	}
}

// mustEqualBool anchors discrete nilability/policy invariants without reflection.
func mustEqualBool(t *testing.T, got, want bool, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s = %v; want %v", op, got, want)
	}
}

// fixtureMatrix is a purpose-built public Matrix implementation for testing
// boundary conditions that an ordinary finite Dense cannot represent.
// In particular, its At may report an injected error or expose nonfinite data.
type fixtureMatrix struct {
	rows, cols int
	values     []float64
	readErr    error
	errRow     int
	errCol     int
	onRead     func(row, col int)
}

// IsNil implements the nil-safe core.Nilable boundary for typed-nil fixtures.
func (m *fixtureMatrix) IsNil() bool { return m == nil }

// Rows reports a fixed test-supplied row count without dereferencing nil.
func (m *fixtureMatrix) Rows() int {
	if m == nil {
		return 0
	}
	return m.rows
}

// Cols reports a fixed feature width, including intentional invalid sizes.
func (m *fixtureMatrix) Cols() int {
	if m == nil {
		return 0
	}
	return m.cols
}

// At provides controlled read faults or raw nonfinite cells through Matrix.
// The onRead observer deterministically triggers cancellation tests without sleeps.
func (m *fixtureMatrix) At(row, col int) (float64, error) {
	if m == nil {
		return 0, errors.New("fixture: nil receiver should have been rejected")
	}
	if row < 0 || col < 0 || row >= m.rows || col >= m.cols {
		return 0, matrix.ErrOutOfRange
	}
	if m.onRead != nil {
		m.onRead(row, col)
	}
	if m.readErr != nil && row == m.errRow && col == m.errCol {
		return 0, m.readErr
	}
	return m.values[row*m.cols+col], nil
}

// Set is a minimal shape-checked test adapter write, not an OLS entry point.
func (m *fixtureMatrix) Set(row, col int, value float64) error {
	if m == nil || row < 0 || col < 0 || row >= m.rows || col >= m.cols {
		return matrix.ErrOutOfRange
	}
	m.values[row*m.cols+col] = value
	return nil
}

// Clone supplies the interface-mandated independent fixture value snapshot.
func (m *fixtureMatrix) Clone() matrix.Matrix {
	if m == nil {
		return nil
	}
	clone := *m
	clone.values = append([]float64(nil), m.values...)
	return &clone
}

// mustMatrixValues compares original caller-owned values after Fit for mutation.
// It also checks shapes so a silent matrix resizing cannot go undetected.
func mustMatrixValues(t *testing.T, got matrix.Matrix, want [][]float64) {
	t.Helper()
	mustEqualInt(t, got.Rows(), len(want), "matrix rows")
	if len(want) == 0 {
		return
	}
	mustEqualInt(t, got.Cols(), len(want[0]), "matrix cols")
	for row := range want {
		for col, value := range want[row] {
			cell, err := got.At(row, col)
			if err != nil || cell != value {
				t.Fatalf("matrix changed at (%d,%d): %v, %v; want %v", row, col, cell, err, value)
			}
		}
	}
}
