// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package lstm_test defines explicit non-reflecting numerical, sentinel and
// nilability assertions shared by independent external contract tests.
package lstm_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/lstm"
	"github.com/lvlath/go/matrix"
)

func mustMatrix(t *testing.T, r, c int, v []float64) *matrix.Dense {
	t.Helper()
	m, e := matrix.NewDense(r, c)
	if e != nil {
		t.Fatal(e)
	}
	for k, x := range v {
		if e = m.Set(k/c, k%c, x); e != nil {
			t.Fatal(e)
		}
	}
	return m
}
func fixture(t *testing.T, d, h int, bias []float64) *lstm.Model {
	t.Helper()
	w := lstm.Weights{Input: mustMatrix(t, 4*h, d, nil), Recurrent: mustMatrix(t, 4*h, h, nil), Bias: bias, Hidden: h}
	if bias == nil {
		w.Bias = make([]float64, 4*h)
	}
	m, e := lstm.Import(w)
	if e != nil {
		t.Fatal(e)
	}
	return m
}
func mustApprox(t *testing.T, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-13 {
		t.Fatalf("got %.17g, want %.17g", got, want)
	}
}
func mustErrorIs(t *testing.T, err, target error) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Fatalf("got %v, want %v", err, target)
	}
}
func at(t *testing.T, m matrix.Matrix, r, c int) float64 {
	t.Helper()
	v, e := m.At(r, c)
	if e != nil {
		t.Fatal(e)
	}
	return v
}
func state(t *testing.T, m *lstm.Model, c, h float64) *lstm.State {
	t.Helper()
	return &lstm.State{Hidden: mustMatrix(t, 1, 1, []float64{h}), Cell: mustMatrix(t, 1, 1, []float64{c}), ModelID: m.ID()}
}

// mustEqualString compares protocol identifiers without reflection or error-message parsing.
func mustEqualString(t *testing.T, got, want, operation string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %q, want %q", operation, got, want)
	}
}

// mustEqualInt checks a bounded integer contract such as shape or valid sample count.
func mustEqualInt(t *testing.T, got, want int, operation string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %d, want %d", operation, got, want)
	}
}

// mustEqualBool checks a semantic boolean without generic reflect-based assertions.
func mustEqualBool(t *testing.T, got, want bool, operation string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %v, want %v", operation, got, want)
	}
}

// isNilLike checks only explicit nilable values; unsupported pointer kinds are not guessed.
func isNilLike(value any) bool {
	if value == nil {
		return true
	}
	if nilable, ok := value.(interface{ IsNil() bool }); ok {
		return nilable.IsNil()
	}
	return false
}

// mustNilState anchors a Nilable receiver's state without requiring reflect.
func mustNilState(t *testing.T, value any, wantNil bool, operation string) {
	t.Helper()
	mustEqualBool(t, isNilLike(value), wantNil, operation)
}
