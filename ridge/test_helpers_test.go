// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ridge_test provides typed, reflection-free contract assertions.
// Helpers deliberately assert mathematical values and error identities, not text.
package ridge_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/ridge"
)

// dense builds an owning rectangular Matrix fixture with fully checked writes.
// A malformed fixture fails immediately in the calling test goroutine.
func dense(t *testing.T, values [][]float64) *matrix.Dense {
	t.Helper()
	if len(values) == 0 || len(values[0]) == 0 {
		t.Fatal("empty matrix fixture")
	}
	result, err := matrix.NewDense(len(values), len(values[0]))
	if err != nil {
		t.Fatal(err)
	}
	for i, row := range values {
		if len(row) != len(values[0]) {
			t.Fatal("ragged matrix fixture")
		}
		for j, value := range row {
			if err = result.Set(i, j, value); err != nil {
				t.Fatal(err)
			}
		}
	}
	return result
}

// closeTo checks a scale-aware floating point equality for ordinary model fixtures.
// Exact-binary rational oracle tests use the stricter assertRationalClose instead.
func closeTo(t *testing.T, got, want float64) {
	t.Helper()
	if math.IsNaN(got) || math.IsInf(got, 0) || math.Abs(got-want) > 1e-9*(1+math.Abs(want)) {
		t.Fatalf("got %.17g want %.17g", got, want)
	}
}

// mustErrorIs checks sentinel ancestry without depending on diagnostic wording.
func mustErrorIs(t *testing.T, err, target error) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Fatalf("error=%v, want errors.Is(_, %v)", err, target)
	}
}

// errorIs retains the existing test helper spelling while sharing one assertion.
func errorIs(t *testing.T, err, target error) { mustErrorIs(t, err, target) }

// isNilLike reports recognized typed-nil states without runtime reflection.
// Additional custom fixture types must explicitly implement core.Nilable.
func isNilLike(value any) bool {
	if value == nil {
		return true
	}
	if candidate, ok := value.(core.Nilable); ok {
		return candidate.IsNil()
	}
	switch typed := value.(type) {
	case *ridge.Model:
		return typed == nil
	default:
		return false
	}
}

// mustNilState checks typed-nil semantics through the supported Nilable protocol.
func mustNilState(t *testing.T, value any, wantNil bool, operation string) {
	t.Helper()
	if actual := isNilLike(value); actual != wantNil {
		t.Fatalf("%s: nil state=%t want=%t", operation, actual, wantNil)
	}
}

// mustEqualBool asserts an exact Boolean business or contract condition.
func mustEqualBool(t *testing.T, actual, expected bool, operation string) {
	t.Helper()
	if actual != expected {
		t.Fatalf("%s: got=%t want=%t", operation, actual, expected)
	}
}

// mustEqualInt asserts an exact dimensional or discrete contract value.
func mustEqualInt(t *testing.T, actual, expected int, operation string) {
	t.Helper()
	if actual != expected {
		t.Fatalf("%s: got=%d want=%d", operation, actual, expected)
	}
}

// mustEqualString asserts exact deterministic labels without assertion frameworks.
func mustEqualString(t *testing.T, actual, expected, operation string) {
	t.Helper()
	if actual != expected {
		t.Fatalf("%s: got=%q want=%q", operation, actual, expected)
	}
}
