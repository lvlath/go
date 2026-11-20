// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pacf_test centralizes contract-oriented assertions without reflection.
// Error identity, typed-nil state, and scalar accuracy have explicit helpers.
package pacf_test

import (
	"errors"
	"math"
	"testing"
)

// mustErrorIs compares sentinel classification through errors.Is, never strings.
// It preserves the production contract that wrapped errors remain classifiable.
func mustErrorIs(t *testing.T, err, want error) {
	t.Helper()
	if !errors.Is(err, want) {
		t.Fatalf("error = %v, want errors.Is(...,%v)", err, want)
	}
}

// isNilLike implements the reflect-free Nilable recognition contract.
// It recognizes a nil interface or an allocated interface whose value implements
// IsNil; unrecognized non-nil values are not assumed to have hidden nil state.
func isNilLike(value any) bool {
	if value == nil {
		return true
	}
	if nilable, ok := value.(interface{ IsNil() bool }); ok {
		return nilable.IsNil()
	}
	return false
}

// mustNilState checks nilability even when a typed nil is stored in an interface.
// It avoids any direct reflection or heuristics based on result slice contents.
func mustNilState(t *testing.T, value any, want bool, operation string) {
	t.Helper()
	got := isNilLike(value)
	if got != want {
		t.Fatalf("%s nil = %t, want %t", operation, got, want)
	}
}

// mustEqualBool compares a boolean contract property and reports the named operation.
func mustEqualBool(t *testing.T, got, want bool, operation string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %t want %t", operation, got, want)
	}
}

// mustEqualInt compares exact integer dimensions and observation counts.
func mustEqualInt(t *testing.T, got, want int, operation string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %d want %d", operation, got, want)
	}
}

// closeEnough uses a relative tolerance above unit magnitude and an absolute
// floor below it; dedicated ULP tests check subnormal representations exactly.
func closeEnough(got, want, tolerance float64) bool {
	return !math.IsNaN(got) && !math.IsInf(got, 0) &&
		math.Abs(got-want) <= tolerance*math.Max(1, math.Abs(want))
}

// mustSlices checks equal shape and every binary64 coefficient within an
// explicit tolerance, with index-specific diagnostics instead of DeepEqual.
func mustSlices(t *testing.T, got, want []float64, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length %d want %d", len(got), len(want))
	}
	for i := range got {
		if !closeEnough(got[i], want[i], tolerance) {
			t.Fatalf("index %d: got %.17g want %.17g", i, got[i], want[i])
		}
	}
}
