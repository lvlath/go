// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gjrgarch_test defines transparent, reflect-free UES contract assertions.
// Failure messages describe the failing semantic assertion rather than field dumps.
package gjrgarch_test

import (
	"errors"
	"math"
	"testing"
)

// mustErrorIs enforces sentinel identity through the canonical errors.Is protocol.
func mustErrorIs(t *testing.T, err, target error) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Fatalf("error %v does not wrap sentinel %v", err, target)
	}
}

// nilState is the tiny, reflection-free subset used by core.Nilable results.
type nilState interface{ IsNil() bool }

// isNilLike handles nil interfaces and typed-nil values implementing IsNil.
func isNilLike(value any) bool {
	if value == nil {
		return true
	}
	nilable, ok := value.(nilState)
	return ok && nilable.IsNil()
}

// mustNilState asserts the exact typed-nil interpretation expected by UES.
func mustNilState(t *testing.T, value any, wantNil bool, operation string) {
	t.Helper()
	if got := isNilLike(value); got != wantNil {
		t.Fatalf("%s: nil-state got=%t want=%t", operation, got, wantNil)
	}
}

// mustEqualBool validates one explicit Boolean contract claim.
func mustEqualBool(t *testing.T, got, want bool, operation string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %t want %t", operation, got, want)
	}
}

// mustEqualInt validates an exact count such as Steps or iterations.
func mustEqualInt(t *testing.T, got, want int, operation string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %d want %d", operation, got, want)
	}
}

// mustNearRelative asserts a numerical value using a scale-aware error budget.
func mustNearRelative(t *testing.T, got, want, tolerance float64, operation string) {
	t.Helper()
	if !finiteTest(got) || math.Abs(got-want) > tolerance*math.Max(1, math.Abs(want)) {
		t.Fatalf("%s: got %.17g want %.17g with tolerance %.3g", operation, got, want, tolerance)
	}
}

// finiteTest avoids importing internal numeric validators in the external package.
func finiteTest(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }
