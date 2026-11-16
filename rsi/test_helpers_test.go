// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// File test_helpers_test.go implements explicit external contract assertions.
package rsi_test

import (
	"errors"
	"math"
	"testing"
)

// mustErrorIs enforces sentinel identity through errors.Is rather than error text.
// It accepts wrapped identities and rejects unrelated categories.
func mustErrorIs(t *testing.T, err, target error) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Fatalf("error=%v, want errors.Is(_, %v)", err, target)
	}
}

// mustEqualBool checks an explicit readiness or protocol flag.
// It reports the failing operation without reflection.
func mustEqualBool(t *testing.T, got, want bool, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %v want %v", op, got, want)
	}
}

// mustEqualInt checks shape, count, and length invariants.
// The operation label identifies the observable contract field.
func mustEqualInt(t *testing.T, got, want int, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %d want %d", op, got, want)
	}
}

// mustEqualString compares identifiers and versioned configuration fingerprints.
// Exact string comparison anchors the public snapshot ABI.
func mustEqualString(t *testing.T, got, want string, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %q want %q", op, got, want)
	}
}

// isNilLike honors the public IsNil method before testing the interface itself.
// It detects typed-nil *rsi.Result values without using reflect.
func isNilLike(value any) bool {
	if x, ok := value.(interface{ IsNil() bool }); ok {
		return x.IsNil()
	}
	return value == nil
}

// mustNilState checks nilable interface semantics without inspecting private fields.
// It distinguishes a typed-nil result from an allocated zero-value Result.
func mustNilState(t *testing.T, v any, wantNil bool, op string) {
	t.Helper()
	mustEqualBool(t, isNilLike(v), wantNil, op)
}

// mustSameFloat compares raw binary64 bits, including signed zero and NaN payloads.
// Unlike approximate equality, this protects the batch/stream parity contract.
func mustSameFloat(t *testing.T, a, b float64) {
	t.Helper()
	if math.Float64bits(a) != math.Float64bits(b) {
		t.Fatalf("float mismatch: %.17g vs %.17g", a, b)
	}
}
