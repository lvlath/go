// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// External test helpers compare public contracts without reflection or
// assertion frameworks. Every helper below is used by independent sax_test
// fixtures to keep sentinel, nil-state, and numerical failures actionable.
package sax_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/core"
)

// mustErrorIs checks a branchable sentinel, never the unstable error string.
func mustErrorIs(t *testing.T, err, sentinel error) {
	t.Helper()
	if !errors.Is(err, sentinel) {
		t.Fatalf("error=%v; expected errors.Is(..., %v)", err, sentinel)
	}
}

// isNilLike follows the same typed-nil core.Nilable protocol as the library.
// It is intentionally not a general reflection-based nil detector.
func isNilLike(value any) bool {
	if value == nil {
		return true
	}
	if nilable, ok := value.(core.Nilable); ok {
		return nilable.IsNil()
	}
	return false
}

// mustNilState distinguishes a typed-nil pointer from a non-nil empty result.
func mustNilState(t *testing.T, value any, wantNil bool, operation string) {
	t.Helper()
	if got := isNilLike(value); got != wantNil {
		t.Fatalf("%s: nil-state=%v; want %v", operation, got, wantNil)
	}
}

// mustEqualBool checks deterministic flags such as Degenerate and IsNil.
func mustEqualBool(t *testing.T, got, want bool, operation string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %t; want %t", operation, got, want)
	}
}

// mustEqualInt checks sample, alphabet, segment, and symbol dimensions.
func mustEqualInt(t *testing.T, got, want int, operation string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %d; want %d", operation, got, want)
	}
}

// mustEqualString checks canonical hashes or audited labels exactly.
func mustEqualString(t *testing.T, got, want, operation string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %q; want %q", operation, got, want)
	}
}

// mustEqualSymbols asserts position and shape, not only a vague slice mismatch.
func mustEqualSymbols(t *testing.T, got, want []uint16, operation string) {
	t.Helper()
	mustEqualInt(t, len(got), len(want), operation+" length")
	for index, symbol := range want {
		if got[index] != symbol {
			t.Fatalf("%s[%d]: got %d; want %d", operation, index, got[index], symbol)
		}
	}
}

// mustNear checks finite values with an explicit absolute/relative oracle budget.
func mustNear(t *testing.T, got, want, atol, rtol float64, operation string) {
	t.Helper()
	if math.IsNaN(got) || math.IsInf(got, 0) || math.IsNaN(want) || math.IsInf(want, 0) {
		t.Fatalf("%s: non-finite got=%g want=%g", operation, got, want)
	}
	budget := atol + rtol*math.Max(math.Abs(got), math.Abs(want))
	if math.Abs(got-want) > budget {
		t.Fatalf("%s: got %.17g; want %.17g; allowed %.4g", operation, got, want, budget)
	}
}
