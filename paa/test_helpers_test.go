// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
//
// Explicit external-package contract assertions.

package paa_test

import (
	"errors"
	"math"
	"testing"
)

// mustErrorIs verifies the exact sentinel class without comparing diagnostic text.
// It is intentionally shared by all external tests for stable error precedence.
func mustErrorIs(t *testing.T, err, target error) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Fatalf("error = %v, want errors.Is(...,%v)", err, target)
	}
}

// mustEqualInt compares shape and index metadata exactly.
// An index mismatch is a contract violation, not a numerical tolerance issue.
func mustEqualInt(t *testing.T, got, want int, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %d want %d", op, got, want)
	}
}

// mustEqualBool checks discrete predicates without reflection or assertions.
// A descriptive operation label identifies the violated state law.
func mustEqualBool(t *testing.T, got, want bool, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %t want %t", op, got, want)
	}
}

// mustEqualString compares user-visible deterministic string classifications.
// It is intentionally exact; message text is never an error protocol.
func mustEqualString(t *testing.T, got, want, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %q want %q", op, got, want)
	}
}

// mustClose compares well-conditioned example magnitudes with scale-aware tolerance.
// Exact-rounding obligations use a separate big.Rat oracle and bit comparisons.
func mustClose(t *testing.T, got, want float64) {
	t.Helper()
	bound := 1e-14 * math.Max(1, math.Max(math.Abs(got), math.Abs(want)))
	if math.IsNaN(got) || math.IsInf(got, 0) || math.Abs(got-want) > bound {
		t.Fatalf("got %.17g want %.17g, tolerance %.17g", got, want, bound)
	}
}
