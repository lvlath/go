// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package acf_test provides explicit, reflect-free assertion helpers for ACF tests.
// Helpers report the contractual value being compared rather than generic deep-equality.
package acf_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/acf"
)

// mustErrorIs enforces sentinel classification with errors.Is, never error text.
// It reports the actual and expected error values to the owning test goroutine.
func mustErrorIs(t *testing.T, err, target error) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Fatalf("error=%v want %v", err, target)
	}
}

// mustClose compares finite ACF coefficients within a fixed binary64 budget.
// The combined absolute/relative bound is suitable for normalized small-window ratios.
func mustClose(t *testing.T, got, want float64) {
	t.Helper()
	if math.IsNaN(got) || math.IsInf(got, 0) || math.Abs(got-want) > 2e-13*math.Max(1, math.Abs(want)) {
		t.Fatalf("got %.17g want %.17g", got, want)
	}
}

// mustValues checks exact slice length before ordered numeric comparisons.
// It rejects missing lags rather than comparing only the available prefix.
func mustValues(t *testing.T, got []float64, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length %d != %d", len(got), len(want))
	}
	for i := range got {
		mustClose(t, got[i], want[i])
	}
}

// mustPairCounts checks length parity and n-k at every reported lag.
// Structural pair counts are exact integers, never floating-point estimates.
func mustPairCounts(t *testing.T, r *acf.Result, n int) {
	t.Helper()
	if r == nil {
		t.Fatal("nil result has no pair counts")
	}
	if len(r.PairCounts) != len(r.Values) {
		t.Fatalf("pair count length=%d coefficient length=%d", len(r.PairCounts), len(r.Values))
	}
	for i, c := range r.PairCounts {
		if c != n-i {
			t.Fatalf("pairs[%d]=%d want %d", i, c, n-i)
		}
	}
}

// mustEqualBool checks a contract predicate without general-purpose reflection.
// Its operation label identifies the failed observable rule.
func mustEqualBool(t *testing.T, got, want bool, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s got %t want %t", op, got, want)
	}
}

// mustEqualInt compares exact integer metadata and option enum values.
// It never permits numeric tolerance for discrete invariants.
func mustEqualInt(t *testing.T, got, want int, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s got %d want %d", op, got, want)
	}
}

// mustEqualString checks exact textual metadata when a domain requires it.
// The helper does not parse, classify, or compare error message text.
func mustEqualString(t *testing.T, got, want string, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s got %q want %q", op, got, want)
	}
}

// isNilLike applies the Nilable-style interface contract without reflection.
// Only a nil interface or a dynamic value implementing IsNil can be nil-like.
func isNilLike(value any) bool {
	if value == nil {
		return true
	}
	if n, ok := value.(interface{ IsNil() bool }); ok {
		return n.IsNil()
	}
	return false
}

// mustNilState checks nil-state classification through the public method set.
// It works for typed nil pointers held inside a non-nil interface.
func mustNilState(t *testing.T, value any, want bool, op string) {
	t.Helper()
	mustEqualBool(t, isNilLike(value), want, op)
}
