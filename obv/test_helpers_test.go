// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package obv_test holds source-independent fixtures and explicit assertions.
// Helpers use only the public OBV API and never hide numerical tolerance rules.
package obv_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/obv"
)

// bars constructs an aligned OHLCV fixture from independent closes and volumes.
// Assigning the same value to OHLC produces valid flat-intrabar geometry while
// leaving interbar Close changes and signed-volume contributions observable.
//
// Inputs:
//   - closes: chronological, positive finite closing prices.
//   - volumes: chronological, finite nonnegative base-volume units of equal length.
//
// Returns:
//   - obv.Bars: detached column slices, with Open=High=Low=Close at each index.
//
// Complexity:
//   - Time O(n), Space O(n) for the five owned columns.
//
// AI-Hints:
//   - This is a controlled test fixture, not a production OHLCV normalizer.
func bars(closes, volumes []float64) obv.Bars {
	count := len(closes)
	fixture := obv.Bars{
		Open: make([]float64, count), High: make([]float64, count),
		Low: make([]float64, count), Close: append([]float64(nil), closes...),
		Volume: append([]float64(nil), volumes...),
	}
	for index, closingPrice := range closes {
		fixture.Open[index] = closingPrice
		fixture.High[index] = closingPrice
		fixture.Low[index] = closingPrice
	}
	return fixture
}

// mustErrorIs asserts stable sentinel identity without inspecting diagnostic text.
// It accepts wrapped errors because errors.Is is the published classification API.
//
// Inputs:
//   - t: the owning test goroutine's testing handle.
//   - err: error returned by the public facade.
//   - target: expected exported sentinel error.
//
// Complexity:
//   - Time O(w), Space O(1), where w is the wrapped-error chain depth.
//
// AI-Hints:
//   - Never convert a sentinel assertion into strings.Contains on Error().
func mustErrorIs(t *testing.T, err, target error) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Fatalf("error=%v; want errors.Is(..., %v)", err, target)
	}
}

// isNilLike recognizes the package's explicit Nilable convention without reflect.
// An allocated empty Result remains non-nil even when its public slices are nil.
//
// Inputs:
//   - value: an optional nil interface or object implementing IsNil.
//
// Returns:
//   - bool: true for a nil interface or an explicitly nilable nil receiver.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// AI-Hints:
//   - A typed nil pointer inside an interface is not equal to a nil interface.
func isNilLike(value any) bool {
	if value == nil {
		return true
	}
	if nilable, ok := value.(interface{ IsNil() bool }); ok {
		return nilable.IsNil()
	}
	return false
}

// mustNilState verifies pointer-level result presence using its public contract.
// It does not confuse a zero-length slice with an absent Result object.
//
// Inputs:
//   - t: current test owner; value: public result in interface form.
//   - wantNil: expected pointer-null classification; op: diagnostic context.
//
// Complexity:
//   - Time O(1), Space O(1).
func mustNilState(t *testing.T, value any, wantNil bool, op string) {
	t.Helper()
	if got := isNilLike(value); got != wantNil {
		t.Fatalf("%s: nil-state=%t, want %t", op, got, wantNil)
	}
}

// mustEqualBool checks an exact boolean contract without permissive coercion.
// It is suitable for Valid masks and transactional status flags.
//
// Complexity:
//   - Time O(1), Space O(1).
func mustEqualBool(t *testing.T, got, want bool, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %t, want %t", op, got, want)
	}
}

// mustEqualInt compares index and count quantities without string conversion.
// It preserves the exact integer-domain contract of the assertion.
//
// Complexity:
//   - Time O(1), Space O(1).
func mustEqualInt(t *testing.T, got, want int, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %d, want %d", op, got, want)
	}
}

// mustEqualString verifies stable version IDs and compatibility fingerprints.
// It is not an appropriate substitute for errors.Is classification.
//
// Complexity:
//   - Time O(n), Space O(1), where n is the compared string length.
func mustEqualString(t *testing.T, got, want, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %q, want %q", op, got, want)
	}
}

// mustValues compares a complete OBV fixture to an independent exact-bit oracle.
// Successful batch output must have a true mask at every index, including zero.
//
// Inputs:
//   - t: owning test; result: public batch artifact; want: independent values.
//
// Complexity:
//   - Time O(n), Space O(1) beyond the caller-owned expected slice.
//
// AI-Hints:
//   - Exact bits, not epsilon equality, defend batch/stream and seed semantics.
func mustValues(t *testing.T, result *obv.Result, want []float64) {
	t.Helper()
	if result == nil {
		t.Fatal("result is nil after an expected successful computation")
	}
	mustEqualInt(t, len(result.Values), len(want), "Values length")
	mustEqualInt(t, len(result.Valid), len(want), "Valid length")
	for index, expected := range want {
		if math.Float64bits(result.Values[index]) != math.Float64bits(expected) {
			t.Fatalf("bar %d: OBV=%v, want %v", index, result.Values[index], expected)
		}
		mustEqualBool(t, result.Valid[index], true, "accepted observation validity")
	}
}

// sameSnapshot compares full checkpoint metadata and all numeric bits.
// It checks compensation separately so an unchanged displayed total cannot
// disguise a mutated low-order recurrence component after a failed Push.
//
// Inputs:
//   - a, b: detached versioned snapshots from public Stream.Snapshot calls.
//
// Returns:
//   - bool: exact equality of scalar metadata, slice lengths and element bits.
//
// Complexity:
//   - Time O(f+i), Space O(1), for float and integer snapshot lengths.
//
// AI-Hints:
//   - Float comparisons must include sign-zero bits and all compensation slots.
func sameSnapshot(a, b obv.Snapshot) bool {
	if a.Algorithm != b.Algorithm || a.Config != b.Config || a.Count != b.Count ||
		len(a.FloatState) != len(b.FloatState) || len(a.IntState) != len(b.IntState) {
		return false
	}
	for index := range a.FloatState {
		if math.Float64bits(a.FloatState[index]) != math.Float64bits(b.FloatState[index]) {
			return false
		}
	}
	for index := range a.IntState {
		if a.IntState[index] != b.IntState[index] {
			return false
		}
	}
	return true
}
