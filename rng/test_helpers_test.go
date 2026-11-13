// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package rng_test contains explicit, reflection-free contract assertions.
// These helpers compare public observations rather than private generator state.
package rng_test

import (
	"math"
	"testing"
)

type nilable interface{ IsNil() bool }

// isNilLike supports the UES pointer-presence convention without reflection.
// The RNG interface remains three-method and is not extended with IsNil.
func isNilLike(value any) bool {
	if value == nil {
		return true
	}
	if typed, ok := value.(nilable); ok {
		return typed.IsNil()
	}
	return false
}

// mustNilState checks either a nil interface or a typed nilable receiver.
func mustNilState(t *testing.T, value any, want bool, where string) {
	t.Helper()
	if got := isNilLike(value); got != want {
		t.Fatalf("%s: IsNil got %t want %t", where, got, want)
	}
}

// mustEqualBool compares a contract flag without an assertion framework.
func mustEqualBool(t *testing.T, got, want bool, where string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %t want %t", where, got, want)
	}
}

// mustEqualInt compares draw counts and path sizes exactly.
func mustEqualInt(t *testing.T, got, want int, where string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %d want %d", where, got, want)
	}
}

// mustEqualString compares stable protocol revision IDs exactly.
func mustEqualString(t *testing.T, got, want string, where string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %q want %q", where, got, want)
	}
}

// mustEqualWord checks raw bitstream equality with hexadecimal diagnostics.
func mustEqualWord(t *testing.T, got, want uint64, where string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %#016x want %#016x", where, got, want)
	}
}

// mustSameFloatBits checks exact replay of IEEE-754 results on this platform.
func mustSameFloatBits(t *testing.T, got, want float64, where string) {
	t.Helper()
	if math.Float64bits(got) != math.Float64bits(want) {
		t.Fatalf("%s: got %.17g (%x), want %.17g (%x)", where,
			got, math.Float64bits(got), want, math.Float64bits(want))
	}
}
