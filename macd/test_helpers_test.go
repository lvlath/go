// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package macd_test provides reflect-free assertion helpers for
// error identity, state nilability, bitwise floats and detached snapshots.
package macd_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/macd"
)

// mustErrorIs Asserts a semantic sentinel category without relying on error strings.
func mustErrorIs(t *testing.T, err, target error) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Fatalf("error = %v; want errors.Is(..., %v)", err, target)
	}
}

// mustEqualBool Reports a named mismatch in a warmup or state boolean invariant.
func mustEqualBool(t *testing.T, got, want bool, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: %v want %v", op, got, want)
	}
}

// mustEqualString Reports a named mismatch in a fixed textual protocol field.
func mustEqualString(t *testing.T, got, want, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: %s want %s", op, got, want)
	}
}

// isNilLike Classifies typed-nil Result pointers without importing reflect.
func isNilLike(v any) bool {
	if v == nil {
		return true
	}
	if p, ok := v.(*macd.Result); ok {
		return p == nil
	}
	return false
}

// mustNilState Applies the nilability helper and reports the intended operation.
func mustNilState(t *testing.T, v any, want bool, op string) {
	t.Helper()
	mustEqualBool(t, isNilLike(v), want, op)
}

// mustClose Compares independent rational fixture values with a fixed floating tolerance.
func mustClose(t *testing.T, got, want float64, where string) {
	t.Helper()
	if math.Abs(got-want) > 2e-14*math.Max(1, math.Abs(want)) {
		t.Fatalf("%s got %.17g want %.17g", where, got, want)
	}
}

// mustBits Compares full IEEE-754 bit patterns, including signed zero.
func mustBits(t *testing.T, a, b float64, where string) {
	t.Helper()
	if math.Float64bits(a) != math.Float64bits(b) {
		t.Fatalf("%s bits %x != %x", where, math.Float64bits(a), math.Float64bits(b))
	}
}

// mustSnapshotsEqual Verifies version/config/count and every phase field without reflection.
func mustSnapshotsEqual(t *testing.T, a, b macd.Snapshot) {
	t.Helper()
	if a.Count != b.Count || a.Algorithm != b.Algorithm || a.Config != b.Config || len(a.FloatState) != len(b.FloatState) || len(a.IntState) != len(b.IntState) {
		t.Fatalf("snapshots structural mismatch: %+v vs %+v", a, b)
	}
	for i := range a.FloatState {
		mustBits(t, a.FloatState[i], b.FloatState[i], "snapshot mean")
	}
	for i := range a.IntState {
		if a.IntState[i] != b.IntState[i] {
			t.Fatalf("snapshot count %d: %d != %d", i, a.IntState[i], b.IntState[i])
		}
	}
}
