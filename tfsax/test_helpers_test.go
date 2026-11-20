// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tfsax_test provides explicit public-surface test assertions without reflect.
// Helpers preserve errors.Is and nilable-interface semantics in independent test packages.
package tfsax_test

import (
	"errors"
	"testing"

	"github.com/lvlath/go/tfsax"
)

// mustErrorIs fails on a missing sentinel identity rather than comparing error text.
// Only the owning test goroutine may call the failure methods on testing.T.
func mustErrorIs(t *testing.T, err, target error) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Fatalf("got %v, want errors.Is(...,%v)", err, target)
	}
}

// isNilLike recognizes literal nil or the library's typed nilable convention.
// It deliberately does not inspect arbitrary typed nils using reflection.
func isNilLike(value any) bool {
	if value == nil {
		return true
	}
	if object, ok := value.(interface{ IsNil() bool }); ok {
		return object.IsNil()
	}
	return false
}

// mustNilState checks typed nil vs nonnil result semantics through IsNil.
// It does not mistake a zero-valued nonnil result for an absent result.
func mustNilState(t *testing.T, value any, wantNil bool, operation string) {
	t.Helper()
	if isNilLike(value) != wantNil {
		t.Fatalf("%s: nil-state got=%t want=%t", operation, isNilLike(value), wantNil)
	}
}

// mustEqualBool compares observable boolean contracts without coercing values.
func mustEqualBool(t *testing.T, got, want bool, operation string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got=%t want=%t", operation, got, want)
	}
}

// mustEqualInt compares exact counts, indexes and work-accounting invariants.
func mustEqualInt(t *testing.T, got, want int, operation string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got=%d want=%d", operation, got, want)
	}
}

// mustEqualString compares deterministic schema and representation identities.
func mustEqualString(t *testing.T, got, want, operation string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got=%q want=%q", operation, got, want)
	}
}

func mustIDs(t *testing.T, got, want []tfsax.CandidateID) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("candidate count got=%d want=%d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("candidate[%d] got=%d want=%d", i, got[i], want[i])
		}
	}
}
