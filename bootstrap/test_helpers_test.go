// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package bootstrap_test centralizes exact, reflect-free contract assertions.
// Every comparison reports semantic indices or error classes instead of string matching.
package bootstrap_test

import (
	"errors"
	"testing"

	"github.com/lvlath/go/bootstrap"
)

// mustErrorIs preserves sentinel identity under arbitrary error wrapping.
func mustErrorIs(t *testing.T, err, target error) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Fatalf("error=%v; want errors.Is(...,%v)", err, target)
	}
}

// isNilLike recognizes nil interfaces and pointer receivers implementing IsNil.
// It intentionally does not use reflect or interpret empty allocated results as nil.
func isNilLike(value any) bool {
	if value == nil {
		return true
	}
	if nilable, ok := value.(interface{ IsNil() bool }); ok {
		return nilable.IsNil()
	}
	return false
}

// mustNilState checks typed-nil behavior without unsafe pointer introspection.
func mustNilState(t *testing.T, value any, wantNil bool, op string) {
	t.Helper()
	if got := isNilLike(value); got != wantNil {
		t.Fatalf("%s: nil-state=%v, want %v", op, got, wantNil)
	}
}

// mustEqualBool compares a declared boolean contract property.
func mustEqualBool(t *testing.T, got, want bool, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got=%v, want=%v", op, got, want)
	}
}

// mustEqualInt compares an exactly defined count, offset or index.
func mustEqualInt(t *testing.T, got, want int, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got=%d, want=%d", op, got, want)
	}
}

// mustEqualString compares contract metadata exactly, including RNG revisions.
func mustEqualString(t *testing.T, got, want, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got=%q, want=%q", op, got, want)
	}
}

// mustSameRows checks both row counts and each observation index exactly.
func mustSameRows(t *testing.T, got, want [][]int, op string) {
	t.Helper()
	mustEqualInt(t, len(got), len(want), op+" row count")
	for row := range got {
		mustEqualInt(t, len(got[row]), len(want[row]), op+" row length")
		for col, actual := range got[row] {
			if expected := want[row][col]; actual != expected {
				t.Fatalf("%s: row=%d col=%d got=%d want=%d", op, row, col, actual, expected)
			}
		}
	}
}

// mustResultShape validates a fully populated result's ownership-independent laws.
func mustResultShape(t *testing.T, result *bootstrap.Result, rows, length, completed int) {
	t.Helper()
	if result == nil {
		t.Fatal("expected a non-nil Result")
	}
	mustEqualInt(t, result.Completed, completed, "Completed")
	mustEqualInt(t, len(result.Indices), completed, "Indices rows")
	if result.Indices == nil {
		t.Fatal("Indices slice must be allocated even on zero-completion partial results")
	}
	for replicate, indices := range result.Indices {
		mustEqualInt(t, len(indices), length, "replicate row length")
		for col, index := range indices {
			if index < 0 || index >= rows {
				t.Fatalf("invalid index at replicate=%d col=%d: got=%d, rows=%d", replicate, col, index, rows)
			}
		}
	}
	if result.RNG == "" {
		t.Fatal("RNG revision must be nonempty on computed results")
	}
}
