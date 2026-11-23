// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package match_test checks the public protocol without internal engine access.
// Helpers use explicit types and errors.Is instead of reflection or assertions.
package match_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/match"
	"github.com/lvlath/go/matrix"
)

// mustErrorIs asserts the exact branching contract of a returned error.
// It keeps an original cause readable when classification differs.
func mustErrorIs(t *testing.T, err, target error) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Fatalf("errors.Is(%v, %v) = false", err, target)
	}
}

// isNilLike checks the actually supported Nilable result and matrix types.
// It deliberately does not promise generic typed-nil detection through reflection.
func isNilLike(value any) bool {
	if value == nil {
		return true
	}
	if nilable, ok := value.(interface{ IsNil() bool }); ok {
		return nilable.IsNil()
	}
	return false
}

// mustNilState asserts the public nil-state contract of supported values.
// It distinguishes an empty non-nil matching result from a nil result pointer.
func mustNilState(t *testing.T, value any, wantNil bool, op string) {
	t.Helper()
	if got := isNilLike(value); got != wantNil {
		t.Fatalf("%s nil state = %v, want %v", op, got, wantNil)
	}
}

// mustEqualBool compares a boolean result without testing-framework dependencies.
func mustEqualBool(t *testing.T, got, want bool, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s = %v, want %v", op, got, want)
	}
}

// mustPairsEqual compares canonical pairs while preserving exact slice order.
// It also catches accidental local-index output in a global-index test.
func mustPairsEqual(t *testing.T, got, want []match.Pair) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("pairs=%v, want %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("pair[%d]=%+v, want %+v", i, got[i], want[i])
		}
	}
}

// mustFiniteCost checks the public arithmetic domain before any numeric comparison.
func mustFiniteCost(t *testing.T, cost float64) {
	t.Helper()
	if math.IsNaN(cost) || math.IsInf(cost, 0) || cost < 0 {
		t.Fatalf("cost is not finite nonnegative: %.17g", cost)
	}
}

// newFixtureMatrix builds a deterministic symmetric matrix for public API tests.
// It fails at setup time rather than silently ignoring matrix construction errors.
func newFixtureMatrix(t *testing.T, rows [][]float64) *matrix.Dense {
	t.Helper()
	n := len(rows)
	m, err := matrix.NewDense(n, n)
	if err != nil {
		t.Fatalf("matrix.NewDense: %v", err)
	}
	for i, row := range rows {
		if len(row) != n {
			t.Fatalf("row %d length %d != %d", i, len(row), n)
		}
		for j, value := range row {
			if err := m.Set(i, j, value); err != nil {
				t.Fatalf("Set(%d,%d,%.17g): %v", i, j, value, err)
			}
		}
	}
	return m
}

// pairCoverage verifies original-index membership and canonical publication.
// It is independent of any production mate or verification helper.
func pairCoverage(t *testing.T, vertices []int, pairs []match.Pair) {
	t.Helper()
	if len(pairs) != len(vertices)/2 {
		t.Fatalf("pair count %d, selected %d", len(pairs), len(vertices))
	}
	members := make(map[int]bool, len(vertices))
	for _, v := range vertices {
		members[v] = true
	}
	seen := make(map[int]bool, len(vertices))
	previousU := -1
	for _, p := range pairs {
		if !(p.U < p.V) || p.U <= previousU {
			t.Fatalf("noncanonical public pair order: %+v, after %d", p, previousU)
		}
		if !members[p.U] || !members[p.V] || seen[p.U] || seen[p.V] {
			t.Fatalf("invalid coverage pair=%+v vertices=%v", p, vertices)
		}
		seen[p.U], seen[p.V] = true, true
		previousU = p.U
	}
	if len(seen) != len(members) {
		t.Fatalf("missing vertices: selected=%v pairs=%v", vertices, pairs)
	}
}
