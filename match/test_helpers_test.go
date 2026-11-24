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

// newInvariantEngine creates one independent engine and its live test-only view.
// It uses the real constructor so every fault is introduced into an otherwise valid shape.
// Implementation:
//   - Stage 1: Build a deterministic finite symmetric local problem.
//   - Stage 2: Construct the actual weighted engine and obtain its state view.
//
// Behavior highlights:
//   - Each case owns all mutable engine data.
//
// Inputs:
//   - t: external test harness; n: small even matching order.
//
// Returns:
//   - A valid engine and its live state view.
//
// Errors:
//   - Fixture construction failures stop the current test.
//
// Determinism:
//   - Fixed seed and local vertex ordering.
//
// Complexity:
//   - Time and Space O(n^2), including dense edges and local weights.
//
// Notes:
//   - Never reuse a corrupted fixture in another subtest.
//
// AI-Hints:
//   - Test rejection of one invariant at a time; do not fabricate invalid pointers.
func newInvariantEngine(t *testing.T, n int) (*match.BlossomEngineForTest, match.BlossomStateForTest) {
	t.Helper()
	e, err := match.NewBlossomEngineForTest(invariantSeededProblem(n, 23), match.DefaultEpsilon)
	if err != nil || e == nil {
		t.Fatalf("construct engine: %v", err)
	}
	return e, match.InspectBlossomStateForTest(e)
}

// requireInvariantOK stops a test at the first failed setup operation.
// Subsequent assertions must never depend on partially constructed state.
func requireInvariantOK(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

// contractedTriangle creates a real contracted outer blossom over vertices 0,1,2.
// The alternating forest uses matched edge 1-2, grow edge 0-2 and closing edge 0-1.
// Implementation:
//   - Stage 1: Seed the reciprocal internal match and exposed root.
//   - Stage 2: Apply the production grow and shrink operations.
//   - Stage 3: Return the actual contracted node with its live state.
//
// Behavior highlights:
//   - No fabricated cycle or duplicate contraction implementation.
//
// Inputs:
//   - t: external test harness.
//
// Returns:
//   - Six-vertex engine, live state, allocated triangle node.
//
// Errors:
//   - Any rejected setup event fails the test immediately.
//
// Determinism:
//   - The same three original vertices and edge IDs are used on every invocation.
//
// Complexity:
//   - O(1) for the fixed fixture; construction uses the ordinary dense engine.
//
// Notes:
//   - The fixture exercises structural event semantics, not tightness selection.
//
// AI-Hints:
//   - Use actual grow/shrink to obtain valid cycle metadata before corrupting one field.
func contractedTriangle(t *testing.T) (*match.BlossomEngineForTest, match.BlossomStateForTest, int) {
	t.Helper()
	e, s := newInvariantEngine(t, 6)
	requireInvariantOK(t, match.SetMatchedEdgeForTest(e, invariantEdgeID(t, e, 1, 2)))
	match.ResetForestForTest(e)
	match.AssignOuterRootForTest(e, 0)
	grow := match.NewBlossomEventForTest(match.EventGrowForTest, invariantEdgeID(t, e, 0, 2), 0, 2, 0, 2)
	requireInvariantOK(t, match.GrowForTest(e, grow))
	node := *s.NextNode
	shrink := match.NewBlossomEventForTest(match.EventShrinkForTest, invariantEdgeID(t, e, 0, 1), 0, 1, 0, 1)
	requireInvariantOK(t, match.ShrinkForTest(e, shrink))
	if !s.Active[node] || s.Base[node] != 0 || len(s.Cycles[node]) != 3 {
		t.Fatalf("invalid contracted triangle: node=%d base=%d cycle=%v", node, s.Base[node], s.Cycles[node])
	}
	return e, s, node
}

// invariantProblem converts the existing deterministic square fixture into local coordinates.
// It allocates an identity subset and row-major storage; time and space are O(k^2).
func invariantProblem(rows [][]float64) match.MatchingProblemForTest {
	k := len(rows)
	vertices := make([]int, k)
	weights := make([]float64, 0, k*k)
	for i := range rows {
		vertices[i] = i
		weights = append(weights, rows[i]...)
	}
	return match.NewMatchingProblemForTest(k, vertices, weights)
}

// invariantSeededProblem reuses the historical integer-cent fixture generator.
// The identity map makes forest node assertions independent of public pair sorting.
func invariantSeededProblem(k int, seed int64) match.MatchingProblemForTest {
	return invariantProblem(legacySeededWeights(k, seed))
}

// invariantNearTieProblem reuses the historical near-tie fixture without rescaling.
// Its wide and narrow dual tolerances are intentionally distinguishable.
func invariantNearTieProblem(k int, seed int64) match.MatchingProblemForTest {
	return invariantProblem(legacyNearTieWeights(k, seed))
}

// invariantEdgeID locates a real edge and fails the external test if absent.
// It avoids independently reimplementing the dense engine's edge-numbering formula.
func invariantEdgeID(t *testing.T, e *match.BlossomEngineForTest, u, v int) int {
	t.Helper()
	id, ok := match.FindBlossomEdgeForTest(e, u, v)
	if !ok {
		t.Fatalf("missing engine edge %d-%d", u, v)
	}
	return id
}
