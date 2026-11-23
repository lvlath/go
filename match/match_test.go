// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package match_test verifies the public matching boundary and original indices.
// Mathematical output checks remain independent of internal engine predicates.
package match_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/match"
	"github.com/lvlath/go/matrix"
)

// TestPerfect_GreedySuboptimality confirms exact-vs-heuristic selection semantics.
// This counterexample cannot be satisfied by a hidden greedy fallback in Blossom.
func TestPerfect_GreedySuboptimality(t *testing.T) {
	m := newFixtureMatrix(t, [][]float64{
		{0, 2, 2, 1},
		{2, 0, 100, 2},
		{2, 100, 0, 3},
		{1, 2, 3, 0},
	})
	vertices := []int{0, 1, 2, 3}
	blossom, err := match.Perfect(m, vertices)
	if err != nil {
		t.Fatalf("Blossom: %v", err)
	}
	if blossom == nil {
		t.Fatal("solver returned a nil result without an error")
	}
	greedy, err := match.Perfect(m, vertices, match.WithAlgorithm(match.Greedy))
	if err != nil {
		t.Fatalf("Greedy: %v", err)
	}
	if greedy == nil {
		t.Fatal("solver returned a nil result without an error")
	}
	pairCoverage(t, vertices, blossom.Pairs)
	pairCoverage(t, vertices, greedy.Pairs)
	mustPairsEqual(t, blossom.Pairs, []match.Pair{{U: 0, V: 2}, {U: 1, V: 3}})
	mustPairsEqual(t, greedy.Pairs, []match.Pair{{U: 0, V: 3}, {U: 1, V: 2}})
	if blossom.Cost != 4 || greedy.Cost != 101 {
		t.Fatalf("objective Blossom=%.17g Greedy=%.17g, want 4/101", blossom.Cost, greedy.Cost)
	}
	mustEqualBool(t, blossom.Optimal, true, "Blossom internal optimality check")
	mustEqualBool(t, greedy.Optimal, false, "Greedy optimality claim")
}

// TestPerfect_IndexMappingPreservation proves public pairs contain original indices.
// The solver input is deliberately nonmonotone and must not be sorted beforehand.
func TestPerfect_IndexMappingPreservation(t *testing.T) {
	const n = 8
	rows := make([][]float64, n)
	for i := 0; i < n; i++ {
		rows[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			if i != j {
				rows[i][j] = 20
			}
		}
	}
	rows[5][7], rows[7][5] = 1, 1
	rows[1][3], rows[3][1] = 2, 2
	m := newFixtureMatrix(t, rows)
	vertices := []int{5, 1, 7, 3}
	original := append([]int(nil), vertices...)
	for _, algo := range []match.Algorithm{match.Blossom, match.Greedy} {
		result, err := match.Perfect(m, vertices, match.WithAlgorithm(algo))
		if err != nil {
			t.Fatalf("algorithm=%v: %v", algo, err)
		}
		if result == nil {
			t.Fatal("solver returned a nil result without an error")
		}
		pairCoverage(t, vertices, result.Pairs)
		mustPairsEqual(t, result.Pairs, []match.Pair{{U: 1, V: 3}, {U: 5, V: 7}})
		if result.Cost != 3 {
			t.Fatalf("algorithm %v: cost=%v, want 3", algo, result.Cost)
		}
	}
	for i := range vertices {
		if vertices[i] != original[i] {
			t.Fatalf("caller subset was mutated: %v vs %v", vertices, original)
		}
	}
}

// TestPerfect_OddVerticesRejected asserts the odd preflight before matrix reads.
// A nil matrix is intentionally supplied to anchor precedence.
func TestPerfect_OddVerticesRejected(t *testing.T) {
	result, err := match.Perfect(nil, []int{0, 2, 1})
	mustErrorIs(t, err, match.ErrOddVertices)
	mustNilState(t, result, true, "odd result")
}

// TestPerfect_ZeroAndEmptySubset checks the valid zero-cardinality specialization.
// Neither a nil nor an empty subset means the whole matrix is selected.
func TestPerfect_ZeroAndEmptySubset(t *testing.T) {
	for _, vertices := range [][]int{nil, {}} {
		result, err := match.Perfect(nil, vertices)
		if err != nil {
			t.Fatalf("empty subset: %v", err)
		}
		if result == nil {
			t.Fatal("solver returned a nil result without an error")
		}
		mustNilState(t, result, false, "empty result")
		if result.Pairs == nil || len(result.Pairs) != 0 || result.Cost != 0 || !result.Optimal {
			t.Fatalf("incorrect empty matching: %+v", result)
		}
	}
	greedy, err := match.Perfect(nil, nil, match.WithAlgorithm(match.Greedy))
	if err != nil || greedy == nil || greedy.Optimal {
		t.Fatalf("greedy empty flag must not claim optimum: %+v, %v", greedy, err)
	}
}

// TestPerfect_ResultCloneAndNilability anchors detached result memory.
// Editing one clone must not mutate another result or the selected input.
func TestPerfect_ResultCloneAndNilability(t *testing.T) {
	var nilResult *match.Result
	mustEqualBool(t, nilResult.IsNil(), true, "nil result")
	if nilResult.Clone() != nil {
		t.Fatal("nil Clone should be nil")
	}
	m := newFixtureMatrix(t, [][]float64{{0, 3}, {3, 0}})
	subset := []int{1, 0}
	result, err := match.Perfect(m, subset)
	if err != nil {
		t.Fatalf("Perfect: %v", err)
	}
	if result == nil {
		t.Fatal("solver returned a nil result without an error")
	}
	clone := result.Clone()
	if clone == nil || clone == result || &clone.Pairs[0] == &result.Pairs[0] {
		t.Fatal("Clone did not detach pairs")
	}
	clone.Pairs[0].U = 99
	if result.Pairs[0].U != 0 || subset[0] != 1 {
		t.Fatalf("unexpected retained alias: result=%+v subset=%v", result, subset)
	}
	if err := m.Set(0, 1, 9); err != nil {
		t.Fatalf("Set: %v", err)
	}
	if result.Cost != 3 {
		t.Fatalf("result retains a live source cost: %.17g", result.Cost)
	}
}

// TestPerfect_NumericCostBoundaries distinguishes raw cost and sum overflow.
// It rejects a successful infinite cost while preserving very small finite weights.
func TestPerfect_NumericCostBoundaries(t *testing.T) {
	for _, weight := range []float64{1e300, 4e-10} {
		m := newFixtureMatrix(t, [][]float64{{0, weight}, {weight, 0}})
		for _, algo := range []match.Algorithm{match.Blossom, match.Greedy} {
			result, err := match.Perfect(m, []int{0, 1}, match.WithAlgorithm(algo))
			if err != nil {
				t.Fatalf("weight=%g algo=%v: %v", weight, algo, err)
			}
			if result == nil {
				t.Fatal("solver returned a nil result without an error")
			}
			mustFiniteCost(t, result.Cost)
			if result.Cost != weight {
				t.Fatalf("weight=%g algo=%v cost=%.17g", weight, algo, result.Cost)
			}
		}
	}
	rows := [][]float64{
		{0, 1e308, 1e308, 1e308},
		{1e308, 0, 1e308, 1e308},
		{1e308, 1e308, 0, 1e308},
		{1e308, 1e308, 1e308, 0},
	}
	m := newFixtureMatrix(t, rows)
	result, err := match.Perfect(m, []int{0, 1, 2, 3}, match.WithAlgorithm(match.Greedy))
	mustErrorIs(t, err, match.ErrNumericOverflow)
	mustNilState(t, result, true, "overflow result")
}

// TestPerfect_OptionsAndValidation verifies first-error precedence and error causes.
// It never relies on errors.Error() text for classification.
func TestPerfect_OptionsAndValidation(t *testing.T) {
	defaultPolicy := match.DefaultOptions()
	if defaultPolicy.Algorithm != match.Blossom || defaultPolicy.Eps != 1e-12 {
		t.Fatalf("defaults changed: %+v", defaultPolicy)
	}
	invalid := []match.Option{
		nil,
		match.WithAlgorithm(match.Algorithm(255)),
		match.WithEpsilon(0),
		match.WithEpsilon(math.NaN()),
		match.WithEpsilon(math.Inf(1)),
	}
	for _, opt := range invalid {
		result, err := match.Perfect(nil, nil, opt)
		mustErrorIs(t, err, match.ErrInvalidOptions)
		mustNilState(t, result, true, "invalid option")
	}
	sourceCause := errors.New("custom option cause")
	result, err := match.Perfect(nil, nil, func(*match.Options) error { return sourceCause })
	if result != nil || !errors.Is(err, sourceCause) {
		t.Fatalf("callback cause was lost: %v", err)
	}
	result, err = match.Perfect(nil, nil, func(o *match.Options) error {
		o.Eps = 0
		return nil
	}, match.WithEpsilon(1e-12))
	mustErrorIs(t, err, match.ErrInvalidOptions)
	mustNilState(t, result, true, "intermediate invalid option")
}

// TestPerfect_InputAdmission verifies whole-matrix validation before subset bounds.
// Invalid cells outside the requested subset must not disappear silently.
func TestPerfect_InputAdmission(t *testing.T) {
	rows := [][]float64{
		{0, 1, 2, 3},
		{1, 0, 4, 5},
		{2, 4, 0, 6},
		{3, 5, 6, 0},
	}
	m := newFixtureMatrix(t, rows)
	for _, subset := range [][]int{{0, 0}, {0, 5}} {
		result, err := match.Perfect(m, subset)
		mustErrorIs(t, err, match.ErrInvalidVertices)
		mustNilState(t, result, true, "invalid vertex result")
	}
	rows[2][3] = -1
	bad := &testMatrix{rows: rows}
	result, err := match.Perfect(bad, []int{0, 0})
	mustErrorIs(t, err, match.ErrNegativeWeight)
	mustNilState(t, result, true, "outside-subset invalid matrix")
}

// testMatrix is a test-only mutable matrix backend with explicit read failures.
// It is never used by production match or any exported helper.
type testMatrix struct {
	rows      [][]float64
	atFailure error
}

func (m *testMatrix) Rows() int { return len(m.rows) }
func (m *testMatrix) Cols() int {
	if len(m.rows) == 0 {
		return 0
	}
	return len(m.rows[0])
}
func (m *testMatrix) At(i, j int) (float64, error) {
	if m.atFailure != nil {
		return 0, m.atFailure
	}
	if i < 0 || j < 0 || i >= m.Rows() || j >= len(m.rows[i]) {
		return 0, matrix.ErrOutOfRange
	}
	return m.rows[i][j], nil
}
func (m *testMatrix) Set(i, j int, value float64) error {
	if i < 0 || j < 0 || i >= m.Rows() || j >= len(m.rows[i]) {
		return matrix.ErrOutOfRange
	}
	m.rows[i][j] = value
	return nil
}
func (m *testMatrix) Clone() matrix.Matrix {
	clone := &testMatrix{rows: make([][]float64, len(m.rows)), atFailure: m.atFailure}
	for i := range m.rows {
		clone.rows[i] = append([]float64(nil), m.rows[i]...)
	}
	return clone
}

// TestPerfect_AtCausePreservation ensures a backend failure remains discoverable.
func TestPerfect_AtCausePreservation(t *testing.T) {
	cause := errors.New("backend At cause")
	m := &testMatrix{rows: [][]float64{{0, 1}, {1, 0}}, atFailure: cause}
	result, err := match.Perfect(m, []int{1, 0})
	mustNilState(t, result, true, "At error result")
	mustErrorIs(t, err, cause)
	mustErrorIs(t, err, match.ErrDimensionMismatch)
}

// TestResult_IsNilInterface checks match.Result through the actual Nilable interface.
// The interface remains non-nil while the stored receiver pointer is nil.
func TestResult_IsNilInterface(t *testing.T) {
	var result *match.Result
	var nilable interface{ IsNil() bool } = result
	if nilable == nil || !nilable.IsNil() {
		t.Fatal("typed-nil result must implement the real nilable interface")
	}
	mustNilState(t, result, true, "typed-nil result")
}

// TestPerfect_TypedNilMatrixConvention checks existing matrix/core nil detection.
// A nil *matrix.Dense is handled without reflection or a panic.
func TestPerfect_TypedNilMatrixConvention(t *testing.T) {
	var dense *matrix.Dense
	result, err := match.Perfect(dense, []int{0, 1})
	mustNilState(t, result, true, "typed-nil matrix result")
	mustErrorIs(t, err, match.ErrNilMatrix)
	mustErrorIs(t, err, matrix.ErrNilMatrix)
}
