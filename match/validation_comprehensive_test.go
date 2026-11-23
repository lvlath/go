// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package match_test anchors error precedence and safe API ownership.
// These external tests exercise only the exported match contract.
package match_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/match"
	"github.com/lvlath/go/matrix"
)

// observedMatrix is a test-only Matrix that records source reads and can inject
// a distinct backend error in any one of the three admission/copy phases.
type observedMatrix struct {
	rows        [][]float64
	rowCount    int
	columnCount int
	reads       int
	failRead    int
	cause       error
}

// Rows reports a possibly synthetic dimension for representability tests.
func (m *observedMatrix) Rows() int {
	if m.rowCount != 0 {
		return m.rowCount
	}
	return len(m.rows)
}

// Cols reports a possibly synthetic shape without accessing source entries.
func (m *observedMatrix) Cols() int {
	if m.columnCount != 0 {
		return m.columnCount
	}
	if len(m.rows) == 0 {
		return 0
	}
	return len(m.rows[0])
}

// At returns one original entry or a deterministic injected backend error.
func (m *observedMatrix) At(i, j int) (float64, error) {
	m.reads++
	if m.cause != nil && m.reads == m.failRead {
		return 0, m.cause
	}
	if i < 0 || i >= len(m.rows) || j < 0 || j >= len(m.rows[i]) {
		return 0, matrix.ErrOutOfRange
	}
	return m.rows[i][j], nil
}

// Set changes only this test backend's own storage.
func (m *observedMatrix) Set(i, j int, v float64) error {
	if i < 0 || i >= len(m.rows) || j < 0 || j >= len(m.rows[i]) {
		return matrix.ErrOutOfRange
	}
	m.rows[i][j] = v
	return nil
}

// Clone duplicates all observable matrix cells and the injected failure policy.
func (m *observedMatrix) Clone() matrix.Matrix {
	copyOfMatrix := *m
	copyOfMatrix.rows = make([][]float64, len(m.rows))
	for i, row := range m.rows {
		copyOfMatrix.rows[i] = append([]float64(nil), row...)
	}
	return &copyOfMatrix
}

// TestPerfectWholeMatrixPolicy checks errors independent of subset membership.
// Both algorithms must reject corruption elsewhere in the complete matrix.
func TestPerfectWholeMatrixPolicy(t *testing.T) {
	base := [][]float64{
		{0, 1, 2, 3},
		{1, 0, 4, 5},
		{2, 4, 0, 6},
		{3, 5, 6, 0},
	}
	tests := []struct {
		name string
		edit func([][]float64)
		want error
	}{
		{"nan", func(rows [][]float64) { rows[2][3] = math.NaN() }, match.ErrNaNInf},
		{"negative-infinity", func(rows [][]float64) { rows[2][3] = math.Inf(-1) }, match.ErrNaNInf},
		{"positive-infinity", func(rows [][]float64) { rows[2][3] = math.Inf(1) }, match.ErrIncompleteGraph},
		{"negative", func(rows [][]float64) { rows[2][3] = -4 }, match.ErrNegativeWeight},
		{"diagonal-nonzero", func(rows [][]float64) { rows[2][2] = 1e-8 }, match.ErrNonZeroDiagonal},
		{"diagonal-positive-infinity", func(rows [][]float64) { rows[2][2] = math.Inf(1) }, match.ErrNonZeroDiagonal},
		{"asymmetric", func(rows [][]float64) { rows[2][3] = 6.1 }, match.ErrAsymmetry},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rows := make([][]float64, len(base))
			for i, row := range base {
				rows[i] = append([]float64(nil), row...)
			}
			tc.edit(rows)
			for _, algo := range []match.Algorithm{match.Blossom, match.Greedy} {
				result, err := match.Perfect(&observedMatrix{rows: rows}, []int{0, 1}, match.WithAlgorithm(algo))
				mustErrorIs(t, err, tc.want)
				if result != nil {
					t.Fatalf("algo=%v returned partial result: %+v", algo, result)
				}
			}
		})
	}
}

// TestPerfectValidationPrecedence ensures malformed cardinality and shape short-circuit safely.
// It also verifies size admission occurs before any allocated matrix cells are read.
func TestPerfectValidationPrecedence(t *testing.T) {
	cause := errors.New("no At call expected")
	badShape := &observedMatrix{rowCount: 3, columnCount: 4, cause: cause, failRead: 1}
	if result, err := match.Perfect(badShape, []int{0, 1}); result != nil || !errors.Is(err, match.ErrNonSquare) || badShape.reads != 0 {
		t.Fatalf("nonsquare: result=%+v err=%v reads=%d", result, err, badShape.reads)
	}
	tooSmall := &observedMatrix{rows: [][]float64{{0}}, cause: cause, failRead: 1}
	if result, err := match.Perfect(tooSmall, []int{0, 1}); result != nil || !errors.Is(err, match.ErrInvalidDimensions) || tooSmall.reads != 0 {
		t.Fatalf("one-row: result=%+v err=%v reads=%d", result, err, tooSmall.reads)
	}
	maxInt := int(^uint(0) >> 1)
	tooLarge := &observedMatrix{rowCount: maxInt, columnCount: maxInt, cause: cause, failRead: 1}
	if result, err := match.Perfect(tooLarge, []int{0, 1}); result != nil || !errors.Is(err, match.ErrSizeOverflow) || tooLarge.reads != 0 {
		t.Fatalf("oversized shape: result=%+v err=%v reads=%d", result, err, tooLarge.reads)
	}
	for _, algo := range []match.Algorithm{match.Blossom, match.Greedy} {
		input := &observedMatrix{cause: cause, failRead: 1}
		if result, err := match.Perfect(input, []int{0, 1, 2}, match.WithAlgorithm(algo)); result != nil || !errors.Is(err, match.ErrOddVertices) || input.reads != 0 {
			t.Fatalf("odd precedence algo=%v err=%v", algo, err)
		}
		if result, err := match.Perfect(input, nil, match.WithAlgorithm(algo)); result == nil || err != nil || input.reads != 0 {
			t.Fatalf("empty reads input algo=%v", algo)
		}
	}
}

// TestPerfectAtCauseAcrossAdmissionStages anchors the original source read sequence.
// A backend failure remains accessible through errors.Is and never yields a result.
func TestPerfectAtCauseAcrossAdmissionStages(t *testing.T) {
	for _, tc := range []struct {
		name string
		read int
	}{
		{"scalar", 1}, {"symmetry", 5}, {"copy", 7},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cause := errors.New("matrix backend temporary failure")
			matrixOfCosts := &observedMatrix{rows: [][]float64{{0, 1}, {1, 0}}, cause: cause, failRead: tc.read}
			result, err := match.Perfect(matrixOfCosts, []int{1, 0})
			if result != nil {
				t.Fatalf("unexpected half-matching: %+v", result)
			}
			mustErrorIs(t, err, match.ErrDimensionMismatch)
			mustErrorIs(t, err, cause)
			if matrixOfCosts.reads != tc.read {
				t.Fatalf("read=%d want=%d", matrixOfCosts.reads, tc.read)
			}
		})
	}
}

// TestPerfectNearSymmetricDirectionAndStrictEpsilon verifies two distinct tolerances.
// New public cost uses the selected local direction, while local Blossom checks Eps.
func TestPerfectNearSymmetricDirectionAndStrictEpsilon(t *testing.T) {
	const difference = 5e-13
	rows := [][]float64{{0, 10}, {10 + difference, 0}}
	m := &observedMatrix{rows: rows}
	for _, algo := range []match.Algorithm{match.Blossom, match.Greedy} {
		result, err := match.Perfect(m, []int{1, 0}, match.WithAlgorithm(algo))
		if err != nil {
			t.Fatalf("algo=%v: %v", algo, err)
		}
		if result == nil {
			t.Fatal("solver returned a nil result without an error")
		}
		mustPairsEqual(t, result.Pairs, []match.Pair{{U: 0, V: 1}})
		// Blossom sums local index 0->1, whereas Greedy chooses its last
		// remaining pivot (global 0) and pays directional cost 0->1.
		want := rows[1][0]
		if algo == match.Greedy {
			want = rows[0][1]
		}
		if result.Cost != want {
			t.Fatalf("algo=%v changed selected directional cost got %.17g want %.17g", algo, result.Cost, want)
		}
	}
	result, err := match.Perfect(m, []int{1, 0}, match.WithEpsilon(1e-14))
	if result != nil {
		t.Fatalf("strict local asymmetry returned result: %+v", result)
	}
	mustErrorIs(t, err, match.ErrAsymmetry)
	// Greedy does not borrow the caller's Blossom Eps for partner selection.
	result, err = match.Perfect(m, []int{1, 0}, match.WithAlgorithm(match.Greedy), match.WithEpsilon(1e-14))
	if err != nil || result == nil {
		t.Fatalf("unexpected greedy epsilon effect: %+v %v", result, err)
	}
}

// TestPerfectOptionStateAndHelpers verifies the public option callbacks and defaults.
// No callback may panic on a nil *Options, and broken intermediate state is rejected.
func TestPerfectOptionStateAndHelpers(t *testing.T) {
	if err := match.WithAlgorithm(match.Greedy)(nil); !errors.Is(err, match.ErrInvalidOptions) {
		t.Fatalf("nil setter: %v", err)
	}
	if err := match.WithEpsilon(0)(nil); !errors.Is(err, match.ErrInvalidOptions) {
		t.Fatalf("nil tolerance setter: %v", err)
	}
	for _, v := range []float64{0, -1, math.NaN(), math.Inf(-1), math.Inf(1)} {
		result, err := match.Perfect(nil, nil, match.WithEpsilon(v))
		if result != nil || !errors.Is(err, match.ErrInvalidOptions) {
			t.Fatalf("epsilon=%v result=%+v err=%v", v, result, err)
		}
	}
	first, err := match.Perfect(nil, nil, match.WithAlgorithm(match.Greedy), match.WithAlgorithm(match.Blossom))
	if err != nil || first == nil || first.Algorithm != match.Blossom {
		t.Fatalf("last valid assignment: %+v %v", first, err)
	}
	invalid, err := match.Perfect(nil, nil, func(o *match.Options) error { o.Algorithm = match.Algorithm(127); return nil }, match.WithAlgorithm(match.Blossom))
	if invalid != nil || !errors.Is(err, match.ErrInvalidOptions) {
		t.Fatalf("intermediate invalid enum: %+v %v", invalid, err)
	}
}

// TestPerfectGreedyPivotAndTolerance keeps the historical partner-selection rule.
// The smallest original index may win a near-tie after a strictly cheaper candidate.
func TestPerfectGreedyPivotAndTolerance(t *testing.T) {
	rows := [][]float64{
		{0, 20, 20, 1 + 5e-13},
		{20, 0, 3, 1},
		{20, 3, 0, 1},
		{1 + 5e-13, 1, 1, 0},
	}
	result, err := match.Perfect(&observedMatrix{rows: rows}, []int{2, 1, 0, 3}, match.WithAlgorithm(match.Greedy))
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Fatal("solver returned a nil result without an error")
	}
	pairCoverage(t, []int{2, 1, 0, 3}, result.Pairs)
	// Pivot 3 reads candidate 2, then 1, then 0; index 0 wins a near-tie.
	mustPairsEqual(t, result.Pairs, []match.Pair{{U: 0, V: 3}, {U: 1, V: 2}})
	want := rows[3][0] + rows[1][2]
	if result.Cost != want {
		t.Fatalf("selection order sum: got %.17g want %.17g", result.Cost, want)
	}
}
