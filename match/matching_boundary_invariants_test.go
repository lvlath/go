// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package match_test protects the local model, size admission and output boundary.
// It separates public source failures from malformed private solver artifacts.
package match_test

import (
	"errors"
	"math"
	"math/bits"
	"slices"
	"testing"

	"github.com/lvlath/go/match"
	"github.com/lvlath/go/matrix"
)

// TestMatchingSizeArithmetic verifies both int-element and byte-capacity guards.
// The tested dimensions depend on the target word size and never allocate a matrix.
func TestMatchingSizeArithmetic(t *testing.T) {
	// Squaring this order fits int, but its float64 byte count does not.
	byteOverflowOrder := 1 << (bits.UintSize/2 - 1)
	for _, n := range []int{-1, math.MaxInt, byteOverflowOrder} {
		count, err := match.CheckedSquareLengthForTest(n)
		mustErrorIs(t, err, match.ErrSizeOverflow)
		if count != 0 {
			t.Fatalf("n=%d returned unsafe element count %d", n, count)
		}
	}
	for _, n := range []int{0, 1, 4} {
		count, err := match.CheckedSquareLengthForTest(n)
		if err != nil || count != n*n {
			t.Fatalf("n=%d count=%d err=%v", n, count, err)
		}
	}
	mustErrorIs(t, match.CheckBlossomSizesForTest(-1), match.ErrInvalidMatching)
	mustErrorIs(t, match.CheckBlossomSizesForTest(math.MaxInt), match.ErrSizeOverflow)
	// This square fits its float64 buffer, but dense edge/event storage does not.
	edgeOverflowOrder := (1 << (bits.UintSize/2 - 2)) - 1
	if _, err := match.CheckedSquareLengthForTest(edgeOverflowOrder); err != nil {
		t.Fatalf("fixture square must fit: %v", err)
	}
	mustErrorIs(t, match.CheckBlossomSizesForTest(edgeOverflowOrder), match.ErrSizeOverflow)
	for _, n := range []int{0, 2, 64} {
		requireInvariantOK(t, match.CheckBlossomSizesForTest(n))
	}
}

// TestLocalMatchingProblemMappingAndRefusal checks the local-domain preparation law.
// Selected weights retain ordered original indices and a detached square buffer.
func TestLocalMatchingProblemMappingAndRefusal(t *testing.T) {
	rows := [][]float64{{0, 9, 1, 8}, {9, 0, 7, 2}, {1, 7, 0, 3}, {8, 2, 3, 0}}
	dist := newFixtureMatrix(t, rows)
	vertices := []int{2, 0, 3, 1}
	p, err := match.BuildMatchingProblemForTest(vertices, dist)
	requireInvariantOK(t, err)
	n, indices, weights := match.InspectMatchingProblemForTest(p)
	if n != 4 || !slices.Equal(indices, vertices) || weights[1] != 1 || weights[2*4+3] != 2 {
		t.Fatalf("local mapping n=%d indices=%v weights=%v", n, indices, weights)
	}
	vertices[0] = 1
	requireInvariantOK(t, dist.Set(2, 0, 99))
	_, again, againWeights := match.InspectMatchingProblemForTest(p)
	if again[0] != 2 || againWeights[1] != 1 {
		t.Fatal("local problem aliases caller storage")
	}
	p, err = match.BuildMatchingProblemForTest(nil, nil)
	requireInvariantOK(t, err)
	n, indices, weights = match.InspectMatchingProblemForTest(p)
	if n != 0 || len(indices) != 0 || len(weights) != 0 {
		t.Fatal("empty local problem has unexpected storage")
	}
	_, err = match.BuildMatchingProblemForTest([]int{0}, nil)
	mustErrorIs(t, err, match.ErrOddVertices)
	mustErrorIs(t, err, match.ErrInvalidMatching)
	_, err = match.BuildMatchingProblemForTest([]int{0, 1}, nil)
	mustErrorIs(t, err, match.ErrNilMatrix)
}

// TestLocalMateValidationAndCost checks exact mate symmetry and raw weight publication.
// No invalid local array is allowed to be indexed as a perfect matching.
func TestLocalMateValidationAndCost(t *testing.T) {
	for _, mates := range [][]int{{1}, {-1, 0}, {2, 0}, {0, 1}, {1, 2, 3, 2}} {
		mustErrorIs(t, match.VerifyPerfectMatchingForTest(mates), match.ErrInvalidMatching)
	}
	mates := []int{1, 0, 3, 2}
	requireInvariantOK(t, match.VerifyPerfectMatchingForTest(mates))
	indices := []int{0, 1, 2, 3}
	weights := []float64{0, 2, 5, 6, 2, 0, 7, 8, 5, 7, 0, 3, 6, 8, 3, 0}
	valid := match.NewMatchingProblemForTest(4, indices, weights)
	cost, err := match.MatchingCostForTest(valid, mates)
	if err != nil || cost != 5 {
		t.Fatalf("cost=%v err=%v", cost, err)
	}
	for _, tc := range []struct {
		name string
		p    match.MatchingProblemForTest
		mate []int
		want error
	}{
		{"malformed-mates", valid, []int{0, 1}, match.ErrInvalidMatching},
		{"wrong-order", match.NewMatchingProblemForTest(2, indices[:2], weights[:4]), mates, match.ErrInvalidMatching},
		{"wrong-map", match.NewMatchingProblemForTest(4, indices[:2], weights), mates, match.ErrInvalidMatching},
		{"short-weights", match.NewMatchingProblemForTest(4, indices, weights[:4]), mates, match.ErrInvalidMatching},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cost, err := match.MatchingCostForTest(tc.p, tc.mate)
			mustErrorIs(t, err, tc.want)
			if cost != 0 {
				t.Fatalf("partial cost=%v", cost)
			}
		})
	}
	for _, bad := range []float64{math.NaN(), math.Inf(1), -1} {
		invalid := slices.Clone(weights)
		invalid[1] = bad
		cost, err := match.MatchingCostForTest(match.NewMatchingProblemForTest(4, indices, invalid), mates)
		mustErrorIs(t, err, match.ErrInvalidMatching)
		if cost != 0 {
			t.Fatalf("invalid weight published cost %v", cost)
		}
	}
	large := slices.Clone(weights)
	large[1], large[11] = math.MaxFloat64, math.MaxFloat64
	cost, err = match.MatchingCostForTest(match.NewMatchingProblemForTest(4, indices, large), mates)
	mustErrorIs(t, err, match.ErrNumericOverflow)
	if cost != 0 {
		t.Fatalf("overflow returned %v", cost)
	}
}

// TestCanonicalPairsRejectMalformedCoverage checks publication independently of solving.
// Invalid pair count, self-pairs, foreign endpoints and duplicates are all rejected.
func TestCanonicalPairsRejectMalformedCoverage(t *testing.T) {
	vertices := []int{7, 1, 5, 3}
	for _, tc := range []struct {
		name     string
		vertices []int
		pairs    []match.Pair
	}{
		{"odd", vertices[:3], []match.Pair{{U: 1, V: 3}}},
		{"pair-count", vertices, []match.Pair{{U: 1, V: 3}}},
		{"duplicate-subset", []int{1, 3, 3, 7}, []match.Pair{{U: 1, V: 3}, {U: 3, V: 7}}},
		{"self-pair", vertices, []match.Pair{{U: 1, V: 1}, {U: 5, V: 7}}},
		{"foreign", vertices, []match.Pair{{U: 1, V: 9}, {U: 5, V: 7}}},
		{"duplicate-endpoint", vertices, []match.Pair{{U: 1, V: 3}, {U: 1, V: 7}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mustErrorIs(t, match.CanonicalizeMatchingPairsForTest(tc.pairs, tc.vertices), match.ErrInvalidMatching)
		})
	}
	pairs := []match.Pair{{U: 7, V: 5}, {U: 3, V: 1}}
	requireInvariantOK(t, match.CanonicalizeMatchingPairsForTest(pairs, vertices))
	mustPairsEqual(t, pairs, []match.Pair{{U: 1, V: 3}, {U: 5, V: 7}})
	if !slices.Equal(vertices, []int{7, 1, 5, 3}) {
		t.Fatal("output sort changed input tie order")
	}
}

// TestLocalBlossomProblemAdmission verifies private numerical preparation separately.
// Public matrix admission normally prevents malformed local shape and nonfinite costs.
func TestLocalBlossomProblemAdmission(t *testing.T) {
	for _, tc := range []struct {
		name    string
		n       int
		indices []int
		weights []float64
		eps     float64
		want    error
	}{
		{"negative", -1, nil, nil, match.DefaultEpsilon, match.ErrInvalidMatching},
		{"odd", 1, []int{0}, []float64{0}, match.DefaultEpsilon, match.ErrInvalidMatching},
		{"short-map", 2, []int{0}, []float64{0, 1, 1, 0}, match.DefaultEpsilon, match.ErrInvalidMatching},
		{"short-buffer", 2, []int{0, 1}, []float64{0, 1, 1}, match.DefaultEpsilon, match.ErrInvalidMatching},
		{"nan", 2, []int{0, 1}, []float64{0, math.NaN(), 1, 0}, match.DefaultEpsilon, match.ErrNaNInf},
		{"negative-edge", 2, []int{0, 1}, []float64{0, -1, -1, 0}, match.DefaultEpsilon, match.ErrNegativeWeight},
		{"asymmetric", 2, []int{0, 1}, []float64{0, 1, 2, 0}, match.DefaultEpsilon, match.ErrAsymmetry},
		{"epsilon", 2, []int{0, 1}, []float64{0, 1, 1, 0}, 0, match.ErrInvalidOptions},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := match.NewMatchingProblemForTest(tc.n, tc.indices, tc.weights)
			e, err := match.NewBlossomEngineForTest(p, tc.eps)
			mustErrorIs(t, err, tc.want)
			if e != nil {
				t.Fatal("invalid local problem constructed an engine")
			}
			mates, cost, err := match.SolveBlossomProblemForTest(p, tc.eps)
			mustErrorIs(t, err, tc.want)
			if mates != nil || cost != 0 {
				t.Fatalf("invalid local problem published %v %v", mates, cost)
			}
		})
	}
	p := match.NewMatchingProblemForTest(0, nil, nil)
	mates, cost, err := match.SolveBlossomProblemForTest(p, match.DefaultEpsilon)
	if err != nil || mates == nil || len(mates) != 0 || cost != 0 {
		t.Fatalf("empty local solve %v %v %v", mates, cost, err)
	}
	// Direct edge construction validates the costs it consumes, even if called
	// independently of the engine's earlier full local-model validation.
	for _, bad := range []float64{math.Inf(1), -1} {
		_, _, _, err := match.BuildBlossomEdgesForTest(match.NewMatchingProblemForTest(2, []int{0, 1}, []float64{0, bad, bad, 0}))
		want := match.ErrNaNInf
		if bad < 0 {
			want = match.ErrNegativeWeight
		}
		mustErrorIs(t, err, want)
	}
}

// lateMatchingValue injects one backend scalar violation after normal admission.
// This deliberately broken backend is not a supported concurrent mutation model.
type lateMatchingValue struct {
	*observedMatrix
	badRead  int
	badValue float64
}

// At returns the underlying observation except for one deliberate invalid value.
// The read counter belongs to the embedded test backend; no source cell is edited.
func (m *lateMatchingValue) At(i, j int) (float64, error) {
	value, err := m.observedMatrix.At(i, j)
	if err == nil && m.reads == m.badRead {
		return m.badValue, nil
	}
	return value, err
}

// TestPerfectCopyPhaseRejectsInvalidBackendValues checks the second-pass safety wall.
// A source that breaks its stable-value contract cannot yield a partial matching.
func TestPerfectCopyPhaseRejectsInvalidBackendValues(t *testing.T) {
	// For n=2, four scalar reads and two symmetry reads precede copying.
	const diagonalCopyRead = 7
	const edgeCopyRead = 8
	for _, tc := range []struct {
		name  string
		read  int
		value float64
		want  error
	}{
		{"nan", edgeCopyRead, math.NaN(), match.ErrNaNInf},
		{"negative-inf", edgeCopyRead, math.Inf(-1), match.ErrNaNInf},
		{"positive-inf", edgeCopyRead, math.Inf(1), match.ErrIncompleteGraph},
		{"negative-edge", edgeCopyRead, -1, match.ErrNegativeWeight},
		{"nonzero-diagonal", diagonalCopyRead, 1, match.ErrNonZeroDiagonal},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, algo := range []match.Algorithm{match.Blossom, match.Greedy} {
				backend := &lateMatchingValue{observedMatrix: &observedMatrix{rows: [][]float64{{0, 1}, {1, 0}}}, badRead: tc.read, badValue: tc.value}
				result, err := match.Perfect(backend, []int{0, 1}, match.WithAlgorithm(algo))
				mustErrorIs(t, err, tc.want)
				if result != nil || backend.reads != tc.read {
					t.Fatalf("algo=%v result=%+v reads=%d", algo, result, backend.reads)
				}
				if errors.Is(err, match.ErrNaNInf) && !errors.Is(err, matrix.ErrNaNInf) {
					t.Fatal("matrix numeric sentinel lost")
				}
			}
		})
	}
}
