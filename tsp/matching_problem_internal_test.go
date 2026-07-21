// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package tsp

import (
	"errors"
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/lvlath/go/matrix"
)

type internalDense struct {
	rows [][]float64
}

var _ matrix.Matrix = internalDense{}

func (m internalDense) Rows() int { return len(m.rows) }

func (m internalDense) Cols() int {
	if len(m.rows) == 0 {
		return 0
	}

	return len(m.rows[0])
}

func (m internalDense) At(row int, col int) (float64, error) {
	if row < 0 || row >= m.Rows() || col < 0 || col >= m.Cols() {
		return 0, matrix.ErrIndexOutOfBounds
	}

	return m.rows[row][col], nil
}

func (m internalDense) Set(row int, col int, value float64) error {
	if row < 0 || row >= m.Rows() || col < 0 || col >= m.Cols() {
		return matrix.ErrIndexOutOfBounds
	}

	m.rows[row][col] = value

	return nil
}

func (m internalDense) Clone() matrix.Matrix {
	clone := make([][]float64, len(m.rows))
	for row := range m.rows {
		clone[row] = append([]float64(nil), m.rows[row]...)
	}

	return internalDense{rows: clone}
}

func internalMatrixFromRows(rows [][]float64) internalDense {
	clone := make([][]float64, len(rows))
	for row := range rows {
		clone[row] = append([]float64(nil), rows[row]...)
	}

	return internalDense{rows: clone}
}

func internalCompleteRows(n int) [][]float64 {
	rows := make([][]float64, n)

	for row := 0; row < n; row++ {
		rows[row] = make([]float64, n)
		for col := 0; col < n; col++ {
			if row == col {
				continue
			}

			rows[row][col] = 1
		}
	}

	return rows
}

func internalSeededMatchingProblem(k int, seed int64) matchingProblem {
	rng := rand.New(rand.NewSource(seed))

	problem := matchingProblem{
		odd: make([]int, k),
		n:   k,
		w:   make([]float64, k*k),
	}

	for vertex := 0; vertex < k; vertex++ {
		problem.odd[vertex] = vertex
	}

	for row := 0; row < k; row++ {
		for col := row + 1; col < k; col++ {
			weight := 1 + float64(rng.Intn(10_000))/100
			problem.w[row*k+col] = weight
			problem.w[col*k+row] = weight
		}
	}

	return problem
}

func internalCloneAdj(adj [][]int) [][]int {
	clone := make([][]int, len(adj))
	for row := range adj {
		clone[row] = append([]int(nil), adj[row]...)
	}

	return clone
}

func internalAdjEqual(a [][]int, b [][]int) bool {
	if len(a) != len(b) {
		return false
	}

	for row := range a {
		if !slices.Equal(a[row], b[row]) {
			return false
		}
	}

	return true
}

func internalEdgeID(t *testing.T, engine *blossomEngine, u int, v int) int {
	t.Helper()

	for edgeID, edge := range engine.edges {
		if (edge.u == u && edge.v == v) || (edge.u == v && edge.v == u) {
			return edgeID
		}
	}

	t.Fatalf("missing edge %d-%d", u, v)
	return noEdge
}

func exactMatchingOracleForTest(problem matchingProblem) ([]int, float64, error) {
	if problem.n == 0 {
		return []int{}, 0, nil
	}
	if (problem.n&1) == 1 || len(problem.odd) != problem.n || len(problem.w) != problem.n*problem.n {
		return nil, 0, ErrInvalidMatching
	}

	used := make([]bool, problem.n)
	current := make([]int, problem.n)
	best := make([]int, problem.n)

	for vertex := range current {
		current[vertex] = matchingUnmatched
		best[vertex] = matchingUnmatched
	}

	bestCost := math.Inf(1)

	var search func(float64)
	search = func(cost float64) {
		first := matchingUnmatched
		for vertex := 0; vertex < problem.n; vertex++ {
			if !used[vertex] {
				first = vertex
				break
			}
		}

		if first == matchingUnmatched {
			if cost < bestCost {
				bestCost = cost
				copy(best, current)
			}
			return
		}

		used[first] = true
		for partner := first + 1; partner < problem.n; partner++ {
			if used[partner] {
				continue
			}

			used[partner] = true
			current[first] = partner
			current[partner] = first

			search(cost + problem.at(first, partner))

			current[first] = matchingUnmatched
			current[partner] = matchingUnmatched
			used[partner] = false
		}
		used[first] = false
	}

	search(0)

	if math.IsInf(bestCost, 1) {
		return nil, 0, ErrIncompleteGraph
	}
	if err := verifyPerfectMatching(best); err != nil {
		return nil, 0, err
	}

	return best, round1e9(bestCost), nil
}

func TestBuildMatchingProblemCopiesLocalCostsInOddOrder(t *testing.T) {
	rows := [][]float64{
		{0, 9, 1, 8},
		{9, 0, 7, 2},
		{1, 7, 0, 3},
		{8, 2, 3, 0},
	}

	problem, err := buildMatchingProblem([]int{2, 0, 3, 1}, internalMatrixFromRows(rows))
	if err != nil {
		t.Fatalf("buildMatchingProblem: %v", err)
	}

	if problem.n != 4 {
		t.Fatalf("problem.n=%d, want 4", problem.n)
	}
	if !slices.Equal(problem.odd, []int{2, 0, 3, 1}) {
		t.Fatalf("odd order changed: %v", problem.odd)
	}
	if problem.at(0, 1) != 1 || problem.at(2, 3) != 2 {
		t.Fatalf("local costs not copied in odd order")
	}
}

func TestBuildMatchingProblemRejectsMalformedOddSet(t *testing.T) {
	dist := internalMatrixFromRows(internalCompleteRows(4))

	if _, err := buildMatchingProblem([]int{0, 1, 2}, dist); !errors.Is(err, ErrInvalidMatching) {
		t.Fatalf("odd cardinality: got %v", err)
	}
	if _, err := buildMatchingProblem([]int{0, 1, 1, 2}, dist); !errors.Is(err, ErrInvalidMatching) {
		t.Fatalf("duplicate odd vertex: got %v", err)
	}
	if _, err := buildMatchingProblem([]int{0, 1, 2, 9}, dist); !errors.Is(err, ErrInvalidMatching) {
		t.Fatalf("out of range odd vertex: got %v", err)
	}
}

func TestVerifyPerfectMatchingAndMatchingCost(t *testing.T) {
	problem := matchingProblem{
		odd: []int{0, 1, 2, 3},
		n:   4,
		w: []float64{
			0, 2, 5, 6,
			2, 0, 7, 8,
			5, 7, 0, 3,
			6, 8, 3, 0,
		},
	}

	match := []int{1, 0, 3, 2}

	if err := verifyPerfectMatching(match); err != nil {
		t.Fatalf("verifyPerfectMatching: %v", err)
	}

	cost, err := matchingCost(problem, match)
	if err != nil {
		t.Fatalf("matchingCost: %v", err)
	}
	if cost != 5 {
		t.Fatalf("matching cost got %.3f want 5", cost)
	}
}

func TestAppendPerfectMatchingAndRollbackAdjacency(t *testing.T) {
	problem := matchingProblem{
		odd: []int{3, 1, 2, 0},
		n:   4,
		w:   make([]float64, 16),
	}
	match := []int{1, 0, 3, 2}

	adj := [][]int{
		{1},
		{0},
		{},
		{},
	}
	before := internalCloneAdj(adj)
	lengths := snapshotAdjLengths(adj)

	if err := appendPerfectMatching(problem, match, adj); err != nil {
		t.Fatalf("appendPerfectMatching: %v", err)
	}

	if len(adj[3]) != len(before[3])+1 || len(adj[1]) != len(before[1])+1 {
		t.Fatalf("first matched pair was not appended: before=%v after=%v", before, adj)
	}

	rollbackAdj(adj, lengths)
	if !internalAdjEqual(adj, before) {
		t.Fatalf("rollback failed: before=%v after=%v", before, adj)
	}
}
