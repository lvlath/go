// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package match_test independently checks Blossom results against bounded pairing enumeration.
// The oracle is test-only and must never be used as a production fallback.
package match_test

import (
	"math/rand"
	"testing"

	"github.com/lvlath/go/match"
)

// exhaustivePairingCost independently searches every small perfect pairing.
// A strict state budget prevents accidental exponential test resource leaks.
// Implementation:
//   - Stage 1: Choose the first unpaired vertex.
//   - Stage 2: Try each remaining partner recursively.
//   - Stage 3: Retain the minimum integer objective over completed pairings.
//
// Behavior highlights:
//   - No production matching helpers or rounded costs are consulted.
//
// Inputs:
//   - weights: small symmetric integer matrix; budget: maximum recursive calls.
//
// Returns:
//   - int64: minimum cost, int: consumed state budget, bool: complete search.
//
// Errors:
//   - Exhaustion is represented as complete=false, never as a false optimum.
//
// Determinism:
//   - First-unmatched vertex, increasing partner index.
//
// Complexity:
//   - Exponential in k; restricted to test k<=10.
//
// Notes:
//   - Integer sums are exact for the bounded fixture weights.
//
// AI-Hints:
//   - Do not use this oracle for production or publish a truncated minimum.
func exhaustivePairingCost(weights [][]int64, budget int) (int64, int, bool) {
	k := len(weights)
	if k < 0 || k > 10 || k&1 != 0 || budget <= 0 {
		return 0, 0, false
	}
	used := make([]bool, k)
	states := 0
	best := int64(1 << 62)
	complete := true
	var search func(int64)
	search = func(cost int64) {
		if !complete {
			return
		}
		states++
		if states > budget {
			complete = false
			return
		}
		first := -1
		for i := 0; i < k; i++ {
			if !used[i] {
				first = i
				break
			}
		}
		if first < 0 {
			if cost < best {
				best = cost
			}
			return
		}
		used[first] = true
		for partner := first + 1; partner < k; partner++ {
			if used[partner] {
				continue
			}
			used[partner] = true
			search(cost + weights[first][partner])
			used[partner] = false
		}
		used[first] = false
	}
	search(0)
	if !complete {
		return 0, states, false
	}
	return best, states, true
}

// TestPerfect_SmallExactOracle verifies the original objective independently.
// Input generation is fixed-seed and uses only integer-exact weights.
func TestPerfect_SmallExactOracle(t *testing.T) {
	for _, k := range []int{2, 4, 6, 8, 10} {
		for _, seed := range []int64{1, 3, 23, 86} {
			rng := rand.New(rand.NewSource(seed))
			weights := make([][]int64, k)
			rows := make([][]float64, k)
			for i := range rows {
				weights[i] = make([]int64, k)
				rows[i] = make([]float64, k)
			}
			for i := 0; i < k; i++ {
				for j := i + 1; j < k; j++ {
					w := int64(1 + rng.Intn(200))
					weights[i][j], weights[j][i] = w, w
					rows[i][j], rows[j][i] = float64(w), float64(w)
				}
			}
			want, states, ok := exhaustivePairingCost(weights, 10000)
			if !ok {
				t.Fatalf("oracle budget exhausted k=%d seed=%d states=%d", k, seed, states)
			}
			m := newFixtureMatrix(t, rows)
			vertices := make([]int, k)
			for i := range vertices {
				vertices[i] = i
			}
			result, err := match.Perfect(m, vertices)
			if err != nil {
				t.Fatalf("Blossom k=%d seed=%d: %v", k, seed, err)
			}
			if result == nil {
				t.Fatal("solver returned a nil result without an error")
			}
			pairCoverage(t, vertices, result.Pairs)
			if result.Cost != float64(want) {
				t.Fatalf("wrong optimum k=%d seed=%d: got=%v want=%d pairs=%v", k, seed, result.Cost, want, result.Pairs)
			}
			independentCost := int64(0)
			for _, pair := range result.Pairs {
				independentCost += weights[pair.U][pair.V]
			}
			if independentCost != want {
				t.Fatalf("wrong selected edges k=%d seed=%d: got=%d want=%d", k, seed, independentCost, want)
			}
		}
	}
}

// TestPerfect_NestedBlossoms exposes multiple deterministic dense stress cases.
// The original internal contraction/lifting regression corpus is also moved intact.
func TestPerfect_NestedBlossoms(t *testing.T) {
	for _, seed := range []int64{2, 5, 23, 80, 86} {
		k := 8
		rng := rand.New(rand.NewSource(seed))
		weights := make([][]int64, k)
		rows := make([][]float64, k)
		for i := 0; i < k; i++ {
			weights[i] = make([]int64, k)
			rows[i] = make([]float64, k)
		}
		for i := 0; i < k; i++ {
			for j := i + 1; j < k; j++ {
				w := int64(1 + rng.Intn(5))
				weights[i][j], weights[j][i] = w, w
				rows[i][j], rows[j][i] = float64(w), float64(w)
			}
		}
		want, _, ok := exhaustivePairingCost(weights, 10000)
		if !ok {
			t.Fatalf("oracle limit seed=%d", seed)
		}
		vertices := []int{0, 1, 2, 3, 4, 5, 6, 7}
		result, err := match.Perfect(newFixtureMatrix(t, rows), vertices)
		if err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		if result == nil {
			t.Fatal("solver returned a nil result without an error")
		}
		pairCoverage(t, vertices, result.Pairs)
		if result.Cost != float64(want) {
			t.Fatalf("seed=%d got=%v want=%d", seed, result.Cost, want)
		}
	}
}

// TestOracleBudgetCannotPublishPartialMinimum guards test-only resource policy.
func TestOracleBudgetCannotPublishPartialMinimum(t *testing.T) {
	const k = 10
	weights := make([][]int64, k)
	for i := range weights {
		weights[i] = make([]int64, k)
	}
	_, states, complete := exhaustivePairingCost(weights, 3)
	if complete || states <= 3 {
		t.Fatalf("expected explicit oracle budget refusal: states=%d complete=%v", states, complete)
	}
}
