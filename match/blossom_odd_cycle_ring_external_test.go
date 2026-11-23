// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package match_test checks complete matching across many coupled odd cycles.
// The construction has an analytic objective lower bound independent of Blossom.
package match_test

import (
	"slices"
	"testing"

	"github.com/lvlath/go/match"
)

// TestPerfect_OddCycleBridgeRing verifies perfect coverage and exactness at k=24.
// Eight three-vertex groups require intergroup connections because each
// individual group has odd cardinality. An eight-group cycle provides enough
// bridges to cover all unmatched group vertices without costly remote edges.
//
// Implementation:
//   - Stage 1: Construct complete nonnegative symmetric costs for eight triangles.
//   - Stage 2: Add a cycle of intergroup bridges and permute original indices.
//   - Stage 3: Compare Blossom's objective with a combinatorial lower bound.
//
// Behavior highlights:
//   - Each perfect matching needs at least four intergroup edges between odd groups.
//   - Four cheap bridges and eight unit internal pairs attain cost 4*1.5+8*1=14.
//   - Every alternative distant edge costs 100 and cannot improve this bound.
//
// Inputs:
//   - t: test handle; the 24-vertex weighted graph is fixed and deterministic.
//
// Returns:
//   - None; assertions cover cost, indices, deterministic repeatability and mode.
//
// Errors:
//   - Any solver failure fails this valid complete-graph test.
//
// Determinism:
//   - Original indices are deliberately permuted; the kernel receives that order.
//   - Repeated calls must publish the same canonical witness.
//
// Complexity:
//   - O(k^2) fixture construction plus the existing weighted Blossom engine.
//
// Notes:
//   - This test does not claim a particular blossom-shrink count or internal path.
//   - The lower bound is an independent objective proof, not a dual certificate.
//
// AI-Hints:
//   - Do not replace weighted Blossom with a global cheap-edge greedy heuristic.
//   - Do not hardcode the oracle's expected pair IDs when several minima exist.
func TestPerfect_OddCycleBridgeRing(t *testing.T) {
	const (
		groupCount     = 8
		groupSize      = 3
		vertexCount    = groupCount * groupSize
		localCost      = 1.0
		bridgeCost     = 1.5
		remoteCost     = 100.0
		expectedCost   = 14.0
		permutationMul = 5
	)
	rows := make([][]float64, vertexCount)
	for u := 0; u < vertexCount; u++ {
		rows[u] = make([]float64, vertexCount)
		for v := 0; v < vertexCount; v++ {
			if u == v {
				continue
			}
			cost := remoteCost
			if u/groupSize == v/groupSize {
				cost = localCost
			}
			rows[u][v] = cost
		}
	}
	// Each group contributes a bridge from its last vertex to the next group's
	// first vertex. The closed group cycle admits four disjoint bridges.
	for group := 0; group < groupCount; group++ {
		u := group*groupSize + groupSize - 1
		v := ((group + 1) % groupCount) * groupSize
		rows[u][v], rows[v][u] = bridgeCost, bridgeCost
	}
	ordered := make([]int, vertexCount)
	for i := range ordered {
		ordered[i] = (i * permutationMul) % vertexCount
	}
	input := newFixtureMatrix(t, rows)
	first, err := match.Perfect(input, ordered)
	if err != nil {
		t.Fatal(err)
	}
	if first == nil {
		t.Fatal("solver returned a nil result without an error")
	}
	pairCoverage(t, ordered, first.Pairs)
	if first.Algorithm != match.Blossom || !first.Optimal || first.Cost != expectedCost {
		t.Fatalf("invalid exact matching: %+v", first)
	}
	second, err := match.Perfect(input, ordered)
	if err != nil {
		t.Fatal(err)
	}
	if second == nil {
		t.Fatal("solver returned a nil result without an error")
	}
	if !slices.Equal(first.Pairs, second.Pairs) || first.Cost != second.Cost {
		t.Fatalf("nondeterministic original-index pairing: %v versus %v", first.Pairs, second.Pairs)
	}
}
