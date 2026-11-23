// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package match_test exercises difficult weighted general-graph matchings.
// The externally observable checks protect deterministic complete pairings and
// the original-cost objective without requiring private Blossom instrumentation.
package match_test

import (
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/lvlath/go/match"
)

// challengingCompleteMatchingWeights constructs repeatable weighted stress graphs.
// Integer, near-tie and geometric modes expose distinct patterns of tight-edge
// events without altering the public solver's numerical or ordering contracts.
//
// Implementation:
//   - Stage 1: Draw an explicit, fixed set of points from the local seed.
//   - Stage 2: Generate upper-triangle costs according to the selected family.
//   - Stage 3: Mirror every finite edge and retain an exactly zero diagonal.
//
// Behavior highlights:
//   - Values model dense general-graph costs, not a bipartite-only subproblem.
//   - Geometric costs use Euclidean distance but no production metric closure.
//
// Inputs:
//   - k: even graph order; seed: deterministic generator seed.
//   - mode: 0 for varied integral costs, 1 for weighted near ties, 2 for Euclidean.
//
// Returns:
//   - [][]float64: detached symmetric finite complete source matrix.
//
// Errors:
//   - None for the fixed small positive sizes used by these external tests.
//
// Determinism:
//   - Point sampling and upper-triangle generation occur in fixed PRNG order.
//
// Complexity:
//   - Time O(k^2), Space O(k^2).
//
// Notes:
//   - The generator itself never invokes a production matching helper.
//
// AI-Hints:
//   - Keep both mirrored values bitwise identical to isolate Blossom behavior.
//   - The exact original-index matrix is the only source of published cost.
func challengingCompleteMatchingWeights(k int, seed int64, mode int) [][]float64 {
	rng := rand.New(rand.NewSource(seed))
	rows := make([][]float64, k)
	points := make([][2]float64, k)
	for vertex := 0; vertex < k; vertex++ {
		rows[vertex] = make([]float64, k)
		points[vertex] = [2]float64{rng.Float64() * 100, rng.Float64() * 100}
	}
	for u := 0; u < k; u++ {
		for v := u + 1; v < k; v++ {
			var weight float64
			switch mode {
			case 0:
				weight = float64(1 + rng.Intn(500))
			case 1:
				weight = 1000 + float64(rng.Intn(3)) + rng.Float64()*1e-6
			default:
				weight = math.Hypot(points[u][0]-points[v][0], points[u][1]-points[v][1])
			}
			rows[u][v], rows[v][u] = weight, weight
		}
	}
	return rows
}

// TestPerfect_DifficultWeightedBlossomFamilies checks the exact public solver
// against deterministic integer, near-degenerate and Euclidean instances.
// These inputs are selected to exercise both contraction and reopening of
// weighted odd-cycle structures through the original numerical implementation.
//
// Implementation:
//   - Stage 1: Build each complete weighted graph from fixed generator inputs.
//   - Stage 2: Invoke public Perfect twice and verify every original endpoint.
//   - Stage 3: Recompute the source-weight objective independently and, for the
//     bounded 12-vertex case, compare against the exact rational oracle.
//
// Behavior highlights:
//   - Tests the genuine weighted Blossom kernel rather than a greedy substitute.
//   - Does not assume that distinct valid optimum witnesses are ordered identically
//     by a different solver: only this package's repeated-call determinism is checked.
//
// Inputs:
//   - t: test runner; dimensions and seeds define reproducible stress instances.
//
// Returns:
//   - None; failures report the instance identity and its published witness.
//
// Errors:
//   - Solver errors or incomplete witnesses fail the corresponding test case.
//
// Determinism:
//   - Same ordered input and options must reproduce canonical pairs and raw cost.
//
// Complexity:
//   - O(k^2) preparation plus two Blossom solves per test instance.
//   - The k=12 oracle uses bounded exponential test-only enumeration.
//
// Notes:
//   - Large cases check a necessary objective invariant, not a separate proof
//     of global optimality; the smaller independent oracle checks optimality.
//
// AI-Hints:
//   - Avoid hard-coded private shrink counts in external contract tests.
//   - Do not replace original edge sums with transformed Blossom profit values.
func TestPerfect_DifficultWeightedBlossomFamilies(t *testing.T) {
	cases := []struct {
		name string
		k    int
		seed int64
		mode int
	}{
		{"multiple_small_odd_cycles", 12, 48, 0},
		{"weighted_nested_cycles", 18, 45, 0},
		{"nearly_degenerate_duals", 24, 31, 1},
		{"dense_32_vertex_instance", 32, 32, 0},
		{"geometric_dense_general_graph", 32, 70, 2},
		{"near_tie_reopening", 32, 294, 1},
		{"large_dense_integer_case", 64, 392, 0},
		{"geometric_nested_cycles", 64, 184, 2},
		{"geometric_many_odd_cycles", 64, 421, 2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rows := challengingCompleteMatchingWeights(tc.k, tc.seed, tc.mode)
			vertices := make([]int, tc.k)
			for i := range vertices {
				vertices[i] = i
			}
			input := newFixtureMatrix(t, rows)
			first, err := match.Perfect(input, vertices)
			if err != nil || first == nil {
				t.Fatalf("first exact solve: result=%+v err=%v", first, err)
			}
			second, err := match.Perfect(input, vertices)
			if err != nil || second == nil {
				t.Fatalf("repeat exact solve: result=%+v err=%v", second, err)
			}
			pairCoverage(t, vertices, first.Pairs)
			pairCoverage(t, vertices, second.Pairs)
			if !first.Optimal || !second.Optimal || first.Algorithm != match.Blossom {
				t.Fatalf("unexpected algorithm identity and proof status: %+v", first)
			}
			if first.Cost != second.Cost || !slices.Equal(first.Pairs, second.Pairs) {
				t.Fatalf("nondeterministic exact matching: %+v versus %+v", first, second)
			}
			objective := 0.0
			for _, pair := range first.Pairs {
				objective += rows[pair.U][pair.V]
			}
			if objective != first.Cost {
				t.Fatalf("cost from selected original weights %.17g differs from returned %.17g", objective, first.Cost)
			}
			if tc.k == 12 {
				minimum, _, complete := exhaustiveRationalMinimum(rows, exhaustiveMatchingStateBudget)
				if !complete || independentPairCost(rows, first.Pairs).Cmp(minimum) != 0 {
					t.Fatalf("independent rational optimum differs: pairs=%v minimum=%v complete=%v", first.Pairs, minimum, complete)
				}
			}
		})
	}
}
