// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package match_test verifies exact matching invariance under vertex potentials.
// Every perfect matching uses each selected endpoint once, making this a strong
// optimum-level metamorphic oracle independent of Blossom's internal duals.
package match_test

import (
	"math/rand"
	"testing"

	"github.com/lvlath/go/match"
)

// TestPerfect_VertexPotentialObjectiveLaw checks the assignment-independent
// objective shift caused by adding one potential to every endpoint of an edge.
// The law applies to every perfect pairing, so the minimum must shift by the
// sum of the potentials without relying on a particular tied matching witness.
//
// Implementation:
//   - Stage 1: Construct deterministic integer-weight complete graphs at 12, 16
//     and 24 vertices, using nonmonotone original-index subset order.
//   - Stage 2: Solve the original costs and costs w'(u,v)=w(u,v)+p(u)+p(v).
//   - Stage 3: Independently check complete vertex coverage, both published
//     objectives and the predicted additive optimum shift.
//
// Behavior highlights:
//   - All fixture weights and sums are exactly representable as float64 integers.
//   - The invariant checks optimum value, not incidental tie-breaking or pair IDs.
//   - The shifted matrix is positive, symmetric, complete, and zero-diagonal.
//
// Inputs:
//   - t: test receiver; all instance sizes and PRNG seeds are fixed locally.
//
// Returns:
//   - No value; a failed comparison rejects an incorrect objective publication.
//
// Errors:
//   - Any matching error fails; no approximate or greedy fallback is accepted.
//
// Determinism:
//   - Integer PRNG draws occur in upper-triangle row-major order; selected
//     original indices are permuted by a coprime modular multiplier.
//
// Complexity:
//   - O(k^2) fixture construction and two exact matching calls per instance.
//   - Solver complexity follows the existing weighted Blossom implementation.
//
// Notes:
//   - Vertex potentials are distinct from uniform edge shifts: every edge has
//     a different additive correction, yet the optimum changes by a constant.
//   - This is a test of the exposed mathematical contract, not private dual data.
//
// AI-Hints:
//   - Never add potentials to diagonal self-costs: the input domain requires zero.
//   - Do not demand identical pair IDs when multiple optima may legitimately tie.
func TestPerfect_VertexPotentialObjectiveLaw(t *testing.T) {
	for _, k := range []int{12, 16, 24} {
		for _, seed := range []int64{5, 23, 86} {
			rng := rand.New(rand.NewSource(seed))
			base := make([][]float64, k)
			shifted := make([][]float64, k)
			potentials := make([]float64, k)
			vertices := make([]int, k)
			potentialSum := 0.0
			for i := 0; i < k; i++ {
				base[i] = make([]float64, k)
				shifted[i] = make([]float64, k)
				potentials[i] = float64(1 + (i*17)%31)
				potentialSum += potentials[i]
				vertices[i] = (i * 5) % k
			}
			for u := 0; u < k; u++ {
				for v := u + 1; v < k; v++ {
					cost := float64(100 + rng.Intn(1000))
					base[u][v], base[v][u] = cost, cost
					shift := cost + potentials[u] + potentials[v]
					shifted[u][v], shifted[v][u] = shift, shift
				}
			}
			original, err := match.Perfect(newFixtureMatrix(t, base), vertices)
			if err != nil || original == nil {
				t.Fatalf("original k=%d seed=%d: result=%+v err=%v", k, seed, original, err)
			}
			changed, err := match.Perfect(newFixtureMatrix(t, shifted), vertices)
			if err != nil || changed == nil {
				t.Fatalf("shifted k=%d seed=%d: result=%+v err=%v", k, seed, changed, err)
			}
			pairCoverage(t, vertices, original.Pairs)
			pairCoverage(t, vertices, changed.Pairs)
			if !original.Optimal || !changed.Optimal {
				t.Fatalf("Blossom must complete its internal verification")
			}
			if changed.Cost != original.Cost+potentialSum {
				t.Fatalf("vertex-potential law k=%d seed=%d: shifted %.17g, original %.17g, potential sum %.17g", k, seed, changed.Cost, original.Cost, potentialSum)
			}
			for _, result := range []*match.Result{original, changed} {
				rows := base
				if result == changed {
					rows = shifted
				}
				direct := 0.0
				for _, pair := range result.Pairs {
					direct += rows[pair.U][pair.V]
				}
				if result.Cost != direct {
					t.Fatalf("published cost disagrees with the original selected edges: %.17g != %.17g", result.Cost, direct)
				}
			}
		}
	}
}
