// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package match_test exercises formerly internal weighted-Blossom regression inputs.
// It tests the public matching contract using independent objective arithmetic.
package match_test

import (
	"fmt"
	"math"
	"math/big"
	"math/rand"
	"testing"

	"github.com/lvlath/go/match"
)

const exhaustiveMatchingStateBudget = 500_000

// legacySeededWeights reconstructs the deterministic integer-cent input family.
// A seed describes an actual dense numerical instance, not a mock of the solver.
// Implementation:
//   - Stage 1: Allocate an even-order square matrix with a zero diagonal.
//   - Stage 2: Use the original stable Go PRNG and upper-triangle scan order.
//   - Stage 3: Copy each sampled weight to its symmetric mirror.
//
// Behavior highlights:
//   - Weight sequence is identical to the historical Blossom regression family.
//
// Inputs:
//   - k: even matching order; seed: reproducible PRNG seed.
//
// Returns:
//   - [][]float64: detached complete symmetric costs.
//
// Errors:
//   - None for the small fixed test sizes used here.
//
// Determinism:
//   - Upper triangle is visited in increasing row, then column order.
//
// Complexity:
//   - Time O(k^2), Space O(k^2).
//
// Notes:
//   - The distribution includes near ties at cent-level precision.
//
// AI-Hints:
//   - Do not change the PRNG call count or convert to whole-number costs.
func legacySeededWeights(k int, seed int64) [][]float64 {
	rng := rand.New(rand.NewSource(seed))
	rows := make([][]float64, k)
	for i := range rows {
		rows[i] = make([]float64, k)
	}
	for row := 0; row < k; row++ {
		for col := row + 1; col < k; col++ {
			weight := 1 + float64(rng.Intn(10_000))/100
			rows[row][col], rows[col][row] = weight, weight
		}
	}
	return rows
}

// legacyNearTieWeights reconstructs the original tight-slack numeric regression family.
// It preserves the historical PRNG draws and float64 values for precise replay.
// Implementation:
//   - Stage 1: Fill every off-diagonal symmetric weight.
//   - Stage 2: Combine an integral base with a micro-scale fractional perturbation.
//
// Behavior highlights:
//   - Near-tied edges stress dual selection and contraction/lifting.
//
// Inputs:
//   - k: even matching order; seed: deterministic PRNG seed.
//
// Returns:
//   - [][]float64: detached symmetric weighted instance.
//
// Errors:
//   - None for fixed small fixture sizes.
//
// Determinism:
//   - Upper-triangle row-major PRNG sampling.
//
// Complexity:
//   - Time O(k^2), Space O(k^2).
//
// Notes:
//   - An exact rational oracle distinguishes genuinely different float64 inputs.
//
// AI-Hints:
//   - Tolerance comparisons must not silently redefine the objective.
func legacyNearTieWeights(k int, seed int64) [][]float64 {
	rng := rand.New(rand.NewSource(seed))
	rows := make([][]float64, k)
	for i := range rows {
		rows[i] = make([]float64, k)
	}
	for row := 0; row < k; row++ {
		for col := row + 1; col < k; col++ {
			weight := 1000 + float64(rng.Intn(5)) + rng.Float64()*1e-6
			rows[row][col], rows[col][row] = weight, weight
		}
	}
	return rows
}

// legacyWideRangeWeights reconstructs wide-magnitude dense Blossom inputs.
// Every edge remains a finite nonnegative original cost, never a profit transform.
// Implementation:
//   - Stage 1: Sample a power-of-ten scale for each upper-triangle edge.
//   - Stage 2: Sample a floating multiplier and copy to the symmetric cell.
//
// Behavior highlights:
//   - Replays the input order and arithmetic of existing scale regressions.
//
// Inputs:
//   - k: even matching order; seed: deterministic PRNG seed.
//
// Returns:
//   - [][]float64: detached complete symmetric matrix.
//
// Errors:
//   - None for the accepted fixed test sizes.
//
// Determinism:
//   - Exact historical upper-triangle PRNG schedule.
//
// Complexity:
//   - Time O(k^2), Space O(k^2).
//
// Notes:
//   - These tests probe conditioning rather than an arbitrary large-value threshold.
//
// AI-Hints:
//   - Never normalize all costs in production to satisfy test magnitudes.
func legacyWideRangeWeights(k int, seed int64) [][]float64 {
	rng := rand.New(rand.NewSource(seed))
	rows := make([][]float64, k)
	for i := range rows {
		rows[i] = make([]float64, k)
	}
	for row := 0; row < k; row++ {
		for col := row + 1; col < k; col++ {
			scale := math.Pow10(rng.Intn(6))
			weight := scale * (1 + rng.Float64()*1000)
			rows[row][col], rows[col][row] = weight, weight
		}
	}
	return rows
}

// exhaustiveRationalMinimum computes the exact optimum of the represented float64 costs.
// Its search is independent of production mate validation, duals and output rounding.
// Implementation:
//   - Stage 1: Check even order and the explicit recursion budget.
//   - Stage 2: Pair the first unmatched vertex with each remaining vertex in order.
//   - Stage 3: Compare completed sums with exact big.Rat arithmetic.
//
// Behavior highlights:
//   - Exceeding the work budget refuses to publish an unproven optimum.
//
// Inputs:
//   - rows: small complete square matrix; budget: maximum visited recursion nodes.
//
// Returns:
//   - *big.Rat: independent minimum, int: visited nodes, bool: complete search.
//
// Errors:
//   - None; an over-budget search returns complete=false and a nil result.
//
// Determinism:
//   - First-unmatched row, then ascending partner order.
//
// Complexity:
//   - Time exponential in k, bounded to k<=14; Space O(k^2 + recursion).
//
// Notes:
//   - This oracle must not become an alternative production matching kernel.
//
// AI-Hints:
//   - Rational comparison is required for close float64 weights, not decimal strings.
func exhaustiveRationalMinimum(rows [][]float64, budget int) (*big.Rat, int, bool) {
	k := len(rows)
	if k > 14 || k&1 != 0 || budget <= 0 {
		return nil, 0, false
	}
	for _, row := range rows {
		if len(row) != k {
			return nil, 0, false
		}
	}
	used := make([]bool, k)
	weights := make([][]*big.Rat, k)
	for i := range weights {
		weights[i] = make([]*big.Rat, k)
		for j := range weights[i] {
			weights[i][j] = new(big.Rat).SetFloat64(rows[i][j])
			if weights[i][j] == nil {
				return nil, 0, false
			}
		}
	}
	var best *big.Rat
	states, complete := 0, true
	var search func(sum *big.Rat)
	search = func(sum *big.Rat) {
		if !complete {
			return
		}
		states++
		if states > budget {
			complete = false
			return
		}
		first := -1
		for v, selected := range used {
			if !selected {
				first = v
				break
			}
		}
		if first == -1 {
			if best == nil || sum.Cmp(best) < 0 {
				best = new(big.Rat).Set(sum)
			}
			return
		}
		used[first] = true
		for partner := first + 1; partner < k; partner++ {
			if used[partner] {
				continue
			}
			used[partner] = true
			search(new(big.Rat).Add(sum, weights[first][partner]))
			used[partner] = false
		}
		used[first] = false
	}
	search(new(big.Rat))
	if !complete {
		return nil, states, false
	}
	return best, states, true
}

// independentPairCost adds selected public edges using exact input-float64 rational values.
// It checks the mathematical witness without invoking production verification helpers.
// Implementation:
//   - Stage 1: Start at exactly zero.
//   - Stage 2: Add source costs in returned pair order, without rounding.
//
// Behavior highlights:
//   - Original indices are read directly from the immutable test fixture.
//
// Inputs:
//   - rows: full original matrix; pairs: validated canonical matching witnesses.
//
// Returns:
//   - *big.Rat: mathematically exact represented-input sum.
//
// Errors:
//   - None for the structurally validated cases in these tests.
//
// Determinism:
//   - Public pair-slice order.
//
// Complexity:
//   - Time O(k), Space O(k) for big-number storage.
//
// Notes:
//   - This independent calculation must not use result.Cost as its input.
//
// AI-Hints:
//   - Do not confuse transformed edge profit with original source cost.
func independentPairCost(rows [][]float64, pairs []match.Pair) *big.Rat {
	sum := new(big.Rat)
	for _, pair := range pairs {
		sum.Add(sum, new(big.Rat).SetFloat64(rows[pair.U][pair.V]))
	}
	return sum
}

// checkBlossomRationalOptimum compares the public witness with the bounded independent oracle.
// It defends the mathematical objective and the distinct float64 publication contract.
// Implementation:
//   - Stage 1: Solve the full matrix with the canonical public facade.
//   - Stage 2: Check perfect original-index coverage and independent rational objective.
//   - Stage 3: Compare raw published float64 cost with the exact result's float64 representation.
//
// Behavior highlights:
//   - No alternative matching solver is called by production.
//
// Inputs:
//   - t: test; rows: input; vertices: original index order; budget: oracle work.
//
// Returns:
//   - None; assertions report the exact objective mismatch.
//
// Errors:
//   - Test failure for incomplete oracle, solver failure or wrong objective.
//
// Determinism:
//   - The solver's output order is checked separately by pairCoverage.
//
// Complexity:
//   - Oracle time exponential in k; test-only budget limits work.
//
// Notes:
//   - Floating summation may differ from exact rational conversion by a few ULP.
//
// AI-Hints:
//   - Never hide a real non-optimal witness by comparing only rounded output cost.
func checkBlossomRationalOptimum(t *testing.T, rows [][]float64, vertices []int, budget int) {
	t.Helper()
	best, states, ok := exhaustiveRationalMinimum(rows, budget)
	if !ok {
		t.Fatalf("independent oracle refused k=%d states=%d budget=%d", len(rows), states, budget)
	}
	result, err := match.Perfect(newFixtureMatrix(t, rows), vertices)
	if err != nil {
		t.Fatalf("Blossom solve: %v", err)
	}
	if result == nil {
		t.Fatal("solver returned a nil result without an error")
	}
	pairCoverage(t, vertices, result.Pairs)
	witnessCost := independentPairCost(rows, result.Pairs)
	if witnessCost.Cmp(best) != 0 {
		t.Fatalf("suboptimal matching: pairs=%v witness=%s oracle=%s", result.Pairs, witnessCost.RatString(), best.RatString())
	}
	want, _ := best.Float64()
	if math.Abs(result.Cost-want) > 2e-12*math.Max(1, math.Abs(want)) {
		t.Fatalf("raw float64 sum drift: got=%.17g expected=%.17g", result.Cost, want)
	}
}

// TestBlossomHistoricalRegressionSeeds replays the known contraction and lifting inputs.
// For k<=14 it checks objective optimality with an independent exact rational oracle.
func TestBlossomHistoricalRegressionSeeds(t *testing.T) {
	cases := []struct {
		k    int
		seed int64
	}{
		{6, 23}, {8, 5}, {8, 2}, {8, 6}, {8, 11},
		{10, 14}, {10, 18}, {12, 6}, {12, 8}, {12, 9}, {12, 19}, {12, 20},
		{14, 1}, {14, 3}, {14, 9}, {14, 14}, {14, 23}, {14, 25},
	}
	for _, tc := range cases {
		t.Run(fmt.Sprintf("k%d/seed%d", tc.k, tc.seed), func(t *testing.T) {
			rows := legacySeededWeights(tc.k, tc.seed)
			selected := make([]int, tc.k)
			for i := range selected {
				selected[i] = i
			}
			checkBlossomRationalOptimum(t, rows, selected, exhaustiveMatchingStateBudget)
		})
	}
}

// TestBlossomNearTieNumerics guards dual movement under micro-scale edge differences.
// Exact rational input-cost comparison avoids mistaking rounded equality for optimality.
func TestBlossomNearTieNumerics(t *testing.T) {
	for _, seed := range []int64{80, 86} {
		t.Run(fmt.Sprintf("seed%d", seed), func(t *testing.T) {
			rows := legacyNearTieWeights(12, seed)
			selected := make([]int, 12)
			for i := range selected {
				selected[i] = i
			}
			checkBlossomRationalOptimum(t, rows, selected, exhaustiveMatchingStateBudget)
		})
	}
}

// TestBlossomWideRangeOriginalCosts checks float magnitudes without profit rounding.
// Bounded exact-rational comparison tests which edges are selected, not just sum precision.
func TestBlossomWideRangeOriginalCosts(t *testing.T) {
	for _, seed := range []int64{7, 42, 1001, 9001, 17017, 424242} {
		t.Run(fmt.Sprintf("seed%d", seed), func(t *testing.T) {
			rows := legacyWideRangeWeights(12, seed)
			selected := make([]int, 12)
			for i := range selected {
				selected[i] = i
			}
			checkBlossomRationalOptimum(t, rows, selected, exhaustiveMatchingStateBudget)
		})
	}
}

// TestBlossomEqualWeightDeterminism checks stable witness order across independent solves.
// Multiple distinct optimal matchings are allowed, but one policy must be repeatable.
func TestBlossomEqualWeightDeterminism(t *testing.T) {
	for _, k := range []int{4, 8, 16, 32, 64} {
		t.Run(fmt.Sprintf("k%d", k), func(t *testing.T) {
			rows := make([][]float64, k)
			for i := range rows {
				rows[i] = make([]float64, k)
				for j := range rows[i] {
					if i != j {
						rows[i][j] = 1
					}
				}
			}
			m := newFixtureMatrix(t, rows)
			selected := make([]int, k)
			for i := range selected {
				selected[i] = i
			}
			first, err := match.Perfect(m, selected)
			if err != nil {
				t.Fatal(err)
			}
			for attempt := 0; attempt < 4; attempt++ {
				next, err := match.Perfect(m, selected)
				if err != nil {
					t.Fatalf("attempt %d: %v", attempt, err)
				}
				pairCoverage(t, selected, next.Pairs)
				mustPairsEqual(t, next.Pairs, first.Pairs)
				if next.Cost != first.Cost || next.Cost != float64(k/2) {
					t.Fatalf("unexpected repeated result: %+v vs %+v", next, first)
				}
			}
		})
	}
}

// TestBlossomDenseLargeInputCompletes covers deterministic medium-size solves.
// It checks structure and output detachment without an exponential reference oracle.
func TestBlossomDenseLargeInputCompletes(t *testing.T) {
	for _, k := range []int{64, 96, 128, 192, 256} {
		t.Run(fmt.Sprintf("k%d", k), func(t *testing.T) {
			rows := legacySeededWeights(k, int64(k*100))
			selected := make([]int, k)
			for i := range selected {
				selected[i] = i
			}
			m := newFixtureMatrix(t, rows)
			result, err := match.Perfect(m, selected)
			if err != nil {
				t.Fatalf("solve k%d: %v", k, err)
			}
			pairCoverage(t, selected, result.Pairs)
			if result.Cost <= 0 || result.Cost > float64(k/2)*101 {
				t.Fatalf("invalid cost k%d: %.17g", k, result.Cost)
			}
		})
	}
}
