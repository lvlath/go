// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package match_test fuzzes the actual public numerical solver against an independent oracle.
// Every seed uses a fixed input generator, bounded cardinality, and an explicit search budget.
package match_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/lvlath/go/match"
)

// FuzzPerfectExactObjective checks small dense instances with an exact rational oracle.
// Its seed corpus covers integer-cent, nearly tied, constant and wide-scale weights;
// arbitrary fuzz values cannot trigger exponential unbounded oracle work.
func FuzzPerfectExactObjective(f *testing.F) {
	for _, seed := range []int64{1, 2, 3, 5, 8, 13, 21, 34, 55, 80, 86, 1001} {
		for _, k := range []int{4, 6, 8, 10} {
			for mode := 0; mode < 4; mode++ {
				f.Add(seed, k, mode)
			}
		}
	}
	f.Fuzz(func(t *testing.T, seed int64, size, mode int) {
		// Normalize into the bounded mathematical oracle domain.
		k := (size%5 + 5) % 5
		k = 2*k + 2
		mode = (mode%4 + 4) % 4
		var rows [][]float64
		switch mode {
		case 0:
			rows = legacySeededWeights(k, seed)
		case 1:
			rows = legacyNearTieWeights(k, seed)
		case 2:
			rows = legacyWideRangeWeights(k, seed)
		default:
			rows = make([][]float64, k)
			for i := range rows {
				rows[i] = make([]float64, k)
				for j := range rows[i] {
					if i != j {
						rows[i][j] = 7
					}
				}
			}
		}
		vertices := make([]int, k)
		for i := range vertices {
			vertices[i] = i
		}
		checkBlossomRationalOptimum(t, rows, vertices, exhaustiveMatchingStateBudget)
	})
}

// FuzzPerfectSubsetPermutation tests original-index mapping without assuming tie equity.
// The induced k*k exact objective is compared to the public result in n*k context.
func FuzzPerfectSubsetPermutation(f *testing.F) {
	for _, seed := range []int64{1, 2, 3, 8, 13, 21, 86, 1001} {
		for _, k := range []int{4, 6, 8, 10} {
			f.Add(seed, k)
		}
	}
	f.Fuzz(func(t *testing.T, seed int64, rawSize int) {
		k := 2 * ((rawSize%4+4)%4 + 2)
		n := k + 4
		full := legacySeededWeights(n, seed)
		rng := rand.New(rand.NewSource(seed ^ 0x62B2))
		perm := rng.Perm(n)
		selected := append([]int(nil), perm[:k]...)
		induced := make([][]float64, k)
		for i, u := range selected {
			induced[i] = make([]float64, k)
			for j, v := range selected {
				induced[i][j] = full[u][v]
			}
		}
		best, _, complete := exhaustiveRationalMinimum(induced, exhaustiveMatchingStateBudget)
		if !complete {
			t.Fatal("bounded independent oracle failed")
		}
		result, err := match.Perfect(newFixtureMatrix(t, full), selected)
		if err != nil {
			t.Fatalf("Blossom subset: %v", err)
		}
		pairCoverage(t, selected, result.Pairs)
		got := independentPairCost(full, result.Pairs)
		if got.Cmp(best) != 0 {
			t.Fatalf("incorrect induced optimum k=%d seed=%d pairs=%v actual=%v best=%v", k, seed, result.Pairs, got, best)
		}
		want, _ := best.Float64()
		if math.Abs(result.Cost-want) > 2e-12*math.Max(1, math.Abs(want)) {
			t.Fatalf("raw subset cost drift: %.17g versus %.17g", result.Cost, want)
		}
	})
}
