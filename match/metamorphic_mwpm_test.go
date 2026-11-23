// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package match_test checks exact matching invariants under safe weight transforms.
// None of the metamorphic comparisons assumes a unique witness without proving one.
package match_test

import (
	"fmt"
	"slices"
	"testing"

	"github.com/lvlath/go/match"
)

// TestPerfect_UniformShiftAndPositiveScaling checks MWPM's fixed-cardinality law.
// A unique optimum on ten selected vertices remains unchanged by affine positive
// transforms; its exact cost changes by (k/2)*shift or the scale factor.
func TestPerfect_UniformShiftAndPositiveScaling(t *testing.T) {
	const k = 10
	base := make([][]float64, k)
	for i := range base {
		base[i] = make([]float64, k)
		for j := range base[i] {
			if i == j {
				continue
			}
			if i^1 == j {
				base[i][j] = 1
				continue
			}
			base[i][j] = float64(100 + i + j)
		}
	}
	subset := []int{9, 2, 7, 4, 5, 0, 8, 1, 6, 3}
	expected := []match.Pair{{U: 0, V: 1}, {U: 2, V: 3}, {U: 4, V: 5}, {U: 6, V: 7}, {U: 8, V: 9}}
	for _, tc := range []struct {
		name               string
		scale, shift, cost float64
	}{
		{"original", 1, 0, 5},
		{"positive_scale", 7, 0, 35},
		{"uniform_edge_shift", 1, 4, 25},
		{"combined", 7, 4, 55},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rows := make([][]float64, k)
			for i := range rows {
				rows[i] = make([]float64, k)
				for j := range rows[i] {
					if i != j {
						rows[i][j] = base[i][j]*tc.scale + tc.shift
					}
				}
			}
			result, err := match.Perfect(newFixtureMatrix(t, rows), subset)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(result.Pairs, expected) || result.Cost != tc.cost {
				t.Fatalf("affine objective law %s: %+v", tc.name, result)
			}
			checkBlossomRationalOptimum(t, rows, []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, exhaustiveMatchingStateBudget)
		})
	}
}

// TestPerfect_ContractedTriangleFamilies constructs interlocked odd-cycle groups.
// Each triple contains cheap internal links, while cross-group bridge weights
// force complete pairing decisions across the odd components of the local graph.
func TestPerfect_ContractedTriangleFamilies(t *testing.T) {
	for _, bridge := range []float64{1.5, 2, 3} {
		t.Run(fmt.Sprintf("bridge_%.1f", bridge), func(t *testing.T) {
			const k = 12
			rows := make([][]float64, k)
			for i := range rows {
				rows[i] = make([]float64, k)
				for j := range rows[i] {
					if i != j {
						rows[i][j] = 50
					}
				}
			}
			for group := 0; group < 4; group++ {
				for u := group * 3; u < group*3+3; u++ {
					for v := u + 1; v < group*3+3; v++ {
						rows[u][v], rows[v][u] = 1, 1
					}
				}
			}
			for _, edge := range [][2]int{{2, 3}, {5, 6}, {8, 9}, {11, 0}} {
				rows[edge[0]][edge[1]] = bridge
				rows[edge[1]][edge[0]] = bridge
			}
			selected := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
			checkBlossomRationalOptimum(t, rows, selected, exhaustiveMatchingStateBudget)
		})
	}
}
