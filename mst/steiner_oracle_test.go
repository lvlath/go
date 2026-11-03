// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mst_test independently enumerates small original-edge subsets for Steiner OPT.
// The oracle does not call production DSU, Kruskal selection, pruning, or path reconstruction.
package mst_test

import (
	"math"
	"math/rand"
	"sort"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/mst"
)

// oracleConnectedTerminals checks whether a selected subset connects every required terminal.
// Its BFS is intentionally independent of production MST and Steiner implementation helpers.
func oracleConnectedTerminals(edges []core.Edge, terminals []string) bool {
	if len(terminals) == 0 {
		return false
	}
	neighbors := make(map[string][]string)
	for _, edge := range edges {
		neighbors[edge.From] = append(neighbors[edge.From], edge.To)
		neighbors[edge.To] = append(neighbors[edge.To], edge.From)
	}
	seen := map[string]bool{terminals[0]: true}
	queue := []string{terminals[0]}
	for head := 0; head < len(queue); head++ {
		for _, next := range neighbors[queue[head]] {
			if !seen[next] {
				seen[next] = true
				queue = append(queue, next)
			}
		}
	}
	for _, terminal := range terminals {
		if !seen[terminal] {
			return false
		}
	}
	return true
}

// oracleSteinerMinimumCost exhausts all original-edge subsets for bounded tiny graphs.
// It returns a true graph-Steiner optimum, rather than reusing the KMB approximation kernel.
func oracleSteinerMinimumCost(t *testing.T, all []core.Edge, terminals []string) float64 {
	t.Helper()
	if len(all) >= 20 {
		t.Fatal("tiny exhaustive oracle requires fewer than 20 candidate edges")
	}
	best := math.Inf(1)
	for mask := 0; mask < 1<<len(all); mask++ {
		chosen := make([]core.Edge, 0, len(all))
		cost := float64(0)
		for index, edge := range all {
			if mask&(1<<index) == 0 {
				continue
			}
			chosen = append(chosen, edge)
			cost += edge.Weight
		}
		if cost < best && oracleConnectedTerminals(chosen, terminals) {
			best = cost
		}
	}
	return best
}

// TestSteinerTree_ExactIntegerExhaustiveOracle checks KMB's mathematical bound.
// The comparison uses exact small integers and never requires KMB to equal the true OPT.
func TestSteinerTree_ExactIntegerExhaustiveOracle(t *testing.T) {
	generator := rand.New(rand.NewSource(20261103))
	vertexIDs := []string{"A", "B", "C", "D", "X"}
	for caseIndex := 0; caseIndex < 36; caseIndex++ {
		g := steinerFixtureGraph(t, core.WithMultiEdges())
		for _, id := range vertexIDs {
			if err := g.AddVertex(id); err != nil {
				t.Fatalf("case %d: AddVertex(%q): %v", caseIndex, id, err)
			}
		}
		original := make([]core.Edge, 0, 9)
		add := func(from, to string, weight float64) {
			id := "oracle-" + string(rune('a'+len(original)))
			steinerFixtureEdge(t, g, id, from, to, weight)
			original = append(original, core.Edge{ID: id, From: from, To: to, Weight: weight})
		}

		// A connected randomized base supports all terminals; additional links
		// include independent shortcuts and occasional parallel competitors.
		for index := 1; index < len(vertexIDs); index++ {
			add(vertexIDs[index], vertexIDs[generator.Intn(index)], float64(generator.Intn(7)))
		}
		for extra := 0; extra < 4; extra++ {
			left := generator.Intn(len(vertexIDs))
			right := generator.Intn(len(vertexIDs) - 1)
			if right >= left {
				right++
			}
			add(vertexIDs[left], vertexIDs[right], float64(generator.Intn(8)))
		}
		terminals := []string{"D", "A", "C"}
		optimal := oracleSteinerMinimumCost(t, original, terminals)
		if math.IsInf(optimal, 1) {
			t.Fatalf("case %d: connected construction has no oracle solution", caseIndex)
		}
		for _, mode := range []mst.SteinerPathStorage{mst.SteinerRecomputePaths, mst.SteinerRetainPaths} {
			result, err := mst.SteinerTree(g, terminals, mst.WithSteinerPathStorage(mode))
			if err != nil {
				t.Fatalf("case %d mode %d: SteinerTree: %v", caseIndex, mode, err)
			}
			if len(result.Edges) != len(result.Vertices)-1 ||
				!oracleConnectedTerminals(result.Edges, result.Vertices) {
				t.Fatalf("case %d mode %d: invalid tree: %+v", caseIndex, mode, result)
			}
			ids := make(map[string]core.Edge, len(original))
			for _, edge := range original {
				ids[edge.ID] = edge
			}
			weight := float64(0)
			for index, edge := range result.Edges {
				if originalEdge, ok := ids[edge.ID]; !ok || originalEdge != edge ||
					(index > 0 && edge.ID <= result.Edges[index-1].ID) {
					t.Fatalf("case %d mode %d: foreign or unordered original edge: %+v", caseIndex, mode, edge)
				}
				weight += edge.Weight
			}
			if weight != result.TotalWeight {
				t.Fatalf("case %d mode %d: published cost %g differs from original sum %g", caseIndex, mode, result.TotalWeight, weight)
			}
			// k=3 gives cost(KMB) <= 2*(1-1/k)*OPT, so the
			// integer-scaled comparison avoids any ratio or 0/0 handling.
			const k = 3
			if float64(k)*result.TotalWeight > float64(2*(k-1))*optimal {
				t.Fatalf("case %d mode %d: cost=%g exceeds KMB bound for OPT=%g", caseIndex, mode, result.TotalWeight, optimal)
			}
			if !sort.StringsAreSorted(result.Vertices) || !sort.StringsAreSorted(result.Terminals) {
				t.Fatalf("case %d mode %d: output order not canonical: %+v", caseIndex, mode, result)
			}
		}
	}
}
