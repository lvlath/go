// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package mst_test

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/mst"
)

type exactMSTOracle struct {
	weight float64
	ids    []string
}

// exactStrictMSTOracle computes the exact MST by enumerating every edge subset of size |V|-1.
// It is intentionally exponential and must only be used for bounded test graphs.
//
// Implementation:
//   - Stage 1: Copy non-loop graph edges into deterministic edge order.
//   - Stage 2: Enumerate all combinations of |V|-1 edges.
//   - Stage 3: Keep only subsets that form one connected acyclic spanning tree.
//   - Stage 4: Select the minimum total finite weight.
//
// Behavior highlights:
//   - Negative finite weights are supported.
//   - Equal optimal costs are allowed; the oracle validates optimal weight, not representative identity.
//   - Disconnected inputs fail the test because strict MST is undefined.
//
// Inputs:
//   - graph: small/medium weighted undirected graph.
//
// Returns:
//   - exactMSTOracle: exact optimal weight and one optimal edge-ID witness.
//
// Determinism:
//   - Enumeration follows core.Edges() order.
//   - Equal-cost witness selection is stable by enumeration order.
//
// Complexity:
//   - Time O(C(E,V-1)·E·α(V)), Space O(V+E).
//
// AI-Hints:
//   - Do not use this oracle in benchmarks or large tests; it is a correctness oracle, not production logic.
func exactStrictMSTOracle(t *testing.T, graph *core.Graph) exactMSTOracle {
	t.Helper()

	vertices := graph.Vertices()
	if len(vertices) == 0 {
		t.Fatalf("exact oracle: empty graph has no strict MST")
	}

	candidates := make([]core.Edge, 0, graph.EdgeCount())
	for _, edge := range graph.Edges() {
		if edge.From == edge.To {
			continue
		}
		candidates = append(candidates, *edge)
	}

	targetEdges := maxTreeEdges(len(vertices))
	if targetEdges == 0 {
		return exactMSTOracle{weight: 0, ids: nil}
	}

	bestWeight := math.Inf(1)
	bestIDs := []string(nil)
	selected := make([]core.Edge, 0, targetEdges)

	var enumerate func(start int)
	enumerate = func(start int) {
		if len(selected) == targetEdges {
			weight, ok := strictTreeWeight(vertices, selected)
			if !ok {
				return
			}
			if weight < bestWeight {
				bestWeight = weight
				bestIDs = edgeIDs(selected)
			}
			return
		}

		remainingSlots := targetEdges - len(selected)
		for i := start; i <= len(candidates)-remainingSlots; i++ {
			selected = append(selected, candidates[i])
			enumerate(i + 1)
			selected = selected[:len(selected)-1]
		}
	}

	enumerate(0)

	if math.IsInf(bestWeight, 1) {
		t.Fatalf("exact oracle: no spanning tree exists for vertices=%v edges=%d", vertices, len(candidates))
	}

	return exactMSTOracle{weight: bestWeight, ids: bestIDs}
}

func strictTreeWeight(vertices []string, edges []core.Edge) (float64, bool) {
	set := newTestDisjointSet(vertices)

	var total float64
	for _, edge := range edges {
		if edge.From == edge.To {
			return 0, false
		}
		if !set.union(edge.From, edge.To) {
			return 0, false
		}
		total += edge.Weight
	}

	root := set.find(vertices[0])
	for _, vertexID := range vertices[1:] {
		if set.find(vertexID) != root {
			return 0, false
		}
	}

	return total, true
}

func edgeIDs(edges []core.Edge) []string {
	ids := make([]string, 0, len(edges))
	for _, edge := range edges {
		ids = append(ids, edge.ID)
	}
	return ids
}

func mustAlgorithmsMatchExactOracle(t *testing.T, graph *core.Graph, root string, op string) {
	t.Helper()

	oracle := exactStrictMSTOracle(t, graph)

	kruskalResult, err := mst.Kruskal(graph)
	mustNoError(t, err, op+" Kruskal")
	mustValidStrictMST(t, graph, kruskalResult, oracle.weight)

	primResult, err := mst.Prim(graph, root)
	mustNoError(t, err, op+" Prim")
	mustValidStrictMST(t, graph, primResult, oracle.weight)

	mustFloatClose(t, kruskalResult.TotalWeight, primResult.TotalWeight, op+" Kruskal vs Prim")
}

func TestOracle_HandBuiltNonTrivialGraphMatchesExactMST(t *testing.T) {
	graph := mustWeightedGraph(t)

	_, _ = graph.AddEdge("A", "B", 4)
	_, _ = graph.AddEdge("A", "C", -3)
	_, _ = graph.AddEdge("A", "D", 8)
	_, _ = graph.AddEdge("B", "C", 2)
	_, _ = graph.AddEdge("B", "E", 7)
	_, _ = graph.AddEdge("C", "D", 1)
	_, _ = graph.AddEdge("C", "E", 6)
	_, _ = graph.AddEdge("D", "E", -1)
	_, _ = graph.AddEdge("B", "D", 5)

	mustAlgorithmsMatchExactOracle(t, graph, "A", "hand-built non-trivial graph")
}

func TestOracle_CompleteGraphSevenVerticesMatchesExactMST(t *testing.T) {
	graph := mustWeightedGraph(t)

	vertices := []string{"A", "B", "C", "D", "E", "F", "G"}
	for i := 0; i < len(vertices); i++ {
		for j := i + 1; j < len(vertices); j++ {
			weight := float64(((i+3)*(j+5))%17 - 8)
			_, _ = graph.AddEdge(vertices[i], vertices[j], weight)
		}
	}

	mustAlgorithmsMatchExactOracle(t, graph, "A", "complete graph K7")
}

func TestOracle_DeterministicRandomConnectedGraphsMatchExactMST(t *testing.T) {
	for seed := int64(1); seed <= 40; seed++ {
		t.Run(fmt.Sprintf("seed_%02d", seed), func(t *testing.T) {
			graph := mustRandomConnectedGraph(t, 8, 14, seed)

			mustAlgorithmsMatchExactOracle(t, graph, "V0", "random connected graph")
		})
	}
}

func mustRandomConnectedGraph(t *testing.T, vertexCount int, targetEdges int, seed int64) *core.Graph {
	t.Helper()

	if vertexCount <= 0 {
		t.Fatalf("random graph: vertexCount must be positive, got %d", vertexCount)
	}
	if targetEdges < vertexCount-1 {
		t.Fatalf("random graph: targetEdges=%d below connectivity minimum %d", targetEdges, vertexCount-1)
	}

	graph := mustWeightedGraph(t)
	for i := 0; i < vertexCount; i++ {
		_ = graph.AddVertex(fmt.Sprintf("V%d", i))
	}

	rng := rand.New(rand.NewSource(seed))

	for i := 1; i < vertexCount; i++ {
		from := fmt.Sprintf("V%d", i-1)
		to := fmt.Sprintf("V%d", i)
		weight := randomOracleWeight(rng)

		_, _ = graph.AddEdge(from, to, weight)
	}

	for graph.EdgeCount() < targetEdges {
		fromIndex := rng.Intn(vertexCount)
		toIndex := rng.Intn(vertexCount)
		if fromIndex == toIndex {
			continue
		}

		from := fmt.Sprintf("V%d", fromIndex)
		to := fmt.Sprintf("V%d", toIndex)
		weight := randomOracleWeight(rng)

		if _, err := graph.AddEdge(from, to, weight); err != nil {
			if errors.Is(err, core.ErrMultiEdgeNotAllowed) {
				continue
			}
			t.Fatalf("random graph: AddEdge(%q,%q): %v", from, to, err)
		}
	}

	return graph
}

func randomOracleWeight(rng *rand.Rand) float64 {
	return float64(rng.Intn(41)-20) / 3
}

func TestForestOracle_DisconnectedComponentsMatchExactComponentMSTs(t *testing.T) {
	graph := mustWeightedGraph(t)

	_, _ = graph.AddEdge("A", "B", 4)
	_, _ = graph.AddEdge("A", "C", 1)
	_, _ = graph.AddEdge("B", "C", 2)

	_, _ = graph.AddEdge("D", "E", -3)
	_, _ = graph.AddEdge("E", "F", 5)
	_, _ = graph.AddEdge("D", "F", 0)

	_ = graph.AddVertex("G")

	wantComponents, wantWeight := exactForestOracle(t, graph)

	kruskalResult, err := mst.MinimumSpanningTree(graph, mst.WithForest())
	mustNoError(t, err, "Kruskal forest oracle")
	mustValidForest(t, graph, kruskalResult, wantComponents, wantWeight)

	primResult, err := mst.MinimumSpanningTree(
		graph,
		mst.WithAlgorithm(mst.AlgorithmPrim),
		mst.WithForest(),
	)
	mustNoError(t, err, "Prim forest oracle")
	mustValidForest(t, graph, primResult, wantComponents, wantWeight)
}

func exactForestOracle(t *testing.T, graph *core.Graph) (int, float64) {
	t.Helper()

	vertices := graph.Vertices()
	components := graphComponents(vertices, graph.Edges())

	var total float64
	for _, component := range components {
		if len(component) <= 1 {
			continue
		}

		subgraph := mustWeightedGraph(t)
		for _, vertexID := range component {
			_ = subgraph.AddVertex(vertexID)
		}

		componentSet := make(map[string]struct{}, len(component))
		for _, vertexID := range component {
			componentSet[vertexID] = struct{}{}
		}

		for _, edge := range graph.Edges() {
			if _, ok := componentSet[edge.From]; !ok {
				continue
			}
			if _, ok := componentSet[edge.To]; !ok {
				continue
			}
			if edge.From == edge.To {
				continue
			}
			_, _ = subgraph.AddEdge(edge.From, edge.To, edge.Weight)
		}

		total += exactStrictMSTOracle(t, subgraph).weight
	}

	return len(components), total
}

func graphComponents(vertices []string, edges []*core.Edge) [][]string {
	set := newTestDisjointSet(vertices)

	for _, edge := range edges {
		if edge.From == edge.To {
			continue
		}
		set.union(edge.From, edge.To)
	}

	groupByRoot := make(map[string][]string, len(vertices))
	for _, vertexID := range vertices {
		root := set.find(vertexID)
		groupByRoot[root] = append(groupByRoot[root], vertexID)
	}

	components := make([][]string, 0, len(groupByRoot))
	for _, component := range groupByRoot {
		components = append(components, component)
	}

	return components
}
