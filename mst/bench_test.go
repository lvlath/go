// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package mst_test

import (
	"errors"
	"fmt"
	"math/rand"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/mst"
)

const (
	benchmarkGraphSeed int64 = 42

	benchmarkSparseVertices = 500
	benchmarkSparseEdges    = 2000

	benchmarkDenseVertices = 200
	benchmarkDenseEdges    = 12000

	benchmarkChainWeightLimit  = 10
	benchmarkRandomWeightLimit = 100
)

var benchmarkResult *mst.Result

// BenchmarkMST_Kruskal_SparseConnected_500v_2000e measures global edge-sort cost
// on a connected sparse-ish graph where E is much smaller than V².
//
// The fixture is built before ResetTimer and fails fast on construction errors.
func BenchmarkMST_Kruskal_SparseConnected_500v_2000e(b *testing.B) {
	graph := buildBenchmarkConnectedGraph(b, benchmarkSparseVertices, benchmarkSparseEdges)

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		result, err := mst.Kruskal(graph)
		if err != nil {
			b.Fatal(err)
		}
		benchmarkResult = result
	}
}

func BenchmarkMST_Prim_SparseConnected_500v_2000e(b *testing.B) {
	graph := buildBenchmarkConnectedGraph(b, benchmarkSparseVertices, benchmarkSparseEdges)

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		result, err := mst.Prim(graph, benchmarkRootVertexID)
		if err != nil {
			b.Fatal(err)
		}
		benchmarkResult = result
	}
}

func BenchmarkMST_Kruskal_DenseConnected_200v_12000e(b *testing.B) {
	graph := buildBenchmarkConnectedGraph(b, benchmarkDenseVertices, benchmarkDenseEdges)

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		result, err := mst.Kruskal(graph)
		if err != nil {
			b.Fatal(err)
		}
		benchmarkResult = result
	}
}

// BenchmarkMST_Prim_DenseConnected_200v_12000e measures edge-frontier heap behavior
// on a dense connected graph.
//
// This regime is intentionally allocation-sensitive: Prim stores many frontier candidates,
// so B/op and allocs/op are as important as ns/op.
func BenchmarkMST_Prim_DenseConnected_200v_12000e(b *testing.B) {
	graph := buildBenchmarkConnectedGraph(b, benchmarkDenseVertices, benchmarkDenseEdges)

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		result, err := mst.Prim(graph, benchmarkRootVertexID)
		if err != nil {
			b.Fatal(err)
		}
		benchmarkResult = result
	}
}

const benchmarkRootVertexID = "V0"

// buildBenchmarkConnectedGraph builds a deterministic connected benchmark graph.
// It first creates a chain to guarantee connectivity, then adds deterministic random edges
// until targetEdges is reached.
//
// Implementation:
//   - Stage 1: Validate benchmark dimensions.
//   - Stage 2: Build a weighted graph and add vertices V0..V(n-1).
//   - Stage 3: Add a deterministic chain so MST algorithms never benchmark setup errors.
//   - Stage 4: Add deterministic pseudo-random non-loop edges until the requested size is reached.
//
// Behavior highlights:
//   - Uses a fixed seed for reproducible benchmark topology.
//   - Duplicate-edge attempts are skipped only when core rejects multi-edges.
//   - Any unexpected construction error fails the benchmark immediately.
//
// Inputs:
//   - b: benchmark handle used for fail-fast setup.
//   - vertexCount: positive number of vertices.
//   - targetEdges: desired total edge count, at least vertexCount-1.
//
// Returns:
//   - *core.Graph: connected weighted graph ready for MST benchmarking.
//
// Determinism:
//   - Vertex IDs and pseudo-random edges are deterministic for benchmarkGraphSeed.
//
// Complexity:
//   - Setup time is outside b.ResetTimer and not measured.
//   - Setup space is O(V + E).
//
// AI-Hints:
//   - Do not move this builder inside the benchmark hot loop.
//   - Do not ignore unexpected AddEdge errors; benchmark fixtures must be trustworthy.
func buildBenchmarkConnectedGraph(b *testing.B, vertexCount int, targetEdges int) *core.Graph {
	b.Helper()

	if vertexCount <= 0 {
		b.Fatalf("vertexCount must be positive: got %d", vertexCount)
	}
	if targetEdges < vertexCount-1 {
		b.Fatalf("targetEdges=%d must be at least vertexCount-1=%d", targetEdges, vertexCount-1)
	}

	graph, err := core.NewGraph(core.WithWeighted())
	if err != nil {
		b.Fatalf("core.NewGraph(core.WithWeighted()): %v", err)
	}

	for i := 0; i < vertexCount; i++ {
		vertexID := benchmarkVertexID(i)
		if err = graph.AddVertex(vertexID); err != nil {
			b.Fatalf("Graph.AddVertex(%q): %v", vertexID, err)
		}
	}

	rng := rand.New(rand.NewSource(benchmarkGraphSeed))

	for i := 1; i < vertexCount; i++ {
		from := benchmarkVertexID(i - 1)
		to := benchmarkVertexID(i)
		weight := 1 + float64(rng.Intn(benchmarkChainWeightLimit))

		if _, err = graph.AddEdge(from, to, weight); err != nil {
			b.Fatalf("chain Graph.AddEdge(%q,%q): %v", from, to, err)
		}
	}

	for graph.EdgeCount() < targetEdges {
		fromIndex := rng.Intn(vertexCount)
		toIndex := rng.Intn(vertexCount)
		if fromIndex == toIndex {
			continue
		}

		from := benchmarkVertexID(fromIndex)
		to := benchmarkVertexID(toIndex)
		weight := 1 + float64(rng.Intn(benchmarkRandomWeightLimit))

		if _, err = graph.AddEdge(from, to, weight); err != nil {
			if errors.Is(err, core.ErrMultiEdgeNotAllowed) {
				continue
			}
			b.Fatalf("random Graph.AddEdge(%q,%q): %v", from, to, err)
		}
	}

	return graph
}

func benchmarkVertexID(index int) string {
	return fmt.Sprintf("V%d", index)
}
