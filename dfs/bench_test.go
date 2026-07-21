// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package dfs_test

import (
	"fmt"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/dfs"
)

const (
	// benchmarkChainVertexCount is the number of vertices used in the long-chain DFS benchmark.
	benchmarkChainVertexCount = 10001

	// benchmarkTopoLayerCount controls the number of layers in the layered DAG benchmark.
	benchmarkTopoLayerCount = 80

	// benchmarkTopoLayerWidth controls the width of each DAG layer in the layered DAG benchmark.
	benchmarkTopoLayerWidth = 16

	// benchmarkCycleSparseVertexCount controls the sparse-cycle benchmark graph size.
	benchmarkCycleSparseVertexCount = 4000

	// benchmarkCycleDenseCliqueSize controls the dense-but-safe cycle benchmark graph size.
	// Keep this small enough to avoid uncontrolled witness explosion.
	benchmarkCycleDenseCliqueSize = 7
)

// BenchmarkDFS_Chain10000 measures the successful hot path of DFS on a long directed chain.
//
// Implementation:
//   - Stage 1: Build a valid directed chain graph once, outside the timed loop.
//   - Stage 2: Reset the timer and report allocations.
//   - Stage 3: Repeatedly run DFS from the first vertex and fail on any unexpected error.
//
// Behavior highlights:
//   - Measures DFS traversal only; graph construction is excluded.
//   - Uses a deterministic graph and a guaranteed-valid start vertex.
//
// Inputs:
//   - N/A.
//
// Returns:
//   - N/A.
//
// Errors:
//   - Fatal benchmark failure if graph setup or traversal fails.
//
// Determinism:
//   - Deterministic for the same benchmark constants.
//
// Complexity:
//   - Setup time O(V).
//   - Each measured DFS run is O(V+E), which is O(V) for a chain.
//
// Notes:
//   - This benchmark intentionally measures the successful hot path only.
//
// AI-Hints:
//   - Benchmark only the successful hot path.
//   - Keep graph construction and validation outside timed loops.
//   - Report allocations for every core benchmark.
func BenchmarkDFS_Chain10000(b *testing.B) {
	graph := buildBenchmarkDirectedChain(b, benchmarkChainVertexCount)
	startID := "N0"

	b.ReportAllocs()
	b.ResetTimer()

	for index := 0; index < b.N; index++ {
		result, err := dfs.DFS(graph, startID)
		if err != nil {
			b.Fatalf("DFS failed: %v", err)
		}
		if result == nil {
			b.Fatal("DFS returned nil result on a valid hot path")
		}
	}
}

// BenchmarkDFS_FullTraversal_Forest measures DFS forest traversal over multiple disconnected components.
//
// Implementation:
//   - Stage 1: Build a deterministic graph with several disconnected directed chains.
//   - Stage 2: Reset the timer and report allocations.
//   - Stage 3: Repeatedly run full-traversal DFS and fail on any unexpected error.
//
// Behavior highlights:
//   - Measures forest traversal semantics rather than a single-tree run.
//   - Uses WithFullTraversal on a graph with guaranteed disconnected structure.
//
// Inputs:
//   - N/A.
//
// Returns:
//   - N/A.
//
// Errors:
//   - Fatal benchmark failure if graph setup or traversal fails.
//
// Determinism:
//   - Deterministic for the same benchmark constants and graph construction order.
//
// Complexity:
//   - Setup time O(V+E).
//   - Each measured run is O(V+E).
//
// Notes:
//   - This benchmark complements the single-chain benchmark by exercising forest-root restart logic.
//
// AI-Hints:
//   - Benchmark configuration-dependent code paths explicitly when they affect traversal shape.
func BenchmarkDFS_FullTraversal_Forest(b *testing.B) {
	graph := buildBenchmarkForestGraph(b, 32, 256)

	b.ReportAllocs()
	b.ResetTimer()

	for index := 0; index < b.N; index++ {
		result, err := dfs.DFS(graph, "", dfs.WithFullTraversal())
		if err != nil {
			b.Fatalf("DFS full traversal failed: %v", err)
		}
		if result == nil {
			b.Fatal("DFS full traversal returned nil result on a valid hot path")
		}
	}
}

// BenchmarkTopologicalSort_LayeredDAG measures topological sorting on a deterministic layered DAG.
//
// Implementation:
//   - Stage 1: Build a valid directed acyclic layered graph once.
//   - Stage 2: Reset the timer and report allocations.
//   - Stage 3: Repeatedly run TopologicalSort and fail on any unexpected error.
//
// Behavior highlights:
//   - Measures the successful hot path of topological sorting.
//   - Uses a graph shape with many edges but guaranteed acyclicity.
//
// Inputs:
//   - N/A.
//
// Returns:
//   - N/A.
//
// Errors:
//   - Fatal benchmark failure if graph setup or sorting fails.
//
// Determinism:
//   - Deterministic for the same benchmark constants.
//
// Complexity:
//   - Setup time O(V+E).
//   - Each measured run is O(V+E).
//
// Notes:
//   - Layered DAGs are a useful middle ground between trivial chains and dense arbitrary DAGs.
//
// AI-Hints:
//   - Do not accidentally benchmark a cycle-detection error path when measuring topo hot paths.
func BenchmarkTopologicalSort_LayeredDAG(b *testing.B) {
	graph := buildBenchmarkLayeredDAG(b, benchmarkTopoLayerCount, benchmarkTopoLayerWidth)

	b.ReportAllocs()
	b.ResetTimer()

	for index := 0; index < b.N; index++ {
		order, err := dfs.TopologicalSort(graph)
		if err != nil {
			b.Fatalf("TopologicalSort failed: %v", err)
		}
		if order == nil && graph.VertexCount() != 0 {
			b.Fatal("TopologicalSort returned nil order on a valid hot path")
		}
	}
}

// BenchmarkDetectCycles_SparseSingleWitness measures cycle detection on a large sparse graph
// containing one controlled witness cycle.
//
// Implementation:
//   - Stage 1: Build a sparse directed chain and close exactly one cycle near the tail.
//   - Stage 2: Reset the timer and report allocations.
//   - Stage 3: Repeatedly run DetectCycles and fail on any unexpected error.
//
// Behavior highlights:
//   - Keeps cycle count controlled while still exercising witness detection.
//   - Avoids dense-cycle explosion that would distort the benchmark into output-growth cost.
//
// Inputs:
//   - N/A.
//
// Returns:
//   - N/A.
//
// Errors:
//   - Fatal benchmark failure if graph setup or detection fails.
//
// Determinism:
//   - Deterministic for the same benchmark constants.
//
// Complexity:
//   - Setup time O(V).
//   - Each measured run is O(V+E+W·L) with W intentionally kept near 1.
//
// Notes:
//   - This is the preferred baseline DetectCycles benchmark because it exercises the algorithm
//     without producing an uncontrollable number of witness cycles.
//
// AI-Hints:
//   - Keep cycle benchmark inputs controlled; dense graphs can benchmark witness explosion instead of traversal.
func BenchmarkDetectCycles_SparseSingleWitness(b *testing.B) {
	graph := buildBenchmarkSparseCycleGraph(b, benchmarkCycleSparseVertexCount)

	b.ReportAllocs()
	b.ResetTimer()

	for index := 0; index < b.N; index++ {
		result, err := dfs.DetectCycles(graph)
		if err != nil {
			b.Fatalf("DetectCycles failed: %v", err)
		}
		if result == nil {
			b.Fatal("DetectCycles returned nil result on a valid hot path")
		}
	}
}

// BenchmarkDetectCycles_DenseControlled measures cycle detection on a small dense directed clique.
//
// Implementation:
//   - Stage 1: Build a small complete directed graph once.
//   - Stage 2: Reset the timer and report allocations.
//   - Stage 3: Repeatedly run DetectCycles and fail on any unexpected error.
//
// Behavior highlights:
//   - Exercises a dense-cycle regime deliberately.
//   - The graph size is intentionally capped to avoid an uncontrolled combinatorial blow-up.
//
// Inputs:
//   - N/A.
//
// Returns:
//   - N/A.
//
// Errors:
//   - Fatal benchmark failure if graph setup or detection fails.
//
// Determinism:
//   - Deterministic for the same benchmark constants.
//
// Complexity:
//   - Setup time O(V^2).
//   - Measured complexity depends on witness reconstruction volume, which is intentionally bounded here.
//
// Notes:
//   - Keep the clique size conservative.
//   - This benchmark is for dense-regime comparison, not for maximum-scale throughput claims.
//
// AI-Hints:
//   - Dense cycle benchmarks must be size-capped deliberately to stay meaningful and reproducible.
func BenchmarkDetectCycles_DenseControlled(b *testing.B) {
	graph := buildBenchmarkDenseCycleGraph(b, benchmarkCycleDenseCliqueSize)

	b.ReportAllocs()
	b.ResetTimer()

	for index := 0; index < b.N; index++ {
		result, err := dfs.DetectCycles(graph)
		if err != nil {
			b.Fatalf("DetectCycles dense benchmark failed: %v", err)
		}
		if result == nil {
			b.Fatal("DetectCycles returned nil result on a valid dense hot path")
		}
	}
}

// buildBenchmarkDirectedChain constructs a deterministic directed chain graph for benchmarks.
//
// Implementation:
//   - Stage 1: Validate the requested vertex count.
//   - Stage 2: Build the graph once, failing immediately on any setup error.
//   - Stage 3: Return the fully constructed graph.
//
// Behavior highlights:
//   - Setup failures abort the benchmark before timing begins.
//
// Inputs:
//   - vertexCount: number of vertices in the chain.
//
// Returns:
//   - *core.Graph: fully built directed chain graph.
//
// Errors:
//   - Fatal benchmark failure on invalid size or graph-construction failure.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(V), Space O(V).
//
// Notes:
//   - The resulting graph contains vertexCount-1 edges.
//
// AI-Hints:
//   - Never ignore setup failures in benchmarks for core algorithms.
func buildBenchmarkDirectedChain(b *testing.B, vertexCount int) *core.Graph {
	b.Helper()

	if vertexCount < 2 {
		b.Fatalf("invalid chain vertex count: got=%d want>=2", vertexCount)
	}

	graph, _ := core.NewGraph(core.WithDirected(true))

	for index := 0; index < vertexCount-1; index++ {
		fromID := fmt.Sprintf("N%d", index)
		toID := fmt.Sprintf("N%d", index+1)

		if err := graph.AddVertex(fromID); err != nil {
			b.Fatalf("AddVertex(%q) failed: %v", fromID, err)
		}
		if err := graph.AddVertex(toID); err != nil {
			b.Fatalf("AddVertex(%q) failed: %v", toID, err)
		}
		if _, err := graph.AddEdge(fromID, toID, 0); err != nil {
			b.Fatalf("AddEdge(%q,%q) failed: %v", fromID, toID, err)
		}
	}

	return graph
}

// buildBenchmarkForestGraph constructs a deterministic forest-shaped directed graph
// consisting of multiple disconnected chains.
//
// AI-Hints:
//   - This builder exists to benchmark full-traversal restart logic explicitly.
func buildBenchmarkForestGraph(b *testing.B, componentCount, componentLength int) *core.Graph {
	b.Helper()

	if componentCount < 1 {
		b.Fatalf("invalid component count: got=%d want>=1", componentCount)
	}
	if componentLength < 1 {
		b.Fatalf("invalid component length: got=%d want>=1", componentLength)
	}

	graph, _ := core.NewGraph(core.WithDirected(true))

	for componentIndex := 0; componentIndex < componentCount; componentIndex++ {
		for localIndex := 0; localIndex < componentLength; localIndex++ {
			vertexID := fmt.Sprintf("C%03d-N%03d", componentIndex, localIndex)
			if err := graph.AddVertex(vertexID); err != nil {
				b.Fatalf("AddVertex(%q) failed: %v", vertexID, err)
			}

			if localIndex == 0 {
				continue
			}

			parentID := fmt.Sprintf("C%03d-N%03d", componentIndex, localIndex-1)
			if _, err := graph.AddEdge(parentID, vertexID, 0); err != nil {
				b.Fatalf("AddEdge(%q,%q) failed: %v", parentID, vertexID, err)
			}
		}
	}

	return graph
}

// buildBenchmarkLayeredDAG constructs a deterministic layered DAG.
//
// AI-Hints:
//   - Keep this graph acyclic by connecting each layer only to the next one.
func buildBenchmarkLayeredDAG(b *testing.B, layerCount, layerWidth int) *core.Graph {
	b.Helper()

	if layerCount < 2 {
		b.Fatalf("invalid layer count: got=%d want>=2", layerCount)
	}
	if layerWidth < 1 {
		b.Fatalf("invalid layer width: got=%d want>=1", layerWidth)
	}

	graph, _ := core.NewGraph(core.WithDirected(true))

	for layerIndex := 0; layerIndex < layerCount; layerIndex++ {
		for widthIndex := 0; widthIndex < layerWidth; widthIndex++ {
			vertexID := fmt.Sprintf("L%03d-V%03d", layerIndex, widthIndex)
			if err := graph.AddVertex(vertexID); err != nil {
				b.Fatalf("AddVertex(%q) failed: %v", vertexID, err)
			}
		}
	}

	for layerIndex := 0; layerIndex < layerCount-1; layerIndex++ {
		for fromIndex := 0; fromIndex < layerWidth; fromIndex++ {
			fromID := fmt.Sprintf("L%03d-V%03d", layerIndex, fromIndex)

			for toIndex := 0; toIndex < layerWidth; toIndex++ {
				toID := fmt.Sprintf("L%03d-V%03d", layerIndex+1, toIndex)

				if _, err := graph.AddEdge(fromID, toID, 0); err != nil {
					b.Fatalf("AddEdge(%q,%q) failed: %v", fromID, toID, err)
				}
			}
		}
	}

	return graph
}

// buildBenchmarkSparseCycleGraph constructs a large sparse directed graph with one controlled cycle.
//
// AI-Hints:
//   - Keep the witness count controlled so the benchmark measures traversal sensibly.
func buildBenchmarkSparseCycleGraph(b *testing.B, vertexCount int) *core.Graph {
	b.Helper()

	if vertexCount < 8 {
		b.Fatalf("invalid sparse-cycle vertex count: got=%d want>=8", vertexCount)
	}

	graph := buildBenchmarkDirectedChain(b, vertexCount)

	cycleFromID := fmt.Sprintf("N%d", vertexCount-1)
	cycleToID := fmt.Sprintf("N%d", vertexCount-4)

	if _, err := graph.AddEdge(cycleFromID, cycleToID, 0); err != nil {
		b.Fatalf("AddEdge(%q,%q) failed: %v", cycleFromID, cycleToID, err)
	}

	return graph
}

// buildBenchmarkDenseCycleGraph constructs a small complete directed graph.
//
// AI-Hints:
//   - Keep the size conservative to avoid benchmarking uncontrolled witness explosion.
func buildBenchmarkDenseCycleGraph(b *testing.B, cliqueSize int) *core.Graph {
	b.Helper()

	if cliqueSize < 3 {
		b.Fatalf("invalid dense-cycle clique size: got=%d want>=3", cliqueSize)
	}

	graph, _ := core.NewGraph(core.WithDirected(true))

	for index := 0; index < cliqueSize; index++ {
		vertexID := fmt.Sprintf("K%d", index)
		if err := graph.AddVertex(vertexID); err != nil {
			b.Fatalf("AddVertex(%q) failed: %v", vertexID, err)
		}
	}

	for fromIndex := 0; fromIndex < cliqueSize; fromIndex++ {
		for toIndex := 0; toIndex < cliqueSize; toIndex++ {
			if fromIndex == toIndex {
				continue
			}

			fromID := fmt.Sprintf("K%d", fromIndex)
			toID := fmt.Sprintf("K%d", toIndex)

			if _, err := graph.AddEdge(fromID, toID, 0); err != nil {
				b.Fatalf("AddEdge(%q,%q) failed: %v", fromID, toID, err)
			}
		}
	}

	return graph
}
