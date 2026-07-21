// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package flow_test

import (
	"strconv"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/flow"
)

var benchSinkFlowResult *flow.MaxFlowResult

// buildLayeredBenchmarkGraph builds a deterministic two-layer capacity network.
// The graph shape guarantees that source and sink are connected and that max-flow is known.
//
// Implementation:
//   - Stage 1: Create a directed weighted graph.
//   - Stage 2: Add source and sink vertices.
//   - Stage 3: Add width independent middle vertices.
//   - Stage 4: Connect S -> A_i -> T with identical capacities.
//
// Behavior highlights:
//   - No random graph generation is used.
//   - Every benchmark measures a successful max-flow regime.
//   - All setup errors fail fast through b.Fatalf.
//
// Inputs:
//   - b: benchmark handle.
//   - width: number of independent middle vertices.
//   - capacity: capacity for each S->A_i and A_i->T edge.
//
// Returns:
//   - *core.Graph: deterministic benchmark graph.
//
// Errors:
//   - Setup failures call b.Fatalf immediately.
//
// Determinism:
//   - Vertex IDs and edge insertion order are stable.
//   - Max-flow value is width * capacity.
//
// Complexity:
//   - Time O(width), Space O(width).
//
// Notes:
//   - This shape is especially favorable to Dinic and still useful for allocation tracking.
//
// AI-Hints:
//   - Do not replace this with random graphs in hot benchmarks.
//   - Use random graphs only in explicitly named stochastic/stress benchmarks.
func buildLayeredBenchmarkGraph(b *testing.B, width int, capacity float64) *core.Graph {
	b.Helper()

	g, err := core.NewGraph(core.WithDirected(true), core.WithWeighted())
	if err != nil {
		b.Fatalf("NewGraph: %v", err)
	}

	if err = g.AddVertex("S"); err != nil {
		b.Fatalf("AddVertex(S): %v", err)
	}
	if err = g.AddVertex("T"); err != nil {
		b.Fatalf("AddVertex(T): %v", err)
	}

	for i := 0; i < width; i++ {
		middle := "A" + strconv.Itoa(i)

		if err = g.AddVertex(middle); err != nil {
			b.Fatalf("AddVertex(%s): %v", middle, err)
		}
		if _, err = g.AddEdge("S", middle, capacity); err != nil {
			b.Fatalf("AddEdge(S,%s): %v", middle, err)
		}
		if _, err = g.AddEdge(middle, "T", capacity); err != nil {
			b.Fatalf("AddEdge(%s,T): %v", middle, err)
		}
	}

	return g
}

// buildDenseBipartiteBenchmarkGraph builds a deterministic dense middle-layer network.
// It stresses residual adjacency scanning more than the simple layered graph.
//
// Implementation:
//   - Stage 1: Add source, sink, left partition, and right partition.
//   - Stage 2: Connect S to every left vertex.
//   - Stage 3: Connect every left vertex to every right vertex.
//   - Stage 4: Connect every right vertex to T.
//
// Behavior highlights:
//   - Graph is connected by construction.
//   - Edge count is O(left*right), so it exercises residual traversal cost.
//
// Inputs:
//   - b: benchmark handle.
//   - left: number of left partition vertices.
//   - right: number of right partition vertices.
//   - capacity: capacity for all edges.
//
// Returns:
//   - *core.Graph: deterministic dense bipartite capacity graph.
//
// Errors:
//   - Setup failures call b.Fatalf immediately.
//
// Determinism:
//   - Stable vertex and edge insertion order.
//
// Complexity:
//   - Time O(left*right), Space O(left*right).
//
// AI-Hints:
//   - Keep dimensions modest for Edmonds-Karp and Ford-Fulkerson.
func buildDenseBipartiteBenchmarkGraph(
	b *testing.B,
	left int,
	right int,
	capacity float64,
) *core.Graph {
	b.Helper()

	g, err := core.NewGraph(core.WithDirected(true), core.WithWeighted())
	if err != nil {
		b.Fatalf("NewGraph: %v", err)
	}

	if err = g.AddVertex("S"); err != nil {
		b.Fatalf("AddVertex(S): %v", err)
	}
	if err = g.AddVertex("T"); err != nil {
		b.Fatalf("AddVertex(T): %v", err)
	}

	for i := 0; i < left; i++ {
		leftID := "L" + strconv.Itoa(i)
		if err = g.AddVertex(leftID); err != nil {
			b.Fatalf("AddVertex(%s): %v", leftID, err)
		}
		if _, err = g.AddEdge("S", leftID, capacity); err != nil {
			b.Fatalf("AddEdge(S,%s): %v", leftID, err)
		}
	}

	for j := 0; j < right; j++ {
		rightID := "R" + strconv.Itoa(j)
		if err = g.AddVertex(rightID); err != nil {
			b.Fatalf("AddVertex(%s): %v", rightID, err)
		}
		if _, err = g.AddEdge(rightID, "T", capacity); err != nil {
			b.Fatalf("AddEdge(%s,T): %v", rightID, err)
		}
	}

	for i := 0; i < left; i++ {
		leftID := "L" + strconv.Itoa(i)

		for j := 0; j < right; j++ {
			rightID := "R" + strconv.Itoa(j)
			if _, err = g.AddEdge(leftID, rightID, capacity); err != nil {
				b.Fatalf("AddEdge(%s,%s): %v", leftID, rightID, err)
			}
		}
	}

	return g
}

// benchmarkMaxFlow runs MaxFlow in one named algorithm regime.
// It validates one warm-up run before measuring hot-loop performance.
//
// Implementation:
//   - Stage 1: Run one setup validation call before ResetTimer.
//   - Stage 2: Report allocations for algorithmic memory analysis.
//   - Stage 3: Execute b.N successful MaxFlow calls.
//   - Stage 4: Store the result in a package-level sink to prevent dead-code elimination.
//
// Behavior highlights:
//   - Hot loop never ignores errors.
//   - Benchmarks measure successful max-flow computation, not setup or error paths.
//
// Inputs:
//   - b: benchmark handle.
//   - g: deterministic graph built before timing.
//   - source: source vertex ID.
//   - sink: sink vertex ID.
//   - opts: canonical flow options.
//
// Returns:
//   - None.
//
// Errors:
//   - Any setup or hot-loop error calls b.Fatal.
//
// Determinism:
//   - Determinism is inherited from MaxFlow and graph construction.
//
// Complexity:
//   - Benchmark cost is b.N times selected algorithm complexity.
//
// AI-Hints:
//   - Do not move graph construction into the timed loop unless measuring construction.
func benchmarkMaxFlow(
	b *testing.B,
	g *core.Graph,
	source string,
	sink string,
	opts ...flow.Option,
) {
	b.Helper()

	result, err := flow.MaxFlow(g, source, sink, opts...)
	if err != nil {
		b.Fatalf("setup MaxFlow: %v", err)
	}
	if result == nil {
		b.Fatalf("setup MaxFlow: nil result")
	}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		result, err = flow.MaxFlow(g, source, sink, opts...)
		if err != nil {
			b.Fatal(err)
		}

		benchSinkFlowResult = result
	}
}

func BenchmarkMaxFlow_Dinic_LayeredWide(b *testing.B) {
	g := buildLayeredBenchmarkGraph(b, 1000, 1)

	benchmarkMaxFlow(
		b,
		g,
		"S",
		"T",
		flow.WithAlgorithm(flow.AlgorithmDinic),
	)
}

func BenchmarkMaxFlow_EdmondsKarp_LayeredMedium(b *testing.B) {
	g := buildLayeredBenchmarkGraph(b, 200, 1)

	benchmarkMaxFlow(
		b,
		g,
		"S",
		"T",
		flow.WithAlgorithm(flow.AlgorithmEdmondsKarp),
	)
}

func BenchmarkMaxFlow_FordFulkerson_LayeredSmall(b *testing.B) {
	g := buildLayeredBenchmarkGraph(b, 64, 1)

	benchmarkMaxFlow(
		b,
		g,
		"S",
		"T",
		flow.WithAlgorithm(flow.AlgorithmFordFulkerson),
		flow.WithMaxAugmentations(128),
	)
}

func BenchmarkMaxFlow_Dinic_DenseBipartite(b *testing.B) {
	g := buildDenseBipartiteBenchmarkGraph(b, 64, 64, 1)

	benchmarkMaxFlow(
		b,
		g,
		"S",
		"T",
		flow.WithAlgorithm(flow.AlgorithmDinic),
	)
}

func BenchmarkCapacityMatrix_DenseBipartite(b *testing.B) {
	g := buildDenseBipartiteBenchmarkGraph(b, 64, 64, 1)

	matrixValue, order, err := flow.CapacityMatrix(g)
	if err != nil {
		b.Fatalf("setup CapacityMatrix: %v", err)
	}
	if matrixValue == nil || len(order) == 0 {
		b.Fatalf("setup CapacityMatrix: invalid output")
	}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		matrixValue, order, err = flow.CapacityMatrix(g)
		if err != nil {
			b.Fatal(err)
		}
		if matrixValue == nil || len(order) == 0 {
			b.Fatal("CapacityMatrix returned invalid output")
		}
	}
}
