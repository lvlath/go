// Package core_test provides benchmarks for core.Graph operations.
package core_test

import (
	"fmt"
	"testing"

	"github.com/lvlath/go/core"
)

// Benchmark sinks prevent accidental dead-code elimination.
var (
	benchSinkString string
	benchSinkEdges  []*core.Edge
	benchSinkGraph  *core.Graph
)

// BenchmarkAddEdge_Unweighted measures AddEdge throughput under the default policy (unweighted, undirected),
// excluding string formatting costs from the timed region.
//
// Improvements:
//   - Strings are pre-generated to measure Graph performance, not fmt performance.
//   - Errors are captured to ensure API correctness without adding branching overhead in the hot loop.
//
// Behavior highlights:
//   - Exercises the unweighted fast-path (weight must be 0).
//
// Complexity:
//   - Per iteration: expected O(1) amortized (hash-map + adjacency updates).
func BenchmarkAddEdge_Unweighted(b *testing.B) {
	// Create a new default Graph (undirected, unweighted)
	g, _ := core.NewGraph()

	var i int
	ids := make([]string, b.N)
	for i = 0; i < b.N; i++ {
		ids[i] = fmt.Sprintf("N%d", i)
	}

	// Report memory allocations per operation
	b.ReportAllocs()
	// Reset timer to exclude setup cost
	b.ResetTimer()

	var err error
	for i = 0; i < b.N; i++ {
		// Use weight=0 for unweighted graph.
		_, err = g.AddEdge("Root", ids[i], 0)
	}

	// Verify correctness outside the hot path.
	if err != nil {
		b.Fatal(err)
	}
}

// BenchmarkAddEdge_Weighted measures AddEdge throughput when weights are enabled, excluding vertex-ID formatting
// from the timed region.
//
// Complexity:
//   - Per iteration: expected O(1) amortized.
func BenchmarkAddEdge_Weighted(b *testing.B) {
	// Create a weighted Graph
	g, _ := core.NewGraph(core.WithWeighted())

	var i int
	ids := make([]string, b.N)
	for i = 0; i < b.N; i++ {
		ids[i] = fmt.Sprintf("N%d", i)
	}

	b.ReportAllocs()
	b.ResetTimer()

	var err error
	for i = 0; i < b.N; i++ {
		// Use 'i' as weight.
		_, err = g.AddEdge("Root", ids[i], float64(i))
	}

	if err != nil {
		b.Fatal(err)
	}
}

// BenchmarkAddEdge_MultiEdges measures AddEdge under high parallel-edge pressure by repeatedly targeting a small,
// fixed set of destination vertices while multi-edges are enabled.
//
// Complexity:
//   - Per iteration: expected O(1) amortized.
func BenchmarkAddEdge_MultiEdges(b *testing.B) {
	// Create graph allowing multi-edges and weights
	g, _ := core.NewGraph(core.WithWeighted(), core.WithMultiEdges())

	// Keep endpoint cardinality small to force parallel edges.
	const targets = 100
	var i int
	ids := make([]string, targets)
	for i = 0; i < targets; i++ {
		ids[i] = fmt.Sprintf("N%d", i)
	}

	b.ReportAllocs()
	b.ResetTimer()

	var err error
	for i = 0; i < b.N; i++ {
		// Cycle through targets to create many parallel edges.
		_, err = g.AddEdge("Root", ids[i%targets], float64(i))
	}

	if err != nil {
		b.Fatal(err)
	}
}

// BenchmarkNeighbors measures Neighbors("Center") on a fixed star topology, focusing on the
// per-call cost of assembling and sorting the neighbor edge slice.
//
// Complexity:
//   - Per iteration: O(d log d), where d is the degree of "Center".
func BenchmarkNeighbors(b *testing.B) {
	// Create graph with multi-edge support
	g, _ := core.NewGraph(core.WithMultiEdges())
	// Build a star with 1000 leaves: Center→Node{i}
	var i int
	for i = 0; i < 1000; i++ {
		_, _ = g.AddEdge("Center", fmt.Sprintf("Node%d", i), 0)
	}

	b.ReportAllocs()
	b.ResetTimer()

	var edges []*core.Edge
	var err error
	for i = 0; i < b.N; i++ {
		edges, err = g.Neighbors("Center")
		benchSinkEdges = edges
	}

	if err != nil {
		b.Fatal(err)
	}
}

// BenchmarkClone measures Clone() cost for a pre-populated graph, focusing on the deep-copy
// of vertices, edges, and adjacency structures.
//
// Complexity:
//   - Per iteration: O(V+E).
func BenchmarkClone(b *testing.B) {
	// Create graph with loops, multi-edges, and weights
	g, _ := core.NewGraph(core.WithWeighted(), core.WithMultiEdges(), core.WithLoops())
	// Populate with 1000 edges A→V{i}

	var i int
	for i = 0; i < 1000; i++ {
		_, _ = g.AddEdge("A", fmt.Sprintf("V%d", i), float64(i))
	}

	b.ReportAllocs()
	b.ResetTimer()

	for i = 0; i < b.N; i++ {
		// Clone performs O(V+E) copy
		benchSinkGraph = g.Clone()
	}
}
