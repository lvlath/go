// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mst_test benchmarks the public Steiner construction and covering-walk APIs.
// Each regime measures a complete public call while varying one dominant algorithmic phase.
package mst_test

import (
	"fmt"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/mst"
)

const (
	steinerBenchmarkVertices      = 240
	steinerBenchmarkEdges         = 700
	steinerBenchmarkWalkSpine     = 240
	steinerBenchmarkBranchEvery   = 3
	steinerBenchmarkExpectedCycle = 211
)

var (
	steinerBenchmarkNetwork *mst.SteinerResult
	steinerBenchmarkWalk    *mst.WalkResult
)

// BenchmarkSteinerTree_AdmissionSingleton measures the full-graph admission and clone floor.
// A single required vertex avoids shortest paths, closure selection, and union pruning.
// Implementation:
//   - Stage 1: Construct one connected deterministic 240-vertex, 700-edge graph.
//   - Stage 2: Time SteinerTree with one terminal and no optional limits.
//   - Stage 3: Retain the returned result to prevent compiler elimination.
//
// Behavior highlights:
//   - The timed call still validates every original edge, including irrelevant ones.
//
// Inputs:
//   - b: benchmark harness; graph setup is excluded from the measured loop.
//
// Returns:
//   - ns/op and allocs/op for the admission/singleton public-call regime.
//
// Errors:
//   - Fixture or algorithm errors stop the benchmark immediately.
//
// Determinism:
//   - Fixed vertex IDs, topology seed, and selected terminal.
//
// Complexity:
//   - Time dominated by graph Clone and sorted catalogs; Space O(V+E).
//
// Notes:
//   - This is not a standalone timing of core.Clone; it includes Steiner admission.
//
// AI-Hints:
//   - Never build the graph or check its shape inside b.N.
func BenchmarkSteinerTree_AdmissionSingleton(b *testing.B) {
	graph := buildBenchmarkConnectedGraph(b, steinerBenchmarkVertices, steinerBenchmarkEdges)
	terminals := []string{benchmarkVertexID(0)}
	b.ReportAllocs()
	b.ResetTimer()
	for iteration := 0; iteration < b.N; iteration++ {
		result, err := mst.SteinerTree(graph, terminals)
		if err != nil {
			b.Fatal(err)
		}
		steinerBenchmarkNetwork = result
	}
}

// BenchmarkSteinerTree_SSSPAndClosure compares terminal-count and predecessor-storage regimes.
// The same graph supports k=2 (one tracked SSSP), k=4, and k=12 (terminal closure).
// Implementation:
//   - Stage 1: Build and validate the connected source graph outside all timed loops.
//   - Stage 2: Run separate sub-benchmarks for terminal cardinality and memory policy.
//   - Stage 3: Time only complete SteinerTree calls and retain their published results.
//
// Behavior highlights:
//   - k=2 has no closure; k>=3 runs one SSSP per lower-index terminal.
//   - Retain trades O(kV) predecessor maps for fewer repeated source searches.
//
// Inputs:
//   - b: benchmark harness; stable exact vertex/edge counts and terminal subsets.
//
// Returns:
//   - Comparable end-to-end timing/allocation regimes, not exclusive per-phase timings.
//
// Errors:
//   - Any unexpected invalid graph, disconnected terminal, or numeric error fails the case.
//
// Determinism:
//   - Same fixed-seed topology for all sub-benchmarks, no random work in the loop.
//
// Complexity:
//   - k>=3 time (k-1+g)Tsssp+O(k^2 log k+Eu log Eu); retain O(kV) live state.
//
// Notes:
//   - Public APIs deliberately do not expose private closure/pruning phase timers.
//   - Interpret differences as phase pressure, not an isolated SSSP microbenchmark.
//
// AI-Hints:
//   - Never label the entire KMB call an isolated Kruskal or SSSP kernel measurement.
//   - Benchmark both modes on the same graph before choosing a default for an application.
func BenchmarkSteinerTree_SSSPAndClosure(b *testing.B) {
	graph := buildBenchmarkConnectedGraph(b, steinerBenchmarkVertices, steinerBenchmarkEdges)
	regimes := []struct {
		name      string
		terminals int
		storage   mst.SteinerPathStorage
	}{
		{"pair-k02", 2, mst.SteinerRecomputePaths},
		{"closure-k04-recompute", 4, mst.SteinerRecomputePaths},
		{"closure-k04-retain", 4, mst.SteinerRetainPaths},
		{"closure-k12-recompute", 12, mst.SteinerRecomputePaths},
		{"closure-k12-retain", 12, mst.SteinerRetainPaths},
	}
	for _, regime := range regimes {
		terminals := make([]string, regime.terminals)
		for index := range terminals {
			// The selected sources are spread across the same connected catalog.
			terminals[index] = benchmarkVertexID(index * (steinerBenchmarkVertices / regime.terminals))
		}
		storage := mst.WithSteinerPathStorage(regime.storage)
		b.Run(regime.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for iteration := 0; iteration < b.N; iteration++ {
				result, err := mst.SteinerTree(graph, terminals, storage)
				if err != nil {
					b.Fatal(err)
				}
				steinerBenchmarkNetwork = result
			}
		})
	}
}

// BenchmarkSteinerTree_CycleUnionAndPruning measures a small topology requiring all KMB stages.
// Selected terminal paths create a cyclic union, then the union MST leaves a removable nonterminal leaf.
// Implementation:
//   - Stage 1: Build the six-edge original graph and verify the known output once.
//   - Stage 2: Time full KMB in each memory mode, not just terminal closure selection.
//   - Stage 3: Retain every completed result outside any validation postprocessing.
//
// Behavior highlights:
//   - The fixture reaches second Kruskal and iterative pruning rather than taking a pair fast-path.
//
// Inputs:
//   - b: benchmark harness; all graph construction happens before ResetTimer.
//
// Returns:
//   - ns/op and allocs/op for a branch-rich, cycle-removing KMB workload.
//
// Errors:
//   - Unexpected shape, output cost, or algorithm errors fail immediately.
//
// Determinism:
//   - Six named original edges with fixed integer weights and three terminals.
//
// Complexity:
//   - End-to-end (k-1+g)Tsssp + O(Eu log Eu+Vu+Eu) for fixed k=3.
//
// Notes:
//   - This intentionally small structural regression is not a dense-graph scalability claim.
//
// AI-Hints:
//   - Do not replace this regime with a tree-shaped union: then pruning/cycle work disappears.
func BenchmarkSteinerTree_CycleUnionAndPruning(b *testing.B) {
	graph, err := core.NewGraph(core.WithWeighted())
	if err != nil {
		b.Fatal(err)
	}
	for _, edge := range []struct {
		id, from, to string
		cost         float64
	}{
		{"AX", "A", "X", 100}, {"CX", "C", "X", 100},
		{"XY", "X", "Y", 1}, {"BY", "B", "Y", 10},
		{"XZ", "X", "Z", 5}, {"BZ", "B", "Z", 6},
	} {
		if _, err := graph.AddEdge(edge.from, edge.to, edge.cost, core.WithID(edge.id)); err != nil {
			b.Fatal(err)
		}
	}
	terminals := []string{"A", "B", "C"}
	for _, mode := range []mst.SteinerPathStorage{mst.SteinerRecomputePaths, mst.SteinerRetainPaths} {
		name := "recompute"
		if mode == mst.SteinerRetainPaths {
			name = "retain"
		}
		option := mst.WithSteinerPathStorage(mode)
		b.Run(name, func(b *testing.B) {
			// Verify the semantic shape once; exclude this check from the hot loop.
			validated, err := mst.SteinerTree(graph, terminals, option)
			if err != nil || validated == nil || validated.TotalWeight != steinerBenchmarkExpectedCycle ||
				len(validated.Edges) != 4 {
				b.Fatalf("cycle/pruning fixture: result=%+v error=%v", validated, err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for iteration := 0; iteration < b.N; iteration++ {
				result, err := mst.SteinerTree(graph, terminals, option)
				if err != nil {
					b.Fatal(err)
				}
				steinerBenchmarkNetwork = result
			}
		})
	}
}

// buildSteinerBenchmarkWalkTree constructs a deep fixed-end spine with attached branches.
// Its caller-owned SteinerResult is structurally valid and priced by exact integer weights.
// Implementation:
//   - Stage 1: Append one original edge per successive spine vertex.
//   - Stage 2: Attach a leaf at every third spine vertex to create off-spine work.
//   - Stage 3: Return a network with accurate canonical unique-edge TotalWeight.
//
// Behavior highlights:
//   - All edges cost one; terminal endpoints lie at opposite ends of the long spine.
//
// Inputs:
//   - b: benchmark harness used to reject incorrect setup dimensions.
//
// Returns:
//   - *SteinerResult: caller-owned test network prepared entirely outside timing.
//
// Errors:
//   - Invalid fixture configuration fails via b.Fatal.
//
// Determinism:
//   - Fixed ID formatting and edge insertion order; no map or random iteration.
//
// Complexity:
//   - Setup Time O(V+E), Space O(V+E); no graph allocation inside timed loops.
//
// Notes:
//   - Walk benchmarks include the public tree validator and output allocation.
//
// AI-Hints:
//   - Passing a handcrafted tree is supported: SteinerWalk validates structure, not provenance.
func buildSteinerBenchmarkWalkTree(b *testing.B) *mst.SteinerResult {
	b.Helper()
	if steinerBenchmarkWalkSpine < 2 || steinerBenchmarkBranchEvery <= 0 {
		b.Fatal("invalid walk benchmark dimensions")
	}
	vertices := make([]string, 0, steinerBenchmarkWalkSpine*2)
	edges := make([]core.Edge, 0, steinerBenchmarkWalkSpine*2)
	for index := 0; index < steinerBenchmarkWalkSpine; index++ {
		current := fmt.Sprintf("spine-%03d", index)
		vertices = append(vertices, current)
		if index > 0 {
			previous := fmt.Sprintf("spine-%03d", index-1)
			edges = append(edges, core.Edge{
				ID:   fmt.Sprintf("spine-link-%03d", index),
				From: previous, To: current, Weight: 1,
			})
		}
		if index%steinerBenchmarkBranchEvery == 0 {
			branch := fmt.Sprintf("branch-%03d", index)
			vertices = append(vertices, branch)
			edges = append(edges, core.Edge{
				ID:   fmt.Sprintf("branch-link-%03d", index),
				From: current, To: branch, Weight: 1,
			})
		}
	}
	return &mst.SteinerResult{
		Terminals:   []string{"spine-000", fmt.Sprintf("spine-%03d", steinerBenchmarkWalkSpine-1)},
		Vertices:    vertices,
		Edges:       edges,
		TotalWeight: float64(len(edges)),
	}
}

// BenchmarkSteinerWalk_ClosedAndFixed measures complete tree validation plus itinerary construction.
// Closed traversal repeats every edge; fixed-end traversal repeats only off-spine branches.
// Implementation:
//   - Stage 1: Construct one valid long-spine, branch-rich caller-owned tree outside timing.
//   - Stage 2: Time closed and fixed-end public calls as independent sub-benchmarks.
//   - Stage 3: Keep the route result alive so allocations and repeated-ID output are measurable.
//
// Behavior highlights:
//   - Distinct output lengths reveal costs of actual passages, not just the scalar distance formula.
//
// Inputs:
//   - b: benchmark harness; a network of hundreds of edges is shared read-only.
//
// Returns:
//   - ns/op, B/op, and allocs/op for closed and open edge-covering public calls.
//
// Errors:
//   - Any rejected tree, missing endpoint, or numeric failure aborts the run.
//
// Determinism:
//   - Canonical per-vertex neighbor order and original edge IDs govern itinerary order.
//
// Complexity:
//   - Validation O(V log V+E log E+sum deg log deg), traversal O(S), Space O(V+E+S).
//
// Notes:
//   - Results include copy/sort/DSU preparation; they are not isolated DFS timings.
//
// AI-Hints:
//   - Do not compute 2*TotalWeight instead of a route: passage memory is the measured artifact.
func BenchmarkSteinerWalk_ClosedAndFixed(b *testing.B) {
	tree := buildSteinerBenchmarkWalkTree(b)
	end := fmt.Sprintf("spine-%03d", steinerBenchmarkWalkSpine-1)
	for _, caseData := range []struct {
		name, start, finish string
	}{
		{"closed-all-branches", "spine-000", "spine-000"},
		{"fixed-long-spine", "spine-000", end},
	} {
		b.Run(caseData.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for iteration := 0; iteration < b.N; iteration++ {
				result, err := mst.SteinerWalk(tree, caseData.start, caseData.finish)
				if err != nil {
					b.Fatal(err)
				}
				steinerBenchmarkWalk = result
			}
		})
	}
}
