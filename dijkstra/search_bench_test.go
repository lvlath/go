// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dijkstra_test benchmarks all target methods on matched graph fixtures.
// This file reuses the deterministic graph builders from bench_test.go.
package dijkstra_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/lvlath/go/dijkstra"
)

const (
	// benchmarkSearchGridSide fixes a target-search grid with real route competition.
	benchmarkSearchGridSide = 32

	// benchmarkSearchCutoffVertices keeps a long chain behind a short query budget.
	benchmarkSearchCutoffVertices = 6_001

	// benchmarkSearchCutoffDistance is the inclusive original-cost limit for the chain.
	benchmarkSearchCutoffDistance = 256.0
)

// benchmarkPathResultSink retains one complete target witness from each measured call.
// A single pointer is sufficient because the benchmark reports elapsed execution and allocations.
var benchmarkPathResultSink *dijkstra.PathResult

// benchmarkSearchVariant identifies one method on the same prebuilt route fixture.
// It stores a bound API call rather than duplicating switch branches in the timed loop.
// Implementation:
//   - Stage 1: Retain the method label and an input-bound public call.
//   - Stage 2: Execute one validation call outside the timer.
//   - Stage 3: Invoke the same call inside the measured loop.
//
// Behavior highlights:
//   - Each variant keeps its own real options and heuristic evaluation overhead.
//
// Inputs:
//   - name: benchmark subcase label; method: required executed search mode.
//   - run: fully configured route query.
//
// Returns:
//   - benchmarkSearchVariant: immutable benchmark-only value.
//
// Errors:
//   - The run function propagates public API errors to the caller.
//
// Determinism:
//   - The fixed fixture and method selection do not change between iterations.
//
// Complexity:
//   - Storage O(1); the run function has the selected solver's full complexity.
//
// Notes:
//   - This is a benchmark fixture abstraction, not an additional search facade.
//
// AI-Hints:
//   - Do not cache route results in the run closure; the measured work must be real.
type benchmarkSearchVariant struct {
	name   string
	method dijkstra.SearchMethod
	run    func() (*dijkstra.PathResult, error)
}

// benchmarkSearchCases measures exact public target queries after setup validation.
// It keeps fixture construction and the preliminary correctness check outside timing.
// Implementation:
//   - Stage 1: Run one candidate query and verify reachable/unreachable contract.
//   - Stage 2: Reset the timer and report per-operation allocations.
//   - Stage 3: Execute the same configured public method b.N times.
//
// Behavior highlights:
//   - Timings include snapshot cloning, validation, heuristics and path publication.
//   - Method names identify the same topology and terminal pair, not different fixtures.
//
// Inputs:
//   - b: benchmark handle; variants: immutable bound queries; expectedCost: original cost.
//   - sourceID, targetID: original terminal IDs for witness validation.
//   - reachable: expected target reachability under the shared policy.
//
// Returns:
//   - None; subbenchmarks publish normal Go benchmark metrics.
//
// Errors:
//   - Fatal benchmark failure if preflight or a timed call violates the public contract.
//
// Determinism:
//   - Fixed variant order and fixed inputs for every iteration.
//
// Complexity:
//   - Each iteration has the full cost of its selected public search method.
//   - Setup is excluded from measured time.
//
// Notes:
//   - No MB/s is reported: vertex-plus-edge counts are not real byte throughput.
//
// AI-Hints:
//   - Do not time graph setup or use a fake result sink to manufacture zero work.
//   - Keep the hot loop free of formatting, graph mutation and fixture generation.
func benchmarkSearchCases(
	b *testing.B,
	variants []benchmarkSearchVariant,
	expectedCost float64,
	reachable bool,
	sourceID, targetID string,
) {
	b.Helper()

	for _, variant := range variants {
		b.Run(variant.name, func(b *testing.B) {
			// One genuine public call proves that the timed path is the intended
			// success branch, not an early error or accidentally truncated graph.
			b.StopTimer()
			preflight, err := variant.run()
			if err != nil {
				b.Fatalf("%s preflight failed: %v", variant.name, err)
			}
			if preflight == nil || preflight.Method != variant.method ||
				preflight.Reachable != reachable || preflight.Distance != expectedCost {
				b.Fatalf("%s preflight returned wrong method/reachability/cost: %+v", variant.name, preflight)
			}
			// A correct number with no original-edge witness would not be a valid
			// target-search result. Check that every timed method has this contract.
			if reachable {
				if len(preflight.Vertices) != len(preflight.EdgeIDs)+1 ||
					len(preflight.Vertices) == 0 || preflight.Vertices[0] != sourceID ||
					preflight.Vertices[len(preflight.Vertices)-1] != targetID {
					b.Fatalf("%s preflight returned malformed source-to-target witness: %+v", variant.name, preflight)
				}
			} else if preflight.Vertices != nil || preflight.EdgeIDs != nil {
				b.Fatalf("%s preflight published forbidden unreachable witness: %+v", variant.name, preflight)
			}

			b.ReportAllocs()
			b.ResetTimer()
			b.StartTimer()
			for iteration := 0; iteration < b.N; iteration++ {
				result, err := variant.run()
				if err != nil {
					b.Fatal(err)
				}
				benchmarkPathResultSink = result
			}
		})
	}
}

// BenchmarkSearch_GridFourMethods compares every implemented target kernel fairly.
// The shared directed grid has equal-cost routes, finite integral weights and exact potentials.
// Implementation:
//   - Stage 1: Build one 32x32 graph and precompute both Manhattan heuristic tables.
//   - Stage 2: Bind four public method calls to the same source and target.
//   - Stage 3: Preflight each route, then measure complete end-to-end searches.
//
// Behavior highlights:
//   - A* and balanced A* evaluate actual frozen heuristics, not an omitted callback.
//   - Original Dijkstra, bidirectional Dijkstra and both A* methods share graph and terminals.
//
// Inputs:
//   - b: benchmark handle; graph and callbacks are prepared before subbenchmarks run.
//
// Returns:
//   - None; reports ns/op, B/op, allocs/op for each method.
//
// Errors:
//   - Fails immediately on graph construction, invalid heuristic or incorrect preflight cost.
//
// Determinism:
//   - Stable row/column IDs and all weights equal one; no randomness.
//
// Complexity:
//   - Setup O(V+E); each run includes search preparation and the selected kernel.
//   - Dense-grid E=2*side*(side-1), V=side^2.
//
// Notes:
//   - End-to-end results must not be advertised as pure heap-kernel microbenchmarks.
//
// AI-Hints:
//   - Do not time heuristic-table construction or compare different grid sizes.
//   - Relative performance depends on topology, heuristic quality and snapshot cost.
func BenchmarkSearch_GridFourMethods(b *testing.B) {
	fixture := buildBenchmarkDirectedGrid(b, benchmarkSearchGridSide)
	source := fixture.sourceID
	target := benchmarkGridVertexID(benchmarkSearchGridSide-1, benchmarkSearchGridSide-1)
	finalStep := benchmarkSearchGridSide - 1
	expectedCost := float64(2 * finalStep)

	// Immutable integer Manhattan tables avoid test-side parsing overhead per
	// callback while preserving the real one-call-per-vertex search contract.
	toTarget := make(map[string]float64, fixture.vertexCount)
	fromSource := make(map[string]float64, fixture.vertexCount)
	for row := 0; row < benchmarkSearchGridSide; row++ {
		for column := 0; column < benchmarkSearchGridSide; column++ {
			vertexID := benchmarkGridVertexID(row, column)
			toTarget[vertexID] = float64(2*finalStep - row - column)
			fromSource[vertexID] = float64(row + column)
		}
	}
	estimateTarget := func(id string) (float64, error) {
		value, exists := toTarget[id]
		if !exists {
			return 0, fmt.Errorf("benchmark: unknown to-target vertex %q", id)
		}
		return value, nil
	}
	estimateSource := func(id string) (float64, error) {
		value, exists := fromSource[id]
		if !exists {
			return 0, fmt.Errorf("benchmark: unknown from-source vertex %q", id)
		}
		return value, nil
	}

	variants := []benchmarkSearchVariant{
		{"TargetDijkstra", dijkstra.SearchDijkstra, func() (*dijkstra.PathResult, error) {
			return dijkstra.Search(fixture.graph, source, target)
		}},
		{"BidirectionalDijkstra", dijkstra.SearchBidirectional, func() (*dijkstra.PathResult, error) {
			return dijkstra.Bidirectional(fixture.graph, source, target)
		}},
		{"ConsistentAStar", dijkstra.SearchAStar, func() (*dijkstra.PathResult, error) {
			return dijkstra.AStar(fixture.graph, source, target, estimateTarget)
		}},
		{"BalancedBidirectionalAStar", dijkstra.SearchBidirectionalAStar, func() (*dijkstra.PathResult, error) {
			return dijkstra.BidirectionalAStar(fixture.graph, source, target, estimateTarget, estimateSource)
		}},
	}
	benchmarkSearchCases(b, variants, expectedCost, true, source, target)
}

// BenchmarkSearch_CutoffDirectedChain compares one-frontier and two-frontier pruning.
// A fixed finite original-cost budget excludes the distant target on the same long chain.
// Implementation:
//   - Stage 1: Build a 6001-vertex directed unit-cost chain before timing.
//   - Stage 2: Bind target Search and Bidirectional to one cutoff configuration.
//   - Stage 3: Preflight known-unreachable outcomes and measure whole public calls.
//
// Behavior highlights:
//   - Both variants pay their real snapshot/validation costs despite early traversal cutoff.
//   - Returning +Inf with nil error is required, not a shortcut error branch.
//
// Inputs:
//   - b: benchmark state; all fixture parameters are fixed constants.
//
// Returns:
//   - None; reports actual execution/allocations, not synthetic work counts.
//
// Errors:
//   - Fails on construction or unexpected reachable/errored outcomes.
//
// Determinism:
//   - One directed route with fixed unit weights and an inclusive finite cutoff.
//
// Complexity:
//   - Setup O(V), each iteration includes O(V log V+E log E) preparation
//     and the cutoff-limited frontier exploration.
//
// Notes:
//   - This isolates cutoff behavior from the four-method fully reachable grid regime.
//
// AI-Hints:
//   - Do not compare this cutoff workload to an unbounded grid as an algorithm speedup.
//   - The target remains present in the graph even though no admissible route is published.
func BenchmarkSearch_CutoffDirectedChain(b *testing.B) {
	fixture := buildBenchmarkDirectedChain(b, benchmarkSearchCutoffVertices)
	source := fixture.sourceID
	target := benchmarkVertexID(benchmarkSearchCutoffVertices - 1)

	variants := []benchmarkSearchVariant{
		{"TargetDijkstra", dijkstra.SearchDijkstra, func() (*dijkstra.PathResult, error) {
			return dijkstra.Search(fixture.graph, source, target,
				dijkstra.WithSearchMaxDistance(benchmarkSearchCutoffDistance))
		}},
		{"BidirectionalDijkstra", dijkstra.SearchBidirectional, func() (*dijkstra.PathResult, error) {
			return dijkstra.Bidirectional(fixture.graph, source, target,
				dijkstra.WithSearchMaxDistance(benchmarkSearchCutoffDistance))
		}},
	}
	benchmarkSearchCases(b, variants, math.Inf(1), false, source, target)
}
