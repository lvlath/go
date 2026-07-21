// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package dijkstra_test

import (
	"strconv"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/dijkstra"
)

// AI-HINTS (file):
//   - Benchmarks must measure the public Dijkstra API, not graph construction.
//   - All fixture generation must complete before b.ResetTimer().
//   - Every setup error must fail immediately; never benchmark an unknown topology.
//   - Keep benchmark fixtures deterministic and shape-stable.
//   - Use b.ReportAllocs() for every benchmark.
//   - Use b.SetBytes(...) with a stable topology-size proxy so runs stay comparable.
//   - The hot loop must contain only Dijkstra (+ minimal err check).
//   - Path-tracking comparison benchmarks must use the same graph shape, source, and workload law.

const (
	// benchmarkChainVertexCount controls the size of the sparse directed chain benchmark.
	benchmarkChainVertexCount = 10_001

	// benchmarkGridSide controls the side length of the directed grid benchmark.
	benchmarkGridSide = 96

	// benchmarkMixedSegmentCount controls the size of the mixed-edge corridor benchmark.
	benchmarkMixedSegmentCount = 4_096

	// benchmarkTrackingVertexCount controls the size of the identical fixtures used for
	// WithPathTracking vs WithoutPathTracking comparison.
	benchmarkTrackingVertexCount = 8_001

	// benchmarkCutoffVertexCount controls the size of the MaxDistance cutoff benchmark chain.
	benchmarkCutoffVertexCount = 12_001

	// benchmarkWallSegmentCount controls the size of the threshold-wall corridor benchmark.
	benchmarkWallSegmentCount = 4_096

	// benchmarkWeightUnit is the canonical baseline traversable edge weight.
	benchmarkWeightUnit = 1.0

	// benchmarkWeightMixedHeavy is a slightly heavier main-edge weight used in mixed fixtures.
	benchmarkWeightMixedHeavy = 2.0

	// benchmarkWeightShortcut is the deterministic forward shortcut weight in mixed fixtures.
	benchmarkWeightShortcut = 3.0

	// benchmarkMaxDistanceCutoff is the public cutoff used in the cutoff benchmark.
	benchmarkMaxDistanceCutoff = 256.0

	// benchmarkWallThreshold is the public edge-wall threshold used in the wall benchmark.
	benchmarkWallThreshold = 1.5

	// benchmarkWallDirectWeight is the direct edge weight that becomes impassable under the threshold.
	benchmarkWallDirectWeight = 1.5

	// benchmarkWallBypassWeight is the weight of each bypass edge around a blocked direct edge.
	benchmarkWallBypassWeight = 1.0

	// benchmarkWallPeriod controls how often a wall segment appears in the wall benchmark.
	benchmarkWallPeriod = 4

	// benchmarkWallOffset controls which segment positions become blocked direct edges.
	benchmarkWallOffset = 1
)

// benchmarkResultSink prevents accidental dead-code elimination of benchmark results.
var benchmarkResultSink *dijkstra.Result

// benchmarkFixture stores one fully built deterministic benchmark topology together
// with its workload metadata.
// The structure exists so each benchmark can report a stable topology-size proxy
// and reuse the exact source vertex without reconstructing graph knowledge.
//
// Implementation:
//   - Stage 1: Store the built graph.
//   - Stage 2: Store the deterministic source vertex identifier.
//   - Stage 3: Store exact vertex and edge counts for benchmark accounting.
//
// Behavior highlights:
//   - The fixture is immutable after construction.
//   - vertexCount and edgeCount describe the actual built workload.
//
// Inputs:
//   - graph: the constructed graph fixture.
//   - sourceID: the deterministic benchmark source vertex.
//   - vertexCount: the number of vertices in the built graph.
//   - edgeCount: the number of edges inserted into the built graph.
//
// Returns:
//   - benchmarkFixture: one deterministic benchmark input package.
//
// Errors:
//   - None directly; builders fail the benchmark on setup errors instead.
//
// Determinism:
//   - Deterministic for the same builder inputs.
//
// Complexity:
//   - Storage cost O(1) beyond the graph pointer.
//
// Notes:
//   - edgeCount is tracked explicitly because the benchmark accounting law is part of the suite design.
//
// AI-Hints:
//   - Keep fixture metadata exact; do not estimate topology size loosely in benchmark accounting.
type benchmarkFixture struct {
	graph       *core.Graph
	sourceID    string
	vertexCount int
	edgeCount   int
}

// benchmarkVertexID formats a deterministic benchmark vertex identifier.
//
// Implementation:
//   - Stage 1: Convert the integer index to decimal text.
//   - Stage 2: Prefix it with a stable benchmark namespace.
//
// Behavior highlights:
//   - Stable textual IDs keep fixture generation reproducible.
//
// Inputs:
//   - index: vertex index within a synthetic benchmark topology.
//
// Returns:
//   - string: deterministic benchmark vertex identifier.
//
// Errors:
//   - None.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(log10(index+1)), Space O(log10(index+1)).
//
// Notes:
//   - This helper is benchmark-setup only and never enters the timed region.
//
// AI-Hints:
//   - Keep benchmark IDs simple and deterministic.
func benchmarkVertexID(index int) string {
	return "v" + strconv.Itoa(index)
}

// benchmarkGridVertexID formats a deterministic vertex identifier for a grid fixture.
//
// Implementation:
//   - Stage 1: Convert row and column indices to decimal text.
//   - Stage 2: Combine them into a stable grid coordinate identifier.
//
// Behavior highlights:
//   - Stable coordinate IDs keep grid shape reproducible.
//
// Inputs:
//   - row: grid row index.
//   - column: grid column index.
//
// Returns:
//   - string: deterministic grid vertex identifier.
//
// Errors:
//   - None.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(log(row)+log(column)), Space O(log(row)+log(column)).
//
// Notes:
//   - This helper is benchmark-setup only and never enters the timed region.
//
// AI-Hints:
//   - Keep coordinate formatting deterministic so benchmark topology stays stable.
func benchmarkGridVertexID(row, column int) string {
	return "g" + strconv.Itoa(row) + "_" + strconv.Itoa(column)
}

// benchmarkAuxVertexID formats a deterministic auxiliary vertex identifier used by
// threshold-wall bypass fixtures.
//
// Implementation:
//   - Stage 1: Convert the segment index to decimal text.
//   - Stage 2: Prefix it with an auxiliary namespace.
//
// Behavior highlights:
//   - Auxiliary IDs remain disjoint from primary corridor IDs.
//
// Inputs:
//   - index: wall-segment index.
//
// Returns:
//   - string: deterministic auxiliary vertex identifier.
//
// Errors:
//   - None.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(log10(index+1)), Space O(log10(index+1)).
//
// Notes:
//   - This helper is benchmark-setup only.
//
// AI-Hints:
//   - Keep bypass-node naming explicit so topology reasoning stays simple.
func benchmarkAuxVertexID(index int) string {
	return "aux" + strconv.Itoa(index)
}

// buildBenchmarkDirectedChain constructs a deterministic weighted directed chain.
// The resulting workload is the clean baseline regime for Dijkstra: one dominant
// forward route, sparse topology, and predictable frontier growth.
//
// Implementation:
//   - Stage 1: Validate the requested vertex count.
//   - Stage 2: Precompute deterministic vertex IDs.
//   - Stage 3: Insert exactly vertexCount-1 directed weighted edges.
//   - Stage 4: Return the fully built fixture with exact workload metadata.
//
// Behavior highlights:
//   - Exactly V=vertexCount vertices are reachable from the source.
//   - Exactly E=vertexCount-1 edges are inserted.
//   - Directed edges remove reverse-adjacency noise from the baseline regime.
//
// Inputs:
//   - vertexCount: the number of vertices in the chain.
//
// Returns:
//   - benchmarkFixture: the fully built benchmark topology.
//
// Errors:
//   - Fatal benchmark failure if the requested size is invalid.
//   - Fatal benchmark failure if graph construction fails.
//
// Determinism:
//   - Deterministic for the same vertexCount.
//
// Complexity:
//   - Setup time O(V).
//   - Setup space O(V).
//
// Notes:
//   - The source is always the first vertex in the chain.
//
// AI-Hints:
//   - Use a directed chain here intentionally; it gives the cleanest baseline for sparse weighted traversal.
func buildBenchmarkDirectedChain(b *testing.B, vertexCount int) benchmarkFixture {
	b.Helper()

	if vertexCount < 2 {
		b.Fatalf("invalid chain vertex count: got=%d want>=2", vertexCount)
	}

	vertexIDs := make([]string, vertexCount)
	for index := 0; index < vertexCount; index++ {
		vertexIDs[index] = benchmarkVertexID(index)
	}

	graph, _ := core.NewGraph(core.WithDirected(true), core.WithWeighted())
	edgeCount := 0

	for index := 0; index < vertexCount-1; index++ {
		_, err := graph.AddEdge(vertexIDs[index], vertexIDs[index+1], benchmarkWeightUnit)
		if err != nil {
			b.Fatalf("setup AddEdge(%q,%q) failed: %v", vertexIDs[index], vertexIDs[index+1], err)
		}
		edgeCount++
	}

	return benchmarkFixture{
		graph:       graph,
		sourceID:    vertexIDs[0],
		vertexCount: graph.VertexCount(),
		edgeCount:   edgeCount,
	}
}

// buildBenchmarkDirectedGrid constructs a deterministic weighted directed grid
// with rightward and downward edges only.
// This is the high-competition regime: many alternative routes with repeated
// candidate competition in the priority queue.
//
// Implementation:
//   - Stage 1: Validate the requested grid side length.
//   - Stage 2: Precompute deterministic grid vertex IDs.
//   - Stage 3: Insert all vertices explicitly.
//   - Stage 4: Insert exactly rightward and downward directed edges.
//   - Stage 5: Return the fully built fixture with exact workload metadata.
//
// Behavior highlights:
//   - Exactly V=side*side vertices are inserted.
//   - Exactly E=2*side*(side-1) directed edges are inserted.
//   - The top-left source reaches the entire grid.
//
// Inputs:
//   - side: number of rows and columns in the square grid.
//
// Returns:
//   - benchmarkFixture: the fully built benchmark topology.
//
// Errors:
//   - Fatal benchmark failure if the requested size is invalid.
//   - Fatal benchmark failure if graph construction fails.
//
// Determinism:
//   - Deterministic for the same side length.
//
// Complexity:
//   - Setup time O(V+E).
//   - Setup space O(V).
//
// Notes:
//   - This regime is intentionally more heap-competitive than the chain regime.
//
// AI-Hints:
//   - Keep the grid directed and acyclic here so the workload emphasizes route competition, not backtracking noise.
func buildBenchmarkDirectedGrid(b *testing.B, side int) benchmarkFixture {
	b.Helper()

	if side < 2 {
		b.Fatalf("invalid grid side: got=%d want>=2", side)
	}

	vertexIDs := make([][]string, side)
	for row := 0; row < side; row++ {
		rowIDs := make([]string, side)
		for column := 0; column < side; column++ {
			rowIDs[column] = benchmarkGridVertexID(row, column)
		}
		vertexIDs[row] = rowIDs
	}

	graph, _ := core.NewGraph(core.WithDirected(true), core.WithWeighted())

	for row := 0; row < side; row++ {
		for column := 0; column < side; column++ {
			if err := graph.AddVertex(vertexIDs[row][column]); err != nil {
				b.Fatalf("setup AddVertex(%q) failed: %v", vertexIDs[row][column], err)
			}
		}
	}

	edgeCount := 0

	for row := 0; row < side; row++ {
		for column := 0; column < side; column++ {
			currentID := vertexIDs[row][column]

			if column+1 < side {
				rightID := vertexIDs[row][column+1]
				_, err := graph.AddEdge(currentID, rightID, benchmarkWeightUnit)
				if err != nil {
					b.Fatalf("setup AddEdge(%q,%q) failed: %v", currentID, rightID, err)
				}
				edgeCount++
			}

			if row+1 < side {
				downID := vertexIDs[row+1][column]
				_, err := graph.AddEdge(currentID, downID, benchmarkWeightUnit)
				if err != nil {
					b.Fatalf("setup AddEdge(%q,%q) failed: %v", currentID, downID, err)
				}
				edgeCount++
			}
		}
	}

	return benchmarkFixture{
		graph:       graph,
		sourceID:    vertexIDs[0][0],
		vertexCount: graph.VertexCount(),
		edgeCount:   edgeCount,
	}
}

// buildBenchmarkMixedCorridor constructs a deterministic weighted mixed-edge graph.
// The primary corridor is always forward-reachable from the source, while edge mode
// alternates between directed and undirected segments and deterministic shortcuts
// create additional candidate competition.
//
// Implementation:
//   - Stage 1: Validate the requested segment count.
//   - Stage 2: Precompute deterministic corridor vertex IDs.
//   - Stage 3: Insert the primary corridor with alternating per-edge direction policy.
//   - Stage 4: Insert forward shortcut edges with stable weights.
//   - Stage 5: Return the fully built fixture with exact workload metadata.
//
// Behavior highlights:
//   - Mixed-edge semantics are exercised on every run.
//   - The source still reaches the entire corridor.
//   - Shortcut competition keeps the workload richer than a plain chain.
//
// Inputs:
//   - segmentCount: number of primary corridor segments.
//
// Returns:
//   - benchmarkFixture: the fully built benchmark topology.
//
// Errors:
//   - Fatal benchmark failure if the requested size is invalid.
//   - Fatal benchmark failure if graph construction fails.
//
// Determinism:
//   - Deterministic for the same segmentCount.
//
// Complexity:
//   - Setup time O(segmentCount).
//   - Setup space O(segmentCount).
//
// Notes:
//   - The graph uses core.WithMixedEdges() intentionally to benchmark endpoint-resolution overhead.
//
// AI-Hints:
//   - Keep this fixture contract-faithful: mixed semantics are the point of the regime, not decorative noise.
func buildBenchmarkMixedCorridor(b *testing.B, segmentCount int) benchmarkFixture {
	b.Helper()

	if segmentCount < 2 {
		b.Fatalf("invalid mixed segment count: got=%d want>=2", segmentCount)
	}

	vertexIDs := make([]string, segmentCount+1)
	for index := 0; index <= segmentCount; index++ {
		vertexIDs[index] = benchmarkVertexID(index)
	}

	graph, _ := core.NewGraph(core.WithWeighted(), core.WithMixedEdges())
	edgeCount := 0

	for index := 0; index < segmentCount; index++ {
		fromID := vertexIDs[index]
		toID := vertexIDs[index+1]

		directed := index%2 == 0
		weight := benchmarkWeightUnit
		if index%5 == 0 {
			weight = benchmarkWeightMixedHeavy
		}

		_, err := graph.AddEdge(fromID, toID, weight, core.WithEdgeDirected(directed))
		if err != nil {
			b.Fatalf("setup AddEdge(%q,%q) failed: %v", fromID, toID, err)
		}
		edgeCount++

		if index+2 <= segmentCount {
			shortcutID := vertexIDs[index+2]
			_, err = graph.AddEdge(fromID, shortcutID, benchmarkWeightShortcut, core.WithEdgeDirected(true))
			if err != nil {
				b.Fatalf("setup AddEdge(%q,%q) failed: %v", fromID, shortcutID, err)
			}
			edgeCount++
		}
	}

	return benchmarkFixture{
		graph:       graph,
		sourceID:    vertexIDs[0],
		vertexCount: graph.VertexCount(),
		edgeCount:   edgeCount,
	}
}

// buildBenchmarkThresholdWallCorridor constructs a deterministic weighted directed corridor
// where every benchmarkWallPeriod-th direct edge becomes impassable under the configured
// InfEdgeThreshold and must be bypassed through a two-edge detour.
//
// Implementation:
//   - Stage 1: Validate the requested segment count.
//   - Stage 2: Precompute deterministic main-corridor vertex IDs.
//   - Stage 3: Insert normal traversable direct edges on non-wall segments.
//   - Stage 4: Insert blocked direct edges plus explicit bypass nodes on wall segments.
//   - Stage 5: Return the fully built fixture with exact workload metadata.
//
// Behavior highlights:
//   - The wall policy changes actual traversal behavior, not just branch coverage.
//   - Direct blocked edges are lighter than the detour, so threshold policy is semantically relevant.
//   - The source still reaches the full corridor through bypass nodes.
//
// Inputs:
//   - segmentCount: number of corridor segments.
//
// Returns:
//   - benchmarkFixture: the fully built benchmark topology.
//
// Errors:
//   - Fatal benchmark failure if the requested size is invalid.
//   - Fatal benchmark failure if graph construction fails.
//
// Determinism:
//   - Deterministic for the same segmentCount.
//
// Complexity:
//   - Setup time O(segmentCount).
//   - Setup space O(segmentCount).
//
// Notes:
//   - This fixture is intentionally policy-sensitive: without the threshold, some blocked direct edges
//     would be optimal; with the threshold, detours become mandatory.
//
// AI-Hints:
//   - Keep threshold-sensitive fixtures semantically real; do not benchmark a threshold branch that changes nothing.
func buildBenchmarkThresholdWallCorridor(b *testing.B, segmentCount int) benchmarkFixture {
	b.Helper()

	if segmentCount < 2 {
		b.Fatalf("invalid wall segment count: got=%d want>=2", segmentCount)
	}

	mainIDs := make([]string, segmentCount+1)
	for index := 0; index <= segmentCount; index++ {
		mainIDs[index] = benchmarkVertexID(index)
	}

	graph, _ := core.NewGraph(core.WithDirected(true), core.WithWeighted())
	edgeCount := 0

	for index := 0; index < segmentCount; index++ {
		fromID := mainIDs[index]
		toID := mainIDs[index+1]

		if index%benchmarkWallPeriod == benchmarkWallOffset {
			_, err := graph.AddEdge(fromID, toID, benchmarkWallDirectWeight)
			if err != nil {
				b.Fatalf("setup AddEdge(%q,%q) failed: %v", fromID, toID, err)
			}
			edgeCount++

			bypassID := benchmarkAuxVertexID(index)

			_, err = graph.AddEdge(fromID, bypassID, benchmarkWallBypassWeight)
			if err != nil {
				b.Fatalf("setup AddEdge(%q,%q) failed: %v", fromID, bypassID, err)
			}
			edgeCount++

			_, err = graph.AddEdge(bypassID, toID, benchmarkWallBypassWeight)
			if err != nil {
				b.Fatalf("setup AddEdge(%q,%q) failed: %v", bypassID, toID, err)
			}
			edgeCount++

			continue
		}

		_, err := graph.AddEdge(fromID, toID, benchmarkWeightUnit)
		if err != nil {
			b.Fatalf("setup AddEdge(%q,%q) failed: %v", fromID, toID, err)
		}
		edgeCount++
	}

	return benchmarkFixture{
		graph:       graph,
		sourceID:    mainIDs[0],
		vertexCount: graph.VertexCount(),
		edgeCount:   edgeCount,
	}
}

// BenchmarkDijkstra_Chain measures Dijkstra on a deterministic sparse weighted directed chain.
//
// Implementation:
//   - Stage 1: Build the chain fixture before timing begins.
//   - Stage 2: Report allocations and topology-size proxy.
//   - Stage 3: Reset the timer.
//   - Stage 4: Repeatedly run Dijkstra in the hot loop.
//
// Behavior highlights:
//   - This is the baseline sparse weighted regime.
//   - The frontier has one dominant forward route.
//
// AI-Hints:
//   - Use this benchmark as the base throughput reference, not as evidence for richer route-competition regimes.
func BenchmarkDijkstra_Chain(b *testing.B) {
	fixture := buildBenchmarkDirectedChain(b, benchmarkChainVertexCount)

	b.ReportAllocs()
	b.SetBytes(int64(fixture.vertexCount + fixture.edgeCount))
	b.ResetTimer()

	for index := 0; index < b.N; index++ {
		result, err := dijkstra.Dijkstra(fixture.graph, fixture.sourceID)
		if err != nil {
			b.Fatalf("Dijkstra(%q) failed: %v", fixture.sourceID, err)
		}
		benchmarkResultSink = result
	}
}

// BenchmarkDijkstra_Grid measures Dijkstra on a deterministic weighted directed grid.
//
// Implementation:
//   - Stage 1: Build the grid fixture before timing begins.
//   - Stage 2: Report allocations and topology-size proxy.
//   - Stage 3: Reset the timer.
//   - Stage 4: Repeatedly run Dijkstra in the hot loop.
//
// Behavior highlights:
//   - This regime exercises many alternative routes and repeated candidate competition.
//   - The workload is intentionally more heap-intensive than the chain regime.
//
// AI-Hints:
//   - Compare this against the chain benchmark to observe heap-pressure and route-competition effects.
func BenchmarkDijkstra_Grid(b *testing.B) {
	fixture := buildBenchmarkDirectedGrid(b, benchmarkGridSide)

	b.ReportAllocs()
	b.SetBytes(int64(fixture.vertexCount + fixture.edgeCount))
	b.ResetTimer()

	for index := 0; index < b.N; index++ {
		result, err := dijkstra.Dijkstra(fixture.graph, fixture.sourceID)
		if err != nil {
			b.Fatalf("Dijkstra(%q) failed: %v", fixture.sourceID, err)
		}
		benchmarkResultSink = result
	}
}

// BenchmarkDijkstra_MixedEdges measures Dijkstra on a deterministic mixed-edge corridor.
//
// Implementation:
//   - Stage 1: Build the mixed-edge fixture before timing begins.
//   - Stage 2: Report allocations and topology-size proxy.
//   - Stage 3: Reset the timer.
//   - Stage 4: Repeatedly run Dijkstra in the hot loop.
//
// Behavior highlights:
//   - This regime exercises mixed directed/undirected semantics and endpoint resolution.
//   - The workload is still fully deterministic and forward-reachable from the source.
//
// AI-Hints:
//   - Treat this as a contract-regime benchmark, not just a generic throughput number.
func BenchmarkDijkstra_MixedEdges(b *testing.B) {
	fixture := buildBenchmarkMixedCorridor(b, benchmarkMixedSegmentCount)

	b.ReportAllocs()
	b.SetBytes(int64(fixture.vertexCount + fixture.edgeCount))
	b.ResetTimer()

	for index := 0; index < b.N; index++ {
		result, err := dijkstra.Dijkstra(fixture.graph, fixture.sourceID)
		if err != nil {
			b.Fatalf("Dijkstra(%q) failed: %v", fixture.sourceID, err)
		}
		benchmarkResultSink = result
	}
}

// BenchmarkDijkstra_WithPathTracking measures Dijkstra when predecessor tracking
// is enabled explicitly on a deterministic baseline chain.
//
// Implementation:
//   - Stage 1: Build the comparison fixture before timing begins.
//   - Stage 2: Report allocations and topology-size proxy.
//   - Stage 3: Reset the timer.
//   - Stage 4: Repeatedly run Dijkstra with WithPathTracking in the hot loop.
//
// Behavior highlights:
//   - This regime isolates the additional predecessor-bookkeeping cost.
//   - The graph shape is intentionally identical to BenchmarkDijkstra_WithoutPathTracking.
//
// AI-Hints:
//   - Never compare tracked vs untracked modes on different topologies.
func BenchmarkDijkstra_WithPathTracking(b *testing.B) {
	fixture := buildBenchmarkDirectedChain(b, benchmarkTrackingVertexCount)

	b.ReportAllocs()
	b.SetBytes(int64(fixture.vertexCount + fixture.edgeCount))
	b.ResetTimer()

	for index := 0; index < b.N; index++ {
		result, err := dijkstra.Dijkstra(fixture.graph, fixture.sourceID, dijkstra.WithPathTracking())
		if err != nil {
			b.Fatalf("Dijkstra(%q, WithPathTracking) failed: %v", fixture.sourceID, err)
		}
		benchmarkResultSink = result
	}
}

// BenchmarkDijkstra_WithoutPathTracking measures Dijkstra on the exact same deterministic
// baseline chain used by BenchmarkDijkstra_WithPathTracking, but without predecessor storage.
//
// Implementation:
//   - Stage 1: Build the comparison fixture before timing begins.
//   - Stage 2: Report allocations and topology-size proxy.
//   - Stage 3: Reset the timer.
//   - Stage 4: Repeatedly run Dijkstra without path tracking in the hot loop.
//
// Behavior highlights:
//   - This regime isolates the pure distance-only baseline.
//   - The graph shape is intentionally identical to BenchmarkDijkstra_WithPathTracking.
//
// AI-Hints:
//   - Keep this benchmark topology identical to the tracked variant or the comparison becomes dishonest.
func BenchmarkDijkstra_WithoutPathTracking(b *testing.B) {
	fixture := buildBenchmarkDirectedChain(b, benchmarkTrackingVertexCount)

	b.ReportAllocs()
	b.SetBytes(int64(fixture.vertexCount + fixture.edgeCount))
	b.ResetTimer()

	for index := 0; index < b.N; index++ {
		result, err := dijkstra.Dijkstra(fixture.graph, fixture.sourceID)
		if err != nil {
			b.Fatalf("Dijkstra(%q) failed: %v", fixture.sourceID, err)
		}
		benchmarkResultSink = result
	}
}

// BenchmarkDijkstra_MaxDistanceCutoff measures Dijkstra on a long deterministic chain
// when MaxDistance terminates the frontier early.
//
// Implementation:
//   - Stage 1: Build the cutoff fixture before timing begins.
//   - Stage 2: Report allocations and topology-size proxy.
//   - Stage 3: Reset the timer.
//   - Stage 4: Repeatedly run Dijkstra with WithMaxDistance in the hot loop.
//
// Behavior highlights:
//   - This regime exercises the public cutoff policy branch as a real traversal mode.
//   - The chain is intentionally much longer than the cutoff radius.
//
// AI-Hints:
//   - Keep cutoff benchmarks on graphs where the cutoff changes actual work, not just published output.
func BenchmarkDijkstra_MaxDistanceCutoff(b *testing.B) {
	fixture := buildBenchmarkDirectedChain(b, benchmarkCutoffVertexCount)

	b.ReportAllocs()
	b.SetBytes(int64(fixture.vertexCount + fixture.edgeCount))
	b.ResetTimer()

	for index := 0; index < b.N; index++ {
		result, err := dijkstra.Dijkstra(
			fixture.graph,
			fixture.sourceID,
			dijkstra.WithMaxDistance(benchmarkMaxDistanceCutoff),
		)
		if err != nil {
			b.Fatalf("Dijkstra(%q, WithMaxDistance) failed: %v", fixture.sourceID, err)
		}
		benchmarkResultSink = result
	}
}

// BenchmarkDijkstra_InfEdgeThresholdWall measures Dijkstra on a deterministic corridor
// where the threshold-wall policy changes which edges are traversable.
//
// Implementation:
//   - Stage 1: Build the threshold-sensitive fixture before timing begins.
//   - Stage 2: Report allocations and topology-size proxy.
//   - Stage 3: Reset the timer.
//   - Stage 4: Repeatedly run Dijkstra with WithInfEdgeThreshold in the hot loop.
//
// Behavior highlights:
//   - This regime exercises a real wall-policy traversal branch.
//   - Some direct edges would otherwise be attractive but become impassable under the threshold.
//
// AI-Hints:
//   - Threshold-wall benchmarks should reflect an actual traversal-policy change, not just an extra if-statement.
func BenchmarkDijkstra_InfEdgeThresholdWall(b *testing.B) {
	fixture := buildBenchmarkThresholdWallCorridor(b, benchmarkWallSegmentCount)

	b.ReportAllocs()
	b.SetBytes(int64(fixture.vertexCount + fixture.edgeCount))
	b.ResetTimer()

	for index := 0; index < b.N; index++ {
		result, err := dijkstra.Dijkstra(
			fixture.graph,
			fixture.sourceID,
			dijkstra.WithInfEdgeThreshold(benchmarkWallThreshold),
		)
		if err != nil {
			b.Fatalf("Dijkstra(%q, WithInfEdgeThreshold) failed: %v", fixture.sourceID, err)
		}
		benchmarkResultSink = result
	}
}
