// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package flow_test provides shared fixtures and independent proof assertions.
// This file centralizes external test helpers without widening the production API.
package flow_test

import (
	"errors"
	"io"
	"math"
	"os"
	"slices"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/flow"
	"github.com/lvlath/go/matrix"
)

func mustNoError(t *testing.T, err error, op string) {
	t.Helper()
	if err != nil {
		t.Fatalf("%s: unexpected error: %v", op, err)
	}
}

func mustErrorIs(t *testing.T, err error, target error, op string) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Fatalf("%s: want errors.Is(err, %v); got %v", op, target, err)
	}
}

func mustEqualFloat(t *testing.T, got, want float64, op string) {
	t.Helper()
	if math.IsNaN(got) || math.Abs(got-want) > 1e-9 {
		// Both operands must be finite; Inf-Inf and NaN must never satisfy a finite comparison.
		if math.IsNaN(got) || math.IsInf(got, 0) || math.IsNaN(want) || math.IsInf(want, 0) || math.Abs(got-want) > flowTestFloatTolerance {
			t.Fatalf("%s: got=%g want=%g", op, got, want)
		}
	}
}

func mustEqualBool(t *testing.T, got, want bool, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got=%t want=%t", op, got, want)
	}
}

func mustGraph(t *testing.T, opts ...core.GraphOption) *core.Graph {
	t.Helper()

	g, err := core.NewGraph(opts...)
	mustNoError(t, err, "core.NewGraph")

	return g
}

func mustAddVertex(t *testing.T, g *core.Graph, id string) {
	t.Helper()

	err := g.AddVertex(id)
	mustNoError(t, err, "AddVertex("+id+")")
}

func mustAddEdge(t *testing.T, g *core.Graph, from, to string, capacity float64) {
	t.Helper()

	_, err := g.AddEdge(from, to, capacity)
	mustNoError(t, err, "AddEdge("+from+","+to+")")
}

func mustFlowValue(t *testing.T, result *flow.MaxFlowResult, want float64, op string) {
	t.Helper()

	if result == nil {
		t.Fatalf("%s: nil result", op)
	}
	mustEqualFloat(t, result.Value, want, op)
}

// mustNoAugmentingPath verifies that a residual graph has no positive residual path.
// It is a proof helper for maximum-flow optimality.
//
// Implementation:
//   - Stage 1: BFS from source in the published residual graph.
//   - Stage 2: Traverse only directed residual edges with Weight > epsilon.
//   - Stage 3: Fail if sink becomes reachable.
//
// Behavior highlights:
//   - Uses core.Neighbors only on the already-published directed residual graph.
//   - Does not infer undirected input semantics.
//
// Inputs:
//   - t: test handle.
//   - residual: directed weighted residual graph.
//   - source: source vertex ID.
//   - sink: sink vertex ID.
//   - epsilon: residual threshold.
//
// Returns:
//   - map[string]bool: independently discovered source-side residual vertices.
//
// Errors:
//   - Test failure on invalid terminals, core.Neighbors error, or reachable sink.
//
// Determinism:
//   - core.Neighbors provides deterministic edge ordering.
//
// Complexity:
//   - Time O(V + E_res) excluding core neighbor ordering; Space O(V + d_max).
//
// AI-Hints:
//   - Use this on result.Residual, not on the input graph.
func mustNoAugmentingPath(
	t *testing.T,
	residual *core.Graph,
	source string,
	sink string,
	epsilon float64,
) map[string]bool {
	t.Helper()

	if residual == nil {
		t.Fatalf("residual graph is nil")
	}

	if source == sink || !residual.HasVertex(source) || !residual.HasVertex(sink) {
		t.Fatalf("invalid residual terminals: source=%q sink=%q", source, sink)
	}

	seen := map[string]bool{source: true}
	queue := []string{source}

	for head := 0; head < len(queue); head++ {
		from := queue[head]

		edges, err := residual.Neighbors(from)
		mustNoError(t, err, "residual.Neighbors("+from+")")

		for _, edge := range edges {
			if edge.Weight <= epsilon || seen[edge.To] {
				continue
			}
			if edge.To == sink {
				t.Fatalf("residual graph still has positive path to sink via %s -> %s", from, sink)
			}

			seen[edge.To] = true
			queue = append(queue, edge.To)
		}
	}
	// Reuse this independent traversal to validate the complete published cut.
	return seen
}

// mustResidualDirectedWeighted verifies residual publication policy.
//
// Implementation:
//   - Stage 1: Reject nil result/residual.
//   - Stage 2: Check directed graph flag.
//   - Stage 3: Check weighted graph flag.
//
// Behavior highlights:
//   - The residual graph must not inherit undirected/unweighted input flags.
//
// Inputs:
//   - t: test handle.
//   - result: MaxFlowResult to verify.
//
// Errors:
//   - Test failure when residual policy is violated.
//
// Determinism:
//   - Pure graph-flag checks.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// AI-Hints:
//   - This helper protects the residual graph law after refactors.
func mustResidualDirectedWeighted(t *testing.T, result *flow.MaxFlowResult) {
	t.Helper()

	if result == nil {
		t.Fatalf("nil MaxFlowResult")
	}
	if result.Residual == nil {
		t.Fatalf("nil residual graph")
	}
	if !result.Residual.Directed() {
		t.Fatalf("residual graph must be directed")
	}
	if !result.Residual.Weighted() {
		t.Fatalf("residual graph must be weighted")
	}
}

// mustCutCapacityEqualsFlow verifies the max-flow/min-cut certificate.
// It compares MaxFlowResult.Value with the effective original cut capacity.
//
// Implementation:
//   - Stage 1: Compute effective cut capacity through cutCapacityFromResult.
//   - Stage 2: Compare it against result.Value using mustEqualFloat.
//
// Behavior highlights:
//   - Honors epsilon filtering.
//   - Works for directed and undirected input graph policies.
//
// Inputs:
//   - t: test handle.
//   - original: original weighted capacity graph.
//   - result: successful MaxFlowResult.
//   - epsilon: same threshold used by MaxFlow.
//   - op: diagnostic operation name.
//
// Errors:
//   - Test failure on nil inputs or mismatched theorem value.
//
// Determinism:
//   - Deterministic for fixed graph, result, and epsilon.
//
// Complexity:
//   - Time O(V_cut + E), Space O(V_cut).
//
// AI-Hints:
//   - Always pass the same epsilon used by the MaxFlow call.
func mustCutCapacityEqualsFlow(
	t *testing.T,
	original *core.Graph,
	result *flow.MaxFlowResult,
	epsilon float64,
	op string,
) {
	t.Helper()

	if original == nil {
		t.Fatalf("nil original graph")
	}
	if result == nil {
		t.Fatalf("nil MaxFlowResult")
	}

	sourceSide := make(map[string]bool, len(result.CutSourceSide))
	for _, vertexID := range result.CutSourceSide {
		sourceSide[vertexID] = true
	}

	capacity := 0.0
	for _, edge := range original.Edges() {
		if edge.From == edge.To {
			continue
		}
		if edge.Weight <= epsilon {
			continue
		}

		fromSourceSide := sourceSide[edge.From]
		toSourceSide := sourceSide[edge.To]

		if edge.Directed {
			if fromSourceSide && !toSourceSide {
				capacity += edge.Weight
			}
			continue
		}

		if fromSourceSide != toSourceSide {
			capacity += edge.Weight
		}
	}

	mustEqualFloat(t, capacity, result.Value, op)
}

// mustSuccessfulCertificate verifies the full success certificate for a max-flow run.
// It checks value, completion state, residual policy, residual no-path, and min-cut theorem.
//
// Implementation:
//   - Stage 1: Verify scalar max-flow value.
//   - Stage 2: Verify the run is complete rather than partial.
//   - Stage 3: Verify residual graph publication policy.
//   - Stage 4: Verify residual reachability and the complete ordered cut partition.
//   - Stage 5: Verify effective min-cut capacity equals flow value.
//
// Behavior highlights:
//   - This is the main proof helper for successful runs.
//   - It intentionally validates more than the scalar value.
//
// Inputs:
//   - t: test handle.
//   - original: original weighted capacity graph.
//   - result: MaxFlowResult returned by MaxFlow.
//   - source: source vertex ID.
//   - sink: sink vertex ID.
//   - want: expected flow value.
//   - epsilon: same threshold used by MaxFlow.
//   - op: diagnostic operation name.
//
// Errors:
//   - Test failure when any certificate component is invalid.
//
// Determinism:
//   - Stable for fixed graph, algorithm, epsilon, and residual traversal order.
//
// Complexity:
//   - Time O(V + E + E_res) excluding core enumeration sorting.
//   - Space O(V + E) including core edge enumeration.
//
// AI-Hints:
//   - Prefer this helper over tests that only compare result.Value.
func mustSuccessfulCertificate(
	t *testing.T,
	original *core.Graph,
	result *flow.MaxFlowResult,
	source string,
	sink string,
	want float64,
	epsilon float64,
	op string,
) {
	t.Helper()

	mustFlowValue(t, result, want, op+" value")
	mustEqualBool(t, result.Partial, false, op+" partial")
	mustResidualDirectedWeighted(t, result)
	reachable := mustNoAugmentingPath(t, result.Residual, source, sink, epsilon)
	mustCutPartition(t, original, result, source, sink, reachable)
	mustCutCapacityEqualsFlow(t, original, result, epsilon, op+" min-cut theorem")
}

// test helper
func residualCapacity(t *testing.T, residual *core.Graph, from string, to string) float64 {
	t.Helper()

	if residual == nil {
		t.Fatalf("residualCapacity(%s,%s): nil residual graph", from, to)
	}

	edges, err := residual.Neighbors(from)
	mustNoError(t, err, "residual.Neighbors("+from+")")

	total := 0.0
	for _, edge := range edges {
		if edge.To == to {
			total += edge.Weight
		}
	}

	return total
}

// Constants distinguish public default policy from assertion tolerance.
const (
	flowTestDefaultEpsilon = 1e-9
	flowTestFloatTolerance = 1e-9
	flowTestSource         = "S"
	flowTestSink           = "T"
	flowTestShortNode      = "A"
	flowTestLongNode       = "B"
	flowTestRelay          = "C"
	flowTestDeadEnd        = "D"
	flowTestIsolated       = "I"
	flowTestShortValue     = 3.0
	flowTestLongValue      = 5.0
	flowTestTotalValue     = flowTestShortValue + flowTestLongValue
	flowTestEventCount     = 2
	flowTestBackboneValue  = 27.0
)

// flowAlgorithmCase binds each canonical algorithm to its compatibility facade.
// It keeps all cross-algorithm assertions on one shared algorithm inventory.
type flowAlgorithmCase struct {
	name      string
	algorithm flow.Algorithm
	legacy    func(*core.Graph, string, string, flow.Options) (float64, *core.Graph, error)
}

// allAlgorithms returns every supported engine in stable test execution order.
// The legacy function and canonical selector must identify the same engine.
// Implementation:
//   - Stage 1: Pair each exported wrapper with its Algorithm constant.
//
// Behavior highlights:
//   - Every shared test executes against all three algorithms.
//
// Inputs: None.
// Returns: A fresh ordered slice of algorithm cases.
// Errors: None.
// Determinism: Dinic, Edmonds-Karp, Ford-Fulkerson.
// Complexity: O(1) time and space for the fixed engine inventory.
// Notes: Keep this inventory synchronized with the public dispatcher.
// AI-Hints: Do not omit a wrapper when introducing another algorithm.
func allAlgorithms() []flowAlgorithmCase {
	return []flowAlgorithmCase{
		{name: "Dinic", algorithm: flow.AlgorithmDinic, legacy: flow.Dinic},
		{name: "EdmondsKarp", algorithm: flow.AlgorithmEdmondsKarp, legacy: flow.EdmondsKarp},
		{name: "FordFulkerson", algorithm: flow.AlgorithmFordFulkerson, legacy: flow.FordFulkerson},
	}
}

// buildEnterpriseBackboneProofNetwork builds a layered capacity network.
// Its terminal links bound the optimum by 15 + 12 = 27 units.
// Implementation:
//   - Stage 1: Allocate a directed weighted graph.
//   - Stage 2: Add ingress, core, aggregation, and terminal links.
//
// Behavior highlights: Multiple competing routes must share the terminal cut.
// Inputs: t is the test handle used for fail-fast graph construction.
// Returns: A fresh capacity graph owned by the calling test.
// Errors: Any core construction error fails the test immediately.
// Determinism: The edge list has fixed insertion order.
// Complexity: O(V+E) expected graph construction time and space.
// Notes: The fixture uses exactly representable integer capacities.
// AI-Hints: Keep the known optimum independent of the algorithm under test.
func buildEnterpriseBackboneProofNetwork(t *testing.T) *core.Graph {
	t.Helper()

	g := mustGraph(t, core.WithDirected(true), core.WithWeighted())

	// Ingress capacity exceeds the terminal cut.
	mustAddEdge(t, g, "S", "EdgeA", 10)
	mustAddEdge(t, g, "S", "EdgeB", 15)
	mustAddEdge(t, g, "S", "EdgeC", 10)

	// Core links allow competing paths and rerouting between aggregation tiers.
	mustAddEdge(t, g, "EdgeA", "CoreD", 9)
	mustAddEdge(t, g, "EdgeA", "CoreE", 4)
	mustAddEdge(t, g, "EdgeB", "CoreD", 5)
	mustAddEdge(t, g, "EdgeB", "CoreE", 8)
	mustAddEdge(t, g, "EdgeB", "CoreF", 4)
	mustAddEdge(t, g, "EdgeC", "CoreE", 6)
	mustAddEdge(t, g, "EdgeC", "CoreF", 8)

	mustAddEdge(t, g, "CoreD", "AggG", 10)
	mustAddEdge(t, g, "CoreE", "AggG", 7)
	mustAddEdge(t, g, "CoreE", "AggH", 8)
	mustAddEdge(t, g, "CoreF", "AggH", 10)

	// These links form a cut with total capacity 27.
	mustAddEdge(t, g, "AggG", "T", 15)
	mustAddEdge(t, g, "AggH", "T", 12)

	return g
}

// buildRegionalUndirectedBackbone builds a bidirectional regional transport graph.
// The two links incident to Tbilisi form a cut of capacity 100.
// Implementation:
//   - Stage 1: Construct an undirected weighted graph.
//   - Stage 2: Insert the fixed regional capacity links.
//
// Behavior highlights: Reverse traversal must use both orientations of each link.
// Inputs: t is the test handle.
// Returns: A fresh graph with deterministic link inventory.
// Errors: Any graph setup error fails the test.
// Determinism: Fixed insertion order and stable core vertex/edge ordering.
// Complexity: O(V+E) expected time and space.
// Notes: No shared mutable fixture is retained across subtests.
// AI-Hints: Do not replace undirected links with duplicated core edges.
func buildRegionalUndirectedBackbone(t *testing.T) *core.Graph {
	t.Helper()

	g := mustGraph(t, core.WithDirected(false), core.WithWeighted())
	mustAddEdge(t, g, "Frankfurt", "Warsaw", 160)
	mustAddEdge(t, g, "Frankfurt", "Prague", 120)
	mustAddEdge(t, g, "Warsaw", "Kyiv", 90)
	mustAddEdge(t, g, "Prague", "Kyiv", 70)
	mustAddEdge(t, g, "Kyiv", "Tbilisi", 55)
	mustAddEdge(t, g, "Tbilisi", "Bucharest", 45)
	mustAddEdge(t, g, "Bucharest", "Prague", 40)
	mustAddEdge(t, g, "Bucharest", "Warsaw", 20)

	return g
}

// buildFlowObservationNetwork constructs two independent augmenting routes.
// Different path lengths distinguish BFS/level traversal from LIFO DFS.
// Implementation:
//   - Stage 1: Add a two-arc path with bottleneck 3.
//   - Stage 2: Add a three-arc path with bottleneck 5.
//   - Stage 3: Add a reachable dead end and an isolated vertex.
//
// Behavior highlights: The optimum is 8, reached by exactly two augmentations.
// Inputs: t is the test handle.
// Returns: A fresh directed capacity graph.
// Errors: Any graph setup error fails the test.
// Determinism: Vertex IDs fix the adjacency and stack ordering.
// Complexity: O(V+E) expected time and space.
// Notes: Surplus ingress capacities leave meaningful forward residual arcs.
// AI-Hints: Do not equalize the bottlenecks; that hides event ordering errors.
func buildFlowObservationNetwork(t *testing.T) *core.Graph {
	t.Helper()

	g := mustGraph(t, core.WithDirected(true), core.WithWeighted())
	mustAddEdge(t, g, flowTestSource, flowTestShortNode, 4)
	mustAddEdge(t, g, flowTestShortNode, flowTestSink, flowTestShortValue)
	mustAddEdge(t, g, flowTestSource, flowTestLongNode, 6)
	mustAddEdge(t, g, flowTestLongNode, flowTestRelay, flowTestLongValue)
	mustAddEdge(t, g, flowTestRelay, flowTestSink, flowTestLongValue)
	mustAddEdge(t, g, flowTestSource, flowTestDeadEnd, 2)
	mustAddVertex(t, g, flowTestIsolated)

	return g
}

// expectedFlowEvents derives the two augmentation witnesses for the fixed fixture.
// It does not call MaxFlow to obtain expected values.
// Implementation:
//   - Stage 1: Form the independent short and long path witnesses.
//   - Stage 2: Apply the documented BFS or LIFO traversal order.
//   - Stage 3: Assign one-based indexes and cumulative totals.
//
// Behavior highlights: Dinic publishes nil paths, not fabricated path witnesses.
// Inputs: algorithm identifies a supported engine.
// Returns: Two independent event values in expected publication order.
// Errors: None for the supported engine inventory.
// Determinism: Short-first for Dinic/EK; long-first for the current LIFO DFS.
// Complexity: O(1) time and space for this fixed fixture.
// Notes: This oracle relies only on the fixture and public traversal contract.
// AI-Hints: Do not derive golden events by recording another run of the same engine.
func expectedFlowEvents(algorithm flow.Algorithm) []flow.AugmentationEvent {
	short := flow.AugmentationEvent{
		Algorithm: algorithm,
		Path:      []string{flowTestSource, flowTestShortNode, flowTestSink},
		Delta:     flowTestShortValue,
	}
	long := flow.AugmentationEvent{
		Algorithm: algorithm,
		Path:      []string{flowTestSource, flowTestLongNode, flowTestRelay, flowTestSink},
		Delta:     flowTestLongValue,
	}
	// Dinic's observer contract does not expose a materialized augmenting path.
	if algorithm == flow.AlgorithmDinic {
		short.Path, long.Path = nil, nil
	}
	// The DFS stack processes the later lexical branch before the earlier one.
	if algorithm == flow.AlgorithmFordFulkerson {
		short, long = long, short
	}
	short.Index, short.Total = 1, short.Delta
	long.Index, long.Total = flowTestEventCount, flowTestTotalValue
	return []flow.AugmentationEvent{short, long}
}

// mustFlowEvents checks complete observer payloads rather than callback counts.
// Both the numeric sequence and the engine-specific path contract are validated.
// Implementation:
//   - Stage 1: Compare sequence lengths.
//   - Stage 2: Check algorithm, index, delta, total, and path for each event.
//
// Behavior highlights: Distinguishes a nil path from a non-nil empty path.
// Inputs: t is the test handle; got and want are ordered event sequences.
// Returns: None.
// Errors: Any differing field fails the test with its event index.
// Determinism: Compare events and paths in publication order.
// Complexity: O(K+P) time, O(1) auxiliary space for K events and P path entries.
// Notes: All fixture capacities are exactly representable.
// AI-Hints: Do not sort event paths or compare only the final total.
func mustFlowEvents(t *testing.T, got, want []flow.AugmentationEvent) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("augmentation count: got %d want %d", len(got), len(want))
	}
	for i, expected := range want {
		actual := got[i]
		if actual.Algorithm != expected.Algorithm || actual.Index != expected.Index ||
			actual.Delta != expected.Delta || actual.Total != expected.Total ||
			(actual.Path == nil) != (expected.Path == nil) || !slices.Equal(actual.Path, expected.Path) {
			t.Fatalf("augmentation[%d]: got %+v want %+v", i, actual, expected)
		}
	}
}

// mustEqualFlowGraphs verifies deterministic topology and capacity inventories.
// It compares public graph data without mutating the read-only Edge pointers.
// Implementation:
//   - Stage 1: Check non-nil graphs, directedness, weightedness, and vertices.
//   - Stage 2: Compare edge counts and complete Edge values in ID order.
//
// Behavior highlights: Detects lost isolated vertices, reverse arcs, or capacities.
// Inputs: t is the test handle; got and want are stable graph snapshots; op is a label.
// Returns: None.
// Errors: A structural mismatch fails the test.
// Determinism: Uses core.Vertices and core.Edges ordering.
// Complexity: O(V log V + E log E) time and O(V+E) enumeration space.
// Notes: Vertex metadata is outside this topology comparison.
// AI-Hints: Do not compare pointer identities as a substitute for edge values.
func mustEqualFlowGraphs(t *testing.T, got, want *core.Graph, op string) {
	t.Helper()
	if got == nil || want == nil {
		t.Fatalf("%s: nil graph: got=%p want=%p", op, got, want)
	}
	if got.Directed() != want.Directed() || got.Weighted() != want.Weighted() {
		t.Fatalf("%s: graph policy differs", op)
	}
	if gv, wv := got.Vertices(), want.Vertices(); !slices.Equal(gv, wv) {
		t.Fatalf("%s: vertices got=%v want=%v", op, gv, wv)
	}
	gotEdges, wantEdges := got.Edges(), want.Edges()
	if len(gotEdges) != len(wantEdges) {
		t.Fatalf("%s: edge count got=%d want=%d", op, len(gotEdges), len(wantEdges))
	}
	for i, expected := range wantEdges {
		actual := gotEdges[i]
		if actual == nil || expected == nil || *actual != *expected {
			t.Fatalf("%s: edge[%d] got=%+v want=%+v", op, i, actual, expected)
		}
	}
}

// mustCutPartition verifies the complete residual reachability certificate.
// A minimum-cut partition must be disjoint, exhaustive, ordered, and terminal-anchored.
// Implementation:
//   - Stage 1: Check source/sink metadata and residual vertex preservation.
//   - Stage 2: Derive both expected sides from independent BFS reachability.
//   - Stage 3: Compare the published sides in core vertex order.
//
// Behavior highlights: Rejects fabricated cut partitions even when their sum is correct.
// Inputs: original/result identify the solved network; reachable is the BFS visited set.
// Returns: None.
// Errors: Any missing vertex or mismatched partition fails the test.
// Determinism: Uses sorted original vertex IDs.
// Complexity: O(V log V) time including enumeration, O(V) space.
// Notes: Called only after a successful run with no residual source-sink path.
// AI-Hints: Do not infer cut membership from the returned cut capacity alone.
func mustCutPartition(t *testing.T, original *core.Graph, result *flow.MaxFlowResult, source, sink string, reachable map[string]bool) {
	t.Helper()
	if result.Source != source || result.Sink != sink || !reachable[source] || reachable[sink] {
		t.Fatalf("invalid terminal certificate: source=%q sink=%q reachable=%v", result.Source, result.Sink, reachable)
	}
	vertices := original.Vertices()
	if residualVertices := result.Residual.Vertices(); !slices.Equal(residualVertices, vertices) {
		t.Fatalf("residual vertex inventory: got=%v want=%v", residualVertices, vertices)
	}
	wantSource := make([]string, 0, len(vertices))
	wantSink := make([]string, 0, len(vertices))
	for _, vertex := range vertices {
		if reachable[vertex] {
			wantSource = append(wantSource, vertex)
		} else {
			wantSink = append(wantSink, vertex)
		}
	}
	if !slices.Equal(result.CutSourceSide, wantSource) || !slices.Equal(result.CutSinkSide, wantSink) {
		t.Fatalf("cut partition: got %v | %v, want %v | %v", result.CutSourceSide, result.CutSinkSide, wantSource, wantSink)
	}
}

// mustExactFlowBalance reconstructs the signed net flow from capacities and residuals.
// It independently verifies skew symmetry, bounds, and conservation for exact fixtures.
// Implementation:
//   - Stage 1: Aggregate effective original capacities by ordered endpoint pair.
//   - Stage 2: Aggregate positive published residual capacities over the same pairs.
//   - Stage 3: Evaluate f(u,v)=C(u,v)-R(u,v), including reverse-only arcs.
//   - Stage 4: Check per-vertex net outflow against the source/sink balance law.
//
// Behavior highlights: Supports parallel and antiparallel arcs without confusing gross and net flow.
// Inputs: original/residual are stable graph snapshots; value is the expected net flow;
// epsilon is the input filtering threshold used for the run.
// Returns: None.
// Errors: Any non-finite residual, bound violation, or conservation failure fails the test.
// Determinism: Visits pairs in their first-seen core edge order.
// Complexity: O(V log V + E log E + R log R) time including sorted core enumeration;
// O(V+E+R) auxiliary space, where R is the number of residual arcs.
// Notes: Use only exact integer/dyadic fixtures where residual clamping discards no positive remainder.
// AI-Hints: Never require exact conservation for a fixture that intentionally drops nonzero residuals.
func mustExactFlowBalance(t *testing.T, original, residual *core.Graph, source, sink string, value, epsilon float64) {
	t.Helper()
	type arc struct{ from, to string }
	capacity := make(map[arc]float64)
	remaining := make(map[arc]float64)
	seen := make(map[arc]bool)
	pairs := make([]arc, 0, original.EdgeCount()+residual.EdgeCount())
	// Keep the union sparse so missing reverse arcs are checked as zero capacities.
	remember := func(key arc) {
		if !seen[key] {
			seen[key] = true
			pairs = append(pairs, key)
		}
	}
	for _, edge := range original.Edges() {
		if edge.From == edge.To || edge.Weight <= epsilon {
			continue
		}
		key := arc{edge.From, edge.To}
		remember(key)
		capacity[key] += edge.Weight
		if !edge.Directed {
			reverse := arc{edge.To, edge.From}
			remember(reverse)
			capacity[reverse] += edge.Weight
		}
	}
	for _, edge := range residual.Edges() {
		if !edge.Directed || edge.From == edge.To || edge.Weight <= epsilon || math.IsNaN(edge.Weight) || math.IsInf(edge.Weight, 0) {
			t.Fatalf("invalid published residual edge: %+v", edge)
		}
		key := arc{edge.From, edge.To}
		remember(key)
		remaining[key] += edge.Weight
	}
	vertices := original.Vertices()
	balance := make(map[string]float64, len(vertices))
	for _, key := range pairs {
		reverse := arc{key.to, key.from}
		net := capacity[key] - remaining[key]
		reverseNet := capacity[reverse] - remaining[reverse]
		if net != -reverseNet || net > capacity[key] || net < -capacity[reverse] {
			t.Fatalf("net flow %s->%s: f=%g reverse=%g bounds=[%g,%g]", key.from, key.to, net, reverseNet, -capacity[reverse], capacity[key])
		}
		balance[key.from] += net
	}
	for _, vertex := range vertices {
		want := 0.0
		switch vertex {
		case source:
			want = value
		case sink:
			want = -value
		}
		if balance[vertex] != want {
			t.Fatalf("net outflow at %q: got=%g want=%g", vertex, balance[vertex], want)
		}
	}
}

// mustPartialFlowResult verifies every retained and cleared partial-result field.
// An interrupted augmentation sequence must not publish an optimality certificate.
// Implementation:
//   - Stage 1: Check the completed augmentation count, value, and run identity.
//   - Stage 2: Check Partial and require all certificate fields to remain nil.
//
// Behavior highlights: Distinguishes interruption after a commit from pre-run failure.
// Inputs: t is the test handle; result is the interrupted run; expected identifies the last event.
// Returns: None.
// Errors: A missing or misleading partial result fails the test.
// Determinism: Field comparisons only.
// Complexity: O(1) time and space.
// Notes: This helper follows newPartialResult's existing publication law.
// AI-Hints: Do not accept a successful-looking residual or min-cut on a partial result.
func mustPartialFlowResult(t *testing.T, result *flow.MaxFlowResult, expected flow.AugmentationEvent) {
	t.Helper()
	if result == nil {
		t.Fatal("interrupted run returned no partial result")
	}
	if !result.Partial || result.Algorithm != expected.Algorithm || result.Source != flowTestSource || result.Sink != flowTestSink || result.Augmentations != expected.Index || result.Value != expected.Total {
		t.Fatalf("partial result: got %+v want algorithm=%s, value=%g, count=%d", result, expected.Algorithm, expected.Total, expected.Index)
	}
	if result.Residual != nil || result.CutSourceSide != nil || result.CutSinkSide != nil {
		t.Fatal("partial result must not publish an optimality certificate")
	}
}

// captureFlowStdout records the legacy verbose stream without an unbounded pipe.
// A temporary file avoids deadlock when the callback emits more than a pipe buffer.
// Implementation:
//   - Stage 1: Create a test-owned temporary file and preserve os.Stdout.
//   - Stage 2: Execute the synchronous callback with stdout redirected.
//   - Stage 3: Restore stdout, rewind, and read the complete output.
//
// Behavior highlights: Restores stdout even if the callback fails the test.
// Inputs: t is a sequential test handle; run is a synchronous callback.
// Returns: The complete captured output string.
// Errors: File creation, seek, read, or close failures fail the test.
// Determinism: No concurrent stdout writers are permitted by the caller.
// Complexity: O(B) time and space for B bytes of captured output.
// Notes: This helper mutates process-wide stdout; do not use t.Parallel with it.
// AI-Hints: Do not use a pipe and wait to read until after a potentially large write.
func captureFlowStdout(t *testing.T, run func()) string {
	t.Helper()
	output, err := os.CreateTemp(t.TempDir(), "flow-verbose-*")
	mustNoError(t, err, "create stdout capture")
	previous := os.Stdout
	defer func() {
		os.Stdout = previous
		if err := output.Close(); err != nil {
			t.Errorf("close stdout capture: %v", err)
		}
	}()
	os.Stdout = output
	run()
	os.Stdout = previous
	_, err = output.Seek(0, io.SeekStart)
	mustNoError(t, err, "rewind stdout capture")
	data, err := io.ReadAll(output)
	mustNoError(t, err, "read stdout capture")
	return string(data)
}

// resetFlowOption deliberately resets a generic option target to its zero value.
// Go type inference permits assigning this function to the public flow.Option type.
// Implementation:
//   - Stage 1: Construct the zero value of the inferred target type.
//   - Stage 2: Assign it through the supplied pointer.
//
// Behavior highlights: Exercises final option validation without unsafe or private exports.
// Inputs: target is the option accumulator supplied by the public resolver.
// Returns: nil, making the resolver responsible for validating the resulting state.
// Errors: None under the resolver's non-nil accumulator contract.
// Determinism: Always writes the same zero value for a given type.
// Complexity: O(1) time and space for the fixed-size flow accumulator.
// Notes: This is an intentionally invalid external custom option, not an application example.
// AI-Hints: Do not export the private options type merely to test the final guard.
func resetFlowOption[T any](target *T) error {
	var zero T
	*target = zero
	return nil
}

// mustCapacityCells checks a dense capacity snapshot against independent ordered cells.
// The comparison includes zero cells, which detect unexpected reverse or loop entries.
// Implementation:
//   - Stage 1: Check matrix shape and exact vertex order.
//   - Stage 2: Read and compare every row-major cell.
//
// Behavior highlights: Exact integer/dyadic fixtures require exact float64 equality.
// Inputs: got/order are published artifacts; want/wantOrder are independent test oracles.
// Returns: None.
// Errors: Invalid shape, access failure, or any unequal cell fails the test.
// Determinism: Fixed row-major traversal order.
// Complexity: O(V^2) time and O(1) additional space.
// AI-Hints: Do not derive expected zero cells from the returned matrix.
func mustCapacityCells(t *testing.T, got *matrix.Dense, order []string, want [][]float64, wantOrder []string) {
	t.Helper()
	if got == nil || got.Rows() != len(wantOrder) || got.Cols() != len(wantOrder) || !slices.Equal(order, wantOrder) {
		t.Fatalf("capacity matrix shape/order: matrix=%p order=%v want=%v", got, order, wantOrder)
	}
	if len(want) != len(wantOrder) {
		t.Fatal("invalid expected matrix row count")
	}
	for row, values := range want {
		if len(values) != len(wantOrder) {
			t.Fatalf("invalid expected matrix row %d width", row)
		}
		for col, expected := range values {
			actual, err := got.At(row, col)
			mustNoError(t, err, "read capacity cell")
			if actual != expected {
				t.Fatalf("capacity[%s,%s]: got=%g want=%g", order[row], order[col], actual, expected)
			}
		}
	}
}
