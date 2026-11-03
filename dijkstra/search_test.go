// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dijkstra_test checks the target-directed public API against original graph edges.
// AI-Hints:
//   - Tests validate EdgeIDs and directedness, not only a list of vertex labels.
//   - Oracle distances are computed independently of all production search kernels.
//   - Do not use epsilon comparisons to excuse incorrect integer/dyadic shortest paths.
package dijkstra_test

import (
	"context"
	"errors"
	"fmt"
	"github.com/lvlath/go/dijkstra"
	"math"
	"math/rand"
	"testing"

	"github.com/lvlath/go/core"
)

// searchTestGraph builds a weighted graph and fails immediately on invalid setup.
// The fixture builder preserves explicit original edge IDs for witness checks.
func searchTestGraph(t *testing.T, options ...core.GraphOption) *core.Graph {
	t.Helper()
	g, err := core.NewGraph(append([]core.GraphOption{core.WithWeighted()}, options...)...)
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	return g
}

// searchTestEdge adds one fixture edge with its chosen identity and policy.
// Every fixture error is fatal rather than silently changing its topology.
func searchTestEdge(t *testing.T, g *core.Graph, from, to, id string, weight float64, opts ...core.EdgeOption) {
	t.Helper()
	args := append([]core.EdgeOption{core.WithID(id)}, opts...)
	if _, err := g.AddEdge(from, to, weight, args...); err != nil {
		t.Fatalf("AddEdge(%q,%q,%q): %v", from, to, id, err)
	}
}

// searchTestWitness checks each original edge, direction and total fixture cost.
// It uses the immutable fixture graph, not the production search adapter.
func searchTestWitness(t *testing.T, g *core.Graph, result *dijkstra.PathResult) {
	t.Helper()
	if result == nil {
		t.Fatal("nil dijkstra.PathResult")
	}
	if !result.Reachable {
		if !math.IsInf(result.Distance, 1) || result.Vertices != nil || result.EdgeIDs != nil {
			t.Fatalf("invalid unreachable publication: %+v", result)
		}
		return
	}
	if len(result.Vertices) != len(result.EdgeIDs)+1 || result.Vertices[0] != result.SourceID ||
		result.Vertices[len(result.Vertices)-1] != result.TargetID {
		t.Fatalf("invalid witness endpoints/length: %+v", result)
	}
	cost := float64(0)
	for index, edgeID := range result.EdgeIDs {
		edge, err := g.GetEdge(edgeID)
		if err != nil {
			t.Fatalf("original edge %q missing: %v", edgeID, err)
		}
		from, to := result.Vertices[index], result.Vertices[index+1]
		if !(edge.From == from && edge.To == to) && (edge.Directed || edge.From != to || edge.To != from) {
			t.Fatalf("invalid orientation on edge %q: %q->%q, original=%+v", edgeID, from, to, edge)
		}
		cost += edge.Weight
	}
	if cost != result.Distance {
		t.Fatalf("witness original cost %g != published %g", cost, result.Distance)
	}
}

// TestSearch_ParallelOriginalIdentity verifies strict relaxation and edge ownership.
// Both parallel edges share endpoints but only the faster ID is a valid optimum witness.
func TestSearch_ParallelOriginalIdentity(t *testing.T) {
	g := searchTestGraph(t, core.WithDirected(true), core.WithMultiEdges())
	searchTestEdge(t, g, "S", "T", "slow", 7)
	searchTestEdge(t, g, "S", "T", "fast", 3)
	got, err := dijkstra.Search(g, "S", "T")
	if err != nil {
		t.Fatal(err)
	}
	searchTestWitness(t, g, got)
	if got.Distance != 3 || got.Method != dijkstra.SearchDijkstra || len(got.EdgeIDs) != 1 || got.EdgeIDs[0] != "fast" {
		t.Fatalf("wrong parallel edge selection: %+v", got)
	}
	vertices, edgeIDs, err := got.Path()
	if err != nil || len(vertices) != 2 || vertices[0] != "S" || vertices[1] != "T" || len(edgeIDs) != 1 || edgeIDs[0] != "fast" {
		t.Fatalf("Path: %v, %v, %v", vertices, edgeIDs, err)
	}

	// Once detached, neither normal core mutations nor mutations of a helper's
	// returned slices may change the original result.
	cloned := got.Clone()
	vertices[0] = "changed"
	edgeIDs[0] = "changed"
	if err := g.RemoveEdge("fast"); err != nil {
		t.Fatal(err)
	}
	if got.Vertices[0] != "S" || got.EdgeIDs[0] != "fast" {
		t.Fatalf("result aliasing after mutation: %+v", got)
	}
	cloned.Vertices[0] = "elsewhere"
	if got.Vertices[0] != "S" {
		t.Fatal("Clone shares the vertex slice")
	}
	got.EdgeIDs[0] = "corrupt"
	if cloned.EdgeIDs[0] != "fast" {
		t.Fatal("Clone shares the edge slice")
	}
}

// TestSearch_UndirectedReverseAndMixedPolicy verifies per-edge orientation.
// Directed incoming edges do not become traversable in reverse on mixed graphs.
func TestSearch_UndirectedReverseAndMixedPolicy(t *testing.T) {
	g := searchTestGraph(t, core.WithDirected(false))
	searchTestEdge(t, g, "A", "B", "undirected", 4)
	got, err := dijkstra.Search(g, "B", "A")
	if err != nil {
		t.Fatal(err)
	}
	searchTestWitness(t, g, got)
	if got.Distance != 4 || got.EdgeIDs[0] != "undirected" {
		t.Fatalf("reverse endpoint failure: %+v", got)
	}

	mixed, err := core.NewMixedGraph(core.WithWeighted(), core.WithDirected(true))
	if err != nil {
		t.Fatal(err)
	}
	searchTestEdge(t, mixed, "S", "A", "oneway", 2)
	searchTestEdge(t, mixed, "A", "B", "shared", 0, core.WithEdgeDirected(false))
	searchTestEdge(t, mixed, "B", "T", "finish", 3)
	forward, err := dijkstra.Search(mixed, "S", "T")
	if err != nil {
		t.Fatal(err)
	}
	searchTestWitness(t, mixed, forward)
	if forward.Distance != 5 || len(forward.EdgeIDs) != 3 {
		t.Fatalf("mixed path unexpected: %+v", forward)
	}
	backward, err := dijkstra.Search(mixed, "T", "S")
	if err != nil {
		t.Fatal(err)
	}
	searchTestWitness(t, mixed, backward)
	if backward.Reachable {
		t.Fatalf("directed edges incorrectly reversed: %+v", backward)
	}
}

// TestSearch_SettledTargetNotFirstEnqueue checks early-stop and stale heap entries.
// Target discovery at cost 10 must not override the later optimal cost 3.
func TestSearch_SettledTargetNotFirstEnqueue(t *testing.T) {
	g := searchTestGraph(t, core.WithDirected(true))
	searchTestEdge(t, g, "S", "T", "direct", 10)
	searchTestEdge(t, g, "S", "A", "slowA", 10)
	searchTestEdge(t, g, "S", "B", "first", 1)
	searchTestEdge(t, g, "B", "A", "improve", 1)
	searchTestEdge(t, g, "A", "T", "last", 1)
	got, err := dijkstra.Search(g, "S", "T")
	if err != nil {
		t.Fatal(err)
	}
	searchTestWitness(t, g, got)
	if got.Distance != 3 || len(got.EdgeIDs) != 3 || got.EdgeIDs[0] != "first" || got.EdgeIDs[1] != "improve" || got.EdgeIDs[2] != "last" {
		t.Fatalf("early target enqueue or stale-label failure: %+v", got)
	}
}

// TestSearch_ZeroCycleAndSelf validates strict improvements and final pre-scan.
// A zero cycle cannot install a predecessor for the source or loop forever.
func TestSearch_ZeroCycleAndSelf(t *testing.T) {
	g := searchTestGraph(t, core.WithDirected(true), core.WithLoops())
	searchTestEdge(t, g, "S", "A", "start", 0)
	searchTestEdge(t, g, "A", "S", "back", 0)
	searchTestEdge(t, g, "A", "T", "finish", 1)
	searchTestEdge(t, g, "S", "S", "loop", 0)
	got, err := dijkstra.Search(g, "S", "T")
	if err != nil {
		t.Fatal(err)
	}
	searchTestWitness(t, g, got)
	if got.Distance != 1 || len(got.EdgeIDs) != 2 {
		t.Fatalf("zero-cycle witness: %+v", got)
	}
	self, err := dijkstra.Search(g, "S", "S", dijkstra.WithSearchMaxDistance(0))
	if err != nil {
		t.Fatal(err)
	}
	searchTestWitness(t, g, self)
	if self.Distance != 0 || !self.Reachable || len(self.Vertices) != 1 || self.Vertices[0] != "S" || self.EdgeIDs != nil {
		t.Fatalf("self witness: %+v", self)
	}
}

// TestSearch_CutoffPrecisionAndOverflow verifies exact objective and error separation.
// The new target cutoff must not borrow legacy SSSP's rounded-candidate semantics.
func TestSearch_CutoffPrecisionAndOverflow(t *testing.T) {
	g := searchTestGraph(t, core.WithDirected(true))
	first, second, limit := 0.9, 0.1, 1.0
	searchTestEdge(t, g, "S", "A", "fraction0", first)
	searchTestEdge(t, g, "A", "T", "fraction1", second)
	blocked, err := dijkstra.Search(g, "S", "T", dijkstra.WithSearchMaxDistance(limit))
	if err != nil {
		t.Fatal(err)
	}
	searchTestWitness(t, g, blocked)
	if blocked.Reachable || blocked.Distance != math.Inf(1) {
		t.Fatalf("exact cutoff wrongly admitted rounded sum: %+v", blocked)
	}
	for _, bound := range []float64{math.Inf(1), 2} {
		got, searchErr := dijkstra.Search(g, "S", "T", dijkstra.WithSearchMaxDistance(bound))
		if got != nil || !errors.Is(searchErr, dijkstra.ErrInexactArithmetic) {
			t.Fatalf("inexact admitted sum: result=%+v, err=%v", got, searchErr)
		}
	}

	bigGraph := searchTestGraph(t, core.WithDirected(true))
	large := float64(math.MaxFloat64)
	searchTestEdge(t, bigGraph, "S", "A", "max", large)
	searchTestEdge(t, bigGraph, "A", "T", "over", large)
	cut, err := dijkstra.Search(bigGraph, "S", "T", dijkstra.WithSearchMaxDistance(large))
	if err != nil || cut == nil || cut.Reachable {
		t.Fatalf("overflowing out-of-cutoff candidate: result=%+v, err=%v", cut, err)
	}
	noBound, err := dijkstra.Search(bigGraph, "S", "T")
	if noBound != nil || !errors.Is(err, dijkstra.ErrDistanceOverflow) {
		t.Fatalf("unbounded overflow not reported: result=%+v, err=%v", noBound, err)
	}
}

// TestSearch_ValidationAndCancellation defends error precedence and no partial results.
// Callback failures and scalar violations must not be corrected by later setters.
func TestSearch_ValidationAndCancellation(t *testing.T) {
	var nilGraph *core.Graph
	if got, err := dijkstra.Search(nilGraph, "", ""); got != nil || !errors.Is(err, dijkstra.ErrNilGraph) {
		t.Fatalf("nil graph precedence: %v", err)
	}
	g := searchTestGraph(t, core.WithDirected(true))
	searchTestEdge(t, g, "S", "T", "edge", 2)
	if got, err := dijkstra.Search(g, "", ""); got != nil || !errors.Is(err, dijkstra.ErrEmptySourceID) {
		t.Fatalf("empty source precedence: %v", err)
	}
	if got, err := dijkstra.Search(g, "S", ""); got != nil || !errors.Is(err, dijkstra.ErrEmptyTargetID) {
		t.Fatalf("empty target precedence: %v", err)
	}
	if got, err := dijkstra.Search(g, "S", "unknown"); got != nil || !errors.Is(err, dijkstra.ErrTargetNotFound) {
		t.Fatalf("unknown target: %v", err)
	}
	for _, option := range []dijkstra.SearchOption{
		nil,
		dijkstra.WithSearchContext(nil),
		dijkstra.WithSearchMethod(dijkstra.SearchMethod(123)),
		dijkstra.WithSearchMaxDistance(-1),
		dijkstra.WithSearchInfEdgeThreshold(0),
	} {
		if got, err := dijkstra.Search(g, "S", "T", option); got != nil || err == nil {
			t.Fatalf("invalid option accepted: result=%+v", got)
		}
	}
	customCause := errors.New("options callback cause")
	badOption := dijkstra.SearchOption(func(*dijkstra.SearchOptions) error { return customCause })
	if got, err := dijkstra.Search(g, "S", "T", badOption); got != nil || !errors.Is(err, customCause) {
		t.Fatalf("callback error not preserved: result=%+v, err=%v", got, err)
	}
	setInvalid := dijkstra.SearchOption(func(config *dijkstra.SearchOptions) error {
		config.MaxDistance = -1
		return nil
	})
	if got, err := dijkstra.Search(g, "S", "T", setInvalid, dijkstra.WithSearchMaxDistance(10)); got != nil || !errors.Is(err, dijkstra.ErrBadMaxDistance) {
		t.Fatalf("later option repaired invalid scalar: result=%+v, err=%v", got, err)
	}
	ctx, cancel := context.WithCancelCause(context.Background())
	stopCause := errors.New("stop requested")
	cancel(stopCause)
	if got, err := dijkstra.Search(g, "S", "T", dijkstra.WithSearchContext(ctx)); got != nil ||
		!errors.Is(err, context.Canceled) || !errors.Is(err, stopCause) {
		t.Fatalf("cancel causes: result=%+v err=%v", got, err)
	}
}

// TestSearch_TraversalProjectionAndZeroCutoff validates policy projection and
// exact zero-cost reachability without transferring legacy TrackPaths semantics.
func TestSearch_TraversalProjectionAndZeroCutoff(t *testing.T) {
	g := searchTestGraph(t, core.WithDirected(true))
	searchTestEdge(t, g, "S", "A", "zero", 0)
	searchTestEdge(t, g, "A", "T", "positive", 1)
	policy := dijkstra.DefaultOptions()
	policy.TrackPaths = false
	policy.MaxDistance = 0
	policy.InfEdgeThreshold = 2
	got, err := dijkstra.Search(g, "S", "A", dijkstra.WithTraversalPolicy(policy))
	if err != nil {
		t.Fatal(err)
	}
	searchTestWitness(t, g, got)
	if !got.Reachable || got.Distance != 0 || got.MaxDistance != 0 ||
		got.InfEdgeThreshold != 2 || len(got.EdgeIDs) != 1 || got.EdgeIDs[0] != "zero" {
		t.Fatalf("zero cutoff or policy projection wrong: %+v", got)
	}
	blocked, err := dijkstra.Search(g, "S", "T", dijkstra.WithTraversalPolicy(policy))
	if err != nil || blocked == nil || blocked.Reachable {
		t.Fatalf("cutoff did not exclude positive extension: %+v, err=%v", blocked, err)
	}
	// Caller options are value inputs. Updating them after dijkstra.Search cannot alter
	// the published policy or route.
	policy.MaxDistance = 100
	if got.MaxDistance != 0 {
		t.Fatal("dijkstra.PathResult aliases mutable traversal options")
	}
}

// TestSearch_WallAndInputValidation checks wall policy and complete input scanning.
// Even a blocked bad edge is rejected before any shortest-path computation.
func TestSearch_WallAndInputValidation(t *testing.T) {
	g := searchTestGraph(t, core.WithDirected(true))
	searchTestEdge(t, g, "S", "T", "slow", 3)
	blocked, err := dijkstra.Search(g, "S", "T", dijkstra.WithSearchInfEdgeThreshold(3))
	if err != nil || blocked == nil || blocked.Reachable {
		t.Fatalf("wall should make target unreachable: %+v, %v", blocked, err)
	}

	// Deliberate single-threaded corruption of a live alias exercises the
	// defensive pre-scan. Concurrent unsynchronized alias mutation is forbidden.
	edge, err := g.GetEdge("slow")
	if err != nil {
		t.Fatal(err)
	}
	edge.Weight = math.NaN()
	defer func() { edge.Weight = 3 }()
	if got, searchErr := dijkstra.Search(g, "S", "S", dijkstra.WithSearchInfEdgeThreshold(3)); got != nil || !errors.Is(searchErr, dijkstra.ErrInvalidWeight) {
		t.Fatalf("self or wall masked invalid input: result=%+v, err=%v", got, searchErr)
	}
}

// TestSearch_PathReceiverStructuralIntegrity protects copy, nil and corruption laws.
// The public helper must terminate on caller-modified, structurally invalid fields.
func TestSearch_PathReceiverStructuralIntegrity(t *testing.T) {
	var absent *dijkstra.PathResult
	if !absent.IsNil() || absent.Clone() != nil {
		t.Fatal("nil result helper contract")
	}
	if _, _, err := absent.Path(); !errors.Is(err, dijkstra.ErrNilResult) {
		t.Fatalf("nil Path: %v", err)
	}
	g := searchTestGraph(t, core.WithDirected(true))
	searchTestEdge(t, g, "S", "T", "a", 1)
	got, err := dijkstra.Search(g, "S", "T")
	if err != nil {
		t.Fatal(err)
	}
	got.Vertices = []string{"S", "S", "T"}
	got.EdgeIDs = []string{"a", "a"}
	if _, _, err := got.Path(); !errors.Is(err, dijkstra.ErrNoPath) {
		t.Fatalf("cyclic caller witness: %v", err)
	}
	unreachable := &dijkstra.PathResult{Reachable: false, Distance: math.Inf(1)}
	if _, _, err := unreachable.Path(); !errors.Is(err, dijkstra.ErrNoPath) {
		t.Fatalf("unreachable Path: %v", err)
	}
	clone := unreachable.Clone()
	if clone == nil || clone.Vertices != nil || clone.EdgeIDs != nil {
		t.Fatalf("nil slices not preserved: %+v", clone)
	}
}

// oracleArc holds one original test graph relation independent of the search adapter.
// Directedness and parallel identities remain explicit in the oracle's own input.
type oracleArc struct {
	from, to int
	weight   int64
	directed bool
}

// oracleDistances computes exact small bounded all-pairs distances by Floyd-Warshall.
// Integer weights are sufficiently small that every sum fits int64 exactly.
func oracleDistances(vertices int, edges []oracleArc) [][]int64 {
	const infinity int64 = 1 << 55
	result := make([][]int64, vertices)
	for from := range result {
		result[from] = make([]int64, vertices)
		for to := range result[from] {
			result[from][to] = infinity
		}
		result[from][from] = 0
	}
	for _, edge := range edges {
		if edge.weight < result[edge.from][edge.to] {
			result[edge.from][edge.to] = edge.weight
		}
		if !edge.directed && edge.weight < result[edge.to][edge.from] {
			result[edge.to][edge.from] = edge.weight
		}
	}
	for via := 0; via < vertices; via++ {
		for from := 0; from < vertices; from++ {
			for to := 0; to < vertices; to++ {
				if result[from][via] == infinity || result[via][to] == infinity {
					continue
				}
				candidate := result[from][via] + result[via][to]
				if candidate < result[from][to] {
					result[from][to] = candidate
				}
			}
		}
	}
	return result
}

// TestSearch_BoundedIndependentOracle checks mixed, directed and undirected multigraphs.
// A separate integer Floyd-Warshall oracle proves costs; edge-by-edge validation proves witness.
func TestSearch_BoundedIndependentOracle(t *testing.T) {
	const (
		seed      int64 = 20261030
		caseCount       = 48
		maxWeight int64 = 9
		infinity  int64 = 1 << 55
	)
	rng := rand.New(rand.NewSource(seed))
	for caseIndex := 0; caseIndex < caseCount; caseIndex++ {
		vertexCount := 2 + rng.Intn(5)
		g, err := core.NewMixedGraph(core.WithWeighted(), core.WithDirected(true), core.WithMultiEdges(), core.WithLoops())
		if err != nil {
			t.Fatal(err)
		}
		ids := make([]string, vertexCount)
		for vertex := range ids {
			ids[vertex] = fmt.Sprintf("v%d", vertex)
			if err := g.AddVertex(ids[vertex]); err != nil {
				t.Fatal(err)
			}
		}
		arcs := make([]oracleArc, 0)
		for edgeIndex := 0; edgeIndex < vertexCount*3; edgeIndex++ {
			from, to := rng.Intn(vertexCount), rng.Intn(vertexCount)
			weight := int64(rng.Intn(int(maxWeight)))
			directed := rng.Intn(2) == 0
			id := fmt.Sprintf("case%d_edge%03d", caseIndex, edgeIndex)
			searchTestEdge(t, g, ids[from], ids[to], id, float64(weight), core.WithEdgeDirected(directed))
			arcs = append(arcs, oracleArc{from: from, to: to, weight: weight, directed: directed})
		}
		oracle := oracleDistances(vertexCount, arcs)
		for source := 0; source < vertexCount; source++ {
			for target := 0; target < vertexCount; target++ {
				got, err := dijkstra.Search(g, ids[source], ids[target])
				if err != nil {
					t.Fatalf("case=%d %s->%s: %v", caseIndex, ids[source], ids[target], err)
				}
				searchTestWitness(t, g, got)
				expected := oracle[source][target]
				if expected == infinity {
					if got.Reachable || !math.IsInf(got.Distance, 1) {
						t.Fatalf("case=%d %s->%s: expected unreachable, got %+v", caseIndex, ids[source], ids[target], got)
					}
				} else if !got.Reachable || got.Distance != float64(expected) {
					t.Fatalf("case=%d %s->%s: expected %d, got %+v", caseIndex, ids[source], ids[target], expected, got)
				}
			}
		}
	}
}
