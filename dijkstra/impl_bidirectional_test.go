// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dijkstra_test tests certified two-frontier routing and original edge witnesses.
// These tests use the public Search/Bidirectional result surface and test-only numeric probes.
// AI-Hints:
//   - The first meeting must be observed before claiming it is not a stopping bound.
//   - Verify exact original costs and directed edge IDs, never only vertex sequences.
package dijkstra_test

import (
	"context"
	"errors"
	"fmt"
	"github.com/lvlath/go/dijkstra"
	"math"
	"math/big"
	"math/rand"
	"slices"
	"testing"

	"github.com/lvlath/go/core"
)

// TestBidirectional_FirstMeetingIsNotStoppingCertificate rejects premature stopping.
// Its 101-cost first crossing must not hide the observable 52-cost optimum.
func TestBidirectional_FirstMeetingIsNotStoppingCertificate(t *testing.T) {
	graph := searchTestGraph(t, core.WithDirected(true))
	searchTestEdge(t, graph, "S", "A", "startA", 1)
	searchTestEdge(t, graph, "A", "T", "finishA", 100)
	searchTestEdge(t, graph, "S", "B", "startB", 50)
	searchTestEdge(t, graph, "B", "T", "finishB", 2)

	// Observe the actual production frontier transitions via a test-binary
	// bridge, retaining the original intermediate regression invariants.
	trace, err := dijkstra.SearchTestProbeFirstMeeting(graph, "S", "T")
	if err != nil {
		t.Fatal(err)
	}
	if !trace.ForwardRootCertified || !trace.BackwardRootCertified ||
		!trace.FirstWitness || trace.FirstCost != 101 || trace.ForwardMinimum != 50 ||
		trace.BackwardMinimum != 2 || trace.LowerBoundStops || trace.FinalCost != 52 {
		t.Fatalf("bad intermediate first-crossing or stopping certificate: %+v", trace)
	}

	// The cheaper first expansion through A yields an initial complete
	// crossing at 101, but a sound stopping rule must continue to the
	// 52-cost S->B->T route. This checks observable optimality without
	// depending on private frontier settlement order.
	result, err := dijkstra.Bidirectional(graph, "S", "T")
	if err != nil {
		t.Fatal(err)
	}
	searchTestWitness(t, graph, result)
	if result.Method != dijkstra.SearchBidirectional || result.Distance != 52 ||
		!slices.Equal(result.EdgeIDs, []string{"startB", "finishB"}) {
		t.Fatalf("first crossing was incorrectly treated as optimal: %+v", result)
	}
}

// TestBidirectional_RootCertificatesBeforePop checks a direct source-target edge.
// No longer path or unrelated reverse-directed relation may replace its witness.
func TestBidirectional_RootCertificatesBeforePop(t *testing.T) {
	graph := searchTestGraph(t, core.WithDirected(true))
	searchTestEdge(t, graph, "S", "T", "direct", 3)
	settled, witnessed, firstCost, err := dijkstra.SearchTestProbeRootJoin(graph, "S", "T")
	if err != nil || settled || !witnessed || firstCost != 3 {
		t.Fatalf("root certificate not recognized before backward settlement: settled=%t witnessed=%t cost=%g err=%v",
			settled, witnessed, firstCost, err)
	}
	result, err := dijkstra.Bidirectional(graph, "S", "T")
	if err != nil {
		t.Fatal(err)
	}
	searchTestWitness(t, graph, result)
	if result.Distance != 3 || !slices.Equal(result.EdgeIDs, []string{"direct"}) {
		t.Fatalf("root-to-root crossing not recognized: %+v", result)
	}
}

// TestBidirectional_UsesIncomingDirectedArcs protects the reverse traversal law.
// An outgoing arc from T to A must not fabricate a path from S through A to T.
func TestBidirectional_UsesIncomingDirectedArcs(t *testing.T) {
	g := searchTestGraph(t, core.WithDirected(true))
	searchTestEdge(t, g, "S", "A", "start", 1)
	searchTestEdge(t, g, "T", "A", "reverseTrap", 1)
	bad, err := dijkstra.Bidirectional(g, "S", "T")
	if err != nil {
		t.Fatal(err)
	}
	searchTestWitness(t, g, bad)
	if bad.Reachable {
		t.Fatalf("reverse trap incorrectly reached target: %+v", bad)
	}
	searchTestEdge(t, g, "A", "T", "realIncoming", 2)
	good, err := dijkstra.Bidirectional(g, "S", "T")
	if err != nil {
		t.Fatal(err)
	}
	searchTestWitness(t, g, good)
	if !good.Reachable || good.Distance != 3 || len(good.EdgeIDs) != 2 ||
		good.EdgeIDs[0] != "start" || good.EdgeIDs[1] != "realIncoming" {
		t.Fatalf("failed directed incoming route: %+v", good)
	}
	// The reverse query is a different relation. It may not use A->T backward.
	reverse, err := dijkstra.Bidirectional(g, "T", "S")
	if err != nil || reverse == nil || reverse.Reachable {
		t.Fatalf("invalid reversed directed route: %+v, err=%v", reverse, err)
	}
}

// TestBidirectional_ParallelAndUndirectedWitness protects original EdgeIDs for
// parallel channels and both legal orientations of an undirected connection.
func TestBidirectional_ParallelAndUndirectedWitness(t *testing.T) {
	multi := searchTestGraph(t, core.WithDirected(true), core.WithMultiEdges())
	searchTestEdge(t, multi, "S", "T", "slow", 7)
	searchTestEdge(t, multi, "S", "T", "fast", 3)
	got, err := dijkstra.Bidirectional(multi, "S", "T")
	if err != nil {
		t.Fatal(err)
	}
	searchTestWitness(t, multi, got)
	if got.Distance != 3 || len(got.EdgeIDs) != 1 || got.EdgeIDs[0] != "fast" {
		t.Fatalf("parallel channel identity lost: %+v", got)
	}
	copied := got.Clone()
	got.Vertices[0] = "changed"
	got.EdgeIDs[0] = "changed"
	if copied.Vertices[0] != "S" || copied.EdgeIDs[0] != "fast" {
		t.Fatalf("Clone aliases published witness: %+v", copied)
	}

	undirected := searchTestGraph(t, core.WithDirected(false))
	searchTestEdge(t, undirected, "A", "B", "shared", 4)
	reverse, err := dijkstra.Bidirectional(undirected, "B", "A")
	if err != nil {
		t.Fatal(err)
	}
	searchTestWitness(t, undirected, reverse)
	if reverse.Distance != 4 || len(reverse.EdgeIDs) != 1 || reverse.EdgeIDs[0] != "shared" {
		t.Fatalf("undirected orientation/identity lost: %+v", reverse)
	}
}

// TestBidirectional_ZeroPlateauAndExhaustion verifies finite predecessor chains,
// zero cutoff reachability, self paths and a disconnected direction's exhaustion.
func TestBidirectional_ZeroPlateauAndExhaustion(t *testing.T) {
	g := searchTestGraph(t, core.WithDirected(true), core.WithLoops())
	searchTestEdge(t, g, "S", "A", "zeroStart", 0)
	searchTestEdge(t, g, "A", "S", "zeroBack", 0)
	searchTestEdge(t, g, "A", "T", "positive", 1)
	searchTestEdge(t, g, "A", "A", "selfLoop", 0)
	got, err := dijkstra.Bidirectional(g, "S", "T")
	if err != nil {
		t.Fatal(err)
	}
	searchTestWitness(t, g, got)
	if !got.Reachable || got.Distance != 1 || len(got.EdgeIDs) != 2 {
		t.Fatalf("zero-cycle route: %+v", got)
	}
	zero, err := dijkstra.Bidirectional(g, "S", "A", dijkstra.WithSearchMaxDistance(0))
	if err != nil {
		t.Fatal(err)
	}
	searchTestWitness(t, g, zero)
	if !zero.Reachable || zero.Distance != 0 || len(zero.EdgeIDs) != 1 {
		t.Fatalf("zero cutoff should admit a free route: %+v", zero)
	}
	self, err := dijkstra.Bidirectional(g, "S", "S", dijkstra.WithSearchMaxDistance(0))
	if err != nil {
		t.Fatal(err)
	}
	searchTestWitness(t, g, self)
	if self.Distance != 0 || len(self.Vertices) != 1 || self.EdgeIDs != nil {
		t.Fatalf("self route: %+v", self)
	}

	disconnected := searchTestGraph(t, core.WithDirected(true))
	searchTestEdge(t, disconnected, "S", "A", "sourceOnly", 1)
	searchTestEdge(t, disconnected, "B", "T", "targetOnly", 1)
	unreachable, err := dijkstra.Bidirectional(disconnected, "S", "T")
	if err != nil {
		t.Fatal(err)
	}
	searchTestWitness(t, disconnected, unreachable)
	if unreachable.Reachable || !math.IsInf(unreachable.Distance, 1) {
		t.Fatalf("queue exhaustion fabricated a route: %+v", unreachable)
	}
}

// TestBidirectional_ExactCutoffAndOverflow tests the original-cost objective
// and distinguishes out-of-policy overflow from an admitted unbounded overflow.
func TestBidirectional_ExactCutoffAndOverflow(t *testing.T) {
	fractional := searchTestGraph(t, core.WithDirected(true))
	first, second, cutoff := 0.9, 0.1, 1.0
	searchTestEdge(t, fractional, "S", "A", "first", first)
	searchTestEdge(t, fractional, "A", "T", "second", second)
	outOfCutoff, err := dijkstra.Bidirectional(fractional, "S", "T", dijkstra.WithSearchMaxDistance(cutoff))
	if err != nil {
		t.Fatal(err)
	}
	if outOfCutoff.Reachable || !math.IsInf(outOfCutoff.Distance, 1) {
		t.Fatalf("exact sum above finite cutoff was admitted: %+v", outOfCutoff)
	}
	if got, searchErr := dijkstra.Bidirectional(fractional, "S", "T"); got != nil || !errors.Is(searchErr, dijkstra.ErrInexactArithmetic) {
		t.Fatalf("inexact admitted path must fail: result=%+v err=%v", got, searchErr)
	}

	large := searchTestGraph(t, core.WithDirected(true))
	maximum := float64(math.MaxFloat64)
	searchTestEdge(t, large, "S", "A", "max0", maximum)
	searchTestEdge(t, large, "A", "T", "max1", maximum)
	outside, err := dijkstra.Bidirectional(large, "S", "T", dijkstra.WithSearchMaxDistance(maximum))
	if err != nil || outside == nil || outside.Reachable {
		t.Fatalf("out-of-cutoff overflow must be skipped: result=%+v err=%v", outside, err)
	}
	if got, searchErr := dijkstra.Bidirectional(large, "S", "T"); got != nil || !errors.Is(searchErr, dijkstra.ErrDistanceOverflow) {
		t.Fatalf("unbounded original-cost overflow must fail: result=%+v err=%v", got, searchErr)
	}

	// The maximum finite original cost is legal when it requires no addition
	// beyond the binary64 domain; only the second maximum edge overflows.
	direct := searchTestGraph(t, core.WithDirected(true))
	searchTestEdge(t, direct, "S", "T", "maximum", maximum)
	maximumRoute, err := dijkstra.Bidirectional(direct, "S", "T", dijkstra.WithSearchMaxDistance(maximum))
	if err != nil {
		t.Fatal(err)
	}
	searchTestWitness(t, direct, maximumRoute)
	if !maximumRoute.Reachable || maximumRoute.Distance != maximum {
		t.Fatalf("legal MaxFloat64 route rejected: %+v", maximumRoute)
	}
}

// TestBidirectional_SumBoundRationalOracle verifies the exact >= stopping predicate.
// A negative residual at rounded equality must not certify early termination.
func TestBidirectional_SumBoundRationalOracle(t *testing.T) {
	belowOne := math.Nextafter(1, 0)
	tie := math.Ldexp(1, -54)
	if !(float64(belowOne+tie) == 1) {
		t.Fatal("fixture must round to 1 at a binary64 midpoint")
	}
	if dijkstra.SearchTestSumAtLeast(belowOne, tie, 1) {
		t.Fatal("rounded equality with negative residual stopped too early")
	}

	values := []float64{
		0, math.SmallestNonzeroFloat64, math.Ldexp(1, -100), 0.1, 0.2, 0.9,
		belowOne, tie, 1, math.Nextafter(1, math.Inf(1)),
		math.Ldexp(1, 53), math.MaxFloat64,
	}
	for _, left := range values {
		for _, right := range values {
			exact := new(big.Rat).Add(new(big.Rat).SetFloat64(left), new(big.Rat).SetFloat64(right))
			for _, bound := range values {
				want := exact.Cmp(new(big.Rat).SetFloat64(bound)) >= 0
				if got := dijkstra.SearchTestSumAtLeast(left, right, bound); got != want {
					t.Fatalf("exact >= mismatch: %g + %g >= %g: got %t, want %t", left, right, bound, got, want)
				}
			}
		}
	}
}

// TestBidirectional_WrapperAndCancellation ensures the wrapper enforces its method
// without overwriting caller storage, and preserves context causes on error.
func TestBidirectional_WrapperAndCancellation(t *testing.T) {
	g := searchTestGraph(t, core.WithDirected(true))
	searchTestEdge(t, g, "S", "T", "direct", 1)
	opts := make([]dijkstra.SearchOption, 1, 4)
	opts[0] = dijkstra.WithSearchMethod(dijkstra.SearchDijkstra)
	spare := errors.New("spare callback unexpectedly called")
	backing := opts[:cap(opts)]
	backing[1] = func(*dijkstra.SearchOptions) error { return spare }
	wrapped, err := dijkstra.Bidirectional(g, "S", "T", opts...)
	if err != nil {
		t.Fatal(err)
	}
	if wrapped.Method != dijkstra.SearchBidirectional || wrapped.Distance != 1 {
		t.Fatalf("wrapper failed to enforce method: %+v", wrapped)
	}
	if err := backing[1](&dijkstra.SearchOptions{}); !errors.Is(err, spare) {
		t.Fatalf("wrapper modified caller backing storage: %v", err)
	}
	selected, err := dijkstra.Search(g, "S", "T", dijkstra.WithSearchMethod(dijkstra.SearchBidirectional))
	if err != nil || selected.Distance != wrapped.Distance || selected.Method != wrapped.Method {
		t.Fatalf("direct dijkstra.Search dispatch differs: %+v err=%v", selected, err)
	}

	ctx, cancel := context.WithCancelCause(context.Background())
	cause := errors.New("bidirectional stop cause")
	cancel(cause)
	if result, searchErr := dijkstra.Bidirectional(g, "S", "T", dijkstra.WithSearchContext(ctx)); result != nil ||
		!errors.Is(searchErr, context.Canceled) || !errors.Is(searchErr, cause) {
		t.Fatalf("cancel produced a partial result or lost its causes: %+v, %v", result, searchErr)
	}
	callbackFailure := errors.New("custom bidirectional option failure")
	if result, searchErr := dijkstra.Bidirectional(g, "S", "T", func(*dijkstra.SearchOptions) error { return callbackFailure }); result != nil || !errors.Is(searchErr, callbackFailure) {
		t.Fatalf("custom callback error lost: result=%+v err=%v", result, searchErr)
	}
	if result, searchErr := dijkstra.Bidirectional(g, "S", "T", dijkstra.WithSearchMaxDistance(-1), dijkstra.WithSearchMaxDistance(10)); result != nil || !errors.Is(searchErr, dijkstra.ErrBadMaxDistance) {
		t.Fatalf("invalid scalar repaired by later option: %+v, %v", result, searchErr)
	}
	if result, searchErr := dijkstra.Bidirectional(g, "S", "missing"); result != nil || !errors.Is(searchErr, dijkstra.ErrTargetNotFound) {
		t.Fatalf("missing target classification lost: %+v, %v", result, searchErr)
	}

	// Cancel after real forward work. The next backward expansion must fail
	// without treating the already-existing tentative state as an answer.
	runningCtx, stop := context.WithCancelCause(context.Background())
	if err := dijkstra.SearchTestCancelAfterForward(runningCtx, g, "S", "T", func() { stop(cause) }); !errors.Is(err, context.Canceled) || !errors.Is(err, cause) {
		t.Fatalf("mid-search cancellation lost causes: %v", err)
	}
}

// TestBidirectional_StaleHeadsAndDamagedSuffix checks public stale-label safety.
// A caller-damaged cyclic result witness must be rejected without partial output.
func TestBidirectional_StaleHeadsAndDamagedSuffix(t *testing.T) {
	cost, present, remaining := dijkstra.SearchTestProbeStaleHead()
	if !present || cost != 2 || remaining {
		t.Fatalf("stale root or settled queue candidate survived: cost=%g present=%t remaining=%t",
			cost, present, remaining)
	}
	graph := searchTestGraph(t, core.WithDirected(true))
	searchTestEdge(t, graph, "S", "A", "slow", 10)
	searchTestEdge(t, graph, "S", "B", "first", 1)
	searchTestEdge(t, graph, "B", "A", "improve", 1)
	searchTestEdge(t, graph, "A", "T", "finish", 1)
	searchTestEdge(t, graph, "S", "T", "direct", 20)
	result, err := dijkstra.Bidirectional(graph, "S", "T")
	if err != nil {
		t.Fatal(err)
	}
	searchTestWitness(t, graph, result)
	if result.Distance != 3 || !slices.Equal(result.EdgeIDs, []string{"first", "improve", "finish"}) {
		t.Fatalf("stale candidate corrupted route: %+v", result)
	}
	result.Vertices = []string{"S", "A", "A", "T"}
	result.EdgeIDs = []string{"slow", "slow", "finish"}
	if _, _, err := result.Path(); !errors.Is(err, dijkstra.ErrNoPath) {
		t.Fatalf("mutated cyclic suffix unexpectedly published: %v", err)
	}
	if err := dijkstra.SearchTestProbeDamagedSuffix(graph, "S", "T", "A", "slow"); !errors.Is(err, dijkstra.ErrNoPath) {
		t.Fatalf("missing private backward successor was accepted: %v", err)
	}
}

// TestBidirectional_OracleMixedMultigraphs compares bidirectional output with an
// independent bounded integer Floyd-Warshall oracle under real wall/cutoff policies.
func TestBidirectional_OracleMixedMultigraphs(t *testing.T) {
	const (
		seed        int64 = 20261030
		caseCount         = 40
		wall              = 6
		cutoff            = 13
		unreachable int64 = 1 << 55
	)
	rng := rand.New(rand.NewSource(seed))
	for caseIndex := 0; caseIndex < caseCount; caseIndex++ {
		vertexCount := 2 + rng.Intn(6)
		g, err := core.NewMixedGraph(core.WithWeighted(), core.WithDirected(true),
			core.WithMultiEdges(), core.WithLoops())
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
		allowed := make([]oracleArc, 0, vertexCount*4)
		for edgeIndex := 0; edgeIndex < vertexCount*4; edgeIndex++ {
			from, to := rng.Intn(vertexCount), rng.Intn(vertexCount)
			weight := int64(rng.Intn(10))
			directed := rng.Intn(2) == 0
			id := fmt.Sprintf("test%d_edge%03d", caseIndex, edgeIndex)
			searchTestEdge(t, g, ids[from], ids[to], id, float64(weight), core.WithEdgeDirected(directed))
			if weight < wall {
				allowed = append(allowed, oracleArc{from: from, to: to, weight: weight, directed: directed})
			}
		}
		oracle := oracleDistances(vertexCount, allowed)
		options := []dijkstra.SearchOption{dijkstra.WithSearchInfEdgeThreshold(wall), dijkstra.WithSearchMaxDistance(cutoff)}
		for source := range ids {
			for target := range ids {
				result, err := dijkstra.Bidirectional(g, ids[source], ids[target], options...)
				if err != nil {
					t.Fatalf("case=%d %s->%s: %v", caseIndex, ids[source], ids[target], err)
				}
				searchTestWitness(t, g, result)
				want := oracle[source][target]
				if want == unreachable || want > cutoff {
					if result.Reachable || !math.IsInf(result.Distance, 1) {
						t.Fatalf("case=%d %s->%s: expected no admitted path: %+v", caseIndex, ids[source], ids[target], result)
					}
					continue
				}
				if !result.Reachable || result.Distance != float64(want) || result.Method != dijkstra.SearchBidirectional {
					t.Fatalf("case=%d %s->%s: expected %d, got %+v", caseIndex, ids[source], ids[target], want, result)
				}
				for _, edgeID := range result.EdgeIDs {
					edge, err := g.GetEdge(edgeID)
					if err != nil || edge.Weight >= wall {
						t.Fatalf("case=%d crossed forbidden wall %q: %v", caseIndex, edgeID, err)
					}
				}
			}
		}
	}
}

// TestBidirectional_InsertionOrderDoesNotAffectWitness verifies lexical ID
// enumeration by constructing the same equal-cost topology in opposite insertion orders.
func TestBidirectional_InsertionOrderDoesNotAffectWitness(t *testing.T) {
	builder := func(reverse bool) *core.Graph {
		g := searchTestGraph(t, core.WithDirected(true))
		edges := []struct {
			from, to, id string
			weight       float64
		}{
			{"S", "A", "aStart", 0}, {"S", "B", "bStart", 0},
			{"A", "T", "aFinish", 1}, {"B", "T", "bFinish", 1},
		}
		for i := range edges {
			index := i
			if reverse {
				index = len(edges) - 1 - i
			}
			e := edges[index]
			searchTestEdge(t, g, e.from, e.to, e.id, e.weight)
		}
		return g
	}
	first, err := dijkstra.Bidirectional(builder(false), "S", "T")
	if err != nil {
		t.Fatal(err)
	}
	second, err := dijkstra.Bidirectional(builder(true), "S", "T")
	if err != nil {
		t.Fatal(err)
	}
	if first.Distance != 1 || second.Distance != 1 ||
		!slices.Equal(first.Vertices, second.Vertices) ||
		!slices.Equal(first.EdgeIDs, second.EdgeIDs) {
		t.Fatalf("insertion order changed deterministic witness: first=%+v second=%+v", first, second)
	}
}

// TestBidirectional_ThreeTermCutoffRational verifies exact admission of a
// crossing even when its intermediate forward-plus-edge sum is not storable.
func TestBidirectional_ThreeTermCutoffRational(t *testing.T) {
	values := []float64{
		0, math.SmallestNonzeroFloat64, math.Ldexp(1, -54), 0.1, 0.2,
		0.4, 0.9, 1, 2, math.Ldexp(1, 53), math.MaxFloat64,
	}
	for _, first := range values {
		for _, middle := range values {
			for _, last := range values {
				whole := new(big.Rat).Add(new(big.Rat).SetFloat64(first), new(big.Rat).SetFloat64(middle))
				whole.Add(whole, new(big.Rat).SetFloat64(last))
				for _, bound := range []float64{0, 1, 1.5, math.Ldexp(1, 53), math.MaxFloat64} {
					want := whole.Cmp(new(big.Rat).SetFloat64(bound)) > 0
					if got := dijkstra.SearchTestThreeAbove(first, middle, last, bound); got != want {
						t.Fatalf("%g+%g+%g > %g: got %t want %t", first, middle, last, bound, got, want)
					}
				}
			}
		}
	}

	// A complete S->A->B->T crossing above cutoff must be discarded BEFORE
	// demanding exact binary64 storage of its first two fractional terms.
	graph := searchTestGraph(t, core.WithDirected(true))
	searchTestEdge(t, graph, "S", "A", "prefix", 0.9)
	searchTestEdge(t, graph, "A", "B", "bridge", 0.1)
	searchTestEdge(t, graph, "B", "T", "suffix", 0.9)
	witnessed, err := dijkstra.SearchTestProbePrunedJoin(
		graph, "S", "T", "A", "B", "bridge", 0.9, 0.9, 1.5,
	)
	if err != nil || witnessed {
		t.Fatalf("out-of-cutoff complete crossing wrongly admitted or rejected: witnessed=%t err=%v", witnessed, err)
	}
}
