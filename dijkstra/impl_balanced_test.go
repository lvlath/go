// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dijkstra_test verifies balanced bidirectional A* through its public API.
// Mathematical oracles use original core edges, never the private search engine.
package dijkstra_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/big"
	"math/rand"
	"slices"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/dijkstra"
)

// TestBidirectionalAStar_ZeroPotentialsPreserveDijkstra verifies exact reduction.
// With both callbacks zero, every frontier key and incumbent bound is unchanged.
func TestBidirectionalAStar_ZeroPotentialsPreserveDijkstra(t *testing.T) {
	graph := searchTestGraph(t, core.WithDirected(true), core.WithMultiEdges())
	searchTestEdge(t, graph, "S", "A", "zStart", 0)
	searchTestEdge(t, graph, "A", "B", "zMiddle", 0)
	searchTestEdge(t, graph, "B", "T", "zEnd", 3)
	searchTestEdge(t, graph, "S", "T", "slowDirect", 8)
	searchTestEdge(t, graph, "S", "T", "fastDirect", 5)
	zero := func(string) (float64, error) { return 0, nil }

	baseline, err := dijkstra.Bidirectional(graph, "S", "T")
	if err != nil {
		t.Fatal(err)
	}
	got, err := dijkstra.BidirectionalAStar(graph, "S", "T", zero, zero)
	if err != nil {
		t.Fatal(err)
	}
	mustBalancedWitness(t, graph, got)
	if got.Method != dijkstra.SearchBidirectionalAStar || got.Distance != 3 ||
		got.Distance != baseline.Distance || !slices.Equal(got.Vertices, baseline.Vertices) ||
		!slices.Equal(got.EdgeIDs, baseline.EdgeIDs) {
		t.Fatalf("zero potential changed shortest witness: baseline=%+v balanced=%+v", baseline, got)
	}
	selected, err := dijkstra.Search(graph, "S", "T",
		dijkstra.WithSearchMethod(dijkstra.SearchBidirectionalAStar),
		dijkstra.WithBalancedHeuristics(zero, zero))
	if err != nil || selected == nil || selected.Method != dijkstra.SearchBidirectionalAStar ||
		!slices.Equal(selected.EdgeIDs, got.EdgeIDs) {
		t.Fatalf("canonical facade and wrapper disagree: %+v err=%v", selected, err)
	}
}

// TestBidirectionalAStar_ShiftedIncumbentAvoidsEarlyStopping checks the balanced bound.
// The first crossing costs 101, but the normalized stopping proof must retain 52.
func TestBidirectionalAStar_ShiftedIncumbentAvoidsEarlyStopping(t *testing.T) {
	graph := searchTestGraph(t, core.WithDirected(true))
	searchTestEdge(t, graph, "S", "A", "startA", 1)
	searchTestEdge(t, graph, "A", "T", "finishA", 100)
	searchTestEdge(t, graph, "S", "B", "startB", 50)
	searchTestEdge(t, graph, "B", "T", "finishB", 2)
	toTargetValues := map[string]float64{"S": -100, "A": -100, "B": -100, "T": 0}
	fromSourceValues := map[string]float64{"S": 0, "A": -100, "B": -100, "T": -100}
	toTarget := func(id string) (float64, error) { return toTargetValues[id], nil }
	fromSource := func(id string) (float64, error) { return fromSourceValues[id], nil }

	// p(S)=-50, p(A)=p(B)=0, p(T)=50. The incumbent must be
	// compared against mu+p(T)-p(S), which is 201 for the first join.
	// Clean minima of 100 and 52 sum to 152: a comparison with mu=101
	// would incorrectly stop, but 152<201 requires continued expansion.
	got, err := dijkstra.BidirectionalAStar(graph, "S", "T", toTarget, fromSource)
	if err != nil {
		t.Fatal(err)
	}
	mustBalancedWitness(t, graph, got)
	if got.Distance != 52 || !slices.Equal(got.Vertices, []string{"S", "B", "T"}) ||
		!slices.Equal(got.EdgeIDs, []string{"startB", "finishB"}) {
		t.Fatalf("early-stop violated balanced mathematical bound: %+v", got)
	}

	// The endpoint potential shift is 100, but the cutoff is still in
	// original cost units. Exactly 52 must be admitted; 51 must reject it.
	admitted, err := dijkstra.BidirectionalAStar(graph, "S", "T", toTarget, fromSource,
		dijkstra.WithSearchMaxDistance(52))
	if err != nil || admitted == nil || !admitted.Reachable || admitted.Distance != 52 {
		t.Fatalf("original-cost cutoff wrongly used reduced incumbent: %+v err=%v", admitted, err)
	}
	excluded, err := dijkstra.BidirectionalAStar(graph, "S", "T", toTarget, fromSource,
		dijkstra.WithSearchMaxDistance(51))
	if err != nil || excluded == nil || excluded.Reachable || !math.IsInf(excluded.Distance, 1) {
		t.Fatalf("original-cost cutoff accepted expensive route: %+v err=%v", excluded, err)
	}
	blocked, err := dijkstra.BidirectionalAStar(graph, "S", "T", toTarget, fromSource,
		dijkstra.WithSearchInfEdgeThreshold(50))
	if err != nil || blocked == nil || blocked.Reachable || !math.IsInf(blocked.Distance, 1) {
		t.Fatalf("wall threshold ignored original edge costs: %+v err=%v", blocked, err)
	}
}

// TestBidirectionalAStar_AsymmetricTopologyAndOriginalIDs defends incoming arcs.
// Reversing a directed trap or merging parallel edges would fabricate wrong witnesses.
func TestBidirectionalAStar_AsymmetricTopologyAndOriginalIDs(t *testing.T) {
	graph, err := core.NewMixedGraph(core.WithWeighted(), core.WithDirected(true), core.WithMultiEdges())
	if err != nil {
		t.Fatal(err)
	}
	searchTestEdge(t, graph, "S", "A", "source", 1)
	searchTestEdge(t, graph, "T", "A", "reverseTrap", 0)
	searchTestEdge(t, graph, "A", "B", "shared", 1, core.WithEdgeDirected(false))
	searchTestEdge(t, graph, "B", "T", "long", 8)
	searchTestEdge(t, graph, "B", "T", "short", 2)
	zero := func(string) (float64, error) { return 0, nil }
	got, err := dijkstra.BidirectionalAStar(graph, "S", "T", zero, zero)
	if err != nil {
		t.Fatal(err)
	}
	mustBalancedWitness(t, graph, got)
	if got.Distance != 4 || !slices.Equal(got.EdgeIDs, []string{"source", "shared", "short"}) {
		t.Fatalf("bad directed or parallel original ID witness: %+v", got)
	}
	reverse, err := dijkstra.BidirectionalAStar(graph, "T", "S", zero, zero)
	if err != nil {
		t.Fatal(err)
	}
	mustBalancedWitness(t, graph, reverse)
	if reverse.Reachable {
		t.Fatalf("reverse directed trap fabricated a source path: %+v", reverse)
	}
}

// TestBidirectionalAStar_DirectionalAnchorsAndExactConsistency validates both laws.
// Reverse consistency must reject rounded equality just as forward consistency does.
func TestBidirectionalAStar_DirectionalAnchorsAndExactConsistency(t *testing.T) {
	graph := searchTestGraph(t, core.WithDirected(true))
	searchTestEdge(t, graph, "S", "A", "start", 1)
	searchTestEdge(t, graph, "A", "B", "near", math.Ldexp(1, -54))
	searchTestEdge(t, graph, "B", "T", "end", 1)
	zero := func(string) (float64, error) { return 0, nil }
	below := math.Nextafter(1, 0)

	badTarget := func(id string) (float64, error) {
		if id == "A" {
			return 1, nil
		}
		if id == "B" {
			return below, nil
		}
		return 0, nil
	}
	got, err := dijkstra.BidirectionalAStar(graph, "S", "T", badTarget, zero)
	mustBalancedError(t, got, err, dijkstra.ErrInconsistentHeuristic)

	badSource := func(id string) (float64, error) {
		switch id {
		case "A":
			return below, nil
		case "B":
			return 1, nil
		default:
			return 0, nil
		}
	}
	got, err = dijkstra.BidirectionalAStar(graph, "S", "T", zero, badSource)
	mustBalancedError(t, got, err, dijkstra.ErrInconsistentHeuristic)

	wrongTargetAnchor := func(id string) (float64, error) {
		if id == "T" {
			return 1, nil
		}
		return 0, nil
	}
	got, err = dijkstra.BidirectionalAStar(graph, "S", "T", wrongTargetAnchor, zero)
	mustBalancedError(t, got, err, dijkstra.ErrInvalidHeuristic)
	wrongSourceAnchor := func(id string) (float64, error) {
		if id == "S" {
			return 1, nil
		}
		return 0, nil
	}
	got, err = dijkstra.BidirectionalAStar(graph, "S", "T", zero, wrongSourceAnchor)
	mustBalancedError(t, got, err, dijkstra.ErrInvalidHeuristic)
}

// TestBidirectionalAStar_CallbackOrderOwnershipAndCauses checks callback freezing.
// All ToTarget callbacks precede all FromSource callbacks in lexical ID order.
func TestBidirectionalAStar_CallbackOrderOwnershipAndCauses(t *testing.T) {
	graph := searchTestGraph(t, core.WithDirected(true))
	searchTestEdge(t, graph, "S", "T", "direct", 3)
	if err := graph.AddVertex("Z"); err != nil {
		t.Fatal(err)
	}
	calls := make([]string, 0, 6)
	toTarget := func(id string) (float64, error) {
		calls = append(calls, "T:"+id)
		return 0, nil
	}
	fromSource := func(id string) (float64, error) {
		calls = append(calls, "S:"+id)
		return 0, nil
	}
	got, err := dijkstra.BidirectionalAStar(graph, "S", "T", toTarget, fromSource)
	if err != nil {
		t.Fatal(err)
	}
	if got.Distance != 3 || !slices.Equal(calls, []string{
		"T:S", "T:T", "T:Z", "S:S", "S:T", "S:Z",
	}) {
		t.Fatalf("callback invocation was not frozen and lexical: cost=%g calls=%v", got.Distance, calls)
	}

	// A wrapper must enforce its method while never overwriting the caller's
	// option slice backing capacity beyond len, even with spare space.
	opts := make([]dijkstra.SearchOption, 1, 4)
	opts[0] = dijkstra.WithSearchMethod(dijkstra.SearchDijkstra)
	spare := errors.New("caller spare option")
	backing := opts[:cap(opts)]
	backing[1] = func(*dijkstra.SearchOptions) error { return spare }
	got, err = dijkstra.BidirectionalAStar(graph, "S", "T", toTarget, fromSource, opts...)
	if err != nil || got.Method != dijkstra.SearchBidirectionalAStar {
		t.Fatalf("wrapper method override failed: %+v err=%v", got, err)
	}
	if err := backing[1](&dijkstra.SearchOptions{}); !errors.Is(err, spare) {
		t.Fatalf("wrapper mutated unused caller backing capacity: %v", err)
	}

	cause := errors.New("upstream heuristic failure")
	got, err = dijkstra.BidirectionalAStar(graph, "S", "T", toTarget,
		func(string) (float64, error) { return 0, cause })
	mustBalancedError(t, got, err, cause)
	ctx, cancel := context.WithCancelCause(context.Background())
	stop := errors.New("provider canceled")
	canceling := func(string) (float64, error) {
		cancel(stop)
		return 0, nil
	}
	got, err = dijkstra.BidirectionalAStar(graph, "S", "T", canceling, fromSource,
		dijkstra.WithSearchContext(ctx))
	if got != nil || !errors.Is(err, context.Canceled) || !errors.Is(err, stop) {
		t.Fatalf("lost mid-callback cancellation: result=%+v err=%v", got, err)
	}
}

// TestBidirectionalAStar_OptionCompletenessAndSpecialPaths validates public policy.
// Missing callbacks are errors; self paths and unreachable targets still validate both.
func TestBidirectionalAStar_OptionCompletenessAndSpecialPaths(t *testing.T) {
	graph := searchTestGraph(t, core.WithDirected(true))
	searchTestEdge(t, graph, "S", "A", "free", 0)
	searchTestEdge(t, graph, "A", "T", "positive", 2)
	if err := graph.AddVertex("U"); err != nil {
		t.Fatal(err)
	}
	zero := func(string) (float64, error) { return 0, nil }
	for _, test := range []struct {
		name     string
		opts     []dijkstra.SearchOption
		expected error
	}{
		{"missing pair", []dijkstra.SearchOption{
			dijkstra.WithSearchMethod(dijkstra.SearchBidirectionalAStar),
		}, dijkstra.ErrNilHeuristic},
		{"target-only pair", []dijkstra.SearchOption{
			dijkstra.WithSearchMethod(dijkstra.SearchBidirectionalAStar),
			dijkstra.WithHeuristic(zero),
		}, dijkstra.ErrNilHeuristic},
		{"unused pair in Dijkstra", []dijkstra.SearchOption{
			dijkstra.WithBalancedHeuristics(zero, zero),
		}, dijkstra.ErrInvalidSearchOptions},
		{"unused source estimate in AStar", []dijkstra.SearchOption{
			dijkstra.WithSearchMethod(dijkstra.SearchAStar),
			dijkstra.WithBalancedHeuristics(zero, zero),
		}, dijkstra.ErrInvalidSearchOptions},
		{"explicit nil cannot be repaired", []dijkstra.SearchOption{
			dijkstra.WithSearchMethod(dijkstra.SearchBidirectionalAStar),
			dijkstra.WithBalancedHeuristics(nil, zero),
			dijkstra.WithBalancedHeuristics(zero, zero),
		}, dijkstra.ErrNilHeuristic},
	} {
		t.Run(test.name, func(t *testing.T) {
			result, err := dijkstra.Search(graph, "S", "T", test.opts...)
			mustBalancedError(t, result, err, test.expected)
		})
	}
	valid, err := dijkstra.Search(graph, "S", "T",
		dijkstra.WithSearchMethod(dijkstra.SearchBidirectionalAStar),
		dijkstra.WithBalancedHeuristics(zero, zero))
	if err != nil || valid == nil || valid.Distance != 2 {
		t.Fatalf("valid completed pair rejected: result=%+v err=%v", valid, err)
	}
	got, err := dijkstra.BidirectionalAStar(graph, "S", "A", zero, zero, dijkstra.WithSearchMaxDistance(0))
	if err != nil {
		t.Fatal(err)
	}
	mustBalancedWitness(t, graph, got)
	if !got.Reachable || got.Distance != 0 || !slices.Equal(got.EdgeIDs, []string{"free"}) {
		t.Fatalf("zero cutoff excluded a valid zero route: %+v", got)
	}
	self, err := dijkstra.BidirectionalAStar(graph, "S", "S", zero, zero)
	if err != nil || self == nil || !self.Reachable || self.Distance != 0 ||
		!slices.Equal(self.Vertices, []string{"S"}) || self.EdgeIDs != nil {
		t.Fatalf("self path contract: %+v err=%v", self, err)
	}
	unreachable, err := dijkstra.BidirectionalAStar(graph, "S", "U", zero, zero)
	if err != nil {
		t.Fatal(err)
	}
	mustBalancedWitness(t, graph, unreachable)
	if unreachable.Reachable || !math.IsInf(unreachable.Distance, 1) {
		t.Fatalf("unreachable publication: %+v", unreachable)
	}
	for _, test := range []struct {
		to, from dijkstra.Heuristic
	}{
		{nil, zero}, {zero, nil}, {nil, nil},
	} {
		result, err := dijkstra.BidirectionalAStar(graph, "S", "T", test.to, test.from)
		mustBalancedError(t, result, err, dijkstra.ErrNilHeuristic)
	}
	for _, invalid := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		bad := func(string) (float64, error) { return invalid, nil }
		result, err := dijkstra.BidirectionalAStar(graph, "S", "T", bad, zero)
		mustBalancedError(t, result, err, dijkstra.ErrInvalidHeuristic)
		result, err = dijkstra.BidirectionalAStar(graph, "S", "T", zero, bad)
		mustBalancedError(t, result, err, dijkstra.ErrInvalidHeuristic)
	}

	// The public setter must reject an incomplete pair atomically; no callback
	// is retained in a configuration when either required function is absent.
	config := dijkstra.DefaultSearchOptions()
	if err := dijkstra.WithBalancedHeuristics(zero, nil)(&config); !errors.Is(err, dijkstra.ErrNilHeuristic) ||
		config.ToTarget != nil || config.FromSource != nil {
		t.Fatalf("invalid setter partially mutated heuristic pair: %+v err=%v", config, err)
	}
}

// TestBidirectionalAStar_NumericExtremes checks representability and key separation.
// Large opposite heuristic values can produce a finite exact half-difference.
func TestBidirectionalAStar_NumericExtremes(t *testing.T) {
	t.Run("overflow-safe half difference", func(t *testing.T) {
		graph := searchTestGraph(t, core.WithDirected(true))
		searchTestEdge(t, graph, "S", "T", "free", 0)
		if err := graph.AddVertex("A"); err != nil {
			t.Fatal(err)
		}
		ht := func(id string) (float64, error) {
			if id == "A" {
				return math.MaxFloat64, nil
			}
			return 0, nil
		}
		hs := func(id string) (float64, error) {
			if id == "A" || id == "T" {
				return -math.MaxFloat64, nil
			}
			return 0, nil
		}
		result, err := dijkstra.BidirectionalAStar(graph, "S", "T", ht, hs)
		if err != nil {
			t.Fatal(err)
		}
		mustBalancedWitness(t, graph, result)
		if !result.Reachable || result.Distance != 0 {
			t.Fatalf("finite half difference treated as overflow: %+v", result)
		}
	})
	t.Run("subnormal half is inexact", func(t *testing.T) {
		graph := searchTestGraph(t, core.WithDirected(true))
		searchTestEdge(t, graph, "S", "T", "small", math.SmallestNonzeroFloat64)
		ht := func(id string) (float64, error) {
			if id == "S" {
				return math.SmallestNonzeroFloat64, nil
			}
			return 0, nil
		}
		zero := func(string) (float64, error) { return 0, nil }
		result, err := dijkstra.BidirectionalAStar(graph, "S", "T", ht, zero)
		mustBalancedError(t, result, err, dijkstra.ErrInexactArithmetic)
	})
	t.Run("normalized key overflow", func(t *testing.T) {
		graph := searchTestGraph(t, core.WithDirected(true))
		searchTestEdge(t, graph, "S", "T", "maximum", math.MaxFloat64)
		ht := func(id string) (float64, error) {
			if id == "S" {
				return -math.MaxFloat64, nil
			}
			return 0, nil
		}
		hs := func(id string) (float64, error) {
			if id == "T" {
				return -math.MaxFloat64, nil
			}
			return 0, nil
		}
		result, err := dijkstra.BidirectionalAStar(graph, "S", "T", ht, hs)
		mustBalancedError(t, result, err, dijkstra.ErrSearchKeyOverflow)
	})
	t.Run("nonrepresentable key", func(t *testing.T) {
		graph := searchTestGraph(t, core.WithDirected(true))
		searchTestEdge(t, graph, "S", "T", "one", 1)
		ht := func(id string) (float64, error) {
			if id == "S" {
				return -math.Ldexp(1, 54), nil
			}
			return 0, nil
		}
		zero := func(string) (float64, error) { return 0, nil }
		result, err := dijkstra.BidirectionalAStar(graph, "S", "T", ht, zero)
		mustBalancedError(t, result, err, dijkstra.ErrInexactArithmetic)
	})
	t.Run("exact finite cutoff vs overflow", func(t *testing.T) {
		graph := searchTestGraph(t, core.WithDirected(true))
		searchTestEdge(t, graph, "S", "A", "first", math.MaxFloat64)
		searchTestEdge(t, graph, "A", "T", "second", math.MaxFloat64)
		zero := func(string) (float64, error) { return 0, nil }
		out, err := dijkstra.BidirectionalAStar(graph, "S", "T", zero, zero,
			dijkstra.WithSearchMaxDistance(math.MaxFloat64))
		if err != nil || out == nil || out.Reachable {
			t.Fatalf("cutoff should prune overflow: %+v err=%v", out, err)
		}
		out, err = dijkstra.BidirectionalAStar(graph, "S", "T", zero, zero)
		mustBalancedError(t, out, err, dijkstra.ErrDistanceOverflow)
	})
}

// balancedOracleDistance independently computes shortest costs in a small graph.
// Floyd-Warshall operates on original edge values, including directedness and duplicates.
func balancedOracleDistance(graph *core.Graph, ids []string) [][]int64 {
	const infinity int64 = 1 << 50
	index := make(map[string]int, len(ids))
	for i, id := range ids {
		index[id] = i
	}
	result := make([][]int64, len(ids))
	for u := range result {
		result[u] = make([]int64, len(ids))
		for v := range result[u] {
			result[u][v] = infinity
		}
		result[u][u] = 0
	}
	for _, edge := range graph.Edges() {
		from, to := index[edge.From], index[edge.To]
		cost := int64(edge.Weight)
		if cost < result[from][to] {
			result[from][to] = cost
		}
		if !edge.Directed && cost < result[to][from] {
			result[to][from] = cost
		}
	}
	for k := range ids {
		for u := range ids {
			for v := range ids {
				if result[u][k] == infinity || result[k][v] == infinity {
					continue
				}
				cost := result[u][k] + result[k][v]
				if cost < result[u][v] {
					result[u][v] = cost
				}
			}
		}
	}
	return result
}

// TestBidirectionalAStar_IndependentMixedGraphOracle proves optimality under both potentials.
// Exact integer shortest-distance estimates are independently generated per query.
func TestBidirectionalAStar_IndependentMixedGraphOracle(t *testing.T) {
	const (
		seed      int64 = 20261031
		caseCount       = 32
		infinity  int64 = 1 << 50
		farBound  int64 = 1000
	)
	rng := rand.New(rand.NewSource(seed))
	for trial := 0; trial < caseCount; trial++ {
		graph, err := core.NewMixedGraph(core.WithWeighted(), core.WithDirected(true),
			core.WithMultiEdges(), core.WithLoops())
		if err != nil {
			t.Fatal(err)
		}
		ids := make([]string, 2+rng.Intn(5))
		for i := range ids {
			ids[i] = fmt.Sprintf("v%d", i)
			if err := graph.AddVertex(ids[i]); err != nil {
				t.Fatal(err)
			}
		}
		for i := 0; i < len(ids)*4; i++ {
			from, to := rng.Intn(len(ids)), rng.Intn(len(ids))
			weight := float64(rng.Intn(8))
			directed := rng.Intn(2) == 0
			searchTestEdge(t, graph, ids[from], ids[to],
				fmt.Sprintf("trial%d_edge%d", trial, i), weight, core.WithEdgeDirected(directed))
		}
		oracle := balancedOracleDistance(graph, ids)
		for source := range ids {
			for target := range ids {
				toTarget := func(id string) (float64, error) {
					var index int
					if _, err := fmt.Sscanf(id, "v%d", &index); err != nil {
						return 0, err
					}
					if oracle[index][target] == infinity {
						return float64(farBound), nil
					}
					return float64(oracle[index][target]), nil
				}
				fromSource := func(id string) (float64, error) {
					var index int
					if _, err := fmt.Sscanf(id, "v%d", &index); err != nil {
						return 0, err
					}
					if oracle[source][index] == infinity {
						return float64(farBound), nil
					}
					return float64(oracle[source][index]), nil
				}
				got, err := dijkstra.BidirectionalAStar(graph, ids[source], ids[target], toTarget, fromSource)
				if err != nil {
					t.Fatalf("trial=%d %s->%s: %v", trial, ids[source], ids[target], err)
				}
				mustBalancedWitness(t, graph, got)
				if want := oracle[source][target]; want == infinity {
					if got.Reachable || !math.IsInf(got.Distance, 1) {
						t.Fatalf("trial=%d %s->%s: want unreachable, got %+v", trial, ids[source], ids[target], got)
					}
				} else if !got.Reachable || got.Distance != float64(want) ||
					got.Method != dijkstra.SearchBidirectionalAStar {
					t.Fatalf("trial=%d %s->%s want %d, got %+v", trial, ids[source], ids[target], want, got)
				}
			}
		}
	}
}

// TestBidirectionalAStar_BalancedPotentialRationalOracle certifies the half-difference.
// Exact rationals are formed from the original binary64 bits, not rounded decimals.
func TestBidirectionalAStar_BalancedPotentialRationalOracle(t *testing.T) {
	const sampleCount = 2048
	rng := rand.New(rand.NewSource(20261031))
	values := []float64{
		0, math.SmallestNonzeroFloat64, -math.SmallestNonzeroFloat64,
		math.Ldexp(1, -1022), -math.Ldexp(1, -1022),
		math.Ldexp(1, 53), -math.Ldexp(1, 53),
		math.MaxFloat64, -math.MaxFloat64,
		math.Nextafter(1, 0), math.Nextafter(1, math.Inf(1)),
	}
	for i := 0; i < sampleCount; i++ {
		random := math.Float64frombits(rng.Uint64())
		if !math.IsNaN(random) && !math.IsInf(random, 0) {
			values = append(values, random)
		}
	}
	for index := range values {
		// Deterministically test both near-neighbor and opposite-sign operands.
		for _, other := range []float64{values[(index+1)%len(values)], -values[index]} {
			exact := new(big.Rat).Sub(
				new(big.Rat).SetFloat64(values[index]),
				new(big.Rat).SetFloat64(other),
			)
			exact.Quo(exact, big.NewRat(2, 1))
			rounded, _ := exact.Float64()
			wantExact := !math.IsInf(rounded, 0) &&
				new(big.Rat).SetFloat64(rounded).Cmp(exact) == 0
			got, err := dijkstra.SearchTestBalancedPotential(values[index], other)
			if wantExact {
				if err != nil || got != rounded {
					t.Fatalf("exact potential failed: hT=%g hS=%g want=%g got=%g err=%v",
						values[index], other, rounded, got, err)
				}
			} else if !errors.Is(err, dijkstra.ErrInexactArithmetic) {
				t.Fatalf("nonrepresentable half-difference accepted: hT=%g hS=%g got=%g err=%v",
					values[index], other, got, err)
			}
		}
	}
}
