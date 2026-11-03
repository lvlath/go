// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dijkstra_test tests consistent-heuristic A* against original edge witnesses.
// The oracles use independent integer graph distances and exact rational comparisons.
// AI-Hints:
//   - Do not replace exact inequality checks with tolerance-based acceptance.
//   - Callback-count checks require the prepared lexical vertex domain, not heap visits.
//   - An inexact heap key is a runtime error, not proof of target unreachability.
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

// TestAStar_GridConsistentManhattan verifies a genuine shortest-path A* call.
// Manhattan is consistent on cardinal unit edges and the output retains original IDs.
// Implementation:
//   - Build nine vertices and twelve undirected unit-cost connections.
//   - Invoke both dijkstra.AStar and dijkstra.Search with the same exact integer potential.
//   - Compare the original-cost witness, edge identities and zero-potential reduction.
//
// Behavior highlights:
//   - Multiple optimal grid routes are legal; this test requires cost and feasibility,
//     not an arbitrary preferred geometry that the API does not promise.
//
// Determinism:
//   - Both entry points must publish the same witness for the same ordered graph.
//
// AI-Hints:
//   - Manhattan is valid here only because each cardinal move costs exactly one.
func TestAStar_GridConsistentManhattan(t *testing.T) {
	const side = 3
	graph := searchTestGraph(t, core.WithDirected(false))
	id := func(row, column int) string { return fmt.Sprintf("r%dc%d", row, column) }
	for row := 0; row < side; row++ {
		for column := 0; column < side; column++ {
			if row+1 < side {
				searchTestEdge(t, graph, id(row, column), id(row+1, column),
					fmt.Sprintf("v%d_%d", row, column), 1)
			}
			if column+1 < side {
				searchTestEdge(t, graph, id(row, column), id(row, column+1),
					fmt.Sprintf("h%d_%d", row, column), 1)
			}
		}
	}
	estimate := func(vertex string) (float64, error) {
		var row, column int
		if _, err := fmt.Sscanf(vertex, "r%dc%d", &row, &column); err != nil {
			return 0, err
		}
		return float64((side - 1 - row) + (side - 1 - column)), nil
	}

	got, err := dijkstra.AStar(graph, id(0, 0), id(2, 2), estimate)
	if err != nil {
		t.Fatal(err)
	}
	searchTestWitness(t, graph, got)
	if got.Method != dijkstra.SearchAStar || got.Distance != 4 || !got.Reachable ||
		len(got.Vertices) != 5 || len(got.EdgeIDs) != 4 {
		t.Fatalf("Manhattan A* returned invalid original-cost result: %+v", got)
	}
	canonical, err := dijkstra.Search(graph, id(0, 0), id(2, 2),
		dijkstra.WithSearchMethod(dijkstra.SearchAStar), dijkstra.WithHeuristic(estimate))
	if err != nil {
		t.Fatal(err)
	}
	if canonical.Distance != 4 || canonical.Method != dijkstra.SearchAStar ||
		!slices.Equal(canonical.EdgeIDs, got.EdgeIDs) {
		t.Fatalf("facade and wrapper differ: canonical=%+v wrapper=%+v", canonical, got)
	}

	// A zero potential uses the same label-setting kernel, but must keep the
	// selected A* method rather than relabeling its result as Dijkstra.
	zero := func(string) (float64, error) { return 0, nil }
	withoutHeuristic, err := dijkstra.AStar(graph, id(0, 0), id(2, 2), zero)
	if err != nil {
		t.Fatal(err)
	}
	targetOnly, err := dijkstra.Search(graph, id(0, 0), id(2, 2))
	if err != nil {
		t.Fatal(err)
	}
	if withoutHeuristic.Method != dijkstra.SearchAStar ||
		withoutHeuristic.Distance != targetOnly.Distance ||
		withoutHeuristic.Reachable != targetOnly.Reachable {
		t.Fatalf("zero potential differs from target Dijkstra: %+v vs %+v", withoutHeuristic, targetOnly)
	}
}

// TestAStar_CallbacksFrozenInLexicalOrder ensures each vertex is evaluated once.
// An isolated vertex is still checked before traversal; repeated heap pushes do not call h.
func TestAStar_CallbacksFrozenInLexicalOrder(t *testing.T) {
	graph := searchTestGraph(t, core.WithDirected(true))
	searchTestEdge(t, graph, "S", "A", "start", 1)
	searchTestEdge(t, graph, "A", "T", "finish", 1)
	searchTestEdge(t, graph, "S", "T", "direct", 10)
	if err := graph.AddVertex("Z"); err != nil {
		t.Fatal(err)
	}
	values := map[string]float64{"A": 1, "S": 2, "T": 0, "Z": 9}
	calls := make([]string, 0, len(values))
	callback := func(id string) (float64, error) {
		calls = append(calls, id)
		return values[id], nil
	}
	got, err := dijkstra.AStar(graph, "S", "T", callback)
	if err != nil {
		t.Fatal(err)
	}
	searchTestWitness(t, graph, got)
	if got.Distance != 2 || !slices.Equal(calls, []string{"A", "S", "T", "Z"}) {
		t.Fatalf("cached lexical callback contract: cost=%g calls=%v", got.Distance, calls)
	}
}

// TestAStar_InconsistentHeuristicExactBound rejects a rounded-equality trap.
// Mathematically 1-2^-53+2^-54 < 1, even though float64 addition rounds to 1.
func TestAStar_InconsistentHeuristicExactBound(t *testing.T) {
	belowOne := math.Nextafter(1, 0)
	step := math.Ldexp(1, -54)
	if float64(belowOne+step) != 1 {
		t.Fatal("fixture must have rounded equality despite exact inconsistency")
	}
	graph := searchTestGraph(t, core.WithDirected(true))
	searchTestEdge(t, graph, "S", "A", "roundTrap", step)
	searchTestEdge(t, graph, "A", "T", "finish", belowOne)
	values := map[string]float64{"S": 1, "A": belowOne, "T": 0}
	callback := func(id string) (float64, error) { return values[id], nil }
	got, err := dijkstra.AStar(graph, "S", "T", callback)
	if got != nil || !errors.Is(err, dijkstra.ErrInconsistentHeuristic) {
		t.Fatalf("inconsistent rounded-equality edge accepted: result=%+v error=%v", got, err)
	}

	// A different rounded equality is mathematically valid: binary64(0.9)
	// plus binary64(0.1) is strictly greater than the represented binary64(1).
	positive := searchTestGraph(t, core.WithDirected(true))
	searchTestEdge(t, positive, "S", "A", "valid", 0.1)
	searchTestEdge(t, positive, "A", "T", "validFinish", 0.9)
	validEstimate := func(id string) (float64, error) {
		return map[string]float64{"S": 1, "A": 0.9, "T": 0}[id], nil
	}
	// The exact preprocessing certificate must succeed: the rounding trap
	// is mathematically consistent and its normalized offsets are representable.
	if err := dijkstra.SearchTestValidateAStar(positive, "S", "T", validEstimate); err != nil {
		t.Fatalf("mathematically consistent rounded-equality edge rejected: %v", err)
	}
}

// TestAStar_HeuristicErrorsAndOptions exercises method completeness and precedence.
// Inapplicable heuristics cannot be silently ignored by nonheuristic kernels.
// Implementation:
//   - Reject missing, explicitly nil and inapplicable callbacks through public dijkstra.Search.
//   - Accept either valid option ordering before graph work begins.
//   - Reject non-finite estimates, a nonzero target anchor and inconsistent values.
//   - Accept a finite negative consistent potential to defend the signed domain.
//
// Behavior highlights:
//   - The first invalid option or method/heuristic failure yields nil dijkstra.PathResult.
//
// AI-Hints:
//   - Do not select A* automatically in dijkstra.WithHeuristic or accept nil as zero h.
func TestAStar_HeuristicErrorsAndOptions(t *testing.T) {
	graph := searchTestGraph(t, core.WithDirected(true))
	searchTestEdge(t, graph, "S", "T", "direct", 2)
	zero := func(string) (float64, error) { return 0, nil }

	for _, tc := range []struct {
		name string
		opts []dijkstra.SearchOption
		want error
	}{
		{"missing A* callback", []dijkstra.SearchOption{dijkstra.WithSearchMethod(dijkstra.SearchAStar)}, dijkstra.ErrNilHeuristic},
		{"explicit nil callback", []dijkstra.SearchOption{dijkstra.WithSearchMethod(dijkstra.SearchAStar), dijkstra.WithHeuristic(nil)}, dijkstra.ErrNilHeuristic},
		{"unexpected Dijkstra callback", []dijkstra.SearchOption{dijkstra.WithHeuristic(zero)}, dijkstra.ErrInvalidSearchOptions},
		{"unexpected bidirectional callback", []dijkstra.SearchOption{dijkstra.WithSearchMethod(dijkstra.SearchBidirectional), dijkstra.WithHeuristic(zero)}, dijkstra.ErrInvalidSearchOptions},
		{"nil cannot be repaired", []dijkstra.SearchOption{dijkstra.WithHeuristic(nil), dijkstra.WithHeuristic(zero), dijkstra.WithSearchMethod(dijkstra.SearchAStar)}, dijkstra.ErrNilHeuristic},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := dijkstra.Search(graph, "S", "T", tc.opts...)
			if got != nil || !errors.Is(err, tc.want) {
				t.Fatalf("result=%+v err=%v, want errors.Is(%v)", got, err, tc.want)
			}
		})
	}
	for _, options := range [][]dijkstra.SearchOption{
		{dijkstra.WithSearchMethod(dijkstra.SearchAStar), dijkstra.WithHeuristic(zero)},
		{dijkstra.WithHeuristic(zero), dijkstra.WithSearchMethod(dijkstra.SearchAStar)},
	} {
		got, err := dijkstra.Search(graph, "S", "T", options...)
		if err != nil || got == nil || got.Method != dijkstra.SearchAStar || got.Distance != 2 {
			t.Fatalf("valid order unexpectedly rejected: result=%+v err=%v", got, err)
		}
	}
	if got, err := dijkstra.AStar(graph, "S", "T", nil); got != nil || !errors.Is(err, dijkstra.ErrNilHeuristic) {
		t.Fatalf("nil direct callback: result=%+v err=%v", got, err)
	}

	for _, bad := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		estimate := func(id string) (float64, error) {
			if id == "S" {
				return bad, nil
			}
			return 0, nil
		}
		if got, err := dijkstra.AStar(graph, "S", "T", estimate); got != nil || !errors.Is(err, dijkstra.ErrInvalidHeuristic) {
			t.Fatalf("non-finite heuristic=%g: result=%+v err=%v", bad, got, err)
		}
	}
	if got, err := dijkstra.AStar(graph, "S", "T", func(string) (float64, error) { return 1, nil }); got != nil || !errors.Is(err, dijkstra.ErrInvalidHeuristic) {
		t.Fatalf("nonzero terminal estimate accepted: result=%+v err=%v", got, err)
	}
	if got, err := dijkstra.AStar(graph, "S", "T", func(id string) (float64, error) {
		if id == "S" {
			return 3, nil
		}
		return 0, nil
	}); got != nil || !errors.Is(err, dijkstra.ErrInconsistentHeuristic) {
		t.Fatalf("h(S)=3 across S->T=2 accepted: result=%+v err=%v", got, err)
	}

	// A valid negative heuristic remains supported; it is not mistaken for
	// negative original edge weight or an invalid result distance.
	negative := func(id string) (float64, error) {
		if id == "S" {
			return -5, nil
		}
		return 0, nil
	}
	got, err := dijkstra.AStar(graph, "S", "T", negative)
	if err != nil || got == nil || got.Distance != 2 {
		t.Fatalf("valid signed potential rejected: result=%+v err=%v", got, err)
	}
}

// TestAStar_CallbackCauseAndCancellation preserves errors.Is for user causes.
// Cancellation during heuristic freezing returns nil result and is not unreachable.
func TestAStar_CallbackCauseAndCancellation(t *testing.T) {
	graph := searchTestGraph(t, core.WithDirected(true))
	searchTestEdge(t, graph, "S", "T", "direct", 1)
	callbackCause := errors.New("heuristic provider failed")
	callback := func(string) (float64, error) { return 0, callbackCause }
	if got, err := dijkstra.AStar(graph, "S", "T", callback); got != nil || !errors.Is(err, callbackCause) {
		t.Fatalf("callback cause lost: result=%+v err=%v", got, err)
	}

	ctx, cancel := context.WithCancelCause(context.Background())
	stopReason := errors.New("estimate data revoked")
	calls := 0
	canceling := func(string) (float64, error) {
		calls++
		cancel(stopReason)
		return 0, nil
	}
	got, err := dijkstra.AStar(graph, "S", "T", canceling, dijkstra.WithSearchContext(ctx))
	if got != nil || calls != 1 || !errors.Is(err, context.Canceled) || !errors.Is(err, stopReason) {
		t.Fatalf("mid-preprocessing cancellation: result=%+v calls=%d err=%v", got, calls, err)
	}
}

// TestAStar_OriginalCutoffAndWallConsistency keeps policy in original-cost units.
// The heuristic is certified only on admitted effective arcs after global validation.
func TestAStar_OriginalCutoffAndWallConsistency(t *testing.T) {
	graph := searchTestGraph(t, core.WithDirected(true))
	searchTestEdge(t, graph, "S", "A", "blocked", 10)
	searchTestEdge(t, graph, "A", "T", "finish", 1)
	values := map[string]float64{"S": 20, "A": 0, "T": 0}
	estimate := func(id string) (float64, error) { return values[id], nil }
	out, err := dijkstra.AStar(graph, "S", "T", estimate, dijkstra.WithSearchInfEdgeThreshold(5))
	if err != nil || out == nil || out.Reachable || !math.IsInf(out.Distance, 1) {
		t.Fatalf("wall/heuristic effective graph mismatch: result=%+v err=%v", out, err)
	}
	if got, err := dijkstra.AStar(graph, "S", "T", estimate); got != nil || !errors.Is(err, dijkstra.ErrInconsistentHeuristic) {
		t.Fatalf("unblocked inconsistent arc ignored: result=%+v err=%v", got, err)
	}

	fractional := searchTestGraph(t, core.WithDirected(true))
	searchTestEdge(t, fractional, "S", "A", "first", 0.9)
	searchTestEdge(t, fractional, "A", "T", "second", 0.1)
	zero := func(string) (float64, error) { return 0, nil }
	blocked, err := dijkstra.AStar(fractional, "S", "T", zero, dijkstra.WithSearchMaxDistance(1))
	if err != nil || blocked == nil || blocked.Reachable {
		t.Fatalf("exact original-cost cutoff: result=%+v err=%v", blocked, err)
	}
	if got, err := dijkstra.AStar(fractional, "S", "T", zero); got != nil || !errors.Is(err, dijkstra.ErrInexactArithmetic) {
		t.Fatalf("inexact admitted original path not refused: result=%+v err=%v", got, err)
	}
}

// TestAStar_SignedOffsetAndKeyErrors separates original-distance and key domains.
// The same finite graph weight can be legal while its normalized priority overflows.
func TestAStar_SignedOffsetAndKeyErrors(t *testing.T) {
	const hugeExponent = 53
	t.Run("inexact offset", func(t *testing.T) {
		graph := searchTestGraph(t, core.WithDirected(true))
		searchTestEdge(t, graph, "S", "A", "start", 1)
		searchTestEdge(t, graph, "A", "T", "end", 1)
		values := map[string]float64{"S": 1, "A": 0.1, "T": 0}
		got, err := dijkstra.AStar(graph, "S", "T", func(id string) (float64, error) { return values[id], nil })
		if got != nil || !errors.Is(err, dijkstra.ErrInexactArithmetic) {
			t.Fatalf("inexact offset admitted: result=%+v err=%v", got, err)
		}
	})
	t.Run("inexact priority", func(t *testing.T) {
		graph := searchTestGraph(t, core.WithDirected(true))
		searchTestEdge(t, graph, "S", "T", "direct", 1)
		got, err := dijkstra.AStar(graph, "S", "T", func(id string) (float64, error) {
			if id == "S" {
				return -math.Ldexp(1, hugeExponent), nil
			}
			return 0, nil
		})
		if got != nil || !errors.Is(err, dijkstra.ErrInexactArithmetic) {
			t.Fatalf("inexact g+offset admitted: result=%+v err=%v", got, err)
		}
	})
	t.Run("priority overflow", func(t *testing.T) {
		graph := searchTestGraph(t, core.WithDirected(true))
		searchTestEdge(t, graph, "S", "T", "maximum", math.MaxFloat64)
		got, err := dijkstra.AStar(graph, "S", "T", func(id string) (float64, error) {
			if id == "S" {
				return -math.MaxFloat64, nil
			}
			return 0, nil
		})
		if got != nil || !errors.Is(err, dijkstra.ErrSearchKeyOverflow) {
			t.Fatalf("overflowing queue key admitted: result=%+v err=%v", got, err)
		}
	})
	t.Run("offset overflow on disconnected vertex", func(t *testing.T) {
		graph := searchTestGraph(t, core.WithDirected(true))
		searchTestEdge(t, graph, "S", "T", "direct", 1)
		if err := graph.AddVertex("Z"); err != nil {
			t.Fatal(err)
		}
		values := map[string]float64{"S": -math.MaxFloat64, "T": 0, "Z": math.MaxFloat64}
		got, err := dijkstra.AStar(graph, "S", "T", func(id string) (float64, error) { return values[id], nil })
		if got != nil || !errors.Is(err, dijkstra.ErrSearchKeyOverflow) {
			t.Fatalf("offset overflow ignored: result=%+v err=%v", got, err)
		}
	})
}

// TestAStar_ExactSignedInequalityRationalOracle compares actual binary values.
// It guards the consistency predicate across signed, subnormal and overflow scales.
// Implementation:
//   - Interpret every finite float64 bit pattern as an exact rational value.
//   - Compare h(from)<=weight+h(to) without calling the production sum helper.
//   - Independently certify that successful signed key materialization is exact.
//
// Behavior highlights:
//   - Rounded equality does not imply exact consistency; positive overflow does.
//
// Determinism:
//   - A fixed enumerated scalar domain replaces probabilistic tolerance checks.
//
// Complexity:
//   - Test-only rational operations are excluded from production hot loops.
//
// AI-Hints:
//   - A float64 expression on both sides of a test is not an independent oracle.
func TestAStar_ExactSignedInequalityRationalOracle(t *testing.T) {
	values := []float64{
		-math.MaxFloat64, -math.Ldexp(1, 53), -1, -math.Nextafter(1, 0),
		-0.9, -math.SmallestNonzeroFloat64, 0, math.SmallestNonzeroFloat64,
		math.Ldexp(1, -54), 0.1, 0.9, math.Nextafter(1, 0), 1,
		math.Ldexp(1, 53), math.MaxFloat64,
	}
	weights := []float64{0, math.SmallestNonzeroFloat64, math.Ldexp(1, -54), 0.1, 0.9, 1, math.MaxFloat64}
	for _, from := range values {
		for _, to := range values {
			for _, weight := range weights {
				bound := new(big.Rat).SetFloat64(from)
				right := new(big.Rat).Add(new(big.Rat).SetFloat64(weight), new(big.Rat).SetFloat64(to))
				want := bound.Cmp(right) <= 0
				if got := dijkstra.SearchTestHeuristicConsistent(from, weight, to); got != want {
					t.Fatalf("inexact comparison: %g <= %g+%g: got=%v want=%v", from, weight, to, got, want)
				}
			}
		}
	}

	// The same signed-sum certificate is used by offsets and heap keys.
	for _, left := range values {
		for _, right := range values {
			exact := new(big.Rat).Add(new(big.Rat).SetFloat64(left), new(big.Rat).SetFloat64(right))
			got, err := dijkstra.SearchTestCertifyKey(left, right)
			if err == nil && new(big.Rat).SetFloat64(got).Cmp(exact) != 0 {
				t.Fatalf("key materialization changed represented sum %g+%g=%g", left, right, got)
			}
			if err != nil && !errors.Is(err, dijkstra.ErrSearchKeyOverflow) && !errors.Is(err, dijkstra.ErrInexactArithmetic) {
				t.Fatalf("wrong key numeric class: %g+%g: %v", left, right, err)
			}
		}
	}
}

// TestAStar_IndependentMixedMultigraphOracle validates optimal original costs.
// A target-specific integer Floyd-Warshall potential is consistent on every arc.
// Implementation:
//   - Construct reproducible directed/undirected, parallel and loop-bearing graphs.
//   - Compute all-pairs integer distances independently of the A* adapter.
//   - Supply exact distances-to-target as a frozen callback for each query.
//   - Verify reachability, optimum cost and every original edge in the witness.
//
// Behavior highlights:
//   - Finite integer costs avoid false test tolerance and key-rounding ambiguity.
//   - Unreachable components use zero estimates, consistent with their own edges.
//
// Determinism:
//   - Fixed generator seed and complete ordered source-target enumeration.
//
// Complexity:
//   - Test-only O(V^3) oracle work is never charged to the production kernel.
//
// AI-Hints:
//   - Do not build the oracle from searchGraph or reuse runTarget distances.
func TestAStar_IndependentMixedMultigraphOracle(t *testing.T) {
	const (
		seed        int64 = 20261030
		caseCount         = 35
		unreachable int64 = 1 << 55
	)
	rng := rand.New(rand.NewSource(seed))
	for caseIndex := 0; caseIndex < caseCount; caseIndex++ {
		vertexCount := 2 + rng.Intn(5)
		graph, err := core.NewMixedGraph(core.WithWeighted(), core.WithDirected(true),
			core.WithMultiEdges(), core.WithLoops())
		if err != nil {
			t.Fatal(err)
		}
		ids := make([]string, vertexCount)
		for vertex := range ids {
			ids[vertex] = fmt.Sprintf("v%d", vertex)
			if err := graph.AddVertex(ids[vertex]); err != nil {
				t.Fatal(err)
			}
		}
		edges := make([]oracleArc, 0, vertexCount*4)
		for edgeIndex := 0; edgeIndex < vertexCount*4; edgeIndex++ {
			from, to := rng.Intn(vertexCount), rng.Intn(vertexCount)
			weight := int64(rng.Intn(8))
			directed := rng.Intn(2) == 0
			searchTestEdge(t, graph, ids[from], ids[to],
				fmt.Sprintf("g%d-e%d", caseIndex, edgeIndex), float64(weight),
				core.WithEdgeDirected(directed))
			edges = append(edges, oracleArc{from: from, to: to, weight: weight, directed: directed})
		}
		oracle := oracleDistances(vertexCount, edges)
		for target := range ids {
			potential := make(map[string]float64, vertexCount)
			for vertex := range ids {
				// Unreachable-to-target components cannot have outgoing arcs to
				// a finite-distance component, so zero is consistent there.
				if oracle[vertex][target] != unreachable {
					potential[ids[vertex]] = float64(oracle[vertex][target])
				}
			}
			callback := func(id string) (float64, error) { return potential[id], nil }
			for source := range ids {
				got, err := dijkstra.AStar(graph, ids[source], ids[target], callback)
				if err != nil {
					t.Fatalf("case=%d %s->%s: %v", caseIndex, ids[source], ids[target], err)
				}
				searchTestWitness(t, graph, got)
				want := oracle[source][target]
				if want == unreachable {
					if got.Reachable || !math.IsInf(got.Distance, 1) {
						t.Fatalf("case=%d %s->%s should be unreachable: %+v", caseIndex, ids[source], ids[target], got)
					}
				} else if !got.Reachable || got.Distance != float64(want) || got.Method != dijkstra.SearchAStar {
					t.Fatalf("case=%d %s->%s want=%d, got=%+v", caseIndex, ids[source], ids[target], want, got)
				}
			}
		}
	}
}

// TestAStar_WrapperOptionOwnershipAndSelfValidation verifies that argument
// overrides neither mutate caller capacity nor skip terminal heuristic validation.
func TestAStar_WrapperOptionOwnershipAndSelfValidation(t *testing.T) {
	graph := searchTestGraph(t, core.WithDirected(true))
	searchTestEdge(t, graph, "S", "T", "one", 1)
	zero := func(string) (float64, error) { return 0, nil }
	opts := make([]dijkstra.SearchOption, 1, 4)
	opts[0] = dijkstra.WithSearchMethod(dijkstra.SearchBidirectional)
	backing := opts[:cap(opts)]
	sentinel := errors.New("caller option storage used")
	backing[1] = func(*dijkstra.SearchOptions) error { return sentinel }
	got, err := dijkstra.AStar(graph, "S", "T", zero, opts...)
	if err != nil || got == nil || got.Distance != 1 || got.Method != dijkstra.SearchAStar {
		t.Fatalf("wrapper override failed: result=%+v err=%v", got, err)
	}
	if err := backing[1](&dijkstra.SearchOptions{}); !errors.Is(err, sentinel) {
		t.Fatalf("wrapper damaged caller backing store: %v", err)
	}
	if result, err := dijkstra.AStar(graph, "S", "S", func(string) (float64, error) { return 1, nil }); result != nil || !errors.Is(err, dijkstra.ErrInvalidHeuristic) {
		t.Fatalf("self path masked invalid terminal anchor: result=%+v err=%v", result, err)
	}
}

// TestAStar_StableEqualKeyWitness verifies deterministic ties in the A* queue.
// Original edge IDs and lexical vertex ordering, not insertion order, select the witness.
func TestAStar_StableEqualKeyWitness(t *testing.T) {
	build := func(reverse bool) *core.Graph {
		graph := searchTestGraph(t, core.WithDirected(true))
		edges := []struct {
			from, to, id string
		}{
			{"S", "A", "startA"}, {"S", "B", "startB"},
			{"A", "T", "finishA"}, {"B", "T", "finishB"},
		}
		for index := range edges {
			selected := index
			if reverse {
				selected = len(edges) - 1 - index
			}
			edge := edges[selected]
			searchTestEdge(t, graph, edge.from, edge.to, edge.id, 1)
		}
		return graph
	}
	values := map[string]float64{"S": 2, "A": 1, "B": 1, "T": 0}
	estimate := func(id string) (float64, error) { return values[id], nil }
	first, err := dijkstra.AStar(build(false), "S", "T", estimate)
	if err != nil {
		t.Fatal(err)
	}
	second, err := dijkstra.AStar(build(true), "S", "T", estimate)
	if err != nil {
		t.Fatal(err)
	}
	if first.Distance != 2 || second.Distance != 2 ||
		!slices.Equal(first.EdgeIDs, second.EdgeIDs) ||
		!slices.Equal(first.Vertices, second.Vertices) ||
		len(first.EdgeIDs) != 2 || first.EdgeIDs[0] != "startA" || first.EdgeIDs[1] != "finishA" {
		t.Fatalf("equal priority changed stable witness: first=%+v second=%+v", first, second)
	}
}
