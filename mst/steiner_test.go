// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mst_test exercises public SteinerTree and its original-edge contract.
// The cases anchor actual KMB behavior rather than asserting an arbitrary optimal representative.
package mst_test

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/mst"
)

// steinerFixtureEdge adds one original named edge, failing at construction time.
// It ensures rejected input setup cannot silently degrade a mathematical test fixture.
func steinerFixtureEdge(t *testing.T, g *core.Graph, id, from, to string, weight float64) {
	t.Helper()
	gotID, err := g.AddEdge(from, to, weight, core.WithID(id))
	if err != nil || gotID != id {
		t.Fatalf("AddEdge(%q,%q,%q,%g): id=%q error=%v", id, from, to, weight, gotID, err)
	}
}

// steinerFixtureGraph constructs a weighted graph with explicit policy flags.
// Its returned source graph is owned only by the current test.
func steinerFixtureGraph(t *testing.T, extras ...core.GraphOption) *core.Graph {
	t.Helper()
	options := append([]core.GraphOption{core.WithWeighted()}, extras...)
	graph, err := core.NewGraph(options...)
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	return graph
}

// TestSteinerTree_HubOriginalIDsAndMemoryModes protects the complete KMB pipeline.
// Closure cost four must not be mistaken for the three-edge final network cost three.
func TestSteinerTree_HubOriginalIDsAndMemoryModes(t *testing.T) {
	g := steinerFixtureGraph(t)
	for _, e := range []struct {
		id, from, to string
		weight       float64
	}{
		{"AX", "A", "X", 1}, {"BX", "B", "X", 1}, {"CX", "C", "X", 1},
		{"AB", "A", "B", 3}, {"BC", "B", "C", 3}, {"AC", "A", "C", 3},
	} {
		steinerFixtureEdge(t, g, e.id, e.from, e.to, e.weight)
	}
	terminals := []string{"C", "A", "B"}
	for _, mode := range []mst.SteinerPathStorage{mst.SteinerRecomputePaths, mst.SteinerRetainPaths} {
		t.Run(map[mst.SteinerPathStorage]string{mst.SteinerRecomputePaths: "recompute", mst.SteinerRetainPaths: "retain"}[mode], func(t *testing.T) {
			result, err := mst.SteinerTree(g, terminals, mst.WithSteinerPathStorage(mode))
			if err != nil {
				t.Fatalf("SteinerTree: %v", err)
			}
			if result.TotalWeight != 3 || len(result.Edges) != 3 {
				t.Fatalf("network cost/size = %g/%d, want 3/3", result.TotalWeight, len(result.Edges))
			}
			if !slices.Equal(result.Terminals, []string{"A", "B", "C"}) ||
				!slices.Equal(result.Vertices, []string{"A", "B", "C", "X"}) {
				t.Fatalf("terminal/network vertices: %+v", result)
			}
			for index, wantID := range []string{"AX", "BX", "CX"} {
				if result.Edges[index].ID != wantID {
					t.Fatalf("edge[%d]=%q, want %q", index, result.Edges[index].ID, wantID)
				}
			}
		})
	}
	if !slices.Equal(terminals, []string{"C", "A", "B"}) || g.EdgeCount() != 6 || g.VertexCount() != 4 {
		t.Fatal("SteinerTree changed caller terminal order or source graph topology")
	}
}

// TestSteinerTree_UnionMSTAndLeafPruning protects both mandatory KMB completion stages.
// Two distinct shortest paths form a cycle; union MST leaves a useless nonterminal leaf Y.
func TestSteinerTree_UnionMSTAndLeafPruning(t *testing.T) {
	g := steinerFixtureGraph(t)
	for _, edge := range []struct {
		id, from, to string
		weight       float64
	}{
		{"AX", "A", "X", 100},
		{"CX", "C", "X", 100},
		{"XY", "X", "Y", 1},
		{"BY", "B", "Y", 10},
		{"XZ", "X", "Z", 5},
		{"BZ", "B", "Z", 6},
	} {
		steinerFixtureEdge(t, g, edge.id, edge.from, edge.to, edge.weight)
	}
	for _, mode := range []mst.SteinerPathStorage{mst.SteinerRecomputePaths, mst.SteinerRetainPaths} {
		result, err := mst.SteinerTree(g, []string{"A", "B", "C"}, mst.WithSteinerPathStorage(mode))
		if err != nil {
			t.Fatalf("mode=%d: %v", mode, err)
		}
		if result.TotalWeight != 211 || !slices.Equal(result.Vertices, []string{"A", "B", "C", "X", "Z"}) {
			t.Fatalf("mode=%d: unpruned network: %+v", mode, result)
		}
		if !slices.Equal([]string{result.Edges[0].ID, result.Edges[1].ID,
			result.Edges[2].ID, result.Edges[3].ID}, []string{"AX", "BZ", "CX", "XZ"}) {
			t.Fatalf("mode=%d: wrong original edges: %+v", mode, result.Edges)
		}
	}
}

// TestSteinerTree_SingletonAndTwoTerminalPath locks degenerate mathematical reductions.
// Both paths must perform admission first and publish only original selected vertices.
func TestSteinerTree_SingletonAndTwoTerminalPath(t *testing.T) {
	g := steinerFixtureGraph(t)
	steinerFixtureEdge(t, g, "AX", "X", "A", 1)
	steinerFixtureEdge(t, g, "XC", "C", "X", 2)
	steinerFixtureEdge(t, g, "AC", "A", "C", 5)
	if err := g.AddVertex("Z"); err != nil {
		t.Fatal(err)
	}

	singleton, err := mst.SteinerTree(g, []string{"Z"})
	if err != nil || singleton.TotalWeight != 0 || len(singleton.Edges) != 0 ||
		!slices.Equal(singleton.Vertices, []string{"Z"}) ||
		!slices.Equal(singleton.Terminals, []string{"Z"}) {
		t.Fatalf("singleton=%+v err=%v", singleton, err)
	}
	for _, storage := range []mst.SteinerPathStorage{mst.SteinerRecomputePaths, mst.SteinerRetainPaths} {
		result, err := mst.SteinerTree(g, []string{"C", "A"}, mst.WithSteinerPathStorage(storage))
		if err != nil || result.TotalWeight != 3 ||
			!slices.Equal(result.Vertices, []string{"A", "C", "X"}) {
			t.Fatalf("pair mode=%v result=%+v err=%v", storage, result, err)
		}
		if !slices.Equal([]string{result.Edges[0].ID, result.Edges[1].ID}, []string{"AX", "XC"}) {
			t.Fatalf("pair original IDs: %+v", result.Edges)
		}
	}
}

// TestSteinerTree_IslandsAndDisconnectedTerminals separates full-graph spanning from terminal connectivity.
// A valid unrelated island is allowed, but selecting a terminal on that island must fail.
func TestSteinerTree_IslandsAndDisconnectedTerminals(t *testing.T) {
	g := steinerFixtureGraph(t)
	steinerFixtureEdge(t, g, "ab", "A", "B", 2)
	steinerFixtureEdge(t, g, "xy", "X", "Y", 1)
	res, err := mst.SteinerTree(g, []string{"B", "A"})
	if err != nil || res.TotalWeight != 2 || len(res.Vertices) != 2 {
		t.Fatalf("independent island: result=%+v err=%v", res, err)
	}
	for _, storage := range []mst.SteinerPathStorage{mst.SteinerRecomputePaths, mst.SteinerRetainPaths} {
		res, err = mst.SteinerTree(g, []string{"A", "Y"}, mst.WithSteinerPathStorage(storage))
		if res != nil || !errors.Is(err, mst.ErrDisconnectedTerminals) {
			t.Fatalf("disconnected terminals: result=%+v error=%v", res, err)
		}
	}
}

// TestSteinerTree_ParallelEdgeIDsAndZeroWeights protects original edge identity.
// Endpoint orientation and equal-cost parallel edges cannot erase the canonical ID witness.
func TestSteinerTree_ParallelEdgeIDsAndZeroWeights(t *testing.T) {
	g := steinerFixtureGraph(t, core.WithMultiEdges())
	steinerFixtureEdge(t, g, "edge-z", "B", "A", 0)
	steinerFixtureEdge(t, g, "edge-a", "A", "B", 0)
	steinerFixtureEdge(t, g, "edge-heavy", "A", "B", 4)
	steinerFixtureEdge(t, g, "BC", "C", "B", 1)
	result, err := mst.SteinerTree(g, []string{"A", "C"})
	if err != nil || result.TotalWeight != 1 || len(result.Edges) != 2 || result.Edges[1].ID != "edge-a" {
		t.Fatalf("parallel witness: result=%+v err=%v", result, err)
	}
	for repeat := 0; repeat < 4; repeat++ {
		current, err := mst.SteinerTree(g, []string{"A", "C"})
		if err != nil || !slices.Equal(result.Edges, current.Edges) {
			t.Fatalf("nondeterministic parallel witness repeat=%d result=%+v err=%v", repeat, current, err)
		}
	}
}

// TestSteinerTree_TerminalErrorsAndOptions fixes the public admission precedence.
// Invalid options are checked before graph admission, and missing IDs keep core identities.
func TestSteinerTree_TerminalErrorsAndOptions(t *testing.T) {
	g := steinerFixtureGraph(t)
	steinerFixtureEdge(t, g, "AB", "A", "B", 1)
	cases := []struct {
		name      string
		graph     *core.Graph
		terminals []string
		opts      []mst.SteinerOption
		want      error
	}{
		{"nilGraph", nil, []string{"A"}, nil, mst.ErrNilGraph},
		{"emptyTerminals", g, nil, nil, mst.ErrEmptyTerminals},
		{"emptyID", g, []string{""}, nil, core.ErrEmptyVertexID},
		{"duplicateID", g, []string{"B", "A", "A"}, nil, mst.ErrDuplicateTerminal},
		{"unknownID", g, []string{"A", "missing"}, nil, core.ErrVertexNotFound},
		{"nilOption", nil, nil, []mst.SteinerOption{nil}, mst.ErrNilOption},
		{"nilContext", g, []string{"A"}, []mst.SteinerOption{mst.WithSteinerContext(nil)}, mst.ErrInvalidOption},
		{"badMode", g, []string{"A"}, []mst.SteinerOption{mst.WithSteinerPathStorage(255)}, mst.ErrInvalidOption},
		{"verticesCap", g, []string{"A"}, []mst.SteinerOption{mst.WithSteinerMaxVertices(1)}, mst.ErrResourceLimit},
		{"terminalsCap", g, []string{"A", "B"}, []mst.SteinerOption{mst.WithSteinerMaxTerminals(1)}, mst.ErrResourceLimit},
	}
	for _, tc := range cases {
		if tc.want == nil {
			continue
		}
		t.Run(tc.name, func(t *testing.T) {
			result, err := mst.SteinerTree(tc.graph, tc.terminals, tc.opts...)
			if result != nil || !errors.Is(err, tc.want) {
				t.Fatalf("result=%+v error=%v, want %v", result, err, tc.want)
			}
		})
	}
	if _, err := mst.SteinerTree(g, []string{"A"}, mst.WithSteinerMaxEdges(1)); err != nil {
		t.Fatalf("exact edge cap rejected: %v", err)
	}
	if _, err := mst.SteinerTree(g, []string{"A"}, mst.WithSteinerMaxEdges(-1)); !errors.Is(err, mst.ErrInvalidOption) {
		t.Fatalf("negative cap not rejected: %v", err)
	}
	if _, err := mst.SteinerTree(g, []string{"A"}, mst.WithSteinerMaxEdges(0)); err != nil {
		t.Fatalf("zero edge cap must mean unset: %v", err)
	}
	steinerFixtureEdge(t, g, "BC", "B", "C", 1)
	if result, err := mst.SteinerTree(g, []string{"A"}, mst.WithSteinerMaxEdges(1)); result != nil || !errors.Is(err, mst.ErrResourceLimit) {
		t.Fatalf("edge cap result=%+v err=%v", result, err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	result, err := mst.SteinerTree(g, []string{"A"}, mst.WithSteinerContext(ctx))
	if result != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation: result=%+v error=%v", result, err)
	}
}

// TestSteinerTree_RejectsInvalidEdgesEvenOutsideSelectedNetwork protects global admission.
// Self-loops and irrelevant islands are validated before excluding them from KMB candidates.
func TestSteinerTree_RejectsInvalidEdgesEvenOutsideSelectedNetwork(t *testing.T) {
	for _, tc := range []struct {
		name     string
		weight   float64
		sentinel error
	}{
		{"negative-loop", -1, mst.ErrNegativeWeight},
		{"negative-island", -4, mst.ErrNegativeWeight},
		{"nan-loop", math.NaN(), mst.ErrNaNInfWeight},
		{"inf-island", math.Inf(1), mst.ErrNaNInfWeight},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := steinerFixtureGraph(t, core.WithLoops())
			steinerFixtureEdge(t, g, "AB", "A", "B", 1)
			isLoop := tc.name == "negative-loop" || tc.name == "nan-loop"
			from, to := "X", "Y"
			if isLoop {
				to = "X"
			}
			weight := tc.weight
			if math.IsNaN(weight) || math.IsInf(weight, 0) {
				weight = 1
			}
			steinerFixtureEdge(t, g, "bad", from, to, weight)
			if math.IsNaN(tc.weight) || math.IsInf(tc.weight, 0) {
				pointer, err := g.GetEdge("bad")
				if err != nil {
					t.Fatal(err)
				}
				// Defensive corruption tests intentionally exercise the graph's live-edge
				// alias; normal core.AddEdge forbids non-finite weights on insertion.
				pointer.Weight = tc.weight
			}
			result, err := mst.SteinerTree(g, []string{"A", "B"})
			if result != nil || !errors.Is(err, tc.sentinel) {
				t.Fatalf("result=%+v err=%v, want %v", result, err, tc.sentinel)
			}
		})
	}
}

// TestSteinerTree_DirectedGraphPoliciesReuseMSTSentinels anchors undirected admission.
// Graph-level and edge-level direction remain separate errors with the old broad sentinel.
func TestSteinerTree_DirectedGraphPoliciesReuseMSTSentinels(t *testing.T) {
	directed := steinerFixtureGraph(t, core.WithDirected(true))
	steinerFixtureEdge(t, directed, "AB", "A", "B", 1)
	if result, err := mst.SteinerTree(directed, []string{"A", "B"}); result != nil || !errors.Is(err, mst.ErrInvalidGraph) || !errors.Is(err, mst.ErrDirectedGraph) {
		t.Fatalf("directed graph result=%+v err=%v", result, err)
	}

	mixed := steinerFixtureGraph(t, core.WithMixedEdges())
	if _, err := mixed.AddEdge("A", "B", 1, core.WithID("AB"), core.WithEdgeDirected(true)); err != nil {
		t.Fatal(err)
	}
	if result, err := mst.SteinerTree(mixed, []string{"A", "B"}); result != nil || !errors.Is(err, mst.ErrInvalidGraph) || !errors.Is(err, mst.ErrDirectedEdge) {
		t.Fatalf("directed edge result=%+v err=%v", result, err)
	}
}

// TestSteinerTree_OverflowBoundaries distinguishes unnecessary closure arithmetic from final publication.
// A three-spoke huge star succeeds; a four-spoke selected network cannot publish a finite total.
func TestSteinerTree_OverflowBoundaries(t *testing.T) {
	const w = 0x1p1022
	for _, leaves := range [][]string{{"A", "B", "C"}, {"A", "B", "C", "D"}} {
		g := steinerFixtureGraph(t)
		for _, terminal := range leaves {
			steinerFixtureEdge(t, g, terminal+"X", terminal, "X", w)
		}
		for _, mode := range []mst.SteinerPathStorage{mst.SteinerRecomputePaths, mst.SteinerRetainPaths} {
			result, err := mst.SteinerTree(g, leaves, mst.WithSteinerPathStorage(mode))
			if len(leaves) == 3 {
				if err != nil || result.TotalWeight != 3*w || len(result.Edges) != 3 {
					t.Fatalf("finite star mode=%d result=%+v err=%v", mode, result, err)
				}
			} else if result != nil || !errors.Is(err, mst.ErrWeightOverflow) || errors.Is(err, mst.ErrDisconnectedTerminals) {
				t.Fatalf("overflow star mode=%d result=%+v err=%v", mode, result, err)
			}
		}
	}
}

// TestSteinerTree_DetachedOwnershipAndNilHelpers protects result slice boundaries.
// Caller-owned values may be changed without mutating the graph or other result copies.
func TestSteinerTree_DetachedOwnershipAndNilHelpers(t *testing.T) {
	var nilResult *mst.SteinerResult
	if !nilResult.IsNil() || nilResult.Clone() != nil {
		t.Fatal("nil result helpers do not follow the MST convention")
	}
	if edges, err := nilResult.EdgeValues(); edges != nil || !errors.Is(err, mst.ErrNilResult) {
		t.Fatalf("nil EdgeValues: edges=%v err=%v", edges, err)
	}
	g := steinerFixtureGraph(t)
	steinerFixtureEdge(t, g, "ab", "A", "B", 1)
	result, err := mst.SteinerTree(g, []string{"A", "B"})
	if err != nil {
		t.Fatal(err)
	}
	copyResult := result.Clone()
	edges, err := result.EdgeValues()
	if err != nil {
		t.Fatal(err)
	}
	edges[0].Weight = 99
	copyResult.Edges[0].Weight = 88
	copyResult.Terminals[0] = "changed"
	copyResult.Vertices[0] = "changed"
	if result.Edges[0].Weight != 1 || result.Terminals[0] != "A" || result.Vertices[0] != "A" {
		t.Fatalf("result slices were aliased: %+v", result)
	}
	if got, err := g.GetEdge("ab"); err != nil || got.Weight != 1 {
		t.Fatalf("source graph was mutated: edge=%+v err=%v", got, err)
	}
}

// TestSteinerTree_PathOrderAndNetworkOrderAreDistinctNumericContracts prevents a false equality.
// A shortest-path prefix fold and a canonical ID-ordered network fold can round differently.
func TestSteinerTree_PathOrderAndNetworkOrderAreDistinctNumericContracts(t *testing.T) {
	g := steinerFixtureGraph(t)
	steinerFixtureEdge(t, g, "z-heavy", "A", "X", 1e16)
	steinerFixtureEdge(t, g, "a-unit", "X", "Y", 1)
	steinerFixtureEdge(t, g, "b-unit", "Y", "B", 1)
	result, err := mst.SteinerTree(g, []string{"A", "B"})
	if err != nil || result.TotalWeight != 10000000000000002 {
		t.Fatalf("canonical fold result=%+v err=%v", result, err)
	}
}

// TestMST_CheckedTotalsPreserveStructuralErrorPriority protects ordinary Prim and Kruskal.
// A disconnected strict graph reports topology first; forest publication checks overflow.
func TestMST_CheckedTotalsPreserveStructuralErrorPriority(t *testing.T) {
	g := steinerFixtureGraph(t)
	steinerFixtureEdge(t, g, "AB", "A", "B", math.MaxFloat64)
	steinerFixtureEdge(t, g, "BC", "B", "C", math.MaxFloat64)
	if err := g.AddVertex("Z"); err != nil {
		t.Fatal(err)
	}
	for _, algo := range []mst.Algorithm{mst.AlgorithmKruskal, mst.AlgorithmPrim} {
		opts := []mst.Option{mst.WithAlgorithm(algo)}
		if algo == mst.AlgorithmPrim {
			opts = append(opts, mst.WithRoot("A"))
		}
		result, err := mst.MinimumSpanningTree(g, opts...)
		if result != nil || !errors.Is(err, mst.ErrDisconnected) || errors.Is(err, mst.ErrWeightOverflow) {
			t.Fatalf("strict %s: result=%+v err=%v", algo, result, err)
		}
		forestOpts := append(append([]mst.Option{}, opts...), mst.WithForest())
		result, err = mst.MinimumSpanningTree(g, forestOpts...)
		if result != nil || !errors.Is(err, mst.ErrWeightOverflow) {
			t.Fatalf("forest %s: result=%+v err=%v", algo, result, err)
		}
	}

	negative := steinerFixtureGraph(t)
	steinerFixtureEdge(t, negative, "ab", "A", "B", -3)
	steinerFixtureEdge(t, negative, "bc", "B", "C", -1)
	for _, algo := range []mst.Algorithm{mst.AlgorithmKruskal, mst.AlgorithmPrim} {
		opts := []mst.Option{mst.WithAlgorithm(algo)}
		if algo == mst.AlgorithmPrim {
			opts = append(opts, mst.WithRoot("C"))
		}
		result, err := mst.MinimumSpanningTree(negative, opts...)
		if err != nil || result.TotalWeight != -4 {
			t.Fatalf("negative ordinary %s: result=%+v err=%v", algo, result, err)
		}
	}
}
