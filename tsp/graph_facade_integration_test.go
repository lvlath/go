// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp_test validates the real graph-to-matrix routing adapter.
// It protects edge-distance semantics and stable vertex-ID projection.
package tsp_test

import (
	"errors"
	"fmt"
	"slices"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/tsp"
)

// testRoutingGraph builds five named service hubs using only public core.Graph APIs.
// The connected-chain mode requires explicit metric closure before route construction.
func testRoutingGraph(t *testing.T, complete bool) *core.Graph {
	t.Helper()
	graph, err := core.NewGraph(core.WithWeighted())
	if err != nil {
		t.Fatalf("core.NewGraph: %v", err)
	}
	names := []string{"dock-0", "dock-1", "dock-2", "dock-3", "dock-4"}
	for _, id := range names {
		if err := graph.AddVertex(id); err != nil {
			t.Fatalf("AddVertex(%s): %v", id, err)
		}
	}
	for i := range names {
		for j := i + 1; j < len(names); j++ {
			if !complete && j != i+1 {
				continue
			}
			if _, err := graph.AddEdge(names[i], names[j], float64(j-i)); err != nil {
				t.Fatalf("AddEdge %s-%s: %v", names[i], names[j], err)
			}
		}
	}
	return graph
}

// TestTSPGraphFacade_MetricDistanceAndDetachedIDs checks complete weighted routing.
// The five line-metric stations have a minimum closed walk of exactly eight units.
func TestTSPGraphFacade_MetricDistanceAndDetachedIDs(t *testing.T) {
	graph := testRoutingGraph(t, true)
	opts := tsp.DefaultOptions()
	opts.EnableLocalSearch = false
	result, err := tsp.SolveGraph(graph, opts)
	if err != nil || result == nil {
		t.Fatalf("SolveGraph complete: %+v, %v", result, err)
	}
	expectedIDs := []string{"dock-0", "dock-1", "dock-2", "dock-3", "dock-4"}
	if !slices.Equal(result.IDs, expectedIDs) || result.Cost != 8 || result.MetricClosureApplied {
		t.Fatalf("wrong complete graph result: %+v", result)
	}
	ids, err := result.VertexTour()
	if err != nil || len(ids) != len(expectedIDs)+1 || ids[0] != "dock-0" || ids[len(ids)-1] != "dock-0" {
		t.Fatalf("graph-ID tour=%v err=%v", ids, err)
	}
	result.IDs[0] = "caller-mutated"
	if !slices.Equal(graph.Vertices(), expectedIDs) {
		t.Fatalf("result alias corrupted core graph IDs: %s", fmt.Sprint(graph.Vertices()))
	}
}

// TestTSPGraphFacade_ClosureOverSparseChain checks explicitly requested APSP.
// Missing direct edges are traversable after shortest-path closure but not before.
func TestTSPGraphFacade_ClosureOverSparseChain(t *testing.T) {
	graph := testRoutingGraph(t, false)
	opts := tsp.DefaultOptions()
	opts.RunMetricClosure = true
	opts.EnableLocalSearch = false
	result, err := tsp.SolveGraph(graph, opts)
	if err != nil || result == nil {
		t.Fatalf("SolveGraph closure: %+v, %v", result, err)
	}
	if !result.MetricClosureApplied || result.Cost != 8 || len(result.IDs) != 5 {
		t.Fatalf("unexpected closed graph tour: %+v", result)
	}
}

// TestTSPGraphFacade_DirectedAndMixedPolicies verifies core-to-matrix directionality.
// A fully bidirectional directed graph is legal for ATSP, but mixed edge mode is
// refused rather than silently flattening incompatible topological semantics.
func TestTSPGraphFacade_DirectedAndMixedPolicies(t *testing.T) {
	directed, err := core.NewGraph(core.WithDirected(true), core.WithWeighted())
	if err != nil {
		t.Fatalf("directed core.NewGraph: %v", err)
	}
	names := []string{"airport", "clinic", "station", "warehouse"}
	for _, name := range names {
		if err := directed.AddVertex(name); err != nil {
			t.Fatal(err)
		}
	}
	for i, from := range names {
		for j, to := range names {
			if i == j {
				continue
			}
			step := i - j
			if step < 0 {
				step = -step
			}
			if _, err := directed.AddEdge(from, to, float64(step)); err != nil {
				t.Fatalf("directed AddEdge %s->%s: %v", from, to, err)
			}
		}
	}
	opts := tsp.DefaultOptions()
	opts.Symmetric = false
	opts.Algo = tsp.ExactHeldKarp
	result, err := tsp.SolveGraph(directed, opts)
	if err != nil || result == nil || result.Cost != 6 || result.Algorithm != tsp.ExactHeldKarp {
		t.Fatalf("directed exact graph: %+v %v", result, err)
	}
	mixed, err := core.NewMixedGraph(core.WithWeighted())
	if err != nil {
		t.Fatalf("mixed core.NewMixedGraph: %v", err)
	}
	rejected, err := tsp.SolveGraph(mixed, opts)
	if rejected != nil || !errors.Is(err, tsp.ErrMixedGraphNotRepresentable) {
		t.Fatalf("mixed graph must be refused: %+v %v", rejected, err)
	}
}
