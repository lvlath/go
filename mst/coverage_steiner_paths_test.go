// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mst_test validates Steiner's original-edge witness and closure boundaries.
// Independent fixture records exercise defensive failures without changing Dijkstra or graph state.
package mst_test

import (
	"context"
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/dijkstra"
	"github.com/lvlath/go/mst"
)

// TestSteinerWitness_VerifiesEveryOriginalPathStep checks the complete expansion boundary.
// A path that matches its final cost can still contain a wrong intermediate distance or original edge.
// Implementation:
//   - Stage 1: Construct an explicit two-edge SSSP witness with reverse stored orientation.
//   - Stage 2: Vary one structural or arithmetic invariant at a time on isolated fixture data.
//   - Stage 3: Check exact sentinel identities and original union records on successful expansion.
//
// Behavior highlights:
//   - Original IDs, incidence, directedness and every rounded prefix are independent checks.
//   - Invalid Dijkstra predecessor results preserve the originating Dijkstra error.
//
// Inputs:
//   - t: active test; detached edge inventories and public dijkstra.Result values.
//
// Returns:
//   - None; malformed witnesses are rejected and the positive control preserves exact edges.
//
// Errors:
//   - ErrInvalidWitness, ErrWeightOverflow, context.Canceled, and Dijkstra path errors.
//
// Determinism:
//   - Fixed vertex path A-X-B and explicit canonical endpoint keys.
//
// Complexity:
//   - O(P) per expansion for P path steps; P=2 in these fixtures.
//
// Notes:
//   - The union is private scratch state and may contain verified prefixes on failure.
//   - No fabricated error-returning Dijkstra implementation is used.
//
// AI-Hints:
//   - Do not require a private partial union to be empty; public publication, not scratch state,
//     is atomic. Exercise stale original records without changing a live core.Edge pointer.
func TestSteinerWitness_VerifiesEveryOriginalPathStep(t *testing.T) {
	const expected = 3.0
	first := core.Edge{ID: "AX", From: "X", To: "A", Weight: 1}
	second := core.Edge{ID: "BX", From: "B", To: "X", Weight: 2}
	originals := []core.Edge{first, second}
	pairs := map[[2]string]core.Edge{{"A", "X"}: first, {"B", "X"}: second}
	prepared := mst.NewSteinerPreparedForTest(nil, []string{"A", "B"}, originals, pairs)
	valid := &dijkstra.Result{
		SourceID: "A", Distances: map[string]float64{"A": 0, "X": 1, "B": expected},
		Prev: map[string]string{"X": "A", "B": "X"},
	}

	// The positive control verifies both stored edge directions and deduplicated identities.
	union := make(map[string]core.Edge)
	mustNoError(t, mst.ExpandSteinerVertexPathForTest(context.Background(), prepared, valid, "A", "B", expected, union), "valid expansion")
	mustEqualInt(t, len(union), 2, "original path edge count")
	if union[first.ID] != first || union[second.ID] != second {
		t.Fatalf("path expansion lost original records: %v", union)
	}
	mustNoError(t, mst.ExpandSteinerVertexPathForTest(context.Background(), prepared, valid, "A", "B", expected, union), "repeat expansion")
	mustEqualInt(t, len(union), 2, "ID-based deduplication")

	// Corrupt only the public SSSP result, not a real graph or shared production state.
	cases := []struct {
		name   string
		change func(*dijkstra.Result)
		want   error
	}{
		{"wrong source", func(r *dijkstra.Result) { r.SourceID = "B" }, mst.ErrInvalidWitness},
		{"no tracking", func(r *dijkstra.Result) { r.Prev = nil }, dijkstra.ErrPathTrackingDisabled},
		{"unknown target", func(r *dijkstra.Result) { delete(r.Distances, "B") }, dijkstra.ErrTargetNotFound},
		{"unreachable target", func(r *dijkstra.Result) { r.Distances["B"] = math.Inf(1) }, dijkstra.ErrNoPath},
		{"broken predecessor", func(r *dijkstra.Result) { delete(r.Prev, "X") }, dijkstra.ErrNoPath},
		{"cyclic predecessor", func(r *dijkstra.Result) { r.Prev["X"] = "B" }, dijkstra.ErrNoPath},
		{"missing intermediate distance", func(r *dijkstra.Result) { delete(r.Distances, "X") }, dijkstra.ErrNoPath},
		{"wrong intermediate prefix", func(r *dijkstra.Result) { r.Distances["X"] = 2 }, mst.ErrInvalidWitness},
		{"wrong final prefix", func(r *dijkstra.Result) { r.Distances["B"] = 4 }, mst.ErrInvalidWitness},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result := valid.Clone()
			tc.change(result)
			err := mst.ExpandSteinerVertexPathForTest(context.Background(), prepared, result, "A", "B", expected, make(map[string]core.Edge))
			mustErrorIs(t, err, tc.want, "invalid SSSP witness")
		})
	}
	mustErrorIs(t, mst.ExpandSteinerVertexPathForTest(context.Background(), prepared, nil, "A", "B", expected, make(map[string]core.Edge)), mst.ErrInvalidWitness, "nil SSSP witness")
	mustErrorIs(t, mst.ExpandSteinerVertexPathForTest(context.Background(), prepared, valid, "A", "A", 0, make(map[string]core.Edge)), mst.ErrInvalidWitness, "unexpected singleton path")
	mustErrorIs(t, mst.ExpandSteinerVertexPathForTest(context.Background(), prepared, valid, "A", "B", expected+1, make(map[string]core.Edge)), mst.ErrInvalidWitness, "closure cost mismatch")

	// Vary the adapter's pair record without modifying the authentic original catalog.
	for _, tc := range []struct {
		name string
		edge core.Edge
	}{
		{"empty identity", core.Edge{From: "A", To: "X", Weight: 1}},
		{"directed identity", core.Edge{ID: "AX", From: "A", To: "X", Weight: 1, Directed: true}},
		{"wrong incidence", core.Edge{ID: "AX", From: "A", To: "B", Weight: 1}},
		{"unknown original ID", core.Edge{ID: "foreign", From: "X", To: "A", Weight: 1}},
		{"changed original weight", core.Edge{ID: "AX", From: "X", To: "A", Weight: 2}},
		{"changed stored orientation", core.Edge{ID: "AX", From: "A", To: "X", Weight: 1}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			invalid := mst.NewSteinerPreparedForTest(nil, []string{"A", "B"}, originals, map[[2]string]core.Edge{{"A", "X"}: tc.edge, {"B", "X"}: second})
			err := mst.ExpandSteinerVertexPathForTest(context.Background(), invalid, valid, "A", "B", expected, make(map[string]core.Edge))
			mustErrorIs(t, err, mst.ErrInvalidWitness, "invalid original inventory")
		})
	}
	missingPair := mst.NewSteinerPreparedForTest(nil, []string{"A", "B"}, originals, nil)
	mustErrorIs(t, mst.ExpandSteinerVertexPathForTest(context.Background(), missingPair, valid, "A", "B", expected, make(map[string]core.Edge)), mst.ErrInvalidWitness, "missing original endpoint pair")

	// The final target label is finite but the supplied original path would overflow on its second step.
	largeFirst, largeSecond := first, second
	largeFirst.Weight, largeSecond.Weight = math.MaxFloat64, math.MaxFloat64
	large := mst.NewSteinerPreparedForTest(nil, []string{"A", "B"}, []core.Edge{largeFirst, largeSecond}, map[[2]string]core.Edge{{"A", "X"}: largeFirst, {"B", "X"}: largeSecond})
	largeResult := valid.Clone()
	largeResult.Distances["X"], largeResult.Distances["B"] = math.MaxFloat64, math.MaxFloat64
	err := mst.ExpandSteinerVertexPathForTest(context.Background(), large, largeResult, "A", "B", math.MaxFloat64, make(map[string]core.Edge))
	mustErrorIs(t, err, mst.ErrWeightOverflow, "unrepresentable original path prefix")
}

// TestSteinerClosure_OriginalDistanceAndRetainedState verifies the metric-closure boundary.
// Closure costs are per-terminal shortest distances and never replace original union edges.
// Implementation:
//   - Stage 1: Build a small graph with a known three-terminal metric closure.
//   - Stage 2: Compare all unordered distances and the two storage policies.
//   - Stage 3: Reject incomplete or inconsistent retained witness state and recomputation failures.
//
// Behavior highlights:
//   - Each selected connection must agree exactly with the corresponding SSSP target distance.
//   - A missing or disabled retained witness must not silently fall back to recomputation.
//
// Inputs:
//   - t: active test; real core.Graph and explicit detached closure links.
//
// Returns:
//   - None; success outputs and nil-on-failure slice contracts are checked.
//
// Errors:
//   - ErrInvalidWitness, dijkstra.ErrTargetNotFound, dijkstra.ErrPathTrackingDisabled,
//     dijkstra.ErrDistanceOverflow, and dijkstra.ErrSourceNotFound.
//
// Determinism:
//   - Fixed lexical terminals A, B, C; selected links A-B and A-C.
//
// Complexity:
//   - O(Tsssp(V,E)+k^2) per small closure, plus selected path reconstruction.
//
// Notes:
//   - Invalid state is introduced at the adapter boundary only, not into core's graph internals.
//
// AI-Hints:
//   - Check every pair's cost, not just the aggregate closure MST cost.
func TestSteinerClosure_OriginalDistanceAndRetainedState(t *testing.T) {
	graph := mustWeightedGraph(t)
	originals := []core.Edge{
		{ID: "AX", From: "A", To: "X", Weight: 1},
		{ID: "BX", From: "B", To: "X", Weight: 2},
		{ID: "CX", From: "C", To: "X", Weight: 4},
	}
	for _, edge := range originals {
		steinerFixtureEdge(t, graph, edge.ID, edge.From, edge.To, edge.Weight)
	}
	prepared := mst.NewSteinerPreparedForTest(graph, []string{"A", "B", "C"}, originals, map[[2]string]core.Edge{
		{"A", "X"}: originals[0], {"B", "X"}: originals[1], {"C", "X"}: originals[2],
	})
	config := mst.DefaultSteinerOptions()
	closure, retained, err := mst.BuildSteinerClosureForTest(prepared, config)
	mustNoError(t, err, "distance-only closure")
	if retained != nil || len(closure) != 3 {
		t.Fatalf("unexpected closure storage: edges=%v retained=%v", closure, retained)
	}
	wantPairs := []struct {
		from, to string
		cost     float64
	}{{"A", "B", 3}, {"A", "C", 5}, {"B", "C", 6}}
	for i, want := range wantPairs {
		if closure[i].From != want.from || closure[i].To != want.to || closure[i].Weight != want.cost || closure[i].Directed {
			t.Fatalf("closure[%d]=%+v, want %+v", i, closure[i], want)
		}
	}
	config.PathStorage = mst.SteinerRetainPaths
	tracked, retained, err := mst.BuildSteinerClosureForTest(prepared, config)
	mustNoError(t, err, "tracked closure")
	if !slices.Equal(closure, tracked) || len(retained) != 2 || retained["A"] == nil || retained["B"] == nil || retained["A"].Prev == nil || retained["B"].Prev == nil {
		t.Fatalf("retained closure disagrees with distance-only closure: %v %v", tracked, retained)
	}

	// The selected connections must expand to one deduplicated original hub, not synthetic IDs.
	selected := tracked[:2]
	vertices, edges, err := mst.ExpandSteinerClosureForTest(prepared, config, selected, retained)
	mustNoError(t, err, "retained expansion")
	if !slices.Equal(vertices, []string{"A", "B", "C", "X"}) || !slices.Equal(edges, originals) {
		t.Fatalf("unexpected original union: vertices=%v edges=%v", vertices, edges)
	}

	// A malformed retained result can be represented using the real public Dijkstra result type.
	for _, tc := range []struct {
		name   string
		change func(map[string]*dijkstra.Result)
		want   error
	}{
		{"absent source", func(r map[string]*dijkstra.Result) { delete(r, "A") }, mst.ErrInvalidWitness},
		{"absent target", func(r map[string]*dijkstra.Result) { delete(r["A"].Distances, "B") }, dijkstra.ErrTargetNotFound},
		{"changed distance", func(r map[string]*dijkstra.Result) { r["A"].Distances["B"]++ }, mst.ErrInvalidWitness},
		{"disabled tracking", func(r map[string]*dijkstra.Result) { r["A"].Prev = nil }, dijkstra.ErrPathTrackingDisabled},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Clone only the retained state required by these selected source groups.
			state := map[string]*dijkstra.Result{"A": retained["A"].Clone()}
			tc.change(state)
			gotVertices, gotEdges, err := mst.ExpandSteinerClosureForTest(prepared, config, selected, state)
			mustErrorIs(t, err, tc.want, "retained witness failure")
			if gotVertices != nil || gotEdges != nil {
				t.Fatalf("failed expansion published slices: %v %v", gotVertices, gotEdges)
			}
		})
	}

	// Recompute must preserve a real SSSP arithmetic error rather than switching modes.
	overflowGraph := mustWeightedGraph(t)
	steinerFixtureEdge(t, overflowGraph, "AB", "A", "B", math.MaxFloat64)
	steinerFixtureEdge(t, overflowGraph, "BC", "B", "C", math.MaxFloat64)
	overflow := mst.NewSteinerPreparedForTest(overflowGraph, []string{"A", "B", "C"}, nil, nil)
	config.PathStorage = mst.SteinerRecomputePaths
	vertices, edges, err = mst.ExpandSteinerClosureForTest(overflow, config, selected, nil)
	mustErrorIs(t, err, dijkstra.ErrDistanceOverflow, "recomputed distance overflow")
	if vertices != nil || edges != nil {
		t.Fatalf("failed recomputation published slices: %v %v", vertices, edges)
	}

	// A missing source or target should be rejected before any valid closure is published.
	for _, tc := range []struct {
		terminals []string
		want      error
	}{
		{[]string{"0-missing", "A", "B"}, dijkstra.ErrSourceNotFound},
		{[]string{"A", "B", "Z-missing"}, dijkstra.ErrTargetNotFound},
	} {
		invalid := mst.NewSteinerPreparedForTest(graph, tc.terminals, nil, nil)
		badClosure, state, err := mst.BuildSteinerClosureForTest(invalid, config)
		mustErrorIs(t, err, tc.want, "closure target/source domain")
		if badClosure != nil || state != nil {
			t.Fatalf("failed closure published state: %v %v", badClosure, state)
		}
	}
}

// TestSteinerKernel_DoesNotPublishFailedExpansion protects the dispatch boundary.
// Correct SSSP distances cannot justify publishing original edges absent from the prepared inventory.
// Implementation:
//   - Stage 1: Construct a valid path graph with known source and targets.
//   - Stage 2: Present a missing target or an intentionally incomplete original inventory.
//   - Stage 3: Exercise pair and full KMB dispatch and require nil results with preserved errors.
//
// Behavior highlights:
//   - Both storage modes must reject inconsistent adapter state; no hidden recomputation repair.
//
// Inputs:
//   - t: active test; the graph is valid and never mutated by the fixture.
//
// Returns:
//   - None; no partial public result may escape either branch.
//
// Errors:
//   - ErrInvalidWitness and dijkstra.ErrTargetNotFound.
//
// Determinism:
//   - Lexical terminal lists and a two-edge graph.
//
// Complexity:
//   - O(Tsssp(V,E)+k^2 log k) for the bounded fixture.
//
// Notes:
//   - This tests an internal invariant boundary, not admission of malformed public graphs.
//
// AI-Hints:
//   - Do not add production options that allow callers to inject invalid predecessor maps.
func TestSteinerKernel_DoesNotPublishFailedExpansion(t *testing.T) {
	graph := mustWeightedGraph(t)
	steinerFixtureEdge(t, graph, "AB", "A", "B", 1)
	steinerFixtureEdge(t, graph, "BC", "B", "C", 2)
	for _, storage := range []mst.SteinerPathStorage{mst.SteinerRecomputePaths, mst.SteinerRetainPaths} {
		config := mst.DefaultSteinerOptions()
		config.PathStorage = storage
		for _, terminals := range [][]string{{"A", "B"}, {"A", "B", "C"}} {
			t.Run(fmt.Sprintf("storage_%d_k%d", storage, len(terminals)), func(t *testing.T) {
				// SSSP succeeds, but expansion has no verified original records.
				invalid := mst.NewSteinerPreparedForTest(graph, terminals, nil, nil)
				result, err := mst.RunSteinerForTest(invalid, config)
				mustSteinerBoundaryError(t, result, err, mst.ErrInvalidWitness, "incomplete original inventory")
			})
		}
		invalid := mst.NewSteinerPreparedForTest(graph, []string{"A", "Z-missing"}, nil, nil)
		result, err := mst.RunSteinerForTest(invalid, config)
		mustSteinerBoundaryError(t, result, err, dijkstra.ErrTargetNotFound, "missing pair target")
	}
}
