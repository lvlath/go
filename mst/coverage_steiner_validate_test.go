// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mst_test exercises pruning and publication as separate Steiner invariant boundaries.
// These tests supply detached candidate trees and compare their structure with independently known outcomes.
package mst_test

import (
	"context"
	"math"
	"slices"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/mst"
)

// TestSteinerPruning_CascadingNonterminalLeaves verifies degree-queue pruning on a real tree.
// Removing a leaf must enqueue its newly exposed nonterminal neighbor without removing terminal paths.
// Implementation:
//   - Stage 1: Supply a terminal backbone with a two-edge nonterminal tail.
//   - Stage 2: Prune both tail edges, including the previously removed incident-edge entry.
//   - Stage 3: Check surviving original records, order, cancellation, and invalid incidence.
//
// Behavior highlights:
//   - A zero-cost terminal edge is still a required connection.
//   - Both stored endpoint orientations are used by the cascade.
//
// Inputs:
//   - t: active test; explicitly ordered detached vertex and edge slices.
//
// Returns:
//   - None; the expected surviving tree is compared by original edge value.
//
// Errors:
//   - ErrInvalidSteinerResult for missing terminal or invalid incidence; context.Canceled on interruption.
//
// Determinism:
//   - Initial queue and incident-edge scan orders follow the supplied vertex and edge slices.
//
// Complexity:
//   - O(V+E) time and space per fixture, including the test's independent copies.
//
// Notes:
//   - Pruning assumes a selected connected tree; it is not an independent connectivity validator.
//
// AI-Hints:
//   - A one-edge tail does not exercise cascading degrees or an already removed incident entry.
func TestSteinerPruning_CascadingNonterminalLeaves(t *testing.T) {
	vertices := []string{"A", "B", "C", "D", "E"}
	terminals := []string{"A", "D"}
	// CE is listed before BC so the second leaf scan encounters the removed edge first.
	// E is stored as From, while C is stored as To on its remaining edge BC.
	edges := []core.Edge{
		{ID: "CE", From: "E", To: "C", Weight: 5},
		{ID: "BC", From: "B", To: "C", Weight: 4},
		{ID: "AB", From: "A", To: "B", Weight: 0},
		{ID: "BD", From: "B", To: "D", Weight: 3},
	}
	before := slices.Clone(edges)
	pruned, err := mst.PruneSteinerLeavesForTest(context.Background(), vertices, edges, terminals)
	mustNoError(t, err, "cascade pruning")
	if !slices.Equal(pruned, edges[2:]) || !slices.Equal(edges, before) {
		t.Fatalf("wrong surviving records or mutated candidates: pruned=%v input=%v", pruned, edges)
	}

	// A single terminal at the end of a branch must survive the complete cascade.
	pruned, err = mst.PruneSteinerLeavesForTest(context.Background(), []string{"A", "B"}, edges[2:3], []string{"A"})
	mustNoError(t, err, "single-terminal pruning")
	mustEqualInt(t, len(pruned), 0, "nonterminal leaf removed")

	// An unknown terminal and malformed edge endpoints are rejected at their own stage.
	for _, tc := range []struct {
		name      string
		edges     []core.Edge
		terminals []string
	}{
		{"unknown terminal", edges, []string{"A", "missing"}},
		{"unknown from", []core.Edge{{ID: "bad", From: "missing", To: "B", Weight: 1}}, terminals},
		{"unknown to", []core.Edge{{ID: "bad", From: "B", To: "missing", Weight: 1}}, terminals},
		{"self loop", []core.Edge{{ID: "bad", From: "B", To: "B", Weight: 1}}, terminals},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := mst.PruneSteinerLeavesForTest(context.Background(), vertices, tc.edges, tc.terminals)
			mustErrorIs(t, err, mst.ErrInvalidSteinerResult, "pruning incidence")
			if got != nil {
				t.Fatalf("invalid pruning produced %v", got)
			}
		})
	}

	// Cancel before preparation and again at the second removable leaf.
	for _, checkpoint := range []int64{1, 3} {
		ctx, cancel := newSteinerCheckpointContext(checkpoint)
		got, err := mst.PruneSteinerLeavesForTest(ctx, vertices, edges, terminals)
		cancel()
		mustErrorIs(t, err, context.Canceled, "pruning cancellation")
		if got != nil {
			t.Fatalf("canceled pruning produced %v", got)
		}
	}
}

// TestSteinerVerification_RejectsMalformedCandidateTrees protects final KMB provenance and structure.
// A candidate with |E|=|V|-1 still needs unique canonical IDs, valid incidence, original records and acyclicity.
// Implementation:
//   - Stage 1: Construct a valid three-vertex terminal path and independent original-edge catalog.
//   - Stage 2: Change one structural, numeric or identity invariant per case.
//   - Stage 3: Verify precise error identity, a cycle with correct cardinality, and nonterminal-leaf rejection.
//
// Behavior highlights:
//   - This is stricter than the public walk validator: final KMB output must have no nonterminal leaves.
//   - Foreign or altered original-edge records are classified separately from invalid tree shape.
//
// Inputs:
//   - t: active test; detached SteinerResult values and a stable catalog fixture.
//
// Returns:
//   - None; each invalid candidate must fail before cost publication.
//
// Errors:
//   - ErrInvalidSteinerResult for structural or numeric domain violations; ErrInvalidWitness for provenance.
//
// Determinism:
//   - Canonical vertex, terminal and edge order is explicit in the fixture.
//
// Complexity:
//   - O(V+E*(alpha(V)+log V)) per bounded candidate, plus detached copies.
//
// Notes:
//   - TotalWeight is not checked here because publication computes it after structural verification.
//
// AI-Hints:
//   - Keep a foreign-edge test structurally valid, otherwise cardinality or incidence will hide the provenance check.
func TestSteinerVerification_RejectsMalformedCandidateTrees(t *testing.T) {
	originals := []core.Edge{
		{ID: "AX", From: "A", To: "X", Weight: 1},
		{ID: "BX", From: "X", To: "B", Weight: 2},
	}
	base := &mst.SteinerResult{
		Terminals: []string{"A", "B"}, Vertices: []string{"A", "B", "X"}, Edges: originals,
	}
	prepared := mst.NewSteinerPreparedForTest(nil, base.Terminals, originals, nil)
	mustNoError(t, mst.VerifySteinerNetworkForTest(context.Background(), prepared, base), "valid original tree")
	mustErrorIs(t, mst.VerifySteinerNetworkForTest(context.Background(), prepared, nil), mst.ErrInvalidSteinerResult, "nil candidate")

	for _, tc := range []struct {
		name   string
		change func(*mst.SteinerResult)
		want   error
	}{
		{"no terminals", func(r *mst.SteinerResult) { r.Terminals = nil }, mst.ErrInvalidSteinerResult},
		{"no vertices", func(r *mst.SteinerResult) { r.Vertices = nil }, mst.ErrInvalidSteinerResult},
		{"edge count", func(r *mst.SteinerResult) { r.Edges = r.Edges[:1] }, mst.ErrInvalidSteinerResult},
		{"empty vertex", func(r *mst.SteinerResult) { r.Vertices[0] = "" }, mst.ErrInvalidSteinerResult},
		{"duplicate vertex", func(r *mst.SteinerResult) { r.Vertices[1] = "A" }, mst.ErrInvalidSteinerResult},
		{"unordered vertices", func(r *mst.SteinerResult) { r.Vertices[0], r.Vertices[1] = r.Vertices[1], r.Vertices[0] }, mst.ErrInvalidSteinerResult},
		{"empty terminal", func(r *mst.SteinerResult) { r.Terminals[0] = "" }, mst.ErrInvalidSteinerResult},
		{"duplicate terminal", func(r *mst.SteinerResult) { r.Terminals[1] = "A" }, mst.ErrInvalidSteinerResult},
		{"unordered terminals", func(r *mst.SteinerResult) { r.Terminals[0], r.Terminals[1] = r.Terminals[1], r.Terminals[0] }, mst.ErrInvalidSteinerResult},
		{"foreign terminal", func(r *mst.SteinerResult) { r.Terminals[1] = "Z" }, mst.ErrInvalidSteinerResult},
		{"empty edge ID", func(r *mst.SteinerResult) { r.Edges[0].ID = "" }, mst.ErrInvalidSteinerResult},
		{"duplicate edge ID", func(r *mst.SteinerResult) { r.Edges[1].ID = "AX" }, mst.ErrInvalidSteinerResult},
		{"unordered edges", func(r *mst.SteinerResult) { r.Edges[0], r.Edges[1] = r.Edges[1], r.Edges[0] }, mst.ErrInvalidSteinerResult},
		{"loop", func(r *mst.SteinerResult) { r.Edges[0].To = "A" }, mst.ErrInvalidSteinerResult},
		{"directed", func(r *mst.SteinerResult) { r.Edges[0].Directed = true }, mst.ErrInvalidSteinerResult},
		{"unknown from", func(r *mst.SteinerResult) { r.Edges[0].From = "Z" }, mst.ErrInvalidSteinerResult},
		{"unknown to", func(r *mst.SteinerResult) { r.Edges[0].To = "Z" }, mst.ErrInvalidSteinerResult},
		{"negative cost", func(r *mst.SteinerResult) { r.Edges[0].Weight = -1 }, mst.ErrInvalidSteinerResult},
		{"NaN cost", func(r *mst.SteinerResult) { r.Edges[0].Weight = math.NaN() }, mst.ErrInvalidSteinerResult},
		{"positive infinite cost", func(r *mst.SteinerResult) { r.Edges[0].Weight = math.Inf(1) }, mst.ErrInvalidSteinerResult},
		{"negative infinite cost", func(r *mst.SteinerResult) { r.Edges[0].Weight = math.Inf(-1) }, mst.ErrInvalidSteinerResult},
		{"foreign edge", func(r *mst.SteinerResult) { r.Edges[1].ID = "ZZ" }, mst.ErrInvalidWitness},
		{"changed original cost", func(r *mst.SteinerResult) { r.Edges[0].Weight = 2 }, mst.ErrInvalidWitness},
		{"changed original orientation", func(r *mst.SteinerResult) { r.Edges[0].From, r.Edges[0].To = r.Edges[0].To, r.Edges[0].From }, mst.ErrInvalidWitness},
		{"nonterminal leaf", func(r *mst.SteinerResult) { r.Terminals = []string{"A", "X"} }, mst.ErrInvalidSteinerResult},
	} {
		t.Run(tc.name, func(t *testing.T) {
			candidate := base.Clone()
			tc.change(candidate)
			mustErrorIs(t, mst.VerifySteinerNetworkForTest(context.Background(), prepared, candidate), tc.want, "candidate verification")
		})
	}

	// Two distinct parallel original IDs form a cycle while an isolated vertex preserves V-1 cardinality.
	cycle := &mst.SteinerResult{
		Terminals: []string{"A", "B"}, Vertices: []string{"A", "B", "C"},
		Edges: []core.Edge{{ID: "p1", From: "A", To: "B", Weight: 1}, {ID: "p2", From: "B", To: "A", Weight: 2}},
	}
	cyclePrepared := mst.NewSteinerPreparedForTest(nil, cycle.Terminals, cycle.Edges, nil)
	mustErrorIs(t, mst.VerifySteinerNetworkForTest(context.Background(), cyclePrepared, cycle), mst.ErrInvalidSteinerResult, "parallel cycle plus isolate")
}

// TestSteinerPublication_RejectsInvalidNetworkBeforeCost verifies the final publication boundary.
// Structural verification must precede aggregation, and canonical order must not change the input slice.
// Implementation:
//   - Stage 1: Build original records for a valid terminal path and a separate invalid candidate.
//   - Stage 2: Publish a reordered valid path and reject foreign or missing original connections.
//   - Stage 3: Verify cancellation and distinguish a real final aggregate overflow.
//
// Behavior highlights:
//   - On any failure the public SteinerResult pointer is nil.
//   - Original Edge.ID order defines the numeric fold, not candidate order.
//
// Inputs:
//   - t: active test; read-only detached candidate records and terminal set.
//
// Returns:
//   - None; successful canonical publication and failed-publication contracts are asserted.
//
// Errors:
//   - ErrInvalidWitness, ErrInvalidSteinerResult, ErrWeightOverflow, and context.Canceled.
//
// Determinism:
//   - Fixed edge ID order and exactly representable small positive-control weights.
//
// Complexity:
//   - O(E log E+V log V+E*alpha(V)) time and O(V+E) space per publication.
//
// Notes:
//   - This boundary is not a replacement for end-to-end SteinerTree tests.
//
// AI-Hints:
//   - Do not classify an overflowing but structurally valid final tree as disconnected.
func TestSteinerPublication_RejectsInvalidNetworkBeforeCost(t *testing.T) {
	first := core.Edge{ID: "AX", From: "A", To: "X", Weight: 1}
	second := core.Edge{ID: "BX", From: "X", To: "B", Weight: 2}
	originals := []core.Edge{first, second}
	prepared := mst.NewSteinerPreparedForTest(nil, []string{"A", "B"}, originals, nil)
	selected := []core.Edge{second, first}
	result, err := mst.PublishSteinerNetworkForTest(context.Background(), prepared, selected)
	mustNoError(t, err, "valid publication")
	if result == nil || result.TotalWeight != 3 || !slices.Equal(result.Edges, originals) || !slices.Equal(result.Vertices, []string{"A", "B", "X"}) || selected[0] != second {
		t.Fatalf("canonical publication or input ownership failed: result=%+v input=%v", result, selected)
	}

	// Provenance failure takes priority over a scalar aggregate on the candidate.
	foreign := second
	foreign.Weight = math.MaxFloat64
	result, err = mst.PublishSteinerNetworkForTest(context.Background(), prepared, []core.Edge{first, foreign})
	mustSteinerBoundaryError(t, result, err, mst.ErrInvalidWitness, "stale original edge")
	result, err = mst.PublishSteinerNetworkForTest(context.Background(), prepared, []core.Edge{first})
	mustSteinerBoundaryError(t, result, err, mst.ErrInvalidSteinerResult, "missing connection")

	// Both records are original and finite, but the selected network's aggregate overflows.
	first.Weight, second.Weight = math.MaxFloat64, math.MaxFloat64
	largeEdges := []core.Edge{first, second}
	large := mst.NewSteinerPreparedForTest(nil, []string{"A", "B"}, largeEdges, nil)
	result, err = mst.PublishSteinerNetworkForTest(context.Background(), large, largeEdges)
	mustSteinerBoundaryError(t, result, err, mst.ErrWeightOverflow, "final aggregate overflow")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	result, err = mst.PublishSteinerNetworkForTest(ctx, prepared, selected)
	mustSteinerBoundaryError(t, result, err, context.Canceled, "publication cancellation")
}
