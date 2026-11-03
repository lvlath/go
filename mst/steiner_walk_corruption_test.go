// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mst_test validates adversarial caller-owned SteinerResult values.
// The cases assert structural admission and error identity without accessing mst internals.
package mst_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/mst"
)

// TestSteinerWalk_RejectsMalformedCallerOwnedTrees checks independent corruptions of a valid tree.
// Every error must prevent publication, including failures in nonterminal or unrelated fields.
// Implementation:
//   - Stage 1: Build one valid original-edge tree with stable, distinct identities.
//   - Stage 2: Mutate exactly the field defining each malformed counterexample.
//   - Stage 3: Require a nil walk and errors.Is identity for the advertised failure class.
//
// Behavior highlights:
//   - A caller may corrupt structural fields even when TotalWeight remains unchanged.
//
// Inputs:
//   - Table of test-only mutations; no mutation to a core.Graph is needed.
//
// Returns:
//   - Test failure on any accepted invalid result or discarded error cause.
//
// Errors:
//   - ErrInvalidSteinerResult for structural/stale data, joined MST/core numeric/ID causes.
//
// Determinism:
//   - The same canonical copy is validated for every independent counterexample.
//
// Complexity:
//   - Time O(C) for a fixed set of constant-size counterexamples; Space O(1) per case.
//
// Notes:
//   - Tree validation cannot attest KMB provenance or existence in an omitted source graph.
//
// AI-Hints:
//   - Do not repair corrupted public TotalWeight or silently remove an invalid edge.
func TestSteinerWalk_RejectsMalformedCallerOwnedTrees(t *testing.T) {
	base := walkFixtureTree(
		[]string{"A", "B"},
		[]string{"A", "B", "X"},
		[]core.Edge{
			{ID: "AX", From: "A", To: "X", Weight: 1},
			{ID: "BX", From: "B", To: "X", Weight: 2},
		},
	)
	cases := []struct {
		name   string
		mutate func(*mst.SteinerResult)
		cause  error
	}{
		{"no-terminals", func(r *mst.SteinerResult) { r.Terminals = nil }, nil},
		{"empty-terminal-id", func(r *mst.SteinerResult) { r.Terminals[0] = "" }, core.ErrEmptyVertexID},
		{"duplicate-terminal", func(r *mst.SteinerResult) { r.Terminals[1] = "A" }, nil},
		{"foreign-terminal", func(r *mst.SteinerResult) { r.Terminals[1] = "foreign" }, core.ErrVertexNotFound},
		{"no-vertices", func(r *mst.SteinerResult) { r.Vertices = nil }, nil},
		{"empty-vertex-id", func(r *mst.SteinerResult) { r.Vertices[0] = "" }, core.ErrEmptyVertexID},
		{"duplicate-vertex", func(r *mst.SteinerResult) { r.Vertices[1] = "A" }, nil},
		{"empty-edge-id", func(r *mst.SteinerResult) { r.Edges[0].ID = "" }, core.ErrEmptyEdgeID},
		{"duplicate-edge-id", func(r *mst.SteinerResult) { r.Edges[1].ID = "AX" }, nil},
		{"empty-edge-endpoint", func(r *mst.SteinerResult) { r.Edges[0].From = "" }, core.ErrEmptyVertexID},
		{"foreign-edge-endpoint", func(r *mst.SteinerResult) { r.Edges[0].From = "foreign" }, core.ErrVertexNotFound},
		{"self-loop", func(r *mst.SteinerResult) { r.Edges[0].From = "X" }, nil},
		{"directed", func(r *mst.SteinerResult) { r.Edges[0].Directed = true }, mst.ErrDirectedEdge},
		{"negative", func(r *mst.SteinerResult) { r.Edges[0].Weight = -1 }, mst.ErrNegativeWeight},
		{"nan-edge", func(r *mst.SteinerResult) { r.Edges[0].Weight = math.NaN() }, mst.ErrNaNInfWeight},
		{"positive-infinite-edge", func(r *mst.SteinerResult) { r.Edges[0].Weight = math.Inf(1) }, mst.ErrNaNInfWeight},
		{"negative-infinite-edge", func(r *mst.SteinerResult) { r.Edges[0].Weight = math.Inf(-1) }, mst.ErrNaNInfWeight},
		{"wrong-total", func(r *mst.SteinerResult) { r.TotalWeight = 2 }, nil},
		{"negative-total", func(r *mst.SteinerResult) { r.TotalWeight = -1 }, nil},
		{"nan-total", func(r *mst.SteinerResult) { r.TotalWeight = math.NaN() }, nil},
		{"infinite-total", func(r *mst.SteinerResult) { r.TotalWeight = math.Inf(1) }, nil},
		{"disconnected-cardinality", func(r *mst.SteinerResult) {
			r.Vertices = append(r.Vertices, "isolated")
		}, nil},
		{"excess-edge-count", func(r *mst.SteinerResult) {
			r.Edges = append(r.Edges, core.Edge{ID: "extra", From: "A", To: "B", Weight: 1})
		}, nil},
		{"cycle-plus-isolate", func(r *mst.SteinerResult) {
			r.Vertices = append(r.Vertices, "isolated")
			r.Edges = append(r.Edges, core.Edge{ID: "AB", From: "A", To: "B", Weight: 5})
		}, nil},
		{"parallel-cycle", func(r *mst.SteinerResult) {
			r.Vertices = []string{"A", "B", "C"}
			r.Terminals = []string{"A", "B"}
			r.Edges = []core.Edge{
				{ID: "p1", From: "A", To: "B", Weight: 1},
				{ID: "p2", From: "B", To: "A", Weight: 1},
			}
		}, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			damaged := base.Clone()
			tc.mutate(damaged)
			walk, err := mst.SteinerWalk(damaged, "A", "B")
			if walk != nil || !errors.Is(err, mst.ErrInvalidSteinerResult) {
				t.Fatalf("malformed result admitted: walk=%+v error=%v", walkSummary(walk), err)
			}
			if tc.cause != nil && !errors.Is(err, tc.cause) {
				t.Fatalf("lost specific errors.Is cause %v: %v", tc.cause, err)
			}
		})
	}
}

// TestSteinerWalk_RejectsUnrepresentableUniqueNetworkTotal distinguishes tree and route overflow.
// An invalid unique-edge aggregate fails admission before a walk can visit any tree edge.
// Implementation:
//   - Stage 1: Create two individually finite original edges that cannot be summed in float64.
//   - Stage 2: Keep the public total finite so the canonical recomputation is exercised.
//   - Stage 3: Require ErrWeightOverflow with no published WalkResult.
//
// Behavior highlights:
//   - Network-cost overflow is a numeric error, not malformed connectivity.
//
// Inputs:
//   - A two-edge three-vertex tree with finite positive weights.
//
// Returns:
//   - Test failure if an overflowed network is accepted or misclassified.
//
// Errors:
//   - ErrWeightOverflow, never ErrDisconnected or a successful infinite Distance.
//
// Determinism:
//   - Fixed Edge.ID order and checked left fold.
//
// Complexity:
//   - Time O(1), Space O(1) for this small example.
//
// Notes:
//   - This is distinct from a valid finite network whose repeated route overflows.
//
// AI-Hints:
//   - Do not classify canonical-network overflow as a stale scalar mismatch.
func TestSteinerWalk_RejectsUnrepresentableUniqueNetworkTotal(t *testing.T) {
	const heavy = 0x1p1023
	tree := &mst.SteinerResult{
		Terminals: []string{"A", "B"},
		Vertices:  []string{"A", "X", "B"},
		Edges: []core.Edge{
			{ID: "AX", From: "A", To: "X", Weight: heavy},
			{ID: "BX", From: "B", To: "X", Weight: heavy},
		},
		TotalWeight: heavy,
	}
	walk, err := mst.SteinerWalk(tree, "A", "B")
	if walk != nil || !errors.Is(err, mst.ErrWeightOverflow) || errors.Is(err, mst.ErrInvalidSteinerResult) {
		t.Fatalf("network arithmetic must fail before walking: walk=%+v error=%v", walk, err)
	}
}

// TestSteinerWalk_RejectsStaleNumericFoldButAcceptsValidOrderDifference checks float64 order.
// The canonical unique-edge sum can differ from the real path's accumulation order.
// Implementation:
//   - Stage 1: Build a path whose heavy edge comes first in traversal but last in edge-ID order.
//   - Stage 2: Require the walk to accept the canonical ID-ordered network total.
//   - Stage 3: Reject a corrupted total incorrectly replaced with the path-ordered value.
//
// Behavior highlights:
//   - Both sums use checked ordinary binary64 left folds, each in its declared order.
//
// Inputs:
//   - Three finite original edges with a large and two small weights.
//
// Returns:
//   - Test failure on false witness/cost coupling or stale public aggregate acceptance.
//
// Errors:
//   - ErrInvalidSteinerResult only after corrupting the caller's public TotalWeight.
//
// Determinism:
//   - Exact Edge.ID-ordered network fold versus exact traversal-ordered passage fold.
//
// Complexity:
//   - Time O(1), Space O(1) for the fixed path fixture.
//
// Notes:
//   - This is a numeric counterexample, not an approximation theorem proof.
//
// AI-Hints:
//   - Do not demand equality between a path-ordered sum and a separately ordered network sum.
func TestSteinerWalk_RejectsStaleNumericFoldButAcceptsValidOrderDifference(t *testing.T) {
	tree := walkFixtureTree([]string{"A", "B"}, []string{"A", "X", "Y", "B"}, []core.Edge{
		{ID: "z-heavy", From: "A", To: "X", Weight: 1e16},
		{ID: "a-unit", From: "X", To: "Y", Weight: 1},
		{ID: "b-unit", From: "Y", To: "B", Weight: 1},
	})
	const canonicalTotal = 10000000000000002
	if tree.TotalWeight != canonicalTotal {
		t.Fatalf("incorrect numeric counterexample: %g", tree.TotalWeight)
	}
	walk, err := mst.SteinerWalk(tree, "A", "B")
	if err != nil || walk.NetworkWeight != canonicalTotal || walk.Distance != 1e16 {
		t.Fatalf("valid differing folds: walk=%+v error=%v", walk, err)
	}
	assertWalkSteps(t, walk, tree.Edges)
	stale := tree.Clone()
	stale.TotalWeight = 1e16
	if got, err := mst.SteinerWalk(stale, "A", "B"); got != nil || !errors.Is(err, mst.ErrInvalidSteinerResult) {
		t.Fatalf("stale path-ordered total must be rejected: walk=%+v error=%v", got, err)
	}
}
