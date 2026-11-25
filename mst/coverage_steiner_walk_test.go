// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mst_test verifies distinct walk-overflow and step-incidence boundaries.
// Finite tree weights can still yield an unrepresentable repeated-passage route.
package mst_test

import (
	"errors"
	"slices"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/mst"
)

// TestSteinerWalk_OverflowAtDistinctPassageStages verifies error propagation from actual traversal steps.
// A finite canonical network can overflow on branch entry, a deeper descent, or a spine passage.
// Implementation:
//   - Stage 1: Construct exactly representable weighted trees with known traversal prefixes.
//   - Stage 2: Request closed and fixed-end walks whose next actual passage overflows.
//   - Stage 3: Require ErrWeightOverflow, nil result, and unchanged input data.
//
// Behavior highlights:
//   - Cost is accumulated by actual original-edge traversal; no 2*W intermediate is required.
//   - All fixtures have finite unique-edge cost and valid topology.
//
// Inputs:
//   - t: active test context; caller-owned tree fixtures with independent backing arrays.
//
// Returns:
//   - None; each overflow path is rejected without exposing a partial route.
//
// Errors:
//   - ErrWeightOverflow; neither connectivity nor structural errors are expected.
//
// Determinism:
//   - Lexical neighbor A is visited before B, and fixed-end B is reserved for the spine.
//
// Complexity:
//   - O(V log V+E log E) preparation and O(V+E) traversal on bounded fixtures.
//
// Notes:
//   - Weights are exact powers-of-two fractions, so the decisive prefixes do not depend on decimal rounding.
//
// AI-Hints:
//   - A single closed overflow test may exercise only the branch-return error; vary the traversal shape.
func TestSteinerWalk_OverflowAtDistinctPassageStages(t *testing.T) {
	const w = 0x1p1022
	star := walkFixtureTree([]string{"A", "B"}, []string{"A", "B", "X"}, []core.Edge{
		{ID: "AX", From: "A", To: "X", Weight: 1.5 * w},
		{ID: "BX", From: "B", To: "X", Weight: w},
	})
	deep := walkFixtureTree([]string{"A", "C"}, []string{"A", "B", "C", "X"}, []core.Edge{
		{ID: "AX", From: "A", To: "X", Weight: 1.5 * w},
		{ID: "BC", From: "B", To: "C", Weight: 0.5 * w},
		{ID: "BX", From: "B", To: "X", Weight: 0.5 * w},
	})
	for _, tc := range []struct {
		name string
		tree *mst.SteinerResult
		end  string
	}{
		// Returning from A costs 3w; entering B costs another w and overflows.
		{"branch entry", star, "X"},
		// After A returns and B is reached, the deeper B-C step raises 3.5w to 4w.
		{"deeper descent", deep, "X"},
		// After the off-spine A branch returns, the final X-B spine step overflows.
		{"spine passage", star, "B"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			before := tc.tree.Clone()
			walk, err := mst.SteinerWalk(tc.tree, "X", tc.end)
			mustErrorIs(t, err, mst.ErrWeightOverflow, "actual passage overflow")
			if walk != nil || errors.Is(err, mst.ErrInvalidSteinerResult) || errors.Is(err, mst.ErrDisconnected) {
				t.Fatalf("walk overflow misclassified: result=%+v error=%v", walk, err)
			}
			if !slices.Equal(tc.tree.Edges, before.Edges) || !slices.Equal(tc.tree.Vertices, before.Vertices) ||
				!slices.Equal(tc.tree.Terminals, before.Terminals) || tc.tree.TotalWeight != before.TotalWeight {
				t.Fatal("failed walk mutated the caller's network")
			}
		})
	}
}

// TestSteinerWalk_PreparedPassageRejectsInvalidIncidence defends the final step-publication guard.
// Invalid indexes, wrong current vertices, and wrong original edges must not change an existing route prefix.
// Implementation:
//   - Stage 1: Prepare one real, valid original-edge tree through the actual walk validator.
//   - Stage 2: Request inconsistent step indexes or incidence on isolated WalkResult copies.
//   - Stage 3: Check error identity and unchanged prefix, followed by a legal reverse-oriented step.
//
// Behavior highlights:
//   - A step uses original Edge.ID, not just a valid destination vertex.
//   - A malformed transition fails before either route slice or Distance changes.
//
// Inputs:
//   - t: active test context; immutable prepared tree and independently owned route prefixes.
//
// Returns:
//   - None; the step validator must reject invalid operations atomically.
//
// Errors:
//   - ErrInvalidSteinerResult for invalid indexes or incidence; ErrNilResult for nil preparation.
//
// Determinism:
//   - Canonical vertices A, B, X and original edges AX, BX fix the indexes independently of input order.
//
// Complexity:
//   - O(V log V+E log E) preparation, O(1) per preallocated route-step check.
//
// Notes:
//   - Direct step tests exercise defensive internal invariants, not an exported production route API.
//
// AI-Hints:
//   - Never manufacture invalid adjacency slices or change valid graph incidence to reach these guards.
func TestSteinerWalk_PreparedPassageRejectsInvalidIncidence(t *testing.T) {
	if prepared, err := mst.PrepareSteinerWalkTreeForTest(nil); prepared != nil || !errors.Is(err, mst.ErrNilResult) {
		t.Fatalf("nil preparation: state=%v error=%v", prepared, err)
	}
	network := walkFixtureTree([]string{"A", "B"}, []string{"A", "B", "X"}, []core.Edge{
		{ID: "AX", From: "X", To: "A", Weight: 1},
		{ID: "BX", From: "B", To: "X", Weight: 2},
	})
	prepared, err := mst.PrepareSteinerWalkTreeForTest(network)
	mustNoError(t, err, "prepare valid walk tree")
	const (
		indexA  = 0
		indexB  = 1
		indexX  = 2
		indexAX = 0
		indexBX = 1
	)
	for _, tc := range []struct {
		name           string
		from, to, edge int
		vertices       []string
	}{
		{"negative source", -1, indexX, indexAX, []string{"A"}},
		{"out-of-range target", indexA, 3, indexAX, []string{"A"}},
		{"out-of-range edge", indexA, indexX, 2, []string{"A"}},
		{"empty prefix", indexA, indexX, indexAX, nil},
		{"wrong current vertex", indexA, indexX, indexAX, []string{"B"}},
		{"wrong original edge", indexA, indexX, indexBX, []string{"A"}},
		{"wrong endpoint", indexA, indexB, indexAX, []string{"A"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			prefix := &mst.WalkResult{Vertices: slices.Clone(tc.vertices), Distance: 7}
			before := prefix.Clone()
			err := mst.AppendSteinerWalkPassageForTest(prepared, prefix, tc.from, tc.to, tc.edge)
			mustErrorIs(t, err, mst.ErrInvalidSteinerResult, "invalid passage")
			if !slices.Equal(prefix.Vertices, before.Vertices) || !slices.Equal(prefix.EdgeIDs, before.EdgeIDs) || prefix.Distance != before.Distance {
				t.Fatalf("invalid passage changed prefix: before=%+v after=%+v", before, prefix)
			}
		})
	}
	prefix := &mst.WalkResult{Vertices: []string{"A"}}
	mustNoError(t, mst.AppendSteinerWalkPassageForTest(prepared, prefix, indexA, indexX, indexAX), "reverse original passage")
	if !slices.Equal(prefix.Vertices, []string{"A", "X"}) || !slices.Equal(prefix.EdgeIDs, []string{"AX"}) || prefix.Distance != 1 {
		t.Fatalf("legal reverse passage failed: %+v", prefix)
	}

	// Test the right endpoint guard separately from the already tested left endpoint guard.
	malformed := network.Clone()
	malformed.Edges[0].To = "missing"
	walk, err := mst.SteinerWalk(malformed, "A", "B")
	mustErrorIs(t, err, mst.ErrInvalidSteinerResult, "unknown right endpoint")
	mustErrorIs(t, err, core.ErrVertexNotFound, "unknown right endpoint cause")
	if walk != nil {
		t.Fatalf("invalid endpoint published %+v", walk)
	}
}
