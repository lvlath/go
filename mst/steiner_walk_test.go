// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mst_test exercises the public Steiner tree-covering walk contract.
// Every fixture is checked through exported APIs, including edge-step identities and ownership.
package mst_test

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"sort"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/mst"
)

// walkFixtureTree constructs a caller-owned tree with the declared edge order preserved.
// It sums only an independent ID-sorted copy to model the public canonical network cost.
// Implementation:
//   - Stage 1: Copy all input slices without modifying caller storage.
//   - Stage 2: Sort only a temporary edge copy by original Edge.ID.
//   - Stage 3: Sum the fixture weights in that order and publish a mutable test value.
//
// Behavior highlights:
//   - The production walk validator must independently validate every field of this value.
//
// Inputs:
//   - terminals, vertices: fixture ID slices; edges: original-edge value slice.
//
// Returns:
//   - *mst.SteinerResult: independent mutable fixture with unchanged edge ordering.
//
// Errors:
//   - None; special malformed cases are constructed by later caller mutations.
//
// Determinism:
//   - Canonical ID order is used for the fixture total only.
//
// Complexity:
//   - Time O(E log E+V+k), Space O(E+V+k).
//
// Notes:
//   - No production graph, DSU, path reconstruction or private validator is invoked.
//
// AI-Hints:
//   - Never sort the returned edges: permutation acceptance must be tested explicitly.
func walkFixtureTree(terminals, vertices []string, edges []core.Edge) *mst.SteinerResult {
	ordered := append([]core.Edge(nil), edges...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i].ID < ordered[j].ID })
	var total float64
	for _, edge := range ordered {
		total += edge.Weight
	}
	return &mst.SteinerResult{
		Terminals:   append([]string(nil), terminals...),
		Vertices:    append([]string(nil), vertices...),
		Edges:       append([]core.Edge(nil), edges...),
		TotalWeight: total,
	}
}

// assertWalkSteps independently checks every published step against original edge identities.
// It verifies repeated traversals by their real endpoint incidence, not by a total-cost formula.
// Implementation:
//   - Stage 1: Check endpoints and the route-witness length invariant.
//   - Stage 2: Resolve each passage ID in an independent edge map and verify both orientations.
//   - Stage 3: Accumulate actual step weights and return their passage counts.
//
// Behavior highlights:
//   - Original Edge.IDs can occur repeatedly but synthetic/foreign IDs cannot.
//
// Inputs:
//   - t: test handle; walk: public route; edges: independent original edge inventory.
//
// Returns:
//   - map[string]int: number of passages over each supplied original edge ID.
//
// Errors:
//   - Fail the test immediately on any invalid vertex transition or cost witness.
//
// Determinism:
//   - Steps are checked in exactly their returned order.
//
// Complexity:
//   - Time O(S+E), Space O(E) for S published passages.
//
// Notes:
//   - This checker deliberately does not call production traversal or validation helpers.
//
// AI-Hints:
//   - A matching Distance alone cannot prove correctness: confirm every edge/vertex transition.
func assertWalkSteps(t *testing.T, walk *mst.WalkResult, edges []core.Edge) map[string]int {
	t.Helper()
	if walk == nil || len(walk.Vertices) != len(walk.EdgeIDs)+1 || len(walk.Vertices) == 0 ||
		walk.Vertices[0] != walk.StartID || walk.Vertices[len(walk.Vertices)-1] != walk.EndID {
		t.Fatalf("invalid published walk shape: %+v", walk)
	}
	inventory := make(map[string]core.Edge, len(edges))
	for _, edge := range edges {
		inventory[edge.ID] = edge
	}
	counts := make(map[string]int, len(edges))
	var actual float64
	for stepIndex, edgeID := range walk.EdgeIDs {
		edge, exists := inventory[edgeID]
		from, to := walk.Vertices[stepIndex], walk.Vertices[stepIndex+1]
		if !exists || !((edge.From == from && edge.To == to) || (edge.To == from && edge.From == to)) {
			t.Fatalf("invalid passage %d: id=%q from=%q to=%q", stepIndex, edgeID, from, to)
		}
		actual += edge.Weight
		counts[edgeID]++
	}
	if actual != walk.Distance {
		t.Fatalf("distance=%g differs from original step sum=%g", walk.Distance, actual)
	}
	return counts
}

// TestSteinerWalk_HubClosedAndFixedEnd checks exact paths and costs from a real KMB tree.
// The hub is cheaper than its terminal closure; closed traversal returns and fixed-end does not.
// Implementation:
//   - Stage 1: Construct the original weighted hub and run both KMB memory policies.
//   - Stage 2: Check the exact closed and fixed-end edge/vertex passage sequences.
//   - Stage 3: Verify repeated original IDs, distances, and input result immutability.
//
// Behavior highlights:
//   - A->B leaves the C branch twice but does not return along the A-X-B spine.
//
// Inputs:
//   - A six-edge, three-terminal original graph; both implemented memory policies.
//
// Returns:
//   - Test failure on incorrect route structure, network/walk cost, or ownership.
//
// Errors:
//   - Any unexpected SteinerTree/SteinerWalk error fails the test immediately.
//
// Determinism:
//   - Canonical branch-neighbor order fixes one reproducible walk witness.
//
// Complexity:
//   - O(Tsssp(V,E)+E log E+S) per KMB/route invocation for the tiny fixture.
//
// Notes:
//   - The test checks an optimal covering walk on the selected tree, not a global graph tour.
//
// AI-Hints:
//   - A scalar 6 or 4 alone cannot certify EdgeIDs or return passages.
func TestSteinerWalk_HubClosedAndFixedEnd(t *testing.T) {
	g := steinerFixtureGraph(t)
	for _, edge := range []core.Edge{
		{ID: "AX", From: "A", To: "X", Weight: 1},
		{ID: "BX", From: "B", To: "X", Weight: 1},
		{ID: "CX", From: "C", To: "X", Weight: 1},
		{ID: "AB", From: "A", To: "B", Weight: 3},
		{ID: "BC", From: "B", To: "C", Weight: 3},
		{ID: "AC", From: "A", To: "C", Weight: 3},
	} {
		steinerFixtureEdge(t, g, edge.ID, edge.From, edge.To, edge.Weight)
	}
	for _, mode := range []mst.SteinerPathStorage{mst.SteinerRecomputePaths, mst.SteinerRetainPaths} {
		result, err := mst.SteinerTree(g, []string{"C", "A", "B"}, mst.WithSteinerPathStorage(mode))
		if err != nil {
			t.Fatalf("SteinerTree mode=%v: %v", mode, err)
		}
		if result.TotalWeight != 3 {
			t.Fatalf("unexpected hub network cost: %g", result.TotalWeight)
		}
		original := result.Clone()

		closed, err := mst.SteinerWalk(result, "A", "A")
		if err != nil {
			t.Fatalf("closed mode=%v: %v", mode, err)
		}
		if closed.Distance != 6 || closed.NetworkWeight != 3 ||
			!slices.Equal(closed.Vertices, []string{"A", "X", "B", "X", "C", "X", "A"}) ||
			!slices.Equal(closed.EdgeIDs, []string{"AX", "BX", "BX", "CX", "CX", "AX"}) {
			t.Fatalf("closed hub route mode=%v: %+v", mode, closed)
		}
		for id, count := range assertWalkSteps(t, closed, result.Edges) {
			if count != 2 {
				t.Fatalf("closed traversal of %q: got %d passages, want 2", id, count)
			}
		}

		fixed, err := mst.SteinerWalk(result, "A", "B")
		if err != nil {
			t.Fatalf("fixed-end mode=%v: %v", mode, err)
		}
		if fixed.Distance != 4 || fixed.NetworkWeight != 3 ||
			!slices.Equal(fixed.Vertices, []string{"A", "X", "C", "X", "B"}) ||
			!slices.Equal(fixed.EdgeIDs, []string{"AX", "CX", "CX", "BX"}) {
			t.Fatalf("fixed-end hub route mode=%v: %+v", mode, fixed)
		}
		counts := assertWalkSteps(t, fixed, result.Edges)
		if counts["AX"] != 1 || counts["BX"] != 1 || counts["CX"] != 2 {
			t.Fatalf("fixed-end spine/branch passage counts: %+v", counts)
		}
		if !slices.Equal(original.Edges, result.Edges) ||
			!slices.Equal(original.Vertices, result.Vertices) ||
			!slices.Equal(original.Terminals, result.Terminals) || original.TotalWeight != result.TotalWeight {
			t.Fatal("walk mutated the caller-owned SteinerResult")
		}
	}
}

// TestSteinerWalk_ValidPermutationsAndNonterminalLeaves protects tree-only admission.
// The input is deliberately not canonical and has a nonterminal leaf that must be traversed.
// Implementation:
//   - Stage 1: Build a valid caller-owned tree with reversed orientation and unsorted arrays.
//   - Stage 2: Trace open/closed walks and a walk with nonterminal endpoints.
//   - Stage 3: Permute all public slices and confirm unchanged route and input ownership.
//
// Behavior highlights:
//   - Nonterminal leaf Y is required network structure and must not be pruned.
//
// Inputs:
//   - A four-vertex tree with one nonterminal leaf and both required terminals.
//
// Returns:
//   - Test failure if validation normalizes by mutation or rejects a valid tree.
//
// Errors:
//   - Any unexpected ErrInvalidSteinerResult is a broken admission contract.
//
// Determinism:
//   - Neighbor/edge IDs determine identical published walks under all valid permutations.
//
// Complexity:
//   - O(V log V+E log E+S) validation and walk for this fixture.
//
// Notes:
//   - KMB provenance is deliberately absent; tree validity must suffice.
//
// AI-Hints:
//   - Do not turn the validator into a hidden Steiner pruning algorithm.
func TestSteinerWalk_ValidPermutationsAndNonterminalLeaves(t *testing.T) {
	original := []core.Edge{
		{ID: "XY", From: "Y", To: "X", Weight: 5},
		{ID: "XB", From: "B", To: "X", Weight: 3},
		{ID: "AX", From: "X", To: "A", Weight: 2},
	}
	tree := walkFixtureTree([]string{"B", "A"}, []string{"Y", "B", "X", "A"}, original)
	before := tree.Clone()

	open, err := mst.SteinerWalk(tree, "A", "B")
	if err != nil {
		t.Fatal(err)
	}
	if open.NetworkWeight != 10 || open.Distance != 15 ||
		!slices.Equal(open.Vertices, []string{"A", "X", "Y", "X", "B"}) ||
		!slices.Equal(open.EdgeIDs, []string{"AX", "XY", "XY", "XB"}) {
		t.Fatalf("off-spine nonterminal leaf omitted: %+v", open)
	}
	assertWalkSteps(t, open, original)

	closed, err := mst.SteinerWalk(tree, "A", "A")
	if err != nil || closed.NetworkWeight != 10 || closed.Distance != 20 {
		t.Fatalf("closed nonterminal tree: %+v error=%v", closed, err)
	}
	assertWalkSteps(t, closed, original)

	// A nonterminal leaf and a nonterminal internal vertex are also legal
	// explicitly requested endpoints: their membership, not KMB provenance, matters.
	nonterminal, err := mst.SteinerWalk(tree, "Y", "X")
	if err != nil || nonterminal.Distance != 15 || nonterminal.NetworkWeight != 10 {
		t.Fatalf("nonterminal endpoints: %+v error=%v", nonterminal, err)
	}
	assertWalkSteps(t, nonterminal, original)

	permuted := tree.Clone()
	slices.Reverse(permuted.Edges)
	slices.Reverse(permuted.Vertices)
	slices.Reverse(permuted.Terminals)
	permutedRoute, err := mst.SteinerWalk(permuted, "A", "B")
	if err != nil || !slices.Equal(permutedRoute.Vertices, open.Vertices) ||
		!slices.Equal(permutedRoute.EdgeIDs, open.EdgeIDs) || permutedRoute.Distance != open.Distance {
		t.Fatalf("valid input permutation changed traversal: %+v error=%v", permutedRoute, err)
	}
	if !slices.Equal(tree.Edges, before.Edges) ||
		!slices.Equal(tree.Vertices, before.Vertices) || !slices.Equal(tree.Terminals, before.Terminals) {
		t.Fatal("validation reordered the caller's edge or vertex slices")
	}
}

// TestSteinerWalk_SingletonAndResultCopies verifies degenerate route and nilability laws.
// It also protects independent backing arrays for both public walk slices.
// Implementation:
//   - Stage 1: Exercise nil-safe WalkResult methods and the zero-edge singleton.
//   - Stage 2: Clone both singleton and nonempty route artifacts.
//   - Stage 3: Mutate clones and assert independent route slices and unchanged originals.
//
// Behavior highlights:
//   - A singleton visits its vertex exactly once and has no passage IDs.
//
// Inputs:
//   - One single-vertex tree and one two-vertex tree.
//
// Returns:
//   - Test failure for wrong nilability, result shape, cost, or aliasing.
//
// Errors:
//   - Any unexpected error for a valid singleton or pair fails the test.
//
// Determinism:
//   - Exact scalar and slice equality without concurrent modification.
//
// Complexity:
//   - O(V+E+S) for tiny fixtures and route clone operations.
//
// Notes:
//   - Copy methods do not validate caller-corrupted results.
//
// AI-Hints:
//   - Distinct slice backing arrays are required even when the result is caller-owned.
func TestSteinerWalk_SingletonAndResultCopies(t *testing.T) {
	var absent *mst.WalkResult
	if !absent.IsNil() || absent.Clone() != nil {
		t.Fatal("nil WalkResult helpers must be safe")
	}
	// Non-nil empty slices remain non-nil after Clone. Their nilability is
	// an ownership property, not a request to normalize caller-mutated state.
	emptySlices := (&mst.WalkResult{Vertices: []string{}, EdgeIDs: []string{}}).Clone()
	if emptySlices.Vertices == nil || emptySlices.EdgeIDs == nil {
		t.Fatal("Clone discarded distinct non-nil empty slice states")
	}
	result, err := mst.SteinerWalk(walkFixtureTree([]string{"S"}, []string{"S"}, nil), "S", "S")
	if err != nil || result.IsNil() || result.Distance != 0 || result.NetworkWeight != 0 ||
		!slices.Equal(result.Vertices, []string{"S"}) || len(result.EdgeIDs) != 0 {
		t.Fatalf("singleton result=%+v error=%v", result, err)
	}
	copyResult := result.Clone()
	copyResult.Vertices[0] = "mutated"
	if result.Vertices[0] != "S" {
		t.Fatal("Clone shared the singleton Vertices backing array")
	}
	full := walkFixtureTree([]string{"A", "B"}, []string{"A", "B"}, []core.Edge{
		{ID: "ab", From: "B", To: "A", Weight: 3},
	})
	walk, err := mst.SteinerWalk(full, "A", "B")
	if err != nil {
		t.Fatal(err)
	}
	clone := walk.Clone()
	clone.Vertices[0] = "changed"
	clone.EdgeIDs[0] = "changed"
	if walk.Vertices[0] != "A" || walk.EdgeIDs[0] != "ab" {
		t.Fatal("Clone must detach both route slice backings")
	}
	if walk.NetworkWeight != 3 || walk.Distance != 3 {
		t.Fatal("singleton/pair numerical semantics changed")
	}
}

// TestSteinerWalk_EndpointValidation applies explicit nil, empty, and unknown ID rules.
// A bad tree is rejected independently of the caller's otherwise valid endpoint request.
// Implementation:
//   - Stage 1: Construct a valid two-vertex tree with explicit endpoints.
//   - Stage 2: Call SteinerWalk with nil tree, empty IDs, and missing IDs.
//   - Stage 3: Check errors.Is and tree-validation precedence over endpoint membership.
//
// Behavior highlights:
//   - Empty strings never mean a default endpoint or closed-walk selection.
//
// Inputs:
//   - Invalid request variants and one separately corrupted tree.
//
// Returns:
//   - Test failure if any invalid request publishes even a partial WalkResult.
//
// Errors:
//   - ErrNilResult, core.ErrEmptyVertexID, core.ErrVertexNotFound, ErrInvalidSteinerResult.
//
// Determinism:
//   - Stable first-failure validation order for the same request and tree state.
//
// Complexity:
//   - O(V log V+E log E) for the complete validation cases.
//
// Notes:
//   - Empty endpoint rejection occurs before tree copying; nonempty membership occurs after it.
//
// AI-Hints:
//   - Do not silently replace endID=="" with startID.
func TestSteinerWalk_EndpointValidation(t *testing.T) {
	valid := walkFixtureTree([]string{"A", "B"}, []string{"A", "B"}, []core.Edge{
		{ID: "ab", From: "A", To: "B", Weight: 2},
	})
	for _, tc := range []struct {
		name       string
		tree       *mst.SteinerResult
		start, end string
		want       error
	}{
		{"nil-result", nil, "A", "B", mst.ErrNilResult},
		{"empty-start", valid, "", "B", core.ErrEmptyVertexID},
		{"empty-end", valid, "A", "", core.ErrEmptyVertexID},
		{"unknown-start", valid, "X", "B", core.ErrVertexNotFound},
		{"unknown-end", valid, "A", "X", core.ErrVertexNotFound},
		{"unknown-closed", valid, "X", "X", core.ErrVertexNotFound},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := mst.SteinerWalk(tc.tree, tc.start, tc.end)
			if got != nil || !errors.Is(err, tc.want) {
				t.Fatalf("walk=%+v error=%v, want %v", got, err, tc.want)
			}
		})
	}
	broken := valid.Clone()
	broken.TotalWeight = 999
	if got, err := mst.SteinerWalk(broken, "missing", "B"); got != nil || !errors.Is(err, mst.ErrInvalidSteinerResult) {
		t.Fatalf("tree integrity must precede nonempty endpoint membership: result=%+v error=%v", got, err)
	}
}

// TestSteinerWalk_ActualStepOverflowVsFiniteOpen verifies the two separate numeric domains.
// The closed route can overflow even though the same valid network has a finite open route.
// Implementation:
//   - Stage 1: Choose a finite one-edge network whose second passage overflows.
//   - Stage 2: Check one-way success and closed-walk ErrWeightOverflow with nil output.
//   - Stage 3: Check a finite three-spoke network with an overflowing fixed-end walk.
//
// Behavior highlights:
//   - Actual passage arithmetic, not only final network arithmetic, determines success.
//
// Inputs:
//   - Binary-exact powers of two near the representable float64 maximum.
//
// Returns:
//   - Test failure if overflow is ignored, mislabeled, or leaks a partial route.
//
// Errors:
//   - ErrWeightOverflow for a nonfinite repeated-step prefix.
//
// Determinism:
//   - Exact powers of two eliminate threshold/tolerance ambiguity.
//
// Complexity:
//   - O(V+E+S) on constant-size tree fixtures.
//
// Notes:
//   - Open/closed endpoint choice can change representability while NetworkWeight stays finite.
//
// AI-Hints:
//   - Never substitute unchecked 2*NetworkWeight-pathDistance for the route's step fold.
func TestSteinerWalk_ActualStepOverflowVsFiniteOpen(t *testing.T) {
	const huge = 0x1p1023
	tree := walkFixtureTree([]string{"A", "B"}, []string{"B", "A"}, []core.Edge{
		{ID: "AB", From: "B", To: "A", Weight: huge},
	})
	open, err := mst.SteinerWalk(tree, "A", "B")
	if err != nil || open.Distance != huge || open.NetworkWeight != huge {
		t.Fatalf("finite one-way cost: %+v error=%v", open, err)
	}
	assertWalkSteps(t, open, tree.Edges)
	closed, err := mst.SteinerWalk(tree, "A", "A")
	if closed != nil || !errors.Is(err, mst.ErrWeightOverflow) || errors.Is(err, mst.ErrInvalidSteinerResult) {
		t.Fatalf("actual repeated passage must overflow without a partial route: %+v error=%v", closed, err)
	}

	// All three unique edges fit in one total, but the A-to-B route must
	// visit the C branch twice and therefore cannot fit its fourth huge step.
	const spoke = 0x1p1022
	star := walkFixtureTree([]string{"A", "B", "C"}, []string{"A", "B", "C", "X"}, []core.Edge{
		{ID: "AX", From: "A", To: "X", Weight: spoke},
		{ID: "BX", From: "B", To: "X", Weight: spoke},
		{ID: "CX", From: "C", To: "X", Weight: spoke},
	})
	if star.TotalWeight != 3*spoke || math.IsInf(star.TotalWeight, 0) {
		t.Fatalf("fixture must have finite unique-edge total: %g", star.TotalWeight)
	}
	if walk, err := mst.SteinerWalk(star, "A", "B"); walk != nil || !errors.Is(err, mst.ErrWeightOverflow) {
		t.Fatalf("fourth actual traversal must overflow: walk=%+v error=%v", walk, err)
	}
}

// TestSteinerWalk_DeepChainUsesIterativeTraversal exercises long closed and fixed-end routes.
// Its depth is far beyond ordinary example sizes and does not rely on recursive Go calls.
// Implementation:
//   - Stage 1: Construct a long valid path with 30,000 vertices and unit weights.
//   - Stage 2: Cover it from end to end without return passages.
//   - Stage 3: Cover it as a closed walk with exactly two passages per edge.
//
// Behavior highlights:
//   - Explicit traversal frames handle deep nesting without recursive call growth.
//
// Inputs:
//   - A path of deterministic original IDs, requiring O(V) route output.
//
// Returns:
//   - Test failure for lost edges, wrong endpoints, or an incomplete deep traversal.
//
// Errors:
//   - Any error for this finite and structurally valid tree fails the test.
//
// Determinism:
//   - Unique neighbor relationships and lexical IDs remove all branch ties.
//
// Complexity:
//   - O(V log V+E log E+S) including tree preparation and route output.
//
// Notes:
//   - Error output is summarized to avoid dumping tens of thousands of vertices.
//
// AI-Hints:
//   - Do not replace the explicit DFS stack with a deeply nested recursive function.
func TestSteinerWalk_DeepChainUsesIterativeTraversal(t *testing.T) {
	const vertexCount = 30000
	vertices := make([]string, vertexCount)
	for index := range vertices {
		vertices[index] = fmt.Sprintf("v%05d", index)
	}
	edges := make([]core.Edge, 0, vertexCount-1)
	for index := 1; index < vertexCount; index++ {
		edges = append(edges, core.Edge{
			ID: fmt.Sprintf("e%05d", index), From: vertices[index-1],
			To: vertices[index], Weight: 1,
		})
	}
	tree := walkFixtureTree([]string{vertices[0], vertices[vertexCount-1]}, vertices, edges)

	open, err := mst.SteinerWalk(tree, vertices[0], vertices[vertexCount-1])
	if err != nil || len(open.EdgeIDs) != vertexCount-1 || open.Distance != vertexCount-1 {
		t.Fatalf("deep fixed-end traversal: result=%+v error=%v", walkSummary(open), err)
	}
	closed, err := mst.SteinerWalk(tree, vertices[0], vertices[0])
	if err != nil || len(closed.EdgeIDs) != 2*(vertexCount-1) || closed.Distance != 2*(vertexCount-1) {
		t.Fatalf("deep closed traversal: result=%+v error=%v", walkSummary(closed), err)
	}
	if open.Vertices[len(open.Vertices)-1] != vertices[vertexCount-1] ||
		closed.Vertices[len(closed.Vertices)-1] != vertices[0] {
		t.Fatal("deep traversal did not finish at its explicit endpoint")
	}
}

// walkSummary avoids formatting tens of thousands of vertices when a deep-tree assertion fails.
// It returns only sizes and costs, never an alias to the original route's large slices.
// Implementation:
//   - Stage 1: Return nil for a nil result.
//   - Stage 2: Read only route lengths and scalar costs.
//   - Stage 3: Publish a small detached diagnostic value.
//
// Behavior highlights:
//   - Large route slices are never copied or printed in a failing assertion.
//
// Inputs:
//   - walk: optional caller-owned WalkResult, read without modification.
//
// Returns:
//   - any: compact diagnostic record, or nil for a nil input.
//
// Errors:
//   - None; input fields are read directly.
//
// Determinism:
//   - Pure projection of lengths and scalar costs.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This is test-only reporting, not a public API or route validator.
//
// AI-Hints:
//   - Avoid printing a complete 30,000-element walk on test failure.
func walkSummary(walk *mst.WalkResult) any {
	if walk == nil {
		return nil
	}
	return struct {
		VertexVisits, EdgePassages int
		NetworkWeight, Distance    float64
	}{len(walk.Vertices), len(walk.EdgeIDs), walk.NetworkWeight, walk.Distance}
}

// TestSteinerWalk_ExactIntegerRouteOracle checks the edge multiplicity theorem independently.
// Bounded random trees include zero-cost branches and all possible ordered endpoint choices.
// Implementation:
//   - Stage 1: Generate bounded independent trees with small exact integer weights.
//   - Stage 2: Compute all requested closed/fixed-end walks and verify each original passage.
//   - Stage 3: Compare each edge multiplicity and 2W-d_tree(s,t) against an independent BFS oracle.
//
// Behavior highlights:
//   - Zero-cost branches still require passages even though their cost is zero.
//
// Inputs:
//   - Deterministically seeded random trees and every ordered pair of their vertices.
//
// Returns:
//   - Test failure for invalid edge witnesses, multiplicity, or optimal tree-covering length.
//
// Errors:
//   - Any unexpected error on exact-integer valid trees fails the test.
//
// Determinism:
//   - Fixed RNG seed, original edge IDs and canonical walk neighbor order.
//
// Complexity:
//   - O(C*V^2*(V+E+S)) for C bounded tiny random trees.
//
// Notes:
//   - The independent oracle never invokes production Steiner or DSU helpers.
//
// AI-Hints:
//   - Never require KMB to find exact graph Steiner OPT; this oracle tests walks on given trees.
func TestSteinerWalk_ExactIntegerRouteOracle(t *testing.T) {
	generator := rand.New(rand.NewSource(20261103))
	const instances = 28
	for instance := 0; instance < instances; instance++ {
		vertexCount := 3 + generator.Intn(7)
		vertices := make([]string, vertexCount)
		for index := range vertices {
			vertices[index] = fmt.Sprintf("n%02d", index)
		}
		edges := make([]core.Edge, 0, vertexCount-1)
		for index := 1; index < vertexCount; index++ {
			parent := generator.Intn(index)
			edges = append(edges, core.Edge{
				ID: fmt.Sprintf("id%02d", index), From: vertices[index],
				To: vertices[parent], Weight: float64(generator.Intn(9)),
			})
		}
		generator.Shuffle(len(edges), func(i, j int) { edges[i], edges[j] = edges[j], edges[i] })
		tree := walkFixtureTree([]string{vertices[0], vertices[vertexCount-1]}, vertices, edges)
		for _, start := range vertices {
			for _, end := range vertices {
				got, err := mst.SteinerWalk(tree, start, end)
				if err != nil {
					t.Fatalf("instance=%d start=%q end=%q: %v", instance, start, end, err)
				}
				counts := assertWalkSteps(t, got, edges)
				spineDistance, spineIDs := oracleWalkSpine(t, edges, start, end)
				if got.NetworkWeight != tree.TotalWeight || got.Distance != 2*tree.TotalWeight-spineDistance {
					t.Fatalf("instance=%d %s->%s: W=%g L=%g, want L=%g",
						instance, start, end, got.NetworkWeight, got.Distance, 2*tree.TotalWeight-spineDistance)
				}
				for _, edge := range edges {
					want := 2
					if spineIDs[edge.ID] {
						want = 1
					}
					if counts[edge.ID] != want {
						t.Fatalf("instance=%d %s->%s: edge %q crossed %d times; want %d",
							instance, start, end, edge.ID, counts[edge.ID], want)
					}
				}
			}
		}
	}
}

// oracleWalkSpine independently reconstructs the unique tree path for exact-integer oracles.
// It uses raw edge records and a simple BFS rather than the walk's indexed graph or DSU.
// Implementation:
//   - Stage 1: Build independent undirected adjacency from raw original-edge records.
//   - Stage 2: Run a small BFS to record one parent edge per discovered vertex.
//   - Stage 3: Follow the target predecessor chain to produce unique path IDs and cost.
//
// Behavior highlights:
//   - Equal endpoints yield zero distance and an empty spine-edge set.
//
// Inputs:
//   - t: test handle; edges: bounded connected tree; start/end: declared vertex IDs.
//
// Returns:
//   - float64: exact-integer path cost; map[string]bool: path edge membership.
//
// Errors:
//   - Fail test if independent fixture endpoints are not connected.
//
// Determinism:
//   - In a tree only one simple path exists, regardless of BFS adjacency order.
//
// Complexity:
//   - Time O(V+E), Space O(V+E), restricted to small oracle fixtures.
//
// Notes:
//   - No private or exported production shortest-path/kernel helpers are called.
//
// AI-Hints:
//   - Do not reuse the walk's spine finder, or a shared bug would escape detection.
func oracleWalkSpine(t *testing.T, edges []core.Edge, start, end string) (float64, map[string]bool) {
	t.Helper()
	spine := make(map[string]bool)
	if start == end {
		return 0, spine
	}
	neighbors := make(map[string][]struct{ to, edgeID string })
	weights := make(map[string]float64)
	for _, edge := range edges {
		neighbors[edge.From] = append(neighbors[edge.From], struct{ to, edgeID string }{edge.To, edge.ID})
		neighbors[edge.To] = append(neighbors[edge.To], struct{ to, edgeID string }{edge.From, edge.ID})
		weights[edge.ID] = edge.Weight
	}
	parent := map[string]string{start: start}
	parentEdge := make(map[string]string)
	queue := []string{start}
	for head := 0; head < len(queue); head++ {
		vertex := queue[head]
		for _, neighbor := range neighbors[vertex] {
			if _, seen := parent[neighbor.to]; seen {
				continue
			}
			parent[neighbor.to] = vertex
			parentEdge[neighbor.to] = neighbor.edgeID
			queue = append(queue, neighbor.to)
		}
	}
	if _, exists := parent[end]; !exists {
		t.Fatalf("test oracle tree does not connect %q and %q", start, end)
	}
	var distance float64
	for vertex := end; vertex != start; vertex = parent[vertex] {
		id := parentEdge[vertex]
		spine[id] = true
		distance += weights[id]
	}
	return distance, spine
}

// TestSteinerWalk_ZeroCostEdgesStillRequirePassages protects topology from numeric shortcuts.
// An off-spine branch with zero weight still must appear in the published original-edge witness.
// Implementation:
//   - Stage 1: Construct an original tree with a zero-weight spine edge and zero-weight leaf.
//   - Stage 2: Compute a fixed-end walk and inspect every required return passage.
//   - Stage 3: Compute a closed walk and check that all three edges occur twice.
//
// Behavior highlights:
//   - Zero-weight edges cannot be dropped just because their contribution to Distance is zero.
//
// Inputs:
//   - An explicit four-vertex tree with two zero-cost edges.
//
// Returns:
//   - Test failure on a missing edge or incorrect fixed-end/closed route length.
//
// Errors:
//   - No error is expected for finite nonnegative original-edge weights.
//
// Determinism:
//   - Per-vertex lexical neighbor order chooses a stable branch-first route.
//
// Complexity:
//   - Time O(V+E log E+S), Space O(V+E+S) for the fixed fixture.
//
// Notes:
//   - Costs alone cannot verify route coverage when some edges have weight zero.
//
// AI-Hints:
//   - Never use a weight>0 filter to decide whether an original tree edge must be visited.
func TestSteinerWalk_ZeroCostEdgesStillRequirePassages(t *testing.T) {
	tree := walkFixtureTree([]string{"B", "A"}, []string{"A", "X", "Y", "B"}, []core.Edge{
		{ID: "AX", From: "A", To: "X", Weight: 0},
		{ID: "XB", From: "X", To: "B", Weight: 2},
		{ID: "XY", From: "Y", To: "X", Weight: 0},
	})
	open, err := mst.SteinerWalk(tree, "A", "B")
	if err != nil || open.NetworkWeight != 2 || open.Distance != 2 ||
		!slices.Equal(open.EdgeIDs, []string{"AX", "XY", "XY", "XB"}) {
		t.Fatalf("zero-cost fixed-end edge coverage: walk=%+v error=%v", open, err)
	}
	counts := assertWalkSteps(t, open, tree.Edges)
	if counts["AX"] != 1 || counts["XY"] != 2 || counts["XB"] != 1 {
		t.Fatalf("zero-cost branch passage counts: %+v", counts)
	}
	closed, err := mst.SteinerWalk(tree, "A", "A")
	if err != nil || closed.NetworkWeight != 2 || closed.Distance != 4 {
		t.Fatalf("zero-cost closed walk: walk=%+v error=%v", closed, err)
	}
	for id, count := range assertWalkSteps(t, closed, tree.Edges) {
		if count != 2 {
			t.Fatalf("closed passage for %q: %d, want 2", id, count)
		}
	}
}
