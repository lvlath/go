// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package match_test verifies defensive Blossom contracts using isolated fixtures.
// Public objective tests remain independent; this file tests actual private state transitions.
package match_test

import (
	"math"
	"slices"
	"testing"

	"github.com/lvlath/go/match"
)

// TestBlossomLiftedCandidateRollback checks a failure after a real partial flip.
// The first candidate clears the middle match and inserts one boundary edge before
// the conflicting last edge fails. Both mate arrays and bases must be restored.
func TestBlossomLiftedCandidateRollback(t *testing.T) {
	e, s := newInvariantEngine(t, 4)
	left := invariantEdgeID(t, e, 0, 1)
	middle := invariantEdgeID(t, e, 1, 2)
	conflict := invariantEdgeID(t, e, 0, 3)
	right := invariantEdgeID(t, e, 2, 3)
	requireInvariantOK(t, match.SetMatchedEdgeForTest(e, middle))
	beforeMate, beforeEdges, beforeBase := slices.Clone(s.Mate), slices.Clone(s.MateEdge), slices.Clone(s.Base)
	bad := []int{left, middle, conflict}
	good := []int{left, middle, right}
	event := match.NewBlossomEventForTest(match.EventAugmentForTest, left, 0, 3, 0, 3)
	// Both candidates alternate by edge status, but only the second is a valid path.
	requireInvariantOK(t, match.VerifyAugmentingEdgeSequenceForTest(e, bad))
	mustErrorIs(t, match.TryLiftedAugmentingCandidatesForTest(e, event, nil, [][]int{bad}), match.ErrInvalidMatching)
	if !slices.Equal(s.Mate, beforeMate) || !slices.Equal(s.MateEdge, beforeEdges) || !slices.Equal(s.Base, beforeBase) {
		t.Fatalf("failed lifted candidate leaked state: mate=%v edges=%v base=%v", s.Mate, s.MateEdge, s.Base)
	}
	mustErrorIs(t, match.TryLiftedAugmentingCandidatesForTest(e, event, nil, nil), match.ErrInvalidMatching)
	// A failed first choice cannot prevent the valid second choice from committing.
	requireInvariantOK(t, match.TryLiftedAugmentingCandidatesForTest(e, event, nil, [][]int{bad, good}))
	if !slices.Equal(s.Mate, []int{1, 0, 3, 2}) || !slices.Equal(s.MateEdge, []int{left, left, right, right}) {
		t.Fatalf("valid candidate not committed: mate=%v edges=%v", s.Mate, s.MateEdge)
	}
}

// TestBlossomLiftedCandidateBaseRollback checks rollback after base refresh fails.
// An invalid lifted candidate leaves two exposed internal vertices after flipping;
// it cannot be committed as an augmentation of a contracted odd blossom.
func TestBlossomLiftedCandidateBaseRollback(t *testing.T) {
	e, s, node := contractedTriangle(t)
	beforeMate, beforeEdges, beforeBase := slices.Clone(s.Mate), slices.Clone(s.MateEdge), slices.Clone(s.Base)
	bad := []int{invariantEdgeID(t, e, 0, 3), invariantEdgeID(t, e, 1, 2), invariantEdgeID(t, e, 4, 5)}
	event := match.NewBlossomEventForTest(match.EventAugmentForTest, bad[0], node, 3, 0, 3)
	mustErrorIs(t, match.TryLiftedAugmentingCandidatesForTest(e, event, nil, [][]int{bad}), match.ErrInvalidMatching)
	if !slices.Equal(s.Mate, beforeMate) || !slices.Equal(s.MateEdge, beforeEdges) || !slices.Equal(s.Base, beforeBase) {
		t.Fatalf("base-refresh rollback leaked: mates=%v base=%v", s.Mate, s.Base)
	}
	base, err := match.CurrentBlossomBaseFromMatchingForTest(e, node)
	if err != nil || base != 0 {
		t.Fatalf("restored base=%d, err=%v", base, err)
	}
}

// TestBlossomMalformedAlternatingEdges covers range, endpoint, parity and duplicate laws.
// Every rejected operation must preserve the previously committed reciprocal pair.
func TestBlossomMalformedAlternatingEdges(t *testing.T) {
	e, s := newInvariantEngine(t, 4)
	middle := invariantEdgeID(t, e, 1, 2)
	left := invariantEdgeID(t, e, 0, 1)
	right := invariantEdgeID(t, e, 2, 3)
	requireInvariantOK(t, match.SetMatchedEdgeForTest(e, middle))
	beforeMate, beforeEdges := slices.Clone(s.Mate), slices.Clone(s.MateEdge)
	for _, tc := range []struct {
		name string
		path []int
	}{
		{"empty", nil}, {"even", []int{left, right}},
		{"out-of-range", []int{match.NoEdgeForTest}},
		{"duplicate", []int{left, middle, left}},
		{"matched-first", []int{middle}},
		{"matched-last", []int{left, right, middle}},
		{"two-unmatched", []int{left, right, invariantEdgeID(t, e, 0, 3)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mustErrorIs(t, match.FlipAugmentingPathForTest(e, tc.path), match.ErrInvalidMatching)
			if !slices.Equal(s.Mate, beforeMate) || !slices.Equal(s.MateEdge, beforeEdges) {
				t.Fatal("invalid edge sequence changed the committed matching")
			}
		})
	}
	// Direct insert/clear guards protect both endpoints, not only the lower ID.
	for _, id := range []int{match.NoEdgeForTest, left, right} {
		mustErrorIs(t, match.SetMatchedEdgeForTest(e, id), match.ErrInvalidMatching)
		mustErrorIs(t, match.ClearMatchedEdgeForTest(e, id), match.ErrInvalidMatching)
	}
	if match.CanAppendAlternatingForTest(e, nil, []int{match.NoEdgeForTest}) || match.EdgeSequenceAlternatesForTest(e, []int{left, right}) {
		t.Fatal("malformed or same-parity edge sequence accepted")
	}
	if !slices.Equal(s.Mate, beforeMate) || !slices.Equal(s.MateEdge, beforeEdges) {
		t.Fatal("direct rejected insert/clear changed the committed matching")
	}
}

// TestBlossomFinalStateRefusesCorruption isolates mate, ownership and dual failures.
// These states cannot be supplied through Perfect; the test-only bridge is required.
func TestBlossomFinalStateRefusesCorruption(t *testing.T) {
	for _, tc := range []struct {
		name    string
		corrupt func(match.BlossomStateForTest)
	}{
		{"mate-outside-domain", func(s match.BlossomStateForTest) { s.Mate[0] = len(s.Mate) }},
		{"self-mate", func(s match.BlossomStateForTest) { s.Mate[0] = 0 }},
		{"nonreciprocal-mate", func(s match.BlossomStateForTest) { s.Mate[s.Mate[0]] = match.NoVertexForTest }},
		{"missing-mate-edge", func(s match.BlossomStateForTest) { s.MateEdge[0] = match.NoEdgeForTest }},
		{"inactive-owner", func(s match.BlossomStateForTest) { s.Active[s.InBlossom[0]] = false }},
		{"negative-slack", func(s match.BlossomStateForTest) { s.Dual[0] = -1e6 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e, s := newInvariantEngine(t, 4)
			requireInvariantOK(t, match.SolveForTest(e))
			requireInvariantOK(t, match.VerifyOptimalStateForTest(e))
			tc.corrupt(s)
			mustErrorIs(t, match.VerifyOptimalStateForTest(e), match.ErrInvalidMatching)
		})
	}
	e, s := newInvariantEngine(t, 4)
	if mates, err := match.ExportMatchingForTest(e); mates != nil || err == nil {
		t.Fatalf("incomplete matching exported: %v %v", mates, err)
	}
	// Make a complete but inconsistent local mate array and verify export refuses it.
	requireInvariantOK(t, match.SolveForTest(e))
	s.Mate[0] = 0
	mates, err := match.ExportMatchingForTest(e)
	mustErrorIs(t, err, match.ErrInvalidMatching)
	if mates != nil {
		t.Fatalf("invalid mates published: %v", mates)
	}
}

// TestBlossomForestRejectsInvalidEvents checks the event admission laws.
// Cases are separate fixtures so no failed state transition is reused.
func TestBlossomForestRejectsInvalidEvents(t *testing.T) {
	for _, tc := range []struct {
		name      string
		modify    func(match.BlossomStateForTest)
		endpoints [4]int
	}{
		{"out-of-range-node", nil, [4]int{-1, 2, 0, 2}},
		{"inactive-target", func(s match.BlossomStateForTest) { s.Active[2] = false }, [4]int{0, 2, 0, 2}},
		{"wrong-label", func(s match.BlossomStateForTest) { s.Label[0] = match.InnerForTest }, [4]int{0, 2, 0, 2}},
		{"foreign-entry-vertex", nil, [4]int{0, 2, 0, 3}},
		{"missing-mate", func(s match.BlossomStateForTest) { s.Mate[2] = match.NoVertexForTest }, [4]int{0, 2, 0, 2}},
		{"invalid-mate-owner", func(s match.BlossomStateForTest) { s.InBlossom[1] = match.NoNodeForTest }, [4]int{0, 2, 0, 2}},
		{"same-top-owner", func(s match.BlossomStateForTest) { s.InBlossom[1] = 2 }, [4]int{0, 2, 0, 2}},
		{"already-labeled-mate", func(s match.BlossomStateForTest) { s.Label[1] = match.OuterForTest }, [4]int{0, 2, 0, 2}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e, s := newInvariantEngine(t, 4)
			requireInvariantOK(t, match.SetMatchedEdgeForTest(e, invariantEdgeID(t, e, 1, 2)))
			match.AssignOuterRootForTest(e, 0)
			if tc.modify != nil {
				tc.modify(s)
			}
			v := tc.endpoints
			event := match.NewBlossomEventForTest(match.EventGrowForTest, invariantEdgeID(t, e, 0, 2), v[0], v[1], v[2], v[3])
			done, err := match.ApplyBlossomEventForTest(e, event)
			mustErrorIs(t, err, match.ErrInvalidMatching)
			if done {
				t.Fatal("rejected grow reported a completed augmentation")
			}
		})
	}
	e, s := newInvariantEngine(t, 4)
	for _, event := range []match.BlossomEventForTest{
		match.NewBlossomEventForTest(match.EventNoneForTest, -1, -1, -1, -1, -1),
		match.NewBlossomEventForTest(match.EventShrinkForTest, -1, 0, 1, 0, 1),
		match.NewBlossomEventForTest(match.EventAugmentForTest, -1, -1, 1, 0, 1),
		match.NewBlossomEventForTest(match.EventExpandForTest, -1, 0, 0, 0, 0),
	} {
		done, err := match.ApplyBlossomEventForTest(e, event)
		mustErrorIs(t, err, match.ErrInvalidMatching)
		if done {
			t.Fatal("invalid event completed an augmentation")
		}
	}
	// A correctly allocated outer node is required before scanning incident edges.
	_, _, err := match.ScanOuterNodeTightEdgesForTest(e, -1)
	mustErrorIs(t, err, match.ErrInvalidMatching)
	s.Members[0] = []int{len(s.Mate)}
	_, _, err = match.ScanOuterNodeTightEdgesForTest(e, 0)
	mustErrorIs(t, err, match.ErrInvalidMatching)
	s.Members[0] = []int{0}
	s.Incident[0] = []int{match.NoEdgeForTest}
	_, _, err = match.ScanOuterNodeTightEdgesForTest(e, 0)
	mustErrorIs(t, err, match.ErrInvalidMatching)
}

// TestBlossomPathAndCycleRejection checks coordinate ownership and path continuity.
// Local vertex IDs, active node IDs and cycle indices are three distinct domains.
func TestBlossomPathAndCycleRejection(t *testing.T) {
	e, s, node := contractedTriangle(t)
	if _, err := match.TopNodeOfVertexForTest(e, -1); err == nil {
		t.Fatal("negative vertex accepted")
	}
	if match.NodeContainsVertexForTest(e, node, -1) || match.NodeContainsVertexForTest(e, -1, 0) {
		t.Fatal("invalid membership accepted")
	}
	edge := invariantEdgeID(t, e, 0, 3)
	for _, args := range [][3]int{{-1, node, 3}, {edge, -1, 3}, {edge, node, node}, {edge, 1, 3}, {edge, node, 4}} {
		_, _, err := match.OrientEdgeForNodesForTest(e, args[0], args[1], args[2])
		mustErrorIs(t, err, match.ErrInvalidMatching)
	}
	if _, err := match.MakePathStepForTest(e, edge, node, 4); err == nil {
		t.Fatal("foreign boundary endpoint accepted")
	}
	if _, err := match.PathToRootForTest(e, 1); err == nil {
		t.Fatal("inactive child accepted as an outer root")
	}
	if _, err := match.PathToRootForTest(e, 3); err == nil {
		t.Fatal("unlabeled node accepted as outer")
	}
	match.AssignOuterRootForTest(e, 3)
	s.TreeRoot[3] = 4 // A non-root outer node must have a parent edge.
	_, err := match.PathToRootForTest(e, 3)
	mustErrorIs(t, err, match.ErrInvalidMatching)
	_, err = match.NodePathToRootForTest(e, -1)
	mustErrorIs(t, err, match.ErrInvalidMatching)
	_, err = match.CycleIndexPathForTest(e, node, -1, 0, true)
	mustErrorIs(t, err, match.ErrInvalidMatching)
	_, err = match.CycleChildIndexContainingVertexForTest(e, node, 5)
	mustErrorIs(t, err, match.ErrInvalidMatching)
	_, err = match.BaseChildIndexForTest(e, 0)
	mustErrorIs(t, err, match.ErrInvalidMatching)
	_, err = match.LiftThroughBlossomChoicesForTest(e, 3, 3, 4)
	mustErrorIs(t, err, match.ErrInvalidMatching)
	_, err = match.LiftThroughBlossomChoicesForTest(e, node, 0, 5)
	mustErrorIs(t, err, match.ErrInvalidMatching)
	_, err = match.LiftAugmentingPathChoicesForTest(e, nil)
	mustErrorIs(t, err, match.ErrInvalidMatching)
	// A bridge between a blossom and a singleton cannot be followed by a step
	// departing from another singleton. The top-level path must be contiguous.
	first := match.NewBlossomPathStepForTest(edge, node, 3, 0, 3)
	second := match.NewBlossomPathStepForTest(invariantEdgeID(t, e, 4, 5), 4, 5, 4, 5)
	mustErrorIs(t, match.VerifyPathStepContinuityForTest(e, []match.BlossomPathStepForTest{first, second}), match.ErrInvalidMatching)
	_, err = match.ComposeAugmentingPathForTest(e, nil, first, []match.BlossomPathStepForTest{second})
	mustErrorIs(t, err, match.ErrInvalidMatching)
	// Both directions around a valid odd cycle must agree on the physical edge.
	steps := s.Cycles[node]
	for i := range steps {
		j := (i + 1) % len(steps)
		forward, err := match.CycleEdgeBetweenAdjacentIndicesForTest(steps, i, j)
		requireInvariantOK(t, err)
		backward, err := match.CycleEdgeBetweenAdjacentIndicesForTest(steps, j, i)
		requireInvariantOK(t, err)
		if forward != backward || forward != steps[i].EdgeToNextForTest() {
			t.Fatal("cycle edge direction mismatch")
		}
	}
	for _, args := range [][2]int{{-1, 0}, {0, 0}, {0, len(steps)}} {
		_, err := match.CycleEdgeBetweenAdjacentIndicesForTest(steps, args[0], args[1])
		mustErrorIs(t, err, match.ErrInvalidMatching)
	}
	_, err = match.CycleEdgeBetweenAdjacentIndicesForTest(steps[:2], 0, 1)
	mustErrorIs(t, err, match.ErrInvalidMatching)
}

// TestBlossomDualCandidateAdmissionAndTieOrder checks the two tolerance domains.
// Inadmissible candidates never replace a feasible delta; a genuine tie uses IDs.
func TestBlossomDualCandidateAdmissionAndTieOrder(t *testing.T) {
	e, s := newInvariantEngine(t, 4)
	baseline := match.NewBlossomDeltaForTest(match.DeltaJoinForTest, 2, 4, 1)
	for _, value := range []float64{math.NaN(), math.Inf(1), -2} {
		best := baseline
		candidate := match.NewBlossomDeltaForTest(match.DeltaGrowForTest, value, 0, 0)
		match.ImproveDeltaForTest(e, &best, candidate)
		if best != baseline {
			t.Fatalf("invalid delta changed best: %v", value)
		}
		mustErrorIs(t, match.ApplyDeltaForTest(e, candidate), match.ErrInvalidMatching)
	}
	best := baseline
	match.ImproveDeltaForTest(e, &best, match.NewBlossomDeltaForTest(match.DeltaGrowForTest, -match.DualToleranceForTest(e)/2, 0, 0))
	if best.ValueForTest() != 0 {
		t.Fatalf("small negative delta must clamp to zero: %v", best)
	}
	for _, tc := range []struct {
		name string
		a, b match.BlossomDeltaForTest
		want bool
	}{
		{"edge-present", match.NewBlossomDeltaForTest(match.DeltaGrowForTest, 2, 0, -1), match.NewBlossomDeltaForTest(match.DeltaExpandForTest, 2, -1, 1), true},
		{"edge-absent", match.NewBlossomDeltaForTest(match.DeltaExpandForTest, 2, -1, 1), baseline, false},
		{"smaller-edge", match.NewBlossomDeltaForTest(match.DeltaGrowForTest, 2, 1, 0), baseline, true},
		{"node-present", match.NewBlossomDeltaForTest(match.DeltaExpandForTest, 2, -1, 0), match.NewBlossomDeltaForTest(match.DeltaExpandForTest, 2, -1, -1), true},
		{"node-absent", match.NewBlossomDeltaForTest(match.DeltaExpandForTest, 2, -1, -1), match.NewBlossomDeltaForTest(match.DeltaExpandForTest, 2, -1, 0), false},
		{"smaller-node", match.NewBlossomDeltaForTest(match.DeltaExpandForTest, 2, -1, 0), match.NewBlossomDeltaForTest(match.DeltaExpandForTest, 2, -1, 1), true},
		{"kind", match.NewBlossomDeltaForTest(match.DeltaGrowForTest, 2, 4, 1), baseline, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := match.DeltaTieLessForTest(tc.a, tc.b); got != tc.want {
				t.Fatalf("tie order got %v want %v", got, tc.want)
			}
		})
	}
	// Scale fallback must remain finite and positive even for a malformed fixture.
	for _, scale := range []float64{0, math.NaN(), math.Inf(1)} {
		*s.Scale = scale
		if tol := match.DualSelectionToleranceForTest(e); !(tol > 0) || math.IsInf(tol, 0) {
			t.Fatalf("selection tolerance=%v", tol)
		}
	}
	// No labeled roots means there is no admissible grow, join or expansion event.
	_, err := match.NextDeltaForTest(e)
	mustErrorIs(t, err, match.ErrIncompleteGraph)
	mustErrorIs(t, match.ShiftMemberVertexDualsForTest(e, -1, 1), match.ErrInvalidMatching)
	s.Members[0] = []int{len(s.Mate)}
	mustErrorIs(t, match.ShiftMemberVertexDualsForTest(e, 0, 1), match.ErrInvalidMatching)
}

// TestBlossomContractionAdmission rejects malformed cycle and forest metadata.
// Every subtest begins with a fresh engine; allocation or contraction failures
// may consume internal state, so these helpers are not asserted to be transactional.
func TestBlossomContractionAdmission(t *testing.T) {
	for _, tc := range []struct {
		name        string
		base        int
		left, right []int
		edit        func(match.BlossomStateForTest)
		badEdge     bool
	}{
		{"invalid-base", -1, []int{0, 1, 2}, nil, nil, false},
		{"invalid-edge", 0, []int{0, 1, 2}, nil, nil, true},
		{"capacity", 0, []int{0, 1, 2}, nil, func(s match.BlossomStateForTest) { *s.NextNode = len(s.Active) }, false},
		{"even-cycle", 0, []int{0, 1}, nil, nil, false},
		{"inactive-child", 0, []int{0, 1, 2}, nil, func(s match.BlossomStateForTest) { s.Active[1] = false }, false},
		{"foreign-child", 0, []int{0, 1, -1}, nil, nil, false},
		{"missing-tree-edge", 0, []int{0, 1, 2}, nil, nil, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e, s := newInvariantEngine(t, 6)
			if tc.edit != nil {
				tc.edit(s)
			}
			edge := invariantEdgeID(t, e, 0, 2)
			if tc.badEdge {
				edge = match.NoEdgeForTest
			}
			node, err := match.AllocateBlossomNodeForTest(e, tc.base, tc.left, tc.right, edge)
			mustErrorIs(t, err, match.ErrInvalidMatching)
			if node != match.NoNodeForTest {
				t.Fatalf("invalid contraction published node %d", node)
			}
		})
	}
	for _, tc := range []struct {
		name string
		edit func(match.BlossomStateForTest, int)
	}{
		{"no-root", func(s match.BlossomStateForTest, node int) { s.TreeRoot[node] = match.NoNodeForTest }},
		{"inactive-root", func(s match.BlossomStateForTest, node int) { s.TreeRoot[node] = 1 }},
		{"broken-cycle", func(s match.BlossomStateForTest, node int) { s.Cycles[node] = nil }},
		{"invalid-child", func(s match.BlossomStateForTest, node int) {
			s.Cycles[node][0] = match.NewBlossomCycleStepForTest(-1, 0, 0, 1)
		}},
		{"self-child", func(s match.BlossomStateForTest, node int) {
			s.Cycles[node][0] = match.NewBlossomCycleStepForTest(node, 0, 0, 1)
		}},
		{"invalid-parent", func(s match.BlossomStateForTest, _ int) { s.Parent[3] = len(s.Active) }},
		{"missing-parent-edge", func(s match.BlossomStateForTest, _ int) { s.Parent[3] = 1; s.LabelEdge[3] = match.NoEdgeForTest }},
		{"invalid-descendant-root", func(s match.BlossomStateForTest, _ int) { s.TreeRoot[3] = len(s.Active) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e, s, node := contractedTriangle(t)
			tc.edit(s, node)
			mustErrorIs(t, match.RedirectForestThroughContractedBlossomForTest(e, node), match.ErrInvalidMatching)
		})
	}
}

// TestBlossomInnerExpansionAdmission isolates the prerequisites for expansion.
// A zero-dual inner blossom needs an active outer parent, a real entry edge,
// and a valid base child before the engine may restore the alternating forest.
func TestBlossomInnerExpansionAdmission(t *testing.T) {
	for _, tc := range []struct {
		name string
		edit func(match.BlossomStateForTest, int)
	}{
		{"inactive", func(s match.BlossomStateForTest, node int) { s.Active[node] = false }},
		{"outer-label", func(s match.BlossomStateForTest, node int) { s.Label[node] = match.OuterForTest }},
		{"positive-dual", func(s match.BlossomStateForTest, node int) { s.Dual[node] = 1 }},
		{"missing-cycle", func(s match.BlossomStateForTest, node int) { s.Cycles[node] = nil }},
		{"missing-root", func(s match.BlossomStateForTest, node int) { s.TreeRoot[node] = match.NoNodeForTest }},
		{"inactive-parent", func(s match.BlossomStateForTest, node int) { s.Parent[node] = 1 }},
		{"inner-parent", func(s match.BlossomStateForTest, _ int) { s.Label[3] = match.InnerForTest }},
		{"invalid-edge", func(s match.BlossomStateForTest, node int) { s.LabelEdge[node] = len(s.Mate) * len(s.Mate) }},
		{"invalid-base", func(s match.BlossomStateForTest, node int) { s.Base[node] = match.NoVertexForTest }},
		{"foreign-base", func(s match.BlossomStateForTest, node int) { s.Base[node] = 5 }},
		{"unlabeled-continuation", func(s match.BlossomStateForTest, node int) { s.Parent[4] = node }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e, s, node := contractedTriangle(t)
			// Attach the contracted triangle to a distinct exposed outer root.
			match.AssignOuterRootForTest(e, 3)
			s.Label[node] = match.InnerForTest
			s.Dual[node] = 0
			s.TreeRoot[node] = 3
			s.Parent[node] = 3
			s.LabelEdge[node] = invariantEdgeID(t, e, 0, 3)
			tc.edit(s, node)
			mustErrorIs(t, match.ExpandInnerBlossomInForestForTest(e, node), match.ErrInvalidMatching)
		})
	}
	e, s, node := contractedTriangle(t)
	// Ordinary outer expansion is a no-op; positive dual prevents inner release.
	requireInvariantOK(t, match.ExpandForTest(e, node))
	if !s.Active[node] {
		t.Fatal("outer blossom was released")
	}
	s.Active[node] = false
	mustErrorIs(t, match.ExpandForTest(e, node), match.ErrInvalidMatching)
	s.Active[node] = true
	s.Cycles[node] = nil
	mustErrorIs(t, match.ExpandForTest(e, node), match.ErrInvalidMatching)
}
