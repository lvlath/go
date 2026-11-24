// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package match_test checks the engine's local matching, forest, cycle and dual invariants.
// The test-only bridge supplies fixtures; every assertion remains external.
package match_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/match"
)

// TestBlossomShrinkRootCycleMakesContractedNodeForestRoot checks root ownership after contracting a genuine alternating triangle.
// A root blossom must not retain a parent edge or a stale tree-root ID.
func TestBlossomShrinkRootCycleMakesContractedNodeForestRoot(t *testing.T) {
	problem := invariantSeededProblem(4, 2)

	engine, err := match.NewBlossomEngineForTest(problem, match.DefaultEpsilon)
	if err != nil {
		t.Fatalf("newBlossomEngine: %v", err)
	}
	state := match.InspectBlossomStateForTest(engine)

	matched := invariantEdgeID(t, engine, 1, 2)
	if err = match.SetMatchedEdgeForTest(engine, matched); err != nil {
		t.Fatalf("seed matched edge: %v", err)
	}

	match.ResetForestForTest(engine)
	match.AssignOuterRootForTest(engine, 0)
	match.AssignOuterRootForTest(engine, 3)

	growEvent := match.NewBlossomEventForTest(match.EventGrowForTest, invariantEdgeID(t, engine, 3, 2), 3, 2, 3, 2)
	if err = match.GrowForTest(engine, growEvent); err != nil {
		t.Fatalf("grow: %v", err)
	}

	newNode := (*state.NextNode)

	shrinkEvent := match.NewBlossomEventForTest(match.EventShrinkForTest, invariantEdgeID(t, engine, 3, 1), 3, 1, 3, 1)
	if err = match.ShrinkForTest(engine, shrinkEvent); err != nil {
		t.Fatalf("shrink: %v", err)
	}

	if state.Parent[newNode] != match.NoNodeForTest {
		t.Fatalf("parent[%d]=%d, want noNode", newNode, state.Parent[newNode])
	}
	if state.LabelEdge[newNode] != match.NoEdgeForTest {
		t.Fatalf("labelEdge[%d]=%d, want noEdge", newNode, state.LabelEdge[newNode])
	}
	if state.TreeRoot[newNode] != newNode {
		t.Fatalf("treeRoot[%d]=%d, want %d", newNode, state.TreeRoot[newNode], newNode)
	}

	path, err := match.PathToRootForTest(engine, newNode)
	if err != nil {
		t.Fatalf("pathToRoot(new blossom): %v", err)
	}
	if len(path) != 0 {
		t.Fatalf("root blossom path length got %d want 0: %+v", len(path), path)
	}
}

// TestFindBlossomBaseAndCycleSeparatesBaseVertexAndBaseNode checks that an original base vertex is not confused with its allocated node ID.
// The forest arms must terminate at the node, not the vertex coordinate.
func TestFindBlossomBaseAndCycleSeparatesBaseVertexAndBaseNode(t *testing.T) {
	problem := invariantSeededProblem(6, 6100)

	engine, err := match.NewBlossomEngineForTest(problem, match.DefaultEpsilon)
	if err != nil {
		t.Fatalf("newBlossomEngine: %v", err)
	}
	state := match.InspectBlossomStateForTest(engine)

	baseVertex := 3
	baseNode := (*state.NextNode)
	(*state.NextNode)++

	state.Base[baseNode] = baseVertex
	state.Members[baseNode] = []int{baseVertex}
	state.Active[baseNode] = true
	state.Active[baseVertex] = false
	state.InBlossom[baseVertex] = baseNode

	state.Label[baseNode] = match.OuterForTest
	state.Parent[baseNode] = match.NoNodeForTest
	state.LabelEdge[baseNode] = match.NoEdgeForTest
	state.TreeRoot[baseNode] = baseNode

	left := 0
	right := 1

	state.Label[left] = match.OuterForTest
	state.Parent[left] = baseNode
	state.LabelEdge[left] = invariantEdgeID(t, engine, left, baseVertex)
	state.TreeRoot[left] = baseNode

	state.Label[right] = match.OuterForTest
	state.Parent[right] = baseNode
	state.LabelEdge[right] = invariantEdgeID(t, engine, right, baseVertex)
	state.TreeRoot[right] = baseNode

	gotBaseVertex, gotBaseNode, pathA, pathB, err := match.FindBlossomBaseAndCycleForTest(engine, left, right)
	if err != nil {
		t.Fatalf("findBlossomBaseAndCycle: %v", err)
	}
	if gotBaseVertex != baseVertex {
		t.Fatalf("base vertex got %d want %d", gotBaseVertex, baseVertex)
	}
	if gotBaseNode != baseNode {
		t.Fatalf("base node got %d want %d", gotBaseNode, baseNode)
	}
	if len(pathA) == 0 || pathA[len(pathA)-1] != baseNode {
		t.Fatalf("pathA does not end at base node: %v", pathA)
	}
	if len(pathB) == 0 {
		t.Fatalf("pathB must contain right arm before base")
	}
}

// TestBlossomDualOuterBlossomPreservesInternalSlack checks the dual compensation law for internal blossom edges.
// Shifting vertex potentials and the blossom dual must preserve internal slack.
func TestBlossomDualOuterBlossomPreservesInternalSlack(t *testing.T) {
	problem := invariantSeededProblem(4, 2)

	engine, err := match.NewBlossomEngineForTest(problem, match.DefaultEpsilon)
	if err != nil {
		t.Fatalf("newBlossomEngine: %v", err)
	}
	state := match.InspectBlossomStateForTest(engine)

	matched := invariantEdgeID(t, engine, 1, 2)
	if err = match.SetMatchedEdgeForTest(engine, matched); err != nil {
		t.Fatalf("seed matched edge: %v", err)
	}

	match.ResetForestForTest(engine)
	match.AssignOuterRootForTest(engine, 0)
	match.AssignOuterRootForTest(engine, 3)

	growEvent := match.NewBlossomEventForTest(match.EventGrowForTest, invariantEdgeID(t, engine, 3, 2), 3, 2, 3, 2)
	if err = match.GrowForTest(engine, growEvent); err != nil {
		t.Fatalf("grow: %v", err)
	}

	newNode := (*state.NextNode)

	shrinkEvent := match.NewBlossomEventForTest(match.EventShrinkForTest, invariantEdgeID(t, engine, 3, 1), 3, 1, 3, 1)
	if err = match.ShrinkForTest(engine, shrinkEvent); err != nil {
		t.Fatalf("shrink: %v", err)
	}

	steps, err := match.CycleStepsForTest(engine, newNode)
	if err != nil {
		t.Fatalf("cycleSteps: %v", err)
	}

	internalEdge := steps[0].EdgeToNextForTest()
	before := match.SlackForTest(engine, internalEdge)

	if err = match.ApplyDeltaForTest(engine, match.NewBlossomDeltaForTest(match.DeltaJoinForTest, 3, match.NoEdgeForTest, match.NoNodeForTest)); err != nil {
		t.Fatalf("applyDelta: %v", err)
	}

	after := match.SlackForTest(engine, internalEdge)
	if math.Abs(after-before) > match.DefaultEpsilon {
		t.Fatalf("internal slack changed: before %.12f after %.12f", before, after)
	}
}

// TestBlossomDualInnerExpansionUsesHalfBlossomDual checks the half-dual expansion threshold.
// An inner blossom with dual 10 becomes eligible at movement 5.
func TestBlossomDualInnerExpansionUsesHalfBlossomDual(t *testing.T) {
	problem := invariantSeededProblem(4, 2)

	engine, err := match.NewBlossomEngineForTest(problem, match.DefaultEpsilon)
	if err != nil {
		t.Fatalf("newBlossomEngine: %v", err)
	}
	state := match.InspectBlossomStateForTest(engine)

	node := (*state.NextNode)
	(*state.NextNode)++

	state.Base[node] = 0
	state.Members[node] = []int{0, 1, 2}
	state.Cycles[node] = []match.BlossomCycleStepForTest{
		match.NewBlossomCycleStepForTest(0, invariantEdgeID(t, engine, 0, 1), 0, 1),
		match.NewBlossomCycleStepForTest(1, invariantEdgeID(t, engine, 1, 2), 1, 2),
		match.NewBlossomCycleStepForTest(2, invariantEdgeID(t, engine, 2, 0), 2, 0),
	}
	state.Active[node] = true
	state.Label[node] = match.InnerForTest
	state.Dual[node] = 10

	best := match.NewBlossomDeltaForTest(match.DeltaNoneForTest, math.Inf(1), match.NoEdgeForTest, match.NoNodeForTest)

	if err = match.ConsiderExpandDeltasForTest(engine, &best); err != nil {
		t.Fatalf("considerExpandDeltas: %v", err)
	}

	if best.KindForTest() != match.DeltaExpandForTest {
		t.Fatalf("kind got %v want deltaExpandInnerBlossom", best.KindForTest())
	}
	if math.Abs(best.ValueForTest()-5) > match.DefaultEpsilon {
		t.Fatalf("delta got %.12f want 5", best.ValueForTest())
	}
}

// TestBlossomExpansionDeltaIsAppliedImmediately checks expansion following an exact dual-to-zero transition.
// Successful expansion releases the contracted metadata and deactivates its node.
func TestBlossomExpansionDeltaIsAppliedImmediately(t *testing.T) {
	problem := invariantSeededProblem(4, 2)

	engine, err := match.NewBlossomEngineForTest(problem, match.DefaultEpsilon)
	if err != nil {
		t.Fatalf("newBlossomEngine: %v", err)
	}
	state := match.InspectBlossomStateForTest(engine)

	node := (*state.NextNode)
	(*state.NextNode)++

	state.Base[node] = 0
	state.Members[node] = []int{0, 1, 2}
	state.Cycles[node] = []match.BlossomCycleStepForTest{
		match.NewBlossomCycleStepForTest(0, invariantEdgeID(t, engine, 0, 1), 0, 1),
		match.NewBlossomCycleStepForTest(1, invariantEdgeID(t, engine, 1, 2), 1, 2),
		match.NewBlossomCycleStepForTest(2, invariantEdgeID(t, engine, 2, 0), 2, 0),
	}

	// Parent edge enters base child directly in this smoke test.
	parent := 3
	parentEdge := invariantEdgeID(t, engine, parent, 0)

	state.Active[parent] = true
	state.Label[parent] = match.OuterForTest
	state.TreeRoot[parent] = parent

	state.Active[node] = true
	state.Label[node] = match.InnerForTest
	state.TreeRoot[node] = parent
	state.Parent[node] = parent
	state.LabelEdge[node] = parentEdge
	state.Dual[node] = 2

	for _, step := range state.Cycles[node] {
		state.Active[step.NodeForTest()] = false
		for _, vertex := range state.Members[step.NodeForTest()] {
			state.InBlossom[vertex] = node
		}
	}

	delta := match.NewBlossomDeltaForTest(match.DeltaExpandForTest, 1, match.NoEdgeForTest, node)

	if err = match.ApplyDeltaForTest(engine, delta); err != nil {
		t.Fatalf("applyDelta: %v", err)
	}
	if err = match.ExpandForTest(engine, delta.NodeForTest()); err != nil {
		t.Fatalf("expand: %v", err)
	}

	if state.Active[node] {
		t.Fatalf("expanded blossom node remains active")
	}
	if len(state.Cycles[node]) != 0 {
		t.Fatalf("expanded blossom cycle metadata not cleared")
	}
	if state.Dual[node] != 0 {
		t.Fatalf("expanded blossom dual got %.12f want 0", state.Dual[node])
	}
}

// TestBlossomExpandInnerUsesEntryChildForParentEdge checks endpoint-aware inner expansion.
// The incoming forest edge must attach to the child owning its original endpoint.
func TestBlossomExpandInnerUsesEntryChildForParentEdge(t *testing.T) {
	problem := invariantSeededProblem(6, 23)

	engine, err := match.NewBlossomEngineForTest(problem, match.DefaultEpsilon)
	if err != nil {
		t.Fatalf("newBlossomEngine: %v", err)
	}
	state := match.InspectBlossomStateForTest(engine)

	node := (*state.NextNode)
	(*state.NextNode)++

	state.Base[node] = 0
	state.Members[node] = []int{0, 1, 2}
	state.Cycles[node] = []match.BlossomCycleStepForTest{
		match.NewBlossomCycleStepForTest(0, invariantEdgeID(t, engine, 0, 1), 0, 1),
		match.NewBlossomCycleStepForTest(1, invariantEdgeID(t, engine, 1, 2), 1, 2),
		match.NewBlossomCycleStepForTest(2, invariantEdgeID(t, engine, 2, 0), 2, 0),
	}

	// The valid entry->base expansion path is 2 -> 1 -> 0.
	// Therefore 2-1 must be matched and 1-0 must be unmatched.
	if err = match.SetMatchedEdgeForTest(engine, invariantEdgeID(t, engine, 1, 2)); err != nil {
		t.Fatalf("set internal matched edge 1-2: %v", err)
	}

	for _, step := range state.Cycles[node] {
		state.Active[step.NodeForTest()] = false
		for _, vertex := range state.Members[step.NodeForTest()] {
			state.InBlossom[vertex] = node
		}
	}

	parent := 3
	parentEdge := invariantEdgeID(t, engine, parent, 2)

	state.Active[parent] = true
	state.Label[parent] = match.OuterForTest
	state.TreeRoot[parent] = parent

	state.Active[node] = true
	state.Label[node] = match.InnerForTest
	state.Parent[node] = parent
	state.LabelEdge[node] = parentEdge
	state.TreeRoot[node] = parent
	state.Dual[node] = 0

	if err = match.ExpandInnerBlossomInForestForTest(engine, node); err != nil {
		t.Fatalf("expandInnerBlossomInForest: %v", err)
	}

	entryChild := 2
	if state.Parent[entryChild] != parent {
		t.Fatalf("entry child parent got %d want %d", state.Parent[entryChild], parent)
	}
	if state.LabelEdge[entryChild] != parentEdge {
		t.Fatalf("entry child labelEdge got %d want %d", state.LabelEdge[entryChild], parentEdge)
	}
	if state.Label[entryChild] != match.InnerForTest {
		t.Fatalf("entry child label got %v want blossomInner", state.Label[entryChild])
	}
}

// TestBlossomExpandInnerReparentsOuterContinuationChild checks the outgoing matched continuation during expansion.
// The outer continuation must attach to the base child, retaining its edge and root.
func TestBlossomExpandInnerReparentsOuterContinuationChild(t *testing.T) {
	problem := invariantSeededProblem(6, 23)

	engine, err := match.NewBlossomEngineForTest(problem, match.DefaultEpsilon)
	if err != nil {
		t.Fatalf("newBlossomEngine: %v", err)
	}
	state := match.InspectBlossomStateForTest(engine)

	node := (*state.NextNode)
	(*state.NextNode)++

	state.Base[node] = 0
	state.Members[node] = []int{0, 1, 2}
	state.Cycles[node] = []match.BlossomCycleStepForTest{
		match.NewBlossomCycleStepForTest(0, invariantEdgeID(t, engine, 0, 1), 0, 1),
		match.NewBlossomCycleStepForTest(1, invariantEdgeID(t, engine, 1, 2), 1, 2),
		match.NewBlossomCycleStepForTest(2, invariantEdgeID(t, engine, 2, 0), 2, 0),
	}

	// Expansion path 2 -> 1 -> 0 needs 2-1 matched.
	if err = match.SetMatchedEdgeForTest(engine, invariantEdgeID(t, engine, 1, 2)); err != nil {
		t.Fatalf("set internal matched edge 1-2: %v", err)
	}

	for _, step := range state.Cycles[node] {
		state.Active[step.NodeForTest()] = false
		for _, vertex := range state.Members[step.NodeForTest()] {
			state.InBlossom[vertex] = node
		}
	}

	parent := 3
	outerChild := 4
	parentEdge := invariantEdgeID(t, engine, parent, 2)
	outerEdge := invariantEdgeID(t, engine, 0, outerChild)

	if err = match.SetMatchedEdgeForTest(engine, outerEdge); err != nil {
		t.Fatalf("setMatchedEdge outerEdge: %v", err)
	}

	state.Active[parent] = true
	state.Label[parent] = match.OuterForTest
	state.TreeRoot[parent] = parent

	state.Active[node] = true
	state.Label[node] = match.InnerForTest
	state.Parent[node] = parent
	state.LabelEdge[node] = parentEdge
	state.TreeRoot[node] = parent
	state.Dual[node] = 0

	state.Active[outerChild] = true
	state.Label[outerChild] = match.OuterForTest
	state.Parent[outerChild] = node
	state.LabelEdge[outerChild] = outerEdge
	state.TreeRoot[outerChild] = parent

	if err = match.ExpandInnerBlossomInForestForTest(engine, node); err != nil {
		t.Fatalf("expandInnerBlossomInForest: %v", err)
	}

	baseChild := 0
	if state.Parent[outerChild] != baseChild {
		t.Fatalf("outer child parent got %d want base child %d", state.Parent[outerChild], baseChild)
	}
	if state.LabelEdge[outerChild] != outerEdge {
		t.Fatalf("outer child labelEdge got %d want %d", state.LabelEdge[outerChild], outerEdge)
	}
	if state.TreeRoot[outerChild] != parent {
		t.Fatalf("outer child treeRoot got %d want %d", state.TreeRoot[outerChild], parent)
	}
}

// TestBlossomRefreshAllocatedBasesUsesCommittedMatching checks base recovery from the committed matching.
// The old base cannot survive when a different original vertex becomes exposed.
func TestBlossomRefreshAllocatedBasesUsesCommittedMatching(t *testing.T) {
	problem := invariantSeededProblem(6, 23)

	engine, err := match.NewBlossomEngineForTest(problem, match.DefaultEpsilon)
	if err != nil {
		t.Fatalf("newBlossomEngine: %v", err)
	}
	state := match.InspectBlossomStateForTest(engine)

	node := (*state.NextNode)
	(*state.NextNode)++

	state.Base[node] = 0
	state.Members[node] = []int{0, 1, 2}
	state.Cycles[node] = []match.BlossomCycleStepForTest{
		match.NewBlossomCycleStepForTest(0, invariantEdgeID(t, engine, 0, 1), 0, 1),
		match.NewBlossomCycleStepForTest(1, invariantEdgeID(t, engine, 1, 2), 1, 2),
		match.NewBlossomCycleStepForTest(2, invariantEdgeID(t, engine, 2, 0), 2, 0),
	}

	state.Active[node] = true

	for _, step := range state.Cycles[node] {
		state.Active[step.NodeForTest()] = false
		for _, vertex := range state.Members[step.NodeForTest()] {
			state.InBlossom[vertex] = node
		}
	}

	if err = match.SetMatchedEdgeForTest(engine, invariantEdgeID(t, engine, 0, 1)); err != nil {
		t.Fatalf("set internal matched edge: %v", err)
	}

	if err = match.RefreshAllocatedBlossomBasesForTest(engine); err != nil {
		t.Fatalf("refreshAllocatedBlossomBases: %v", err)
	}

	if state.Base[node] != 2 {
		t.Fatalf("base got %d want 2", state.Base[node])
	}
}

// TestBlossomShrinkRedirectsForestDescendantsFromContractedChildren checks redirection of live forest descendants after contraction.
// Parent and tree-root references must target the active contracted node.
func TestBlossomShrinkRedirectsForestDescendantsFromContractedChildren(t *testing.T) {
	problem := invariantSeededProblem(6, 23)

	engine, err := match.NewBlossomEngineForTest(problem, match.DefaultEpsilon)
	if err != nil {
		t.Fatalf("newBlossomEngine: %v", err)
	}
	state := match.InspectBlossomStateForTest(engine)

	newNode := (*state.NextNode)
	(*state.NextNode)++

	state.Base[newNode] = 3
	state.Members[newNode] = []int{3, 4, 5}
	state.Cycles[newNode] = []match.BlossomCycleStepForTest{
		match.NewBlossomCycleStepForTest(3, invariantEdgeID(t, engine, 3, 5), 3, 5),
		match.NewBlossomCycleStepForTest(5, invariantEdgeID(t, engine, 5, 4), 5, 4),
		match.NewBlossomCycleStepForTest(4, invariantEdgeID(t, engine, 4, 3), 4, 3),
	}

	// Active descendants outside the soon-to-be-contracted blossom.
	state.Active[1] = true
	state.Label[1] = match.OuterForTest
	state.Parent[1] = 2
	state.LabelEdge[1] = invariantEdgeID(t, engine, 1, 2)
	state.TreeRoot[1] = 3

	state.Active[2] = true
	state.Label[2] = match.InnerForTest
	state.Parent[2] = 4
	state.LabelEdge[2] = invariantEdgeID(t, engine, 2, 4)
	state.TreeRoot[2] = 3

	// Old cycle children are active before contraction.
	for _, child := range []int{3, 4, 5} {
		state.Active[child] = true
	}

	if err = match.ContractCycleIntoBlossomForTest(engine, newNode); err != nil {
		t.Fatalf("contractCycleIntoBlossom: %v", err)
	}

	state.Label[newNode] = match.OuterForTest
	state.Parent[newNode] = match.NoNodeForTest
	state.LabelEdge[newNode] = match.NoEdgeForTest
	state.TreeRoot[newNode] = newNode

	if err = match.RedirectForestThroughContractedBlossomForTest(engine, newNode); err != nil {
		t.Fatalf("redirectForestThroughContractedBlossom: %v", err)
	}

	if state.Parent[2] != newNode {
		t.Fatalf("parent[2] got %d want contracted blossom %d", state.Parent[2], newNode)
	}
	if state.TreeRoot[1] != newNode {
		t.Fatalf("treeRoot[1] got %d want %d", state.TreeRoot[1], newNode)
	}
	if state.TreeRoot[2] != newNode {
		t.Fatalf("treeRoot[2] got %d want %d", state.TreeRoot[2], newNode)
	}

	if _, err = match.PathToRootForTest(engine, 1); err != nil {
		t.Fatalf("pathToRoot after redirect: %v", err)
	}
}

// TestBlossomApplyDeltaDoesNotClampPositiveSubToleranceMovement checks that positive representable dual motion is not discarded.
// Feasibility tolerance must not suppress a real search step.
func TestBlossomApplyDeltaDoesNotClampPositiveSubToleranceMovement(t *testing.T) {
	problem := invariantNearTieProblem(4, 80)

	engine, err := match.NewBlossomEngineForTest(problem, match.DefaultEpsilon)
	if err != nil {
		t.Fatalf("newBlossomEngine: %v", err)
	}
	state := match.InspectBlossomStateForTest(engine)

	state.Active[0] = true
	state.Label[0] = match.OuterForTest

	before := state.Dual[0]
	delta := match.DualToleranceForTest(engine) / 2

	if err = match.ApplyDeltaForTest(engine, match.NewBlossomDeltaForTest(match.DeltaGrowForTest, delta, 0, 0)); err != nil {
		t.Fatalf("applyDelta: %v", err)
	}

	if state.Dual[0] == before {
		t.Fatalf("positive sub-tolerance delta was clamped: before %.17g after %.17g delta %.17g tol %.17g",
			before, state.Dual[0], delta, match.DualToleranceForTest(engine))
	}
}

// TestBlossomImproveDeltaUsesNarrowSelectionTolerance checks the distinction between dual feasibility and event selection.
// A genuinely smaller movement must win even if the larger edge ID is less preferred.
func TestBlossomImproveDeltaUsesNarrowSelectionTolerance(t *testing.T) {
	problem := invariantNearTieProblem(4, 86)

	engine, err := match.NewBlossomEngineForTest(problem, match.DefaultEpsilon)
	if err != nil {
		t.Fatalf("newBlossomEngine: %v", err)
	}

	wide := match.DualToleranceForTest(engine) / 2
	narrow := match.DualSelectionToleranceForTest(engine)

	if !(wide > narrow) {
		t.Fatalf("test requires wide tolerance > narrow tolerance, wide=%g narrow=%g", wide, narrow)
	}

	best := match.NewBlossomDeltaForTest(match.DeltaGrowForTest, wide, 1, 1)

	better := match.NewBlossomDeltaForTest(match.DeltaGrowForTest, narrow, 99, 2)

	match.ImproveDeltaForTest(engine, &best, better)

	if best.EdgeForTest() != better.EdgeForTest() {
		t.Fatalf("improveDelta kept larger candidate: got edge=%d value=%.17g want edge=%d value=%.17g wideTol=%.17g narrowTol=%.17g",
			best.EdgeForTest(), best.ValueForTest(), better.EdgeForTest(), better.ValueForTest(), wide, narrow)
	}
}

// TestBlossomSetAndClearMatchedEdgeMaintainSymmetry checks reciprocal mate and mateEdge updates.
// Both endpoint records must agree after insertion and clearing.
func TestBlossomSetAndClearMatchedEdgeMaintainSymmetry(t *testing.T) {
	problem := invariantSeededProblem(4, 20)

	engine, err := match.NewBlossomEngineForTest(problem, match.DefaultEpsilon)
	if err != nil {
		t.Fatalf("newBlossomEngine: %v", err)
	}
	state := match.InspectBlossomStateForTest(engine)

	edgeID := invariantEdgeID(t, engine, 0, 1)

	if err := match.SetMatchedEdgeForTest(engine, edgeID); err != nil {
		t.Fatalf("setMatchedEdge: %v", err)
	}
	if state.Mate[0] != 1 || state.Mate[1] != 0 {
		t.Fatalf("mate symmetry broken after set: mate=%v", state.Mate)
	}
	if state.MateEdge[0] != edgeID || state.MateEdge[1] != edgeID {
		t.Fatalf("mateEdge symmetry broken after set: mateEdge=%v", state.MateEdge)
	}

	if err := match.ClearMatchedEdgeForTest(engine, edgeID); err != nil {
		t.Fatalf("clearMatchedEdge: %v", err)
	}
	if state.Mate[0] != match.NoVertexForTest || state.Mate[1] != match.NoVertexForTest {
		t.Fatalf("mate not cleared: mate=%v", state.Mate)
	}
	if state.MateEdge[0] != match.NoEdgeForTest || state.MateEdge[1] != match.NoEdgeForTest {
		t.Fatalf("mateEdge not cleared: mateEdge=%v", state.MateEdge)
	}
}

// TestBlossomFlipRejectsNonAlternatingPathBeforeMutation checks invalid-path rejection before a matching transaction.
// Both mate arrays must retain their independent before-images.
func TestBlossomFlipRejectsNonAlternatingPathBeforeMutation(t *testing.T) {
	problem := invariantSeededProblem(4, 21)

	engine, err := match.NewBlossomEngineForTest(problem, match.DefaultEpsilon)
	if err != nil {
		t.Fatalf("newBlossomEngine: %v", err)
	}
	state := match.InspectBlossomStateForTest(engine)

	first := invariantEdgeID(t, engine, 0, 1)
	second := invariantEdgeID(t, engine, 2, 3)

	beforeMate := append([]int(nil), state.Mate...)
	beforeMateEdge := append([]int(nil), state.MateEdge...)

	if err = match.FlipAugmentingPathForTest(engine, []int{first, second}); !errors.Is(err, match.ErrInvalidMatching) {
		t.Fatalf("non-alternating path got %v", err)
	}

	for vertex := range state.Mate {
		if state.Mate[vertex] != beforeMate[vertex] {
			t.Fatalf("mate mutated at %d: got %d want %d", vertex, state.Mate[vertex], beforeMate[vertex])
		}
		if state.MateEdge[vertex] != beforeMateEdge[vertex] {
			t.Fatalf("mateEdge mutated at %d: got %d want %d", vertex, state.MateEdge[vertex], beforeMateEdge[vertex])
		}
	}
}

// TestBlossomFlipAugmentingPathMatchesThreeEdgePath checks the canonical unmatched/matched/unmatched augmentation.
// The middle pair is removed and both boundary pairs are committed.
func TestBlossomFlipAugmentingPathMatchesThreeEdgePath(t *testing.T) {
	problem := invariantSeededProblem(4, 22)

	engine, err := match.NewBlossomEngineForTest(problem, match.DefaultEpsilon)
	if err != nil {
		t.Fatalf("newBlossomEngine: %v", err)
	}
	state := match.InspectBlossomStateForTest(engine)

	left := invariantEdgeID(t, engine, 0, 1)
	middle := invariantEdgeID(t, engine, 1, 2)
	right := invariantEdgeID(t, engine, 2, 3)

	if err := match.SetMatchedEdgeForTest(engine, middle); err != nil {
		t.Fatalf("seed middle match: %v", err)
	}

	if err := match.FlipAugmentingPathForTest(engine, []int{left, middle, right}); err != nil {
		t.Fatalf("flipAugmentingPath: %v", err)
	}

	if state.Mate[0] != 1 || state.Mate[1] != 0 {
		t.Fatalf("left edge not matched after flip: mate=%v", state.Mate)
	}
	if state.Mate[2] != 3 || state.Mate[3] != 2 {
		t.Fatalf("right edge not matched after flip: mate=%v", state.Mate)
	}
}

// TestBlossomComposeAugmentingPathUsesRootToRootOrientation checks orientation of two forest arms joined by one bridge.
// Local node IDs are asserted independently of edge weight.
func TestBlossomComposeAugmentingPathUsesRootToRootOrientation(t *testing.T) {
	problem := invariantSeededProblem(6, 23)

	engine, err := match.NewBlossomEngineForTest(problem, match.DefaultEpsilon)
	if err != nil {
		t.Fatalf("newBlossomEngine: %v", err)
	}

	left := []match.BlossomPathStepForTest{
		match.NewBlossomPathStepForTest(invariantEdgeID(t, engine, 3, 1), 3, 1, 3, 1),
		match.NewBlossomPathStepForTest(invariantEdgeID(t, engine, 1, 0), 1, 0, 1, 0),
	}
	right := []match.BlossomPathStepForTest{
		match.NewBlossomPathStepForTest(invariantEdgeID(t, engine, 4, 2), 4, 2, 4, 2),
		match.NewBlossomPathStepForTest(invariantEdgeID(t, engine, 2, 5), 2, 5, 2, 5),
	}
	bridge := match.NewBlossomPathStepForTest(invariantEdgeID(t, engine, 3, 4), 3, 4, 3, 4)

	path, err := match.ComposeAugmentingPathForTest(engine, left, bridge, right)
	if err != nil {
		t.Fatalf("composeAugmentingPath: %v", err)
	}

	wantFrom := []int{0, 1, 3, 4, 2}
	wantTo := []int{1, 3, 4, 2, 5}

	if len(path) != len(wantFrom) {
		t.Fatalf("path length got %d want %d: %+v", len(path), len(wantFrom), path)
	}

	for index := range path {
		if path[index].FromNodeForTest() != wantFrom[index] || path[index].ToNodeForTest() != wantTo[index] {
			t.Fatalf("step %d got %d->%d want %d->%d",
				index,
				path[index].FromNodeForTest(),
				path[index].ToNodeForTest(),
				wantFrom[index],
				wantTo[index],
			)
		}
	}
}

// TestBlossomShrinkStoresCycleAsSingleSourceOfTruth checks child order and boundary endpoint ownership after contraction.
// Every cycle step must connect the current child to its next child.
func TestBlossomShrinkStoresCycleAsSingleSourceOfTruth(t *testing.T) {
	problem := invariantSeededProblem(4, 10)

	engine, err := match.NewBlossomEngineForTest(problem, match.DefaultEpsilon)
	if err != nil {
		t.Fatalf("newBlossomEngine: %v", err)
	}
	state := match.InspectBlossomStateForTest(engine)

	root := 0
	left := 1
	right := 2

	state.Label[root] = match.OuterForTest
	state.TreeRoot[root] = root
	state.Parent[root] = match.NoNodeForTest
	state.LabelEdge[root] = match.NoEdgeForTest

	state.Label[left] = match.OuterForTest
	state.TreeRoot[left] = root
	state.Parent[left] = root
	state.LabelEdge[left] = invariantEdgeID(t, engine, left, root)

	state.Label[right] = match.OuterForTest
	state.TreeRoot[right] = root
	state.Parent[right] = root
	state.LabelEdge[right] = invariantEdgeID(t, engine, right, root)

	event := match.NewBlossomEventForTest(match.EventShrinkForTest, invariantEdgeID(t, engine, left, right), left, right, left, right)

	newNode := (*state.NextNode)

	if err := match.ShrinkForTest(engine, event); err != nil {
		t.Fatalf("shrink: %v", err)
	}

	steps, err := match.CycleStepsForTest(engine, newNode)
	if err != nil {
		t.Fatalf("cycleSteps: %v", err)
	}

	if len(steps) != 3 {
		t.Fatalf("cycle length got %d want 3", len(steps))
	}

	for index, step := range steps {
		if step.NodeForTest() == match.NoNodeForTest {
			t.Fatalf("step[%d].node=noNode", index)
		}
		if step.EdgeToNextForTest() == match.NoEdgeForTest {
			t.Fatalf("step[%d].edgeToNext=noEdge", index)
		}
		if !match.NodeContainsVertexForTest(engine, step.NodeForTest(), step.VertexToNextForTest()) {
			t.Fatalf("step[%d].vertexToNext=%d not owned by node %d", index, step.VertexToNextForTest(), step.NodeForTest())
		}

		next := steps[(index+1)%len(steps)]
		if !match.NodeContainsVertexForTest(engine, next.NodeForTest(), step.NextVertexForTest()) {
			t.Fatalf("step[%d].nextVertex=%d not owned by next node %d", index, step.NextVertexForTest(), next.NodeForTest())
		}
	}
}

// TestBlossomCycleStepsRejectSingletonAndMalformedCycle checks rejection of singleton and even-length contracted cycle metadata.
// Such metadata cannot represent a nontrivial odd blossom.
func TestBlossomCycleStepsRejectSingletonAndMalformedCycle(t *testing.T) {
	problem := invariantSeededProblem(4, 11)

	engine, err := match.NewBlossomEngineForTest(problem, match.DefaultEpsilon)
	if err != nil {
		t.Fatalf("newBlossomEngine: %v", err)
	}
	state := match.InspectBlossomStateForTest(engine)

	if _, err = match.CycleStepsForTest(engine, 0); !errors.Is(err, match.ErrInvalidMatching) {
		t.Fatalf("singleton cycleSteps got %v", err)
	}

	node := (*state.NextNode)
	(*state.NextNode)++
	state.Cycles[node] = []match.BlossomCycleStepForTest{
		match.NewBlossomCycleStepForTest(0, invariantEdgeID(t, engine, 0, 1), 0, 1),
		match.NewBlossomCycleStepForTest(1, invariantEdgeID(t, engine, 1, 0), 1, 0),
	}

	if _, err = match.CycleStepsForTest(engine, node); !errors.Is(err, match.ErrInvalidMatching) {
		t.Fatalf("even malformed cycle got %v", err)
	}
}
