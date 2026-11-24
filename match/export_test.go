// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package match exposes a narrow white-box fixture boundary to match_test.
// This file is compiled only by go test. It contains no tests, assertions,
// solver substitutions or production exports.
// Function-valued fixture bindings retain the identity of private engine types.
// Call them directly; do not reassign them, including in parallel tests.
package match

import "slices"

// BlossomEngineForTest names the actual private engine in the test build.
// External tests can exercise it without creating a second implementation.
type BlossomEngineForTest = blossomEngine

// MatchingProblemForTest names the actual detached local problem in the test build.
// The fixture constructor deliberately permits malformed local shapes.
type MatchingProblemForTest = matchingProblem

// BlossomCycleStepForTest names the actual cycle metadata in the test build.
// Construct individual steps with NewBlossomCycleStepForTest.
type BlossomCycleStepForTest = blossomCycleStep

// BlossomEventForTest names the actual alternating-forest event in the test build.
// Event fields remain private; use NewBlossomEventForTest to construct a fixture.
type BlossomEventForTest = blossomEvent

// BlossomDeltaForTest names the actual dual movement in the test build.
// It preserves the engine's private representation and comparison policy.
type BlossomDeltaForTest = blossomDelta

// BlossomPathStepForTest names one actual oriented boundary step in the test build.
// The factory does not infer or repair endpoint ownership.
type BlossomPathStepForTest = blossomPathStep

// Fixture constants expose the engine's existing identity and policy values.
// They do not establish a new enumeration or production configuration surface.
const (
	NoVertexForTest     = noVertex
	NoEdgeForTest       = noEdge
	NoNodeForTest       = noNode
	OuterForTest        = blossomOuter
	InnerForTest        = blossomInner
	EventNoneForTest    = eventNone
	EventGrowForTest    = eventGrow
	EventShrinkForTest  = eventShrink
	EventAugmentForTest = eventAugment
	EventExpandForTest  = eventExpand
	DeltaNoneForTest    = deltaNone
	DeltaGrowForTest    = deltaGrowToUnlabeled
	DeltaJoinForTest    = deltaJoinOuterTrees
	DeltaExpandForTest  = deltaExpandInnerBlossom
)

// BlossomStateForTest contains live views of fixture-owned engine storage.
// The views intentionally permit one test to inject a precise invalid invariant.
// Implementation:
//   - Stage 1: Construct a fresh engine through its real constructor.
//   - Stage 2: Inspect or edit only the fields required by the test scenario.
//   - Stage 3: Exercise real engine operations and assert externally.
//
// Behavior highlights:
//   - No validation, copying or solver logic occurs in these field views.
//
// Inputs:
//   - One fresh, exclusively owned test engine.
//
// Returns:
//   - Live slices and pointers to two scalar fields, not detached snapshots.
//
// Errors:
//   - None; malformed state is intentionally the responsibility of the test.
//
// Determinism:
//   - All views refer to the same engine instance and preserve its index order.
//
// Complexity:
//   - Time O(1), Space O(1) for acquiring the view.
//
// Notes:
//   - Never share the engine or a view between parallel tests. Clone slices
//     explicitly before a mutation if an independent before-image is needed.
//
// AI-Hints:
//   - Use this boundary for defensive invariant tests, not ordinary public API tests.
type BlossomStateForTest struct {
	Mate, MateEdge              []int                       // Reciprocal local partners and corresponding edge IDs.
	InBlossom, Base             []int                       // Top-level ownership and original base vertices.
	Active                      []bool                      // Active top-level node flags.
	Parent, LabelEdge, TreeRoot []int                       // Alternating-forest ancestry and witnesses.
	Label                       []blossomLabel              // Unlabeled, outer, or inner search state.
	Cycles                      [][]BlossomCycleStepForTest // Ordered child/boundary metadata.
	Members, Incident           [][]int                     // Original vertices per node and incident edges per vertex.
	Dual                        []float64                   // Current vertex and blossom dual variables.
	NextNode                    *int                        // Next available contracted-node ID.
	Scale                       *float64                    // Magnitude used by the dual tolerance policy.
}

// InspectBlossomStateForTest returns a live view of an exclusively owned engine.
// It exposes state identity so failed transactions can be checked without copying engine logic.
// Implementation:
//   - Stage 1: Bind the original slices and scalar addresses.
//
// Behavior highlights:
//   - Delegates to or exposes the original representation without a numerical rewrite.
//
// Inputs:
//   - e: non-nil engine constructed for one test.
//
// Returns:
//   - BlossomStateForTest: aliased state, not a snapshot.
//
// Errors:
//   - None from the bridge itself; any delegated errors retain their original identity.
//
// Determinism:
//   - Preserves caller order and the underlying engine's behavior.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Test build only. No production API or shared solver state is added.
//
// AI-Hints:
//   - Keep assertions in match_test and use one independently owned fixture per case.
func InspectBlossomStateForTest(e *BlossomEngineForTest) BlossomStateForTest {
	return BlossomStateForTest{
		Mate: e.mate, MateEdge: e.mateEdge, InBlossom: e.inBlossom, Base: e.base,
		Active: e.active, Parent: e.parent, LabelEdge: e.labelEdge, TreeRoot: e.treeRoot,
		Label: e.label, Cycles: e.cycles, Members: e.members, Incident: e.incident,
		Dual: e.dual, NextNode: &e.nextNode, Scale: &e.scale,
	}
}

// NewMatchingProblemForTest constructs a detached local problem with explicit shape.
// It intentionally does not pre-validate an input needed to exercise a private validator.
// Implementation:
//   - Stage 1: Copy the local-to-original map and row-major weights.
//
// Behavior highlights:
//   - Delegates to or exposes the original representation without a numerical rewrite.
//
// Inputs:
//   - n: declared order; vertices and weights: caller-owned slices, possibly malformed.
//
// Returns:
//   - MatchingProblemForTest: owned local data.
//
// Errors:
//   - None from the bridge itself; any delegated errors retain their original identity.
//
// Determinism:
//   - Preserves caller order and the underlying engine's behavior.
//
// Complexity:
//   - Time and Space O(len(vertices)+len(weights)).
//
// Notes:
//   - Test build only. No production API or shared solver state is added.
//
// AI-Hints:
//   - Keep assertions in match_test and use one independently owned fixture per case.
var NewMatchingProblemForTest = func(n int, vertices []int, weights []float64) MatchingProblemForTest {
	return matchingProblem{n: n, odd: slices.Clone(vertices), w: slices.Clone(weights)}
}

// InspectMatchingProblemForTest returns detached observations of a local matching problem.
// Tests can verify subset mapping without changing the constructed model.
// Implementation:
//   - Stage 1: Copy the map and flat weights for independent assertions.
//
// Behavior highlights:
//   - Delegates to or exposes the original representation without a numerical rewrite.
//
// Inputs:
//   - p: local problem returned by the real builder.
//
// Returns:
//   - Declared order, original indices and row-major weights.
//
// Errors:
//   - None from the bridge itself; any delegated errors retain their original identity.
//
// Determinism:
//   - Preserves caller order and the underlying engine's behavior.
//
// Complexity:
//   - Time and Space O(len(p.odd)+len(p.w)).
//
// Notes:
//   - Test build only. No production API or shared solver state is added.
//
// AI-Hints:
//   - Keep assertions in match_test and use one independently owned fixture per case.
func InspectMatchingProblemForTest(p MatchingProblemForTest) (int, []int, []float64) {
	return p.n, slices.Clone(p.odd), slices.Clone(p.w)
}

// NewBlossomEngineForTest calls the actual weighted Blossom constructor.
// It does not bypass admission or allocate an alternative numerical state.
// Implementation:
//   - Stage 1: Pass the explicit local problem and epsilon to the real constructor.
//
// Behavior highlights:
//   - Delegates to or exposes the original representation without a numerical rewrite.
//
// Inputs:
//   - p: local problem; eps: numeric policy.
//
// Returns:
//   - Engine pointer or the original constructor error.
//
// Errors:
//   - None from the bridge itself; any delegated errors retain their original identity.
//
// Determinism:
//   - Preserves caller order and the underlying engine's behavior.
//
// Complexity:
//   - Same as newBlossomEngine, including O(k^2) edge storage.
//
// Notes:
//   - Test build only. No production API or shared solver state is added.
//
// AI-Hints:
//   - Keep assertions in match_test and use one independently owned fixture per case.
var NewBlossomEngineForTest = func(p MatchingProblemForTest, eps float64) (*BlossomEngineForTest, error) {
	return newBlossomEngine(p, blossomOptions{Eps: eps})
}

// SolveBlossomProblemForTest runs the real private solver on an explicit local model.
// It permits direct testing of local-domain failures independently of public admission.
// Implementation:
//   - Stage 1: Delegate to the private kernel and omit only telemetry.
//
// Behavior highlights:
//   - Delegates to or exposes the original representation without a numerical rewrite.
//
// Inputs:
//   - p: local problem; eps: numerical policy.
//
// Returns:
//   - Local mates, raw cost and original error.
//
// Errors:
//   - None from the bridge itself; any delegated errors retain their original identity.
//
// Determinism:
//   - Preserves caller order and the underlying engine's behavior.
//
// Complexity:
//   - Same as solveMinimumWeightPerfectMatching.
//
// Notes:
//   - Test build only. No production API or shared solver state is added.
//
// AI-Hints:
//   - Keep assertions in match_test and use one independently owned fixture per case.
func SolveBlossomProblemForTest(p MatchingProblemForTest, eps float64) ([]int, float64, error) {
	mates, cost, _, err := solveMinimumWeightPerfectMatching(p, blossomOptions{Eps: eps})
	return mates, cost, err
}

// NewBlossomCycleStepForTest constructs a cycle metadata record verbatim.
// No inferred endpoints or repair hides malformed metadata.
// Implementation:
//   - Stage 1: Store explicit child and original boundary indices.
//
// Behavior highlights:
//   - Delegates to or exposes the original representation without a numerical rewrite.
//
// Inputs:
//   - node, edge, from, to: caller-supplied local IDs.
//
// Returns:
//   - A private cycle step value.
//
// Errors:
//   - None from the bridge itself; any delegated errors retain their original identity.
//
// Determinism:
//   - Preserves caller order and the underlying engine's behavior.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Test build only. No production API or shared solver state is added.
//
// AI-Hints:
//   - Keep assertions in match_test and use one independently owned fixture per case.
var NewBlossomCycleStepForTest = func(node, edge, from, to int) BlossomCycleStepForTest {
	return blossomCycleStep{node: node, edgeToNext: edge, vertexToNext: from, nextVertex: to}
}

// NewBlossomEventForTest constructs one explicit search event.
// Invalid events remain invalid and must be refused by the kernel.
// Implementation:
//   - Stage 1: Preserve the supplied kind and endpoint ownership fields.
//
// Behavior highlights:
//   - Delegates to or exposes the original representation without a numerical rewrite.
//
// Inputs:
//   - kind: event type; edge, a, b, av, bv: local IDs.
//
// Returns:
//   - The actual engine event.
//
// Errors:
//   - None from the bridge itself; any delegated errors retain their original identity.
//
// Determinism:
//   - Preserves caller order and the underlying engine's behavior.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Test build only. No production API or shared solver state is added.
//
// AI-Hints:
//   - Keep assertions in match_test and use one independently owned fixture per case.
var NewBlossomEventForTest = func(kind blossomEventKind, edge, a, b, av, bv int) BlossomEventForTest {
	return blossomEvent{kind: kind, edge: edge, a: a, b: b, aVertex: av, bVertex: bv}
}

// NewBlossomDeltaForTest constructs an explicit dual-step candidate.
// Negative and nonfinite values are preserved for defensive tests.
// Implementation:
//   - Stage 1: Store the value and tie-break IDs without normalization.
//
// Behavior highlights:
//   - Delegates to or exposes the original representation without a numerical rewrite.
//
// Inputs:
//   - kind: delta kind; value: movement; edge and node: event identifiers.
//
// Returns:
//   - The actual private delta value.
//
// Errors:
//   - None from the bridge itself; any delegated errors retain their original identity.
//
// Determinism:
//   - Preserves caller order and the underlying engine's behavior.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Test build only. No production API or shared solver state is added.
//
// AI-Hints:
//   - Keep assertions in match_test and use one independently owned fixture per case.
var NewBlossomDeltaForTest = func(kind blossomDeltaKind, value float64, edge, node int) BlossomDeltaForTest {
	return blossomDelta{kind: kind, value: value, edge: edge, node: node}
}

// NewBlossomPathStepForTest constructs one oriented path step without guessing orientation.
// This deliberately permits inconsistent endpoints for continuity tests.
// Implementation:
//   - Stage 1: Store both top-level and original endpoint indices.
//
// Behavior highlights:
//   - Delegates to or exposes the original representation without a numerical rewrite.
//
// Inputs:
//   - edge: dense edge ID; fromNode/toNode/fromVertex/toVertex: local IDs.
//
// Returns:
//   - Actual path metadata.
//
// Errors:
//   - None from the bridge itself; any delegated errors retain their original identity.
//
// Determinism:
//   - Preserves caller order and the underlying engine's behavior.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Test build only. No production API or shared solver state is added.
//
// AI-Hints:
//   - Keep assertions in match_test and use one independently owned fixture per case.
var NewBlossomPathStepForTest = func(edge, fromNode, toNode, fromVertex, toVertex int) BlossomPathStepForTest {
	return blossomPathStep{edge: edge, fromNode: fromNode, toNode: toNode, fromVertex: fromVertex, toVertex: toVertex}
}

// FindBlossomEdgeForTest locates an actual dense edge by its two local endpoints.
// Tests do not duplicate or assume the engine edge-index formula.
// Implementation:
//   - Stage 1: Scan stored immutable edge endpoints. Stage 2: Return the first matching edge ID.
//
// Behavior highlights:
//   - Delegates to or exposes the original representation without a numerical rewrite.
//
// Inputs:
//   - e: fresh engine; u and v: local vertices.
//
// Returns:
//   - Edge ID and presence flag.
//
// Errors:
//   - None from the bridge itself; any delegated errors retain their original identity.
//
// Determinism:
//   - Preserves caller order and the underlying engine's behavior.
//
// Complexity:
//   - Time O(k^2), Space O(1).
//
// Notes:
//   - Test build only. No production API or shared solver state is added.
//
// AI-Hints:
//   - Keep assertions in match_test and use one independently owned fixture per case.
func FindBlossomEdgeForTest(e *BlossomEngineForTest, u, v int) (int, bool) {
	for id, edge := range e.edges {
		if (edge.u == u && edge.v == v) || (edge.u == v && edge.v == u) {
			return id, true
		}
	}
	return noEdge, false
}

// NodeForTest reports the stored node field without modifying the fixture.
// This is a constant-time observation of actual engine metadata, compiled only for tests.
func (s blossomCycleStep) NodeForTest() int { return s.node }

// EdgeToNextForTest reports the stored edgeToNext field without modifying the fixture.
// This is a constant-time observation of actual engine metadata, compiled only for tests.
func (s blossomCycleStep) EdgeToNextForTest() int { return s.edgeToNext }

// VertexToNextForTest reports the stored vertexToNext field without modifying the fixture.
// This is a constant-time observation of actual engine metadata, compiled only for tests.
func (s blossomCycleStep) VertexToNextForTest() int { return s.vertexToNext }

// NextVertexForTest reports the stored nextVertex field without modifying the fixture.
// This is a constant-time observation of actual engine metadata, compiled only for tests.
func (s blossomCycleStep) NextVertexForTest() int { return s.nextVertex }

// FromNodeForTest reports the stored fromNode field without modifying the fixture.
// This is a constant-time observation of actual engine metadata, compiled only for tests.
func (s blossomPathStep) FromNodeForTest() int { return s.fromNode }

// ToNodeForTest reports the stored toNode field without modifying the fixture.
// This is a constant-time observation of actual engine metadata, compiled only for tests.
func (s blossomPathStep) ToNodeForTest() int { return s.toNode }

// EdgeForTest reports the stored edge field without modifying the fixture.
// This is a constant-time observation of actual engine metadata, compiled only for tests.
func (s blossomPathStep) EdgeForTest() int { return s.edge }

// KindForTest reports the stored kind field without modifying the fixture.
// This is a constant-time observation of actual engine metadata, compiled only for tests.
func (s blossomDelta) KindForTest() blossomDeltaKind { return s.kind }

// ValueForTest reports the stored value field without modifying the fixture.
// This is a constant-time observation of actual engine metadata, compiled only for tests.
func (s blossomDelta) ValueForTest() float64 { return s.value }

// EdgeForTest reports the stored edge field without modifying the fixture.
// This is a constant-time observation of actual engine metadata, compiled only for tests.
func (s blossomDelta) EdgeForTest() int { return s.edge }

// NodeForTest reports the stored node field without modifying the fixture.
// This is a constant-time observation of actual engine metadata, compiled only for tests.
func (s blossomDelta) NodeForTest() int { return s.node }

// Test-only method expressions. Each alias calls the actual private operation
// with an explicit fixture receiver. No adapter reimplements its mathematics.
var (
	// SetMatchedEdgeForTest exposes setMatchedEdge only to the test binary.
	SetMatchedEdgeForTest = (*blossomEngine).setMatchedEdge
	// ClearMatchedEdgeForTest exposes clearMatchedEdge only to the test binary.
	ClearMatchedEdgeForTest = (*blossomEngine).clearMatchedEdge
	// FlipAugmentingPathForTest exposes flipAugmentingPath only to the test binary.
	FlipAugmentingPathForTest = (*blossomEngine).flipAugmentingPath
	// VerifyAugmentingEdgeSequenceForTest exposes verifyAugmentingEdgeSequence only to the test binary.
	VerifyAugmentingEdgeSequenceForTest = (*blossomEngine).verifyAugmentingEdgeSequence
	// EdgeSequenceAlternatesForTest exposes edgeSequenceAlternates only to the test binary.
	EdgeSequenceAlternatesForTest = (*blossomEngine).edgeSequenceAlternates
	// CanAppendAlternatingForTest exposes canAppendAlternating only to the test binary.
	CanAppendAlternatingForTest = (*blossomEngine).canAppendAlternating
	// RefreshAllocatedBlossomBasesForTest exposes refreshAllocatedBlossomBases only to the test binary.
	RefreshAllocatedBlossomBasesForTest = (*blossomEngine).refreshAllocatedBlossomBases
	// CurrentBlossomBaseFromMatchingForTest exposes currentBlossomBaseFromMatching only to the test binary.
	CurrentBlossomBaseFromMatchingForTest = (*blossomEngine).currentBlossomBaseFromMatching
	// TryLiftedAugmentingCandidatesForTest exposes tryLiftedAugmentingCandidates only to the test binary.
	TryLiftedAugmentingCandidatesForTest = (*blossomEngine).tryLiftedAugmentingCandidates
	// ResetForestForTest exposes resetForest only to the test binary.
	ResetForestForTest = (*blossomEngine).resetForest
	// AssignOuterRootForTest exposes assignOuterRoot only to the test binary.
	AssignOuterRootForTest = (*blossomEngine).assignOuterRoot
	// GrowForTest exposes grow only to the test binary.
	GrowForTest = (*blossomEngine).grow
	// ShrinkForTest exposes shrink only to the test binary.
	ShrinkForTest = (*blossomEngine).shrink
	// ExpandForTest exposes expand only to the test binary.
	ExpandForTest = (*blossomEngine).expand
	// ExpandInnerBlossomInForestForTest exposes expandInnerBlossomInForest only to the test binary.
	ExpandInnerBlossomInForestForTest = (*blossomEngine).expandInnerBlossomInForest
	// CycleStepsForTest exposes cycleSteps only to the test binary.
	CycleStepsForTest = (*blossomEngine).cycleSteps
	// NodeContainsVertexForTest exposes nodeContainsVertex only to the test binary.
	NodeContainsVertexForTest = (*blossomEngine).nodeContainsVertex
	// PathToRootForTest exposes pathToRoot only to the test binary.
	PathToRootForTest = (*blossomEngine).pathToRoot
	// NodePathToRootForTest exposes nodePathToRoot only to the test binary.
	NodePathToRootForTest = (*blossomEngine).nodePathToRoot
	// FindBlossomBaseAndCycleForTest exposes findBlossomBaseAndCycle only to the test binary.
	FindBlossomBaseAndCycleForTest = (*blossomEngine).findBlossomBaseAndCycle
	// AllocateBlossomNodeForTest exposes allocateBlossomNode only to the test binary.
	AllocateBlossomNodeForTest = (*blossomEngine).allocateBlossomNode
	// ContractCycleIntoBlossomForTest exposes contractCycleIntoBlossom only to the test binary.
	ContractCycleIntoBlossomForTest = (*blossomEngine).contractCycleIntoBlossom
	// RedirectForestThroughContractedBlossomForTest exposes redirectForestThroughContractedBlossom only to the test binary.
	RedirectForestThroughContractedBlossomForTest = (*blossomEngine).redirectForestThroughContractedBlossom
	// CycleIndexPathForTest exposes cycleIndexPath only to the test binary.
	CycleIndexPathForTest = (*blossomEngine).cycleIndexPath
	// BaseChildIndexForTest exposes baseChildIndex only to the test binary.
	BaseChildIndexForTest = (*blossomEngine).baseChildIndex
	// TopNodeOfVertexForTest exposes topNodeOfVertex only to the test binary.
	TopNodeOfVertexForTest = (*blossomEngine).topNodeOfVertex
	// OrientEdgeForNodesForTest exposes orientEdgeForNodes only to the test binary.
	OrientEdgeForNodesForTest = (*blossomEngine).orientEdgeForNodes
	// ScanOuterNodeTightEdgesForTest exposes scanOuterNodeTightEdges only to the test binary.
	ScanOuterNodeTightEdgesForTest = (*blossomEngine).scanOuterNodeTightEdges
	// ApplyBlossomEventForTest exposes applyBlossomEvent only to the test binary.
	ApplyBlossomEventForTest = (*blossomEngine).applyBlossomEvent
	// SolveForTest exposes solve only to the test binary.
	SolveForTest = (*blossomEngine).solve
	// SlackForTest exposes slack only to the test binary.
	SlackForTest = (*blossomEngine).slack
	// DualToleranceForTest exposes dualTolerance only to the test binary.
	DualToleranceForTest = (*blossomEngine).dualTolerance
	// DualSelectionToleranceForTest exposes dualSelectionTolerance only to the test binary.
	DualSelectionToleranceForTest = (*blossomEngine).dualSelectionTolerance
	// ApplyDeltaForTest exposes applyDelta only to the test binary.
	ApplyDeltaForTest = (*blossomEngine).applyDelta
	// ConsiderExpandDeltasForTest exposes considerExpandDeltas only to the test binary.
	ConsiderExpandDeltasForTest = (*blossomEngine).considerExpandDeltas
	// ImproveDeltaForTest exposes improveDelta only to the test binary.
	ImproveDeltaForTest = (*blossomEngine).improveDelta
	// NextDeltaForTest exposes nextDelta only to the test binary.
	NextDeltaForTest = (*blossomEngine).nextDelta
	// ShiftMemberVertexDualsForTest exposes shiftMemberVertexDuals only to the test binary.
	ShiftMemberVertexDualsForTest = (*blossomEngine).shiftMemberVertexDuals
	// ComposeAugmentingPathForTest exposes composeAugmentingPath only to the test binary.
	ComposeAugmentingPathForTest = (*blossomEngine).composeAugmentingPath
	// MakePathStepForTest exposes makePathStep only to the test binary.
	MakePathStepForTest = (*blossomEngine).makePathStep
	// VerifyPathStepContinuityForTest exposes verifyPathStepContinuity only to the test binary.
	VerifyPathStepContinuityForTest = (*blossomEngine).verifyPathStepContinuity
	// LiftAugmentingPathChoicesForTest exposes liftAugmentingPathChoices only to the test binary.
	LiftAugmentingPathChoicesForTest = (*blossomEngine).liftAugmentingPathChoices
	// LiftThroughBlossomChoicesForTest exposes liftThroughBlossomChoices only to the test binary.
	LiftThroughBlossomChoicesForTest = (*blossomEngine).liftThroughBlossomChoices
	// CycleChildIndexContainingVertexForTest exposes cycleChildIndexContainingVertex only to the test binary.
	CycleChildIndexContainingVertexForTest = (*blossomEngine).cycleChildIndexContainingVertex
	// ExportMatchingForTest exposes exportMatching only to the test binary.
	ExportMatchingForTest = (*blossomEngine).exportMatching
	// VerifyOptimalStateForTest exposes verifyOptimalState only to the test binary.
	VerifyOptimalStateForTest = (*blossomEngine).verifyOptimalState
	// CheckedSquareLengthForTest exposes checkedSquareLength only to the test binary.
	CheckedSquareLengthForTest = checkedSquareLength
	// CheckBlossomSizesForTest exposes checkBlossomSizes only to the test binary.
	CheckBlossomSizesForTest = checkBlossomSizes
	// BuildMatchingProblemForTest exposes buildMatchingProblem only to the test binary.
	BuildMatchingProblemForTest = buildMatchingProblem
	// VerifyPerfectMatchingForTest exposes verifyPerfectMatching only to the test binary.
	VerifyPerfectMatchingForTest = verifyPerfectMatching
	// MatchingCostForTest exposes matchingCost only to the test binary.
	MatchingCostForTest = matchingCost
	// CanonicalizeMatchingPairsForTest exposes canonicalizeMatchingPairs only to the test binary.
	CanonicalizeMatchingPairsForTest = canonicalizeMatchingPairs
	// CycleEdgeBetweenAdjacentIndicesForTest exposes cycleEdgeBetweenAdjacentIndices only to the test binary.
	CycleEdgeBetweenAdjacentIndicesForTest = cycleEdgeBetweenAdjacentIndices
	// DeltaTieLessForTest exposes deltaTieLess only to the test binary.
	DeltaTieLessForTest = deltaTieLess
	// BuildBlossomEdgesForTest exposes buildBlossomEdges only to the test binary.
	BuildBlossomEdgesForTest = buildBlossomEdges
)
