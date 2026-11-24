// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp provides test-build access to invariant boundaries.
// The aliases delegate to the real kernels; assertions belong in tsp_test.
// Function-valued fixture bindings retain the identity of private engine types.
// Call them directly; do not reassign them, including in parallel tests.
package tsp

import (
	"slices"
	"time"

	"github.com/lvlath/go/matrix"
)

// These aliases exist only in the test build. They expose the actual operations
// without duplicating their implementation or widening the production API.
var (
	BlossomMatchForTest                = blossomMatch
	GreedyMatchForTest                 = greedyMatch
	TranslateMatchingErrorForTest      = translateMatchingError
	AppendMatchingResultForTest        = appendMatchingResult
	ApplyChristofidesMatchingForTest   = applyChristofidesMatching
	ReverseArcInPlaceForTest           = reverseArcInPlace
	CostClosedTourFromWeightsForTest   = costClosedTourFromWeights
	FinishLocalSearchCurrentForTest    = finishLocalSearchCurrent
	AttachLocalSearchProgressForTest   = attachLocalSearchProgress
	CopyWeightsWithModeForTest         = copyWeightsWithMode
	PublishKernelResultForTest         = publishKernelResult
	FinalizeSolverOptionsForTest       = finalizeSolverOptions
	TrivialRingForTest                 = trivialRing
	SolveDegenerateIfAnyForTest        = solveDegenerateIfAny
	PrepareSolverDistanceMatrixForTest = prepareSolverDistanceMatrix
	SolvePreparedMatrixForTest         = solvePreparedMatrix
	ValidateThreeOptMoveForTest        = validateThreeOptMove
	SegmentEndpointsForTest            = segmentEndpoints
	ThreeOptDeltaSymmetricForTest      = threeOptDeltaSymmetric
	ThreeOptDeltaATSPTailSwapForTest   = threeOptDeltaATSPTailSwap
	AppendOrientedSegmentForTest       = appendOrientedSegment
	ApplyThreeOptSymmetricForTest      = applyThreeOptSymmetric
	ApplyThreeOptATSPTailSwapForTest   = applyThreeOptATSPTailSwap
)

// WeightBufferForTest preserves the real detached weight representation.
// Tests may deliberately violate scalar invariants at defensive boundaries.
type WeightBufferForTest = weightBuffer

// LocalSearchResultForTest is the actual internal progress record.
// It permits assertions before public metadata is attached.
type LocalSearchResultForTest = localSearchResult

// PreparedMatrixForTest identifies the dispatcher input after admission.
// Constructing it directly is reserved for tests of the dispatcher's error paths.
type PreparedMatrixForTest = preparedMatrix

// NewWeightBufferForTest constructs an isolated weight snapshot.
// It deliberately bypasses admission so the downstream guards can be tested.
// Implementation:
//   - Copy the supplied row-major weights and retain the declared order.
//
// Inputs: n is the order; weights are row-major arc costs.
// Returns: detached internal buffer, including deliberately invalid scalars.
// Errors: none; no validation is performed here.
// Determinism: copies values in their original order.
// Complexity: O(len(weights)) time and space.
// Notes: never pass an undersized buffer to an unchecked indexed kernel.
// AI-Hints: use public solvers for normal input admission.
var NewWeightBufferForTest = func(n int, weights []float64) WeightBufferForTest {
	return weightBuffer{n: n, w: slices.Clone(weights)}
}

// InspectWeightBufferForTest returns the order and a detached copy of weights.
// The observer cannot mutate the buffer owned by the tested kernel.
// Complexity: O(n*n) time and space for a valid n-by-n buffer.
func InspectWeightBufferForTest(b WeightBufferForTest) (int, []float64) {
	return b.n, slices.Clone(b.w)
}

// NewLocalSearchResultForTest builds a detached progress record.
// It bypasses search to isolate publication and attachment semantics.
// Inputs: tour, its cost, committed move count and interruption flag.
// Returns: actual internal record with an independent tour slice.
// Errors: none; this constructor does not validate a tour.
// Determinism: preserves the supplied tour order and metadata.
// Complexity: O(len(tour)) time and space.
// AI-Hints: do not substitute this fixture for an end-to-end timeout test.
var NewLocalSearchResultForTest = func(tour []int, cost float64, iterations int, timedOut bool) LocalSearchResultForTest {
	return localSearchResult{tour: slices.Clone(tour), cost: cost, iterations: iterations, timedOut: timedOut}
}

// InspectLocalSearchResultForTest observes the actual progress record.
// The returned tour is detached to preserve the record's ownership.
// Complexity: O(len(tour)) time and space.
func InspectLocalSearchResultForTest(r LocalSearchResultForTest) ([]int, float64, int, bool) {
	return slices.Clone(r.tour), r.cost, r.iterations, r.timedOut
}

// NewPreparedMatrixForTest constructs a dispatcher fixture without admission.
// The supplied matrix is borrowed, just as in the real no-closure preparation path.
// Inputs: dist, declared order and metric-closure metadata.
// Returns: the actual internal dispatcher input.
// Errors: none; this fixture intentionally does not validate its arguments.
// Determinism: retains all arguments exactly.
// Complexity: O(1) time and space.
// AI-Hints: use only for dispatch-boundary guards; normal tests call SolveMatrix.
var NewPreparedMatrixForTest = func(dist matrix.Matrix, n int, closure bool) PreparedMatrixForTest {
	return preparedMatrix{dist: dist, n: n, metricClosureApplied: closure}
}

// SetLocalSearchClockForTest installs a deterministic clock for one serial test.
// It replaces only the existing time source, never the actual search or deadline logic.
// Implementation:
//   - Save the real clock, install the supplied clock and return a restoration callback.
//
// Inputs: non-nil now function supplying monotonic test timestamps.
// Returns: restoration callback for t.Cleanup.
// Errors: none; a nil clock is outside this test fixture's domain.
// Determinism: determined by the caller's clock sequence.
// Complexity: O(1) time and space.
// Notes: global test-build mutation; never combine with parallel tests or live goroutines.
// AI-Hints: register the returned callback immediately, including failure paths.
func SetLocalSearchClockForTest(now func() time.Time) func() {
	previous := localSearchNow
	localSearchNow = now
	return func() { localSearchNow = previous }
}

// Deadline masks expose the real candidate-check cadence, not a second policy.
const (
	TwoOptDeadlineMaskForTest   = twoOptDeadlineCheckMask
	ThreeOptDeadlineMaskForTest = threeOptDeadlineCheckMask
	SegmentS1ForTest            = segmentS1
)

// SegmentIDForTest, OrientedSegmentForTest and ThreeOptMoveForTest retain the
// actual move representation used by the symmetric and directed kernels.
type SegmentIDForTest = segmentID
type OrientedSegmentForTest = orientedSegment
type ThreeOptMoveForTest = threeOptMove

// NewOrientedSegmentForTest constructs one actual segment descriptor.
// Invalid IDs are intentional fixtures for the move validator.
// Complexity: O(1) time and space.
var NewOrientedSegmentForTest = func(id SegmentIDForTest, reverse bool) OrientedSegmentForTest {
	return orientedSegment{id: id, reverse: reverse}
}

// NewThreeOptMoveForTest combines two descriptors without validating them.
// This separates the construction of malformed input from its rejection.
// Complexity: O(1) time and space.
var NewThreeOptMoveForTest = func(x, y OrientedSegmentForTest) ThreeOptMoveForTest {
	return threeOptMove{x: x, y: y}
}

// SymmetricThreeOptMovesForTest copies the real ordered move table.
// Tests compare each move with an independently enumerated Hamiltonian cycle.
// Complexity: O(1) time and space because the table has fixed cardinality seven.
var SymmetricThreeOptMovesForTest = func() []ThreeOptMoveForTest {
	return slices.Clone(symmetricThreeOptMoves[:])
}
