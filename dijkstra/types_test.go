// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package dijkstra_test

import (
	"math"
	"testing"
	"time"

	"github.com/lvlath/go/dijkstra"
)

// AI-HINTS (file):
//   - These tests validate Result as a standalone public contract type.
//   - Build result fixtures directly; do not run the algorithm here.
//   - Keep missing-target, unreachable-target, tracking-disabled, and source-path cases separate.
//   - Use exact path assertions because PathTo is deterministic by contract.
//   - Use nil-safe checks and panic-safety anchors where receiver safety matters.
//   - Caller-owned Prev state may be malformed; cyclic reconstruction must terminate.
//   - Never accept a predecessor vertex absent from Distances.
//   - A timeout guard is allowed only for the dedicated non-termination regression;
//     it must not become production timing semantics.

const (
	resultTestSourceID       = "A"
	resultTestMiddleID       = "B"
	resultTestTargetID       = "C"
	resultTestUnknownID      = "Z"
	resultTestBrokenTargetID = "X"

	resultTestDistanceSource = 0.0
	resultTestDistanceMiddle = 2.0
	resultTestDistanceTarget = 5.0

	// resultTestCycleLeftID is the first vertex in a malformed predecessor cycle.
	resultTestCycleLeftID = "cycle:left"

	// resultTestCycleRightID is the second vertex in a malformed predecessor cycle.
	resultTestCycleRightID = "cycle:right"

	// resultTestCycleTargetID is the finite target whose chain enters the cycle.
	resultTestCycleTargetID = "cycle:target"

	// resultTestForeignParentID is a predecessor absent from the result domain.
	resultTestForeignParentID = "foreign:parent"

	// resultTestCycleGuardTimeout prevents an infinite-loop regression from
	// hanging the complete test process indefinitely.
	resultTestCycleGuardTimeout = 2 * time.Second
)

// TestResult_IsNil verifies the Nilable contract and nil-receiver safety
// of Result.
//
// Implementation:
//   - Stage 1: Evaluate IsNil on a nil receiver under a panic-safety guard.
//   - Stage 2: Evaluate IsNil on a non-nil receiver.
//   - Stage 3: Assert the exact boolean outcomes.
//
// Behavior highlights:
//   - Nil receivers are supported safely.
//   - Result participates in the core.Nilable contract.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure on panic or contract mismatch.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This test validates the public result contract, not algorithm execution.
//
// AI-Hints:
//   - Keep nil-receiver safety explicit for result-surface types.
func TestResult_IsNil(t *testing.T) {
	var nilResult *dijkstra.Result

	mustPanicFree(t, func() {
		mustEqualBool(t, nilResult.IsNil(), true, "nil receiver IsNil mismatch: got=%v want=%v", nilResult.IsNil(), true)
	})

	nonNilResult := &dijkstra.Result{}
	mustEqualBool(t, nonNilResult.IsNil(), false, "non-nil receiver IsNil mismatch: got=%v want=%v", nonNilResult.IsNil(), false)
}

// TestResult_DistanceTo verifies target lookup semantics for nil receivers,
// empty targets, unknown targets, reachable targets, and known unreachable targets.
//
// Implementation:
//   - Stage 1: Cover nil-receiver lookup.
//   - Stage 2: Cover empty-target rejection.
//   - Stage 3: Cover unknown-target rejection.
//   - Stage 4: Cover reachable target lookup.
//   - Stage 5: Cover known unreachable target lookup.
//
// Behavior highlights:
//   - Unknown target and unreachable target are distinct cases.
//   - +Inf is returned as canonical data, not as an error.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure on wrong sentinel classification or value mismatch.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1) per subtest.
//
// Notes:
//   - This test validates the result-surface contract directly.
//
// AI-Hints:
//   - Do not collapse ErrTargetNotFound and +Inf unreachable into one behavior.
func TestResult_DistanceTo(t *testing.T) {
	t.Run("nil receiver", func(t *testing.T) {
		var result *dijkstra.Result

		_, err := result.DistanceTo(resultTestTargetID)
		mustErrorIs(t, err, dijkstra.ErrNilResult)
	})

	t.Run("empty target", func(t *testing.T) {
		result := &dijkstra.Result{
			SourceID: resultTestSourceID,
			Distances: map[string]float64{
				resultTestSourceID: resultTestDistanceSource,
			},
		}

		_, err := result.DistanceTo("")
		mustErrorIs(t, err, dijkstra.ErrEmptyTargetID)
	})

	t.Run("target not found", func(t *testing.T) {
		result := &dijkstra.Result{
			SourceID: resultTestSourceID,
			Distances: map[string]float64{
				resultTestSourceID: resultTestDistanceSource,
			},
		}

		_, err := result.DistanceTo(resultTestUnknownID)
		mustErrorIs(t, err, dijkstra.ErrTargetNotFound)
	})

	t.Run("reachable target", func(t *testing.T) {
		result := &dijkstra.Result{
			SourceID: resultTestSourceID,
			Distances: map[string]float64{
				resultTestSourceID: resultTestDistanceSource,
				resultTestTargetID: resultTestDistanceTarget,
			},
		}

		got, err := result.DistanceTo(resultTestTargetID)
		if err != nil {
			t.Fatalf("DistanceTo(%q) error: %v", resultTestTargetID, err)
		}

		mustEqualFloat64(t, got, resultTestDistanceTarget, "DistanceTo(%q) mismatch: got=%v want=%v", resultTestTargetID, got, resultTestDistanceTarget)
	})

	t.Run("known unreachable target", func(t *testing.T) {
		result := &dijkstra.Result{
			SourceID: resultTestSourceID,
			Distances: map[string]float64{
				resultTestSourceID: resultTestDistanceSource,
				resultTestTargetID: math.Inf(1),
			},
		}

		got, err := result.DistanceTo(resultTestTargetID)
		if err != nil {
			t.Fatalf("DistanceTo(%q) error: %v", resultTestTargetID, err)
		}

		assertInfDistance(t, got)
	})
}

// TestResult_HasPathTo verifies reachability semantics derived from the
// distance contract rather than from predecessor storage.
//
// Implementation:
//   - Stage 1: Cover nil-receiver lookup.
//   - Stage 2: Cover empty-target rejection.
//   - Stage 3: Cover unknown-target rejection.
//   - Stage 4: Cover reachable target success.
//   - Stage 5: Cover known unreachable target success with false.
//
// Behavior highlights:
//   - Reachability depends on distance semantics.
//   - Unknown target remains an error.
//   - +Inf means known but unreachable.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure on wrong sentinel classification or boolean mismatch.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1) per subtest.
//
// Notes:
//   - This method does not require predecessor tracking.
//
// AI-Hints:
//   - Keep HasPathTo aligned with DistanceTo semantics.
func TestResult_HasPathTo(t *testing.T) {
	t.Run("nil receiver", func(t *testing.T) {
		var result *dijkstra.Result

		_, err := result.HasPathTo(resultTestTargetID)
		mustErrorIs(t, err, dijkstra.ErrNilResult)
	})

	t.Run("empty target", func(t *testing.T) {
		result := &dijkstra.Result{
			SourceID: resultTestSourceID,
			Distances: map[string]float64{
				resultTestSourceID: resultTestDistanceSource,
			},
		}

		_, err := result.HasPathTo("")
		mustErrorIs(t, err, dijkstra.ErrEmptyTargetID)
	})

	t.Run("target not found", func(t *testing.T) {
		result := &dijkstra.Result{
			SourceID: resultTestSourceID,
			Distances: map[string]float64{
				resultTestSourceID: resultTestDistanceSource,
			},
		}

		_, err := result.HasPathTo(resultTestUnknownID)
		mustErrorIs(t, err, dijkstra.ErrTargetNotFound)
	})

	t.Run("reachable target", func(t *testing.T) {
		result := &dijkstra.Result{
			SourceID: resultTestSourceID,
			Distances: map[string]float64{
				resultTestSourceID: resultTestDistanceSource,
				resultTestTargetID: resultTestDistanceTarget,
			},
		}

		got, err := result.HasPathTo(resultTestTargetID)
		if err != nil {
			t.Fatalf("HasPathTo(%q) error: %v", resultTestTargetID, err)
		}

		mustEqualBool(t, got, true, "HasPathTo(%q) mismatch: got=%v want=%v", resultTestTargetID, got, true)
	})

	t.Run("known unreachable target", func(t *testing.T) {
		result := &dijkstra.Result{
			SourceID: resultTestSourceID,
			Distances: map[string]float64{
				resultTestSourceID: resultTestDistanceSource,
				resultTestTargetID: math.Inf(1),
			},
		}

		got, err := result.HasPathTo(resultTestTargetID)
		if err != nil {
			t.Fatalf("HasPathTo(%q) error: %v", resultTestTargetID, err)
		}

		mustEqualBool(t, got, false, "HasPathTo(%q) mismatch: got=%v want=%v", resultTestTargetID, got, false)
	})
}

// TestResult_PathTo_Source verifies that PathTo returns a single-vertex
// witness for the source when predecessor tracking is enabled.
//
// Implementation:
//   - Stage 1: Construct a tracked result whose source is known.
//   - Stage 2: Query PathTo on the source vertex.
//   - Stage 3: Assert the exact single-vertex path.
//
// Behavior highlights:
//   - Source-to-source path is a valid witness.
//   - Tracking must be enabled for PathTo to operate.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure on unexpected error or path mismatch.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This test uses Prev != nil intentionally because tracking-disabled semantics are covered elsewhere.
//
// AI-Hints:
//   - Keep source-path behavior separate from tracking-disabled behavior.
func TestResult_PathTo_Source(t *testing.T) {
	result := &dijkstra.Result{
		SourceID: resultTestSourceID,
		Distances: map[string]float64{
			resultTestSourceID: resultTestDistanceSource,
		},
		Prev: map[string]string{
			resultTestSourceID: "",
		},
	}

	got, err := result.PathTo(resultTestSourceID)
	if err != nil {
		t.Fatalf("PathTo(%q) error: %v", resultTestSourceID, err)
	}

	assertPathEqual(t, got, []string{resultTestSourceID})
}

// TestResult_PathTo_TargetNotFound verifies that PathTo rejects a target
// that does not exist in the result domain.
//
// Implementation:
//   - Stage 1: Construct a tracked result with a known target domain.
//   - Stage 2: Query PathTo on an unknown target.
//   - Stage 3: Assert ErrTargetNotFound through the sentinel protocol.
//
// Behavior highlights:
//   - Missing target is not equivalent to unreachable target.
//   - Tracking state does not mask lookup failures.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure on wrong sentinel classification.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This test validates result-domain lookup semantics only.
//
// AI-Hints:
//   - Do not map missing-target failures to ErrNoPath.
func TestResult_PathTo_TargetNotFound(t *testing.T) {
	result := &dijkstra.Result{
		SourceID: resultTestSourceID,
		Distances: map[string]float64{
			resultTestSourceID: resultTestDistanceSource,
			resultTestTargetID: resultTestDistanceTarget,
		},
		Prev: map[string]string{
			resultTestSourceID: "",
			resultTestTargetID: resultTestSourceID,
		},
	}

	_, err := result.PathTo(resultTestUnknownID)
	mustErrorIs(t, err, dijkstra.ErrTargetNotFound)
}

// TestResult_PathTo_NoPath verifies that PathTo reports ErrNoPath for a
// known target that is unreachable under the stored distance contract.
//
// Implementation:
//   - Stage 1: Construct a tracked result with a known unreachable target.
//   - Stage 2: Query PathTo on that target.
//   - Stage 3: Assert ErrNoPath through the sentinel protocol.
//
// Behavior highlights:
//   - Known unreachable target is distinct from missing target.
//   - Tracking enabled does not fabricate a path.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure on wrong sentinel classification.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - The target remains present in Distances with +Inf.
//
// AI-Hints:
//   - Keep unreachable-target handling separate from tracking-disabled handling.
func TestResult_PathTo_NoPath(t *testing.T) {
	result := &dijkstra.Result{
		SourceID: resultTestSourceID,
		Distances: map[string]float64{
			resultTestSourceID: resultTestDistanceSource,
			resultTestTargetID: math.Inf(1),
		},
		Prev: map[string]string{
			resultTestSourceID: "",
			resultTestTargetID: "",
		},
	}

	_, err := result.PathTo(resultTestTargetID)
	mustErrorIs(t, err, dijkstra.ErrNoPath)
}

// TestResult_PathTo_TrackingDisabled verifies that PathTo rejects queries
// when predecessor tracking was not enabled for the producing run.
//
// Implementation:
//   - Stage 1: Construct a result with a known reachable target but nil Prev.
//   - Stage 2: Query PathTo on that target.
//   - Stage 3: Assert ErrPathTrackingDisabled through the sentinel protocol.
//
// Behavior highlights:
//   - Prev == nil means tracking disabled, not “no path”.
//   - Reachable distance does not override the tracking requirement.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure on wrong sentinel classification.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This test protects the explicit tracking contract.
//
// AI-Hints:
//   - Do not collapse tracking-disabled and no-path behavior.
func TestResult_PathTo_TrackingDisabled(t *testing.T) {
	result := &dijkstra.Result{
		SourceID: resultTestSourceID,
		Distances: map[string]float64{
			resultTestSourceID: resultTestDistanceSource,
			resultTestTargetID: resultTestDistanceTarget,
		},
		Prev: nil,
	}

	_, err := result.PathTo(resultTestTargetID)
	mustErrorIs(t, err, dijkstra.ErrPathTrackingDisabled)
}

// TestResult_PathTo_Success verifies successful deterministic path
// reconstruction from a valid predecessor chain.
//
// Implementation:
//   - Stage 1: Construct a tracked result with a valid predecessor chain.
//   - Stage 2: Query PathTo on the target.
//   - Stage 3: Assert the exact deterministic witness order.
//
// Behavior highlights:
//   - Path reconstruction follows Prev backward and returns a forward witness.
//   - Exact order matters because predecessor selection is deterministic.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure on unexpected error or path mismatch.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(k), Space O(k), where k is the returned path length.
//
// Notes:
//   - This is a direct public-contract test of witness reconstruction.
//
// AI-Hints:
//   - Do not weaken this into unordered path comparison.
func TestResult_PathTo_Success(t *testing.T) {
	result := &dijkstra.Result{
		SourceID: resultTestSourceID,
		Distances: map[string]float64{
			resultTestSourceID: resultTestDistanceSource,
			resultTestMiddleID: resultTestDistanceMiddle,
			resultTestTargetID: resultTestDistanceTarget,
		},
		Prev: map[string]string{
			resultTestSourceID: "",
			resultTestMiddleID: resultTestSourceID,
			resultTestTargetID: resultTestMiddleID,
		},
	}

	got, err := result.PathTo(resultTestTargetID)
	if err != nil {
		t.Fatalf("PathTo(%q) error: %v", resultTestTargetID, err)
	}

	assertPathEqual(t, got, []string{resultTestSourceID, resultTestMiddleID, resultTestTargetID})
}

// TestResult_PathTo_BrokenPredecessorChain verifies that PathTo reports
// ErrNoPath when the target is finite but the predecessor chain cannot reach the source.
//
// Implementation:
//   - Stage 1: Construct a tracked result with a finite target distance.
//   - Stage 2: Provide a broken predecessor chain that stops before the source.
//   - Stage 3: Query PathTo on the broken target.
//   - Stage 4: Assert ErrNoPath through the sentinel protocol.
//
// Behavior highlights:
//   - Finite distance does not authorize path fabrication.
//   - Broken witness chains remain explicit contract failures.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure on wrong sentinel classification.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(k), Space O(k), where k is the traversed broken chain length.
//
// Notes:
//   - This is a live branch in PathTo and should remain covered.
//
// AI-Hints:
//   - Keep broken-chain coverage explicit; it protects witness honesty.
func TestResult_PathTo_BrokenPredecessorChain(t *testing.T) {
	result := &dijkstra.Result{
		SourceID: resultTestSourceID,
		Distances: map[string]float64{
			resultTestSourceID:       resultTestDistanceSource,
			resultTestBrokenTargetID: resultTestDistanceTarget,
		},
		Prev: map[string]string{
			resultTestSourceID:       "",
			resultTestBrokenTargetID: "",
		},
	}

	_, err := result.PathTo(resultTestBrokenTargetID)
	mustErrorIs(t, err, dijkstra.ErrNoPath)
}

// TestResult_Clone verifies nil safety, deep-copy semantics, and exact
// preservation of nil predecessor state in Clone.
//
// Implementation:
//   - Stage 1: Cover nil-receiver cloning.
//   - Stage 2: Cover deep-copy semantics for tracked results.
//   - Stage 3: Mutate the clone and assert original isolation.
//   - Stage 4: Cover preservation of nil Prev for untracked results.
//
// Behavior highlights:
//   - Clone returns nil for nil receivers.
//   - Clone deep-copies maps and preserves ownership isolation.
//   - Clone preserves Prev == nil exactly.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure on nil-state mismatch, value mismatch, or aliasing leakage.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(V), Space O(V), where V is the number of copied map entries.
//
// Notes:
//   - Clone is part of the public ownership contract, not just a convenience helper.
//
// AI-Hints:
//   - Preserve nil Prev exactly; do not normalize it into an empty map.
func TestResult_Clone(t *testing.T) {
	t.Run("nil receiver", func(t *testing.T) {
		var result *dijkstra.Result

		cloned := result.Clone()
		mustNilState(t, cloned, true, "Clone(nil)")
	})

	t.Run("deep copy", func(t *testing.T) {
		original := &dijkstra.Result{
			SourceID: resultTestSourceID,
			Distances: map[string]float64{
				resultTestSourceID: resultTestDistanceSource,
				resultTestTargetID: resultTestDistanceTarget,
			},
			Prev: map[string]string{
				resultTestSourceID: "",
				resultTestTargetID: resultTestSourceID,
			},
		}

		cloned := original.Clone()
		mustNilState(t, cloned, false, "Clone result")
		mustEqualString(t, cloned.SourceID, original.SourceID, "SourceID mismatch: got=%q want=%q", cloned.SourceID, original.SourceID)

		cloned.Distances[resultTestTargetID] = 99
		cloned.Prev[resultTestTargetID] = resultTestMiddleID

		mustEqualFloat64(t, original.Distances[resultTestTargetID], resultTestDistanceTarget, "original distance mutated through clone: got=%v want=%v", original.Distances[resultTestTargetID], resultTestDistanceTarget)
		mustEqualString(t, original.Prev[resultTestTargetID], resultTestSourceID, "original predecessor mutated through clone: got=%q want=%q", original.Prev[resultTestTargetID], resultTestSourceID)
	})

	t.Run("preserves nil Prev", func(t *testing.T) {
		original := &dijkstra.Result{
			SourceID: resultTestSourceID,
			Distances: map[string]float64{
				resultTestSourceID: resultTestDistanceSource,
			},
			Prev: nil,
		}

		cloned := original.Clone()
		mustNilState(t, cloned, false, "Clone result")
		mustNilState(t, cloned.Prev, true, "Clone preserves nil Prev")
	})
}

// TestResult_PathTo_CyclicPredecessorChain verifies that PathTo detects a
// caller-mutated predecessor cycle and terminates with ErrNoPath.
//
// Implementation:
//   - Stage 1: Construct a finite tracked result whose target enters a two-vertex cycle.
//   - Stage 2: Execute PathTo in a worker and report completion through a buffered channel.
//   - Stage 3: Guard the test process against an infinite-loop regression.
//   - Stage 4: Assert nil path and ErrNoPath.
//
// Behavior highlights:
//   - Cyclic Prev state must never hang path reconstruction.
//   - No partial witness is published.
//   - The worker never calls testing failure methods.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure on panic, timeout, non-nil path, or wrong sentinel.
//
// Determinism:
//   - The expected functional outcome is deterministic.
//   - The timeout is only a CI safety boundary for non-termination regressions.
//
// Complexity:
//   - Correct implementation completes in O(c), where c is the inspected cycle prefix.
//   - Space O(c).
//
// Notes:
//   - Result maps are deliberately malformed to exercise defensive public behavior.
//
// AI-Hints:
//   - Do not remove this regression merely because Dijkstra-generated Prev is acyclic.
//   - Result is caller-owned and publicly constructible.
func TestResult_PathTo_CyclicPredecessorChain(t *testing.T) {
	result := &dijkstra.Result{
		SourceID: resultTestSourceID,
		Distances: map[string]float64{
			resultTestSourceID:      resultTestDistanceSource,
			resultTestCycleLeftID:   1.0,
			resultTestCycleRightID:  2.0,
			resultTestCycleTargetID: 3.0,
		},
		Prev: map[string]string{
			resultTestSourceID:      "",
			resultTestCycleTargetID: resultTestCycleLeftID,
			resultTestCycleLeftID:   resultTestCycleRightID,
			resultTestCycleRightID:  resultTestCycleLeftID,
		},
	}

	type pathOutcome struct {
		path       []string
		err        error
		panicValue any
	}

	outcomeChannel := make(chan pathOutcome, 1)

	go func() {
		outcome := pathOutcome{}

		defer func() {
			outcome.panicValue = recover()
			outcomeChannel <- outcome
		}()

		outcome.path, outcome.err = result.PathTo(resultTestCycleTargetID)
	}()

	timer := time.NewTimer(resultTestCycleGuardTimeout)
	defer timer.Stop()

	select {
	case outcome := <-outcomeChannel:
		if outcome.panicValue != nil {
			t.Fatalf("PathTo panicked on cyclic Prev: %v", outcome.panicValue)
		}

		mustNilState(t, outcome.path, true, "PathTo cyclic predecessor path")
		mustErrorIs(t, outcome.err, dijkstra.ErrNoPath)

	case <-timer.C:
		t.Fatalf(
			"PathTo(%q) did not terminate within %v; cyclic predecessor regression",
			resultTestCycleTargetID,
			resultTestCycleGuardTimeout,
		)
	}
}

// TestResult_PathTo_PredecessorOutsideResultDomain verifies that PathTo rejects
// a predecessor identifier absent from Distances.
//
// Implementation:
//   - Stage 1: Construct a finite target and a predecessor chain that leaves the domain.
//   - Stage 2: Query PathTo.
//   - Stage 3: Assert nil path and ErrNoPath.
//
// Behavior highlights:
//   - Prev cannot introduce vertices that are absent from the result domain.
//   - Finite target distance alone does not authorize witness fabrication.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure on non-nil path or wrong sentinel.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(k), Space O(k), for the short malformed chain.
//
// Notes:
//   - The foreign parent deliberately has a Prev entry but no Distances entry.
//
// AI-Hints:
//   - Keep domain validation separate from missing-map-entry validation.
func TestResult_PathTo_PredecessorOutsideResultDomain(t *testing.T) {
	result := &dijkstra.Result{
		SourceID: resultTestSourceID,
		Distances: map[string]float64{
			resultTestSourceID: resultTestDistanceSource,
			resultTestTargetID: resultTestDistanceTarget,
		},
		Prev: map[string]string{
			resultTestSourceID:        "",
			resultTestTargetID:        resultTestForeignParentID,
			resultTestForeignParentID: resultTestSourceID,
		},
	}

	path, err := result.PathTo(resultTestTargetID)

	mustNilState(t, path, true, "PathTo foreign predecessor path")
	mustErrorIs(t, err, dijkstra.ErrNoPath)
}
