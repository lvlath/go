// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package dijkstra_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/dijkstra"
)

// AI-HINTS (file):
//   - These tests validate exported option constructors and canonical public
//     option assembly.
//   - DefaultOptions is public and must be tested directly.
//   - Built-in Option values may be applied directly to a detached Options value.
//   - Assembly-only laws such as nil-option rejection and call-order behavior
//     must be observed through Dijkstra, not through production TestOnly bridges.
//   - Caller-defined Option implementations must not bypass finalized-state validation.
//   - Use errors.Is protocol checks through mustErrorIs.
//   - Do not introduce panic-based option tests or production testing exports.

const (
	// optionTestSourceID is the source vertex for public assembly tests.
	optionTestSourceID = "options:source"

	// optionTestMiddleID is the intermediate vertex for policy-composition tests.
	optionTestMiddleID = "options:middle"

	// optionTestTargetID is the final target for policy-composition tests.
	optionTestTargetID = "options:target"

	// optionTestEdgeWeight is used by both edges in the assembly fixture.
	optionTestEdgeWeight = 2.0

	// optionTestRestrictiveMaxDistance excludes the first fixture edge.
	optionTestRestrictiveMaxDistance = 1.0

	// optionTestPermissiveMaxDistance includes the complete two-edge route.
	optionTestPermissiveMaxDistance = 4.0

	// optionTestRestrictiveThreshold blocks fixture edges at their exact weight.
	optionTestRestrictiveThreshold = 2.0

	// optionTestPermissiveThreshold allows every fixture edge.
	optionTestPermissiveThreshold = 3.0

	// optionTestNegativeValue is the canonical finite invalid option value.
	optionTestNegativeValue = -1.0

	// optionTestZeroValue is the exact zero boundary.
	optionTestZeroValue = 0.0
)

// TestDefaultOptions verifies the complete canonical default policy and confirms
// that DefaultOptions returns detached values rather than shared mutable state.
//
// Implementation:
//   - Stage 1: Read the first default Options value.
//   - Stage 2: Assert all documented default fields.
//   - Stage 3: Mutate the local value.
//   - Stage 4: Read another default value and assert that defaults remain intact.
//
// Behavior highlights:
//   - TrackPaths defaults to false.
//   - MaxDistance and InfEdgeThreshold default to +Inf.
//   - Returned Options values are detached scalar snapshots.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure on any default-policy or ownership mismatch.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Positive infinity here belongs to the runtime-policy domain.
//
// AI-Hints:
//   - Test DefaultOptions directly; no test-only wrapper is justified.
func TestDefaultOptions(t *testing.T) {
	first := dijkstra.DefaultOptions()

	mustEqualBool(
		t,
		first.TrackPaths,
		false,
		"TrackPaths default: got=%v want=false",
		first.TrackPaths,
	)

	if !math.IsInf(first.MaxDistance, 1) {
		t.Fatalf("MaxDistance default: got=%v want=+Inf", first.MaxDistance)
	}
	if !math.IsInf(first.InfEdgeThreshold, 1) {
		t.Fatalf(
			"InfEdgeThreshold default: got=%v want=+Inf",
			first.InfEdgeThreshold,
		)
	}

	first.TrackPaths = true
	first.MaxDistance = optionTestZeroValue
	first.InfEdgeThreshold = optionTestPermissiveThreshold

	second := dijkstra.DefaultOptions()

	mustEqualBool(
		t,
		second.TrackPaths,
		false,
		"detached TrackPaths default: got=%v want=false",
		second.TrackPaths,
	)

	if !math.IsInf(second.MaxDistance, 1) {
		t.Fatalf(
			"detached MaxDistance default: got=%v want=+Inf",
			second.MaxDistance,
		)
	}
	if !math.IsInf(second.InfEdgeThreshold, 1) {
		t.Fatalf(
			"detached InfEdgeThreshold default: got=%v want=+Inf",
			second.InfEdgeThreshold,
		)
	}
}

// TestApplyOptions_LastWriterWins verifies canonical option call order through
// the public Dijkstra facade.
//
// Implementation:
//   - Stage 1: Build a deterministic two-edge weighted route.
//   - Stage 2: Apply restrictive MaxDistance and threshold policies.
//   - Stage 3: Apply permissive replacements for the same fields.
//   - Stage 4: Enable path tracking.
//   - Stage 5: Assert that the final permissive writers govern traversal.
//
// Behavior highlights:
//   - If either first writer incorrectly survived, the target would remain +Inf.
//   - The exact finite target distance therefore proves both final policies.
//   - Path reconstruction proves that WithPathTracking was also assembled.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure on graph setup, traversal, distance, or path mismatch.
//
// Determinism:
//   - Deterministic for the fixed directed graph and option sequence.
//
// Complexity:
//   - Setup O(1); measured test traversal follows Dijkstra cost for V=3, E=2.
//
// Notes:
//   - External tests intentionally observe private applyOptions through its
//     canonical public consumer.
//
// AI-Hints:
//   - Do not reintroduce GatherOptionsSnapshotTestOnly for this contract.
func TestApplyOptions_LastWriterWins(t *testing.T) {
	graph, err := core.NewGraph(
		core.WithDirected(true),
		core.WithWeighted(),
	)
	if err != nil {
		t.Fatalf("NewGraph(directed, weighted) failed: %v", err)
	}

	if _, err = graph.AddEdge(optionTestSourceID, optionTestMiddleID, optionTestEdgeWeight); err != nil {
		t.Fatalf("AddEdge(%q,%q) failed: %v", optionTestSourceID, optionTestMiddleID, err)
	}

	if _, err = graph.AddEdge(optionTestMiddleID, optionTestTargetID, optionTestEdgeWeight); err != nil {
		t.Fatalf("AddEdge(%q,%q) failed: %v", optionTestMiddleID, optionTestTargetID, err)
	}

	result, err := dijkstra.Dijkstra(
		graph,
		optionTestSourceID,
		dijkstra.WithMaxDistance(optionTestRestrictiveMaxDistance),
		dijkstra.WithInfEdgeThreshold(optionTestRestrictiveThreshold),
		dijkstra.WithMaxDistance(optionTestPermissiveMaxDistance),
		dijkstra.WithInfEdgeThreshold(optionTestPermissiveThreshold),
		dijkstra.WithPathTracking(),
	)
	if err != nil {
		t.Fatalf("Dijkstra(%q) failed: %v", optionTestSourceID, err)
	}

	mustNilState(t, result, false, "last-writer-wins result")
	mustNilState(t, result.Prev, false, "last-writer-wins Prev")

	distance, err := result.DistanceTo(optionTestTargetID)
	if err != nil {
		t.Fatalf("DistanceTo(%q) failed: %v", optionTestTargetID, err)
	}

	mustEqualFloat64(
		t,
		distance,
		optionTestPermissiveMaxDistance,
		"DistanceTo(%q): got=%v want=%v",
		optionTestTargetID,
		distance,
		optionTestPermissiveMaxDistance,
	)

	path, err := result.PathTo(optionTestTargetID)
	if err != nil {
		t.Fatalf("PathTo(%q) failed: %v", optionTestTargetID, err)
	}

	assertPathEqual(t, path, []string{optionTestSourceID, optionTestMiddleID, optionTestTargetID})
}

// TestApplyOptions_NilOption verifies nil-option rejection through the canonical
// public assembly path.
//
// Implementation:
//   - Stage 1: Build a valid weighted graph with a valid source.
//   - Stage 2: Pass a nil Option to Dijkstra.
//   - Stage 3: Assert ErrNilOption and nil result publication.
//
// Behavior highlights:
//   - Nil options are explicit errors.
//   - Panic-based handling is forbidden.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure on setup, sentinel, or publication mismatch.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Valid graph/source inputs ensure option assembly is reached.
//
// AI-Hints:
//   - Never replace this with a panic expectation.
func TestApplyOptions_NilOption(t *testing.T) {
	graph, err := core.NewGraph(core.WithWeighted())
	if err != nil {
		t.Fatalf("NewGraph(WithWeighted) failed: %v", err)
	}

	if err = graph.AddVertex(optionTestSourceID); err != nil {
		t.Fatalf("AddVertex(%q) failed: %v", optionTestSourceID, err)
	}

	var nilOption dijkstra.Option

	result, err := dijkstra.Dijkstra(graph, optionTestSourceID, nilOption)

	mustNilState(t, result, true, "Dijkstra result after nil option")
	mustErrorIs(t, err, dijkstra.ErrNilOption)
}

// TestApplyOptions_RejectsInvalidCustomState verifies that caller-defined options
// cannot bypass the canonical finalized-state validation barrier.
//
// Implementation:
//   - Stage 1: Build a valid weighted graph with a valid source.
//   - Stage 2: Define custom options that return nil after publishing invalid state.
//   - Stage 3: Invoke Dijkstra for every invalid custom state.
//   - Stage 4: Assert the correct sentinel and nil result publication.
//
// Behavior highlights:
//   - Publicly constructible Option values are treated as untrusted policy input.
//   - Built-in constructors are not the only numeric safety boundary.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure on setup, classification, or publication mismatch.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(c), Space O(1), where c is the fixed number of cases.
//
// Notes:
//   - This is a regression anchor for final option-state validation.
//
// AI-Hints:
//   - Do not remove validateOptions from applyOptions while this public Option
//     extension surface exists.
func TestApplyOptions_RejectsInvalidCustomState(t *testing.T) {
	graph, err := core.NewGraph(core.WithWeighted())
	if err != nil {
		t.Fatalf("NewGraph(WithWeighted) failed: %v", err)
	}

	if err = graph.AddVertex(optionTestSourceID); err != nil {
		t.Fatalf("AddVertex(%q) failed: %v", optionTestSourceID, err)
	}

	cases := []struct {
		name   string
		option dijkstra.Option
		target error
	}{
		{
			name: "NaN MaxDistance",
			option: func(options *dijkstra.Options) error {
				options.MaxDistance = math.NaN()
				return nil
			},
			target: dijkstra.ErrBadMaxDistance,
		},
		{
			name: "negative infinity MaxDistance",
			option: func(options *dijkstra.Options) error {
				options.MaxDistance = math.Inf(-1)
				return nil
			},
			target: dijkstra.ErrBadMaxDistance,
		},
		{
			name: "zero InfEdgeThreshold",
			option: func(options *dijkstra.Options) error {
				options.InfEdgeThreshold = optionTestZeroValue
				return nil
			},
			target: dijkstra.ErrBadInfEdgeThreshold,
		},
		{
			name: "NaN InfEdgeThreshold",
			option: func(options *dijkstra.Options) error {
				options.InfEdgeThreshold = math.NaN()
				return nil
			},
			target: dijkstra.ErrBadInfEdgeThreshold,
		},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			result, runErr := dijkstra.Dijkstra(graph, optionTestSourceID, testCase.option)

			mustNilState(t, result, true, "Dijkstra result after invalid custom option")
			mustErrorIs(t, runErr, testCase.target)
		})
	}
}

// TestWithMaxDistance_RejectsNegative verifies that finite negative distance cutoffs
// are rejected by the option contract.
//
// Implementation:
//   - Stage 1: Attempt to assemble a configuration with a negative MaxDistance.
//   - Stage 2: Assert ErrBadMaxDistance through the sentinel protocol.
//
// Behavior highlights:
//   - Negative finite cutoffs are invalid.
//   - The option must fail explicitly instead of clamping or panicking.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure if the sentinel protocol check fails.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Zero remains valid and is not part of this rejection case.
//
// AI-Hints:
//   - Keep finite negative rejection separate from NaN and -Inf coverage.
func TestWithMaxDistance_RejectsNegative(t *testing.T) {
	config := dijkstra.DefaultOptions()

	err := dijkstra.WithMaxDistance(optionTestNegativeValue)(&config)

	mustErrorIs(t, err, dijkstra.ErrBadMaxDistance)

	if !math.IsInf(config.MaxDistance, 1) {
		t.Fatalf("MaxDistance mutated after rejection: got=%v want=+Inf", config.MaxDistance)
	}
}

// TestWithMaxDistance_RejectsNaN verifies that NaN is rejected by the MaxDistance
// option contract.
//
// Implementation:
//   - Stage 1: Attempt to assemble a configuration with MaxDistance set to NaN.
//   - Stage 2: Assert ErrBadMaxDistance through the sentinel protocol.
//
// Behavior highlights:
//   - NaN is invalid configuration data.
//   - The option must fail explicitly instead of silently normalizing.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure if the sentinel protocol check fails.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This test covers a distinct invalid numeric class from finite negative values.
//
// AI-Hints:
//   - Do not remove NaN coverage; it protects the numeric-policy contract.
func TestWithMaxDistance_RejectsNaN(t *testing.T) {
	config := dijkstra.DefaultOptions()

	err := dijkstra.WithMaxDistance(math.NaN())(&config)

	mustErrorIs(t, err, dijkstra.ErrBadMaxDistance)

	if !math.IsInf(config.MaxDistance, 1) {
		t.Fatalf("MaxDistance mutated after rejection: got=%v want=+Inf", config.MaxDistance)
	}
}

// TestWithMaxDistance_RejectsNegativeInfinity verifies that negative infinity is rejected
// explicitly by the MaxDistance option contract.
//
// Implementation:
//   - Stage 1: Attempt to assemble a configuration with MaxDistance set to -Inf.
//   - Stage 2: Assert ErrBadMaxDistance through the sentinel protocol.
//
// Behavior highlights:
//   - -Inf is invalid configuration data.
//   - This test covers a distinct explicit branch in the production code.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure if the sentinel protocol check fails.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This is a live validation branch and should remain covered.
//
// AI-Hints:
//   - Keep negative-infinity coverage separate from finite negative and NaN coverage.
func TestWithMaxDistance_RejectsNegativeInfinity(t *testing.T) {
	config := dijkstra.DefaultOptions()

	err := dijkstra.WithMaxDistance(math.Inf(-1))(&config)

	mustErrorIs(t, err, dijkstra.ErrBadMaxDistance)

	if !math.IsInf(config.MaxDistance, 1) {
		t.Fatalf("MaxDistance mutated after rejection: got=%v want=+Inf", config.MaxDistance)
	}
}

// TestWithMaxDistance_AcceptsPositiveInfinity verifies that +Inf is accepted as the
// canonical “no cutoff” distance policy.
//
// Implementation:
//   - Stage 1: Assemble a configuration with MaxDistance set to +Inf.
//   - Stage 2: Assert successful option assembly.
//   - Stage 3: Assert the finalized value remains +Inf.
//
// Behavior highlights:
//   - +Inf is valid configuration, not an error.
//   - The option stores the accepted value exactly.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure if option assembly unexpectedly fails.
//   - Fatal test failure if the stored value is not +Inf.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This is the canonical no-cutoff mode.
//
// AI-Hints:
//   - Do not weaken this by checking only “no error”; the stored +Inf matters too.
func TestWithMaxDistance_AcceptsPositiveInfinity(t *testing.T) {
	config := dijkstra.DefaultOptions()

	err := dijkstra.WithMaxDistance(math.Inf(1))(&config)
	if err != nil {
		t.Fatalf("WithMaxDistance(+Inf) failed: %v", err)
	}

	if !math.IsInf(config.MaxDistance, 1) {
		t.Fatalf("MaxDistance: got=%v want=+Inf", config.MaxDistance)
	}
}

// TestWithMaxDistance_AcceptsZero verifies the inclusive zero boundary.
func TestWithMaxDistance_AcceptsZero(t *testing.T) {
	config := dijkstra.DefaultOptions()

	err := dijkstra.WithMaxDistance(optionTestZeroValue)(&config)
	if err != nil {
		t.Fatalf("WithMaxDistance(0) failed: %v", err)
	}

	mustEqualFloat64(
		t,
		config.MaxDistance,
		optionTestZeroValue,
		"MaxDistance: got=%v want=%v",
		config.MaxDistance,
		optionTestZeroValue,
	)
}

// TestWithInfEdgeThreshold_RejectsZero verifies that a zero wall threshold is rejected
// because it would incorrectly classify zero-weight edges as impassable.
//
// Implementation:
//   - Stage 1: Attempt to assemble a configuration with InfEdgeThreshold set to zero.
//   - Stage 2: Assert ErrBadInfEdgeThreshold through the sentinel protocol.
//
// Behavior highlights:
//   - Zero is invalid for the edge-wall threshold.
//   - The option must fail explicitly instead of silently adjusting the value.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure if the sentinel protocol check fails.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Positive finite thresholds remain valid and are not part of this rejection case.
//
// AI-Hints:
//   - Keep zero-threshold rejection explicit; it protects zero-weight edge semantics.
func TestWithInfEdgeThreshold_RejectsZero(t *testing.T) {
	config := dijkstra.DefaultOptions()

	err := dijkstra.WithInfEdgeThreshold(optionTestZeroValue)(&config)

	mustErrorIs(t, err, dijkstra.ErrBadInfEdgeThreshold)

	if !math.IsInf(config.InfEdgeThreshold, 1) {
		t.Fatalf("InfEdgeThreshold mutated after rejection: got=%v want=+Inf", config.InfEdgeThreshold)
	}
}

// TestWithInfEdgeThreshold_RejectsNegative verifies finite-negative rejection.
func TestWithInfEdgeThreshold_RejectsNegative(t *testing.T) {
	config := dijkstra.DefaultOptions()

	err := dijkstra.WithInfEdgeThreshold(optionTestNegativeValue)(&config)

	mustErrorIs(t, err, dijkstra.ErrBadInfEdgeThreshold)

	if !math.IsInf(config.InfEdgeThreshold, 1) {
		t.Fatalf("InfEdgeThreshold mutated after rejection: got=%v want=+Inf", config.InfEdgeThreshold)
	}
}

// TestWithInfEdgeThreshold_RejectsNaN verifies that NaN is rejected by the edge-wall
// threshold option contract.
//
// Implementation:
//   - Stage 1: Attempt to assemble a configuration with InfEdgeThreshold set to NaN.
//   - Stage 2: Assert ErrBadInfEdgeThreshold through the sentinel protocol.
//
// Behavior highlights:
//   - NaN is invalid configuration data.
//   - The option must fail explicitly instead of silently normalizing.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure if the sentinel protocol check fails.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This test covers a distinct invalid numeric class from zero and -Inf.
//
// AI-Hints:
//   - Do not remove NaN coverage; it protects the threshold numeric contract.
func TestWithInfEdgeThreshold_RejectsNaN(t *testing.T) {
	config := dijkstra.DefaultOptions()

	err := dijkstra.WithInfEdgeThreshold(math.NaN())(&config)

	mustErrorIs(t, err, dijkstra.ErrBadInfEdgeThreshold)

	if !math.IsInf(config.InfEdgeThreshold, 1) {
		t.Fatalf("InfEdgeThreshold mutated after rejection: got=%v want=+Inf", config.InfEdgeThreshold)
	}
}

// TestWithInfEdgeThreshold_AcceptsPositiveInfinity verifies that +Inf is accepted as the
// canonical “no finite edge is blocked by threshold” mode.
//
// Implementation:
//   - Stage 1: Assemble a configuration with InfEdgeThreshold set to +Inf.
//   - Stage 2: Assert successful option assembly.
//   - Stage 3: Assert the finalized value remains +Inf.
//
// Behavior highlights:
//   - +Inf is valid configuration, not an error.
//   - The option stores the accepted value exactly.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure if option assembly unexpectedly fails.
//   - Fatal test failure if the stored value is not +Inf.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This is the canonical no-wall-threshold mode.
//
// AI-Hints:
//   - Do not weaken this by checking only “no error”; the stored +Inf matters too.
func TestWithInfEdgeThreshold_AcceptsPositiveInfinity(t *testing.T) {
	config := dijkstra.DefaultOptions()

	err := dijkstra.WithInfEdgeThreshold(math.Inf(1))(&config)
	if err != nil {
		t.Fatalf("WithInfEdgeThreshold(+Inf) failed: %v", err)
	}

	if !math.IsInf(config.InfEdgeThreshold, 1) {
		t.Fatalf("InfEdgeThreshold: got=%v want=+Inf", config.InfEdgeThreshold)
	}
}

// TestWithInfEdgeThreshold_RejectsNegativeInfinity verifies that negative infinity is rejected
// explicitly by the edge-wall threshold option contract.
//
// Implementation:
//   - Stage 1: Attempt to assemble a configuration with InfEdgeThreshold set to -Inf.
//   - Stage 2: Assert ErrBadInfEdgeThreshold through the sentinel protocol.
//
// Behavior highlights:
//   - -Inf is invalid configuration data.
//   - This test covers a distinct explicit branch in the production code.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure if the sentinel protocol check fails.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This is a live validation branch and should remain covered.
//
// AI-Hints:
//   - Keep negative-infinity coverage separate from zero and NaN coverage.
func TestWithInfEdgeThreshold_RejectsNegativeInfinity(t *testing.T) {
	config := dijkstra.DefaultOptions()

	err := dijkstra.WithInfEdgeThreshold(math.Inf(-1))(&config)

	mustErrorIs(t, err, dijkstra.ErrBadInfEdgeThreshold)

	if !math.IsInf(config.InfEdgeThreshold, 1) {
		t.Fatalf("InfEdgeThreshold mutated after rejection: got=%v want=+Inf", config.InfEdgeThreshold)
	}
}

// TestWithPathTracking verifies that the path-tracking option enables predecessor
// tracking without perturbing the remaining default policy.
//
// Implementation:
//   - Stage 1: Assemble a configuration with WithPathTracking.
//   - Stage 2: Assert TrackPaths becomes true.
//   - Stage 3: Assert remaining policy fields stay at their documented defaults.
//
// Behavior highlights:
//   - Path tracking is an explicit contract-level request.
//   - The option must not silently alter distance or threshold policy.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure if option assembly unexpectedly fails.
//   - Fatal test failure on any policy mismatch.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This test isolates WithPathTracking from algorithm execution.
//
// AI-Hints:
//   - Keep path-tracking enablement narrow; do not let it mutate unrelated fields.
func TestWithPathTracking(t *testing.T) {
	config := dijkstra.DefaultOptions()

	err := dijkstra.WithPathTracking()(&config)
	if err != nil {
		t.Fatalf("WithPathTracking() failed: %v", err)
	}

	mustEqualBool(t, config.TrackPaths, true, "TrackPaths: got=%v want=true", config.TrackPaths)

	if !math.IsInf(config.MaxDistance, 1) {
		t.Fatalf("MaxDistance changed: got=%v want=+Inf", config.MaxDistance)
	}
	if !math.IsInf(config.InfEdgeThreshold, 1) {
		t.Fatalf("InfEdgeThreshold changed: got=%v want=+Inf", config.InfEdgeThreshold)
	}
}
