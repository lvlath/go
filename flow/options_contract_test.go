// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package flow_test verifies option resolution through the public API.
// No test-only production export of the private option accumulator is needed.
package flow_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/flow"
)

// TestDefaultOptionsReturnsIndependentDocumentedPolicy verifies every legacy default.
// The epsilon comparison is exact because this is a configuration value, not a computed result.
// Implementation:
//   - Stage 1: Obtain and inspect the complete default policy.
//   - Stage 2: Modify the returned value and obtain defaults again.
//
// Behavior highlights: Context is non-cancelable; limits and logging are disabled.
// Inputs: t is the test handle.
// Returns: None.
// Errors: Any missing or altered default fails the test.
// Determinism: No runtime-dependent default policy is expected.
// Complexity: O(1) time and space.
// Notes: A zero-value Options is not equivalent to DefaultOptions for epsilon.
// AI-Hints: Do not use an absolute tolerance equal to epsilon to check the default epsilon.
func TestDefaultOptionsReturnsIndependentDocumentedPolicy(t *testing.T) {
	first := flow.DefaultOptions()
	if first.Ctx == nil || first.Ctx.Err() != nil || first.Ctx.Done() != nil {
		t.Fatal("default context must be non-nil and non-cancelable")
	}
	if _, hasDeadline := first.Ctx.Deadline(); hasDeadline {
		t.Fatal("default context must not impose a deadline")
	}
	if first.Epsilon != flowTestDefaultEpsilon || first.Verbose || first.LevelRebuildInterval != 0 || first.MaxAugmentations != 0 {
		t.Fatalf("unexpected defaults: %+v", first)
	}
	// Modify the local value; no mutation may leak into another caller's defaults.
	first.Ctx = nil
	first.Epsilon = 0
	first.Verbose = true
	first.LevelRebuildInterval = 1
	first.MaxAugmentations = 1
	second := flow.DefaultOptions()
	if first.Ctx != nil || first.Epsilon != 0 || !first.Verbose || first.LevelRebuildInterval != 1 || first.MaxAugmentations != 1 {
		t.Fatal("obtaining defaults changed the first caller-owned value")
	}
	if second.Ctx == nil || second.Ctx.Err() != nil || second.Ctx.Done() != nil || second.Epsilon != flowTestDefaultEpsilon || second.Verbose || second.LevelRebuildInterval != 0 || second.MaxAugmentations != 0 {
		t.Fatalf("defaults were changed by a previous caller: %+v", second)
	}
}

// TestMaxFlowRejectsNilOptionBeforeGraphValidation verifies the assembly gate.
// A nil option must fail deterministically before graph inspection or allocation.
// Implementation:
//   - Stage 1: Supply a valid option followed by a nil option.
//   - Stage 2: Repeat with a nil graph to verify validation precedence.
//
// Behavior highlights: Both calls return ErrInvalidOptions without any result.
// Inputs: t is the test handle.
// Returns: None.
// Errors: Missing sentinel or non-nil result fails the test.
// Determinism: Caller option order controls the first validation failure.
// Complexity: O(K) option assembly, no solver work.
// AI-Hints: Do not replace nil options with defaults; that hides caller mistakes.
func TestMaxFlowRejectsNilOptionBeforeGraphValidation(t *testing.T) {
	g := buildFlowObservationNetwork(t)
	for _, tc := range []struct {
		name  string
		graph *core.Graph
	}{
		{name: "valid_graph", graph: g}, {name: "nil_graph"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			result, err := flow.MaxFlow(tc.graph, flowTestSource, flowTestSink, flow.WithAlgorithm(flow.AlgorithmDinic), nil)
			mustErrorIs(t, err, flow.ErrInvalidOptions, "nil option")
			if result != nil {
				t.Fatalf("nil option published a result: %+v", result)
			}
		})
	}
}

// TestMaxFlowRevalidatesExternallyResetOptionState protects final option assembly.
// A generic function can implement the public Option type without naming its private parameter.
// Implementation:
//   - Stage 1: Use Go type inference to assign a generic reset function to flow.Option.
//   - Stage 2: Invoke MaxFlow with the reset option, erasing the default algorithm.
//   - Stage 3: Require the resolver to reject the resulting invalid configuration.
//
// Behavior highlights: Tests a real language-level boundary without reflection or unsafe.
// Inputs: t is the test handle.
// Returns: None.
// Errors: ErrInvalidOptions and a nil result are required.
// Determinism: The reset produces the same zero configuration on every call.
// Complexity: O(1) option assembly; no graph traversal.
// Notes: The custom option is intentionally invalid; it is not a supported usage example.
// AI-Hints: Keep final validation even when built-in option constructors validate immediately.
func TestMaxFlowRevalidatesExternallyResetOptionState(t *testing.T) {
	var reset flow.Option = resetFlowOption
	g := buildFlowObservationNetwork(t)
	result, err := flow.MaxFlow(g, flowTestSource, flowTestSink, reset)
	mustErrorIs(t, err, flow.ErrInvalidOptions, "reset erased default algorithm")
	if result != nil {
		t.Fatalf("invalid final option state published a result: %+v", result)
	}
}

// TestMaxFlowOptionsUseLastValidWriterAndNormalizeNilContext verifies option composition.
// Repeated valid settings replace previous settings without silently retaining stale callbacks.
// Implementation:
//   - Stage 1: Supply conflicting valid algorithm, epsilon, context, budget, and observer options.
//   - Stage 2: Resolve the final policy to Ford-Fulkerson with no cutoff or limit.
//   - Stage 3: Check the selected algorithm, full result, and only the final observer.
//
// Behavior highlights: An explicitly nil context is normalized after option application.
// Inputs: t is the test handle.
// Returns: None.
// Errors: Unexpected solver errors or stale option effects fail the test.
// Determinism: Canceled context is replaced before the solver starts.
// Complexity: O(K) option resolution plus one fixed-fixture solver run.
// Notes: Last-writer-wins applies to valid options; invalid constructors fail immediately.
// AI-Hints: Do not skip context normalization or accumulate obsolete observers.
func TestMaxFlowOptionsUseLastValidWriterAndNormalizeNilContext(t *testing.T) {
	g := buildFlowObservationNetwork(t)
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	staleCalls := 0
	events := make([]flow.AugmentationEvent, 0, flowTestEventCount)
	result, err := flow.MaxFlow(g, flowTestSource, flowTestSink,
		flow.WithAlgorithm(flow.AlgorithmDinic), flow.WithAlgorithm(flow.AlgorithmFordFulkerson),
		flow.WithEpsilon(flowTestTotalValue), flow.WithEpsilon(0),
		flow.WithContext(canceled), flow.WithContext(nil),
		flow.WithMaxAugmentations(1), flow.WithMaxAugmentations(0),
		flow.WithObserver(func(context.Context, flow.AugmentationEvent) error {
			staleCalls++
			return nil
		}),
		flow.WithObserver(func(ctx context.Context, event flow.AugmentationEvent) error {
			if ctx == nil || ctx.Err() != nil || ctx.Done() != nil {
				t.Fatalf("nil context was not normalized: %v", ctx)
			}
			events = append(events, event)
			return nil
		}))
	mustNoError(t, err, "composed MaxFlow options")
	mustSuccessfulCertificate(t, g, result, flowTestSource, flowTestSink, flowTestTotalValue, 0, "last valid option wins")
	if result.Algorithm != flow.AlgorithmFordFulkerson || staleCalls != 0 {
		t.Fatalf("stale options: algorithm=%s old observer calls=%d", result.Algorithm, staleCalls)
	}
	mustFlowEvents(t, events, expectedFlowEvents(flow.AlgorithmFordFulkerson))
}

// TestMaxFlowOptionsDoNotHideInvalidSetterAndCanDisableObserver protects fail-fast composition.
// Invalid setters are errors even if a later option would otherwise replace their values.
// Implementation:
//   - Stage 1: Supply an invalid epsilon followed by a valid one.
//   - Stage 2: Verify a later nil observer disables a previously installed callback.
//
// Behavior highlights: Invalid configuration is never silently healed.
// Inputs: t is the test handle.
// Returns: None.
// Errors: Missing ErrInvalidEpsilon or unexpected observer activity fails the test.
// Determinism: No time-dependent inputs.
// Complexity: O(K) option assembly plus one fixed-fixture solver run.
// AI-Hints: Distinguish an invalid option from an explicitly disabled optional callback.
func TestMaxFlowOptionsDoNotHideInvalidSetterAndCanDisableObserver(t *testing.T) {
	g := buildFlowObservationNetwork(t)
	result, err := flow.MaxFlow(g, flowTestSource, flowTestSink, flow.WithEpsilon(math.Inf(1)), flow.WithEpsilon(0))
	mustErrorIs(t, err, flow.ErrInvalidEpsilon, "invalid first setter")
	if result != nil {
		t.Fatal("invalid option must not publish a result")
	}
	calls := 0
	result, err = flow.MaxFlow(g, flowTestSource, flowTestSink,
		flow.WithObserver(func(context.Context, flow.AugmentationEvent) error {
			calls++
			return nil
		}), flow.WithObserver(nil))
	mustNoError(t, err, "disabled observer")
	mustSuccessfulCertificate(t, g, result, flowTestSource, flowTestSink, flowTestTotalValue, flowTestDefaultEpsilon, "observer disabled")
	if calls != 0 || result.Algorithm != flow.AlgorithmDinic {
		t.Fatalf("default engine or disabled observer mismatch: algorithm=%s calls=%d", result.Algorithm, calls)
	}
}

// TestLegacyZeroOptionsPreserveExplicitZeroEpsilon verifies the legacy numeric distinction.
// A positive capacity below the default epsilon is retained only when epsilon is explicitly zero.
// Implementation:
//   - Stage 1: Build a single sub-threshold capacity arc.
//   - Stage 2: Compare DefaultOptions with the zero-value Options for each wrapper.
//
// Behavior highlights: Exact comparison avoids hiding the tiny flow behind test tolerance.
// Inputs: t is the test handle.
// Returns: None.
// Errors: A lost or incorrectly retained arc fails the test.
// Determinism: One exact power-of-two-scaled capacity.
// Complexity: Two O(1) solver runs per algorithm on the fixed graph.
// Notes: Nil context is normalized, but zero epsilon is a meaningful numeric policy.
// AI-Hints: Do not use mustEqualFloat here; its tolerance is larger than the fixture capacity.
func TestLegacyZeroOptionsPreserveExplicitZeroEpsilon(t *testing.T) {
	const tiny = flowTestDefaultEpsilon / 2
	g := mustGraph(t, core.WithDirected(true), core.WithWeighted())
	mustAddEdge(t, g, flowTestSource, flowTestSink, tiny)
	for _, tc := range allAlgorithms() {
		t.Run(tc.name, func(t *testing.T) {
			value, residual, err := tc.legacy(g, flowTestSource, flowTestSink, flow.DefaultOptions())
			mustNoError(t, err, "default epsilon")
			if value != 0 || residual == nil || residual.EdgeCount() != 0 {
				t.Fatalf("default epsilon retained a sub-threshold arc: flow=%g", value)
			}
			value, residual, err = tc.legacy(g, flowTestSource, flowTestSink, flow.Options{})
			mustNoError(t, err, "zero epsilon")
			if value != tiny || residual == nil || residual.EdgeCount() != 1 || residualCapacity(t, residual, flowTestSink, flowTestSource) != tiny {
				t.Fatalf("zero epsilon lost the exact tiny capacity: flow=%g want=%g", value, tiny)
			}
		})
	}
}
