// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package flow_test verifies public maximum-flow facades and result publication.
// This file exercises compatibility projection, observer delivery, and error propagation.
package flow_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/flow"
)

// TestLegacyFlowWrappersPreserveCanonicalResult verifies all compatibility facades.
// Each wrapper must preserve the canonical optimum and complete residual topology.
// Implementation:
//   - Stage 1: Construct the shared enterprise backbone and snapshot its topology.
//   - Stage 2: Execute the canonical API and the corresponding legacy wrapper.
//   - Stage 3: Verify the independent optimum, cut, conservation, and residual equality.
//
// Behavior highlights: Exercises all three wrappers using the same weighted network.
// Inputs: t is the test handle.
// Returns: None.
// Errors: Unexpected errors or any topology/value mismatch fail the subtest.
// Determinism: The same algorithm and options must publish identical residual inventories.
// Complexity: Two solver executions per algorithm plus sparse certificate verification.
// Notes: The canonical result retains metadata that the tuple API cannot express.
// AI-Hints: Comparing only wrapper and canonical scalar values is not an independent oracle.
func TestLegacyFlowWrappersPreserveCanonicalResult(t *testing.T) {
	for _, tc := range allAlgorithms() {
		t.Run(tc.name, func(t *testing.T) {
			// Use an independently bounded, non-trivial network rather than a single arc.
			g := buildEnterpriseBackboneProofNetwork(t)
			before := g.Clone()
			legacy := flow.DefaultOptions()
			legacy.Ctx = nil // A nil legacy context must resolve to Background.
			legacy.LevelRebuildInterval = 1

			canonical, err := flow.MaxFlow(g, flowTestSource, flowTestSink,
				flow.WithAlgorithm(tc.algorithm), flow.WithContext(nil),
				flow.WithEpsilon(legacy.Epsilon), flow.WithLevelRebuildInterval(legacy.LevelRebuildInterval))
			mustNoError(t, err, "canonical MaxFlow")
			mustSuccessfulCertificate(t, g, canonical, flowTestSource, flowTestSink,
				flowTestBackboneValue, legacy.Epsilon, tc.name)
			mustExactFlowBalance(t, g, canonical.Residual, flowTestSource, flowTestSink, flowTestBackboneValue, legacy.Epsilon)
			if canonical.Algorithm != tc.algorithm {
				t.Fatalf("algorithm: got %s want %s", canonical.Algorithm, tc.algorithm)
			}

			// Project the same policy through the compatibility wrapper.
			value, residual, err := tc.legacy(g, flowTestSource, flowTestSink, legacy)
			mustNoError(t, err, "legacy max-flow")
			mustEqualFloat(t, value, flowTestBackboneValue, "legacy value")
			mustEqualFlowGraphs(t, residual, canonical.Residual, "legacy residual")
			mustEqualFlowGraphs(t, g, before, "input graph remains unchanged")
		})
	}
}

// TestLegacyFlowWrappersRejectInvalidInputs verifies the legacy error projection.
// Validation and pre-run cancellation must return no residual and zero flow.
// Implementation:
//   - Stage 1: Prepare a valid graph and deterministic canceled context.
//   - Stage 2: Exercise invalid graph, terminal, and legacy option policies.
//   - Stage 3: Check all error identities and the absence of a published result.
//
// Behavior highlights: Invalid numeric policies are not silently replaced by defaults.
// Inputs: t is the test handle.
// Returns: None.
// Errors: Missing sentinel or nonzero failure output fails the subtest.
// Determinism: Cancellation occurs before invocation; no sleeping or deadlines race the solver.
// Complexity: O(K) invocations of fail-fast validation per engine for K cases.
// Notes: Each legacy Options value is copied before modification.
// AI-Hints: Do not require a partial result when the kernel has not started.
func TestLegacyFlowWrappersRejectInvalidInputs(t *testing.T) {
	g := buildFlowObservationNetwork(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	defaults := flow.DefaultOptions()
	badEpsilon, badInterval, badLimit, canceled := defaults, defaults, defaults, defaults
	badEpsilon.Epsilon = math.NaN()
	badInterval.LevelRebuildInterval = -1
	badLimit.MaxAugmentations = -1
	canceled.Ctx = ctx

	cases := []struct {
		name   string
		graph  *core.Graph
		source string
		opts   flow.Options
		want   []error
	}{
		{name: "nil_graph", opts: defaults, source: flowTestSource, want: []error{flow.ErrNilGraph}},
		{name: "empty_source", graph: g, opts: defaults, want: []error{flow.ErrEmptyTerminal, core.ErrEmptyVertexID}},
		{name: "epsilon", graph: g, source: flowTestSource, opts: badEpsilon, want: []error{flow.ErrInvalidEpsilon}},
		{name: "rebuild_interval", graph: g, source: flowTestSource, opts: badInterval, want: []error{flow.ErrInvalidOptions}},
		{name: "augmentation_limit", graph: g, source: flowTestSource, opts: badLimit, want: []error{flow.ErrInvalidOptions}},
		{name: "canceled", graph: g, source: flowTestSource, opts: canceled, want: []error{context.Canceled}},
	}
	for _, algorithm := range allAlgorithms() {
		t.Run(algorithm.name, func(t *testing.T) {
			for _, tc := range cases {
				t.Run(tc.name, func(t *testing.T) {
					value, residual, err := algorithm.legacy(tc.graph, tc.source, flowTestSink, tc.opts)
					for _, target := range tc.want {
						mustErrorIs(t, err, target, "legacy error projection")
					}
					if value != 0 || residual != nil {
						t.Fatalf("invalid legacy result: value=%g residual=%p", value, residual)
					}
				})
			}
		})
	}
}

// TestLegacyFordFulkersonPreservesPartialValueAndExactLimit verifies the budget law.
// The tuple must retain committed flow on interruption and succeed at the exact optimum.
// Implementation:
//   - Stage 1: Configure the two-route fixture with a one-augmentation budget.
//   - Stage 2: Require the first DFS flow and ErrAugmentationLimit without a residual.
//   - Stage 3: Increase the budget to two and verify the complete result.
//
// Behavior highlights: The limit counts successful augmentations, not path searches.
// Inputs: t is the test handle.
// Returns: None.
// Errors: Incorrect error classification or flow value fails the test.
// Determinism: The longer branch is selected first by LIFO DFS.
// Complexity: Two Ford-Fulkerson runs on a fixed graph.
// Notes: Only Ford-Fulkerson currently implements the augmentation budget.
// AI-Hints: Reaching the budget exactly must not be confused with exceeding it.
func TestLegacyFordFulkersonPreservesPartialValueAndExactLimit(t *testing.T) {
	g := buildFlowObservationNetwork(t)
	legacy := flow.DefaultOptions()
	legacy.MaxAugmentations = 1
	value, residual, err := flow.FordFulkerson(g, flowTestSource, flowTestSink, legacy)
	mustErrorIs(t, err, flow.ErrAugmentationLimit, "legacy partial run")
	mustEqualFloat(t, value, flowTestLongValue, "committed DFS flow")
	if residual != nil {
		t.Fatal("legacy partial run published a residual certificate")
	}

	// A search after the last permitted push is still needed to prove optimality.
	legacy.MaxAugmentations = flowTestEventCount
	value, residual, err = flow.FordFulkerson(g, flowTestSource, flowTestSink, legacy)
	mustNoError(t, err, "legacy exact-limit run")
	mustEqualFloat(t, value, flowTestTotalValue, "exact-limit optimum")
	mustNoAugmentingPath(t, residual, flowTestSource, flowTestSink, legacy.Epsilon)
	mustExactFlowBalance(t, g, residual, flowTestSource, flowTestSink, value, legacy.Epsilon)
}

// TestLegacyFlowVerbosePublishesDeterministicAugmentations verifies the logging bridge.
// The legacy flag must emit one diagnostic for every committed augmentation.
// Implementation:
//   - Stage 1: Prepare independently expected event payloads for each engine.
//   - Stage 2: Capture stdout with Verbose enabled and validate the full run.
//   - Stage 3: Repeat with Verbose disabled and require silent execution.
//
// Behavior highlights: Dinic logs an empty path representation by its existing contract.
// Inputs: t is the test handle; this test must remain sequential.
// Returns: None.
// Errors: Solver errors, missing events, or unwanted output fail the test.
// Determinism: Expected strings follow the fixed event order, not map order.
// Complexity: Two solver runs per engine and O(B) diagnostic capture for B bytes.
// Notes: Exact text comparison protects the existing legacy diagnostic format, not error classification.
// AI-Hints: Do not run this test in parallel because stdout is process-wide.
func TestLegacyFlowVerbosePublishesDeterministicAugmentations(t *testing.T) {
	for _, tc := range allAlgorithms() {
		t.Run(tc.name, func(t *testing.T) {
			g := buildFlowObservationNetwork(t)
			legacy := flow.DefaultOptions()
			legacy.Verbose = true
			want := ""
			for _, event := range expectedFlowEvents(tc.algorithm) {
				want += fmt.Sprintf("%s: augmenting path %v with flow %g, total %g\n",
					event.Algorithm, event.Path, event.Delta, event.Total)
			}
			got := captureFlowStdout(t, func() {
				value, residual, err := tc.legacy(g, flowTestSource, flowTestSink, legacy)
				mustNoError(t, err, "verbose legacy run")
				mustEqualFloat(t, value, flowTestTotalValue, "verbose flow value")
				mustNoAugmentingPath(t, residual, flowTestSource, flowTestSink, legacy.Epsilon)
			})
			if got != want {
				t.Fatalf("verbose output: got %q want %q", got, want)
			}
			legacy.Verbose = false
			quiet := captureFlowStdout(t, func() {
				value, residual, err := tc.legacy(g, flowTestSource, flowTestSink, legacy)
				mustNoError(t, err, "quiet legacy run")
				mustEqualFloat(t, value, flowTestTotalValue, "quiet flow value")
				mustNoAugmentingPath(t, residual, flowTestSource, flowTestSink, legacy.Epsilon)
			})
			if quiet != "" {
				t.Fatalf("unexpected quiet output: %q", quiet)
			}
		})
	}
}

// TestMaxFlowObserverReceivesCompleteOrderedEvents verifies successful notification.
// Each callback sees the caller context and the already committed augmentation.
// Implementation:
//   - Stage 1: Run each algorithm with a recording observer.
//   - Stage 2: Compare all events against the independent two-route oracle.
//   - Stage 3: Verify the complete result, cut, conservation, and augmentation count.
//
// Behavior highlights: Callback success must allow subsequent augmentations.
// Inputs: t is the test handle.
// Returns: None.
// Errors: Invalid context, events, or certificate fail the test.
// Determinism: Fixed paths and capacities; no timing assumptions.
// Complexity: One solver run per engine plus O(V+E) certificate checks (excluding core sorting).
// Notes: Observer paths are retained for inspection after the run.
// AI-Hints: Do not require Dinic to publish a path that its API does not currently expose.
func TestMaxFlowObserverReceivesCompleteOrderedEvents(t *testing.T) {
	for _, tc := range allAlgorithms() {
		t.Run(tc.name, func(t *testing.T) {
			g := buildFlowObservationNetwork(t)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			events := make([]flow.AugmentationEvent, 0, flowTestEventCount)
			result, err := flow.MaxFlow(g, flowTestSource, flowTestSink,
				flow.WithAlgorithm(tc.algorithm), flow.WithContext(ctx),
				flow.WithObserver(func(observed context.Context, event flow.AugmentationEvent) error {
					if observed != ctx {
						return errors.New("observer received a different context")
					}
					events = append(events, event)
					return nil
				}))
			mustNoError(t, err, "observed MaxFlow")
			mustFlowEvents(t, events, expectedFlowEvents(tc.algorithm))
			mustSuccessfulCertificate(t, g, result, flowTestSource, flowTestSink, flowTestTotalValue, flowTestDefaultEpsilon, tc.name)
			mustExactFlowBalance(t, g, result.Residual, flowTestSource, flowTestSink, flowTestTotalValue, flowTestDefaultEpsilon)
			if result.Augmentations != flowTestEventCount || result.Algorithm != tc.algorithm {
				t.Fatalf("result metadata: algorithm=%s augmentations=%d", result.Algorithm, result.Augmentations)
			}
		})
	}
}

// TestMaxFlowCancellationAfterFirstAugmentationPreservesPartial verifies post-commit cancellation.
// Cancellation is triggered synchronously from the observer, not by a timer.
// Implementation:
//   - Stage 1: Run each engine with a cancelable context and the two-route fixture.
//   - Stage 2: Cancel immediately after receiving the first committed event.
//   - Stage 3: Require context.Canceled and exactly that event's partial result.
//
// Behavior highlights: No successful certificate may escape an interrupted run.
// Inputs: t is the test handle.
// Returns: None.
// Errors: Incorrect cancellation identity or partial state fails the test.
// Determinism: Cancellation point is tied to an augmentation, not elapsed time.
// Complexity: One committed augmentation and the associated search per engine.
// Notes: The observer returns nil, so cancellation must come from the context.
// AI-Hints: Do not replace deterministic cancellation with time.Sleep or a short timeout.
func TestMaxFlowCancellationAfterFirstAugmentationPreservesPartial(t *testing.T) {
	for _, tc := range allAlgorithms() {
		t.Run(tc.name, func(t *testing.T) {
			g := buildFlowObservationNetwork(t)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			events := make([]flow.AugmentationEvent, 0, 1)
			result, err := flow.MaxFlow(g, flowTestSource, flowTestSink,
				flow.WithAlgorithm(tc.algorithm), flow.WithContext(ctx),
				flow.WithObserver(func(_ context.Context, event flow.AugmentationEvent) error {
					events = append(events, event)
					cancel()
					return nil
				}))
			mustErrorIs(t, err, context.Canceled, "post-commit cancellation")
			want := expectedFlowEvents(tc.algorithm)[:1]
			mustFlowEvents(t, events, want)
			mustPartialFlowResult(t, result, want[0])
		})
	}
}

// TestMaxFlowRejectsUnrepresentableResidualPublication verifies the numeric publication firewall.
// Finite antiparallel capacities can overflow a reverse residual even when the flow value is finite.
// Implementation:
//   - Stage 1: Create S->A->T with a finite MaxFloat64 bottleneck and an antiparallel T->A arc.
//   - Stage 2: Run each engine; augmenting A->T would make R(T,A)=2*MaxFloat64.
//   - Stage 3: Require the preserved core numeric error and no published result.
//
// Behavior highlights: No invalid weights are inserted into the input graph.
// Inputs: t is the test handle.
// Returns: None.
// Errors: core.ErrNaNInf must survive the current residual publication boundary.
// Determinism: One finite augmentation saturates the only source path.
// Complexity: O(V+E) on this fixed single-route fixture.
// Notes: This guards existing late rejection; early overflow detection would be a separate improvement.
// AI-Hints: Do not mutate core's read-only Edge pointers to manufacture invalid capacities.
func TestMaxFlowRejectsUnrepresentableResidualPublication(t *testing.T) {
	const capacity = math.MaxFloat64
	for _, tc := range allAlgorithms() {
		t.Run(tc.name, func(t *testing.T) {
			g := mustGraph(t, core.WithDirected(true), core.WithWeighted())
			mustAddEdge(t, g, flowTestSource, flowTestShortNode, capacity)
			mustAddEdge(t, g, flowTestShortNode, flowTestSink, capacity)
			mustAddEdge(t, g, flowTestSink, flowTestShortNode, capacity)
			before := g.Clone()
			result, err := flow.MaxFlow(g, flowTestSource, flowTestSink, flow.WithAlgorithm(tc.algorithm))
			mustErrorIs(t, err, core.ErrNaNInf, "residual numeric publication")
			if result != nil {
				t.Fatalf("invalid residual escaped finalization: %+v", result)
			}
			mustEqualFlowGraphs(t, g, before, "finite input remains unchanged")
		})
	}
}
