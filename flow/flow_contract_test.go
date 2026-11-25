// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package flow_test

import (
	"context"
	"errors"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/flow"
)

func TestMaxFlow_AllAlgorithms_EnterpriseBackboneProofCertificate(t *testing.T) {
	for _, tt := range allAlgorithms() {
		t.Run(tt.name, func(t *testing.T) {
			g := buildEnterpriseBackboneProofNetwork(t)

			result, err := flow.MaxFlow(
				g,
				"S",
				"T",
				flow.WithAlgorithm(tt.algorithm),
				flow.WithMaxAugmentations(256),
			)

			mustNoError(t, err, "MaxFlow("+string(tt.algorithm)+")")
			mustSuccessfulCertificate(t, g, result, "S", "T", 27, 1e-9, tt.name)
			mustCutCapacityEqualsFlow(t, g, result, 1e-9, tt.name+" explicit cut capacity")
		})
	}
}

func TestMaxFlow_AllAlgorithms_SingleEdgeSaturatesForwardAndPublishesReverse(t *testing.T) {
	for _, tt := range allAlgorithms() {
		t.Run(tt.name, func(t *testing.T) {
			g := mustGraph(t, core.WithDirected(true), core.WithWeighted())
			mustAddEdge(t, g, "S", "T", 7)

			result, err := flow.MaxFlow(
				g,
				"S",
				"T",
				flow.WithAlgorithm(tt.algorithm),
			)

			mustNoError(t, err, "MaxFlow single edge")
			mustSuccessfulCertificate(t, g, result, "S", "T", 7, 1e-9, tt.name)
			mustEqualFloat(t, residualCapacity(t, result.Residual, "S", "T"), 0, "saturated forward residual")
			mustEqualFloat(t, residualCapacity(t, result.Residual, "T", "S"), 7, "reverse residual carries flow")
		})
	}
}

func TestMaxFlow_AllAlgorithms_ParallelEdgesAggregateAcrossLayeredNetwork(t *testing.T) {
	for _, tt := range allAlgorithms() {
		t.Run(tt.name, func(t *testing.T) {
			g := mustGraph(t, core.WithDirected(true), core.WithWeighted(), core.WithMultiEdges())

			mustAddEdge(t, g, "S", "IngressA", 6)
			mustAddEdge(t, g, "S", "IngressA", 4)
			mustAddEdge(t, g, "S", "IngressB", 7)

			mustAddEdge(t, g, "IngressA", "ComputeC", 5)
			mustAddEdge(t, g, "IngressA", "ComputeC", 3)
			mustAddEdge(t, g, "IngressB", "ComputeC", 4)
			mustAddEdge(t, g, "IngressB", "ComputeD", 3)

			mustAddEdge(t, g, "ComputeC", "T", 10)
			mustAddEdge(t, g, "ComputeD", "T", 3)

			result, err := flow.MaxFlow(
				g,
				"S",
				"T",
				flow.WithAlgorithm(tt.algorithm),
				flow.WithMaxAugmentations(64),
			)

			mustNoError(t, err, "MaxFlow parallel layered network")
			mustSuccessfulCertificate(t, g, result, "S", "T", 13, 1e-9, tt.name)

			mustEqualFloat(t, residualCapacity(t, result.Residual, "T", "ComputeC"), 10, "reverse residual T->ComputeC")
			mustEqualFloat(t, residualCapacity(t, result.Residual, "T", "ComputeD"), 3, "reverse residual T->ComputeD")
		})
	}
}

func TestMaxFlow_AllAlgorithms_EpsilonFiltersOnlyEffectiveCapacityNetwork(t *testing.T) {
	const epsilon = 2.0

	for _, tt := range allAlgorithms() {
		t.Run(tt.name, func(t *testing.T) {
			g := mustGraph(t, core.WithDirected(true), core.WithWeighted())

			mustAddEdge(t, g, "S", "TinyA", 1)
			mustAddEdge(t, g, "TinyA", "T", 100)

			mustAddEdge(t, g, "S", "MainA", 3)
			mustAddEdge(t, g, "MainA", "T", 3)

			mustAddEdge(t, g, "S", "MainB", 4)
			mustAddEdge(t, g, "MainB", "T", 1)

			result, err := flow.MaxFlow(
				g,
				"S",
				"T",
				flow.WithAlgorithm(tt.algorithm),
				flow.WithEpsilon(epsilon),
				flow.WithMaxAugmentations(64),
			)

			mustNoError(t, err, "MaxFlow epsilon-filtered network")
			mustSuccessfulCertificate(t, g, result, "S", "T", 3, epsilon, tt.name)

			mustEqualFloat(t, residualCapacity(t, result.Residual, "T", "MainA"), 3, "reverse residual through effective path")
			mustEqualFloat(t, residualCapacity(t, result.Residual, "T", "TinyA"), 0, "no reverse residual through filtered path")
		})
	}
}

func TestMaxFlow_AllAlgorithms_LoopsIgnoredInMultiLayerNetwork(t *testing.T) {
	for _, tt := range allAlgorithms() {
		t.Run(tt.name, func(t *testing.T) {
			g := mustGraph(t, core.WithDirected(true), core.WithWeighted(), core.WithLoops())

			mustAddEdge(t, g, "S", "S", 999)
			mustAddEdge(t, g, "A", "A", 999)
			mustAddEdge(t, g, "T", "T", 999)

			mustAddEdge(t, g, "S", "A", 6)
			mustAddEdge(t, g, "S", "B", 5)
			mustAddEdge(t, g, "A", "C", 4)
			mustAddEdge(t, g, "A", "D", 2)
			mustAddEdge(t, g, "B", "C", 3)
			mustAddEdge(t, g, "B", "D", 2)
			mustAddEdge(t, g, "C", "T", 5)
			mustAddEdge(t, g, "D", "T", 4)

			result, err := flow.MaxFlow(
				g,
				"S",
				"T",
				flow.WithAlgorithm(tt.algorithm),
				flow.WithMaxAugmentations(64),
			)

			mustNoError(t, err, "MaxFlow loops ignored")
			mustSuccessfulCertificate(t, g, result, "S", "T", 9, 1e-9, tt.name)
		})
	}
}

func TestMaxFlow_AllAlgorithms_UndirectedEdgeIsTwoDirectedCapacityArcs(t *testing.T) {
	for _, tt := range allAlgorithms() {
		t.Run(tt.name, func(t *testing.T) {
			g := mustGraph(t, core.WithDirected(false), core.WithWeighted())
			mustAddEdge(t, g, "A", "B", 4)

			result, err := flow.MaxFlow(
				g,
				"B",
				"A",
				flow.WithAlgorithm(tt.algorithm),
			)

			mustNoError(t, err, "MaxFlow undirected reverse direction")
			mustSuccessfulCertificate(t, g, result, "B", "A", 4, 1e-9, tt.name)
		})
	}
}

func TestMaxFlow_Dinic_LevelRebuildIntervalDoesNotChangeValueOrCertificate(t *testing.T) {
	g := mustGraph(t, core.WithDirected(true), core.WithWeighted())

	mustAddEdge(t, g, "S", "A", 2)
	mustAddEdge(t, g, "S", "B", 1)
	mustAddEdge(t, g, "A", "C", 1)
	mustAddEdge(t, g, "B", "C", 1)
	mustAddEdge(t, g, "C", "T", 2)

	defaultResult, err := flow.MaxFlow(g, "S", "T", flow.WithAlgorithm(flow.AlgorithmDinic))
	mustNoError(t, err, "Dinic default")

	rebuildResult, err := flow.MaxFlow(
		g,
		"S",
		"T",
		flow.WithAlgorithm(flow.AlgorithmDinic),
		flow.WithLevelRebuildInterval(2),
	)
	mustNoError(t, err, "Dinic rebuild interval")

	mustEqualFloat(t, rebuildResult.Value, defaultResult.Value, "Dinic rebuild value")
	mustSuccessfulCertificate(t, g, rebuildResult, "S", "T", 2, 1e-9, "Dinic rebuild certificate")
}

func TestMaxFlow_FordFulkersonAugmentationLimitReturnsPartialResult(t *testing.T) {
	g := mustGraph(t, core.WithDirected(true), core.WithWeighted())

	mustAddEdge(t, g, "S", "A", 1)
	mustAddEdge(t, g, "A", "T", 1)
	mustAddEdge(t, g, "S", "B", 1)
	mustAddEdge(t, g, "B", "T", 1)

	result, err := flow.MaxFlow(
		g,
		"S",
		"T",
		flow.WithAlgorithm(flow.AlgorithmFordFulkerson),
		flow.WithMaxAugmentations(1),
	)

	mustErrorIs(t, err, flow.ErrAugmentationLimit, "augmentation limit")
	if result == nil {
		t.Fatalf("augmentation limit must return partial result")
	}
	mustEqualBool(t, result.Partial, true, "augmentation limit partial")
	mustEqualFloat(t, result.Value, 1, "augmentation limit pushed value")
}

// TestMaxFlow_ContextCanceledBeforeRunReturnsNoResult verifies the pre-kernel cancellation law.
// A canceled context prevents residual construction and therefore cannot publish partial flow.
// Implementation: Cancel before MaxFlow, then verify the original context sentinel and nil result.
// Behavior highlights: Distinguishes a rejected run from a partially executed run.
// Inputs: t is the test handle.
// Returns: None.
// Errors: Incorrect cancellation identity or any result publication fails the test.
// Determinism: Cancellation occurs synchronously before invocation.
// Complexity: O(1) cancellation detection after input validation.
// AI-Hints: Do not accept both nil and partial results for the same deterministic cancellation point.
func TestMaxFlow_ContextCanceledBeforeRunReturnsNoResult(t *testing.T) {
	g := buildEnterpriseBackboneProofNetwork(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	result, err := flow.MaxFlow(g, flowTestSource, flowTestSink,
		flow.WithContext(ctx), flow.WithAlgorithm(flow.AlgorithmDinic))
	mustErrorIs(t, err, context.Canceled, "canceled MaxFlow")
	if result != nil {
		t.Fatalf("pre-run cancellation published a result: %+v", result)
	}
}

// TestMaxFlow_ObserverFailureReturnsPartialResult verifies observer rejection across all engines.
// The callback fails after the first committed augmentation, preserving both error identities.
// Implementation:
//   - Stage 1: Run the two-route fixture with a rejecting observer.
//   - Stage 2: Verify the first event and every partial-result field.
//
// Behavior highlights: No later augmentation or optimality certificate may be published.
// Inputs: t is the test handle.
// Returns: None.
// Errors: ErrObserverFailure and the original callback error must both survive wrapping.
// Determinism: The callback stops the run at a fixed augmentation index.
// Complexity: One committed augmentation and its search per engine.
// AI-Hints: Do not replace a callback error with an unrelated cancellation error.
func TestMaxFlow_ObserverFailureReturnsPartialResult(t *testing.T) {
	observerErr := errors.New("observer stopped after first event")
	for _, tc := range allAlgorithms() {
		t.Run(tc.name, func(t *testing.T) {
			g := buildFlowObservationNetwork(t)
			events := make([]flow.AugmentationEvent, 0, 1)
			result, err := flow.MaxFlow(g, flowTestSource, flowTestSink,
				flow.WithAlgorithm(tc.algorithm),
				flow.WithObserver(func(_ context.Context, event flow.AugmentationEvent) error {
					events = append(events, event)
					return observerErr
				}))
			mustErrorIs(t, err, flow.ErrObserverFailure, "observer failure sentinel")
			mustErrorIs(t, err, observerErr, "observer original error")
			want := expectedFlowEvents(tc.algorithm)[:1]
			mustFlowEvents(t, events, want)
			mustPartialFlowResult(t, result, want[0])
		})
	}
}
