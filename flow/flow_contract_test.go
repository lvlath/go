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

func allAlgorithms() []struct {
	name      string
	algorithm flow.Algorithm
} {
	return []struct {
		name      string
		algorithm flow.Algorithm
	}{
		{name: "Dinic", algorithm: flow.AlgorithmDinic},
		{name: "EdmondsKarp", algorithm: flow.AlgorithmEdmondsKarp},
		{name: "FordFulkerson", algorithm: flow.AlgorithmFordFulkerson},
	}
}

func buildEnterpriseBackboneProofNetwork(t *testing.T) *core.Graph {
	t.Helper()

	g := mustGraph(t, core.WithDirected(true), core.WithWeighted())

	mustAddEdge(t, g, "S", "EdgeA", 10)
	mustAddEdge(t, g, "S", "EdgeB", 15)
	mustAddEdge(t, g, "S", "EdgeC", 10)

	mustAddEdge(t, g, "EdgeA", "CoreD", 9)
	mustAddEdge(t, g, "EdgeA", "CoreE", 4)
	mustAddEdge(t, g, "EdgeB", "CoreD", 5)
	mustAddEdge(t, g, "EdgeB", "CoreE", 8)
	mustAddEdge(t, g, "EdgeB", "CoreF", 4)
	mustAddEdge(t, g, "EdgeC", "CoreE", 6)
	mustAddEdge(t, g, "EdgeC", "CoreF", 8)

	mustAddEdge(t, g, "CoreD", "AggG", 10)
	mustAddEdge(t, g, "CoreE", "AggG", 7)
	mustAddEdge(t, g, "CoreE", "AggH", 8)
	mustAddEdge(t, g, "CoreF", "AggH", 10)

	mustAddEdge(t, g, "AggG", "T", 15)
	mustAddEdge(t, g, "AggH", "T", 12)

	return g
}

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

func TestMaxFlow_ContextCanceledBeforeRunReturnsPartialResult(t *testing.T) {
	g := buildEnterpriseBackboneProofNetwork(t)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	result, err := flow.MaxFlow(
		g,
		"S",
		"T",
		flow.WithContext(ctx),
		flow.WithAlgorithm(flow.AlgorithmDinic),
	)

	mustErrorIs(t, err, context.Canceled, "canceled MaxFlow")
	if result != nil {
		mustEqualBool(t, result.Partial, true, "canceled result is partial")
	}
}

func TestMaxFlow_ObserverFailureReturnsPartialResult(t *testing.T) {
	g := mustGraph(t, core.WithDirected(true), core.WithWeighted())

	mustAddEdge(t, g, "S", "A", 1)
	mustAddEdge(t, g, "A", "T", 1)
	mustAddEdge(t, g, "S", "B", 1)
	mustAddEdge(t, g, "B", "T", 1)

	observerErr := errors.New("observer stopped after first event")

	result, err := flow.MaxFlow(
		g,
		"S",
		"T",
		flow.WithAlgorithm(flow.AlgorithmEdmondsKarp),
		flow.WithObserver(func(context.Context, flow.AugmentationEvent) error {
			return observerErr
		}),
	)

	mustErrorIs(t, err, flow.ErrObserverFailure, "observer failure sentinel")
	mustErrorIs(t, err, observerErr, "observer original error")
	if result == nil {
		t.Fatalf("observer failure must return a partial result")
	}
	mustEqualBool(t, result.Partial, true, "observer failure partial")
	mustEqualFloat(t, result.Value, 1, "observer failure pushed value")
}
