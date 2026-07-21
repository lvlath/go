// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package flow_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/flow"
)

func TestMaxFlowRejectsNilGraph(t *testing.T) {
	result, err := flow.MaxFlow(nil, "S", "T")

	mustErrorIs(t, err, flow.ErrNilGraph, "nil graph")
	if result != nil {
		t.Fatalf("nil graph result: got %#v want nil", result)
	}
}

func TestMaxFlowRejectsEmptyTerminalWithCoreSentinel(t *testing.T) {
	g := mustGraph(t, core.WithDirected(true), core.WithWeighted())

	result, err := flow.MaxFlow(g, "", "T")

	mustErrorIs(t, err, flow.ErrEmptyTerminal, "empty terminal flow sentinel")
	mustErrorIs(t, err, core.ErrEmptyVertexID, "empty terminal core sentinel")
	if result != nil {
		t.Fatalf("empty terminal result: got %#v want nil", result)
	}
}

func TestMaxFlowRejectsSameTerminal(t *testing.T) {
	g := mustGraph(t, core.WithDirected(true), core.WithWeighted())
	mustAddVertex(t, g, "S")

	result, err := flow.MaxFlow(g, "S", "S")

	mustErrorIs(t, err, flow.ErrSameTerminal, "same source/sink")
	if result != nil {
		t.Fatalf("same terminal result: got %#v want nil", result)
	}
}

func TestMaxFlowRejectsMissingSourceAndSink(t *testing.T) {
	g := mustGraph(t, core.WithDirected(true), core.WithWeighted())
	mustAddVertex(t, g, "A")

	sourceResult, sourceErr := flow.MaxFlow(g, "X", "A")
	mustErrorIs(t, sourceErr, flow.ErrSourceNotFound, "missing source")
	if sourceResult != nil {
		t.Fatalf("missing source result: got %#v want nil", sourceResult)
	}

	sinkResult, sinkErr := flow.MaxFlow(g, "A", "Z")
	mustErrorIs(t, sinkErr, flow.ErrSinkNotFound, "missing sink")
	if sinkResult != nil {
		t.Fatalf("missing sink result: got %#v want nil", sinkResult)
	}
}

func TestMaxFlowRejectsUnweightedGraph(t *testing.T) {
	g := mustGraph(t, core.WithDirected(true))
	mustAddVertex(t, g, "S")
	mustAddVertex(t, g, "T")

	result, err := flow.MaxFlow(g, "S", "T")

	mustErrorIs(t, err, flow.ErrUnweightedGraph, "unweighted graph")
	mustErrorIs(t, err, core.ErrBadWeight, "unweighted graph core sentinel")
	if result != nil {
		t.Fatalf("unweighted result: got %#v want nil", result)
	}
}

func TestMaxFlowRejectsInvalidOptions(t *testing.T) {
	g := mustGraph(t, core.WithDirected(true), core.WithWeighted())
	mustAddEdge(t, g, "S", "T", 1)

	result, err := flow.MaxFlow(g, "S", "T", flow.WithAlgorithm(flow.Algorithm("unknown")))
	mustErrorIs(t, err, flow.ErrInvalidOptions, "unknown algorithm")
	if result != nil {
		t.Fatalf("unknown algorithm result: got %#v want nil", result)
	}

	result, err = flow.MaxFlow(g, "S", "T", flow.WithLevelRebuildInterval(-1))
	mustErrorIs(t, err, flow.ErrInvalidOptions, "negative level rebuild interval")
	if result != nil {
		t.Fatalf("negative rebuild result: got %#v want nil", result)
	}

	result, err = flow.MaxFlow(g, "S", "T", flow.WithMaxAugmentations(-1))
	mustErrorIs(t, err, flow.ErrInvalidOptions, "negative augmentation limit")
	if result != nil {
		t.Fatalf("negative augmentation limit result: got %#v want nil", result)
	}
}

func TestMaxFlowRejectsInvalidEpsilon(t *testing.T) {
	g := mustGraph(t, core.WithDirected(true), core.WithWeighted())
	mustAddEdge(t, g, "S", "T", 1)

	tests := []struct {
		name    string
		epsilon float64
	}{
		{name: "negative", epsilon: -1},
		{name: "nan", epsilon: math.NaN()},
		{name: "positive_inf", epsilon: math.Inf(1)},
		{name: "negative_inf", epsilon: math.Inf(-1)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := flow.MaxFlow(g, "S", "T", flow.WithEpsilon(tt.epsilon))

			mustErrorIs(t, err, flow.ErrInvalidEpsilon, "invalid epsilon")
			if result != nil {
				t.Fatalf("invalid epsilon result: got %#v want nil", result)
			}
		})
	}
}

func TestMaxFlowRejectsNegativeCapacity(t *testing.T) {
	g := mustGraph(t, core.WithDirected(true), core.WithWeighted())
	mustAddVertex(t, g, "S")
	mustAddVertex(t, g, "T")

	_, err := g.AddEdge("S", "T", -1)
	mustNoError(t, err, "AddEdge negative capacity into weighted graph")

	result, err := flow.MaxFlow(g, "S", "T")

	mustErrorIs(t, err, flow.ErrInvalidCapacity, "negative capacity invalid")
	mustErrorIs(t, err, flow.ErrNegativeCapacity, "negative capacity sentinel")
	if result != nil {
		t.Fatalf("negative capacity result: got %#v want nil", result)
	}
}

func TestCoreRejectsNaNInfWeightsBeforeFlow(t *testing.T) {
	tests := []struct {
		name     string
		capacity float64
	}{
		{name: "nan", capacity: math.NaN()},
		{name: "positive_inf", capacity: math.Inf(1)},
		{name: "negative_inf", capacity: math.Inf(-1)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := mustGraph(t, core.WithDirected(true), core.WithWeighted())
			mustAddVertex(t, g, "S")
			mustAddVertex(t, g, "T")

			_, err := g.AddEdge("S", "T", tt.capacity)

			mustErrorIs(t, err, core.ErrNaNInf, "core rejects NaN/Inf before flow")
		})
	}
}
