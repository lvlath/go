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

// TestValidateCapacityDefensiveNumericBoundaries verifies the internal edge guard through a test-build bridge.
// The fixture uses standalone core.Edge values, never corrupted graph-owned pointers.
// Implementation:
//   - Stage 1: Define nil, non-finite, and adjacent floating-point threshold cases.
//   - Stage 2: Invoke the actual capacity validator through its test-only alias.
//   - Stage 3: Compare the exact normalized value and every required sentinel.
//
// Behavior highlights: Tests both sides of -epsilon and +epsilon without approximate comparisons.
// Inputs: t is the test handle.
// Returns: None.
// Errors: A wrong normalized capacity, missing sentinel, or edge mutation fails the test.
// Determinism: math.Nextafter selects the immediately adjacent representable values.
// Complexity: O(K) time for K constant-size validation cases, O(1) per case.
// Notes: core.AddEdge correctly rejects NaN/Inf before these defensive checks are reachable.
// AI-Hints: Never bypass core's public weight invariant just to increase flow coverage.
func TestValidateCapacityDefensiveNumericBoundaries(t *testing.T) {
	const epsilon = 0.25
	above := math.Nextafter(epsilon, math.Inf(1))
	belowNegative := math.Nextafter(-epsilon, math.Inf(-1))
	cases := []struct {
		name    string
		nilEdge bool
		weight  float64
		want    float64
		errs    []error
	}{
		{name: "nil_edge", nilEdge: true, errs: []error{flow.ErrInvalidCapacity}},
		{name: "nan", weight: math.NaN(), errs: []error{flow.ErrInvalidCapacity, flow.ErrNaNInf}},
		{name: "positive_inf", weight: math.Inf(1), errs: []error{flow.ErrInvalidCapacity, flow.ErrNaNInf}},
		{name: "negative_inf", weight: math.Inf(-1), errs: []error{flow.ErrInvalidCapacity, flow.ErrNaNInf}},
		{name: "below_negative_threshold", weight: belowNegative, errs: []error{flow.ErrInvalidCapacity, flow.ErrNegativeCapacity}},
		{name: "negative_threshold", weight: -epsilon},
		{name: "negative_tolerated", weight: -epsilon / 2},
		{name: "zero"},
		{name: "below_positive_threshold", weight: math.Nextafter(epsilon, 0)},
		{name: "positive_threshold", weight: epsilon},
		{name: "above_positive_threshold", weight: above, want: above},
		{name: "largest_finite", weight: math.MaxFloat64, want: math.MaxFloat64},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var edge *core.Edge
			if !tc.nilEdge {
				edge = &core.Edge{ID: "capacity-fixture", From: flowTestSource, To: flowTestSink, Directed: true, Weight: tc.weight}
			}
			value, err := flow.ValidateCapacityForTest(edge, epsilon)
			if len(tc.errs) == 0 {
				mustNoError(t, err, "valid capacity")
			} else {
				for _, target := range tc.errs {
					mustErrorIs(t, err, target, "capacity numeric policy")
				}
			}
			if value != tc.want {
				t.Fatalf("normalized capacity: got=%g want=%g", value, tc.want)
			}
			// Bit comparison also detects a changed NaN payload, for which equality is undefined.
			if edge != nil && math.Float64bits(edge.Weight) != math.Float64bits(tc.weight) {
				t.Fatal("capacity validation mutated its input")
			}
		})
	}
}
