// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package dfs_test

import (
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/dfs"
)

func TestDetectCycles_NilGraph(t *testing.T) {
	result, err := dfs.DetectCycles(nil)

	mustNilState(t, result, true, "DetectCycles(nil) result")
	mustErrorIs(t, err, dfs.ErrGraphNil)
}

func TestDetectCycles_DirectedNoCycle(t *testing.T) {
	g, _ := core.NewGraph(core.WithDirected(true))

	_, _ = g.AddEdge("A", "B", 0)
	_, _ = g.AddEdge("B", "C", 0)
	_, _ = g.AddEdge("B", "D", 0)
	_, _ = g.AddEdge("C", "G", 0)
	_, _ = g.AddEdge("D", "E", 0)
	_, _ = g.AddEdge("E", "F", 0)
	_, _ = g.AddEdge("F", "H", 0)
	_, _ = g.AddEdge("H", "I", 0)

	result, err := dfs.DetectCycles(g)
	mustNoError(t, err)

	mustCycleResult(t, result, false, nil)
}

func TestDetectCycles_SelfLoopAllowed(t *testing.T) {
	g, _ := core.NewGraph(core.WithDirected(true), core.WithLoops())
	mustNoError(t, g.AddVertex("A"))
	_, err := g.AddEdge("A", "A", 0)
	mustNoError(t, err)

	result, err := dfs.DetectCycles(g)
	mustNoError(t, err)

	mustCycleResult(t, result, true, [][]string{
		{"A", "A"},
	})
}

func TestDetectCycles_SimpleTwoNodeCycle(t *testing.T) {
	g, _ := core.NewGraph(core.WithDirected(true))
	_, _ = g.AddEdge("A", "B", 0)
	_, _ = g.AddEdge("B", "A", 0)

	result, err := dfs.DetectCycles(g)
	mustNoError(t, err)

	mustCycleResult(t, result, true, [][]string{
		{"A", "B", "A"},
	})
}

func TestDetectCycles_ThreeNodeCycle(t *testing.T) {
	g, _ := core.NewGraph(core.WithDirected(true))
	_, _ = g.AddEdge("A", "B", 0)
	_, _ = g.AddEdge("B", "C", 0)
	_, _ = g.AddEdge("C", "A", 0)

	result, err := dfs.DetectCycles(g)
	mustNoError(t, err)

	mustCycleResult(t, result, true, [][]string{
		{"A", "B", "C", "A"},
	})
}

func TestDetectCycles_FourNodeCycle(t *testing.T) {
	g, _ := core.NewGraph(core.WithDirected(true))
	_, _ = g.AddEdge("V", "W", 0)
	_, _ = g.AddEdge("W", "X", 0)
	_, _ = g.AddEdge("X", "Y", 0)
	_, _ = g.AddEdge("Y", "Z", 0)
	_, _ = g.AddEdge("Z", "W", 0)

	result, err := dfs.DetectCycles(g)
	mustNoError(t, err)

	mustCycleResult(t, result, true, [][]string{
		{"W", "X", "Y", "Z", "W"},
	})
}

func TestDetectCycles_UndirectedTriangle_NoFalseTwoCycle(t *testing.T) {
	g, _ := core.NewGraph()
	_, _ = g.AddEdge("A", "B", 0)
	_, _ = g.AddEdge("B", "C", 0)
	_, _ = g.AddEdge("C", "A", 0)

	result, err := dfs.DetectCycles(g)
	mustNoError(t, err)

	mustCycleResult(t, result, true, [][]string{
		{"A", "B", "C", "A"},
	})
}

func TestDetectCycles_UndirectedMultipleDisjointCycles(t *testing.T) {
	g, _ := core.NewGraph()

	_, _ = g.AddEdge("A", "B", 0)
	_, _ = g.AddEdge("B", "C", 0)
	_, _ = g.AddEdge("C", "A", 0)

	_, _ = g.AddEdge("W", "X", 0)
	_, _ = g.AddEdge("X", "Y", 0)
	_, _ = g.AddEdge("Y", "Z", 0)
	_, _ = g.AddEdge("Z", "W", 0)

	result, err := dfs.DetectCycles(g)
	mustNoError(t, err)

	mustCycleResult(t, result, true, [][]string{
		{"A", "B", "C", "A"},
		{"W", "X", "Y", "Z", "W"},
	})
}

func TestDetectCycles_DirectedMultipleDisjointCycles(t *testing.T) {
	g, _ := core.NewGraph(core.WithDirected(true))

	cycleOne := []string{"A", "B", "C", "D", "E", "A"}
	for index := 0; index < len(cycleOne)-1; index++ {
		_, _ = g.AddEdge(cycleOne[index], cycleOne[index+1], 0)
	}

	cycleTwo := []string{"F", "G", "H", "I", "F"}
	for index := 0; index < len(cycleTwo)-1; index++ {
		_, _ = g.AddEdge(cycleTwo[index], cycleTwo[index+1], 0)
	}

	_, _ = g.AddEdge("E", "F", 0)
	_ = g.AddVertex("J")
	_ = g.AddVertex("K")

	result, err := dfs.DetectCycles(g)
	mustNoError(t, err)

	mustCycleResult(t, result, true, [][]string{
		{"A", "B", "C", "D", "E", "A"},
		{"F", "G", "H", "I", "F"},
	})
}

func TestDetectCycles_OverlappingDirectedCycles_WitnessContract(t *testing.T) {
	g, _ := core.NewGraph(core.WithDirected(true))

	// The graph contains:
	//   - A->B->C->A
	//   - A->C->A
	//
	// The contract is witness-set based, not exhaustive simple-cycle enumeration.
	// The deterministic DFS-based algorithm is expected to report a stable witness,
	// not every mathematically possible simple-cycle representation.
	_, _ = g.AddEdge("A", "B", 0)
	_, _ = g.AddEdge("B", "C", 0)
	_, _ = g.AddEdge("C", "A", 0)
	_, _ = g.AddEdge("A", "C", 0)

	result, err := dfs.DetectCycles(g)
	mustNoError(t, err)

	mustCycleResult(t, result, true, [][]string{
		{"A", "B", "C", "A"},
	})
}

func TestDetectCycles_DirectedCanonicalizationPreservesOrientation(t *testing.T) {
	g, _ := core.NewGraph(core.WithDirected(true))

	// The directed cycle is A->C->B->A.
	// Its reversed lexical form A->B->C->A is smaller, but must not win canonicalization
	// because direction is part of the identity of a directed cycle.
	_, _ = g.AddEdge("A", "C", 0)
	_, _ = g.AddEdge("C", "B", 0)
	_, _ = g.AddEdge("B", "A", 0)

	result, err := dfs.DetectCycles(g)
	mustNoError(t, err)

	mustCycleResult(t, result, true, [][]string{
		{"A", "C", "B", "A"},
	})
}

func TestMinimalRotation_DoesNotMutateInputBackingArray(t *testing.T) {
	input := make([]string, 3, 6)
	copy(input, []string{"B", "C", "A"})

	original := append([]string(nil), input...)

	rotation := dfs.TestOnlyMinimalRotation(input)

	if len(input) != len(original) {
		t.Fatalf("input length changed: got=%d want=%d", len(input), len(original))
	}

	for index := range original {
		if input[index] != original[index] {
			t.Fatalf(
				"input mutated at index %d: got=%q want=%q input=%v original=%v",
				index, input[index], original[index], input, original,
			)
		}
	}

	wantRotation := []string{"A", "B", "C"}
	if len(rotation) != len(wantRotation) {
		t.Fatalf("rotation length mismatch: got=%d want=%d rotation=%v", len(rotation), len(wantRotation), rotation)
	}
	for index := range wantRotation {
		if rotation[index] != wantRotation[index] {
			t.Fatalf(
				"rotation mismatch at index %d: got=%q want=%q rotation=%v want=%v",
				index, rotation[index], wantRotation[index], rotation, wantRotation,
			)
		}
	}

	rotation[0] = "X"
	if input[0] != original[0] {
		t.Fatalf("rotation shares mutation-sensitive state with input: input=%v original=%v", input, original)
	}
}
