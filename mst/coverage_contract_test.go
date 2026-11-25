// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mst_test protects public MST option, result ownership, and cancellation contracts.
// The tests use real graph fixtures and classify errors independently of private implementation state.
package mst_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/dijkstra"
	"github.com/lvlath/go/mst"
)

// TestApplyOptions_CustomCallbacksAndStrictOverride protects the legacy final-only option law.
// Invalid final configurations and callback errors must fail before the graph is inspected.
// Implementation:
//   - Stage 1: Supply invalid final fields and an explicit callback failure.
//   - Stage 2: Repair intermediate custom values and explicitly restore strict mode.
//   - Stage 3: Confirm strict disconnection and valid forest publication on the same graph.
//
// Behavior highlights:
//   - A valid later custom assignment may repair an earlier intermediate state.
//   - Callback errors retain identity and prevent later callbacks from executing.
//
// Inputs:
//   - t: test context; options and graph are local fixtures.
//
// Returns:
//   - None; assertions defend the exported option and facade contracts.
//
// Errors:
//   - ErrUnsupportedAlgorithm, ErrInvalidOption, ErrEmptyRoot, ErrNilOption, ErrDisconnected, and a custom cause.
//
// Determinism:
//   - Ordered callback sequence and a fixed two-component graph.
//
// Complexity:
//   - O(1) time and space for the bounded fixture.
//
// Notes:
//   - Steiner's independent option assembler has a different intermediate-validation policy.
//
// AI-Hints:
//   - Use custom callbacks to exercise final validation; built-in setters can reject values earlier.
func TestApplyOptions_CustomCallbacksAndStrictOverride(t *testing.T) {
	const invalidAlgorithm = mst.Algorithm("unknown")
	const invalidMode = mst.Mode("unknown")
	callbackFailure := errors.New("option callback rejected")
	cases := []struct {
		name string
		opts []mst.Option
		want error
	}{
		{"custom algorithm", []mst.Option{func(cfg *mst.Options) error { cfg.Algorithm = invalidAlgorithm; return nil }}, mst.ErrUnsupportedAlgorithm},
		{"custom mode", []mst.Option{func(cfg *mst.Options) error { cfg.Mode = invalidMode; return nil }}, mst.ErrInvalidOption},
		{"custom root", []mst.Option{func(cfg *mst.Options) error { cfg.Algorithm = mst.AlgorithmPrim; return nil }}, mst.ErrEmptyRoot},
		{"nil callback", []mst.Option{nil}, mst.ErrNilOption},
		{"callback failure", []mst.Option{func(*mst.Options) error { return callbackFailure }}, callbackFailure},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Assembly must return its zero configuration on failure.
			cfg, err := mst.ApplyOptions(tc.opts...)
			mustErrorIs(t, err, tc.want, "option assembly")
			if cfg != (mst.Options{}) {
				t.Fatalf("failed options escaped: %+v", cfg)
			}
			// An invalid graph must not supersede an earlier option error.
			result, err := mst.MinimumSpanningTree(nil, tc.opts...)
			mustErrorIs(t, err, tc.want, "facade option precedence")
			if result != nil {
				t.Fatalf("invalid options published %+v", result)
			}
		})
	}

	// Final-only assembly permits a later correction of both invalid fields.
	cfg, err := mst.ApplyOptions(func(cfg *mst.Options) error {
		cfg.Algorithm, cfg.Mode = invalidAlgorithm, invalidMode
		return nil
	}, mst.WithAlgorithm(mst.AlgorithmPrim), mst.WithRoot("B"), mst.WithStrictTree())
	mustNoError(t, err, "repair final configuration")
	if cfg.Algorithm != mst.AlgorithmPrim || cfg.Mode != mst.ModeStrictTree || cfg.Root != "B" {
		t.Fatalf("unexpected finalized configuration: %+v", cfg)
	}

	// A failing callback terminates the ordered option sequence immediately.
	calledLater := false
	_, err = mst.ApplyOptions(func(*mst.Options) error { return callbackFailure }, func(*mst.Options) error {
		calledLater = true
		return nil
	})
	mustErrorIs(t, err, callbackFailure, "first callback failure")
	if calledLater {
		t.Fatal("option executed after a preceding callback failed")
	}

	// Restore strict mode after forest mode and verify the mathematical distinction.
	graph := mustWeightedGraph(t)
	steinerFixtureEdge(t, graph, "AB", "A", "B", 2)
	steinerFixtureEdge(t, graph, "CD", "C", "D", 3)
	result, err := mst.MinimumSpanningTree(graph, mst.WithForest(), mst.WithStrictTree())
	mustErrorIs(t, err, mst.ErrDisconnected, "explicit strict restoration")
	if result != nil {
		t.Fatalf("strict disconnection published %+v", result)
	}
	result, err = mst.MinimumSpanningTree(graph, mst.WithStrictTree(), mst.WithForest())
	mustNoError(t, err, "restore forest")
	mustValidForest(t, graph, result, 2, 5)
}

// TestResult_NilOwnershipAndMetadata covers ordinary MST result access and detached copies.
// Both empty values and real forest results must preserve their documented metadata semantics.
// Implementation:
//   - Stage 1: Check nil and zero-value receiver behavior.
//   - Stage 2: Produce a rooted Prim forest with signed finite weights.
//   - Stage 3: Verify metadata, slice order, and independence from graph and original result.
//
// Behavior highlights:
//   - Clone is a copy operation, not a validator.
//   - Nil result data access returns ErrNilResult rather than panicking.
//
// Inputs:
//   - t: test context; independent graph and results.
//
// Returns:
//   - None; all scalar metadata and slice identities are asserted explicitly.
//
// Errors:
//   - ErrNilResult on nil access; any other failure is unexpected.
//
// Determinism:
//   - Explicit root, edge IDs, signed costs, and known component order.
//
// Complexity:
//   - O(V+E) fixture size plus the public Prim computation.
//
// Notes:
//   - Empty slice nilness is asserted only where the API explicitly promises it.
//
// AI-Hints:
//   - Mutate both copy directions; pointer inequality alone does not prove slice ownership.
func TestResult_NilOwnershipAndMetadata(t *testing.T) {
	var absent *mst.Result
	if !absent.IsNil() || absent.Clone() != nil {
		t.Fatal("nil MST result helpers failed")
	}
	edges, err := absent.EdgeValues()
	mustErrorIs(t, err, mst.ErrNilResult, "nil MST EdgeValues")
	if edges != nil {
		t.Fatal("nil receiver returned edge data")
	}

	// A zero-value result is non-nil but does not represent a verified tree.
	empty := &mst.Result{}
	clone := empty.Clone()
	if empty.IsNil() || clone == nil || clone == empty || len(clone.Edges) != 0 || len(clone.ComponentRoots) != 0 {
		t.Fatalf("zero-value clone: %+v", clone)
	}
	edges, err = empty.EdgeValues()
	mustNoError(t, err, "empty MST EdgeValues")
	mustEqualInt(t, len(edges), 0, "empty edge count")

	// Use a real rooted forest, including an isolate, to exercise component metadata.
	graph := mustWeightedGraph(t)
	steinerFixtureEdge(t, graph, "AB", "A", "B", -2)
	steinerFixtureEdge(t, graph, "CD", "C", "D", 5)
	mustNoError(t, graph.AddVertex("Z"), "add isolate")
	result, err := mst.MinimumSpanningTree(graph, mst.WithAlgorithm(mst.AlgorithmPrim), mst.WithRoot("D"), mst.WithForest())
	mustNoError(t, err, "Prim forest")
	mustValidForest(t, graph, result, 3, 3)
	clone = result.Clone()
	if clone == nil || clone == result || result.IsNil() || clone.Algorithm != result.Algorithm ||
		clone.Mode != result.Mode || clone.Root != result.Root || clone.TotalWeight != result.TotalWeight ||
		clone.VertexCount != result.VertexCount || clone.ComponentCount != result.ComponentCount ||
		!slices.Equal(clone.Edges, result.Edges) || !slices.Equal(clone.ComponentRoots, result.ComponentRoots) {
		t.Fatalf("clone did not preserve the result: original=%+v clone=%+v", result, clone)
	}
	edges, err = result.EdgeValues()
	mustNoError(t, err, "detached EdgeValues")
	if !slices.Equal(edges, result.Edges) {
		t.Fatalf("EdgeValues changed selection order: %v", edges)
	}
	originalFirst := result.Edges[0]
	originalRoot := result.ComponentRoots[0]
	clone.Edges[0].Weight = 100
	clone.ComponentRoots[0] = "changed clone"
	edges[0].ID = "changed copy"
	if result.Edges[0] != originalFirst || result.ComponentRoots[0] != originalRoot {
		t.Fatal("copy mutation changed source result")
	}
	result.Edges[0].Weight = 200
	result.ComponentRoots[0] = "changed source"
	if clone.Edges[0].Weight != 100 || clone.ComponentRoots[0] != "changed clone" || edges[0].Weight != originalFirst.Weight {
		t.Fatal("source mutation changed detached copies")
	}
	original, err := graph.GetEdge(originalFirst.ID)
	mustNoError(t, err, "original graph edge")
	if *original != originalFirst {
		t.Fatal("result or copy mutation changed source graph")
	}

	// A validated singleton has no selected edge, and its accessor succeeds without data.
	valid := mustWeightedGraph(t)
	mustNoError(t, valid.AddVertex("Z"), "valid singleton")
	singleton, err := mst.SteinerTree(valid, []string{"Z"})
	mustNoError(t, err, "singleton SteinerTree")
	edges, err = singleton.EdgeValues()
	mustNoError(t, err, "singleton Steiner EdgeValues")
	if len(edges) != 0 || singleton.TotalWeight != 0 {
		t.Fatalf("singleton edge values=%v result=%+v", edges, singleton)
	}
	// An explicitly nil edge slice stays nil without touching other result fields.
	emptySteiner := &mst.SteinerResult{}
	edges, err = emptySteiner.EdgeValues()
	mustNoError(t, err, "nil Steiner edge slice")
	if edges != nil {
		t.Fatal("nil Steiner edge slice became non-nil")
	}
}

// TestMST_SelfLoopAdmissionAndSelection checks that loops are validated before exclusion.
// Negative finite loops are legal ordinary MST input but cannot participate in an acyclic selection.
// Implementation:
//   - Stage 1: Construct a triangle with a low-cost self-loop.
//   - Stage 2: Compare strict Prim and Kruskal against the known cost and original edge IDs.
//   - Stage 3: Check positive-loop Steiner admission and negative-loop rejection separately.
//
// Behavior highlights:
//   - A loop never connects distinct components; nonfinite loops still fail admission.
//
// Inputs:
//   - t: test context; weighted loop-enabled graph.
//
// Returns:
//   - None; expected edge identities and total costs are checked.
//
// Errors:
//   - ErrNegativeWeight for Steiner; ErrNaNInfWeight for nonfinite loop weights.
//
// Determinism:
//   - Fixed edge catalog and explicit Prim root.
//
// Complexity:
//   - Bounded constant-size graph; general MST selection remains O(E log E).
//
// Notes:
//   - Use separate valid graphs for domain differences; no topology corruption is required.
//
// AI-Hints:
//   - Do not omit admission checks for an edge merely because selection will ignore it.
func TestMST_SelfLoopAdmissionAndSelection(t *testing.T) {
	for _, loopWeight := range []float64{-9, 0, 7} {
		t.Run(fmt.Sprintf("loop_%g", loopWeight), func(t *testing.T) {
			graph := steinerFixtureGraph(t, core.WithLoops())
			steinerFixtureEdge(t, graph, "loop", "A", "A", loopWeight)
			steinerFixtureEdge(t, graph, "AB", "A", "B", 2)
			steinerFixtureEdge(t, graph, "BC", "B", "C", 3)
			steinerFixtureEdge(t, graph, "AC", "A", "C", 8)
			for _, algorithm := range []mst.Algorithm{mst.AlgorithmKruskal, mst.AlgorithmPrim} {
				result, err := mst.MinimumSpanningTree(graph, mst.WithAlgorithm(algorithm), mst.WithRoot("C"))
				mustNoError(t, err, "ordinary MST with loop")
				mustValidStrictMST(t, graph, result, 5)
			}
			// A negative loop is rejected even though it would not enter the tree.
			result, err := mst.SteinerTree(graph, []string{"A", "B", "C"})
			if loopWeight < 0 {
				mustSteinerBoundaryError(t, result, err, mst.ErrNegativeWeight, "negative loop")
				return
			}
			mustNoError(t, err, "nonnegative loop")
			if result.TotalWeight != 5 || len(result.Edges) != 2 || result.Edges[0].ID != "AB" || result.Edges[1].ID != "BC" {
				t.Fatalf("loop altered Steiner selection: %+v", result)
			}
		})
	}
}

// TestSteinerTree_SSSPFailureIdentity distinguishes graph disconnection from distance overflow.
// Each failure is exercised through both path-storage modes and both pair/full-closure branches.
// Implementation:
//   - Stage 1: Build a disconnected graph and a connected chain with an overflowing distance.
//   - Stage 2: Request two and three terminals with each storage policy.
//   - Stage 3: Require nil results and the exact root-cause sentinel, without cross-classification.
//
// Behavior highlights:
//   - All original edge weights are finite, but a path prefix need not be representable.
//
// Inputs:
//   - t: test context; two independently built small graphs.
//
// Returns:
//   - None; public failure semantics are asserted.
//
// Errors:
//   - ErrDisconnectedTerminals and dijkstra.ErrDistanceOverflow remain distinct.
//
// Determinism:
//   - Lexical source A reaches the overflowing prefix through B before C.
//
// Complexity:
//   - O(1) for the bounded three-vertex fixtures.
//
// Notes:
//   - Do not conflate a final-network sum overflow with a shortest-path label overflow.
//
// AI-Hints:
//   - Keep the large edge weights finite so failure occurs in Dijkstra, not graph admission.
func TestSteinerTree_SSSPFailureIdentity(t *testing.T) {
	const huge = math.MaxFloat64
	disconnected := mustWeightedGraph(t)
	steinerFixtureEdge(t, disconnected, "AB", "A", "B", 1)
	mustNoError(t, disconnected.AddVertex("C"), "add disconnected terminal")
	overflow := mustWeightedGraph(t)
	steinerFixtureEdge(t, overflow, "AB", "A", "B", huge)
	steinerFixtureEdge(t, overflow, "BC", "B", "C", huge)
	for _, storage := range []mst.SteinerPathStorage{mst.SteinerRecomputePaths, mst.SteinerRetainPaths} {
		for _, terminals := range [][]string{{"A", "C"}, {"A", "B", "C"}} {
			t.Run(fmt.Sprintf("storage_%d_k%d", storage, len(terminals)), func(t *testing.T) {
				result, err := mst.SteinerTree(disconnected, terminals, mst.WithSteinerPathStorage(storage))
				mustSteinerBoundaryError(t, result, err, mst.ErrDisconnectedTerminals, "unreachable terminal")
				if errors.Is(err, dijkstra.ErrDistanceOverflow) || errors.Is(err, mst.ErrWeightOverflow) {
					t.Fatalf("disconnection mislabeled as arithmetic failure: %v", err)
				}
				result, err = mst.SteinerTree(overflow, terminals, mst.WithSteinerPathStorage(storage))
				mustSteinerBoundaryError(t, result, err, dijkstra.ErrDistanceOverflow, "overflowing distance")
				if errors.Is(err, mst.ErrDisconnectedTerminals) || errors.Is(err, mst.ErrDisconnected) || errors.Is(err, mst.ErrWeightOverflow) {
					t.Fatalf("distance overflow mislabeled: %v", err)
				}
			})
		}
	}
}

// TestSteinerTree_CancellationAtEveryObservedCheckpoint protects atomic result publication.
// A recorded successful execution supplies the cancellation schedule for subsequent identical calls.
// Implementation:
//   - Stage 1: Build the nontrivial KMB cycle/pruning fixture once and record a successful run.
//   - Stage 2: Repeat with cancellation at each observed Err checkpoint for each storage/terminal regime.
//   - Stage 3: Verify cancellation identity, nil result, and unchanged graph/terminal input.
//
// Behavior highlights:
//   - Singleton, pair, recompute and retain paths are exercised without time-based deadlines.
//   - Cancellation is triggered by a real context, including closure selection and publication.
//
// Inputs:
//   - t: test context; stable graph reused serially and fresh context per call.
//
// Returns:
//   - None; every observed checkpoint must suppress partial publication.
//
// Errors:
//   - context.Canceled; no arithmetic or connectivity substitute is accepted.
//
// Determinism:
//   - The successful run defines the checkpoint count; no private line number is hardcoded.
//
// Complexity:
//   - O(P*T) time for P checkpoints and one call cost T; one call's workspace at a time.
//
// Notes:
//   - This checks cooperative cancellation, not cancellation latency inside opaque Dijkstra calls.
//
// AI-Hints:
//   - Do not use sleep, arbitrary timeout durations, or concurrent graph mutation to reach later stages.
func TestSteinerTree_CancellationAtEveryObservedCheckpoint(t *testing.T) {
	graph := mustWeightedGraph(t)
	// Distinct equal-cost terminal paths form a union cycle and leave a removable nonterminal.
	originalEdges := []core.Edge{
		{ID: "AX", From: "A", To: "X", Weight: 100},
		{ID: "CX", From: "C", To: "X", Weight: 100},
		{ID: "XY", From: "X", To: "Y", Weight: 1},
		{ID: "BY", From: "B", To: "Y", Weight: 10},
		{ID: "XZ", From: "X", To: "Z", Weight: 5},
		{ID: "BZ", From: "B", To: "Z", Weight: 6},
	}
	for _, edge := range originalEdges {
		steinerFixtureEdge(t, graph, edge.ID, edge.From, edge.To, edge.Weight)
	}
	originalVertices := graph.Vertices()
	for _, storage := range []mst.SteinerPathStorage{mst.SteinerRecomputePaths, mst.SteinerRetainPaths} {
		for _, terminals := range [][]string{{"A"}, {"B", "A"}, {"C", "A", "B"}} {
			t.Run(fmt.Sprintf("storage_%d_k%d", storage, len(terminals)), func(t *testing.T) {
				// A successful run discovers the complete observation schedule for this regime.
				recorder, cleanup := newSteinerCheckpointContext(0)
				baseline, err := mst.SteinerTree(graph, terminals, mst.WithSteinerContext(recorder), mst.WithSteinerPathStorage(storage))
				checkpoints := recorder.polls.Load()
				cleanup()
				mustNoError(t, err, "record successful checkpoints")
				if baseline == nil || checkpoints <= 1 {
					t.Fatalf("missing successful observation schedule: result=%+v polls=%d", baseline, checkpoints)
				}
				originalTerminals := slices.Clone(terminals)
				for checkpoint := int64(1); checkpoint <= checkpoints; checkpoint++ {
					// A fresh context cancels at exactly this observation and is released immediately.
					ctx, cancel := newSteinerCheckpointContext(checkpoint)
					result, err := mst.SteinerTree(graph, terminals, mst.WithSteinerContext(ctx), mst.WithSteinerPathStorage(storage))
					cancel()
					mustSteinerBoundaryError(t, result, err, context.Canceled, fmt.Sprintf("checkpoint %d/%d", checkpoint, checkpoints))
					// Check complete original identities and weights, not merely unchanged counts.
					if !slices.Equal(terminals, originalTerminals) || !slices.Equal(graph.Vertices(), originalVertices) || graph.EdgeCount() != len(originalEdges) {
						t.Fatalf("checkpoint %d changed caller input", checkpoint)
					}
					for _, original := range originalEdges {
						edge, lookupErr := graph.GetEdge(original.ID)
						mustNoError(t, lookupErr, "original edge after cancellation")
						if edge == nil || *edge != original {
							t.Fatalf("checkpoint %d changed original edge %q: %v", checkpoint, original.ID, edge)
						}
					}
				}
				// A later independent call must still publish the same selected edges and metadata.
				again, err := mst.SteinerTree(graph, terminals, mst.WithSteinerPathStorage(storage))
				mustNoError(t, err, "successful call after cancellations")
				if again == nil || !slices.Equal(again.Edges, baseline.Edges) || !slices.Equal(again.Vertices, baseline.Vertices) || again.TotalWeight != baseline.TotalWeight {
					t.Fatalf("cancellation changed subsequent result: before=%+v after=%+v", baseline, again)
				}
			})
		}
	}
}

// TestSteinerOptions_CustomCallbackAdmission defends the independent Steiner option policy.
// Unlike ordinary ApplyOptions, each successful callback must leave a valid complete configuration.
// Implementation:
//   - Stage 1: Construct a valid small graph and explicit invalid custom options.
//   - Stage 2: Attempt to repair each invalid state with a later callback.
//   - Stage 3: Verify immediate rejection, callback-error identity, and valid limit overrides.
//
// Behavior highlights:
//   - Negative caps and unsupported path storage are not repaired or defaulted silently.
//   - The first callback or scalar error wins before graph validation.
//
// Inputs:
//   - t: active test context; independently applied option callbacks.
//
// Returns:
//   - None; successful options publish a valid tree, invalid options publish none.
//
// Errors:
//   - ErrInvalidOption, ErrResourceLimit, and the caller's original callback error.
//
// Determinism:
//   - Fixed callback order and explicit scalar values.
//
// Complexity:
//   - O(p) option work per case for p callbacks; the positive control uses one singleton call.
//
// Notes:
//   - These tests use only the public facade, not a test export of private option assembly.
//
// AI-Hints:
//   - Do not reuse ordinary ApplyOptions' final-only repair expectation for SteinerOption.
func TestSteinerOptions_CustomCallbackAdmission(t *testing.T) {
	const invalidStorage = mst.SteinerPathStorage(255)
	callbackFailure := errors.New("custom Steiner callback rejected")
	graph := mustWeightedGraph(t)
	steinerFixtureEdge(t, graph, "AB", "A", "B", 1)
	for _, tc := range []struct {
		name string
		opt  mst.SteinerOption
		want error
	}{
		{"custom nil context", func(cfg *mst.SteinerOptions) error { cfg.Context = nil; return nil }, mst.ErrInvalidOption},
		{"custom storage", func(cfg *mst.SteinerOptions) error { cfg.PathStorage = invalidStorage; return nil }, mst.ErrInvalidOption},
		{"custom vertex cap", func(cfg *mst.SteinerOptions) error { cfg.MaxVertices = -1; return nil }, mst.ErrInvalidOption},
		{"custom edge cap", func(cfg *mst.SteinerOptions) error { cfg.MaxEdges = -1; return nil }, mst.ErrInvalidOption},
		{"custom terminal cap", func(cfg *mst.SteinerOptions) error { cfg.MaxTerminals = -1; return nil }, mst.ErrInvalidOption},
		{"negative vertex setter", mst.WithSteinerMaxVertices(-1), mst.ErrInvalidOption},
		{"negative terminal setter", mst.WithSteinerMaxTerminals(-1), mst.ErrInvalidOption},
		{"callback error", func(*mst.SteinerOptions) error { return callbackFailure }, callbackFailure},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// A later valid configuration cannot hide an earlier invalid scalar state.
			calledLater := false
			result, err := mst.SteinerTree(graph, []string{"A"}, tc.opt, func(cfg *mst.SteinerOptions) error {
				calledLater = true
				*cfg = mst.DefaultSteinerOptions()
				return nil
			})
			mustSteinerBoundaryError(t, result, err, tc.want, "Steiner callback admission")
			if calledLater {
				t.Fatal("later callback executed after invalid Steiner options")
			}
		})
	}

	// Valid positive limits may be replaced explicitly before graph admission.
	result, err := mst.SteinerTree(graph, []string{"A", "B"}, mst.WithSteinerMaxVertices(1),
		mst.WithSteinerMaxVertices(2), mst.WithSteinerMaxTerminals(1), mst.WithSteinerMaxTerminals(2),
		mst.WithSteinerMaxEdges(1))
	mustNoError(t, err, "valid cap overrides")
	if result == nil || result.TotalWeight != 1 || len(result.Edges) != 1 {
		t.Fatalf("cap override changed the valid pair result: %+v", result)
	}
}

// TestPrim_RootAdmissionAndConnectedOverflow covers degenerate roots and numeric publication.
// Root existence is validated before traversal, while a connected tree can fail only at its checked total.
// Implementation:
//   - Stage 1: Check implicit and explicit singleton roots.
//   - Stage 2: Check a missing root in a multi-vertex graph.
//   - Stage 3: Exercise connected strict overflow in both ordinary MST kernels.
//
// Behavior highlights:
//   - Forest mode may infer a singleton root; strict mode never guesses a root.
//   - Overflow is not disconnected topology.
//
// Inputs:
//   - t: active test context; independent singleton, pair, and chain fixtures.
//
// Returns:
//   - None; exact root metadata and nil-result error laws are asserted.
//
// Errors:
//   - core.ErrVertexNotFound and ErrWeightOverflow.
//
// Determinism:
//   - Explicit IDs and known connected graph structure.
//
// Complexity:
//   - O(1) for the bounded fixture sizes.
//
// Notes:
//   - Both ordinary MST algorithms accept finite signed weights; this test uses positive overflow.
//
// AI-Hints:
//   - Do not use disconnected input to test strict connected-total publication; topology must succeed first.
func TestPrim_RootAdmissionAndConnectedOverflow(t *testing.T) {
	singleton := mustWeightedGraph(t)
	mustNoError(t, singleton.AddVertex("X"), "add singleton root")
	result, err := mst.MinimumSpanningTree(singleton, mst.WithAlgorithm(mst.AlgorithmPrim), mst.WithForest())
	mustNoError(t, err, "inferred singleton root")
	mustValidForest(t, singleton, result, 1, 0)
	if result.Root != "X" || !slices.Equal(result.ComponentRoots, []string{"X"}) {
		t.Fatalf("incorrect inferred root: %+v", result)
	}
	result, err = mst.Prim(singleton, "missing")
	mustErrorIs(t, err, core.ErrVertexNotFound, "missing singleton root")
	if result != nil {
		t.Fatalf("invalid root published %+v", result)
	}

	graph := mustWeightedGraph(t)
	steinerFixtureEdge(t, graph, "AB", "A", "B", math.MaxFloat64)
	steinerFixtureEdge(t, graph, "BC", "B", "C", math.MaxFloat64)
	result, err = mst.Prim(graph, "missing")
	mustErrorIs(t, err, core.ErrVertexNotFound, "missing multi-vertex root")
	if result != nil {
		t.Fatalf("missing root published %+v", result)
	}
	for _, algorithm := range []mst.Algorithm{mst.AlgorithmKruskal, mst.AlgorithmPrim} {
		result, err := mst.MinimumSpanningTree(graph, mst.WithAlgorithm(algorithm), mst.WithRoot("A"))
		mustErrorIs(t, err, mst.ErrWeightOverflow, "connected strict overflow")
		if result != nil || errors.Is(err, mst.ErrDisconnected) {
			t.Fatalf("connected overflow misclassified: result=%+v error=%v", result, err)
		}
	}
}
