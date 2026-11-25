// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mst exposes narrow internal validation boundaries to external contract tests.
// This file is compiled only by go test and contributes no production API or assertions.
package mst

import (
	"context"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/dijkstra"
)

// SteinerPreparedForTest is a distinct, opaque preparation fixture for external tests.
// Its underlying structure is identical to steinerPrepared, so pointer conversions
// preserve the same fixture without copying inventories or exposing private fields.
//
// Behavior highlights:
//   - The defined type has its own exported identity and exists only in test builds.
//   - Adapters convert its pointer directly to the private kernel state.
//
// Complexity:
//   - Each pointer conversion takes O(1) time and O(1) additional space.
//
// Notes:
//   - Fixtures are read-only during kernel execution; the graph is not cloned here.
//
// AI-Hints:
//   - Do not turn this declaration into a type alias: the fixture must have its own
//     type identity while keeping the actual preparation state private.
type SteinerPreparedForTest steinerPrepared

// Test-only function aliases invoke the actual implementation without a substitute kernel.
// External tests must never reassign these aliases; ordinary builds do not contain them.
var (
	// CheckedSteinerPairCountForTest forwards the scalar closure-cardinality guard.
	CheckedSteinerPairCountForTest = checkedSteinerPairCount
	// CheckedSteinerSliceCapacityForTest forwards the scalar backing-size guard.
	CheckedSteinerSliceCapacityForTest = checkedSteinerSliceCapacity
	// CheckedSteinerCountAddForTest forwards checked composite-count arithmetic.
	CheckedSteinerCountAddForTest = checkedSteinerCountAdd
	// PruneSteinerLeavesForTest forwards degree-queue pruning.
	PruneSteinerLeavesForTest = pruneSteinerLeaves
	// PrepareSteinerWalkTreeForTest forwards the caller-owned tree validator.
	PrepareSteinerWalkTreeForTest = prepareSteinerWalkTree
	// AppendSteinerWalkPassageForTest forwards the checked original-edge step append.
	AppendSteinerWalkPassageForTest = appendSteinerWalkPassage
)

// NewSteinerPreparedForTest constructs an isolated boundary fixture for external tests.
// Explicit inventories permit verification of stale or missing edge witnesses without mutating a live graph.
// Implementation:
//   - Stage 1: Copy terminal IDs into the fixture and retain the supplied read-only graph.
//   - Stage 2: Index the supplied original records and canonical endpoint-pair keys.
//   - Stage 3: Convert the fixture pointer to the distinct test-facing type without copying it.
//
// Behavior highlights:
//   - Invalid inventory records are preserved deliberately for boundary rejection tests.
//   - No shortest paths, minimum-edge choices, or expected outcomes are computed here.
//
// Inputs:
//   - graph: optional stable graph; mandatory for stages that invoke Dijkstra.
//   - terminals: explicitly ordered fixture IDs, copied on entry.
//   - originals: independent original records with unique ID keys.
//   - pairs: explicit canonical endpoint keys (smaller ID first), not inferred from edge records.
//
// Returns:
//   - *SteinerPreparedForTest: non-nil opaque fixture; nil graph is permitted for inventory-only tests.
//
// Errors:
//   - None; the tested production boundary owns error classification.
//
// Determinism:
//   - Explicit unique map keys and a copied terminal order; no map iteration selects a path.
//
// Complexity:
//   - Time O(k+E+P), Space O(k+E+P), for k terminals, E records and P pair entries.
//
// Notes:
//   - The fixture is not a public graph adapter and must never be used as one.
//   - All assertions reside in the external mst_test package.
//
// AI-Hints:
//   - Use the public SteinerTree facade for normal graphs; use this fixture only to test
//     internal contracts that public admission and correct Dijkstra results normally guarantee.
func NewSteinerPreparedForTest(graph *core.Graph, terminals []string, originals []core.Edge,
	pairs map[[2]string]core.Edge,
) *SteinerPreparedForTest {
	// Keep the graph stable and detach the terminal slice from the external fixture.
	prepared := &steinerPrepared{
		graph:     graph,
		terminals: append([]string(nil), terminals...),
		originals: make(map[string]core.Edge, len(originals)),
		minPair:   make(map[steinerEndpointPair]core.Edge, len(pairs)),
	}
	// Populate the exact records; do not repair a stale ID or normalize its endpoints.
	for _, edge := range originals {
		prepared.originals[edge.ID] = edge
	}
	// Preserve explicitly supplied pair keys so incidence failures remain observable.
	for key, edge := range pairs {
		prepared.minPair[steinerEndpointPair{left: key[0], right: key[1]}] = edge
	}
	// Give the same fixture its independent test-facing type without copying its fields.
	return (*SteinerPreparedForTest)(prepared)
}

// BuildSteinerClosureForTest forwards a prepared fixture to the real closure builder.
// It preserves the selected SSSP storage policy, terminal distances, and error identities.
//
// Implementation:
//   - Stage 1: Convert the opaque fixture pointer to the private preparation type.
//   - Stage 2: Execute buildSteinerClosure without intermediate validation or copying.
//   - Stage 3: Return the original closure candidates, retained state, and error.
//
// Behavior highlights:
//   - Both recompute and retain policies use the production shortest-path implementation.
//
// Inputs:
//   - prepared: non-nil fixture with stable graph and ordered terminals.
//   - config: validated Steiner policy, including context and path storage.
//
// Returns:
//   - []core.Edge: synthetic terminal candidates, not original network edges.
//   - map[string]*dijkstra.Result: retained SSSP state or nil in recompute mode.
//   - error: unchanged closure failure, with no partial result.
//
// Errors:
//   - ErrDisconnectedTerminals, ErrInvalidWitness, ErrSizeOverflow, or preserved
//     Dijkstra/context causes from buildSteinerClosure.
//
// Determinism:
//   - The kernel determines terminal-pair order; the adapter adds no ordering decisions.
//
// Complexity:
//   - Adapter Time O(1), Space O(1); kernel uses k-1 SSSP runs and O(k^2) closure storage.
//     Retain additionally stores O(kV) distance/predecessor state.
//
// Notes:
//   - The pointer conversion does not detach or mutate the fixture.
//
// AI-Hints:
//   - Do not rebuild the closure or replace failed SSSP results in this adapter.
func BuildSteinerClosureForTest(
	prepared *SteinerPreparedForTest,
	config SteinerOptions,
) ([]core.Edge, map[string]*dijkstra.Result, error) {
	// Forward the same preparation state and policy to the canonical builder.
	return buildSteinerClosure((*steinerPrepared)(prepared), config)
}

// ExpandSteinerVertexPathForTest forwards one SSSP witness to original-edge validation.
// It preserves source-relative incidence, represented prefix costs, and union identity.
//
// Implementation:
//   - Stage 1: Convert the opaque preparation pointer without copying inventories.
//   - Stage 2: Pass the caller-supplied witness, endpoints, cost, and union to the kernel.
//   - Stage 3: Return the kernel error unchanged.
//
// Behavior highlights:
//   - Malformed witnesses reach the actual boundary rather than being repaired by the adapter.
//
// Inputs:
//   - ctx: non-nil context; prepared: non-nil original-edge fixture.
//   - pathResult: source-specific Dijkstra result, possibly invalid for rejection tests.
//   - source, target: terminal IDs; expected: represented closure distance.
//   - union: caller-owned, initialized original-ID map, modified by the kernel.
//
// Returns:
//   - error: nil for a verified path; otherwise the original kernel failure.
//
// Errors:
//   - ErrInvalidWitness, ErrWeightOverflow, and preserved Dijkstra/context causes.
//
// Determinism:
//   - Production path reconstruction and endpoint-pair lookup determine the outcome.
//
// Complexity:
//   - Adapter Time O(1), Space O(1); kernel takes O(P) time and O(P) path storage
//     for P path vertices, plus storage for new original union records.
//
// Notes:
//   - A failed expansion can leave a private union prefix; callers discard it on error.
//
// AI-Hints:
//   - Use a fresh union map per rejection case; never expect transactional rollback here.
func ExpandSteinerVertexPathForTest(
	ctx context.Context,
	prepared *SteinerPreparedForTest,
	pathResult *dijkstra.Result,
	source, target string,
	expected float64,
	union map[string]core.Edge,
) error {
	// Preserve the original arguments, map ownership, and sentinel identities.
	return expandSteinerVertexPath(ctx, (*steinerPrepared)(prepared), pathResult, source, target, expected, union)
}

// ExpandSteinerClosureForTest forwards selected closure links to original-path expansion.
// It exercises actual retained or recomputed SSSP witnesses without substituting a solver.
//
// Implementation:
//   - Stage 1: Convert the opaque fixture pointer to the private preparation type.
//   - Stage 2: Execute the source-grouped expansion using the supplied retained state.
//   - Stage 3: Return the production union vertices, edges, and error unchanged.
//
// Behavior highlights:
//   - Original Edge.ID deduplication and distance consistency remain kernel responsibilities.
//
// Inputs:
//   - prepared: non-nil graph and original-edge fixture; config: validated Steiner policy.
//   - selected: selected terminal closure edges; retained: source-specific SSSP results.
//
// Returns:
//   - []string: union vertices; []core.Edge: original union edges; error: kernel failure or nil.
//
// Errors:
//   - ErrInvalidWitness, ErrWeightOverflow, and preserved Dijkstra/context causes.
//
// Determinism:
//   - Production source-group order and canonical union publication order are preserved.
//
// Complexity:
//   - Adapter Time O(1), Space O(1); kernel costs include source-grouped SSSP,
//     O(L) witness expansion, and O(Eu log Eu + Vu log Vu) union sorting.
//
// Notes:
//   - The fixture and retained SSSP maps are not copied or modified by the adapter.
//
// AI-Hints:
//   - Do not run an extra SSSP or fill missing retained entries in the adapter.
func ExpandSteinerClosureForTest(
	prepared *SteinerPreparedForTest,
	config SteinerOptions,
	selected []core.Edge,
	retained map[string]*dijkstra.Result,
) ([]string, []core.Edge, error) {
	// Let the production kernel own all witness checks and union construction.
	return expandSteinerClosure((*steinerPrepared)(prepared), config, selected, retained)
}

// PublishSteinerNetworkForTest forwards selected original edges to final publication.
// It retains structural verification, canonical ordering, and checked network-cost semantics.
//
// Implementation:
//   - Stage 1: Convert the opaque fixture pointer to the private preparation type.
//   - Stage 2: Invoke the actual verified publication boundary with the selected edges.
//   - Stage 3: Return its detached network or original failure.
//
// Behavior highlights:
//   - No selected edges or cost values are normalized by the adapter.
//
// Inputs:
//   - ctx: non-nil context; prepared: non-nil terminal and original-edge fixture.
//   - selected: caller-supplied candidate edge values, not mutated by publication.
//
// Returns:
//   - *SteinerResult: verified detached network, or nil on error; error: unchanged failure.
//
// Errors:
//   - ErrInvalidSteinerResult, ErrInvalidWitness, ErrWeightOverflow, ErrSizeOverflow,
//     and preserved context cancellation.
//
// Determinism:
//   - The production publisher owns the canonical vertex, terminal, and edge order.
//
// Complexity:
//   - Adapter Time O(1), Space O(1); publication Time O(E log(E+1) + V log(V+1)),
//     Space O(V+E), for candidate network vertices and edges.
//
// Notes:
//   - No additional result copy is made at this test boundary.
//
// AI-Hints:
//   - Keep provenance and structural verification before numeric publication.
func PublishSteinerNetworkForTest(
	ctx context.Context,
	prepared *SteinerPreparedForTest,
	selected []core.Edge,
) (*SteinerResult, error) {
	// Preserve production validation order and result ownership.
	return publishSteinerNetwork(ctx, (*steinerPrepared)(prepared), selected)
}

// VerifySteinerNetworkForTest forwards a candidate network to structural verification.
// It checks original-record provenance and tree invariants through the actual validator.
//
// Implementation:
//   - Stage 1: Convert the opaque fixture pointer without copying original records.
//   - Stage 2: Pass the supplied candidate directly to verifySteinerNetwork.
//   - Stage 3: Return the original validation outcome.
//
// Behavior highlights:
//   - Invalid ordering, incidence, cycles, and nonterminal leaves are not repaired here.
//
// Inputs:
//   - ctx: non-nil context; prepared: non-nil original-edge fixture.
//   - network: candidate result, including nil or malformed values for rejection tests.
//
// Returns:
//   - error: nil when the candidate satisfies the structural contract; otherwise kernel error.
//
// Errors:
//   - ErrInvalidSteinerResult, ErrInvalidWitness, or preserved context cancellation.
//
// Determinism:
//   - Validation proceeds in the production validator's prescribed slice order.
//
// Complexity:
//   - Adapter Time O(1), Space O(1); verification Time O(V + E*(alpha(V)+log(V+1))),
//     Space O(V+E), where V and E describe the candidate network.
//
// Notes:
//   - Structural verification alone does not publish or recalculate TotalWeight.
//
// AI-Hints:
//   - Pass malformed candidates unchanged to preserve the targeted validation condition.
func VerifySteinerNetworkForTest(
	ctx context.Context,
	prepared *SteinerPreparedForTest,
	network *SteinerResult,
) error {
	// Forward the same candidate and original-edge fixture to the real verifier.
	return verifySteinerNetwork(ctx, (*steinerPrepared)(prepared), network)
}

// RunSteinerForTest forwards a prepared fixture to the production Steiner dispatcher.
// It preserves singleton, pair, and full KMB paths together with failure classification.
//
// Implementation:
//   - Stage 1: Convert the opaque fixture pointer to the private preparation type.
//   - Stage 2: Invoke runSteiner with the unchanged configuration.
//   - Stage 3: Return the production result or error without adaptation.
//
// Behavior highlights:
//   - No graph admission, path reconstruction, or fallback algorithm is added here.
//
// Inputs:
//   - prepared: non-nil fixture with the graph and terminal domain required by the kernel.
//   - config: validated Steiner options, read-only.
//
// Returns:
//   - *SteinerResult: detached network on success; nil on failure.
//   - error: original kernel error or nil.
//
// Errors:
//   - Domain, witness, numeric, Dijkstra, and context errors returned by runSteiner.
//
// Determinism:
//   - All selection and publication order comes from the production kernel.
//
// Complexity:
//   - Adapter Time O(1), Space O(1); total kernel cost depends on terminal count,
//     selected storage policy, SSSP work, closure sorting, and union processing.
//
// Notes:
//   - This is a boundary fixture consumer, not a replacement for public graph admission.
//
// AI-Hints:
//   - Use SteinerTree for ordinary valid inputs; isolate internal failures with this adapter.
func RunSteinerForTest(
	prepared *SteinerPreparedForTest,
	config SteinerOptions,
) (*SteinerResult, error) {
	// Return the actual kernel result without intercepting errors or copying data.
	return runSteiner((*steinerPrepared)(prepared), config)
}
