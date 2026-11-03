// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mst exposes the canonical MST facade and focused algorithm wrappers.
// Public wrappers select an algorithm while preserving the Result contract.
package mst

import "github.com/lvlath/go/core"

// MinimumSpanningTree computes a minimum spanning tree or forest over a core.Graph.
//
// Implementation:
//   - Stage 1: Assemble and validate explicit options.
//   - Stage 2: Adapt core.Graph into a detached deterministic mstSnapshot.
//   - Stage 3: Dispatch to exactly one MST kernel.
//   - Stage 4: Publish Result with ownership and component metadata.
//
// Behavior highlights:
//   - Default policy is Kruskal + strict tree.
//   - Use WithForest to request a minimum spanning forest for disconnected graphs.
//   - Use WithAlgorithm(AlgorithmPrim) and WithRoot(root) for strict Prim.
//   - No partial result is returned on error in this phase.
//
// Inputs:
//   - graph: non-nil weighted undirected *core.Graph with no directed edges.
//   - opts: explicit option list; nil options are rejected.
//
// Returns:
//   - *Result: canonical detached result artifact.
//   - error: sentinel-classified failure.
//
// Errors:
//   - ErrNilOption, ErrUnsupportedAlgorithm, ErrInvalidOption, ErrEmptyRoot from option assembly.
//   - ErrInvalidGraph joined with precise graph-policy sentinels from graph adaptation.
//   - ErrNaNInfWeight for non-finite edge weights.
//   - core.ErrVertexNotFound for missing Prim root.
//   - ErrDisconnected for strict tree mode on disconnected graphs.
//
// Determinism:
//   - Options are applied in caller order.
//   - Graph order is snapshotted from core.Vertices() and core.Edges().
//   - Kernel tie-breaks are fixed by each algorithm.
//
// Complexity:
//   - Kruskal: O(E log E + E·α(V)) time, O(E + V) space.
//   - Prim: O(E log E) time, O(E + V) space.
//
// Notes:
//   - The input graph is not mutated.
//   - Concurrent graph mutation during snapshot construction is unsupported.
//   - Kruskal and Prim are focused wrappers over this facade; they select one algorithm
//     without changing validation, error classification, or result ownership.
//
// AI-Hints:
//   - Do not add graph traversal logic to wrappers; the facade must remain the public single source.
func MinimumSpanningTree(graph *core.Graph, opts ...Option) (*Result, error) {
	cfg, err := ApplyOptions(opts...)
	if err != nil {
		return nil, err
	}

	snapshot, err := newMSTSnapshot(graph)
	if err != nil {
		return nil, err
	}

	switch cfg.Algorithm {
	case AlgorithmKruskal:
		return kruskalKernel(snapshot, cfg)
	case AlgorithmPrim:
		return primKernel(snapshot, cfg)
	default:
		return nil, ErrUnsupportedAlgorithm
	}
}

// Kruskal computes a strict minimum spanning tree using Kruskal's algorithm.
// It is a focused wrapper over MinimumSpanningTree that fixes AlgorithmKruskal.
//
// Implementation:
//   - Stage 1: Delegate to MinimumSpanningTree with WithAlgorithm(AlgorithmKruskal).
//   - Stage 2: Reuse canonical option assembly, graph snapshot validation, and Kruskal kernel.
//   - Stage 3: Return the canonical Result without discarding metadata.
//
// Behavior highlights:
//   - Uses ModeStrictTree by default.
//   - Disconnected graphs return ErrDisconnected.
//   - Forest mode is intentionally not hidden behind this wrapper.
//
// Inputs:
//   - graph: non-nil weighted undirected *core.Graph with no directed edges.
//
// Returns:
//   - *Result: detached canonical Kruskal result.
//   - error: sentinel-classified failure from the canonical facade.
//
// Errors:
//   - ErrInvalidGraph joined with precise graph-policy sentinels.
//   - ErrNaNInfWeight for non-finite edge weights.
//   - ErrDisconnected for disconnected strict tree mode.
//
// Determinism:
//   - Candidates are ordered by finite (Weight, Edge.ID), preserving the former stable-sort outcome.
//   - Equal-weight candidates follow original core.Edges() ID order.
//
// Complexity:
//   - Time O(E log E + E·α(V)), Space O(E + V).
//
// Notes:
//   - Use MinimumSpanningTree with WithForest() when a minimum spanning forest is desired.
//   - The input graph is never mutated.
//
// AI-Hints:
//   - Do not bypass MinimumSpanningTree here; wrapper honesty requires one canonical facade.
func Kruskal(graph *core.Graph) (*Result, error) {
	return MinimumSpanningTree(graph, WithAlgorithm(AlgorithmKruskal))
}

// Prim computes a strict minimum spanning tree using Prim's algorithm from root.
// It is a focused wrapper over MinimumSpanningTree that fixes AlgorithmPrim and Root.
//
// Implementation:
//   - Stage 1: Delegate to MinimumSpanningTree with WithAlgorithm(AlgorithmPrim) and WithRoot(root).
//   - Stage 2: Reuse canonical option assembly, graph snapshot validation, and Prim kernel.
//   - Stage 3: Return the canonical Result without discarding metadata.
//
// Behavior highlights:
//   - Uses ModeStrictTree by default.
//   - root must be non-empty and present in graph.
//   - Disconnected graphs return ErrDisconnected.
//   - Forest mode is intentionally not hidden behind this wrapper.
//
// Inputs:
//   - graph: non-nil weighted undirected *core.Graph with no directed edges.
//   - root: non-empty vertex ID used as the strict Prim start vertex.
//
// Returns:
//   - *Result: detached canonical Prim result.
//   - error: sentinel-classified failure from the canonical facade.
//
// Errors:
//   - ErrEmptyRoot when root is empty.
//   - core.ErrVertexNotFound when root is absent.
//   - ErrInvalidGraph joined with precise graph-policy sentinels.
//   - ErrNaNInfWeight for non-finite edge weights.
//   - ErrDisconnected for disconnected strict tree mode.
//
// Determinism:
//   - The first component starts at root.
//   - Frontier ties are resolved by finite Weight and then Edge.ID.
//
// Complexity:
//   - Time O(E log E), Space O(E + V).
//
// Notes:
//   - Use MinimumSpanningTree with WithForest() and WithAlgorithm(AlgorithmPrim) for explicit forest mode.
//   - The input graph is never mutated.
//
// AI-Hints:
//   - Do not replace this wrapper with a second Prim implementation; that would split the kernel contract.
func Prim(graph *core.Graph, root string) (*Result, error) {
	return MinimumSpanningTree(graph, WithAlgorithm(AlgorithmPrim), WithRoot(root))
}

// SteinerTree computes a deterministic original-edge terminal-covering network using KMB.
// It returns an acyclic connected approximation, not a global exact Steiner optimum.
// Implementation:
//   - Stage 1: Apply independent Steiner policies and observe initial cancellation.
//   - Stage 2: Validate and clone one coherent input topology before KMB processing.
//   - Stage 3: Run singleton, pair, or full KMB and publish a checked detached result.
//
// Behavior highlights:
//   - All original edge weights, including loops and unrelated islands, must be nonnegative finite.
//   - The single-terminal case is validated and has zero edges and zero cost.
//   - Both witness-storage modes execute real SSSP work without hidden switching.
//
// Inputs:
//   - graph: caller-owned undirected weighted core.Graph, immutable throughout the call.
//   - terminals: nonempty unique original IDs; copied and lexically canonicalized.
//   - opts: ordered optional SteinerOption callbacks, not retained.
//
// Returns:
//   - *SteinerResult: detached network with original edges, or nil on any error.
//
// Errors:
//   - Existing MST option/graph sentinels, Steiner terminal/domain/witness errors,
//     ErrWeightOverflow, ErrResourceLimit, ErrSizeOverflow, context cancellation,
//     and preserved core/dijkstra root causes when relevant.
//
// Determinism:
//   - Canonical terminal IDs, terminal-pair order, Kruskal (Weight, Edge.ID),
//     original-edge identity and final Edge.ID publication order.
//
// Complexity:
//   - k-1 SSSP plus up to k-1 recomputation SSSP calls; closure O(k^2 log k),
//     union O(Eu log Eu), including clone and catalog sorting overhead.
//
// Notes:
//   - core.Clone is coherent for topology but shallow-copies Metadata, which is not read.
//   - Legacy Dijkstra has no context; cancellation is cooperative around its calls.
//
// AI-Hints:
//   - Do not expose closure edges or equate a network's unique-edge cost with a walk cost.
//   - Do not use the public MST facade for closure: its mandatory total can overflow.
func SteinerTree(graph *core.Graph, terminals []string, opts ...SteinerOption) (*SteinerResult, error) {
	config, err := applySteinerOptions(opts...)
	if err != nil {
		return nil, err
	}
	if err := config.Context.Err(); err != nil {
		return nil, err
	}
	prepared, err := prepareSteiner(graph, terminals, config)
	if err != nil {
		return nil, err
	}
	return runSteiner(prepared, config)
}
