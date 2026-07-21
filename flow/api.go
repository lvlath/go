// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package flow exposes canonical maximum-flow facades and compatibility wrappers.
//
// This file contains only public API orchestration and result publication helpers.
// Algorithmic mathematics lives in the runDinic, runEdmondsKarp, and runFordFulkerson
// kernels, while residual construction lives in residual.go.
package flow

import (
	"context"
	"errors"
	"fmt"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/matrix"
)

// MaxFlow computes the maximum s-t flow through a weighted capacity network.
// It is the canonical public facade for all maximum-flow kernels in this package.
//
// Implementation:
//   - Stage 1: Apply and validate Option values before touching graph topology.
//   - Stage 2: Validate graph, terminals, and weighted-capacity policy.
//   - Stage 3: Build a deterministic residualNetwork from core.Edges().
//   - Stage 4: Dispatch to the selected kernel without duplicating validation.
//   - Stage 5: Publish MaxFlowResult through the selected kernel finalization path.
//
// Behavior highlights:
//   - The input graph is never mutated.
//   - Directed input edges become one residual arc.
//   - Undirected input edges become two directed capacity arcs.
//   - Parallel edges are aggregated deterministically by the residual adapter.
//   - The residual graph published in MaxFlowResult is always directed and weighted.
//
// Inputs:
//   - g: weighted core.Graph carrying edge capacities in Edge.Weight.
//   - source: non-empty source vertex ID that must exist in g.
//   - sink: non-empty sink vertex ID that must exist in g and differ from source.
//   - opts: optional WithXxx policies; nil options are rejected.
//
// Returns:
//   - *MaxFlowResult: canonical result artifact on success.
//   - error: nil on success, sentinel-classified error on failure.
//
// Errors:
//   - ErrInvalidOptions / ErrInvalidEpsilon from option assembly.
//   - ErrNilGraph, ErrEmptyTerminal, ErrSameTerminal from validation.
//   - ErrSourceNotFound, ErrSinkNotFound from terminal lookup.
//   - ErrUnweightedGraph when g cannot represent capacity weights.
//   - ErrInvalidCapacity, ErrNegativeCapacity, ErrNaNInf from residual construction.
//   - context cancellation errors from cfg.ctx.
//   - ErrObserverFailure if an observer rejects an augmentation event.
//
// Determinism:
//   - Vertex order comes from core.Vertices().
//   - Edge ingestion order comes from core.Edges().
//   - Residual traversal order comes from sorted residualNetwork adjacency lists.
//   - Algorithm dispatch is explicit through WithAlgorithm; no heuristic auto-selection.
//
// Complexity:
//   - Facade overhead is O(V + E + A log A) before kernel execution,
//     where A is the residual adjacency-entry count.
//   - Kernel complexity depends on Algorithm: Dinic, Edmonds-Karp, or Ford-Fulkerson.
//
// Notes:
//   - AlgorithmDinic is the default because it is usually the strongest general-purpose
//     choice among the currently implemented kernels.
//   - Use AlgorithmEdmondsKarp when shortest augmenting paths are desired for easier
//     reasoning/debugging despite weaker asymptotic performance.
//   - Use AlgorithmFordFulkerson for small, simple, integral-like networks or compatibility.
//
// AI-Hints:
//   - Do not move validation into individual public wrappers; MaxFlow is the contract gate.
//   - Do not rebuild residual state inside the switch; the residual adapter is shared truth.
//   - Do not add heuristic algorithm selection without a new explicit Option.
func MaxFlow(g *core.Graph, source, sink string, opts ...Option) (*MaxFlowResult, error) {
	// Apply all user options first so invalid policies fail before graph allocation.
	cfg, err := applyOptions(opts...)
	if err != nil {
		return nil, err
	}

	// Validate graph and terminal laws before any residual network is allocated.
	if err = validateFlowInput(g, source, sink, cfg); err != nil {
		return nil, err
	}

	// Convert core.Graph into the package-owned deterministic residual representation.
	rn, err := buildResidualNetwork(g, cfg)
	if err != nil {
		return nil, err
	}

	// Dispatch to exactly one kernel. The switch changes algorithmic strategy,
	// but not input validation, residual construction, or result publication laws.
	switch cfg.algorithm {
	case AlgorithmDinic:
		// Dinic builds BFS level graphs and pushes blocking flows through them.
		// Prefer it for larger networks where repeated shortest-path BFS is expensive.
		return runDinic(source, sink, rn, cfg)

	case AlgorithmEdmondsKarp:
		// Edmonds-Karp uses BFS to choose the shortest augmenting path each round.
		// Prefer it for auditability, deterministic path witnesses, and simpler proofs.
		return runEdmondsKarp(source, sink, rn, cfg)

	case AlgorithmFordFulkerson:
		// Ford-Fulkerson uses deterministic DFS augmenting paths.
		// Prefer it only for small or compatibility-oriented networks.
		return runFordFulkerson(source, sink, rn, cfg)

	default:
		// This branch is unreachable after applyOptions, but it protects internal misuse.
		return nil, ErrInvalidOptions
	}
}

// Dinic computes maximum flow through the legacy tuple-return API.
// New code should prefer MaxFlow with WithAlgorithm(AlgorithmDinic).
//
// Implementation:
//   - Stage 1: Convert Options into canonical Option values.
//   - Stage 2: Force AlgorithmDinic for the canonical dispatcher.
//   - Stage 3: Delegate to MaxFlow through runMaxFlowLegacy.
//   - Stage 4: Project MaxFlowResult into the historical tuple shape.
//
// Behavior highlights:
//   - This wrapper contains no algorithmic logic.
//   - Validation, residual construction, min-cut extraction, and error policy are
//     inherited from MaxFlow.
//   - The returned residual graph is directed and weighted.
//
// Inputs:
//   - g: weighted capacity graph.
//   - source: non-empty source vertex ID.
//   - sink: non-empty sink vertex ID different from source.
//   - opts: legacy Options.
//
// Returns:
//   - float64: max-flow value or partial pushed value on interruption.
//   - *core.Graph: published residual graph when available.
//   - error: nil on success or sentinel-classified failure.
//
// Errors:
//   - Same as MaxFlow.
//
// Determinism:
//   - Same as MaxFlow and runDinic.
//
// Complexity:
//   - O(1) wrapper overhead plus Dinic kernel complexity.
//
// Notes:
//   - Tuple return cannot represent cut partitions or augmentation metadata.
//   - Prefer MaxFlow for new code.
//
// AI-Hints:
//   - Do not reintroduce validation or residual construction into this wrapper.
func Dinic(
	g *core.Graph,
	source, sink string,
	opts Options,
) (maxFlow float64, residualGraph *core.Graph, err error) {
	return runMaxFlowLegacy(g, source, sink, opts, AlgorithmDinic)
}

// EdmondsKarp computes maximum flow through the legacy tuple-return API.
// New code should prefer MaxFlow with WithAlgorithm(AlgorithmEdmondsKarp).
//
// Implementation:
//   - Stage 1: Convert Options into canonical Option values.
//   - Stage 2: Force AlgorithmEdmondsKarp for the canonical dispatcher.
//   - Stage 3: Delegate to MaxFlow through runMaxFlowLegacy.
//   - Stage 4: Project MaxFlowResult into the historical tuple shape.
//
// Behavior highlights:
//   - This wrapper contains no algorithmic logic.
//   - Edmonds-Karp uses shortest augmenting paths inside the canonical kernel.
//   - The returned residual graph is directed and weighted.
//
// Inputs:
//   - g: weighted capacity graph.
//   - source: non-empty source vertex ID.
//   - sink: non-empty sink vertex ID different from source.
//   - opts: legacy Options.
//
// Returns:
//   - float64: max-flow value or partial pushed value on interruption.
//   - *core.Graph: published residual graph when available.
//   - error: nil on success or sentinel-classified failure.
//
// Errors:
//   - Same as MaxFlow.
//
// Determinism:
//   - Same as MaxFlow and runEdmondsKarp.
//
// Complexity:
//   - O(1) wrapper overhead plus Edmonds-Karp kernel complexity.
//
// Notes:
//   - Tuple return cannot represent cut partitions or augmentation metadata.
//   - Prefer MaxFlow for new code.
//
// AI-Hints:
//   - Do not duplicate BFS logic here; this is only a compatibility adapter.
func EdmondsKarp(
	g *core.Graph,
	source, sink string,
	opts Options,
) (maxFlow float64, residualGraph *core.Graph, err error) {
	return runMaxFlowLegacy(g, source, sink, opts, AlgorithmEdmondsKarp)
}

// FordFulkerson computes maximum flow through the legacy tuple-return API.
// New code should prefer MaxFlow with WithAlgorithm(AlgorithmFordFulkerson).
//
// Implementation:
//   - Stage 1: Convert Options into canonical Option values.
//   - Stage 2: Force AlgorithmFordFulkerson for the canonical dispatcher.
//   - Stage 3: Delegate to MaxFlow through runMaxFlowLegacy.
//   - Stage 4: Project MaxFlowResult into the historical tuple shape.
//
// Behavior highlights:
//   - This wrapper contains no algorithmic logic.
//   - Ford-Fulkerson uses deterministic DFS augmenting paths.
//   - WithMaxAugmentations is recommended for safety-sensitive DFS runs.
//
// Inputs:
//   - g: weighted capacity graph.
//   - source: non-empty source vertex ID.
//   - sink: non-empty sink vertex ID different from source.
//   - opts: legacy Options.
//
// Returns:
//   - float64: max-flow value or partial pushed value on interruption.
//   - *core.Graph: published residual graph when available.
//   - error: nil on success or sentinel-classified failure.
//
// Errors:
//   - Same as MaxFlow, including ErrAugmentationLimit.
//
// Determinism:
//   - Same as MaxFlow and runFordFulkerson.
//
// Complexity:
//   - O(1) wrapper overhead plus Ford-Fulkerson kernel complexity.
//
// Notes:
//   - Tuple return cannot represent cut partitions or augmentation metadata.
//   - Prefer MaxFlow for new code.
//
// AI-Hints:
//   - Do not use this wrapper as a separate algorithm implementation path.
func FordFulkerson(
	g *core.Graph,
	source, sink string,
	opts Options,
) (maxFlow float64, residualGraph *core.Graph, err error) {
	return runMaxFlowLegacy(g, source, sink, opts, AlgorithmFordFulkerson)
}

// CapacityMatrix builds a dense capacity matrix using the same graph-adapter law as MaxFlow.
// It is intended for diagnostics, visualization, tests, and matrix-based post-processing.
//
// Implementation:
//   - Stage 1: Apply Option values and validate graph-level capacity policy.
//   - Stage 2: Build residualNetwork through buildResidualNetwork, reusing the exact
//     directed/undirected/parallel-edge adapter used by MaxFlow.
//   - Stage 3: Allocate a square matrix.Dense with one row and one column per vertex.
//   - Stage 4: Build a vertex-to-index lookup from the deterministic vertex order.
//   - Stage 5: Write positive residual capacities into the dense matrix.
//
// Behavior highlights:
//   - Directed edges write C[from,to].
//   - Undirected edges write both C[from,to] and C[to,from].
//   - Parallel edges are aggregated before matrix publication.
//   - Capacities <= epsilon are treated as absent.
//   - Matrix operations are not used inside max-flow hot loops.
//
// Inputs:
//   - g: weighted core.Graph capacity network.
//   - opts: optional flow policies, mainly WithEpsilon and WithContext.
//
// Returns:
//   - *matrix.Dense: square capacity matrix.
//   - []string: vertex order where order[i] maps to row i and column i.
//   - error: nil on success or sentinel-classified failure.
//
// Errors:
//   - ErrInvalidOptions / ErrInvalidEpsilon from option assembly.
//   - ErrNilGraph / ErrUnweightedGraph from graph validation.
//   - ErrInvalidCapacity / ErrNaNInf / ErrNegativeCapacity from capacity validation.
//   - ErrInvalidCapacity joined with matrix allocation/write errors.
//
// Determinism:
//   - Vertex order is inherited from core.Vertices() through residualNetwork.
//   - Capacity writes scan rn.vertices and rn.adj[from] in deterministic order.
//
// Complexity:
//   - Time O(V^2 + A), Space O(V^2), where A is residual adjacency-entry count.
//
// Notes:
//   - This adapter represents the effective flow capacity network, not the final
//     residual graph after max-flow.
//   - Use MaxFlowResult.Residual for post-run residual certificates.
//
// AI-Hints:
//   - Do not call this inside algorithmic inner loops.
//   - Use this helper when a stable matrix snapshot is needed for docs, tests, or analytics.
func CapacityMatrix(g *core.Graph, opts ...Option) (*matrix.Dense, []string, error) {
	cfg, err := applyOptions(opts...)
	if err != nil {
		return nil, nil, err
	}

	if err = validateFlowGraphOnly(g, cfg); err != nil {
		return nil, nil, err
	}

	rn, err := buildResidualNetwork(g, cfg)
	if err != nil {
		return nil, nil, err
	}

	capacityMatrix, err := matrix.NewPreparedDense(len(rn.vertices), len(rn.vertices))
	if err != nil {
		return nil, nil, errors.Join(ErrInvalidCapacity, err)
	}

	vertexIndex := make(map[string]int, len(rn.vertices))
	for index, vertexID := range rn.vertices {
		vertexIndex[vertexID] = index
	}

	for _, from := range rn.vertices {
		row := vertexIndex[from]

		for _, to := range rn.adj[from] {
			capacity := rn.cap[from][to]
			if capacity <= cfg.epsilon {
				continue
			}

			col := vertexIndex[to]
			if err = capacityMatrix.Set(row, col, capacity); err != nil {
				return nil, nil, errors.Join(ErrInvalidCapacity, err)
			}
		}
	}

	return capacityMatrix, append([]string(nil), rn.vertices...), nil
}

// finalizeResult publishes a successful or explicitly partial maximum-flow result.
// It converts internal residual state into public certificates owned by the caller.
//
// Implementation:
//   - Stage 1: Build a directed weighted core.Graph from positive residual arcs.
//   - Stage 2: Extract the source-side and sink-side min-cut partition.
//   - Stage 3: Assemble MaxFlowResult with algorithm metadata and certificate fields.
//
// Behavior highlights:
//   - Residual output never inherits input graph flags.
//   - Cut slices are built in residualNetwork vertex order and are caller-owned.
//   - The residual graph is detached from both input graph and internal cap maps.
//
// Inputs:
//   - source: source vertex ID used by the run.
//   - sink: sink vertex ID used by the run.
//   - rn: final internal residual network.
//   - cfg: finalized runtime options containing epsilon.
//   - algorithm: kernel identity for metadata.
//   - value: accumulated flow value.
//   - augmentations: number of successful augmenting pushes.
//   - partial: whether the run was interrupted after producing partial state.
//
// Returns:
//   - *MaxFlowResult: result artifact with residual graph and min-cut certificate.
//   - error: nil on success, otherwise residual graph construction error.
//
// Errors:
//   - Any core.NewGraph/AddVertex/AddEdge sentinel surfaced by buildResidualGraph.
//   - ErrInvalidCapacity if a residual capacity cannot be published.
//
// Determinism:
//   - Residual edges are emitted by rn.vertices order and rn.adj[from] order.
//   - Min-cut sides follow rn.vertices order.
//
// Complexity:
//   - Time O(V + A), Space O(V + A), where A is residual adjacency-entry count.
//
// Notes:
//   - This helper is intentionally shared by all kernels to keep publication semantics
//     identical across AlgorithmDinic, AlgorithmEdmondsKarp, and AlgorithmFordFulkerson.
//
// AI-Hints:
//   - Do not call core.CloneEmpty here; residual networks are mathematically directed.
//   - Do not expose rn.cap directly; it is internal mutable state.
func finalizeResult(
	source, sink string,
	rn *residualNetwork,
	cfg options,
	algorithm Algorithm,
	value float64,
	augmentations int,
	partial bool,
) (*MaxFlowResult, error) {
	// Publish only positive residual arcs as a deterministic directed weighted graph.
	residual, err := buildResidualGraph(rn, cfg.epsilon)
	if err != nil {
		return nil, err
	}

	// Extract the min-cut certificate from final residual reachability.
	sourceSide, sinkSide := minCutFromResidual(rn, source, cfg.epsilon)

	// Return a fully self-describing result artifact.
	return &MaxFlowResult{
		Value:         value,
		Source:        source,
		Sink:          sink,
		Algorithm:     algorithm,
		Residual:      residual,
		CutSourceSide: sourceSide,
		CutSinkSide:   sinkSide,
		Augmentations: augmentations,
		Partial:       partial,
	}, nil
}

// notifyAugmentation publishes one successful augmentation event to the observer.
// It is the only diagnostic side-channel used by canonical kernels.
//
// Implementation:
//   - Stage 1: Copy the path slice so observers cannot retain mutable kernel storage.
//   - Stage 2: Print legacy verbose output only when cfg.verbose is enabled.
//   - Stage 3: Invoke cfg.observer when present and wrap observer errors.
//
// Behavior highlights:
//   - Canonical kernels never call fmt.Printf directly.
//   - Observer failure is treated as a controlled interruption.
//   - The event path may be nil for kernels that do not materialize a path witness.
//
// Inputs:
//   - ctx: active run context.
//   - cfg: finalized runtime options.
//   - event: augmentation event with algorithm, delta, total, and optional path.
//
// Returns:
//   - error: nil to continue, non-nil to interrupt the run.
//
// Errors:
//   - ErrObserverFailure joined with the observer-provided error.
//   - Context errors are not created here; callers check ctx at loop boundaries.
//
// Determinism:
//   - Observer invocation order exactly matches augmentation order.
//   - Path order, when present, is source-to-sink.
//
// Complexity:
//   - Time O(P) to copy path of length P; O(1) when event.Path is nil.
//   - Space O(P) for the detached event path.
//
// Notes:
//   - Legacy verbose output is intentionally implemented here, outside kernels.
//   - For high-performance runs, leave both Verbose and observer disabled.
//
// AI-Hints:
//   - Do not let observers mutate residual state; they receive detached data only.
//   - Do not classify observer errors by string; use errors.Is with ErrObserverFailure.
func notifyAugmentation(ctx context.Context, cfg options, event AugmentationEvent) error {
	// Detach the path before any external callback can observe it.
	if event.Path != nil {
		event.Path = append([]string(nil), event.Path...)
	}

	// Preserve legacy Verbose behavior without placing fmt.Printf inside kernels.
	if cfg.verbose {
		fmt.Printf("%s: augmenting path %v with flow %g, total %g\n",
			event.Algorithm, event.Path, event.Delta, event.Total)
	}

	// Absence of observer means the event is intentionally ignored.
	if cfg.observer == nil {
		return nil
	}

	// Observer errors stop the run through a sentinel-preserving wrapper.
	if err := cfg.observer(ctx, event); err != nil {
		return errors.Join(ErrObserverFailure, err)
	}

	return nil
}

// runMaxFlowLegacy adapts the old tuple-return API to the canonical MaxFlow facade.
// It preserves source compatibility while preventing duplicated algorithmic logic.
//
// Implementation:
//   - Stage 1: Convert Options into canonical Option values.
//   - Stage 2: Force the requested legacy algorithm explicitly.
//   - Stage 3: Delegate to MaxFlow.
//   - Stage 4: Project MaxFlowResult into the legacy tuple shape.
//
// Behavior highlights:
//   - Legacy wrappers do not contain max-flow mathematics.
//   - Sentinel classification is preserved from MaxFlow.
//   - Partial result behavior follows MaxFlowResult; legacy tuple keeps Value and Residual.
//
// Inputs:
//   - g: weighted capacity graph.
//   - source: source vertex ID.
//   - sink: sink vertex ID.
//   - legacy: old Options value.
//   - algorithm: algorithm forced by the legacy wrapper.
//
// Returns:
//   - float64: maximum flow value or partial value.
//   - *core.Graph: residual graph when published.
//   - error: nil on success or sentinel-classified failure.
//
// Errors:
//   - Same as MaxFlow, including option, validation, capacity, context, and observer errors.
//
// Determinism:
//   - Same as MaxFlow because all work is delegated to the canonical facade.
//
// Complexity:
//   - O(1) adapter overhead plus MaxFlow complexity.
//
// Notes:
//   - This helper intentionally discards cut metadata and augmentation count because
//     the old tuple API cannot represent them.
//   - New code should call MaxFlow directly.
//
// AI-Hints:
//   - Do not reintroduce old capMap logic in legacy wrappers.
//   - Do not weaken errors by converting them to plain strings.
func runMaxFlowLegacy(
	g *core.Graph,
	source, sink string,
	legacy Options,
	algorithm Algorithm,
) (float64, *core.Graph, error) {
	// Delegate to the canonical facade with explicit algorithm selection.
	result, err := MaxFlow(g, source, sink, optionsFromLegacy(legacy, algorithm)...)
	if result == nil {
		return 0, nil, err
	}

	// Project the canonical result into the legacy tuple contract.
	return result.Value, result.Residual, err
}
