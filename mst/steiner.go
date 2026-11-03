// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mst implements the Kou-Markowsky-Berman graph Steiner approximation.
// The pipeline chooses terminal connections, expands original paths, and prunes one tree.
package mst

import (
	"errors"
	"fmt"
	"math"
	"sort"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/dijkstra"
)

// runSteiner constructs the canonical network for one admitted Steiner request.
// It uses actual original edges and keeps metric-closure and intermediate MST costs unpublished.
// Implementation:
//   - Stage 1: Handle the validated singleton or two-terminal shortest-path reduction.
//   - Stage 2: Run terminal SSSP, select closure MST links, and expand their original paths.
//   - Stage 3: Select union MST, prune nonterminal leaves, verify, and publish final cost.
//
// Behavior highlights:
//   - KMB is an approximation, not a claim of exact graph Steiner optimality.
//   - Original disconnected islands are harmless; unreachable terminals are errors.
//   - Closure sum overflow cannot reject a finite expanded network because it is never computed.
//
// Inputs:
//   - prepared: owned one-clone graph with canonical terminals and original edge IDs.
//   - config: complete validated options, including storage and cooperative context.
//
// Returns:
//   - *SteinerResult: a detached terminal-covering original-edge tree; nil on error.
//
// Errors:
//   - ErrDisconnectedTerminals for missing required connectivity; ErrInvalidWitness,
//     ErrInvalidSteinerResult, ErrWeightOverflow; preserved dijkstra/core/context causes.
//
// Determinism:
//   - Lexical terminal pairs, (Weight, Edge.ID) Kruskal selections, and final ID order.
//
// Complexity:
//   - k=1: O(1) beyond admission; k=2: one tracking Tsssp(V,E)+O(P).
//   - k>=3: (k-1+g)*Tsssp(V,E)+O(k^2 log k+Eu log Eu+L+Vu+Eu),
//     where g<=k-1 in recompute mode and g=0 in retain mode.
//
// Notes:
//   - Source SSSP uses the legacy rounded-distance contract consistently in both memory modes.
//   - New dijkstra.Search has a stricter exact-binary domain and makes a Clone per query;
//     it cannot transparently replace the shared SSSP model or avoid the retained bridge.
//
// AI-Hints:
//   - Never return selected closure links or replace union MST with direct path union.
//   - Do not replace arithmetic failures by ErrDisconnectedTerminals.
func runSteiner(prepared *steinerPrepared, config SteinerOptions) (*SteinerResult, error) {
	if err := config.Context.Err(); err != nil {
		return nil, err
	}
	terminalCount := len(prepared.terminals)
	if terminalCount == 1 {
		return publishSteinerNetwork(config.Context, prepared, nil)
	}
	if terminalCount == 2 {
		return runSteinerPair(prepared, config)
	}

	// A single distance-only SSSP covers every later target for one source.
	// Retain mode keeps tracking maps for expansion; recompute holds none here.
	closure, retained, err := buildSteinerClosure(prepared, config)
	if err != nil {
		return nil, err
	}
	selectedClosure, _, err := selectKruskalEdges(
		config.Context, prepared.terminals, closure, ModeStrictTree,
	)
	if err != nil {
		// Every terminal pair was finite, so failure to connect the complete
		// closure means our own candidate adapter violated its invariant.
		if errors.Is(err, ErrDisconnected) {
			return nil, errors.Join(ErrInvalidWitness, err)
		}
		return nil, err
	}
	if len(selectedClosure) != terminalCount-1 {
		return nil, ErrInvalidWitness
	}

	// The union retains exact original IDs, not terminal-closure identities.
	// No scalar sum is calculated for selectedClosure or for the union MST.
	unionVertices, unionEdges, err := expandSteinerClosure(prepared, config, selectedClosure, retained)
	if err != nil {
		return nil, err
	}
	selectedUnion, _, err := selectKruskalEdges(
		config.Context, unionVertices, unionEdges, ModeStrictTree,
	)
	if err != nil {
		if errors.Is(err, ErrDisconnected) {
			return nil, errors.Join(ErrInvalidWitness, err)
		}
		return nil, err
	}

	pruned, err := pruneSteinerLeaves(
		config.Context, unionVertices, selectedUnion, prepared.terminals,
	)
	if err != nil {
		return nil, err
	}
	return publishSteinerNetwork(config.Context, prepared, pruned)
}

// runSteinerPair reduces the two-terminal problem to one real shortest-path witness.
// It avoids closure allocation, repeated searches, and an unnecessary union MST for k=2.
// Implementation:
//   - Stage 1: Execute one public tracked SSSP from the lower lexical terminal.
//   - Stage 2: Reconstruct and validate original edges for the target terminal.
//   - Stage 3: Verify the path tree and publish its canonical unique-edge cost.
//
// Behavior highlights:
//   - Both storage modes have the same one-run behavior for two terminals.
//   - The shortest-path witness is a simple path; it already contains no cycles or removable leaves.
//
// Inputs:
//   - prepared: clone inventory with exactly two distinct sorted terminals.
//   - config: validated context and storage policy (both use one tracking run).
//
// Returns:
//   - *SteinerResult: one original-edge shortest-path tree, or nil on failure.
//
// Errors:
//   - ErrDisconnectedTerminals for a known +Inf distance; ErrInvalidWitness,
//     ErrWeightOverflow, or preserved Dijkstra/context errors.
//
// Determinism:
//   - SSSP predecessor order and min original parallel edge (Weight, Edge.ID).
//
// Complexity:
//   - Time Tsssp(V,E)+O(P log P), Space O(V+E+P) including prepared clone.
//
// Notes:
//   - A returned path distance need not bitwise-equal its ID-ordered network total
//     under nonassociative float64 addition; each is checked in its own order.
//
// AI-Hints:
//   - Do not compute a full terminal closure or reject a path because edge-ID order changes rounding.
func runSteinerPair(prepared *steinerPrepared, config SteinerOptions) (*SteinerResult, error) {
	source, target := prepared.terminals[0], prepared.terminals[1]
	result, err := dijkstra.Dijkstra(prepared.graph, source, dijkstra.WithPathTracking())
	if err != nil {
		return nil, fmt.Errorf("mst: two-terminal source %q: %w", source, err)
	}
	if err := config.Context.Err(); err != nil {
		return nil, err
	}
	distance, err := result.DistanceTo(target)
	if err != nil {
		return nil, fmt.Errorf("mst: two-terminal target %q: %w", target, err)
	}
	if math.IsInf(distance, 1) {
		return nil, fmt.Errorf("mst: terminal %q cannot reach %q: %w", source, target, ErrDisconnectedTerminals)
	}
	if math.IsNaN(distance) || math.IsInf(distance, -1) || distance < 0 {
		return nil, ErrInvalidWitness
	}

	union := make(map[string]core.Edge)
	if err := expandSteinerVertexPath(config.Context, prepared, result,
		source, target, distance, union); err != nil {
		return nil, err
	}
	edges := make([]core.Edge, 0, len(union))
	for _, edge := range union {
		edges = append(edges, edge)
	}
	sort.Slice(edges, func(i, j int) bool { return edges[i].ID < edges[j].ID })
	return publishSteinerNetwork(config.Context, prepared, edges)
}
