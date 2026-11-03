// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mst adapts public Dijkstra SSSP witnesses to original graph edges.
// Both Steiner memory strategies use one rounded-distance model and validate every path prefix.
package mst

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strconv"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/dijkstra"
)

// steinerClosureIDPrefix separates synthetic terminal candidates from original identity.
// The prefix never appears in the published network, regardless of original ID spelling.
const steinerClosureIDPrefix = "terminal-closure:"

// buildSteinerClosure calculates one oriented distance for each unordered terminal pair.
// It runs one public full SSSP per lower-index terminal, never one search per terminal pair.
// Implementation:
//   - Stage 1: Allocate k*(k-1)/2 candidates and format only k reusable index labels.
//   - Stage 2: Query every later terminal from one source-specific Dijkstra result.
//   - Stage 3: Retain predecessor/distance maps only in the selected storage mode.
//
// Behavior highlights:
//   - Each unordered pair i<j uses distance from terminals[i] to terminals[j].
//   - Rounded distances from reverse-direction searches are not averaged or invented.
//   - A known unreachable target is ErrDisconnectedTerminals, not an arithmetic overflow.
//
// Inputs:
//   - prepared: one immutable cloned graph and canonical terminal set.
//   - config: validated options, including context and path-retention choice.
//
// Returns:
//   - []core.Edge: synthetic closure candidates, none intended for publication.
//   - map[string]*dijkstra.Result: retained SSSP state or nil in recompute mode.
//   - error: nil on success; no partially retained artifacts on error.
//
// Errors:
//   - ErrDisconnectedTerminals for +Inf terminal distance; ErrSizeOverflow;
//     any unchanged dijkstra error or context cancellation.
//
// Determinism:
//   - Sources, targets and closure IDs follow lexical terminal indexes i<j.
//
// Complexity:
//   - Time (k-1)*Tsssp(V,E)+O(k^2); Space O(k^2+V) recompute,
//     O(k^2+kV) retain, beyond the shared cloned graph and SSSP temporary state.
//
// Notes:
//   - Legacy Dijkstra has no nested Clone, but repeats its graph validation and sorted getters.
//   - The new target Search API clones per query and rejects inexact binary sums,
//     so substituting it here would change the accepted SSSP numeric domain.
//
// AI-Hints:
//   - Do not use +Inf as a closure edge weight or translate Dijkstra overflow to unreachable.
//   - Do not call the target API k*(k-1)/2 times when a full SSSP covers many targets.
func buildSteinerClosure(
	prepared *steinerPrepared,
	config SteinerOptions,
) ([]core.Edge, map[string]*dijkstra.Result, error) {
	terminalCount := len(prepared.terminals)
	pairCount, err := checkedSteinerPairCount(terminalCount)
	if err != nil {
		return nil, nil, err
	}
	closure := make([]core.Edge, 0, pairCount)

	var retained map[string]*dijkstra.Result
	if config.PathStorage == SteinerRetainPaths {
		retained = make(map[string]*dijkstra.Result, terminalCount-1)
	}
	indexWidth := len(strconv.Itoa(terminalCount - 1))
	indexLabels := make([]string, terminalCount)
	for index := range indexLabels {
		indexLabels[index] = fmt.Sprintf("%0*d", indexWidth, index)
	}
	for sourceIndex := 0; sourceIndex < terminalCount-1; sourceIndex++ {
		if err := config.Context.Err(); err != nil {
			return nil, nil, err
		}
		source := prepared.terminals[sourceIndex]
		var options []dijkstra.Option
		if config.PathStorage == SteinerRetainPaths {
			options = append(options, dijkstra.WithPathTracking())
		}
		sssp, err := dijkstra.Dijkstra(prepared.graph, source, options...)
		if err != nil {
			return nil, nil, fmt.Errorf("mst: terminal source %q: %w", source, err)
		}
		if err := config.Context.Err(); err != nil {
			return nil, nil, err
		}

		for targetIndex := sourceIndex + 1; targetIndex < terminalCount; targetIndex++ {
			if err := config.Context.Err(); err != nil {
				return nil, nil, err
			}
			target := prepared.terminals[targetIndex]
			distance, err := sssp.DistanceTo(target)
			if err != nil {
				return nil, nil, fmt.Errorf("mst: terminal distance %q to %q: %w", source, target, err)
			}
			if math.IsInf(distance, 1) {
				return nil, nil, fmt.Errorf("mst: terminal %q cannot reach %q: %w", source, target, ErrDisconnectedTerminals)
			}
			if math.IsNaN(distance) || math.IsInf(distance, -1) || distance < 0 {
				return nil, nil, ErrInvalidWitness
			}
			// Fixed-width decimal indexes give stable IDs even when k>=10. These
			// identities exist only in the private closure, never in final edges.
			closureID := steinerClosureIDPrefix + indexLabels[sourceIndex] + ":" + indexLabels[targetIndex]
			closure = append(closure, core.Edge{
				ID: closureID, From: source, To: target, Weight: distance,
			})
		}
		if retained != nil {
			retained[source] = sssp
		}
	}
	return closure, retained, nil
}

// expandSteinerVertexPath verifies and records original edges for one SSSP vertex witness.
// The endpoint-pair minimum is chosen by (Weight, Edge.ID) and checked at each distance prefix.
// Implementation:
//   - Stage 1: Reconstruct the public vertex path with Dijkstra.Result.PathTo.
//   - Stage 2: Resolve each original edge and compare its rounded prefix with Distances.
//   - Stage 3: Add verified original Edge.ID values to the union without endpoint deduplication.
//
// Behavior highlights:
//   - Equal-cost parallel identities use the lowest Edge.ID, not an invented closure edge.
//   - The recovered ID is not claimed to be the private predecessor-edge used by legacy SSSP.
//
// Inputs:
//   - prepared: immutable edge-pair/original-ID inventory.
//   - pathResult: one source-specific public tracked SSSP result.
//   - source, target: original terminal IDs in the selected closure orientation.
//   - expected: stored closure distance in the legacy rounded SSSP domain.
//   - union: kernel-owned original Edge.ID -> core.Edge map, modified on success.
//   - ctx: non-nil caller context for cooperative validation.
//
// Returns:
//   - error: nil on a complete source-to-target original-edge witness.
//
// Errors:
//   - ErrInvalidWitness for missing/mismatched incidence, identity, or cost;
//     ErrWeightOverflow on a nonfinite local prefix; preserved Dijkstra cause; cancellation.
//
// Determinism:
//   - Forward public predecessor path, min (Weight, Edge.ID) per unordered pair.
//
// Complexity:
//   - Time O(P) plus PathTo's O(P), Space O(P) for the path; union uses O(unique IDs).
//
// Notes:
//   - An error does not publish a public network; caller drops its private union map.
//
// AI-Hints:
//   - Do not compare only final path totals: verify every intermediate DistanceTo value.
//   - Do not use edge.To as the next endpoint without checking both undirected orientations.
func expandSteinerVertexPath(
	ctx context.Context,
	prepared *steinerPrepared,
	pathResult *dijkstra.Result,
	source, target string,
	expected float64,
	union map[string]core.Edge,
) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if pathResult == nil || pathResult.SourceID != source {
		return ErrInvalidWitness
	}
	path, err := pathResult.PathTo(target)
	if err != nil {
		return fmt.Errorf("mst: reconstruct terminal path %q to %q: %w", source, target, err)
	}
	if len(path) < 2 || path[0] != source || path[len(path)-1] != target {
		return ErrInvalidWitness
	}

	prefix := float64(0)
	for index := 1; index < len(path); index++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		previousID, currentID := path[index-1], path[index]
		pair := canonicalSteinerPair(previousID, currentID)
		edge, found := prepared.minPair[pair]
		if !found || edge.ID == "" || edge.Directed ||
			!((edge.From == previousID && edge.To == currentID) ||
				(edge.From == currentID && edge.To == previousID)) {
			return fmt.Errorf("mst: no original edge for %q-%q: %w", previousID, currentID, ErrInvalidWitness)
		}
		original, exists := prepared.originals[edge.ID]
		if !exists || original != edge {
			return fmt.Errorf("mst: stale original ID %q: %w", edge.ID, ErrInvalidWitness)
		}

		next, err := checkedMSTWeightAdd(prefix, edge.Weight)
		if err != nil {
			return fmt.Errorf("mst: witness prefix %q-%q: %w", previousID, currentID, err)
		}
		distance, err := pathResult.DistanceTo(currentID)
		if err != nil {
			return fmt.Errorf("mst: witness distance %q: %w", currentID, err)
		}
		if next != distance {
			return fmt.Errorf("mst: inconsistent witness prefix at %q: %w", currentID, ErrInvalidWitness)
		}
		prefix = next
		union[edge.ID] = edge
	}
	if prefix != expected {
		return fmt.Errorf("mst: terminal distance mismatch %q to %q: %w", source, target, ErrInvalidWitness)
	}
	return nil
}

// expandSteinerClosure converts selected terminal links into one original-edge union.
// It reuses retained SSSP predecessors or executes source-grouped tracking SSSP as configured.
// Implementation:
//   - Stage 1: Group selected closure links by their canonical lower terminal source.
//   - Stage 2: Resolve and validate selected paths, deduplicating only original Edge.IDs.
//   - Stage 3: Return sorted union vertices and sorted detached original edge candidates.
//
// Behavior highlights:
//   - Retain uses previously tracked maps; recompute runs at most one new SSSP per used source.
//   - All required terminals remain in the union vertex set, including zero-cost paths.
//
// Inputs:
//   - prepared: one cloned graph with original-ID inventory.
//   - config: validated storage policy and cancellation context.
//   - selected: k-1 terminal closure edges from shared Kruskal selection.
//   - retained: source-specific tracked SSSP results in retain mode, otherwise nil.
//
// Returns:
//   - []string: sorted union vertex IDs; []core.Edge: sorted original union edges.
//   - error: non-nil when any required path cannot be verified; no public partial result.
//
// Errors:
//   - ErrInvalidWitness, underlying dijkstra errors, ErrSizeOverflow and context cancellation.
//
// Determinism:
//   - Source groups in lexical terminal order; original edge output in Edge.ID order.
//
// Complexity:
//   - Time O(L+Eu log Eu+Vu log Vu) plus <=k-1 recomputation SSSP calls.
//   - Space O(Eu+Vu+L+V) for union and one live source result in recompute mode.
//
// Notes:
//   - This stage does not create a core.Graph or copy original graph adjacency.
//
// AI-Hints:
//   - Group target requests by source; do not run k-1 redundant SSSP for repeated sources.
//   - Do not deduplicate parallel edges by their endpoint pair.
func expandSteinerClosure(
	prepared *steinerPrepared,
	config SteinerOptions,
	selected []core.Edge,
	retained map[string]*dijkstra.Result,
) ([]string, []core.Edge, error) {
	groups := make(map[string][]core.Edge, len(prepared.terminals))
	for _, link := range selected {
		groups[link.From] = append(groups[link.From], link)
	}
	union := make(map[string]core.Edge)
	vertices := make(map[string]struct{}, len(prepared.terminals))
	for _, terminal := range prepared.terminals {
		vertices[terminal] = struct{}{}
	}

	for _, source := range prepared.terminals[:len(prepared.terminals)-1] {
		links := groups[source]
		if len(links) == 0 {
			continue
		}
		if err := config.Context.Err(); err != nil {
			return nil, nil, err
		}

		var result *dijkstra.Result
		if config.PathStorage == SteinerRetainPaths {
			result = retained[source]
			if result == nil {
				return nil, nil, ErrInvalidWitness
			}
		} else {
			var err error
			result, err = dijkstra.Dijkstra(prepared.graph, source, dijkstra.WithPathTracking())
			if err != nil {
				return nil, nil, fmt.Errorf("mst: recompute witness source %q: %w", source, err)
			}
		}
		if err := config.Context.Err(); err != nil {
			return nil, nil, err
		}

		for _, link := range links {
			// Verify the distance-only closure result and the tracked witness have
			// the same represented-binary meaning before merging original edges.
			distance, err := result.DistanceTo(link.To)
			if err != nil {
				return nil, nil, fmt.Errorf("mst: witness target %q: %w", link.To, err)
			}
			if distance != link.Weight {
				return nil, nil, fmt.Errorf("mst: closure witness mismatch %q-%q: %w", link.From, link.To, ErrInvalidWitness)
			}
			if err := expandSteinerVertexPath(config.Context, prepared, result,
				link.From, link.To, link.Weight, union); err != nil {
				return nil, nil, err
			}
		}
	}

	// Union members are identified by original IDs. Derive its smallest vertex
	// domain rather than carrying isolated vertices of the original graph forward.
	unionEdges := make([]core.Edge, 0, len(union))
	for _, edge := range union {
		unionEdges = append(unionEdges, edge)
		vertices[edge.From] = struct{}{}
		vertices[edge.To] = struct{}{}
	}
	sort.Slice(unionEdges, func(i, j int) bool { return unionEdges[i].ID < unionEdges[j].ID })
	unionVertices := make([]string, 0, len(vertices))
	for id := range vertices {
		unionVertices = append(unionVertices, id)
	}
	sort.Strings(unionVertices)
	if err := config.Context.Err(); err != nil {
		return nil, nil, err
	}
	return unionVertices, unionEdges, nil
}
