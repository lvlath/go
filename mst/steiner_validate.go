// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mst prepares and verifies original-edge graph Steiner networks.
// The adapter owns one coherent clone and never publishes intermediate graph pointers.
package mst

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
	"unsafe"

	"github.com/lvlath/go/core"
)

// steinerEndpointPair indexes an undirected original endpoint relation.
// Its lexical canonicalization lets different stored edge orientations share one lookup.
type steinerEndpointPair struct {
	left, right string
}

// steinerPrepared owns the immutable inventory used by one KMB construction.
// Its edge map and pair minima are detached core.Edge values from one core.Clone.
type steinerPrepared struct {
	// graph is the sole cloned topology consumed by all public SSSP runs.
	graph *core.Graph
	// vertices and terminals are separate canonical ID slices.
	vertices  []string
	terminals []string
	// originals includes every original edge, even loops excluded from KMB selection.
	originals map[string]core.Edge
	// minPair selects one original parallel record by (Weight, Edge.ID).
	minPair map[steinerEndpointPair]core.Edge
}

// canonicalSteinerPair returns a comparable undirected endpoint key.
// It never changes original Edge.From/To direction or identity.
// Implementation:
//   - Stage 1: Compare the two original vertex IDs.
//   - Stage 2: Put the smaller ID into left and larger into right.
//
// Behavior highlights:
//   - Stored edge orientation and requested traversal direction do not affect the key.
//
// Inputs:
//   - left, right: two original vertex IDs, caller-owned immutable strings.
//
// Returns:
//   - steinerEndpointPair: canonical map key.
//
// Errors:
//   - None; endpoint existence is validated by callers.
//
// Determinism:
//   - Exact lexical string comparison.
//
// Complexity:
//   - Time O(1) ID comparisons, Space O(1).
//
// Notes:
//   - Self-loop keys are possible but self-loops are not entered into minPair.
//
// AI-Hints:
//   - Never identify parallel edges solely by this key in the returned network.
func canonicalSteinerPair(left, right string) steinerEndpointPair {
	if left > right {
		left, right = right, left
	}
	return steinerEndpointPair{left: left, right: right}
}

// prepareSteiner validates the entire original input and builds one immutable KMB inventory.
// It rejects invalid weights in loops or unrelated islands before any singleton/SSSP fast-path.
// Implementation:
//   - Stage 1: Check existing MST graph policy, terminal shape, and explicit input caps.
//   - Stage 2: Take one coherent core.Clone and verify membership and all numeric weights.
//   - Stage 3: Build original Edge.ID and minimal endpoint-pair indexes from detached values.
//
// Behavior highlights:
//   - Terminal IDs are copied and sorted, never reordered in the caller slice.
//   - Negative finite edges are rejected here but remain valid for ordinary MST/MSF.
//   - Metadata is never accessed, including shared shallow clone metadata.
//
// Inputs:
//   - graph: stable caller-owned weighted undirected core.Graph.
//   - terminalIDs: nonempty caller-owned terminal list.
//   - config: already validated SteinerOptions, read-only.
//
// Returns:
//   - *steinerPrepared: one cloned topology plus immutable detached inventory.
//
// Errors:
//   - Existing graph-policy errors, ErrEmptyTerminals, ErrDuplicateTerminal,
//     core.ErrEmptyVertexID, core.ErrVertexNotFound, ErrNaNInfWeight,
//     ErrNegativeWeight, ErrResourceLimit, ErrSizeOverflow, and context cancellation.
//
// Determinism:
//   - Graph capability order, lexical terminal IDs, then core.Edges() ID order.
//
// Complexity:
//   - Time O(V log V + E log E + k log k + V+E) including clone/catalog sorting.
//   - Space O(V+E+k) beyond the original graph, including one cloned graph and indexes.
//
// Notes:
//   - Caller must not mutate graph topology/weights during this operation.
//   - Preflight caps rely on that quiescence; no claim of process RSS control is made.
//
// AI-Hints:
//   - Do not replace core.Clone with separate live Vertices()/Edges() reads.
//   - Validate loops and all disconnected islands before skipping irrelevant candidates.
//
// nolint:gocyclo
func prepareSteiner(graph *core.Graph, terminalIDs []string, config SteinerOptions) (*steinerPrepared, error) {
	if err := validateMSTGraphPolicy(graph); err != nil {
		return nil, err
	}
	if len(terminalIDs) == 0 {
		return nil, ErrEmptyTerminals
	}
	if config.MaxTerminals > 0 && len(terminalIDs) > config.MaxTerminals {
		return nil, ErrResourceLimit
	}
	if _, err := checkedSteinerSliceCapacity(len(terminalIDs), unsafe.Sizeof("")); err != nil {
		return nil, err
	}

	// Empty and duplicate terminal IDs are independent input errors. Resolve
	// these before membership, which must observe only the coherent clone.
	terminals := append([]string(nil), terminalIDs...)
	sort.Strings(terminals)
	for index, id := range terminals {
		if err := config.Context.Err(); err != nil {
			return nil, err
		}
		if id == "" {
			return nil, core.ErrEmptyVertexID
		}
		if index > 0 && id == terminals[index-1] {
			return nil, ErrDuplicateTerminal
		}
	}
	if config.MaxVertices > 0 && graph.VertexCount() > config.MaxVertices {
		return nil, ErrResourceLimit
	}
	if config.MaxEdges > 0 && graph.EdgeCount() > config.MaxEdges {
		return nil, ErrResourceLimit
	}
	if _, err := checkedSteinerPairCount(len(terminals)); err != nil {
		return nil, err
	}

	if err := config.Context.Err(); err != nil {
		return nil, err
	}
	clone := graph.Clone()
	vertices := clone.Vertices()
	for _, id := range terminals {
		if !clone.HasVertex(id) {
			return nil, fmt.Errorf("mst: missing Steiner terminal %q: %w", id, core.ErrVertexNotFound)
		}
	}

	edges := clone.Edges()
	prepared := &steinerPrepared{
		graph:     clone,
		vertices:  vertices,
		terminals: terminals,
		originals: make(map[string]core.Edge, len(edges)),
		minPair:   make(map[steinerEndpointPair]core.Edge, len(edges)),
	}

	// core.Edges() is ordered by original ID. Examine every edge before applying
	// the self-loop exclusion, including islands with no required terminal.
	for _, pointer := range edges {
		if err := config.Context.Err(); err != nil {
			return nil, err
		}
		edge := *pointer
		if math.IsNaN(edge.Weight) || math.IsInf(edge.Weight, 0) {
			return nil, fmt.Errorf("mst: original edge %q: %w", edge.ID, ErrNaNInfWeight)
		}
		if edge.Weight < 0 {
			return nil, fmt.Errorf("mst: original edge %q: %w", edge.ID, ErrNegativeWeight)
		}
		if edge.Directed {
			return nil, errors.Join(ErrInvalidGraph, ErrDirectedEdge)
		}
		if edge.ID == "" || edge.From == "" || edge.To == "" ||
			!clone.HasVertex(edge.From) || !clone.HasVertex(edge.To) {
			return nil, fmt.Errorf("mst: malformed original edge %q: %w", edge.ID, ErrInvalidWitness)
		}
		if _, exists := prepared.originals[edge.ID]; exists {
			return nil, fmt.Errorf("mst: repeated original edge %q: %w", edge.ID, ErrInvalidWitness)
		}
		prepared.originals[edge.ID] = edge
		if edge.From == edge.To {
			continue
		}

		pair := canonicalSteinerPair(edge.From, edge.To)
		best, exists := prepared.minPair[pair]
		if !exists || lessMSTEdge(edge, best) {
			prepared.minPair[pair] = edge
		}
	}
	if err := config.Context.Err(); err != nil {
		return nil, err
	}
	return prepared, nil
}

// publishSteinerNetwork canonicalizes, verifies, and publishes one original-edge tree.
// Its numeric fold occurs only after the final original network has been selected and pruned.
// Implementation:
//   - Stage 1: Copy and sort original selected edges by Edge.ID.
//   - Stage 2: Derive only used vertices and verify every tree/original-ID invariant.
//   - Stage 3: Calculate a checked final cost and publish detached slices.
//
// Behavior highlights:
//   - No closure or preliminary union total is accumulated.
//   - A singleton retains exactly its required vertex and no edges.
//
// Inputs:
//   - ctx: cancellation policy; prepared: immutable one-clone inventory.
//   - selected: kernel-owned original edges, not modified by this function.
//
// Returns:
//   - *SteinerResult: verified detached terminal-covering network; nil on error.
//
// Errors:
//   - ErrInvalidSteinerResult, ErrInvalidWitness, ErrWeightOverflow,
//     ErrSizeOverflow, or context cancellation.
//
// Determinism:
//   - Final edge order is original Edge.ID; vertices and terminals are lexical.
//
// Complexity:
//   - Time O(Es log Es + Vs log Vs + Es*alpha(Vs)), Space O(Vs+Es).
//
// Notes:
//   - Validation establishes structure and provenance against the prepared clone,
//     not global optimality of the selected Steiner approximation.
//
// AI-Hints:
//   - Do not trust selected edges just because the two private MST phases succeeded.
func publishSteinerNetwork(
	ctx context.Context,
	prepared *steinerPrepared,
	selected []core.Edge,
) (*SteinerResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if _, err := checkedSteinerSliceCapacity(len(selected), unsafe.Sizeof(core.Edge{})); err != nil {
		return nil, err
	}
	edges := append([]core.Edge(nil), selected...)
	sort.Slice(edges, func(i, j int) bool { return edges[i].ID < edges[j].ID })

	usedCapacity, err := checkedSteinerCountAdd(len(edges), len(prepared.terminals))
	if err != nil {
		return nil, err
	}
	used := make(map[string]struct{}, usedCapacity)
	for _, terminal := range prepared.terminals {
		used[terminal] = struct{}{}
	}
	for _, edge := range edges {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		used[edge.From] = struct{}{}
		used[edge.To] = struct{}{}
	}
	if _, err := checkedSteinerSliceCapacity(len(used), unsafe.Sizeof("")); err != nil {
		return nil, err
	}
	vertices := make([]string, 0, len(used))
	for id := range used {
		vertices = append(vertices, id)
	}
	sort.Strings(vertices)

	result := &SteinerResult{
		Terminals: append([]string(nil), prepared.terminals...),
		Vertices:  vertices,
		Edges:     edges,
	}
	if err := verifySteinerNetwork(ctx, prepared, result); err != nil {
		return nil, err
	}
	total, err := checkedMSTEdgeTotal(result.Edges)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	result.TotalWeight = total
	return result, nil
}

// verifySteinerNetwork proves that one candidate is a valid original-edge terminal tree.
// It checks the graph-theoretic result contract without assuming or proving optimality.
// Implementation:
//   - Stage 1: Check canonical IDs, cardinality, terminal coverage, and original-edge identity.
//   - Stage 2: Use the existing DSU to reject cycles and verify one connected component.
//   - Stage 3: Reject any nonterminal leaf left by pruning.
//
// Behavior highlights:
//   - Unique parallel Edge.ID values remain distinguishable even with the same endpoints.
//   - The original edge must agree exactly on ID, stored endpoints, weight, and directedness.
//
// Inputs:
//   - ctx: cancellation policy; prepared: immutable original catalog.
//   - network: kernel-owned canonical result before TotalWeight publication.
//
// Returns:
//   - error: nil only when every structural invariant holds.
//
// Errors:
//   - ErrInvalidSteinerResult for topology failures, ErrInvalidWitness for foreign records,
//     or an unchanged context cancellation error.
//
// Determinism:
//   - Lexical vertex/terminal order and original Edge.ID order; no map iteration affects decisions.
//
// Complexity:
//   - Time O(Vs+Es*(alpha(Vs)+log Vs)), Space O(Vs+Es) for sets and DSU.
//
// Notes:
//   - A future caller-owned walk must independently validate mutable public results.
//
// AI-Hints:
//   - Do not mistake an edge-count equality alone for a connectivity proof.
//
// nolint:gocyclo
func verifySteinerNetwork(ctx context.Context, prepared *steinerPrepared, network *SteinerResult) error {
	if network == nil || len(network.Terminals) == 0 || len(network.Vertices) == 0 {
		return ErrInvalidSteinerResult
	}
	if len(network.Edges) != len(network.Vertices)-1 {
		return ErrInvalidSteinerResult
	}
	for index, id := range network.Vertices {
		if id == "" || (index > 0 && network.Vertices[index-1] >= id) {
			return ErrInvalidSteinerResult
		}
	}
	for index, terminal := range network.Terminals {
		if terminal == "" || (index > 0 && network.Terminals[index-1] >= terminal) ||
			!containsSteinerSortedID(network.Vertices, terminal) {
			return ErrInvalidSteinerResult
		}
	}

	set := newDisjointSet(network.Vertices)
	degree := make(map[string]int, len(network.Vertices))
	terminalSet := make(map[string]bool, len(network.Terminals))
	for _, terminal := range network.Terminals {
		terminalSet[terminal] = true
	}
	for index, edge := range network.Edges {
		if err := ctx.Err(); err != nil {
			return err
		}
		if edge.ID == "" || (index > 0 && network.Edges[index-1].ID >= edge.ID) ||
			edge.From == edge.To || edge.Directed ||
			!containsSteinerSortedID(network.Vertices, edge.From) ||
			!containsSteinerSortedID(network.Vertices, edge.To) ||
			math.IsNaN(edge.Weight) || math.IsInf(edge.Weight, 0) || edge.Weight < 0 {
			return ErrInvalidSteinerResult
		}
		original, exists := prepared.originals[edge.ID]
		if !exists || original != edge {
			return fmt.Errorf("mst: non-original edge %q: %w", edge.ID, ErrInvalidWitness)
		}
		if !set.union(edge.From, edge.To) {
			return ErrInvalidSteinerResult
		}
		degree[edge.From]++
		degree[edge.To]++
	}

	// All selected vertices must belong to the same component, including the
	// zero-edge singleton. Every retained nonterminal must have degree >= 2.
	root := set.find(network.Vertices[0])
	for _, id := range network.Vertices {
		if err := ctx.Err(); err != nil {
			return err
		}
		if set.find(id) != root || (!terminalSet[id] && degree[id] <= 1) {
			return ErrInvalidSteinerResult
		}
	}
	return nil
}

// containsSteinerSortedID tests membership in a canonical ascending ID slice.
// It avoids maintaining a second mutable vertex index in the published network.
// Implementation:
//   - Stage 1: Binary-search the lexical lower bound.
//   - Stage 2: Check exact string equality at the returned position.
//
// Behavior highlights:
//   - Empty slices return false without allocations.
//
// Inputs:
//   - ids: sorted unique vertex ID slice; id: candidate original identifier.
//
// Returns:
//   - bool: membership result.
//
// Errors:
//   - None.
//
// Determinism:
//   - Exact string equality after lexical search.
//
// Complexity:
//   - Time O(log V), Space O(1).
//
// Notes:
//   - Sorting is performed before this helper is called.
//
// AI-Hints:
//   - Do not use unsorted caller-owned IDs with this helper.
func containsSteinerSortedID(ids []string, id string) bool {
	position := sort.SearchStrings(ids, id)
	return position < len(ids) && ids[position] == id
}
