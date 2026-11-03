// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dijkstra implements target-directed shortest-path operations.
// This file owns the detached, orientation-preserving graph representation.
package dijkstra

import (
	"context"
	"errors"
	"fmt"

	"github.com/lvlath/go/core"
)

// searchArc identifies one permitted, oriented traversal of an original edge.
// The edge index resolves the immutable original ID, endpoints and weight.
// Implementation:
//   - Stage 1: Resolve the stored original edge.
//   - Stage 2: Traverse from/from to/to in this exact orientation.
//   - Stage 3: Retain the original edge ID through reconstruction.
//
// Behavior highlights:
//   - Undirected edges may generate two arcs sharing one original edge index.
//
// Inputs:
//   - from, to: dense vertex indices in a searchGraph.
//   - edge: index into the corresponding searchGraph.edges slice.
//
// Returns:
//   - searchArc: an internal value, without live core pointers.
//
// Errors:
//   - None directly; the builder establishes all index invariants.
//
// Determinism:
//   - Arc ingestion follows original edge ID order, then orientation order.
//
// Complexity:
//   - Time O(1), Space O(1) per arc.
//
// Notes:
//   - The original graph is never inspected during a search kernel.
//
// AI-Hints:
//   - Do not replace the edge index with just two endpoint vertex IDs.
type searchArc struct {
	from int // Dense source vertex index in the original traversal direction.
	to   int // Dense destination vertex index in the original traversal direction.
	edge int // Index of the original edge record, never a fabricated reverse ID.
}

// searchGraph owns the frozen effective topology for one search invocation.
// It separates original edge identity from admissible directed traversal arcs.
// Implementation:
//   - Stage 1: Store sorted vertex IDs and a dense index map.
//   - Stage 2: Store detached original edge records and admitted oriented arcs.
//   - Stage 3: Index each arc once by its source and once by its destination.
//
// Behavior highlights:
//   - out and in are sparse adjacency lists, never a V-by-V matrix.
//   - Parallel edge IDs remain distinct even for equal endpoints and costs.
//
// Inputs:
//   - ids, index: identical vertex domain under sorted lexical IDs.
//   - edges: value copies of validated core.Edge records.
//   - arcs, out, in: oriented edge indices with consistent endpoints.
//
// Returns:
//   - searchGraph: private detached topology with no source Graph references.
//
// Errors:
//   - None directly; prepareSearchGraph handles input errors.
//
// Determinism:
//   - ids follow core.Vertices; edges follow core.Edges; arc order is stable.
//
// Complexity:
//   - Space O(V + E + A), with A <= 2E for the effective arcs.
//
// Notes:
//   - The original source graph may subsequently change through core mutators.
//   - Caller mutation of live Edge aliases during Clone is not supported.
//
// AI-Hints:
//   - Do not merge parallel edges, discard original IDs or use outgoing arcs for reverse search.
type searchGraph struct {
	ids   []string       // Vertex IDs in lexical order.
	index map[string]int // Dense index lookup for the sorted ID domain.
	edges []core.Edge    // Detached original edge values, including excluded walls.
	arcs  []searchArc    // Admitted oriented arcs referencing original edges.
	out   [][]int        // Arc indices grouped by original source vertex.
	in    [][]int        // The same arc indices grouped by original destination.
}

// checkSearchContext reports cancellation without manufacturing a package error.
// It retains the standard context sentinel and a distinct caller-provided cause.
// Implementation:
//   - Stage 1: Read ctx.Err after the caller has provided a non-nil context.
//   - Stage 2: Read context.Cause only when cancellation has occurred.
//   - Stage 3: Preserve both independent causes through errors.Join when needed.
//
// Behavior highlights:
//   - No partial PathResult is published on cancellation.
//
// Inputs:
//   - ctx: non-nil, caller-owned context.Context.
//
// Returns:
//   - error: nil while active, otherwise a canceled/deadline error with its cause.
//
// Errors:
//   - context.Canceled, context.DeadlineExceeded and any distinct custom cancellation cause.
//
// Determinism:
//   - Reflects the context state observed at this checkpoint.
//
// Complexity:
//   - Time O(1), Space O(1) except for error combination on cancellation.
//
// Notes:
//   - Core cloning and user callbacks remain synchronous and non-preemptive.
//
// AI-Hints:
//   - Do not convert cancellation to ErrNoPath or publish a tentative incumbent.
func checkSearchContext(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		cause := context.Cause(ctx)
		if cause != nil && !errors.Is(err, cause) {
			return errors.Join(err, cause)
		}
		return err
	}
	return nil
}

// appendSearchArc registers an oriented original edge in both adjacency indexes.
// Both indices refer to one arc record, so backward traversal retains true direction.
// Implementation:
//   - Stage 1: Append the arc to the canonical oriented arc slice.
//   - Stage 2: Index the same arc under its outgoing and incoming vertex.
//   - Stage 3: Leave the original edge record unchanged.
//
// Behavior highlights:
//   - Works for loops and parallel edges without special-case merging.
//
// Inputs:
//   - graph: assembler-owned mutable searchGraph.
//   - from, to: valid dense endpoint indices.
//   - edge: index in graph.edges.
//
// Returns:
//   - None; all state is private and mutable only during preparation.
//
// Errors:
//   - None; endpoint validation is the caller's responsibility.
//
// Determinism:
//   - Append order is exactly the ordered edge/orientation ingestion order.
//
// Complexity:
//   - Amortized O(1) time and space per arc.
//
// Notes:
//   - Incoming arcs are not the outgoing arcs of the original target.
//
// AI-Hints:
//   - Do not reverse the stored original arc when adding to the incoming index.
func appendSearchArc(graph *searchGraph, from, to, edge int) {
	arcIndex := len(graph.arcs)
	graph.arcs = append(graph.arcs, searchArc{from: from, to: to, edge: edge})
	graph.out[from] = append(graph.out[from], arcIndex)
	graph.in[to] = append(graph.in[to], arcIndex)
}

// prepareSearchGraph creates one coherent detached topology for target searches.
// Validation and all subsequent traversal use the same private core.Clone state.
// Implementation:
//   - Stage 1: Clone once and validate weighted capability and both endpoints.
//   - Stage 2: Read sorted vertices and edges; classify every edge before any wall filtering.
//   - Stage 3: Copy original edges, then build dense outgoing and incoming lists.
//
// Behavior highlights:
//   - Directed edges produce only From->To; undirected edges produce both orientations.
//   - A wall excludes arcs but cannot hide an invalid raw edge weight.
//   - Metadata is never read; parallel edges and original IDs remain intact.
//
// Inputs:
//   - ctx: non-nil invocation context, checked between preparation operations.
//   - g: non-nil weighted source graph; no writes through live aliases during Clone.
//   - sourceID, targetID: validated non-empty original vertex identifiers.
//   - threshold: validated positive threshold or +Inf in original weight units.
//
// Returns:
//   - *searchGraph: detached topology; int,int: dense source and target indices.
//   - On error, all returned graph/index values must be discarded.
//
// Errors:
//   - ErrUnweightedGraph, ErrSourceNotFound, ErrTargetNotFound for input domain.
//   - ErrInvalidWeight or ErrNegativeWeight with original edge diagnostics.
//   - core.ErrVertexNotFound for a corrupt edge endpoint in the cloned catalog.
//   - Context cancellation, preserved without a partial graph publication.
//
// Determinism:
//   - Lexical vertex IDs, lexical original edge IDs, forward then reverse orientation.
//
// Complexity:
//   - Time O(V+E) Clone, O(V log V+E log E) sorted enumeration and O(V+E) adapter work.
//   - Space O(V+E+A) in the adapter plus transient O(V+E) clone, A<=2E.
//
// Notes:
//   - Clone locks core-owned catalogs coherently against normal core mutators.
//   - Shared Metadata maps are never accessed; direct unsynchronized alias writes are unsupported.
//
// AI-Hints:
//   - Do not call Edges on the caller graph after cloning, or validate and traverse different states.
//   - Do not use graph.Directed defaults in place of per-edge Directed flags.
//   - Never convert sparse adjacency into a dense matrix merely to use another package.
func prepareSearchGraph(
	ctx context.Context,
	g *core.Graph,
	sourceID, targetID string,
	threshold float64,
) (*searchGraph, int, int, error) {
	if err := checkSearchContext(ctx); err != nil {
		return nil, 0, 0, err
	}

	// Core.Clone holds the source's topology locks together. Subsequent reads of
	// this private clone share a single graph state even though the accessors
	// individually sort their own returned slices.
	snapshot := g.Clone()
	if err := validateInputs(snapshot, sourceID); err != nil {
		return nil, 0, 0, err
	}
	if !snapshot.HasVertex(targetID) {
		return nil, 0, 0, ErrTargetNotFound
	}
	if err := checkSearchContext(ctx); err != nil {
		return nil, 0, 0, err
	}

	vertexIDs := snapshot.Vertices()
	edgeRecords := snapshot.Edges()

	// This entire pass is independent of threshold and the target. It prevents
	// an unreachable, blocked, or self-path edge from hiding invalid input.
	for _, edge := range edgeRecords {
		if err := checkSearchContext(ctx); err != nil {
			return nil, 0, 0, err
		}
		if err := classifyWeight(edge.Weight); err != nil {
			return nil, 0, 0, fmt.Errorf(
				"%w: edge_id=%q from=%q to=%q directed=%t weight=%g",
				err, edge.ID, edge.From, edge.To, edge.Directed, edge.Weight,
			)
		}
	}

	graph := &searchGraph{
		ids:   vertexIDs,
		index: make(map[string]int, len(vertexIDs)),
		edges: make([]core.Edge, 0, len(edgeRecords)),
		arcs:  make([]searchArc, 0, len(edgeRecords)),
		out:   make([][]int, len(vertexIDs)),
		in:    make([][]int, len(vertexIDs)),
	}
	for index, vertexID := range vertexIDs {
		graph.index[vertexID] = index
	}

	// Store every original record, including walls. The effective arcs point
	// only to admissible records; an edge's own ID is never regenerated.
	for _, edge := range edgeRecords {
		if err := checkSearchContext(ctx); err != nil {
			return nil, 0, 0, err
		}
		record := core.Edge{
			ID: edge.ID, From: edge.From, To: edge.To,
			Weight: edge.Weight, Directed: edge.Directed,
		}
		edgeIndex := len(graph.edges)
		graph.edges = append(graph.edges, record)
		if record.Weight >= threshold {
			continue
		}

		from, fromOK := graph.index[record.From]
		to, toOK := graph.index[record.To]
		if !fromOK || !toOK {
			return nil, 0, 0, fmt.Errorf(
				"dijkstra: cloned edge %q has a missing endpoint: %w",
				record.ID, core.ErrVertexNotFound,
			)
		}

		// Resolve direction through the same endpoint law as legacy Dijkstra.
		// For an undirected edge the second resolution uses the opposite endpoint.
		forwardTo, valid := otherEndpoint(&record, record.From)
		if !valid || forwardTo != record.To {
			return nil, 0, 0, fmt.Errorf("dijkstra: edge %q has invalid orientation: %w", record.ID, core.ErrVertexNotFound)
		}
		appendSearchArc(graph, from, to, edgeIndex)
		if !record.Directed {
			reverseTo, reverseValid := otherEndpoint(&record, record.To)
			if !reverseValid || reverseTo != record.From {
				return nil, 0, 0, fmt.Errorf("dijkstra: edge %q has invalid reverse endpoint: %w", record.ID, core.ErrVertexNotFound)
			}
			appendSearchArc(graph, to, from, edgeIndex)
		}
	}

	if err := checkSearchContext(ctx); err != nil {
		return nil, 0, 0, err
	}
	return graph, graph.index[sourceID], graph.index[targetID], nil
}
