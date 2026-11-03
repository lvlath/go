// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mst defines validation of caller-owned Steiner tree results before traversal.
// This file implements local canonicalization, structural checks, and sparse adjacency preparation.
package mst

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"unsafe"

	"github.com/lvlath/go/core"
)

// steinerWalkArc identifies one direction of an original undirected tree edge.
// Each non-loop edge appears twice in the local adjacency, once from each endpoint.
// Implementation:
//   - Stage 1: Resolve vertex and edge indexes from the validated tree.
//   - Stage 2: Store the neighboring vertex index and original edge-record index.
//   - Stage 3: Sort each vertex's arcs by neighbor ID, then original edge ID.
//
// Behavior highlights:
//   - The edge index is an identity, not the numeric weight or a synthetic arc ID.
//
// Inputs:
//   - to: index in steinerWalkTree.vertices; edge: index in its sorted edges.
//
// Returns:
//   - steinerWalkArc: immutable O(1)-sized traversal relation.
//
// Errors:
//   - Invalid indexes are rejected by the validating constructor, not by this value.
//
// Determinism:
//   - Neighbor order is established by prepareSteinerWalkTree.
//
// Complexity:
//   - Space O(1) per directed adjacency occurrence.
//
// Notes:
//   - This is private workspace; it never escapes in WalkResult.
//
// AI-Hints:
//   - Resolve the opposite endpoint relative to the current vertex; Edge.To is not a universal neighbor.
type steinerWalkArc struct {
	// to is the vertex reached when traversing this arc from its owning adjacency.
	to int
	// edge is the original edge index used for this step, shared by both arc directions.
	edge int
}

// steinerWalkTree is the immutable, canonical, local view of a supplied SteinerResult.
// It owns detached values and a sparse adjacency suitable for an iterative tree traversal.
// Implementation:
//   - Stage 1: Canonicalize and validate the caller's vertices, terminals, and edges.
//   - Stage 2: Verify one connected acyclic tree and its canonical unique-edge cost.
//   - Stage 3: Build sorted, bidirectional adjacency indexed by local vertices.
//
// Behavior highlights:
//   - No KMB provenance or original graph pointer is needed for a valid covering walk.
//
// Inputs:
//   - Constructed from the caller-owned tree passed to prepareSteinerWalkTree.
//
// Returns:
//   - steinerWalkTree: read-only local workspace with a certified network total.
//
// Errors:
//   - Construction returns ErrInvalidSteinerResult or the applicable numeric error.
//
// Determinism:
//   - Lexical vertex IDs, original Edge.ID order, then (neighbor ID, edge ID) adjacency.
//
// Complexity:
//   - Space O(V+E), excluding the caller's SteinerResult.
//
// Notes:
//   - Caller must not mutate the input result concurrently with its preparation.
//
// AI-Hints:
//   - Do not trust exported SteinerResult slices or TotalWeight: callers can mutate them after SteinerTree.
type steinerWalkTree struct {
	// vertices is the sorted set of network vertex IDs, including all required terminals.
	vertices []string
	// indices maps each vertex ID to its index in vertices, without relying on map iteration order.
	indices map[string]int
	// edges contains one detached record per original edge, in ascending Edge.ID order.
	edges []core.Edge
	// adjacency lists each undirected edge at both endpoints in canonical neighbor order.
	adjacency [][]steinerWalkArc
	// networkWeight is the checked Edge.ID-ordered sum, independently verified against the input.
	networkWeight float64
}

// prepareSteinerWalkTree verifies a mutable public tree and builds detached indexed adjacency.
// It intentionally accepts valid permutations and nonterminal leaves, unlike KMB's pruned output.
// Implementation:
//   - Stage 1: Check cardinalities and canonicalize caller-owned IDs/edge copies.
//   - Stage 2: Validate domain, incidence, unique IDs, DSU acyclicity/connectivity, and total.
//   - Stage 3: Prepare deterministically ordered sparse adjacency for the walk kernel.
//
// Behavior highlights:
//   - All edge weights must be finite and nonnegative, including zero-weight edges.
//   - A singleton has one vertex, at least one terminal, no edges, and zero total.
//   - Exact equality checks the public total against an Edge.ID-ordered float64 fold.
//
// Inputs:
//   - network: caller-owned SteinerResult that must remain unchanged for the duration of this call.
//
// Returns:
//   - *steinerWalkTree: detached canonical workspace, nil on any error.
//
// Errors:
//   - ErrNilResult for nil input; ErrInvalidSteinerResult for malformed topology/total;
//     ErrWeightOverflow for a nonfinite canonical prefix; ErrSizeOverflow for impossible sizes;
//     joined core ID/domain sentinels when their specific condition is encountered.
//
// Determinism:
//   - Local copies are sorted by vertex/terminal IDs and original Edge.ID before checking cost.
//   - Adjacency scans use (neighbor vertex ID, original Edge.ID), independent of input slice order.
//
// Complexity:
//   - Time O(V log V + E log E + E*alpha(V) + sum_v deg(v)log(deg(v)+1)),
//     Space O(V+E), including copies, DSU, and mirrored sparse adjacency.
//
// Notes:
//   - Structural validity is not a proof of optimality, KMB provenance, or original-graph identity.
//   - This method never mutates the caller's network or rewrites its stale TotalWeight.
//
// AI-Hints:
//   - Do not require nonterminal leaves to be absent: a covering walk must traverse them.
//   - Do not sum in caller edge order; a valid permutation must not change cost admission.
//   - Validate the entire tree before using its edges as a step-by-step route witness.
//
// nolint:gocyclo
func prepareSteinerWalkTree(network *SteinerResult) (*steinerWalkTree, error) {
	if network == nil {
		return nil, ErrNilResult
	}
	vertexCount := len(network.Vertices)
	edgeCount := len(network.Edges)

	// An acyclic connected graph on V declared vertices has exactly V-1 edges.
	// Checking cardinality first keeps malformed shapes from allocating traversal state.
	if vertexCount == 0 || len(network.Terminals) == 0 || edgeCount != vertexCount-1 {
		return nil, ErrInvalidSteinerResult
	}
	if _, err := checkedSteinerSliceCapacity(vertexCount, unsafe.Sizeof("")); err != nil {
		return nil, err
	}
	if _, err := checkedSteinerSliceCapacity(len(network.Terminals), unsafe.Sizeof("")); err != nil {
		return nil, err
	}
	if _, err := checkedSteinerSliceCapacity(edgeCount, unsafe.Sizeof(core.Edge{})); err != nil {
		return nil, err
	}

	// A covering walk accepts valid order permutations, but never reorders the
	// input: sorting copies makes both validation and later traversal stable.
	vertices := append([]string(nil), network.Vertices...)
	sort.Strings(vertices)
	indices := make(map[string]int, vertexCount)
	for index, vertexID := range vertices {
		if vertexID == "" {
			return nil, errors.Join(ErrInvalidSteinerResult, core.ErrEmptyVertexID)
		}
		if index > 0 && vertices[index-1] == vertexID {
			return nil, fmt.Errorf("mst: repeated walk vertex %q: %w", vertexID, ErrInvalidSteinerResult)
		}
		indices[vertexID] = index
	}

	terminals := append([]string(nil), network.Terminals...)
	sort.Strings(terminals)
	for index, terminalID := range terminals {
		if terminalID == "" {
			return nil, errors.Join(ErrInvalidSteinerResult, core.ErrEmptyVertexID)
		}
		if index > 0 && terminals[index-1] == terminalID {
			return nil, fmt.Errorf("mst: repeated walk terminal %q: %w", terminalID, ErrInvalidSteinerResult)
		}
		if _, exists := indices[terminalID]; !exists {
			return nil, fmt.Errorf("mst: walk terminal %q is not a tree vertex: %w",
				terminalID, errors.Join(ErrInvalidSteinerResult, core.ErrVertexNotFound))
		}
	}

	edges := append([]core.Edge(nil), network.Edges...)
	sort.Slice(edges, func(i, j int) bool {
		return edges[i].ID < edges[j].ID
	})
	set := newDisjointSet(vertices)
	for index, edge := range edges {
		// Original edge identities are unique. Parallel records between identical
		// endpoints constitute a cycle, and therefore cannot both appear in a tree.
		if edge.ID == "" {
			return nil, errors.Join(ErrInvalidSteinerResult, core.ErrEmptyEdgeID)
		}
		if index > 0 && edges[index-1].ID == edge.ID {
			return nil, fmt.Errorf("mst: repeated walk edge %q: %w", edge.ID, ErrInvalidSteinerResult)
		}
		if edge.From == "" || edge.To == "" {
			return nil, fmt.Errorf("mst: walk edge %q has empty endpoint: %w",
				edge.ID, errors.Join(ErrInvalidSteinerResult, core.ErrEmptyVertexID))
		}
		if edge.Directed {
			return nil, fmt.Errorf("mst: walk edge %q is directed: %w",
				edge.ID, errors.Join(ErrInvalidSteinerResult, ErrDirectedEdge))
		}
		if edge.From == edge.To {
			return nil, fmt.Errorf("mst: walk edge %q is a self-loop: %w", edge.ID, ErrInvalidSteinerResult)
		}
		if math.IsNaN(edge.Weight) || math.IsInf(edge.Weight, 0) {
			return nil, fmt.Errorf("mst: walk edge %q has nonfinite weight: %w",
				edge.ID, errors.Join(ErrInvalidSteinerResult, ErrNaNInfWeight))
		}
		if edge.Weight < 0 {
			return nil, fmt.Errorf("mst: walk edge %q has negative weight: %w",
				edge.ID, errors.Join(ErrInvalidSteinerResult, ErrNegativeWeight))
		}
		if _, leftExists := indices[edge.From]; !leftExists {
			return nil, fmt.Errorf("mst: walk edge %q refers to missing vertex %q: %w",
				edge.ID, edge.From, errors.Join(ErrInvalidSteinerResult, core.ErrVertexNotFound))
		}
		if _, rightExists := indices[edge.To]; !rightExists {
			return nil, fmt.Errorf("mst: walk edge %q refers to missing vertex %q: %w",
				edge.ID, edge.To, errors.Join(ErrInvalidSteinerResult, core.ErrVertexNotFound))
		}
		if !set.union(edge.From, edge.To) {
			return nil, fmt.Errorf("mst: walk edge %q introduces a cycle: %w", edge.ID, ErrInvalidSteinerResult)
		}
	}

	// The edge-count law and cycle check imply connectivity, but explicitly
	// comparing DSU representatives also guards against future validator changes.
	representative := set.find(vertices[0])
	for _, vertexID := range vertices[1:] {
		if set.find(vertexID) != representative {
			return nil, fmt.Errorf("mst: walk tree is disconnected at %q: %w", vertexID, ErrInvalidSteinerResult)
		}
	}

	// The caller's TotalWeight is not trusted. Check the complete structure
	// first, then recalculate the same ID-ordered fold used by SteinerTree.
	if math.IsNaN(network.TotalWeight) || math.IsInf(network.TotalWeight, 0) || network.TotalWeight < 0 {
		return nil, fmt.Errorf("mst: nonfinite or negative public network cost: %w", ErrInvalidSteinerResult)
	}
	networkWeight, err := checkedMSTEdgeTotal(edges)
	if err != nil {
		return nil, fmt.Errorf("mst: walk network cost: %w", err)
	}
	if networkWeight != network.TotalWeight {
		return nil, fmt.Errorf("mst: stale public network cost: %w", ErrInvalidSteinerResult)
	}

	// Allocate only after the result passes semantic and numeric checks.
	// Every original edge creates exactly two adjacency arcs, never a new edge ID.
	arcCount, err := checkedSteinerCountAdd(edgeCount, edgeCount)
	if err != nil {
		return nil, err
	}
	if _, err := checkedSteinerSliceCapacity(arcCount, unsafe.Sizeof(steinerWalkArc{})); err != nil {
		return nil, err
	}
	if _, err := checkedSteinerSliceCapacity(vertexCount, unsafe.Sizeof([]steinerWalkArc{})); err != nil {
		return nil, err
	}
	adjacency := make([][]steinerWalkArc, vertexCount)
	for edgeIndex, edge := range edges {
		left, right := indices[edge.From], indices[edge.To]
		adjacency[left] = append(adjacency[left], steinerWalkArc{to: right, edge: edgeIndex})
		adjacency[right] = append(adjacency[right], steinerWalkArc{to: left, edge: edgeIndex})
	}
	for vertexIndex := range adjacency {
		neighbors := adjacency[vertexIndex]
		sort.Slice(neighbors, func(i, j int) bool {
			leftID := vertices[neighbors[i].to]
			rightID := vertices[neighbors[j].to]
			if leftID != rightID {
				return leftID < rightID
			}
			return edges[neighbors[i].edge].ID < edges[neighbors[j].edge].ID
		})
	}
	return &steinerWalkTree{
		vertices: vertices, indices: indices, edges: edges,
		adjacency: adjacency, networkWeight: networkWeight,
	}, nil
}
