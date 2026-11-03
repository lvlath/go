// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mst prunes nonterminal leaves from an already selected original-edge tree.
// The linear degree-queue pass preserves all terminal paths without rescanning the tree.
package mst

import (
	"context"

	"github.com/lvlath/go/core"
)

// pruneSteinerLeaves removes every removable nonterminal leaf from a connected tree.
// Each removal preserves terminal connectivity and cannot create a cycle or add an edge.
// Implementation:
//   - Stage 1: Build deterministic vertex indexes, degrees, and incident-edge lists.
//   - Stage 2: Seed a head-index queue with nonterminal degree-one vertices.
//   - Stage 3: Remove their incident edges, enqueuing neighbors that become nonterminal leaves.
//
// Behavior highlights:
//   - Terminals are never removed; zero-degree removed vertices are omitted at publication.
//   - The queue uses a head cursor and never retains a popped slice prefix.
//
// Inputs:
//   - ctx: non-nil context for bounded cooperative cancellation.
//   - vertices: sorted unique vertex IDs in the union selection domain.
//   - edges: selected tree edge values (not the entire original graph).
//   - terminals: sorted unique required terminal IDs.
//
// Returns:
//   - []core.Edge: surviving original tree edges in original selection order.
//   - error: nil on completion; no partially pruned public network on error.
//
// Errors:
//   - ErrInvalidSteinerResult on violated internal incidence invariants;
//     or the original context cancellation error.
//
// Determinism:
//   - Initial queue follows vertices; neighbors are enqueued on first degree-one transition.
//
// Complexity:
//   - Time O(Vu+Eu), Space O(Vu+Eu), excluding the earlier Kruskal selection.
//
// Notes:
//   - This function assumes a structurally valid tree; publication verifies that assumption again.
//
// AI-Hints:
//   - Do not replace the queue with repeated full-tree scans or queue=queue[1:].
//   - Do not prune any required terminal regardless of its degree.
func pruneSteinerLeaves(
	ctx context.Context,
	vertices []string,
	edges []core.Edge,
	terminals []string,
) ([]core.Edge, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	indices := make(map[string]int, len(vertices))
	for index, id := range vertices {
		indices[id] = index
	}
	isTerminal := make([]bool, len(vertices))
	for _, terminal := range terminals {
		index, exists := indices[terminal]
		if !exists {
			return nil, ErrInvalidSteinerResult
		}
		isTerminal[index] = true
	}

	degree := make([]int, len(vertices))
	incident := make([][]int, len(vertices))
	for edgeIndex, edge := range edges {
		left, leftExists := indices[edge.From]
		right, rightExists := indices[edge.To]
		if !leftExists || !rightExists || left == right {
			return nil, ErrInvalidSteinerResult
		}
		degree[left]++
		degree[right]++
		incident[left] = append(incident[left], edgeIndex)
		incident[right] = append(incident[right], edgeIndex)
	}

	queue := make([]int, 0, len(vertices))
	for index := range vertices {
		if !isTerminal[index] && degree[index] == 1 {
			queue = append(queue, index)
		}
	}
	removed := make([]bool, len(edges))
	for head := 0; head < len(queue); head++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		leaf := queue[head]
		if isTerminal[leaf] || degree[leaf] != 1 {
			continue
		}

		// Exactly one active incident edge remains at a degree-one vertex.
		// Every edge is inspected at most at both endpoints across the full pass.
		for _, edgeIndex := range incident[leaf] {
			if removed[edgeIndex] {
				continue
			}
			removed[edgeIndex] = true
			edge := edges[edgeIndex]
			neighborID := edge.From
			if vertices[leaf] == edge.From {
				neighborID = edge.To
			}
			neighbor := indices[neighborID]
			degree[leaf]--
			degree[neighbor]--
			if !isTerminal[neighbor] && degree[neighbor] == 1 {
				queue = append(queue, neighbor)
			}
			break
		}
	}

	pruned := make([]core.Edge, 0, len(edges))
	for index, edge := range edges {
		if !removed[index] {
			pruned = append(pruned, edge)
		}
	}
	return pruned, nil
}
