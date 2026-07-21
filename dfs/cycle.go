// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// DFS-based cycle detection implementation and runtime state for core.Graph.
package dfs

import (
	"fmt"
	"sort"

	"github.com/lvlath/go/core"
)

// cycleDetector owns runtime-only state during a single cycle-detection execution.
//
// Implementation:
//   - Stage 1: Hold immutable graph state and traversal metadata.
//   - Stage 2: Track DFS visitation colors and the active DFS path.
//   - Stage 3: Deduplicate canonical witness cycles before exposing them.
//
// Behavior highlights:
//   - The detector records a deterministic witness set of cycles.
//   - The detector does not promise exhaustive enumeration of all simple cycles.
//
// Inputs:
//   - N/A.
//
// Returns:
//   - N/A.
//
// Errors:
//   - N/A.
//
// Determinism:
//   - Deterministic under deterministic graph root order and neighbor order.
//
// Complexity:
//   - Space O(V + W), where V = vertex count and W = total stored witness-cycle output.
//
// Notes:
//   - The struct is internal and per-execution.
//   - Runtime state is intentionally isolated from public configuration structures.
//
// AI-Hints:
//   - Keep witness-cycle discovery state local to one execution.
//   - Do not reuse detector instances across graphs or calls.
type cycleDetector struct {
	// graph is the graph being inspected for cycles.
	graph *core.Graph

	// state stores the DFS visitation color of each vertex.
	state map[string]VertexState

	// path stores the active DFS path for cycle reconstruction.
	path []string

	// seen stores canonical closed-cycle signatures for deduplication.
	seen map[string]struct{}

	// cycles stores the canonical witness cycles discovered during traversal.
	cycles [][]string
}

// runDetectCycles reports graph cyclicity and returns a deterministic witness set of canonical cycles.
//
// Implementation:
//   - Stage 1: Validate the input graph.
//   - Stage 2: Allocate visitation state and deterministic traversal buffers.
//   - Stage 3: Launch DFS from each unvisited vertex in graph vertex order.
//   - Stage 4: Canonicalize, deduplicate, and sort witness cycles for stable output.
//
// Behavior highlights:
//   - The function returns a deterministic witness set, not an exhaustive set of all simple cycles.
//   - Directed-cycle canonicalization preserves orientation.
//   - Undirected-cycle canonicalization may consider reversed orientation as equivalent.
//   - Self-loops are reported only when looped graph policy allows them.
//
// Inputs:
//   - g: graph to inspect for cycles.
//
// Returns:
//   - *CycleDetectionResult: summary cyclicity flag plus canonical witness cycles.
//   - error: nil on success, or a graph/traversal failure.
//
// Errors:
//   - ErrGraphNil: if g is nil.
//   - ErrNeighborFetch: if graph neighbor enumeration fails.
//
// Determinism:
//   - Root order follows g.Vertices().
//   - Neighbor order follows g.Neighbors(id).
//   - Final witness-cycle order is sorted by canonical signature.
//
// Complexity:
//   - Time O(V + E + W·L), where V = vertex count, E = edge count, W = witness-cycle count,
//     and L = average witness-cycle length used during reconstruction and canonicalization.
//   - Space O(V + Lmax + W·L), where Lmax is maximum active DFS path length.
//
// Notes:
//   - The returned cycles are closed sequences of the form [v0, v1, ..., v0].
//   - Absence of a specific cycle in the result does not prove that the graph lacks every
//     mathematically possible simple cycle representation under a different enumeration strategy.
//
// AI-Hints:
//   - This function reports witness cycles, not an exhaustive set of all simple cycles.
//   - Directed cycle canonicalization must preserve edge orientation.
//   - Nil graph is an input error, not an implicit cycle-free success case.
//   - Parent-backtrack suppression for undirected traversal must use true neighbor semantics.
func runDetectCycles(g *core.Graph) (*CycleDetectionResult, error) {
	// Reject a nil graph explicitly so cycle detection follows the package-wide nil-input policy.
	if g == nil {
		return nil, ErrGraphNil
	}

	// Capture deterministic vertex order once for stable DFS-root traversal.
	vertices := g.Vertices()

	// Use the graph's reliable vertex count for preallocation of detector-owned state.
	vertexCount := g.VertexCount()

	// Build the per-execution detector runtime.
	detector := &cycleDetector{
		graph:  g,
		state:  make(map[string]VertexState, vertexCount),
		path:   make([]string, 0, vertexCount),
		seen:   make(map[string]struct{}, vertexCount),
		cycles: make([][]string, 0),
	}

	// Launch DFS from each still-unvisited root in deterministic graph vertex order.
	for _, vertexID := range vertices {
		if detector.state[vertexID] != White {
			continue
		}

		if err := detector.visit(vertexID, ""); err != nil {
			return nil, err
		}
	}

	// Sort canonical witness cycles by their canonical signatures for stable final output.
	sort.Slice(detector.cycles, func(leftIndex, rightIndex int) bool {
		return joinCycleSignature(detector.cycles[leftIndex]) < joinCycleSignature(detector.cycles[rightIndex])
	})

	// Return a stable result object even in the cycle-free case.
	if len(detector.cycles) == 0 {
		return &CycleDetectionResult{
			HasCycle: false,
			Cycles:   nil,
		}, nil
	}

	return &CycleDetectionResult{
		HasCycle: true,
		Cycles:   detector.cycles,
	}, nil
}

// visit performs DFS from currentID while reconstructing cycle witnesses from Gray back-edges.
//
// Implementation:
//   - Stage 1: Mark the current vertex as active and push it onto the DFS path.
//   - Stage 2: Read neighbors exactly once for the current vertex.
//   - Stage 3: Resolve neighbor traversal semantics per edge.
//   - Stage 4: Skip trivial undirected parent backtracks and disallowed self-loops.
//   - Stage 5: Recurse into White neighbors and record Gray back-edge witness cycles.
//   - Stage 6: Pop the current vertex and mark it fully explored.
//
// Behavior highlights:
//   - Gray-to-Gray edges indicate a back-edge into the active DFS path.
//   - Immediate undirected backtracking to the parent is not treated as a cycle witness.
//   - The method records canonical witness cycles only once per canonical signature.
//
// Inputs:
//   - currentID: current DFS vertex.
//   - parentID: immediate DFS parent, or empty string for a DFS root.
//
// Returns:
//   - error: nil on success, or a neighbor-enumeration failure.
//
// Errors:
//   - ErrNeighborFetch: if graph neighbor enumeration fails.
//
// Determinism:
//   - Deterministic under deterministic neighbor order.
//
// Complexity:
//   - Local traversal work is O(out-degree(currentID)) plus cycle-reconstruction work for Gray back-edges.
//   - Overall complexity is bounded by the public DetectCycles contract.
//
// Notes:
//   - Gray state implies that the vertex is currently present in the active path slice.
//   - A defensive impossible-state guard is retained to avoid undefined slice access if an invariant
//     is broken elsewhere in future maintenance.
//
// AI-Hints:
//   - Always resolve neighbor identity via neighborFromEdge before applying graph policy.
//   - Treat trivial undirected parent backtracking separately from real cycle witnesses.
//   - Record only canonical closed cycles in the public result set.
func (d *cycleDetector) visit(currentID, parentID string) error {
	// Mark the vertex as active in the current DFS path.
	d.state[currentID] = Gray

	// Push the vertex onto the active path for later witness reconstruction.
	d.path = append(d.path, currentID)

	// Read all graph-provided neighbors once so traversal logic uses a stable local slice.
	neighbors, err := d.graph.Neighbors(currentID)
	if err != nil {
		return fmt.Errorf("%w: neighbors(%q): %w", ErrNeighborFetch, currentID, err)
	}

	// Process neighbors in graph-provided order to preserve traversal determinism.
	for _, edge := range neighbors {
		// Resolve the actual traversal neighbor using the package-wide per-edge semantics helper.
		neighborID, ok := neighborFromEdge(edge, currentID)
		if !ok {
			continue
		}

		// Apply loop policy separately from neighbor resolution semantics.
		if neighborID == currentID && !d.graph.Looped() {
			continue
		}

		// Suppress the trivial undirected edge back to the immediate DFS parent.
		if !edge.Directed && neighborID == parentID {
			continue
		}

		// React according to the DFS color of the resolved neighbor.
		switch d.state[neighborID] {
		case White:
			// Recurse into an undiscovered neighbor.
			if err = d.visit(neighborID, currentID); err != nil {
				return err
			}

		case Gray:
			// A Gray neighbor indicates a back-edge to an ancestor in the active DFS path.
			startIndex := indexOfString(d.path, neighborID)

			// Gray implies active-path membership; this guard is purely defensive.
			if startIndex == indexNotFound {
				continue
			}

			// The active path suffix from the Gray ancestor to the current vertex forms the cycle base.
			segmentLength := len(d.path) - startIndex

			// Ignore trivial one-vertex cycles when loops are not allowed.
			if segmentLength < 2 && !d.graph.Looped() {
				continue
			}

			// Ignore trivial undirected two-step returns of the form u-v-u.
			if segmentLength == 2 && !edge.Directed {
				continue
			}

			// Record the canonical witness cycle reconstructed from the active path suffix.
			d.recordCycle(startIndex)
		}
	}

	// Pop the current vertex from the active DFS path during backtracking.
	d.path = d.path[:len(d.path)-1]

	// Mark the vertex fully explored after all reachable work is complete.
	d.state[currentID] = Black

	return nil
}

// recordCycle reconstructs, canonicalizes, and deduplicates a cycle from the active DFS path.
//
// Implementation:
//   - Stage 1: Copy the active-path suffix that forms the cycle base.
//   - Stage 2: Canonicalize the base sequence under directed or undirected policy.
//   - Stage 3: Close the cycle by appending the first canonical vertex.
//   - Stage 4: Deduplicate by canonical signature and append new witnesses.
//
// Behavior highlights:
//   - The active DFS path is never mutated by canonicalization.
//   - Directed graphs preserve forward cycle orientation during canonicalization.
//   - Globally undirected graphs allow reverse-equivalent canonicalization.
//
// Inputs:
//   - startIndex: index within the active path where the cycle base begins.
//
// Returns:
//   - N/A.
//
// Errors:
//   - N/A.
//
// Determinism:
//   - Deterministic for the same active path suffix and graph policy.
//
// Complexity:
//   - Time O(L), Space O(L), where L is the cycle base length.
//
// Notes:
//   - The stored cycle is always a closed sequence [v0, v1, ..., v0].
//
// AI-Hints:
//   - Use canonicalCycle on the open base sequence, then close the cycle afterward.
//   - Allow reverse equivalence only when the graph-level cycle model is orientation-symmetric.
func (d *cycleDetector) recordCycle(startIndex int) {
	// Copy the active-path suffix so canonicalization never aliases the detector's DFS path.
	base := append([]string(nil), d.path[startIndex:]...)

	// In globally undirected graphs, reversed orientation is equivalent for cycle identity.
	allowReverse := !d.graph.Directed()

	// Canonicalize the open cycle base under the chosen orientation policy.
	canonicalBase := canonicalCycle(base, allowReverse)

	// Close the canonical cycle by repeating the first vertex at the end.
	closed := append(append([]string(nil), canonicalBase...), canonicalBase[0])

	// Build the stable canonical signature used for deduplication and sorting.
	signature := joinCycleSignature(closed)

	// Keep only the first occurrence of each canonical signature.
	if _, exists := d.seen[signature]; exists {
		return
	}

	d.seen[signature] = struct{}{}
	d.cycles = append(d.cycles, closed)
}
