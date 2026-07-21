// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp defines the local matching problem used by Christofides.
// The model copies odd-set distances into a deterministic local complete graph
// so matching engines do not depend on external matrix ownership or map order.
package tsp

import (
	"github.com/lvlath/go/matrix"
)

// matchingUnmatched marks a local matching vertex that has not been paired.
const matchingUnmatched = -1

// matchingProblem stores the complete graph induced by odd-degree vertices.
// Vertex indices in this structure are local [0..k-1] positions; odd maps them
// back to original TSP matrix vertices.
//
// Implementation:
//   - Stage 1: buildMatchingProblem validates odd-set shape and copies costs.
//   - Stage 2: exactSmallPerfectMatching solves over local indices.
//   - Stage 3: appendPerfectMatching maps local pairs back to original vertices.
//
// Behavior highlights:
//   - Detached from the source matrix.
//   - Row-major local cost layout: w[i*n+j].
//   - Does not mutate Christofides adjacency.
//
// Inputs:
//   - Built from odd-degree MST vertices and the final symmetric distance matrix.
//
// Returns:
//   - Consumed by exact matching helpers.
//
// Errors:
//   - Construction errors are returned by buildMatchingProblem.
//
// Determinism:
//   - Local vertex order preserves the odd slice order produced by MST degree scan.
//
// Complexity:
//   - Space O(k^2), where k=len(odd).
//
// Notes:
//   - This is a matching-local data model, not a public graph representation.
//
// AI-Hints:
//   - Do not expose matchingProblem from public APIs.
//   - Do not reorder odd vertices unless all tie-break tests are updated.
type matchingProblem struct {
	// odd maps local matching indices back to original TSP matrix vertices.
	// odd[local] is used only when appending final matching edges to Christofides adjacency.
	odd []int

	// n is the number of local odd-degree vertices.
	// It must be even for a perfect matching.
	n int

	// w stores local edge costs in row-major n*n layout.
	// w[i*n+j] is the cost between local vertices i and j.
	w []float64
}

// at returns the local matching cost between two local odd-set vertices.
//
// Implementation:
//   - Stage 1: Compute row-major offset i*n+j.
//   - Stage 2: Return the stored cost.
//
// Behavior highlights:
//   - No allocation.
//   - No matrix interface dispatch.
//
// Inputs:
//   - i: local source index.
//   - j: local target index.
//
// Returns:
//   - float64: local complete-graph matching edge cost.
//
// Errors:
//   - None. Callers must pass validated local indices.
//
// Determinism:
//   - Pure indexed lookup.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Diagonal values are unused by the matching DP.
//
// AI-Hints:
//   - Do not use original TSP vertex IDs as indices here; use local indices.
func (p matchingProblem) at(i int, j int) float64 {
	return p.w[i*p.n+j]
}

// matchingCost computes the exact cost of a verified local perfect matching.
// It sums every pair once, using i<match[i] to avoid double-counting symmetric pairs.
//
// Implementation:
//   - Stage 1: Verify the local matching structure.
//   - Stage 2: Scan local vertices in increasing order.
//   - Stage 3: Sum problem.at(i, match[i]) only for i<match[i].
//   - Stage 4: Stabilize the returned cost with round1e9.
//
// Behavior highlights:
//   - Does not mutate problem or match.
//   - Does not inspect adjacency.
//   - Uses the matching-local detached cost matrix.
//
// Inputs:
//   - problem: local complete matching instance.
//   - match: local symmetric perfect matching array.
//
// Returns:
//   - float64: rounded matching cost.
//   - error: nil on valid perfect matching.
//
// Errors:
//   - ErrInvalidMatching from verifyPerfectMatching.
//
// Determinism:
//   - Fixed increasing local-index scan.
//
// Complexity:
//   - Time O(k), Space O(1).
//
// Notes:
//   - The cost is used for verification and tests; Christofides does not need it for adjacency mutation.
//
// AI-Hints:
//   - Do not sum both i→j and j→i.
//   - Do not recompute costs from the original matrix here.
func matchingCost(problem matchingProblem, match []int) (float64, error) {
	if err := verifyPerfectMatching(match); err != nil {
		return 0, err
	}
	if len(problem.odd) != len(match) || problem.n != len(match) {
		return 0, ErrInvalidMatching
	}

	total := 0.0

	for localVertex, partner := range match {
		if localVertex < partner {
			total += problem.at(localVertex, partner)
		}
	}

	return round1e9(total), nil
}

// buildMatchingProblem builds the local complete graph induced by odd vertices.
// It validates local matching shape and copies costs from the shared TSP weight firewall.
//
// Implementation:
//   - Stage 1: Validate even cardinality and copy odd vertices.
//   - Stage 2: Copy the final symmetric complete distance matrix through copyCompleteWeights.
//   - Stage 3: Reject duplicate or out-of-range odd vertices.
//   - Stage 4: Fill a k×k local row-major cost buffer.
//
// Behavior highlights:
//   - Preserves odd order for deterministic tie-breaking.
//   - Rejects malformed odd sets before any adjacency mutation.
//   - Uses weightBuffer instead of repeated matrix.At calls.
//
// Inputs:
//   - odd: odd-degree vertices from MST degree scan.
//   - dist: final symmetric complete TSP matrix.
//
// Returns:
//   - matchingProblem: detached local MWPM instance.
//   - error: nil on success or sentinel-classified failure.
//
// Errors:
//   - ErrInvalidMatching for odd cardinality, duplicates, or out-of-range vertices.
//   - ErrNilDistanceMatrix, ErrNonSquare, ErrNaNInf, ErrNegativeWeight,
//     ErrIncompleteGraph, ErrAsymmetry from copyCompleteWeights.
//
// Determinism:
//   - Fixed increasing local-index scan.
//
// Complexity:
//   - Time O(n^2+k^2), Space O(n^2+k^2).
//
// Notes:
//   - The O(n^2) matrix copy is acceptable because Christofides already runs over dense matrices.
//
// AI-Hints:
//   - Do not sort odd vertices here; MST degree scan already defines deterministic order.
func buildMatchingProblem(odd []int, dist matrix.Matrix) (matchingProblem, error) {
	oddCount := len(odd)
	// Handle the edge case where no odd-degree vertices exist (the MST is already Eulerian).
	if oddCount == 0 {
		return matchingProblem{
			odd: append([]int(nil), odd...),
			n:   0,
			w:   nil,
		}, nil
	}
	if (oddCount & 1) == 1 {
		return matchingProblem{}, ErrInvalidMatching
	}

	// Unpack and validate the global distance model into a continuous local matrix layout.
	// This ensures Numeric Law compliance (no NaN/Inf or negative weights) before matching begins.
	weights, err := copyCompleteWeights(dist, true)
	if err != nil {
		return matchingProblem{}, err
	}

	seen := make([]bool, weights.n)
	localOdd := append([]int(nil), odd...)

	// Pre-declare iteration variables to bypass re-allocation overhead inside the layout loop.
	var (
		localIndex int
		vertex     int
	)
	for localIndex, vertex = range localOdd {
		// Bounds validation: verify that the odd vertex index aligns with the global matrix scope.
		if vertex < 0 || vertex >= weights.n {
			return matchingProblem{}, ErrInvalidMatching
		}
		// Integrity check: fail early if any vertex appears twice in the odd-degree subset.
		if seen[vertex] {
			return matchingProblem{}, ErrInvalidMatching
		}

		// Track vertex registration and save it to the detached local sequence.
		seen[vertex] = true
		localOdd[localIndex] = vertex
	}

	problem := matchingProblem{
		odd: localOdd,
		n:   oddCount,
		w:   make([]float64, oddCount*oddCount),
	}

	// Map the global distances into a tight, row-major k×k local matrix for localized solver caches.
	var (
		row int
		col int
	)
	for row = 0; row < oddCount; row++ {
		for col = 0; col < oddCount; col++ {
			// Skip the diagonal since self-loop matching (matching a vertex to itself) is forbidden.
			if row == col {
				continue
			}

			// Translate local sub-graph indices into global coordinates to pull edge costs.
			problem.w[row*oddCount+col] = weights.at(localOdd[row], localOdd[col])
		}
	}

	return problem, nil
}

// verifyPerfectMatching validates the local symmetric match array.
// It prevents malformed DP or future matching engines from mutating Christofides
// adjacency with duplicate, self, or one-sided pairs.
//
// Implementation:
//   - Stage 1: Reject odd-length match arrays.
//   - Stage 2: Validate each partner index.
//   - Stage 3: Check symmetry match[match[i]]==i and no self-pair.
//
// Behavior highlights:
//   - No allocation.
//   - Does not inspect matrix weights.
//
// Inputs:
//   - match: local match array.
//
// Returns:
//   - error: nil when match is a perfect matching.
//
// Errors:
//   - ErrInvalidMatching for malformed arrays.
//
// Determinism:
//   - Fixed increasing local-index scan.
//
// Complexity:
//   - Time O(k), Space O(1).
//
// Notes:
//   - This is shared by every exact matching engine before adjacency mutation.
//
// AI-Hints:
//   - Do not append matching edges before this verification succeeds.
func verifyPerfectMatching(match []int) error {
	// Structural check: a symmetric matching array must have an even length.
	if (len(match) & 1) == 1 {
		return ErrInvalidMatching
	}

	var (
		index   int
		partner int
	)
	// Iterate through the array to assert bounds safety, loop absence, and reflection.
	for index, partner = range match {
		if partner < 0 || partner >= len(match) {
			return ErrInvalidMatching
		}
		if partner == index {
			return ErrInvalidMatching
		}
		if match[partner] != index {
			return ErrInvalidMatching
		}
	}

	return nil
}

// appendPerfectMatching appends verified local matching pairs to the multigraph.
// It maps local matching indices back to original TSP vertex indices.
//
// Implementation:
//   - Stage 1: Verify the local match array.
//   - Stage 2: Scan local indices in increasing order.
//   - Stage 3: Append each undirected pair exactly once when i<match[i].
//
// Behavior highlights:
//   - Mutates adj only after complete verification.
//   - Preserves deterministic pair append order.
//   - Allows parallel edges because Christofides uses a multigraph.
//
// Inputs:
//   - problem: local matching problem with odd-to-original mapping.
//   - match: verified local match array.
//   - adj: mutable Christofides multigraph adjacency.
//
// Returns:
//   - error: nil after append.
//
// Errors:
//   - ErrInvalidMatching for malformed mappings or adjacency shape.
//
// Determinism:
//   - Fixed increasing local-index scan; each pair appended once.
//
// Complexity:
//   - Time O(k), Space O(1).
//
// Notes:
//   - Caller owns rollback if append must be atomic under later failure.
//
// AI-Hints:
//   - Do not append both i<j and j<i in separate iterations.
func appendPerfectMatching(problem matchingProblem, match []int, adj [][]int) error {
	if err := verifyPerfectMatching(match); err != nil {
		return err
	}
	if len(problem.odd) != len(match) {
		return ErrInvalidMatching
	}

	// Pre-declare iteration primitives to enforce a zero-allocation hot loop profile.
	var (
		local   int
		partner int
		u       int
		v       int
	)
	for local, partner = range match {
		if local > partner {
			continue
		}

		// De-reference local sub-graph indices back into original global TSP vertex IDs.
		u = problem.odd[local]
		v = problem.odd[partner]

		// Bounds firewall: protect the global multigraph against mismatched or corrupt index mappings.
		if u < 0 || u >= len(adj) || v < 0 || v >= len(adj) {
			return ErrInvalidMatching
		}

		// Inject the resolved undirected matching pair into the global Christofides multigraph.
		adj[u] = append(adj[u], v)
		adj[v] = append(adj[v], u)
	}

	return nil
}

// snapshotAdjLengths records current adjacency lengths for atomic rollback.
// It does not copy edge values because matching engines only append edges.
//
// Implementation:
//   - Stage 1: Allocate one length slice.
//   - Stage 2: Store len(adj[v]) for every vertex in deterministic vertex order.
//
// Behavior highlights:
//   - Does not mutate adj.
//   - Does not inspect edge values.
//   - Suitable for Blossom and greedy matching append transactions.
//
// Inputs:
//   - adj: Christofides multigraph adjacency.
//
// Returns:
//   - []int: per-vertex original lengths.
//
// Errors:
//   - None.
//
// Determinism:
//   - Fixed increasing vertex-index scan.
//
// Complexity:
//   - Time O(n), Space O(n).
//
// Notes:
//   - Pair with rollbackAdj in deferred failure paths.
//
// AI-Hints:
//   - Do not deep-copy adjacency for append-only rollback.
//   - Do not reorder adjacency during snapshotting.
func snapshotAdjLengths(adj [][]int) []int {
	lengths := make([]int, len(adj))

	for vertex := range adj {
		lengths[vertex] = len(adj[vertex])
	}

	return lengths
}

// rollbackAdj removes matching edges appended after snapshotAdjLengths.
// It restores slice lengths only and therefore preserves existing edge order.
//
// Implementation:
//   - Stage 1: Scan all vertices.
//   - Stage 2: Validate snapshot bounds defensively.
//   - Stage 3: Truncate adj[v] back to the recorded length.
//
// Behavior highlights:
//   - Mutates adj only by truncation.
//   - Preserves all pre-existing adjacency entries.
//   - Ignores malformed snapshot entries instead of panicking.
//
// Inputs:
//   - adj: mutable adjacency to restore.
//   - lengths: snapshot returned by snapshotAdjLengths.
//
// Returns:
//   - None.
//
// Errors:
//   - None.
//
// Determinism:
//   - Fixed increasing vertex-index scan.
//
// Complexity:
//   - Time O(n), Space O(1).
//
// Notes:
//   - This helper is intentionally private to the matching stage.
//
// AI-Hints:
//   - Do not sort adjacency during rollback; Eulerian traversal depends on stable order.
func rollbackAdj(adj [][]int, lengths []int) {
	for vertex := range adj {
		if vertex >= len(lengths) {
			continue
		}
		if lengths[vertex] < 0 || lengths[vertex] > len(adj[vertex]) {
			continue
		}

		adj[vertex] = adj[vertex][:lengths[vertex]]
	}
}
