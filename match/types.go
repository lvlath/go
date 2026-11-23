// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package match defines algorithm identity and detached perfect-matching results.
// These types carry original matrix indices, not Blossom's private local positions.
package match

// Algorithm selects the one numerical kernel used for a matching request.
// Blossom is the default exact-or-error strategy; Greedy is an explicit heuristic.
// Implementation:
//   - Stage 1: Options validation accepts only the two declared enum values.
//   - Stage 2: Perfect dispatches to the selected existing kernel without fallback.
//
// Behavior highlights:
//   - Enum values are independent of tsp.MatchingAlgo; adapters must map explicitly.
//
// Inputs:
//   - Algorithm: configured through WithAlgorithm or a complete Options value.
//
// Returns:
//   - Algorithm: copied into Result to disclose the selected strategy.
//
// Errors:
//   - ErrInvalidOptions for unsupported enum values before matrix access.
//
// Determinism:
//   - Explicit enum dispatch; no size-dependent algorithm substitution.
//
// Complexity:
//   - Time O(1), Space O(1) for selection.
//
// Notes:
//   - Successful Greedy does not certify minimum weight.
//
// AI-Hints:
//   - Never cast tsp.MatchingAlgo into this enum: their numeric layouts differ.
type Algorithm uint8

const (
	// Blossom requests the migrated weighted Edmonds-style perfect-matching engine.
	// Success requires a complete pairing and its existing internal verification.
	Blossom Algorithm = iota

	// Greedy requests the deterministic last-remaining-pivot selection heuristic.
	// It always produces a full matching or an error, but does not minimize globally.
	Greedy
)

// Pair represents exactly one undirected selected edge in original matrix indices.
// Its endpoints are canonicalized after selection: U < V, with no self-loops.
// Implementation:
//   - Stage 1: Kernel-local mates are mapped through the original subset indices.
//   - Stage 2: Pair coverage and membership are verified before publication.
//   - Stage 3: Each pair is oriented U<V; the result is sorted by ascending U.
//
// Behavior highlights:
//   - Canonical presentation does not sort the input subset or alter kernel ties.
//
// Inputs:
//   - U, V: original nonnegative matrix row/column indices selected by the caller.
//
// Returns:
//   - Pair: a complete-matching edge stored by value, detached from matrix storage.
//
// Errors:
//   - ErrInvalidMatching is returned by Perfect on malformed kernel pairings.
//
// Determinism:
//   - Output order is ascending original U; U is strictly smaller than V.
//
// Complexity:
//   - Time O(1), Space O(1) per pair; full result sorting is O(k log k).
//
// Notes:
//   - Pair fields are publicly mutable; callers can invalidate a published Result.
//
// AI-Hints:
//   - Do not mistake a public endpoint for an index into the unsorted subset slice.
type Pair struct {
	// U is the smaller original matrix endpoint, never a local Blossom vertex index.
	U int

	// V is the larger original matrix endpoint; successful results require U<V.
	V int
}

// Result describes one complete perfect matching returned by either algorithm.
// It owns its pair storage and does not retain a matrix, subset, or mutable TSP adjacency.
// Implementation:
//   - Stage 1: One selected kernel produces reciprocal mates or greedy pair choices.
//   - Stage 2: Perfect verifies full selected-vertex coverage and computes checked raw cost.
//   - Stage 3: It canonicalizes detached pairs and publishes exactly one Result value.
//
// Behavior highlights:
//   - Pairs has length k/2, with ascending U and U<V for every selected edge.
//   - Cost sums selected original-direction weights without presentation rounding.
//   - Optimal is true only for a successfully internally verified Blossom result.
//   - Optimal does not expose an independent mathematical certificate.
//
// Inputs:
//   - Constructed by Perfect; each field can subsequently be modified by callers.
//
// Returns:
//   - A mutable detached value; Clone provides another independently owned pair slice.
//
// Errors:
//   - Perfect returns nil Result on error; no half-matching is published.
//
// Determinism:
//   - Kernel uses input subset order, while Result uses canonical presentation order.
//
// Complexity:
//   - Result storage O(k) for k selected vertices.
//
// Notes:
//   - Editing Pairs does not automatically recompute Cost or Optimal.
//
// AI-Hints:
//   - Do not infer an external dual certificate from the Optimal boolean alone.
type Result struct {
	// Pairs contains one detached canonical edge for every two selected vertices.
	Pairs []Pair

	// Cost is the checked raw float64 sum in the selected algorithm's accumulation order.
	Cost float64

	// Algorithm identifies the actual selected strategy and is not a fallback marker.
	Algorithm Algorithm

	// Optimal means successful internal Blossom verification, not a portable proof object.
	Optimal bool
}

// IsNil reports whether the Result receiver itself is nil.
// It implements core.Nilable semantics without inspecting potentially empty slices.
// Implementation:
//   - Stage 1: Compare the receiver pointer to nil; never dereference it.
//
// Behavior highlights:
//   - A non-nil zero-pair result is valid and reports false.
//
// Inputs:
//   - r: possibly nil Result receiver.
//
// Returns:
//   - bool: true only when the receiver pointer is nil.
//
// Errors:
//   - None; method is nil-receiver safe.
//
// Determinism:
//   - Pure pointer comparison.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Safe when invoked through the core.Nilable interface.
//
// AI-Hints:
//   - Do not infer nilability from len(Pairs), Cost, or Algorithm.
func (r *Result) IsNil() bool { return r == nil }

// Clone returns an independent Result snapshot with detached pair storage.
// It copies every scalar value and preserves existing pair order without recomputation.
// Implementation:
//   - Stage 1: Return nil for a nil receiver.
//   - Stage 2: Copy Algorithm, Cost and Optimal by value.
//   - Stage 3: Allocate and copy Pairs, preserving nil-vs-empty slice state.
//
// Behavior highlights:
//   - Mutating the returned Pairs cannot change the original receiver.
//
// Inputs:
//   - r: caller-owned source, possibly nil or externally modified.
//
// Returns:
//   - *Result: independently owned pair storage, or nil for nil input.
//
// Errors:
//   - None; Clone does not validate the caller's result fields.
//
// Determinism:
//   - Exact original slice order, without sorting or a kernel call.
//
// Complexity:
//   - Time O(len(Pairs)), Space O(len(Pairs)).
//
// Notes:
//   - Clone does not retain or recreate any reference to the input matrix.
//
// AI-Hints:
//   - Never shallow-copy the Pairs backing array across mutable results.
func (r *Result) Clone() *Result {
	if r == nil {
		return nil
	}
	out := *r
	if r.Pairs != nil {
		out.Pairs = make([]Pair, len(r.Pairs))
		copy(out.Pairs, r.Pairs)
	}
	return &out
}
