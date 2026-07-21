// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Internal traversal and cycle-normalization helpers used by DFS-based algorithms.
package dfs

import (
	"strings"

	"github.com/lvlath/go/core"
)

const (
	// cycleSignatureSeparator is the stable internal delimiter used for canonical
	// cycle signatures during deduplication.
	cycleSignatureSeparator = ","

	// indexNotFound is returned when a requested string value is absent from a slice.
	indexNotFound = -1

	// boothFailureUnset marks an absent failure link in Booth's algorithm.
	boothFailureUnset = -1
)

// neighborFromEdge resolves the traversal neighbor reachable from currentID via edge.
//
// Implementation:
//   - Stage 1: Reject nil edges.
//   - Stage 2: Apply per-edge directed semantics for directed edges.
//   - Stage 3: Return the opposite endpoint for undirected edges.
//
// Behavior highlights:
//   - Directed traversal is allowed only from From to To.
//   - Undirected traversal returns the opposite endpoint.
//   - The helper does not apply loop policy, visited policy, or filtering policy.
//
// Inputs:
//   - edge: graph edge under consideration.
//   - currentID: the current traversal vertex ID.
//
// Returns:
//   - string: the resolved neighbor vertex ID.
//   - bool: true if the edge yields a valid traversal neighbor from currentID.
//
// Errors:
//   - N/A.
//
// Determinism:
//   - Deterministic for the same edge and currentID.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This helper encodes traversal semantics only.
//   - Self-loop handling belongs to the caller's traversal policy.
//
// AI-Hints:
//   - This is the only approved edge-to-neighbor interpretation in traversal code.
//   - Never infer an undirected or mixed-edge neighbor from edge.To alone.
//   - Edge direction is defined per edge, not by package-level assumptions.
func neighborFromEdge(edge *core.Edge, currentID string) (string, bool) {
	// Reject nil edges defensively so callers can safely treat them as non-traversable.
	if edge == nil {
		return "", false
	}

	// Directed edges are traversable only from the declared source endpoint.
	if edge.Directed {
		if edge.From != currentID {
			return "", false
		}

		return edge.To, true
	}

	// For undirected edges, return the opposite endpoint if the current vertex matches From.
	if edge.From == currentID {
		return edge.To, true
	}

	// For undirected edges, return the opposite endpoint if the current vertex matches To.
	if edge.To == currentID {
		return edge.From, true
	}

	// The current vertex is not incident to the edge in a traversable way.
	return "", false
}

// indexOfString returns the first index of target in values, or indexNotFound when absent.
//
// Implementation:
//   - Stage 1: Scan values in order.
//   - Stage 2: Return the first matching index.
//   - Stage 3: Return indexNotFound if no match exists.
//
// Behavior highlights:
//   - The first match wins.
//   - The function does not mutate the input slice.
//
// Inputs:
//   - values: ordered string slice.
//   - target: searched string value.
//
// Returns:
//   - int: the first matching index, or indexNotFound.
//
// Errors:
//   - N/A.
//
// Determinism:
//   - Deterministic under deterministic slice order.
//
// Complexity:
//   - Time O(n), Space O(1), where n = len(values).
//
// Notes:
//   - The helper is internal and domain-support oriented.
//
// AI-Hints:
//   - Use this helper only when positional membership matters.
//   - Do not expose internal search helpers as part of the public package API.
func indexOfString(values []string, target string) int {
	// Scan in order so the first matching position is stable and deterministic.
	for index, value := range values {
		if value == target {
			return index
		}
	}

	return indexNotFound
}

// reverseStrings returns a reversed copy of values without mutating the input.
//
// Implementation:
//   - Stage 1: Allocate a new output slice.
//   - Stage 2: Copy elements from the end of the input to the beginning of the output.
//
// Behavior highlights:
//   - The returned slice is a new allocation.
//   - The input slice is never mutated.
//
// Inputs:
//   - values: input string sequence.
//
// Returns:
//   - []string: reversed copy of values.
//
// Errors:
//   - N/A.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(n), Space O(n), where n = len(values).
//
// Notes:
//   - The helper intentionally favors ownership safety over in-place mutation.
//
// AI-Hints:
//   - Keep reverse operations pure when canonicalization must not mutate caller-owned data.
func reverseStrings(values []string) []string {
	// Allocate a new slice so the input remains fully caller-owned and unchanged.
	reversed := make([]string, len(values))

	// Fill the output from the opposite end of the input.
	for index := range values {
		reversed[index] = values[len(values)-1-index]
	}

	return reversed
}

// compareStringSlicesLex compares two string slices using lexicographic order.
//
// Implementation:
//   - Stage 1: Compare shared-prefix elements in order.
//   - Stage 2: Return at the first differing element.
//   - Stage 3: If one slice is a full prefix of the other, the shorter slice is smaller.
//
// Behavior highlights:
//   - The helper is safe for unequal-length slices.
//   - The function does not mutate either input.
//
// Inputs:
//   - left: first string sequence.
//   - right: second string sequence.
//
// Returns:
//   - int: -1 if left < right, 0 if left == right, +1 if left > right.
//
// Errors:
//   - N/A.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(min(len(left), len(right))) for prefix comparison, with O(1) extra space.
//
// Notes:
//   - Equal-length usage is common for cycle normalization, but unequal lengths are supported safely.
//
// AI-Hints:
//   - Use this helper for stable canonical ordering.
//   - Do not assume equal lengths unless the caller's contract explicitly guarantees them.
func compareStringSlicesLex(left, right []string) int {
	// Compare the shared prefix first.
	limit := len(left)
	if len(right) < limit {
		limit = len(right)
	}

	// Return immediately when the first differing element determines the order.
	for index := 0; index < limit; index++ {
		if left[index] < right[index] {
			return -1
		}
		if left[index] > right[index] {
			return 1
		}
	}

	// If all shared elements match, the shorter slice is lexicographically smaller.
	switch {
	case len(left) < len(right):
		return -1
	case len(left) > len(right):
		return 1
	default:
		return 0
	}
}

// joinCycleSignature converts a canonical cycle into a stable deduplication signature.
//
// Implementation:
//   - Stage 1: Join canonical vertex IDs with the stable internal separator.
//
// Behavior highlights:
//   - The signature is intended for stable internal deduplication.
//   - The function does not mutate the input slice.
//
// Inputs:
//   - cycle: canonicalized cycle vertex sequence.
//
// Returns:
//   - string: stable cycle signature.
//
// Errors:
//   - N/A.
//
// Determinism:
//   - Deterministic for the same canonical cycle.
//
// Complexity:
//   - Time O(n + total bytes), Space O(total bytes), where n = len(cycle).
//
// Notes:
//   - This is an internal protocol helper, not a human-oriented formatter.
//
// AI-Hints:
//   - Use this helper for deduplication only.
//   - Keep the separator stable while signature compatibility matters.
func joinCycleSignature(cycle []string) string {
	// Use a stable internal separator so canonical signatures remain deterministic.
	return strings.Join(cycle, cycleSignatureSeparator)
}

// MinimalRotation returns the lexicographically minimal rotation of values.
//
// Implementation:
//   - Stage 1: Handle short inputs with ownership-safe copies.
//   - Stage 2: Build an explicit doubled buffer without aliasing the input.
//   - Stage 3: Run Booth's algorithm to locate the minimal rotation start index.
//   - Stage 4: Copy the resulting rotation into a fresh output slice.
//
// Behavior highlights:
//   - The input slice is never mutated.
//   - The returned slice is always caller-owned.
//   - The implementation is alias-safe and suitable for canonicalization.
//
// Inputs:
//   - values: input cycle sequence.
//
// Returns:
//   - []string: lexicographically minimal rotation of values.
//
// Errors:
//   - N/A.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(n), Space O(n), where n = len(values).
//
// Notes:
//   - Booth's algorithm computes the minimal rotation index in linear time.
//   - A fresh result slice is returned even for short inputs to preserve ownership clarity.
//
// AI-Hints:
//   - Never duplicate the sequence via append(values, values...) when purity matters.
//   - The returned slice must not rely on caller-controlled backing-array capacity.
func minimalRotation(values []string) []string {
	// Preserve caller ownership semantics even for empty or single-element inputs.
	if len(values) <= 1 {
		return append([]string(nil), values...)
	}

	// Record the original length once because Booth's algorithm indexes into a doubled view.
	originalLength := len(values)

	// Allocate the doubled buffer explicitly to avoid aliasing the caller's backing array.
	doubled := make([]string, originalLength*2)

	// Copy the original sequence into the first half of the doubled buffer.
	copy(doubled, values)

	// Copy the original sequence into the second half of the doubled buffer.
	copy(doubled[originalLength:], values)

	// Allocate the Booth failure-link table for the doubled scan space.
	failure := make([]int, originalLength*2)

	// Initialize all failure links to the sentinel unset value.
	for index := range failure {
		failure[index] = boothFailureUnset
	}

	// candidateStart tracks the best known rotation start.
	candidateStart := 0

	// Scan the doubled sequence once to locate the minimal rotation.
	for currentIndex := 1; currentIndex < originalLength*2; currentIndex++ {
		// Read the failure-link state relative to the current candidate start.
		failureIndex := failure[currentIndex-candidateStart-1]

		// Walk failure links while the current symbol disagrees with the current candidate extension.
		for failureIndex != boothFailureUnset &&
			doubled[currentIndex] != doubled[candidateStart+failureIndex+1] {
			// If the current symbol is smaller, a better candidate start has been found.
			if doubled[currentIndex] < doubled[candidateStart+failureIndex+1] {
				candidateStart = currentIndex - failureIndex - 1
			}

			// Follow the next failure link.
			failureIndex = failure[failureIndex]
		}

		// Handle the mismatch or initial no-link case.
		if doubled[currentIndex] != doubled[candidateStart+failureIndex+1] {
			// If the current symbol is smaller than the candidate start symbol, promote it.
			if doubled[currentIndex] < doubled[candidateStart] {
				candidateStart = currentIndex
			}

			// Reset the failure state for the new relative position.
			failure[currentIndex-candidateStart] = boothFailureUnset
			continue
		}

		// Extend the current matched prefix.
		failure[currentIndex-candidateStart] = failureIndex + 1
	}

	// Copy the minimal rotation into a fresh output slice.
	rotation := make([]string, originalLength)
	copy(rotation, doubled[candidateStart:candidateStart+originalLength])

	return rotation
}

// canonicalCycle returns the canonical representation of a cycle sequence.
//
// Implementation:
//   - Stage 1: Compute the minimal rotation of the original sequence.
//   - Stage 2: Optionally compute the minimal rotation of the reversed sequence.
//   - Stage 3: Return the lexicographically smaller canonical representative.
//
// Behavior highlights:
//   - Reversal is optional and policy-driven.
//   - The input slice is never mutated.
//   - Directed-cycle orientation can be preserved by setting allowReverse to false.
//
// Inputs:
//   - cycle: cycle vertex sequence.
//   - allowReverse: whether reversed orientation is considered equivalent.
//
// Returns:
//   - []string: canonical cycle representation.
//
// Errors:
//   - N/A.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(n), Space O(n), where n = len(cycle), because each normalization pass is linear.
//
// Notes:
//   - Use allowReverse=true for orientation-symmetric cycle models.
//   - Use allowReverse=false when direction is part of the cycle identity.
//
// AI-Hints:
//   - Directed cycle canonicalization must preserve orientation.
//   - Reverse comparison is valid only when the cycle model allows orientation symmetry.
func canonicalCycle(cycle []string, allowReverse bool) []string {
	// Normalize the original orientation first.
	best := minimalRotation(cycle)

	// Return immediately when reversal is not part of the canonicalization policy.
	if !allowReverse {
		return best
	}

	// Normalize the reversed orientation independently.
	reversedBest := minimalRotation(reverseStrings(cycle))

	// Select the lexicographically smaller representative.
	if compareStringSlicesLex(reversedBest, best) < 0 {
		return reversedBest
	}

	return best
}

// TestOnlyMinimalRotation exposes minimalRotation for black-box tests in package dfs_test.
//
// AI-Hints:
//   - This helper exists for regression tests only.
//   - Do not use it in production code or examples.
//   - Keep the exported test bridge minimal and side-effect free.
func TestOnlyMinimalRotation(values []string) []string {
	return minimalRotation(values)
}
