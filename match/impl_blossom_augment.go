// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package match mutates the committed matching during Blossom augmentation.
// This file is the only Blossom layer allowed to change mate[] and mateEdge[].
// All path composition and lifting must succeed before mutation begins.
//
// Responsibility:
//   - Validate lifted augmenting edge sequences.
//   - Flip matched/unmatched status along valid augmenting paths.
//   - Maintain mate[] and mateEdge[] symmetry.
//   - Reject non-alternating paths before any mutation.
//
// Boundaries:
//   - Path lifting lives in impl_blossom_path.go.
//   - Forest discovery lives in impl_blossom_forest.go.
//   - Cycle contraction/expansion lives in impl_blossom_contract.go.
//
// AI-Hints:
//   - Do not mutate mate[] before verifyAugmentingEdgeSequence succeeds.
//   - Do not allow duplicate edges in an augmenting path.
//   - Do not combine matching mutation with forest scanning.
package match

import "fmt"

// augment applies one endpoint-aware augmenting path discovered between two outer trees.
// It composes a root-to-root path, lifts contracted blossoms into original edge sequences,
// validates alternation, then flips matching state atomically at the mate[] level.
//
// Implementation:
//   - Stage 1: Reconstruct the left outer-to-root endpoint-aware path.
//   - Stage 2: Reconstruct the right outer-to-root endpoint-aware path.
//   - Stage 3: Build the connecting bridge step from event endpoint ownership.
//   - Stage 4: Compose a root-to-root path in correct orientation.
//   - Stage 5: Lift all contracted blossoms into original dense edge IDs.
//   - Stage 6: Flip the validated augmenting sequence and verify mate symmetry.
//
// Behavior highlights:
//   - Mutates mate[] only after lifting succeeds.
//   - Does not mutate contraction state.
//   - Rejects non-alternating lifted paths.
//   - Supports nested contracted blossoms through recursive lifting.
//
// Inputs:
//   - event: eventAugment with edge, node, and endpoint ownership fields populated.
//
// Returns:
//   - error: nil after a successful matching augmentation.
//
// Errors:
//   - ErrInvalidMatching for malformed event, invalid paths, failed lifting,
//     duplicate edges, broken alternation, or mate/mateEdge corruption.
//
// Determinism:
//   - Parent paths, bridge orientation, candidate lifting, and flip order are deterministic.
//
// Complexity:
//   - Time O(k + L + C), Space O(k + L + C), where L is lifted path length
//     and C is candidate lifting overhead for crossed blossoms.
//
// Notes:
//   - This method must not accept event values without endpoint ownership.
//
// AI-Hints:
//   - Do not flip the top-level path directly.
//   - Do not mutate mate[] before liftAugmentingPath returns successfully.
func (e *blossomEngine) augment(event blossomEvent) error {
	if event.kind != eventAugment {
		return fmt.Errorf("augment: event kind %d: %w", event.kind, ErrInvalidMatching)
	}

	leftPath, err := e.pathToRoot(event.a)
	if err != nil {
		return fmt.Errorf("augment: left path to root from node %d: %w", event.a, err)
	}

	rightPath, err := e.pathToRoot(event.b)
	if err != nil {
		return fmt.Errorf("augment: right path to root from node %d: %w", event.b, err)
	}

	bridge := blossomPathStep{
		edge:       event.edge,
		fromNode:   event.a,
		toNode:     event.b,
		fromVertex: event.aVertex,
		toVertex:   event.bVertex,
	}

	path, err := e.composeAugmentingPath(leftPath, bridge, rightPath)
	if err != nil {
		return fmt.Errorf("augment: compose path event=%+v left=%+v right=%+v: %w", event, leftPath, rightPath, err)
	}

	choices, err := e.liftAugmentingPathChoices(path)
	if err != nil {
		return err
	}

	return e.tryLiftedAugmentingCandidates(event, path, choices)

}

/*func (e *blossomEngine) augment(event blossomEvent) error {
	if event.kind != eventAugment {
		return ErrInvalidMatching
	}

	leftPath, err := e.pathToRoot(event.a)
	if err != nil {
		return err
	}

	rightPath, err := e.pathToRoot(event.b)
	if err != nil {
		return err
	}

	bridge := blossomPathStep{
		edge:       event.edge,
		fromNode:   event.a,
		toNode:     event.b,
		fromVertex: event.aVertex,
		toVertex:   event.bVertex,
	}

	path, err := e.composeAugmentingPath(leftPath, bridge, rightPath)
	if err != nil {
		return err
	}

	lifted, err := e.liftAugmentingPath(path)
	if err != nil {
		return err
	}

	if err = e.flipAugmentingPath(lifted); err != nil {
		return err
	}

	if err = e.refreshAllocatedBlossomBases(); err != nil {
		return err
	}

	return e.verifyMatchingSymmetry()
}*/

// isMatchedEdge reports whether edgeID is currently the symmetric committed mate edge.
// It validates the relation through both mate[] and mateEdge[] so a stale one-sided
// update is never treated as a valid matching edge.
//
// Implementation:
//   - Stage 1: Resolve dense edge endpoints.
//   - Stage 2: Check mate[u]==v and mate[v]==u.
//   - Stage 3: Check mateEdge[u] and mateEdge[v] both point to edgeID.
//
// Behavior highlights:
//   - Does not mutate engine state.
//   - Treats partial mate[] / mateEdge[] corruption as unmatched.
//   - Used by lifting, alternation checks, dual/event logic, and flip preparation.
//
// Inputs:
//   - edgeID: dense local edge identifier.
//
// Returns:
//   - bool: true only when edgeID is the current committed symmetric mate edge.
//
// Errors:
//   - None. Callers must pass a valid dense edge ID.
//
// Determinism:
//   - Pure indexed checks; no iteration.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This method is intentionally stricter than checking mate[] alone.
//
// AI-Hints:
//   - Do not relax this to mate[u]==v only.
//   - Stale mateEdge[] must be detected by returning false or later verification.
func (e *blossomEngine) isMatchedEdge(edgeID int) bool {
	edge := e.edges[edgeID]

	return e.mate[edge.u] == edge.v &&
		e.mate[edge.v] == edge.u &&
		e.mateEdge[edge.u] == edgeID &&
		e.mateEdge[edge.v] == edgeID
}

// extendAlternatingCandidates appends every segment choice to every prefix and keeps
// only edge sequences that still alternate matched/unmatched status under the current
// committed matching.
//
// Implementation:
//   - Stage 1: Preallocate the candidate output using prefix*choice capacity.
//   - Stage 2: Try each prefix/choice pair in deterministic nested-loop order.
//   - Stage 3: Reject joins that break alternation through canAppendAlternating.
//   - Stage 4: Copy accepted edges into a detached candidate slice.
//
// Behavior highlights:
//   - Does not mutate mate[], candidates, or input slices.
//   - Preserves deterministic candidate order.
//   - Accepts empty segment choices as neutral internal routes.
//   - Used by recursive Blossom lifting where multiple parity choices may exist.
//
// Inputs:
//   - prefixes: already-built lifted edge prefixes.
//   - choices: segment alternatives to append.
//
// Returns:
//   - [][]int: detached accepted candidates.
//
// Errors:
//   - None. Invalid edge IDs are rejected by canAppendAlternating.
//
// Determinism:
//   - Stable order: prefix order first, then choice order.
//
// Complexity:
//   - Time O(p*c*s), Space O(p*c*s), where p is prefix count,
//     c is choice count, and s is average appended sequence length.
//
// Notes:
//   - This performs local alternation pruning only.
//   - Full augmenting-path validity is still checked by verifyAugmentingEdgeSequence.
//
// AI-Hints:
//   - Do not return aliases that can be mutated by later appends.
//   - Do not collapse to the first valid candidate before augment tries all candidates.
func (e *blossomEngine) extendAlternatingCandidates(prefixes [][]int, choices [][]int) [][]int {
	out := make([][]int, 0, len(prefixes)*len(choices))

	for _, prefix := range prefixes {
		for _, choice := range choices {
			if !e.canAppendAlternating(prefix, choice) {
				continue
			}

			next := make([]int, 0, len(prefix)+len(choice))
			next = append(next, prefix...)
			next = append(next, choice...)
			out = append(out, next)
		}
	}

	return out
}

// canAppendAlternating reports whether segment can follow prefix without breaking
// matched/unmatched alternation at the join or inside the appended segment.
//
// Implementation:
//   - Stage 1: Treat an empty segment as a neutral append.
//   - Stage 2: Seed previous parity from the prefix tail when the prefix is non-empty.
//   - Stage 3: Scan segment edges in order.
//   - Stage 4: Reject invalid edge IDs or equal adjacent matched status.
//
// Behavior highlights:
//   - Does not mutate engine state.
//   - Works for internal Blossom path fragments and full path prefixes.
//   - Uses current committed matching state, not predicted post-flip state.
//
// Inputs:
//   - prefix: existing lifted edge sequence.
//   - segment: candidate edge sequence to append.
//
// Returns:
//   - bool: true when prefix+segment remains locally alternating.
//
// Errors:
//   - None. Invalid edge IDs return false.
//
// Determinism:
//   - Fixed left-to-right scan.
//
// Complexity:
//   - Time O(len(segment)), Space O(1).
//
// Notes:
//   - Empty prefix and empty segment are both valid neutral cases.
//
// AI-Hints:
//   - Do not require segment to start unmatched here; full augment validation does that.
//   - This is a local pruning helper, not the final augmenting-path validator.
func (e *blossomEngine) canAppendAlternating(prefix []int, segment []int) bool {
	if len(segment) == 0 {
		return true
	}

	previousKnown := false
	previousMatched := false

	if len(prefix) > 0 {
		previousKnown = true
		previousMatched = e.isMatchedEdge(prefix[len(prefix)-1])
	}

	for _, edgeID := range segment {
		if edgeID < 0 || edgeID >= len(e.edges) {
			return false
		}

		currentMatched := e.isMatchedEdge(edgeID)
		if previousKnown && currentMatched == previousMatched {
			return false
		}

		previousKnown = true
		previousMatched = currentMatched
	}

	return true
}

// flipAugmentingPath toggles committed matching status along one validated alternating
// original-edge sequence. It clears currently matched edges before setting newly matched
// edges so transient endpoint conflicts do not reject a valid augmentation.
//
// Implementation:
//   - Stage 1: Validate the full lifted sequence with verifyAugmentingEdgeSequence.
//   - Stage 2: Snapshot current matched status for every edge in the sequence.
//   - Stage 3: Clear edges that were matched before the flip.
//   - Stage 4: Set edges that were unmatched before the flip.
//
// Behavior highlights:
//   - Mutates only mate[] and mateEdge[].
//   - Does not mutate forest labels, duals, base[], cycles[], or members[].
//   - Requires odd-length unmatched/matched/.../unmatched input.
//   - Leaves mate[] symmetric when all helper calls succeed.
//
// Inputs:
//   - edges: lifted original dense edge IDs forming an augmenting path.
//
// Returns:
//   - error: nil after a successful flip.
//
// Errors:
//   - ErrInvalidMatching from invalid edge IDs, duplicate edges, broken alternation,
//     inconsistent pre-existing mate state, or conflicting endpoint matches.
//
// Determinism:
//   - Fixed left-to-right snapshot, clear, then set passes.
//
// Complexity:
//   - Time O(len(edges)), Space O(len(edges)).
//
// Notes:
//   - The clear-before-set policy is essential for paths where adjacent mutations share vertices.
//
// AI-Hints:
//   - Do not interleave clear/set in one pass.
//   - Do not call this before liftAugmentingPathChoices has produced validated candidates.
func (e *blossomEngine) flipAugmentingPath(edges []int) error {
	if err := e.verifyAugmentingEdgeSequence(edges); err != nil {
		return err
	}

	wasMatched := make([]bool, len(edges))

	for index, edgeID := range edges {
		wasMatched[index] = e.isMatchedEdge(edgeID)
	}

	for index, edgeID := range edges {
		if wasMatched[index] {
			if err := e.clearMatchedEdge(edgeID); err != nil {
				return err
			}
		}
	}

	for index, edgeID := range edges {
		if !wasMatched[index] {
			if err := e.setMatchedEdge(edgeID); err != nil {
				return err
			}
		}
	}

	return nil
}

// setMatchedEdge commits one dense edge as the symmetric mate relation of its endpoints.
// It rejects conflicting pre-existing mates and updates mate[] and mateEdge[] together.
//
// Implementation:
//   - Stage 1: Validate edgeID bounds.
//   - Stage 2: Resolve dense endpoints.
//   - Stage 3: Reject endpoint conflicts with unrelated existing mates.
//   - Stage 4: Store symmetric mate[] and mateEdge[] values.
//
// Behavior highlights:
//   - Mutates only mate[] and mateEdge[].
//   - Allows idempotent setting of an already consistent same edge.
//   - Rejects partial or conflicting matching state instead of silently overwriting it.
//
// Inputs:
//   - edgeID: dense local edge identifier to commit.
//
// Returns:
//   - error: nil after endpoints are symmetrically matched.
//
// Errors:
//   - ErrInvalidMatching for invalid edge IDs or conflicting endpoint mates.
//
// Determinism:
//   - Pure indexed endpoint update.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This helper does not validate global perfect matching completeness.
//
// AI-Hints:
//   - Do not overwrite existing unrelated mates.
//   - Keep mate[] and mateEdge[] updates adjacent and symmetric.
func (e *blossomEngine) setMatchedEdge(edgeID int) error {
	if edgeID < 0 || edgeID >= len(e.edges) {
		return ErrInvalidMatching
	}

	edge := e.edges[edgeID]

	if e.mate[edge.u] != noVertex && e.mate[edge.u] != edge.v {
		return ErrInvalidMatching
	}
	if e.mate[edge.v] != noVertex && e.mate[edge.v] != edge.u {
		return ErrInvalidMatching
	}

	e.mate[edge.u] = edge.v
	e.mate[edge.v] = edge.u
	e.mateEdge[edge.u] = edgeID
	e.mateEdge[edge.v] = edgeID

	return nil
}

// clearMatchedEdge removes one currently committed dense mate edge from mate[] and mateEdge[].
// Both endpoints must already agree that edgeID is their mate edge; otherwise the matching
// state is corrupt and ErrInvalidMatching is returned.
//
// Implementation:
//   - Stage 1: Validate edgeID bounds.
//   - Stage 2: Resolve dense endpoints.
//   - Stage 3: Require symmetric mate[] ownership.
//   - Stage 4: Reset both endpoints to noVertex/noEdge.
//
// Behavior highlights:
//   - Mutates only mate[] and mateEdge[].
//   - Rejects stale one-sided state.
//   - Used by flipAugmentingPath after the full path has already been validated.
//
// Inputs:
//   - edgeID: currently matched dense local edge identifier.
//
// Returns:
//   - error: nil after both endpoints are cleared.
//
// Errors:
//   - ErrInvalidMatching for invalid edge IDs or asymmetric matching state.
//
// Determinism:
//   - Pure indexed endpoint update.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This method intentionally does not clear arbitrary incident edges.
//
// AI-Hints:
//   - Do not make this permissive; corruption must be caught immediately.
func (e *blossomEngine) clearMatchedEdge(edgeID int) error {
	if edgeID < 0 || edgeID >= len(e.edges) {
		return ErrInvalidMatching
	}

	edge := e.edges[edgeID]

	if e.mate[edge.u] != edge.v || e.mate[edge.v] != edge.u {
		return ErrInvalidMatching
	}

	e.mate[edge.u] = noVertex
	e.mate[edge.v] = noVertex
	e.mateEdge[edge.u] = noEdge
	e.mateEdge[edge.v] = noEdge

	return nil
}

// verifyAugmentingEdgeSequence validates the lifted original-edge path before mutation.
// A valid augmenting path starts and ends with unmatched edges and alternates matched/unmatched
// status throughout the sequence.
//
// Implementation:
//   - Stage 1: Reject empty or even-length edge sequences.
//   - Stage 2: Validate every edge ID and reject duplicates.
//   - Stage 3: Require the first and last edges to be currently unmatched.
//   - Stage 4: Require strict matched/unmatched alternation between neighbors.
//
// Behavior highlights:
//   - Does not mutate mate[].
//   - Checks original dense edge IDs only.
//   - Protects flipAugmentingPath from corrupt path lifting.
//
// Inputs:
//   - edges: lifted original dense edge sequence.
//
// Returns:
//   - error: nil when the sequence is a valid augmenting path.
//
// Errors:
//   - ErrInvalidMatching for invalid IDs, duplicates, wrong parity, or broken alternation.
//
// Determinism:
//   - Fixed left-to-right scan.
//
// Complexity:
//   - Time O(len(edges)), Space O(|E|) for duplicate detection.
//
// Notes:
//   - This validation is intentionally strict; a failed lift must not partially mutate matching state.
//
// AI-Hints:
//   - Do not weaken first/last unmatched checks.
//   - Do not call setMatchedEdge before this validation succeeds.
func (e *blossomEngine) verifyAugmentingEdgeSequence(edges []int) error {
	if len(edges) == 0 || (len(edges)&1) == 0 {
		return ErrInvalidMatching
	}

	seen := make([]bool, len(e.edges))

	for index, edgeID := range edges {
		if edgeID < 0 || edgeID >= len(e.edges) {
			return ErrInvalidMatching
		}
		if seen[edgeID] {
			return ErrInvalidMatching
		}
		seen[edgeID] = true

		currentMatched := e.isMatchedEdge(edgeID)

		if index == 0 || index == len(edges)-1 {
			if currentMatched {
				return ErrInvalidMatching
			}
		}

		if index > 0 {
			previousMatched := e.isMatchedEdge(edges[index-1])
			if currentMatched == previousMatched {
				return ErrInvalidMatching
			}
		}
	}

	return nil
}

// edgeSequenceAlternates reports whether an edge sequence alternates committed matched status.
// Empty sequences are valid neutral internal Blossom routes because some child transitions
// require no original edge.
//
// Implementation:
//   - Stage 1: Treat length 0 and 1 as alternating.
//   - Stage 2: Scan adjacent pairs.
//   - Stage 3: Reject equal matched status between neighbors.
//
// Behavior highlights:
//   - Does not require first or last edge to be unmatched.
//   - Uses current committed matching state.
//   - Suitable for internal path fragments before full augment validation.
//
// Inputs:
//   - edges: dense original edge IDs.
//
// Returns:
//   - bool: true when adjacent matched status alternates.
//
// Errors:
//   - None. Callers must pass valid edge IDs.
//
// Determinism:
//   - Fixed left-to-right scan.
//
// Complexity:
//   - Time O(len(edges)), Space O(1).
//
// Notes:
//   - verifyAugmentingEdgeSequence is stricter and must still validate full paths.
//
// AI-Hints:
//   - Do not use this as a replacement for verifyAugmentingEdgeSequence.
//   - Empty internal choices must remain valid.
func (e *blossomEngine) edgeSequenceAlternates(edges []int) bool {
	for index := 1; index < len(edges); index++ {
		if e.isMatchedEdge(edges[index]) == e.isMatchedEdge(edges[index-1]) {
			return false
		}
	}

	return true
}

// refreshAllocatedBlossomBases recomputes the base vertex of every live contracted blossom
// from the committed original-vertex matching.
//
// Mathematical invariant:
//   - In a contracted odd blossom, exactly one original member is not matched to another
//     member of the same blossom.
//   - That vertex is the blossom base.
//   - It may be currently unmatched, or it may be matched to a vertex outside the blossom.
//
// Implementation:
//   - Stage 1: Walk allocated blossom nodes in increasing node order.
//   - Stage 2: For each live blossom, find the unique member whose mate is outside the blossom.
//   - Stage 3: Store that member in base[node].
//   - Stage 4: Reject blossoms with zero or multiple external/unmatched members.
//
// Behavior highlights:
//   - Does not mutate mate[] or mateEdge[].
//   - Does not change cycle metadata.
//   - Does not clear dual variables.
//   - Repairs persistent blossom base metadata after a successful augmenting-path flip.
//
// Inputs:
//   - None; reads members[], cycles[], mate[], and base[].
//
// Returns:
//   - error: nil when every allocated blossom has exactly one current base.
//
// Errors:
//   - ErrInvalidMatching for malformed members, invalid mate state, or non-unique base candidates.
//
// Determinism:
//   - Scans blossom IDs and members in deterministic stored order.
//
// Complexity:
//   - Time O(total allocated blossom members * average membership check), Space O(1).
//
// Notes:
//   - This is the correct source of truth after flipAugmentingPath.
//   - Do not infer persistent bases from lift direction alone.
//
// AI-Hints:
//   - Do not clear blossom duals here.
//   - Do not restore all blossoms here.
//   - Do not accept multiple external members; that means the represented matching is corrupt.
func (e *blossomEngine) refreshAllocatedBlossomBases() error {
	for node := e.problem.n; node < e.nextNode; node++ {
		if !e.isAllocatedBlossom(node) {
			continue
		}

		base, err := e.currentBlossomBaseFromMatching(node)
		if err != nil {
			return err
		}

		e.base[node] = base
	}

	return nil
}

// currentBlossomBaseFromMatching returns the unique original member of a live contracted
// blossom whose mate is not another member of the same blossom. This vertex is the
// current base of the represented near-perfect odd set.
//
// Implementation:
//   - Stage 1: Validate that node is an allocated contracted blossom.
//   - Stage 2: Scan original member vertices.
//   - Stage 3: Ignore members matched internally within the same blossom.
//   - Stage 4: Select the unique unmatched-or-externally-matched member as base.
//   - Stage 5: Reject zero or multiple base candidates.
//
// Behavior highlights:
//   - Reads mate[] after a successful speculative or committed flip.
//   - Does not mutate mate[], mateEdge[], cycles[], or members[].
//   - Defines the authoritative post-flip base invariant for persistent blossoms.
//
// Inputs:
//   - node: allocated contracted blossom node.
//
// Returns:
//   - int: original local vertex that is the current blossom base.
//   - error: nil when exactly one base candidate exists.
//
// Errors:
//   - ErrInvalidMatching for invalid node, malformed members, invalid mate values,
//     no base candidate, or multiple base candidates.
//
// Determinism:
//   - Fixed member order scan.
//
// Complexity:
//   - Time O(|members(node)|^2), Space O(1), because membership is checked by scan.
//
// Notes:
//   - Multiple outside-matched members means the lifted path did not preserve the blossom invariant.
//
// AI-Hints:
//   - Do not infer this value from lift direction.
//   - Do not accept multiple candidates; that would corrupt grow/expand semantics.
func (e *blossomEngine) currentBlossomBaseFromMatching(node int) (int, error) {
	if node < e.problem.n || node >= len(e.members) || !e.isAllocatedBlossom(node) {
		return noVertex, ErrInvalidMatching
	}

	base := noVertex

	for _, vertex := range e.members[node] {
		if vertex < 0 || vertex >= e.problem.n {
			return noVertex, ErrInvalidMatching
		}

		mate := e.mate[vertex]
		if mate != noVertex && (mate < 0 || mate >= e.problem.n) {
			return noVertex, ErrInvalidMatching
		}

		if mate != noVertex && e.nodeContainsVertex(node, mate) {
			continue
		}

		if base != noVertex {
			return noVertex, ErrInvalidMatching
		}

		base = vertex
	}

	if base == noVertex {
		return noVertex, ErrInvalidMatching
	}

	return base, nil
}

// tryLiftedAugmentingCandidates tries lifted augmenting candidates transactionally.
// A candidate is committed only when it flips cleanly, preserves live blossom-base
// invariants, and leaves mate[] / mateEdge[] symmetric.
//
// Mathematical contract:
//   - Edge alternation alone is insufficient for nested contracted blossoms.
//   - A candidate that breaks refreshAllocatedBlossomBases is not a solver failure;
//     it is an invalid lift choice and the next deterministic candidate must be tried.
//   - The committed candidate must preserve every live contracted blossom as a valid
//     near-perfect odd set with exactly one base.
//
// Implementation:
//   - Stage 1: Snapshot mate[], mateEdge[], and base[].
//   - Stage 2: Try each lifted candidate in deterministic order.
//   - Stage 3: Flip the candidate.
//   - Stage 4: Refresh live blossom bases from committed matching.
//   - Stage 5: Verify matching symmetry.
//   - Stage 6: Roll back and try the next candidate on any failure.
//
// Behavior highlights:
//   - Does not mutate forest labels or dual variables.
//   - Rolls back partial flip/base updates before trying the next candidate.
//   - Returns the last candidate failure only when every candidate fails.
//
// Inputs:
//   - event: original augment event, used only for diagnostics.
//   - path: endpoint-aware top-level path, used only for diagnostics.
//   - choices: lifted edge candidates from liftAugmentingPathChoices.
//
// Returns:
//   - error: nil after the first candidate that preserves all invariants.
//
// Errors:
//   - ErrInvalidMatching when no lifted candidate can be committed.
//
// Determinism:
//   - Candidate order is stable and inherited from path lifting.
//
// Complexity:
//   - Time O(c * (L + B*m^2)), Space O(k), where c is candidate count.
//
// AI-Hints:
//   - Do not keep a candidate that fails refreshAllocatedBlossomBases.
//   - Do not return immediately on the first failed candidate.
//   - Do not roll back labels/dual; this method mutates only matching/base state.
func (e *blossomEngine) tryLiftedAugmentingCandidates(
	event blossomEvent,
	path []blossomPathStep,
	choices [][]int,
) error {
	if len(choices) == 0 {
		return ErrInvalidMatching
	}

	originalMate := append([]int(nil), e.mate...)
	originalMateEdge := append([]int(nil), e.mateEdge...)
	originalBase := append([]int(nil), e.base...)

	var lastErr error

	for _, lifted := range choices {
		e.restoreMatchingAndBases(originalMate, originalMateEdge, originalBase)

		if err := e.flipAugmentingPath(lifted); err != nil {
			lastErr = err
			continue
		}

		if err := e.refreshAllocatedBlossomBases(); err != nil {
			lastErr = err
			continue
		}

		if err := e.verifyMatchingSymmetry(); err != nil {
			lastErr = err
			continue
		}

		return nil
	}

	e.restoreMatchingAndBases(originalMate, originalMateEdge, originalBase)

	if lastErr != nil {
		return fmt.Errorf(
			"augment: no lifted candidate preserved blossom invariants event=%+v path=%+v choices=%v: %w",
			event,
			path,
			choices,
			lastErr,
		)
	}

	return ErrInvalidMatching
}

// restoreMatchingAndBases restores mate[], mateEdge[], and base[] after a failed
// speculative lifted augmenting candidate. It is used by transactional augmentation
// to try later candidates without leaking partial mutation.
//
// Implementation:
//   - Stage 1: Copy saved mate[] back into engine storage.
//   - Stage 2: Copy saved mateEdge[] back into engine storage.
//   - Stage 3: Copy saved base[] back into engine storage.
//
// Behavior highlights:
//   - Does not mutate labels, parent links, tree roots, duals, cycles, or members.
//   - Assumes input slices were captured from the same engine.
//   - Restores only state that tryLiftedAugmentingCandidates is allowed to mutate.
//
// Inputs:
//   - mate: saved mate[] snapshot.
//   - mateEdge: saved mateEdge[] snapshot.
//   - base: saved base[] snapshot.
//
// Returns:
//   - None.
//
// Errors:
//   - None. Caller must provide correctly sized snapshots.
//
// Determinism:
//   - Pure indexed copy.
//
// Complexity:
//   - Time O(k), Space O(1), excluding snapshot storage owned by the caller.
//
// Notes:
//   - Keep this helper narrow; rolling back forest/dual state here would hide bugs.
//
// AI-Hints:
//   - Do not add cycles[] or members[] rollback unless augmentation starts mutating them.
func (e *blossomEngine) restoreMatchingAndBases(mate []int, mateEdge []int, base []int) {
	copy(e.mate, mate)
	copy(e.mateEdge, mateEdge)
	copy(e.base, base)
}
