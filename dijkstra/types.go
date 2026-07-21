// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package dijkstra

import (
	"math"

	"github.com/lvlath/go/core"
)

// Result stores the detached, queryable outcome of a single-source
// shortest-path run over a weighted graph.
// The result keeps the source identifier, the finalized distance map, and
// the optional predecessor map used for path reconstruction.
//
// Implementation:
//   - Stage 1: Store the finalized source identifier and distance domain.
//   - Stage 2: Optionally store predecessor information when path tracking is enabled.
//   - Stage 3: Expose safe result-surface helpers for distance and path queries.
//
// Behavior highlights:
//   - Distances uses +Inf to represent known but unreachable vertices.
//   - Prev == nil means path tracking was disabled for the producing run.
//   - The result is detached from the graph and remains stable after return.
//
// Inputs:
//   - SourceID: the source vertex identifier used for the originating run.
//   - Distances: the finalized shortest-path distance map.
//   - Prev: the optional predecessor map for path reconstruction.
//
// Returns:
//   - Result: a detached contract type for post-run queries.
//
// Errors:
//   - Result helper methods may return ErrNilResult, ErrEmptyTargetID,
//     ErrTargetNotFound, ErrPathTrackingDisabled, or ErrNoPath.
//
// Determinism:
//   - Query methods are deterministic for the same stored SourceID, Distances, and Prev.
//
// Complexity:
//   - Querying a single distance is O(1).
//   - Path reconstruction is O(k), where k is the number of vertices on the returned path.
//
// Notes:
//   - Distances and Prev are caller-owned after publication.
//   - Direct mutation changes this Result and may invalidate witness invariants.
//   - PathTo defensively rejects broken, out-of-domain, and cyclic predecessor state.
//
// AI-Hints:
//   - Do not treat Prev == nil as "there is no path"; it means path tracking was disabled.
//   - Exported result maps are mutable by the caller; query methods must fail
//     safely when caller mutation invalidates witness state.
//   - Do not collapse unknown-target and unreachable-target semantics into one branch.
//   - Caller-owned Prev state must never be followed without cycle and
//     result-domain validation.
type Result struct {
	SourceID  string
	Distances map[string]float64
	Prev      map[string]string
}

// Ensure that Result satisfies core.Nilable without requiring callers
// to know the concrete result implementation type.
var _ core.Nilable = (*Result)(nil)

// IsNil reports whether the result receiver itself is nil.
// This method exists to satisfy core.Nilable and to let callers perform safe
// nil checks without triggering a nil-pointer dereference.
//
// Implementation:
//   - Stage 1: Compare the receiver pointer against nil.
//
// Behavior highlights:
//   - The method is safe on a nil receiver.
//
// Inputs:
//   - None.
//
// Returns:
//   - bool: true when the receiver is nil; otherwise false.
//
// Errors:
//   - None.
//
// Determinism:
//   - Always returns the same value for the same receiver state.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - A non-nil result may still contain nil maps if that is part of the contract.
//
// AI-Hints:
//   - Prefer IsNil over ad-hoc interface assertions when working with core.Nilable.
//   - Nil receiver safety here does not remove the need for safe checks in other methods.
func (r *Result) IsNil() bool {
	return r == nil
}

// DistanceTo returns the stored shortest-path distance to the requested target vertex.
// A returned distance of +Inf is a valid canonical outcome and means that the target
// is known to the result domain but unreachable from the source.
//
// Implementation:
//   - Stage 1: Validate the receiver and target identifier.
//   - Stage 2: Look up the target in the stored distance map.
//   - Stage 3: Return the stored value without reinterpretation.
//
// Behavior highlights:
//   - Missing targets are rejected explicitly.
//   - +Inf is returned as data, not as an error.
//
// Inputs:
//   - vertexID: the target vertex identifier to query.
//
// Returns:
//   - float64: the stored shortest-path distance for the target vertex.
//
// Errors:
//   - ErrNilResult if the receiver is nil.
//   - ErrEmptyTargetID if vertexID is empty.
//   - ErrTargetNotFound if the target does not exist in the result domain.
//
// Determinism:
//   - Map lookup is deterministic for the same stored result state.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Unknown target and unreachable target are intentionally different states.
//
// AI-Hints:
//   - Do not replace +Inf with zero or another synthetic sentinel value.
//   - Do not treat a missing key as equivalent to an unreachable known vertex.
func (r *Result) DistanceTo(vertexID string) (float64, error) {
	if r == nil {
		return 0, ErrNilResult
	}
	if vertexID == "" {
		return 0, ErrEmptyTargetID
	}

	distance, ok := r.Distances[vertexID]
	if !ok {
		return 0, ErrTargetNotFound
	}

	return distance, nil
}

// HasPathTo reports whether the target vertex is reachable from the source
// under the stored shortest-path result.
// The method distinguishes unknown targets from known-but-unreachable targets.
//
// Implementation:
//   - Stage 1: Resolve the target distance through DistanceTo.
//   - Stage 2: Interpret +Inf as unreachable.
//   - Stage 3: Report reachability for finite distances.
//
// Behavior highlights:
//   - Missing targets remain errors.
//   - Unreachable known targets return false with no error.
//
// Inputs:
//   - vertexID: the target vertex identifier to query.
//
// Returns:
//   - bool: true when the target is reachable; otherwise false.
//
// Errors:
//   - ErrNilResult if the receiver is nil.
//   - ErrEmptyTargetID if vertexID is empty.
//   - ErrTargetNotFound if the target does not exist in the result domain.
//
// Determinism:
//   - Always derives its answer from the stored distance map in a stable way.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This method does not require path tracking.
//
// AI-Hints:
//   - Reachability is derived from the distance contract, not from the presence of Prev.
//   - Do not treat missing-target errors as a simple false result.
func (r *Result) HasPathTo(vertexID string) (bool, error) {
	distance, err := r.DistanceTo(vertexID)
	if err != nil {
		return false, err
	}

	return !math.IsInf(distance, 1), nil
}

// PathTo reconstructs one deterministic shortest-path witness from the source
// to the requested target using the stored predecessor map.
// The method requires path tracking and defensively validates predecessor-chain
// integrity before publishing a witness.
//
// Implementation:
//   - Stage 1: Validate the receiver, target identifier, and target membership.
//   - Stage 2: Reject disabled path tracking and unreachable known targets.
//   - Stage 3: Follow Prev backward while validating result-domain membership.
//   - Stage 4: Detect repeated vertices and reject cyclic predecessor state.
//   - Stage 5: Reject missing or empty predecessor links before the source.
//   - Stage 6: Reverse the validated chain in place and publish the forward path.
//
// Behavior highlights:
//   - Path tracking is explicit and never inferred.
//   - Unknown, unreachable, tracking-disabled, and malformed-witness states remain distinct.
//   - The source queried against itself returns a single-vertex witness.
//   - Caller-mutated cyclic Prev state cannot cause an infinite loop.
//   - No partial path is returned on reconstruction failure.
//
// Inputs:
//   - vertexID: the target vertex identifier to reconstruct.
//
// Returns:
//   - []string: one validated shortest-path witness from SourceID to vertexID.
//   - On error, the returned slice is nil.
//
// Errors:
//   - ErrNilResult if the receiver is nil.
//   - ErrEmptyTargetID if vertexID is empty.
//   - ErrTargetNotFound if vertexID is absent from Distances.
//   - ErrPathTrackingDisabled if Prev is nil.
//   - ErrNoPath if the target distance is +Inf.
//   - ErrNoPath if the predecessor chain is missing, empty before SourceID,
//     leaves the result domain, or contains a cycle.
//
// Determinism:
//   - Reconstruction follows the exact stored Prev relation.
//   - The same SourceID, Distances, and Prev state produces the same path or error.
//
// Complexity:
//   - Time O(k), where k is the number of predecessor vertices inspected.
//   - Space O(k) for the reversed path and cycle-detection set.
//
// Notes:
//   - A Result returned unchanged by Dijkstra satisfies predecessor invariants.
//   - Defensive chain validation protects direct struct construction and
//     caller-owned mutation after publication.
//   - This method returns one witness, not all shortest paths.
//
// AI-Hints:
//   - Do not remove the cycle-detection set as a “redundant optimization”.
//   - Do not trust exported Prev state blindly.
//   - Do not return a partially reconstructed path on failure.
//   - Keep ErrPathTrackingDisabled distinct from ErrNoPath.
func (r *Result) PathTo(vertexID string) ([]string, error) {
	if r == nil {
		return nil, ErrNilResult
	}
	if vertexID == "" {
		return nil, ErrEmptyTargetID
	}

	if _, ok := r.Distances[vertexID]; !ok {
		return nil, ErrTargetNotFound
	}
	if r.Prev == nil {
		return nil, ErrPathTrackingDisabled
	}

	distance, err := r.DistanceTo(vertexID)
	if err != nil {
		return nil, err
	}
	if math.IsInf(distance, 1) {
		return nil, ErrNoPath
	}
	if vertexID == r.SourceID {
		return []string{r.SourceID}, nil
	}

	path := make([]string, 0)
	seenIDs := make(map[string]struct{})
	currentID := vertexID
	var known, repeated bool

	for {
		if _, known = r.Distances[currentID]; !known {
			return nil, ErrNoPath
		}
		if _, repeated = seenIDs[currentID]; repeated {
			return nil, ErrNoPath
		}

		seenIDs[currentID] = struct{}{}
		path = append(path, currentID)

		if currentID == r.SourceID {
			break
		}

		parentID, ok := r.Prev[currentID]
		if !ok || parentID == "" {
			return nil, ErrNoPath
		}

		currentID = parentID
	}

	leftIndex := 0
	rightIndex := len(path) - 1

	for leftIndex < rightIndex {
		path[leftIndex], path[rightIndex] = path[rightIndex], path[leftIndex]

		leftIndex++
		rightIndex--
	}

	return path, nil
}

// Clone returns a detached deep copy of the result.
// Maps are copied deeply so that the returned value can be mutated by the caller
// without affecting the original result.
//
// Implementation:
//   - Stage 1: Handle the nil receiver safely.
//   - Stage 2: Copy scalar fields.
//   - Stage 3: Deep-copy Distances.
//   - Stage 4: Deep-copy Prev only when tracking data exists.
//
// Behavior highlights:
//   - Nil receivers produce nil clones.
//   - Prev remains nil when the original result had path tracking disabled.
//
// Inputs:
//   - None.
//
// Returns:
//   - *Result: a deep copy of the receiver, or nil when the receiver is nil.
//
// Errors:
//   - None.
//
// Determinism:
//   - Produces structurally equivalent copies for the same receiver state.
//
// Complexity:
//   - Time O(V), Space O(V), where V is the total number of entries copied across the stored maps.
//
// Notes:
//   - This method preserves the ownership law of the result surface.
//
// AI-Hints:
//   - Preserve Prev == nil exactly; do not rewrite it into an empty map.
//   - Do not shallow-copy maps here; callers must receive isolated ownership.
func (r *Result) Clone() *Result {
	if r == nil {
		return nil
	}

	clonedResult := &Result{
		SourceID:  r.SourceID,
		Distances: make(map[string]float64, len(r.Distances)),
		Prev:      nil,
	}

	for vertexID, distance := range r.Distances {
		clonedResult.Distances[vertexID] = distance
	}

	if r.Prev != nil {
		clonedResult.Prev = make(map[string]string, len(r.Prev))
		for vertexID, parentID := range r.Prev {
			clonedResult.Prev[vertexID] = parentID
		}
	}

	return clonedResult
}
