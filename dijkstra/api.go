// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package dijkstra

import "github.com/lvlath/go/core"

// Dijkstra runs deterministic single-source shortest paths over a weighted graph
// and returns a detached result object that exposes distances and optional path
// reconstruction helpers.
//
// Implementation:
//   - Stage 1: Validate facade-level inputs.
//   - Stage 2: Assemble the finalized runtime policy through applyOptions.
//   - Stage 3: Delegate the actual shortest-path computation to the canonical kernel.
//
// Behavior highlights:
//   - sourceID is an explicit required input and is never taken from options.
//   - The returned result is detached from the graph and safe for post-run queries.
//   - Path reconstruction remains disabled unless explicitly requested by options or wrappers.
//
// Inputs:
//   - g: the weighted graph to traverse.
//   - sourceID: the source vertex identifier for the traversal.
//   - opts: zero or more functional runtime options.
//
// Returns:
//   - *Result: the detached shortest-path result for the requested source.
//
// Errors:
//   - ErrNilGraph if g is nil.
//   - ErrEmptySourceID if sourceID is empty.
//   - ErrNilOption if a nil option is supplied.
//   - Any option-validation error returned by applyOptions.
//   - Any validation or runtime error returned by the canonical kernel.
//
// Determinism:
//   - The facade preserves the deterministic behavior of the canonical kernel and adds no hidden ordering.
//
// Complexity:
//   - Time O(n) for option assembly plus the delegated kernel cost.
//   - Space O(1) in the facade itself, excluding the delegated kernel result.
//
// Notes:
//   - The facade owns orchestration only and must not embed alternative shortest-path logic.
//
// AI-Hints:
//   - Do not move sourceID back into options.
//   - Do not place graph validation, heap logic, or relaxation logic into this facade.
func Dijkstra(g *core.Graph, sourceID string, opts ...Option) (*Result, error) {
	if g == nil {
		return nil, ErrNilGraph
	}
	if sourceID == "" {
		return nil, ErrEmptySourceID
	}

	config, err := applyOptions(opts...)
	if err != nil {
		return nil, err
	}

	return runDijkstra(g, sourceID, config)
}

// Distances runs Dijkstra and returns a deep copy of the finalized distance map.
// This wrapper is intended for callers that need distances only and do not want
// to interact with the full result object.
//
// Implementation:
//   - Stage 1: Delegate to Dijkstra.
//   - Stage 2: Deep-copy the returned distance map.
//   - Stage 3: Return the isolated copy to the caller.
//
// Behavior highlights:
//   - The wrapper isolates ownership of the returned map from the underlying result.
//   - The wrapper does not require path tracking.
//
// Inputs:
//   - g: the weighted graph to traverse.
//   - sourceID: the source vertex identifier for the traversal.
//   - opts: zero or more functional runtime options.
//
// Returns:
//   - map[string]float64: a detached copy of the finalized distance map.
//
// Errors:
//   - Any error returned by Dijkstra.
//
// Determinism:
//   - The wrapper preserves the deterministic result produced by Dijkstra.
//
// Complexity:
//   - Time O(V) and Space O(V) for copying the distance map, in addition to the delegated kernel cost.
//
// Notes:
//   - Returning the internal result map directly is forbidden because it weakens ownership isolation.
//
// AI-Hints:
//   - Keep the map copy explicit; do not leak shared mutable ownership through convenience wrappers.
//   - Do not call Clone here just to obtain distances; that would also copy Prev unnecessarily.
func Distances(g *core.Graph, sourceID string, opts ...Option) (map[string]float64, error) {
	result, err := Dijkstra(g, sourceID, opts...)
	if err != nil {
		return nil, err
	}

	clonedDistances := make(map[string]float64, len(result.Distances))
	for vertexID, distance := range result.Distances {
		clonedDistances[vertexID] = distance
	}

	return clonedDistances, nil
}

// DistanceTo computes single-source shortest paths and returns the finalized
// distance to one requested target.
// The wrapper provides point-query ergonomics while preserving the canonical
// Dijkstra execution and Result lookup contracts.
//
// Implementation:
//   - Stage 1: Reject an empty target identifier.
//   - Stage 2: Delegate graph validation, option assembly, and traversal to Dijkstra.
//   - Stage 3: Resolve the target through Result.DistanceTo.
//   - Stage 4: Return the stored finite distance or canonical +Inf value unchanged.
//
// Behavior highlights:
//   - The wrapper does not require predecessor tracking.
//   - The wrapper does not perform target-directed early termination; it executes
//     the same full single-source kernel as Dijkstra.
//   - A known unreachable target returns +Inf with nil error.
//   - An unknown target returns ErrTargetNotFound.
//   - Caller-provided options are preserved exactly; passing WithPathTracking is
//     valid but unnecessary when only a distance is consumed.
//
// Inputs:
//   - g: the weighted graph to traverse; must satisfy the Dijkstra graph contract.
//   - sourceID: the required source vertex identifier.
//   - targetID: the required target vertex identifier.
//   - opts: zero or more runtime policy options.
//
// Returns:
//   - float64: the finalized target distance when error is nil.
//   - On error, the returned numeric value is zero and must not be interpreted.
//
// Errors:
//   - ErrEmptyTargetID if targetID is empty.
//   - ErrNilGraph, ErrEmptySourceID, ErrUnweightedGraph, or ErrSourceNotFound
//     when public input validation fails.
//   - ErrNilOption, ErrBadMaxDistance, or ErrBadInfEdgeThreshold
//     when option assembly fails.
//   - ErrInvalidWeight, ErrNegativeWeight, or ErrDistanceOverflow
//     when numeric validation or relaxation fails.
//   - ErrTargetNotFound if targetID is absent from the completed result domain.
//   - Any preserved graph-surface error returned by the canonical kernel.
//
// Determinism:
//   - The wrapper preserves the graph-order, heap tie-break, and strict-improvement
//     laws of Dijkstra.
//   - The point lookup itself is deterministic for the completed Result.
//
// Complexity:
//   - Total time equals the delegated Dijkstra cost plus O(1) target lookup.
//   - Peak space equals the delegated Dijkstra working/result state plus O(1)
//     wrapper-local storage.
//
// Notes:
//   - +Inf is valid only when error is nil and means “known but unreachable under
//     the effective traversal policy”.
//   - Use ShortestPathTo when both the witness and its distance are required.
//
// AI-Hints:
//   - Do not advertise this wrapper as an early-exit target-search optimization.
//   - Do not bypass Result.DistanceTo with direct map access inside the wrapper.
//   - Do not translate +Inf into ErrNoPath; distance and path-query surfaces have
//     intentionally different contracts.
func DistanceTo(g *core.Graph, sourceID, targetID string, opts ...Option) (float64, error) {
	if targetID == "" {
		return 0, ErrEmptyTargetID
	}

	result, err := Dijkstra(g, sourceID, opts...)
	if err != nil {
		return 0, err
	}

	return result.DistanceTo(targetID)
}

// ShortestPathTo runs Dijkstra with path tracking enabled and returns one
// deterministic shortest-path witness together with its finalized distance.
//
// Implementation:
//   - Stage 1: Validate the target identifier.
//   - Stage 2: Build an isolated option slice and append WithPathTracking.
//   - Stage 3: Delegate to Dijkstra.
//   - Stage 4: Resolve the path witness and distance through the result surface.
//
// Behavior highlights:
//   - The wrapper enforces path tracking explicitly.
//   - Caller-owned option slices are not mutated.
//   - The wrapper returns one deterministic shortest-path witness, not all shortest paths.
//
// Inputs:
//   - g: the weighted graph to traverse.
//   - sourceID: the source vertex identifier for the traversal.
//   - targetID: the target vertex identifier to reconstruct.
//   - opts: zero or more functional runtime options.
//
// Returns:
//   - []string: one deterministic shortest-path witness from source to target.
//   - float64: the finalized shortest-path distance to the requested target.
//
// Errors:
//   - ErrEmptyTargetID if targetID is empty.
//   - Any error returned by Dijkstra.
//   - Any result-surface error returned by PathTo or DistanceTo.
//
// Determinism:
//   - The wrapper preserves the deterministic predecessor and path reconstruction laws of the kernel and result surface.
//
// Complexity:
//   - Time O(m + k) after the delegated kernel cost, where m is the number of options copied and k is the path length.
//   - Space O(m + k) in the wrapper and returned path, excluding the delegated kernel result.
//
// Notes:
//   - Wrapper policy wins by explicitly appending WithPathTracking to the local option slice copy.
//
// AI-Hints:
//   - Copy the option slice before appending wrapper-enforced options to avoid aliasing caller-owned storage.
//   - Do not reconstruct paths manually in this wrapper; the result surface is the canonical path API.
func ShortestPathTo(g *core.Graph, sourceID, targetID string, opts ...Option) ([]string, float64, error) {
	if targetID == "" {
		return nil, 0, ErrEmptyTargetID
	}

	wrapperOptions := make([]Option, 0, len(opts)+1)
	wrapperOptions = append(wrapperOptions, opts...)
	wrapperOptions = append(wrapperOptions, WithPathTracking())

	result, err := Dijkstra(g, sourceID, wrapperOptions...)
	if err != nil {
		return nil, 0, err
	}

	path, err := result.PathTo(targetID)
	if err != nil {
		return nil, 0, err
	}

	distance, err := result.DistanceTo(targetID)
	if err != nil {
		return nil, 0, err
	}

	return path, distance, nil
}
