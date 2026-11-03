// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dijkstra exposes shortest-path facades and method dispatch.
// This file owns public wrappers that delegate to shared, validated kernels.
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

// Search computes a target-directed shortest path without finalizing full SSSP state.
// It returns one detached, original-edge-ID witness or a known-unreachable result.
//
// Implementation:
//   - Stage 1: Validate public shape and assemble target-specific options.
//   - Stage 2: Clone and validate one effective graph, then dispatch to a working kernel.
//   - Stage 3: Reconstruct and verify original arcs before publishing PathResult.
//
// Behavior highlights:
//   - Target Dijkstra stops only when the target settles, never on first enqueue.
//   - Bidirectional Dijkstra stops only after clean frontiers certify a complete witness.
//   - A* settles the target with frozen consistent heuristics and normalized keys.
//   - Balanced bidirectional A* uses two frozen heuristics and one shared two-frontier engine.
//   - One Clone supplies both edge admission and traversal; source graph is not reread.
//   - Partial tentative labels and incumbent paths are never returned on error.
//
// Inputs:
//   - g: non-nil weighted core graph; core-owned mutators may run concurrently with Clone.
//   - sourceID, targetID: non-empty existing vertex identifiers.
//   - opts: optional, ordered SearchOption callbacks; the caller retains its slice.
//
// Returns:
//   - *PathResult: detached target witness, or +Inf/Reachable=false for known unreachable.
//   - error: nil on a valid reachable or known-unreachable completion; result nil on failure.
//
// Errors:
//   - ErrNilGraph, ErrEmptySourceID, ErrEmptyTargetID for public input shape.
//   - ErrNilOption, ErrInvalidSearchOptions or scalar/method errors during assembly.
//   - ErrUnweightedGraph, ErrSourceNotFound, ErrTargetNotFound from cloned topology.
//   - ErrInvalidWeight, ErrNegativeWeight, ErrDistanceOverflow, ErrInexactArithmetic.
//   - ErrNilHeuristic, ErrInvalidHeuristic, ErrInconsistentHeuristic, ErrSearchKeyOverflow.
//   - Context cancellation and any separate caller-provided cause.
//   - ErrNoPath only for a malformed internal witness, not known unreachability.
//
// Determinism:
//   - Sorted original IDs, strict new-heap tuple order, and first strict improvement.
//   - Bidirectional ties between frontier minima expand forward first.
//
// Complexity:
//   - Time O(V log V + E log E + (V+A) log(A+1) + V*H + A) plus options;
//     H is heuristic callback cost (zero for Dijkstra methods).
//   - Space O(V+E+A), including temporary Clone and detached result; A<=2E.
//
// Notes:
//   - Direct concurrent caller writes through live Edge pointers are unsupported.
//   - The existing DistanceTo/ShortestPathTo wrappers still execute full legacy SSSP.
//
// AI-Hints:
//   - Do not replace this call with legacy DistanceTo: that always runs full SSSP.
//   - Do not publish the target until reconstruction verifies original edge IDs and cost.
//   - Bidirectional uses incoming arcs backward, not outgoing arcs with synthetic reversal.
//   - A* compares original cost to cutoff, never a shifted heap priority.
//   - Balanced stopping compares normalized frontier priorities to a shifted incumbent.
//   - An error must return nil, never a partial PathResult.
func Search(g *core.Graph, sourceID, targetID string, opts ...SearchOption) (*PathResult, error) {
	if g == nil {
		return nil, ErrNilGraph
	}
	if sourceID == "" {
		return nil, ErrEmptySourceID
	}
	if targetID == "" {
		return nil, ErrEmptyTargetID
	}

	config, err := applySearchOptions(opts...)
	if err != nil {
		return nil, err
	}
	if err = checkSearchContext(config.Context); err != nil {
		return nil, err
	}

	graph, source, target, err := prepareSearchGraph(
		config.Context, g, sourceID, targetID, config.InfEdgeThreshold,
	)
	if err != nil {
		return nil, err
	}
	var arcs []int
	var distance float64
	switch config.Method {
	case SearchDijkstra, SearchAStar:
		// Both single-frontier methods share a certified original-cost
		// predecessor relation; only their queue priorities differ.
		var frontier *searchFrontier
		if config.Method == SearchAStar {
			frontier, err = runAStar(config.Context, graph, source, target,
				config.MaxDistance, config.ToTarget)
		} else {
			frontier, err = runTarget(config.Context, graph, source, target,
				config.MaxDistance, nil)
		}
		if err != nil {
			return nil, err
		}
		distance = frontier.distance[target]
		if frontier.settled[target] {
			arcs, err = reconstructSearchPath(graph, source, target, frontier)
			if err != nil {
				return nil, err
			}
		}
	case SearchBidirectional:
		arcs, distance, err = runBidirectional(config.Context, graph, source, target, config.MaxDistance, nil)
		if err != nil {
			return nil, err
		}
	case SearchBidirectionalAStar:
		arcs, distance, err = runBalanced(config.Context, graph, source, target,
			config.MaxDistance, config.ToTarget, config.FromSource)
		if err != nil {
			return nil, err
		}
	default:
		return nil, ErrInvalidSearchMethod
	}

	// All working kernels publish through one original-edge, original-cost verifier.
	// No tentative frontier state or reduced distance escapes this facade.
	return publishSearchPath(config.Context, graph, source, target, config, arcs, distance)
}

// Bidirectional computes one optimal target path using two opposite Dijkstra frontiers.
// It delegates to Search with the bidirectional method and returns its same PathResult.
// Implementation:
//   - Stage 1: Copy caller-owned options into a private slice.
//   - Stage 2: Append the enforced bidirectional method without changing other policies.
//   - Stage 3: Delegate snapshot preparation, certified stopping and publication to Search.
//
// Behavior highlights:
//   - A first meeting is not enough to finish: both clean frontier minima certify stopping.
//   - Directed backward traversal uses only incoming original arcs.
//   - A valid but unreachable target returns +Inf, Reachable=false and nil error.
//
// Inputs:
//   - g: non-nil weighted graph, read through one coherent Clone by Search.
//   - sourceID, targetID: non-empty existing vertex identifiers.
//   - opts: optional caller-owned SearchOption slice; method is fixed by this wrapper.
//
// Returns:
//   - *PathResult: detached optimal witness with original EdgeIDs, or known unreachable.
//
// Errors:
//   - All Search input, numeric, context and witness errors with their sentinel identities.
//
// Determinism:
//   - Search heap and arc ordering; forward expands first on equal opposite minima.
//
// Complexity:
//   - Time O(V log V + E log E + (V+A)log(A+1) + n) including Clone and options.
//   - Space O(V+E+A+n) including snapshot, two frontiers and wrapper options.
//
// Notes:
//   - Unlike DistanceTo, this wrapper never requests full SSSP distances.
//   - It does not mutate or append into caller-owned option-slice backing storage.
//
// AI-Hints:
//   - Do not add another kernel to this wrapper: Search owns policy and dispatch.
//   - Do not silently ignore custom options or publish a first, nonoptimal crossing.
func Bidirectional(g *core.Graph, sourceID, targetID string, opts ...SearchOption) (*PathResult, error) {
	wrapperOptions := make([]SearchOption, 0, len(opts)+1)
	wrapperOptions = append(wrapperOptions, opts...)
	wrapperOptions = append(wrapperOptions, WithSearchMethod(SearchBidirectional))
	return Search(g, sourceID, targetID, wrapperOptions...)
}

// AStar finds an original-cost shortest path using a consistent target heuristic.
// It freezes and verifies the heuristic before delegating to the canonical Search facade.
// Implementation:
//   - Stage 1: Copy caller options into a new slice without altering backing storage.
//   - Stage 2: Append the enforced A* method and explicit target callback.
//   - Stage 3: Delegate graph validation, priority search and publication to Search.
//
// Behavior highlights:
//   - h(target)=0 and exact h(u)<=w(u,v)+h(v) are mandatory on effective arcs.
//   - Only original edge costs affect MaxDistance and the published Distance.
//   - Inconsistent admissible-only heuristics are rejected, not silently reopened.
//
// Inputs:
//   - g: weighted core graph, consumed through one detached Clone by Search.
//   - sourceID, targetID: non-empty existing vertex IDs.
//   - toTarget: non-nil caller-owned function with finite original-cost estimates.
//   - opts: optional SearchOption callbacks; the slice belongs to the caller.
//
// Returns:
//   - *PathResult: detached original-ID witness, or known-unreachable result.
//   - error: nil on success; result nil on any validation, callback or runtime error.
//
// Errors:
//   - ErrNilHeuristic, ErrInvalidHeuristic, ErrInconsistentHeuristic for heuristics.
//   - ErrInexactArithmetic, ErrSearchKeyOverflow and original-cost errors.
//   - All other Search input, policy, context and witness errors.
//
// Determinism:
//   - One callback per vertex in lexical order, then strict heap tuple ordering.
//
// Complexity:
//   - Time O(n + V log V + E log E + V*H + A + (V+A)log(A+1)).
//   - Space O(n+V+E+A); H is callback time, A<=2E.
//
// Notes:
//   - The method and supplied heuristic override earlier valid wrapper options.
//   - The wrapper cannot mutate caller-owned option-slice capacity beyond len.
//
// AI-Hints:
//   - Do not implement another relaxation loop in this wrapper.
//   - Do not use a scaled Euclidean heuristic without verifying edge-cost units.
//   - Callback errors are caller errors; preserve errors.Is rather than parsing text.
func AStar(
	g *core.Graph,
	sourceID, targetID string,
	toTarget Heuristic,
	opts ...SearchOption,
) (*PathResult, error) {
	wrapperOptions := make([]SearchOption, 0, len(opts)+2)
	wrapperOptions = append(wrapperOptions, opts...)
	wrapperOptions = append(wrapperOptions, WithSearchMethod(SearchAStar), WithHeuristic(toTarget))
	return Search(g, sourceID, targetID, wrapperOptions...)
}

// BidirectionalAStar finds an original-cost shortest path using balanced potentials.
// It requires forward- and reverse-consistent estimates and retains original EdgeIDs.
// Implementation:
//   - Stage 1: Copy caller options into a private slice without mutating its capacity.
//   - Stage 2: Append the implemented balanced method and both required heuristics.
//   - Stage 3: Delegate snapshot validation, search and witness verification to Search.
//
// Behavior highlights:
//   - The two heuristics are frozen exactly once per vertex before either frontier expands.
//   - No separate bidirectional relaxation engine or fallback solver is instantiated.
//
// Inputs:
//   - g: weighted core graph, read through the canonical coherent snapshot.
//   - sourceID, targetID: nonempty existing original vertex identifiers.
//   - toTarget: consistent target estimate anchored at zero for targetID.
//   - fromSource: reverse-consistent source estimate anchored at zero for sourceID.
//   - opts: caller-owned optional policies; this wrapper enforces the method and pair.
//
// Returns:
//   - *PathResult: detached original-cost shortest witness or known unreachable.
//
// Errors:
//   - ErrNilHeuristic, ErrInvalidHeuristic, ErrInconsistentHeuristic on invalid callbacks.
//   - ErrInexactArithmetic, ErrSearchKeyOverflow, ErrDistanceOverflow on arithmetic.
//   - Search validation, cancellation, callback and witness errors via errors.Is.
//
// Determinism:
//   - Canonical lexical callbacks, oriented arcs and strict four-component heap keys.
//
// Complexity:
//   - Time O(n + V log V + E log E + V*H + A + (V+A)log(A+1)).
//   - Space O(n + V + E + A); H is combined callback cost, A <= 2E.
//
// Notes:
//   - Both original-cost cutoff and published Distance exclude reduced potentials.
//   - The option slice's unused backing capacity is never overwritten.
//
// AI-Hints:
//   - Do not pass only an admissible heuristic: both directional consistency laws apply.
//   - A callback pair is required even when both estimates are identically zero.
func BidirectionalAStar(
	g *core.Graph,
	sourceID, targetID string,
	toTarget, fromSource Heuristic,
	opts ...SearchOption,
) (*PathResult, error) {
	wrapperOptions := make([]SearchOption, 0, len(opts)+2)
	wrapperOptions = append(wrapperOptions, opts...)
	wrapperOptions = append(wrapperOptions,
		WithSearchMethod(SearchBidirectionalAStar),
		WithBalancedHeuristics(toTarget, fromSource),
	)
	return Search(g, sourceID, targetID, wrapperOptions...)
}
