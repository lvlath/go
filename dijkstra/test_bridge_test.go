// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dijkstra exposes numeric and preparation probes only inside test binaries.
// No test-only API or diagnostic probe is compiled into the production library.
package dijkstra

import (
	"container/heap"
	"context"

	"github.com/lvlath/go/core"
)

// SearchTestSplitSum makes the error-free two-sum inspectable by external tests.
// The returned values preserve production rounding and residual semantics.
func SearchTestSplitSum(left, right float64) (float64, float64, bool) {
	return splitSearchSum(left, right)
}

// SearchTestAddCost exposes cutoff admission without exporting the private kernel.
// External rational oracles can compare the full actual predicate precisely.
func SearchTestAddCost(prefix, weight, limit float64) (float64, bool, error) {
	return addSearchCost(prefix, weight, limit)
}

// SearchTestSumAtLeast exposes the exact stopping predicate only to tests.
// It must not be treated as an alternate production search mode.
func SearchTestSumAtLeast(left, right, bound float64) bool {
	return searchSumAtLeastBound(left, right, bound)
}

// SearchTestThreeAbove exposes the complete crossing-cutoff predicate only to tests.
// Its three terms are independent exact binary64 values, not an accumulator.
func SearchTestThreeAbove(first, middle, last, bound float64) bool {
	return searchThreeSumAboveBound(first, middle, last, bound)
}

// SearchTestHeuristicConsistent exposes one exact signed arc inequality to external tests.
// It never mutates graph topology or normalizes a callback value.
func SearchTestHeuristicConsistent(from, weight, to float64) bool {
	return searchHeuristicConsistent(from, weight, to)
}

// SearchTestCertifyKey exposes signed key exactness only to the external test package.
// Error categories are preserved, without adding production exports.
func SearchTestCertifyKey(left, right float64) (float64, error) {
	return certifySearchKeySum(left, right)
}

// SearchTestValidateAStar executes just the real A* preprocessing on a coherent clone.
// It separates heuristic consistency from later priority arithmetic in external tests.
func SearchTestValidateAStar(graph *core.Graph, sourceID, targetID string, callback Heuristic) error {
	ctx := context.Background()
	prepared, source, target, err := prepareSearchGraph(ctx, graph, sourceID, targetID, DefaultSearchOptions().InfEdgeThreshold)
	if err != nil {
		return err
	}
	_, err = prepareAStarOffsets(ctx, prepared, source, target, callback)
	return err
}

// SearchTestCancelAfterForward probes deterministic cancellation between two expansions.
// The passed callback is invoked after the real forward kernel returns, before reverse work.
func SearchTestCancelAfterForward(
	ctx context.Context,
	graph *core.Graph,
	sourceID, targetID string,
	cancel func(),
) error {
	prepared, source, target, err := prepareSearchGraph(ctx, graph, sourceID, targetID, DefaultSearchOptions().InfEdgeThreshold)
	if err != nil {
		return err
	}
	state := newBidirectionalSearch(prepared, source, target, DefaultSearchOptions().MaxDistance)
	if err := state.expandSearchForward(ctx); err != nil {
		return err
	}
	cancel()
	return state.expandSearchBackward(ctx)
}

// SearchTestBalancedPotential exposes the exact half-difference certificate only in tests.
// It allows external rational oracles to check signed overflow and subnormal behavior.
func SearchTestBalancedPotential(toTarget, fromSource float64) (float64, error) {
	return certifyBalancedPotential(toTarget, fromSource)
}

// SearchTestMeetingTrace exposes certified facts from four real frontier expansions.
// Its values are immutable observations, not a replacement shortest-path solver.
// Implementation:
//   - Stage 1: Record zero-length root certificates on a fresh workspace.
//   - Stage 2: Execute three actual direction expansions and inspect clean minima.
//   - Stage 3: Expand the opposite frontier once more and retain the improved cost.
//
// Behavior highlights:
//   - FirstWitness precedes the final backward improvement; a first crossing is not final.
//
// Inputs:
//   - Fields: observed root, incumbent, heap-bound and final-crossing states.
//
// Returns:
//   - SearchTestMeetingTrace: detached scalar values without private pointers.
//
// Errors:
//   - None in the value; the constructor returns kernel errors.
//
// Determinism:
//   - Four explicit expansions on a fixed source graph and original arc order.
//
// Complexity:
//   - Space O(1) for the trace; private search state is O(V+A).
//
// Notes:
//   - Present only in the test binary and never returned by production Search.
//
// AI-Hints:
//   - Do not claim stopping at FirstCost; inspect the clean lower bound.
type SearchTestMeetingTrace struct {
	ForwardRootCertified  bool    // The source has a certified zero-cost prefix before pop.
	BackwardRootCertified bool    // The target has a certified zero-cost suffix before pop.
	FirstWitness          bool    // A complete crossing exists after three expansions.
	FirstCost             float64 // Certified first crossing in original edge-cost units.
	ForwardMinimum        float64 // Clean forward queue head after the first crossing.
	BackwardMinimum       float64 // Clean backward queue head after the first crossing.
	LowerBoundStops       bool    // True only when the exact stopping predicate allows it.
	FinalCost             float64 // Certified incumbent after the next backward expansion.
}

// SearchTestProbeFirstMeeting executes the original three-plus-one-step test scenario.
// The caller's test asserts the observed order and both crossing certificates.
// Implementation:
//   - Stage 1: Prepare one coherent snapshot and instantiate both real Dijkstra frontiers.
//   - Stage 2: Expand forward, backward, forward and read certified first mu/minima.
//   - Stage 3: Expand backward, then capture the improved incumbent cost.
//
// Behavior highlights:
//   - Uses production expansion functions; does not synthesize a predecessor chain.
//
// Inputs:
//   - graph: weighted core input with a route whose first crossing is inferior.
//   - sourceID, targetID: source and target vertex IDs from the same graph.
//
// Returns:
//   - SearchTestMeetingTrace: detached observations of the actual search work.
//
// Errors:
//   - All preparation, numeric and cancellation errors from delegated production steps.
//
// Determinism:
//   - Fixed four expansions with lexical arc and heap tie ordering.
//
// Complexity:
//   - Time O(V log V+E log E+A log(A+1)), Space O(V+A).
//
// Notes:
//   - This probe exists only to preserve internal regression coverage externally.
//
// AI-Hints:
//   - The public API remains Bidirectional; never publish this trace in production.
func SearchTestProbeFirstMeeting(graph *core.Graph, sourceID, targetID string) (SearchTestMeetingTrace, error) {
	ctx := context.Background()
	prepared, source, target, err := prepareSearchGraph(ctx, graph, sourceID, targetID, DefaultSearchOptions().InfEdgeThreshold)
	if err != nil {
		return SearchTestMeetingTrace{}, err
	}
	state := newBidirectionalSearch(prepared, source, target, DefaultSearchOptions().MaxDistance)
	trace := SearchTestMeetingTrace{
		ForwardRootCertified:  state.forward.certifiedSearchVertex(source),
		BackwardRootCertified: state.backward.certifiedSearchVertex(target),
	}
	for _, expand := range []func(context.Context) error{
		state.expandSearchForward, state.expandSearchBackward, state.expandSearchForward,
	} {
		if err := expand(ctx); err != nil {
			return SearchTestMeetingTrace{}, err
		}
	}
	trace.FirstWitness = state.hasWitness
	trace.FirstCost = state.incumbent.distance
	f, fOK := state.forward.cleanSearchHead()
	b, bOK := state.backward.cleanSearchHead()
	if fOK && bOK {
		trace.ForwardMinimum = f.distance
		trace.BackwardMinimum = b.distance
		trace.LowerBoundStops = searchSumAtLeastBound(f.distance, b.distance, state.incumbent.distance)
	}
	if err := state.expandSearchBackward(ctx); err != nil {
		return SearchTestMeetingTrace{}, err
	}
	trace.FinalCost = state.incumbent.distance
	return trace, nil
}

// SearchTestProbeRootJoin records a real crossing before target-root settlement.
// It proves the backward root is a certified zero-cost suffix upon initialization.
// Implementation:
//   - Stage 1: Prepare the graph and both root-owned search frontiers.
//   - Stage 2: Expand forward while keeping the backward root in its queue.
//   - Stage 3: Observe whether a complete original-edge crossing was certified.
//
// Behavior highlights:
//   - Only a real source-to-target arc can produce a direct crossing.
//
// Inputs:
//   - graph: weighted core input with original directed edge identities.
//   - sourceID, targetID: valid original vertex identities.
//
// Returns:
//   - backwardSettled: whether target root was popped; witness: certified join flag.
//   - cost: original-cost incumbent; error: any preparation or expansion failure.
//
// Errors:
//   - All numeric/input errors are preserved from the production functions.
//
// Determinism:
//   - Exactly one forward expansion over sorted outgoing original arcs.
//
// Complexity:
//   - Time O(V log V+E log E+A log(A+1)), Space O(V+A).
//
// Notes:
//   - Test-only observation; no additional search entry point exists in production.
//
// AI-Hints:
//   - Root certification is independent of settlement, not a tentative label.
func SearchTestProbeRootJoin(graph *core.Graph, sourceID, targetID string) (backwardSettled, witness bool, cost float64, err error) {
	ctx := context.Background()
	prepared, source, target, err := prepareSearchGraph(ctx, graph, sourceID, targetID, DefaultSearchOptions().InfEdgeThreshold)
	if err != nil {
		return false, false, 0, err
	}
	state := newBidirectionalSearch(prepared, source, target, DefaultSearchOptions().MaxDistance)
	if err := state.expandSearchForward(ctx); err != nil {
		return false, false, 0, err
	}
	return state.backward.settled[target], state.hasWitness, state.incumbent.distance, nil
}

// SearchTestProbeStaleHead drives the real lazy min-heap and its stale cleanup.
// It returns live minimum state before and after the only valid nonroot label settles.
// Implementation:
//   - Stage 1: Insert an obsolete and a current value for the same vertex.
//   - Stage 2: Clean the previously settled root and read the valid head.
//   - Stage 3: Settle that vertex, then verify no live head remains.
//
// Behavior highlights:
//   - A stale heap item never participates in the stopping bound.
//
// Inputs:
//   - None; fixed finite numeric fixture and a three-vertex frontier.
//
// Returns:
//   - cost: currently valid candidate, present: first head validity.
//   - remaining: true if any live candidate remains after settlement.
//
// Errors:
//   - None; heap inputs are private fixed values.
//
// Determinism:
//   - Exact heap tuple with first strict improvement and stale-item removal.
//
// Complexity:
//   - Time O(log H), Space O(1) for a constant-size test queue.
//
// Notes:
//   - No test-only production API is introduced by this observation.
//
// AI-Hints:
//   - Do not read a lazy queue head without discarding settled/stale values.
func SearchTestProbeStaleHead() (cost float64, present, remaining bool) {
	frontier := newSearchFrontier(3, 0)
	frontier.distance[1] = 2
	heap.Push(&frontier.queue, searchItem{vertex: 1, priority: 10, distance: 10, sequence: 1})
	heap.Push(&frontier.queue, searchItem{vertex: 1, priority: 2, distance: 2, sequence: 2})
	frontier.settled[0] = true
	head, ok := frontier.cleanSearchHead()
	if !ok {
		return 0, false, false
	}
	frontier.settled[1] = true
	_, after := frontier.cleanSearchHead()
	return head.distance, true, after
}

// SearchTestProbeDamagedSuffix injects one malformed private successor link.
// Real reconstruction must reject the incomplete path with ErrNoPath.
// Implementation:
//   - Stage 1: Prepare an original S->A->T directed graph snapshot.
//   - Stage 2: Mark A settled backward but remove its required A->T successor.
//   - Stage 3: Ask the real join reconstructor for a complete route.
//
// Behavior highlights:
//   - The reconstructor must not panic, loop or publish partial original EdgeIDs.
//
// Inputs:
//   - graph: weighted graph containing the bridge identified by bridgeID.
//   - sourceID, targetID, middleID: original vertex identities.
//   - bridgeID: actual original edge from sourceID to middleID.
//
// Returns:
//   - error: ErrNoPath for a missing backward successor.
//
// Errors:
//   - Preparation errors or ErrNoPath from reconstruction.
//
// Determinism:
//   - Original edge IDs select the exact bridge, not unordered neighbors.
//
// Complexity:
//   - Time O(V log V+E log E+V), Space O(V+E).
//
// Notes:
//   - This malformed state is possible only in the test bridge, not via public Search.
//
// AI-Hints:
//   - Do not return a partial path from a corrupted predecessor/successor relation.
func SearchTestProbeDamagedSuffix(graph *core.Graph, sourceID, targetID, middleID, bridgeID string) error {
	ctx := context.Background()
	prepared, source, target, err := prepareSearchGraph(ctx, graph, sourceID, targetID, DefaultSearchOptions().InfEdgeThreshold)
	if err != nil {
		return err
	}
	state := newBidirectionalSearch(prepared, source, target, DefaultSearchOptions().MaxDistance)
	middle, exists := prepared.index[middleID]
	if !exists {
		return ErrNoPath
	}
	bridge := noSearchArc
	for _, index := range prepared.out[source] {
		arc := prepared.arcs[index]
		if arc.to == middle && prepared.edges[arc.edge].ID == bridgeID {
			bridge = index
			break
		}
	}
	if bridge == noSearchArc {
		return ErrNoPath
	}
	state.backward.settled[middle] = true
	state.backward.linkArc[middle] = noSearchArc
	_, err = reconstructBidirectionalPath(prepared, source, target, state.forward, state.backward,
		searchJoin{from: source, to: middle, arc: bridge})
	return err
}

// SearchTestProbePrunedJoin tests whole-crossing cutoff before partial materialization.
// The supplied certified segment labels are synthetic, but the join code is production.
// Implementation:
//   - Stage 1: Prepare a graph with a real original bridge arc.
//   - Stage 2: Install two certified segment costs and identify their exact edge ID.
//   - Stage 3: Offer the complete crossing to the real incumbent logic.
//
// Behavior highlights:
//   - A complete route outside cutoff must not fail on an inexact partial sum.
//
// Inputs:
//   - graph: original weighted graph containing bridgeID from fromID to toID.
//   - sourceID, targetID: original terminal IDs; fromID, toID: bridge endpoints.
//   - bridgeID: real oriented edge ID for the crossing.
//   - forwardCost, backwardCost: certified synthetic segment costs.
//   - limit: finite original-cost upper bound.
//
// Returns:
//   - hasWitness: whether an incumbent was published; err: numeric/join error.
//
// Errors:
//   - Any preparation or checked arithmetic failure, preserving sentinel identity.
//
// Determinism:
//   - Original edge ID and fixed certified labels select the unique test join.
//
// Complexity:
//   - Time O(V log V+E log E+E), Space O(V+E).
//
// Notes:
//   - Test-only numerical fixture; production always obtains segment labels by settlement.
//
// AI-Hints:
//   - Do not compute forward+bridge before rejecting a complete out-of-cutoff join.
func SearchTestProbePrunedJoin(
	graph *core.Graph,
	sourceID, targetID, fromID, toID, bridgeID string,
	forwardCost, backwardCost, limit float64,
) (hasWitness bool, err error) {
	ctx := context.Background()
	prepared, source, target, err := prepareSearchGraph(ctx, graph, sourceID, targetID, DefaultSearchOptions().InfEdgeThreshold)
	if err != nil {
		return false, err
	}
	from, fromOK := prepared.index[fromID]
	to, toOK := prepared.index[toID]
	if !fromOK || !toOK {
		return false, ErrNoPath
	}
	bridge := noSearchArc
	for _, index := range prepared.out[from] {
		arc := prepared.arcs[index]
		if arc.to == to && prepared.edges[arc.edge].ID == bridgeID {
			bridge = index
			break
		}
	}
	if bridge == noSearchArc {
		return false, ErrNoPath
	}
	state := newBidirectionalSearch(prepared, source, target, limit)
	state.forward.settled[from] = true
	state.backward.settled[to] = true
	state.forward.distance[from] = forwardCost
	state.backward.distance[to] = backwardCost
	if err := state.offerSearchJoin(from, to, bridge); err != nil {
		return false, err
	}
	return state.hasWitness, nil
}
