// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dijkstra implements target-directed shortest-path operations.
// This file owns shared two-frontier settlement, certified crossings and exact stopping.
package dijkstra

import (
	"container/heap"
	"context"
	"fmt"
	"math"
)

// searchJoin identifies one certified connection between opposite search trees.
// The forward tree ends at from, and the backward tree starts at to; an arc
// between them is optional when both trees contain the same vertex.
// Implementation:
//   - Stage 1: Retain two settled (or zero-cost root) endpoints.
//   - Stage 2: Retain a real crossing arc, or noSearchArc for a shared vertex.
//   - Stage 3: Retain the exact representable original-cost incumbent.
//
// Behavior highlights:
//   - The join never refers to tentative or subsequently replaceable parents.
//
// Inputs:
//   - from, to: certified dense indices; arc: a real original oriented arc or noSearchArc.
//   - distance: complete path cost, not a priority or reduced distance.
//
// Returns:
//   - searchJoin: private value sufficient to reconstruct a complete witness.
//
// Errors:
//   - None; the producer checks arithmetic and endpoints.
//
// Determinism:
//   - Strictly better costs replace a join; equal costs retain the first witness.
//
// Complexity:
//   - Time O(1), Space O(1) per value.
//
// Notes:
//   - Original edge identity is obtained from searchGraph.arcs[arc] during reconstruction.
//
// AI-Hints:
//   - Do not keep only a meeting vertex: a directed crossing edge is often essential.
type searchJoin struct {
	from     int     // Last vertex on the certified source-side prefix.
	to       int     // First vertex on the certified target-side suffix.
	arc      int     // Original oriented bridge; noSearchArc means a common vertex.
	distance float64 // Exact original path cost across the complete join.
}

// bidirectionalSearch coordinates two independent Dijkstra frontiers over one graph.
// Its incumbent is a certified complete witness and never a tentative estimate.
// Optional balanced offsets change only priority ordering and incumbent comparison scale.
// Implementation:
//   - Stage 1: Keep separate forward and backward labels, parent links and heaps.
//   - Stage 2: Offer shared-vertex and concrete-edge connections to the incumbent.
//   - Stage 3: Stop only when clean minima in the same key scale certify optimality.
//
// Behavior highlights:
//   - Forward traverses outgoing arcs; backward traverses original incoming arcs.
//   - Both roots are already certified zero-cost witnesses before their first heap pop.
//
// Inputs:
//   - graph: detached effective topology; limit: inclusive original-cost cutoff.
//   - balanced: optional frozen normalized directional offsets; nil for Dijkstra.
//   - forward, backward: private, disjoint mutable workspaces.
//
// Returns:
//   - bidirectionalSearch: one invocation's private search state.
//
// Errors:
//   - None from the type; expansion methods propagate cost and context errors.
//
// Determinism:
//   - Lexical arc order and the shared heap tuple; equal minima choose forward.
//
// Complexity:
//   - Space O(V+A) for two label sets and lazy heap entries, A<=2E.
//
// Notes:
//   - No pointers to this state escape into PathResult.
//
// AI-Hints:
//   - A first crossing is not a stopping certificate; clean both heaps first.
//   - Never use outgoing arcs to expand the backward frontier on a directed graph.
type bidirectionalSearch struct {
	graph        *searchGraph              // Validated original graph with out/in arc indexes.
	forward      *searchFrontier           // Certified prefixes from the requested source.
	backward     *searchFrontier           // Certified suffixes to the requested target.
	limit        float64                   // Inclusive original-cost policy, or +Inf.
	incumbent    searchJoin                // Best certified complete original-cost crossing.
	incumbentKey float64                   // Its normalized bound: mu + endpoint potential shift.
	hasWitness   bool                      // Distinguishes no crossing from a finite incumbent.
	balanced     *searchBalancedPotentials // Nil for original-cost bidirectional Dijkstra.
}

// newBidirectionalSearch allocates two independent frontiers for one target query.
// Both roots represent immutable zero-cost paths even before their queued pops.
// Implementation:
//   - Stage 1: Allocate both dense label/parent/settlement sets.
//   - Stage 2: Insert one zero-distance root candidate into each heap.
//   - Stage 3: Initialize the incumbent to an absent +Inf witness.
//
// Behavior highlights:
//   - No graph data is cloned or sorted during frontier initialization.
//
// Inputs:
//   - graph: fully prepared effective graph; source, target: valid dense indices.
//   - limit: validated inclusive original-cost cutoff.
//
// Returns:
//   - *bidirectionalSearch: freshly owned, mutable workspace.
//
// Errors:
//   - None for inputs already validated by the Search facade.
//
// Determinism:
//   - Both roots get sequence zero in independent heaps.
//
// Complexity:
//   - Time O(V), Space O(V), excluding future queue growth.
//
// Notes:
//   - This state is not shared between concurrent Search invocations.
//
// AI-Hints:
//   - Keep predecessor arrays independent: backward links encode successors, not forward parents.
func newBidirectionalSearch(graph *searchGraph, source, target int, limit float64) *bidirectionalSearch {
	return &bidirectionalSearch{
		graph:    graph,
		forward:  newSearchFrontier(len(graph.ids), source),
		backward: newSearchFrontier(len(graph.ids), target),
		limit:    limit,
		incumbent: searchJoin{
			arc: noSearchArc, distance: math.Inf(1),
		},
	}
}

// certifiedSearchVertex reports a final path certificate for one frontier vertex.
// The root is certified at initialization; every other vertex is certified on pop.
// Implementation:
//   - Stage 1: Check whether the index names the immutable zero-cost root.
//   - Stage 2: Otherwise require a non-stale, settled distance label.
//
// Behavior highlights:
//   - A finite tentative distance alone is not a complete witness certificate.
//
// Inputs:
//   - vertex: valid dense index in the frontier's graph.
//
// Returns:
//   - bool: true only when the stored parent chain can no longer improve.
//
// Errors:
//   - None for valid indices.
//
// Determinism:
//   - Pure root/settled lookup.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Root certification prevents missing zero-length joins on an unpopped frontier.
//
// AI-Hints:
//   - Do not equate heap membership with settlement; lazy entries may be stale.
func (frontier *searchFrontier) certifiedSearchVertex(vertex int) bool {
	return vertex == frontier.root || frontier.settled[vertex]
}

// cleanSearchHead removes stale heap roots before reading a frontier lower bound.
// The returned head is the smallest live tentative distance in that direction.
// Implementation:
//   - Stage 1: Inspect the minimum queue item without removing a valid item.
//   - Stage 2: Discard entries for settled vertices or superseded distances.
//   - Stage 3: Return the first valid item, or report an exhausted frontier.
//
// Behavior highlights:
//   - A stale key must never participate in the bidirectional stopping condition.
//
// Inputs:
//   - None; receiver is one private mutable frontier.
//
// Returns:
//   - searchItem: the clean minimum, bool: false when no live items remain.
//
// Errors:
//   - None; heap contents were validated before each push.
//
// Determinism:
//   - Uses the fixed (priority, distance, vertex, sequence) heap order.
//
// Complexity:
//   - Amortized O(log(A+1)) per discarded item, O(1) for an immediately valid head.
//
// Notes:
//   - Every stale item is discarded at most once across the entire run.
//
// AI-Hints:
//   - Never compare queue[0] to an incumbent without this stale cleanup.
func (frontier *searchFrontier) cleanSearchHead() (searchItem, bool) {
	for frontier.queue.Len() > 0 {
		item := frontier.queue[0]
		if !frontier.settled[item.vertex] && item.distance == frontier.distance[item.vertex] {
			return item, true
		}
		heap.Pop(&frontier.queue)
	}
	return searchItem{}, false
}

// offerSearchJoin considers one complete source-to-target connection.
// The endpoints already own settled parent chains or immutable root certificates.
// Implementation:
//   - Stage 1: Reject whole out-of-cutoff crossings by exact two-sum comparison.
//   - Stage 2: Extend forward cost and backward suffix with checked arithmetic.
//   - Stage 3: Translate a strictly improved original-cost incumbent into the shared key scale.
//
// Behavior highlights:
//   - A joined path has a real crossing arc; noSearchArc means a shared vertex.
//   - Out-of-cutoff joins do not manufacture overflow or reachable results.
//   - Full three-term cutoff admission precedes intermediate storage checks.
//   - The accepted original-cost incumbent gets a separate checked reduced bound.
//
// Inputs:
//   - from: certified forward vertex; to: certified backward vertex.
//   - bridge: oriented arc index from->to, or noSearchArc for from==to.
//
// Returns:
//   - error: nil on an accepted, pruned, or non-improving complete connection.
//
// Errors:
//   - ErrInexactArithmetic or ErrDistanceOverflow from admitted original-cost joins.
//   - ErrSearchKeyOverflow for an overflowing balanced incumbent bound.
//
// Determinism:
//   - Candidate evaluation order follows pop order and original arc indexes.
//
// Complexity:
//   - Time O(1), Space O(1) per offered connection.
//
// Notes:
//   - A cyclic join cannot strictly improve: any shared ancestor already produced
//     an equal-or-cheaper common-vertex candidate when it became certified twice.
//
// AI-Hints:
//   - Do not update the incumbent with unrelated tentative labels.
//   - Do not use bridge endpoints without preserving its original edge identity.
func (search *bidirectionalSearch) offerSearchJoin(from, to, bridge int) error {
	if !search.forward.certifiedSearchVertex(from) || !search.backward.certifiedSearchVertex(to) {
		return nil
	}

	cost := search.forward.distance[from]
	if bridge != noSearchArc {
		arc := search.graph.arcs[bridge]
		if arc.from != from || arc.to != to {
			return ErrNoPath
		}
		edge := search.graph.edges[arc.edge]

		// The complete crossing has three non-negative terms. Compare that
		// mathematical total with the cutoff before requiring the intermediate
		// forward+bridge sum to fit in one binary64 label.
		if !math.IsInf(search.limit, 1) && searchThreeSumAboveBound(
			cost, edge.Weight, search.backward.distance[to], search.limit,
		) {
			return nil
		}
		next, allowed, err := addSearchCost(cost, edge.Weight, search.limit)
		if err != nil {
			return fmt.Errorf("dijkstra: crossing edge_id=%q from=%q to=%q: %w",
				edge.ID, search.graph.ids[from], search.graph.ids[to], err)
		}
		if !allowed {
			return nil
		}
		cost = next
	} else if from != to {
		return ErrNoPath
	}

	completed, allowed, err := addSearchCost(cost, search.backward.distance[to], search.limit)
	if err != nil {
		return fmt.Errorf("dijkstra: joining forward=%q backward=%q: %w",
			search.graph.ids[from], search.graph.ids[to], err)
	}
	if allowed && (!search.hasWitness || completed < search.incumbent.distance) {
		// The incumbent remains an original-cost witness. Only its stopping
		// certificate is shifted into the two normalized heap-key domains.
		// Check the key BEFORE replacing the accepted original-cost witness.
		incumbentKey := completed
		if search.balanced != nil {
			incumbentKey, err = certifySearchKeySum(completed, search.balanced.shift)
			if err != nil {
				return fmt.Errorf("dijkstra: reduced incumbent from=%q to=%q: %w",
					search.graph.ids[from], search.graph.ids[to], err)
			}
		}
		search.incumbent = searchJoin{from: from, to: to, arc: bridge, distance: completed}
		search.incumbentKey = incumbentKey
		search.hasWitness = true
	}
	return nil
}

// expandSearchForward settles the next clean source-side label and scans outgoing arcs.
// It offers certified crossings before applying same-direction relaxation skips.
// Implementation:
//   - Stage 1: Pop the already-clean minimum and certify its forward prefix.
//   - Stage 2: Offer a common-vertex join and each oriented outgoing crossing.
//   - Stage 3: Strictly improve unsettled neighbors with checked original costs.
//
// Behavior highlights:
//   - A visited neighbor still permits a crossing offer, but not a relaxation.
//   - Balanced mode computes one exact normalized priority per strict improvement.
//   - The source root never acquires a parent through a zero-weight cycle.
//
// Inputs:
//   - ctx: non-nil, cooperatively checked cancellation context.
//
// Returns:
//   - error: nil for successful settlement and complete arc scan.
//
// Errors:
//   - Context cancellation, ErrInexactArithmetic, ErrDistanceOverflow,
//     ErrSearchKeyOverflow, or ErrNoPath.
//
// Determinism:
//   - Minimum heap tuple first; outgoing arcs in original edge/orientation order.
//
// Complexity:
//   - O((degreeOut+1) log(A+1)) time, O(degreeOut) possible new queue entries.
//
// Notes:
//   - The caller cleans both heads and chooses which direction expands.
//
// AI-Hints:
//   - Check a possible crossing before skipping an already-settled neighbor.
func (search *bidirectionalSearch) expandSearchForward(ctx context.Context) error {
	if err := checkSearchContext(ctx); err != nil {
		return err
	}
	item := heap.Pop(&search.forward.queue).(searchItem)
	current := item.vertex
	search.forward.settled[current] = true

	if err := search.offerSearchJoin(current, current, noSearchArc); err != nil {
		return err
	}
	for _, arcIndex := range search.graph.out[current] {
		if err := checkSearchContext(ctx); err != nil {
			return err
		}
		arc := search.graph.arcs[arcIndex]
		if err := search.offerSearchJoin(current, arc.to, arcIndex); err != nil {
			return err
		}
		if search.forward.settled[arc.to] {
			continue
		}
		edge := search.graph.edges[arc.edge]
		candidate, allowed, err := addSearchCost(search.forward.distance[current], edge.Weight, search.limit)
		if err != nil {
			return fmt.Errorf("dijkstra: forward edge_id=%q from=%q to=%q: %w",
				edge.ID, search.graph.ids[current], search.graph.ids[arc.to], err)
		}
		if !allowed || candidate >= search.forward.distance[arc.to] {
			continue
		}
		// The original-cost distance remains the relaxation label. Only
		// the queue priority is translated; a failed key cannot publish a
		// partial parent or alter the certified incumbent.
		priority := candidate
		if search.balanced != nil {
			priority, err = certifySearchKeySum(candidate, search.balanced.forward[arc.to])
			if err != nil {
				return fmt.Errorf("dijkstra: forward reduced key edge_id=%q vertex_id=%q: %w",
					edge.ID, search.graph.ids[arc.to], err)
			}
		}
		search.forward.distance[arc.to] = candidate
		search.forward.linkArc[arc.to] = arcIndex
		heap.Push(&search.forward.queue, searchItem{
			vertex: arc.to, distance: candidate, priority: priority,
			sequence: search.forward.sequence,
		})
		search.forward.sequence++
	}
	return nil
}

// expandSearchBackward settles one target-side suffix on the original reverse relation.
// Incoming arcs u->v are scanned from v, retaining their original u->v identity.
// Implementation:
//   - Stage 1: Certify the current vertex-to-target suffix.
//   - Stage 2: Offer common-vertex and incoming crossing witnesses.
//   - Stage 3: Relax predecessors while recording their original forward successor arcs.
//
// Behavior highlights:
//   - The backward distance at u measures original u->target cost, not target->u.
//   - Backward linkArc[u] is the real u->successor arc for reconstruction.
//   - Its optional normalized key uses p(target)-p(u) and not forward offsets.
//
// Inputs:
//   - ctx: non-nil, cooperatively checked cancellation context.
//
// Returns:
//   - error: nil after processing every incoming arc.
//
// Errors:
//   - Context cancellation, ErrInexactArithmetic, ErrDistanceOverflow,
//     ErrSearchKeyOverflow, or ErrNoPath.
//
// Determinism:
//   - Minimum heap tuple followed by incoming arcs in original edge order.
//
// Complexity:
//   - O((degreeIn+1) log(A+1)) time, O(degreeIn) possible queue entries.
//
// Notes:
//   - Never reverse a directed Edge or synthesize a different edge ID.
//
// AI-Hints:
//   - Use graph.in[current], not graph.out[current], for reverse traversal.
func (search *bidirectionalSearch) expandSearchBackward(ctx context.Context) error {
	if err := checkSearchContext(ctx); err != nil {
		return err
	}
	item := heap.Pop(&search.backward.queue).(searchItem)
	current := item.vertex
	search.backward.settled[current] = true

	if err := search.offerSearchJoin(current, current, noSearchArc); err != nil {
		return err
	}
	for _, arcIndex := range search.graph.in[current] {
		if err := checkSearchContext(ctx); err != nil {
			return err
		}
		arc := search.graph.arcs[arcIndex]
		if err := search.offerSearchJoin(arc.from, current, arcIndex); err != nil {
			return err
		}
		if search.backward.settled[arc.from] {
			continue
		}
		edge := search.graph.edges[arc.edge]
		candidate, allowed, err := addSearchCost(search.backward.distance[current], edge.Weight, search.limit)
		if err != nil {
			return fmt.Errorf("dijkstra: backward edge_id=%q from=%q to=%q: %w",
				edge.ID, search.graph.ids[arc.from], search.graph.ids[current], err)
		}
		if !allowed || candidate >= search.backward.distance[arc.from] {
			continue
		}
		// Reverse original costs are still v->target labels. The queue
		// receives the normalized backward key, never a fabricated reverse
		// edge or a reduced original-cost distance.
		priority := candidate
		if search.balanced != nil {
			priority, err = certifySearchKeySum(candidate, search.balanced.backward[arc.from])
			if err != nil {
				return fmt.Errorf("dijkstra: backward reduced key edge_id=%q vertex_id=%q: %w",
					edge.ID, search.graph.ids[arc.from], err)
			}
		}
		search.backward.distance[arc.from] = candidate
		search.backward.linkArc[arc.from] = arcIndex
		heap.Push(&search.backward.queue, searchItem{
			vertex: arc.from, distance: candidate, priority: priority,
			sequence: search.backward.sequence,
		})
		search.backward.sequence++
	}
	return nil
}

// reconstructBidirectionalPath joins certified parent and successor chains.
// It preserves the real oriented crossing arc and uses no tentative links.
// Implementation:
//   - Stage 1: Reconstruct the source->from settled predecessor chain.
//   - Stage 2: Append the concrete crossing arc, unless both fronts met at one vertex.
//   - Stage 3: Follow backward successor links from to until reaching the target root.
//
// Behavior highlights:
//   - Backward parents are traversed forward in original edge direction.
//   - All traversed pointers are bounded by the vertex count and checked for cycles.
//
// Inputs:
//   - graph: detached effective graph; source,target: original dense terminals.
//   - forward,backward: final private frontiers; join: certified incumbent endpoints.
//
// Returns:
//   - []int: fresh source-to-target original oriented arc indexes.
//
// Errors:
//   - ErrNoPath on missing, cyclic, mismatched, or out-of-range link state.
//
// Determinism:
//   - Exactly the first strictly improving certified join's two frozen chains.
//
// Complexity:
//   - Time O(k+V), Space O(k+V), k<=V-1 for a simple shortest witness.
//
// Notes:
//   - publishSearchPath subsequently verifies edge IDs, direction and original total cost.
//
// AI-Hints:
//   - The backward linkArc[u] points u->successor; do not reverse it on output.
func reconstructBidirectionalPath(
	graph *searchGraph,
	source, target int,
	forward, backward *searchFrontier,
	join searchJoin,
) ([]int, error) {
	if !forward.certifiedSearchVertex(join.from) || !backward.certifiedSearchVertex(join.to) {
		return nil, ErrNoPath
	}

	var arcs []int
	if join.from != source {
		prefix, err := reconstructSearchPath(graph, source, join.from, forward)
		if err != nil {
			return nil, err
		}
		arcs = append(arcs, prefix...)
	}
	if join.arc != noSearchArc {
		if join.arc < 0 || join.arc >= len(graph.arcs) {
			return nil, ErrNoPath
		}
		bridge := graph.arcs[join.arc]
		if bridge.from != join.from || bridge.to != join.to {
			return nil, ErrNoPath
		}
		arcs = append(arcs, join.arc)
	} else if join.from != join.to {
		return nil, ErrNoPath
	}

	seen := make([]bool, len(graph.ids))
	current := join.to
	for current != target {
		if current < 0 || current >= len(graph.ids) || seen[current] {
			return nil, ErrNoPath
		}
		seen[current] = true
		arcIndex := backward.linkArc[current]
		if arcIndex < 0 || arcIndex >= len(graph.arcs) {
			return nil, ErrNoPath
		}
		arc := graph.arcs[arcIndex]
		if arc.from != current {
			return nil, ErrNoPath
		}
		arcs = append(arcs, arcIndex)
		current = arc.to
	}
	return arcs, nil
}

// runBidirectional finds an optimal target witness using two certified search trees.
// Original-cost bidirectional Dijkstra and balanced A* share this exact engine.
// Implementation:
//   - Stage 1: Create two roots over the same validated, detached effective graph.
//   - Stage 2: Clean both heap heads, apply the exact lower bound, and expand one direction.
//   - Stage 3: Reconstruct the best certified join or publish known unreachability.
//
// Behavior highlights:
//   - Only a complete certified incumbent may trigger the stopping condition.
//   - In balanced mode, keys and the incumbent share p(target)-p(source).
//   - The bound minKeyForward+minKeyBackward>=incumbentKey is checked exactly.
//   - With one exhausted queue, its fully explored side certifies completion.
//
// Inputs:
//   - ctx: non-nil cancellation context observed between expansions and arcs.
//   - graph: prepared effective graph; source,target: existing dense vertices.
//   - limit: validated inclusive original-cost cutoff or +Inf.
//   - potentials: nil for bidirectional Dijkstra; frozen normalized balanced values otherwise.
//
// Returns:
//   - []int: detached original arc indexes for a reachable witness, or nil if unreachable.
//   - float64: exact original path cost, or +Inf for known unreachable.
//
// Errors:
//   - ErrInexactArithmetic, ErrDistanceOverflow, ErrSearchKeyOverflow,
//     ErrNoPath and cancellation causes.
//
// Determinism:
//   - Clean heap tuple order; smaller normalized min-priority expands first, equal minima forward.
//   - First strict incumbent improvement wins; original arcs have stable lexical order.
//
// Complexity:
//   - Time O(V + (V+A)log(A+1)) for two lazy heaps and O(A) crossing checks.
//   - Space O(V+A), excluding shared graph snapshot and output, A<=2E.
//
// Notes:
//   - Both frontiers retain their own original-cost distances and non-revisable settled parents.
//   - A source==target witness is finalized only after common upstream validation.
//
// AI-Hints:
//   - Meeting fronts does not justify immediate return; compare cleaned minima to mu.
//   - Do not compare rounded key sums or omit the balanced incumbent offset.
//   - Backward traversal must use original incoming arcs and original edge IDs.
func runBidirectional(
	ctx context.Context,
	graph *searchGraph,
	source, target int,
	limit float64,
	potentials *searchBalancedPotentials,
) ([]int, float64, error) {
	if err := checkSearchContext(ctx); err != nil {
		return nil, 0, err
	}
	if source == target {
		return nil, 0, nil
	}

	search := newBidirectionalSearch(graph, source, target, limit)
	search.balanced = potentials
	for {
		if err := checkSearchContext(ctx); err != nil {
			return nil, 0, err
		}
		forwardMin, forwardOK := search.forward.cleanSearchHead()
		backwardMin, backwardOK := search.backward.cleanSearchHead()

		// One exhausted queue means its side has no remaining unsettled
		// reachable vertices. Every possible complete path has already met
		// a certified vertex or a scanned crossing from that exhausted side.
		if !forwardOK || !backwardOK {
			break
		}
		// Both heads are normalized to their zero-cost roots. The sum of
		// their reduced priorities is a lower bound on every unseen full
		// path. The incumbent is compared in that same shifted scale;
		// original mu alone is NOT a valid balanced termination threshold.
		if search.hasWitness && searchSumAtLeastBound(
			forwardMin.priority, backwardMin.priority, search.incumbentKey,
		) {
			break
		}

		// Both roots start at distance zero. On equal minima the forward
		// direction is deterministic; after that, expand the smaller bound.
		if forwardMin.priority <= backwardMin.priority {
			if err := search.expandSearchForward(ctx); err != nil {
				return nil, 0, err
			}
		} else if err := search.expandSearchBackward(ctx); err != nil {
			return nil, 0, err
		}
	}
	if err := checkSearchContext(ctx); err != nil {
		return nil, 0, err
	}
	if !search.hasWitness {
		return nil, math.Inf(1), nil
	}
	arcs, err := reconstructBidirectionalPath(
		graph, source, target, search.forward, search.backward, search.incumbent,
	)
	if err != nil {
		return nil, 0, err
	}
	return arcs, search.incumbent.distance, nil
}
