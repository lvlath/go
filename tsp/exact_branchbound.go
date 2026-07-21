// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp implements exact Branch-and-Bound search with admissible lower bounds.
//
// Branch-and-Bound enumerates Hamiltonian cycles via deterministic depth-first
// search, admissible pruning, and a governed soft time budget. Both symmetric TSP
// and asymmetric TSP are supported.
//
// Rationale:
//  1. Strict input shape and invariants are enforced before search;
//     the engine prefetches the distance matrix into a dense weightBuffer to remove
//     matrix interface overhead from hot loops.
//  2. Optional seeding of an initial upper bound (UB):
//     Christofides may seed symmetric TSP; a deterministic trivial ring can seed
//     all complete instances. A good UB strengthens pruning but never changes
//     exactness or correctness.
//     UB costs are stored unrounded inside the search state; display rounding
//     happens only when publishing Result.
//  3. Search uses DFS with a degree-1 relaxation lower bound:
//     - For vertices whose outgoing edge is not yet fixed, add minOut[v].
//     - For vertices whose incoming edge is not yet fixed, add minIn[v].
//     - LB_extra = max(sum(minOut), sum(minIn)).
//     - LB = costSoFar + LB_extra. This bound is admissible.
//     The engine prunes whenever LB ≥ UB − eps.
//  4. Branching order tries next vertices from the current last vertex in ascending
//     edge cost with vertex-index tie-breaks. This tightens UB early while remaining
//     fully deterministic.
//  5. Soft time limit checks happen before DFS expansion. When the deadline expires,
//     the engine stops the recursion and returns a valid incumbent as a governed
//     partial result when available.
//
// Complexity:
//   - Worst-case exponential in n.
//   - Per node: O(n) bound plus O(1) state updates.
//   - Memory: O(n) current path + O(n) visited + O(n²) precomputes
//     for min-in/out and neighbor orders.
//
// Governance:
//   - Options.BoundAlgo:
//     NoBound      → disables the lower bound for tests and controlled benchmarks.
//     SimpleBound  → degree-1 relaxation implemented in this file.
//     OneTreeBound → Held-Karp 1-tree lower bound implemented in bound_onetree.go.
package tsp

import (
	"errors"
	"math"
	"sort"
	"time"

	"github.com/lvlath/go/matrix"
)

// bbEngine owns branch-and-bound search state, matrix snapshots, incumbent policy,
// lower-bound buffers, and deterministic neighbor ordering. A dedicated struct keeps
// hot-path dependencies explicit and avoids closure capture surprises.
//
// Implementation:
//   - Stores validated distance access.
//   - Stores current path, visited flags, and incumbent tour.
//   - Stores min-in/min-out arrays for admissible degree-1 lower bounds.
//   - Stores deterministic neighbor order per vertex.
//
// Behavior highlights:
//   - Internal-only.
//   - Not concurrency-safe.
//   - One engine is scoped to one Solve call.
//   - Keeps allocations predictable across recursive search.
//
// Notes:
//   - Public API policy belongs in Options validation, not in this engine.
type bbEngine struct {
	// Configuration / policy
	n         int
	start     int
	symmetric bool
	useBound  bool
	eps       float64

	// Time budget
	useDeadline bool
	deadline    time.Time
	stopped     bool
	// Search telemetry
	nodesExpanded int

	// Graph data (detached dense buffer): weights.at(u,v) is cost u→v.
	weights weightBuffer

	// Precomputes for bound / branching order
	minOut []float64 // per-vertex minimal outgoing edge (excluding self)
	minIn  []float64 // per-vertex minimal incoming edge (excluding self)
	order  [][]int   // for each u: v≠u sorted by w[u→v] (index tiebreak)

	// Current search state
	visited []bool // which vertices are on the current path
	path    []int  // path[0:depth], path[0] == start

	// Current best incumbent (UB)
	bestTour []int
	bestCost float64

	// Whether a feasible Hamiltonian cycle has been found
	foundAny bool
}

// at is a fast accessor into the detached dense weight buffer.
func (e *bbEngine) at(u, v int) float64 { return e.weights.at(u, v) }

// deadlineExpired reports whether the Branch-and-Bound wall-clock budget is exhausted.
// It is intentionally checked before every node expansion so a timeout stops the whole
// DFS tree instead of merely returning from one recursive frame.
//
// Implementation:
//   - Stage 1: Return false when TimeLimit is disabled.
//   - Stage 2: Compare current wall-clock time to the fixed deadline.
//
// Behavior highlights:
//   - Does not mutate engine state.
//   - Used by dfs and finalization.
//
// Inputs:
//   - None; reads engine deadline policy.
//
// Returns:
//   - bool: true when the deadline has passed.
//
// Errors:
//   - None.
//
// Determinism:
//   - Deterministic when TimeLimit==0; wall-clock dependent only when a budget is active.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - The caller sets e.stopped when this returns true.
//
// AI-Hints:
//   - Do not use sparse counters as NodesExpanded.
//   - Do not keep searching sibling branches after this returns true.
func (e *bbEngine) deadlineExpired() bool {
	return e.useDeadline && time.Now().After(e.deadline)
}

// initWeights attaches a validated complete weight buffer to the search engine.
// Branch-and-Bound consumes final solver input, so missing +Inf edges must already
// be rejected by copyCompleteWeights before search starts.
//
// Implementation:
//   - Stage 1: Verify that the buffer order matches the engine order.
//   - Stage 2: Store the detached weight buffer for O(1) hot-loop access.
//
// Behavior highlights:
//   - Does not copy weights again.
//   - Does not mutate the source matrix.
//   - Keeps B&B sentinel classification aligned with all final solver kernels.
//
// Inputs:
//   - weights: detached complete row-major TSP weights.
//
// Returns:
//   - error: nil when the buffer matches engine dimensions.
//
// Errors:
//   - ErrDimensionMismatch when weights.n does not match e.n.
//
// Determinism:
//   - Pure shape check and assignment.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Build weights with copyCompleteWeights, not copyClosureReadyWeights.
//
// AI-Hints:
//   - Do not allow +Inf into B&B final search state.
//   - Do not reintroduce a custom matrix.At prefetch loop here.
func (e *bbEngine) initWeights(weights weightBuffer) error {
	if weights.n != e.n {
		return ErrDimensionMismatch
	}

	e.weights = weights

	return nil
}

// tourCost computes an unrounded tour cost from the engine weight buffer.
// It is used for incumbent bounds; display stabilization happens only when
// publishing Result.
//
// Implementation:
//   - Stage 1: Validate closed-tour length and fixed start.
//   - Stage 2: Scan all arcs in tour order.
//   - Stage 3: Accumulate raw float64 cost without round1e9.
//
// Behavior highlights:
//   - Does not mutate the tour.
//   - Rejects malformed tours before they become pruning bounds.
//   - Reads from weightBuffer, not matrix.Matrix.
//
// Inputs:
//   - tour: closed Hamiltonian cycle candidate.
//
// Returns:
//   - float64: raw unrounded cycle cost.
//   - error: nil on valid finite tour.
//
// Errors:
//   - ErrInvalidTour for malformed tour, out-of-range vertices, or missing closure.
//   - ErrIncompleteGraph for impossible +Inf arc, defensive only after copyCompleteWeights.
//
// Determinism:
//   - Fixed left-to-right arc scan.
//
// Complexity:
//   - Time O(n), Space O(1).
//
// Notes:
//   - This helper intentionally does not call TourCost because TourCost rounds for public metadata.
//
// AI-Hints:
//   - Do not use rounded public result costs as Branch-and-Bound UB.
func (e *bbEngine) tourCost(tour []int) (float64, error) {
	if err := ValidateTour(tour, e.n, e.start); err != nil {
		return 0, err
	}

	var (
		index int
		from  int
		to    int
		cost  float64
		sum   float64
	)
	for index = 0; index < e.n; index++ {
		from = tour[index]
		to = tour[index+1]
		cost = e.at(from, to)

		if math.IsInf(cost, 0) {
			return 0, ErrIncompleteGraph
		}

		sum += cost
	}

	return sum, nil
}

// precomputeMinima computes minimal incoming and outgoing edge costs for every vertex.
// These arrays feed the degree-1 relaxation lower bound used by DFS pruning.
//
// Implementation:
//   - Stage 1: Allocate minOut and minIn arrays.
//   - Stage 2: For every vertex, scan all non-self outgoing and incoming arcs.
//   - Stage 3: Reject the instance when any vertex has no finite in/out candidate.
//
// Behavior highlights:
//   - Uses the detached weightBuffer only.
//   - Excludes self-loops.
//   - Stores raw costs without rounding.
//
// Inputs:
//   - None; reads e.n and e.weights.
//
// Returns:
//   - error: nil when every vertex has at least one finite in/out candidate.
//
// Errors:
//   - ErrIncompleteGraph when a vertex cannot enter or leave any Hamiltonian cycle.
//
// Determinism:
//   - Fixed vertex and neighbor scan order.
//
// Complexity:
//   - Time O(n^2), Space O(n).
//
// AI-Hints:
//   - Do not include self-loops in minima; they are not valid Hamiltonian edges.
func (e *bbEngine) precomputeMinima() error {
	var (
		inf    = math.Inf(1)
		v, u   int
		mo, mi float64
		cOut   float64
		cIn    float64
	)
	e.minOut = make([]float64, e.n)
	e.minIn = make([]float64, e.n)
	for v = 0; v < e.n; v++ {
		mo, mi = inf, inf
		for u = 0; u < e.n; u++ {
			if u == v {
				continue
			}
			cOut = e.at(v, u)
			if cOut < mo {
				mo = cOut
			}
			cIn = e.at(u, v)
			if cIn < mi {
				mi = cIn
			}
		}
		e.minOut[v] = mo
		e.minIn[v] = mi
		if math.IsInf(mo, 0) || math.IsInf(mi, 0) {
			return ErrIncompleteGraph
		}
	}

	return nil
}

// neighborOrder implements sort.Interface for one row of candidate neighbors ordered
// by edge weight and then by vertex ID. Stable deterministic ordering improves pruning
// reproducibility and makes tests independent of map or heap iteration.
//
// Implementation:
//   - Len reports candidate count.
//   - Less compares edge cost, then vertex ID.
//   - Swap exchanges candidate entries in place.
//
// Behavior highlights:
//   - Internal sort adapter.
//   - Does not inspect global solver state after construction.
//   - Keeps equal-weight behavior deterministic.
//
// Notes:
//   - The backing slices are intentionally mutated by sort.Sort.
type neighborOrder struct {
	u   int
	row []int
	e   *bbEngine
}

// Len returns the number of neighbor candidates.
//
// Complexity:
//   - Time O(1), Space O(1).
func (no neighborOrder) Len() int { return len(no.row) }

// Less reports whether candidate i should be explored before candidate j.
// It orders by lower edge weight first and vertex ID second for deterministic ties.
//
// Complexity:
//   - Time O(1), Space O(1).
func (no neighborOrder) Less(i, j int) bool {
	vi, vj := no.row[i], no.row[j]
	wi, wj := no.e.at(no.u, vi), no.e.at(no.u, vj)
	if wi == wj {
		return vi < vj
	}

	return wi < wj
}

// Swap exchanges two neighbor candidates in place.
//
// Complexity:
//   - Time O(1), Space O(1).
func (no *neighborOrder) Swap(i, j int) { no.row[i], no.row[j] = no.row[j], no.row[i] }

// buildNeighborOrder precomputes deterministic DFS branch order for every vertex.
// Each row contains all non-self vertices sorted by ascending edge cost and then by
// vertex index to make equal-cost runs reproducible.
//
// Implementation:
//   - Stage 1: Allocate one neighbor row per source vertex.
//   - Stage 2: Fill each row with all v != u.
//   - Stage 3: Sort through neighborOrder using weight then index tie-break.
//
// Behavior highlights:
//   - Does not mutate weights.
//   - Improves time-to-tight-incumbent without changing exactness.
//
// Inputs:
//   - None; reads e.n and e.weights.
//
// Returns:
//   - None.
//
// Errors:
//   - None.
//
// Determinism:
//   - Stable policy through cost comparison and vertex-index tie-break.
//
// Complexity:
//   - Time O(n^2 log n), Space O(n^2).
//
// AI-Hints:
//   - Do not replace this with map-based ordering.
func (e *bbEngine) buildNeighborOrder() {
	var u, v int
	e.order = make([][]int, e.n)
	for u = 0; u < e.n; u++ {
		row := make([]int, 0, e.n-1)
		for v = 0; v < e.n; v++ {
			if v != u {
				row = append(row, v)
			}
		}
		no := neighborOrder{u: u, row: row, e: e}
		sort.Sort(&no)
		e.order[u] = no.row
	}
}

// recordUB commits a new incumbent upper bound without rounding its cost.
// The bound participates in pruning, so cost must remain the raw accumulated
// value; round1e9 is applied only when publishing Result.
//
// Implementation:
//   - Stage 1: Copy the candidate tour into the engine-owned incumbent buffer.
//   - Stage 2: Store raw search cost as the active upper bound.
//   - Stage 3: Mark that a feasible Hamiltonian cycle exists.
//
// Behavior highlights:
//   - Does not allocate beyond the destination slice already owned by the engine.
//   - Does not round bestCost.
//   - Does not validate the tour; callers must validate or compute cost before committing.
//
// Inputs:
//   - tour: validated closed Hamiltonian cycle.
//   - cost: raw unrounded tour cost.
//
// Returns:
//   - None.
//
// Errors:
//   - None.
//
// Determinism:
//   - Copies the deterministic candidate order exactly.
//
// Complexity:
//   - Time O(n), Space O(1).
//
// Notes:
//   - This is a pruning-state update, not result publication.
//
// AI-Hints:
//   - Do not call round1e9 here; rounded upper bounds can change pruning decisions.
//   - Do not store caller-owned tour slices directly.
func (e *bbEngine) recordUB(tour []int, cost float64) {
	copy(e.bestTour, tour)
	e.bestCost = cost
	e.foundAny = true
}

// tryRecordSeedTour validates a candidate seed tour and records it as incumbent.
// It is a small guard helper used by all Branch-and-Bound seeding paths so raw
// upper-bound semantics stay identical.
//
// Implementation:
//   - Stage 1: Compute raw engine cost through e.tourCost.
//   - Stage 2: Record the tour only when it is feasible under the engine weight buffer.
//
// Behavior highlights:
//   - Non-fatal: invalid seed candidates are ignored.
//   - Preserves raw unrounded cost.
//   - Does not mutate the candidate tour.
//
// Inputs:
//   - tour: candidate closed Hamiltonian cycle.
//
// Returns:
//   - bool: true when the seed became the active incumbent.
//
// Errors:
//   - None. Feasibility failures are intentionally swallowed because seeding is optional.
//
// Determinism:
//   - Deterministic for a fixed tour and weight buffer.
//
// Complexity:
//   - Time O(n), Space O(1).
//
// Notes:
//   - Search correctness never depends on this helper succeeding.
//
// AI-Hints:
//   - Use this helper instead of duplicating e.tourCost + e.recordUB pairs.
func (e *bbEngine) tryRecordSeedTour(tour []int) bool {
	cost, err := e.tourCost(tour)
	if err != nil {
		return false
	}

	e.recordUB(tour, cost)

	return true
}

// trySeedWithChristofides attempts a symmetric Christofides incumbent.
// It is exact-search acceleration only: failure to seed never changes correctness,
// exactness, or the final error returned by Branch-and-Bound.
//
// Implementation:
//   - Stage 1: Reject asymmetric engines because Christofides requires symmetric input.
//   - Stage 2: Force the seed policy to Christofides.
//   - Stage 3: Run the result-native Christofides kernel.
//   - Stage 4: Record the returned feasible tour as raw B&B upper bound.
//
// Behavior highlights:
//   - Non-fatal.
//   - Does not call public wrappers.
//   - Does not use rounded public cost as UB.
//
// Inputs:
//   - dist: same final complete matrix consumed by Branch-and-Bound.
//   - opts: finalized Branch-and-Bound options.
//
// Returns:
//   - bool: true when a Christofides seed was recorded.
//
// Errors:
//   - None. Seed errors are intentionally ignored.
//
// Determinism:
//   - Deterministic because Christofides and tour-cost validation are deterministic.
//
// Complexity:
//   - Time O(n^2) plus matching complexity.
//   - Space follows Christofides matching and Eulerian stages.
//
// Notes:
//   - This helper is an incumbent initializer, not a solver fallback.
//
// AI-Hints:
//   - Keep seedOptions.Algo=Christofides explicit.
//   - Do not call greedy matching unless opts.MatchingAlgo explicitly selects it.
func (e *bbEngine) trySeedWithChristofides(dist matrix.Matrix, opts Options) bool {
	if !e.symmetric {
		return false
	}

	seedOptions := opts
	seedOptions.Algo = Christofides
	seedOptions.Symmetric = true
	seedOptions.RunMetricClosure = false

	result, err := christofides(dist, seedOptions)
	if err != nil || result == nil {
		return false
	}

	return e.tryRecordSeedTour(result.Tour)
}

// trySeedWithTrivialRing records the canonical ring and optionally improves it.
// This path works for symmetric and asymmetric complete matrices because 2-opt uses
// symmetric reversal or ATSP 2-opt* according to opts.Symmetric.
//
// Implementation:
//   - Stage 1: Build the deterministic closed ring from e.start.
//   - Stage 2: Record the ring if feasible.
//   - Stage 3: Optionally run twoOptKernel and record its feasible result.
//   - Stage 4: Accept timeout-local results as seeds when they contain a valid tour.
//
// Behavior highlights:
//   - Non-fatal.
//   - Deterministic when local search uses deterministic policy.
//   - Uses raw engine tour cost for UB even when the local-search result cost is rounded.
//
// Inputs:
//   - dist: same final complete matrix consumed by Branch-and-Bound.
//   - opts: finalized Branch-and-Bound options.
//
// Returns:
//   - None.
//
// Errors:
//   - None. Seed construction/improvement failures are intentionally ignored.
//
// Determinism:
//   - Trivial ring order is fixed.
//   - 2-opt scan order is fixed unless future policy explicitly changes it.
//
// Complexity:
//   - Ring seed O(n).
//   - Optional 2-opt seed O(iter*n^2).
//
// Notes:
//   - The base ring remains useful even when 2-opt fails or times out.
//
// AI-Hints:
//   - Do not treat ErrTimeLimit from twoOptKernel as fatal when local.hasTour() is true.
//   - Do not use local.cost as the B&B UB; recompute raw cost through e.tourCost.
func (e *bbEngine) trySeedWithTrivialRing(dist matrix.Matrix, opts Options) {
	base, err := trivialRing(e.n, e.start)
	if err != nil {
		return
	}

	e.tryRecordSeedTour(base)

	if !opts.EnableLocalSearch || e.n < 4 {
		return
	}

	local, improveErr := twoOptKernel(dist, base, opts)
	if improveErr != nil && !errors.Is(improveErr, ErrTimeLimit) {
		return
	}
	if !local.hasTour() {
		return
	}

	e.tryRecordSeedTour(local.tour)
}

// seedUB initializes the incumbent upper bound used by Branch-and-Bound pruning.
// A strong incumbent shrinks the DFS tree, but every seed path is optional and
// cannot change the mathematical correctness of exact search.
//
// Implementation:
//   - Stage 1: Reset incumbent state to no feasible tour.
//   - Stage 2: Try a symmetric Christofides seed when mathematically allowed.
//   - Stage 3: Fall back to the deterministic trivial ring.
//   - Stage 4: Optionally improve the ring through result-preserving 2-opt.
//
// Behavior highlights:
//   - Non-fatal by design.
//   - Stores raw unrounded costs for pruning.
//   - Keeps result publication separate from incumbent management.
//
// Inputs:
//   - dist: final complete matrix consumed by Branch-and-Bound.
//   - opts: finalized solver policy.
//
// Returns:
//   - None.
//
// Errors:
//   - None. Seeding failures are intentionally ignored.
//
// Determinism:
//   - Deterministic for TimeLimit==0 and deterministic local-search policy.
//
// Complexity:
//   - Christofides seed: O(n^2) plus matching complexity.
//   - Trivial ring: O(n).
//   - Optional 2-opt seed: O(iter*n^2).
//
// Notes:
//   - Exact search remains correct even if no seed is recorded.
//
// AI-Hints:
//   - Do not let seed errors escape from this method.
//   - Do not round incumbent costs before pruning completes.
func (e *bbEngine) seedUB(dist matrix.Matrix, opts Options) {
	e.bestCost = math.Inf(1)
	e.bestTour = make([]int, e.n+1)
	e.foundAny = false

	if e.trySeedWithChristofides(dist, opts) {
		return
	}

	e.trySeedWithTrivialRing(dist, opts)
}

// lowerBound computes the admissible degree-1 relaxation for the current partial path.
// Any Hamiltonian cycle must give every vertex one outgoing and one incoming edge;
// for still-unfixed endpoints, the completion cost is at least the cheapest available
// outgoing/incoming edge contribution.
//
// Implementation:
//   - Stage 1: Start from costSoFar.
//   - Stage 2: Sum minOut for vertices whose outgoing edge is not fixed.
//   - Stage 3: Sum minIn for vertices whose incoming edge is not fixed.
//   - Stage 4: Add max(outRelaxation,inRelaxation) as the remaining-cost lower bound.
//
// Behavior highlights:
//   - Admissible for TSP and ATSP.
//   - Does not mutate search state.
//   - Cheap enough for recursive branch pruning.
//   - Deterministic and allocation-free.
//
// Inputs:
//   - costSoFar: exact cost of the current partial path.
//   - last: last vertex in the partial path.
//
// Returns:
//   - float64: lower bound on any completion of the current branch.
//
// Errors:
//   - None. Invalid precomputed min arrays should already have been caught by validation.
//
// Determinism:
//   - Fixed vertex scans.
//
// Complexity:
//   - Time O(n), Space O(1).
//
// Notes:
//   - Bound strength is modest but safe.
//   - Held-Karp/1-tree bounds can be layered later for symmetric instances.
//
// AI-Hints:
//   - Never overestimate here; branch-and-bound correctness depends on admissibility.
func (e *bbEngine) lowerBound(costSoFar float64, last int) float64 {
	if !e.useBound {
		return costSoFar // NoBound policy (for testing/benchmarking).
	}
	var sumOut, sumIn float64
	sumOut, sumIn = 0, 0

	// Outgoing is fixed for all visited vertices except 'last';
	// incoming is fixed for all visited vertices except 'start'.
	var v int
	for v = 0; v < e.n; v++ {
		if e.visited[v] {
			if v == last {
				sumOut += e.minOut[v]
			}
			if v == e.start {
				sumIn += e.minIn[v]
			}
		} else {
			sumOut += e.minOut[v]
			sumIn += e.minIn[v]
		}
	}
	extra := sumOut
	if sumIn > extra {
		extra = sumIn
	}

	return costSoFar + extra
}

// commit writes the final start-closure and records a new incumbent upper bound.
// The total cost is raw search cost and must not be rounded before pruning completes.
//
// Implementation:
//   - Stage 1: Close the current path with the fixed start vertex.
//   - Stage 2: Copy the full path into bestTour.
//   - Stage 3: Store raw bestCost and mark that an incumbent exists.
//
// Behavior highlights:
//   - Does not allocate.
//   - Does not round bestCost.
//
// Inputs:
//   - total: raw complete-cycle cost.
//
// Returns:
//   - None.
//
// Errors:
//   - None. Caller validates edge feasibility before commit.
//
// Determinism:
//   - Copies the current deterministic DFS path.
//
// Complexity:
//   - Time O(n), Space O(1).
//
// Notes:
//   - Publication-time rounding belongs to result().
//
// AI-Hints:
//   - Do not call round1e9 here; rounded UB can change pruning.
func (e *bbEngine) commit(total float64) {
	e.path[e.n] = e.start
	copy(e.bestTour, e.path)
	e.bestCost = total
	e.foundAny = true
}

// dfs explores Hamiltonian path completions with deterministic Branch-and-Bound.
// It mutates only engine-owned path and visited state and records better incumbents
// through commit when a full cycle improves the upper bound.
//
// Implementation:
//   - Stage 1: Stop immediately on global timeout state.
//   - Stage 2: Check deadline and count the node expansion.
//   - Stage 3: Prune when the admissible lower bound cannot beat the incumbent.
//   - Stage 4: Close a full-depth path into a Hamiltonian cycle.
//   - Stage 5: Recurse over precomputed neighbor order with backtracking.
//
// Behavior highlights:
//   - Exact DFS enumeration with pruning.
//   - Missing closing arcs are rejected defensively.
//   - Backtracking restores visited state before returning.
//
// Inputs:
//   - last: current path tail.
//   - depth: number of vertices currently in path.
//   - costSoFar: raw unrounded partial path cost.
//
// Returns:
//   - None.
//
// Errors:
//   - None. Search state stores timeout and incumbents internally.
//
// Determinism:
//   - Branch order is precomputed and stable.
//
// Complexity:
//   - Worst-case O(n!) nodes.
//   - Per node O(n) lower-bound work plus branching overhead.
//
// AI-Hints:
//   - Do not increment NodesExpanded for pruned children that were never entered.
//   - Do not continue sibling recursion after e.stopped becomes true.
func (e *bbEngine) dfs(last int, depth int, costSoFar float64) {
	if e.stopped {
		return
	}
	if e.deadlineExpired() {
		e.stopped = true
		return
	}
	e.nodesExpanded++

	// Prune by lower bound.
	if lb := e.lowerBound(costSoFar, last); lb >= e.bestCost-e.eps {
		return
	}

	// If we used all vertices, close the cycle at start.
	if depth == e.n {
		c := e.at(last, e.start)
		if math.IsInf(c, 0) {
			return // missing closing edge
		}
		total := costSoFar + c
		if total < e.bestCost-e.eps {
			e.commit(total)
		}

		return
	}

	// Branch: iterate neighbors of 'last' in the precomputed order.
	var (
		v int
		c float64
	)
	for _, v = range e.order[last] {
		if e.stopped {
			return
		}
		if e.visited[v] {
			continue
		}
		c = e.at(last, v)
		if math.IsInf(c, 0) {
			continue
		}
		e.visited[v] = true
		e.path[depth] = v
		e.dfs(v, depth+1, costSoFar+c)
		e.visited[v] = false

		if e.stopped {
			return
		}
	}
}

// result publishes the current Branch-and-Bound incumbent as a canonical Result.
// Implementation:
//   - Stage 1: Detach the best tour.
//   - Stage 2: Stabilize cost.
//   - Stage 3: Attach exact/optimal/timeout/search metadata.
//
// Behavior highlights:
//   - Requires e.bestTour to contain a valid incumbent.
//   - Does not mutate engine state.
//
// Returns:
//   - *Result: detached result snapshot.
//
// Complexity:
//   - Time O(n), Space O(n).
//
// AI-Hints:
//   - Do not expose engine pointers or mutable slices through Result.
func (e *bbEngine) result(optimal bool, timedOut bool) *Result {
	return &Result{
		Tour:          append([]int(nil), e.bestTour...),
		Cost:          round1e9(e.bestCost),
		Algorithm:     BranchAndBound,
		Exact:         true,
		Optimal:       optimal,
		TimedOut:      timedOut,
		NodesExpanded: e.nodesExpanded,
		Symmetric:     e.symmetric,
	}
}

// branchAndBound executes exact Branch-and-Bound and can publish partial timeout results.
// Implementation:
//   - Stage 1: Validate options, matrix, and start vertex.
//   - Stage 2: Initialize dense engine buffers and deterministic branching order.
//   - Stage 3: Seed an incumbent upper bound.
//   - Stage 4: Run DFS with admissible pruning.
//   - Stage 5: Publish optimal, incomplete, or partial-timeout result.
//
// Behavior highlights:
//   - Exact when completed without timeout.
//   - On timeout, returns a partial Result only if a feasible incumbent exists.
//   - Canonical callers preserve timeout and search telemetry metadata.
//
// Inputs:
//   - dist: final complete TSP/ATSP distance matrix.
//   - opts: finalized solver policy.
//
// Returns:
//   - *Result: optimal or partial result.
//   - error: nil, ErrTimeLimit, ErrIncompleteGraph, or validation sentinel.
//
// Errors:
//   - ErrInvalidOptions / ErrUnsupportedAlgorithm.
//   - ErrNilDistanceMatrix / ErrNonSquare / ErrDimensionMismatch.
//   - ErrNaNInf / ErrNegativeWeight / ErrIncompleteGraph.
//   - ErrStartOutOfRange.
//   - ErrTimeLimit with non-nil result when an incumbent exists.
//
// Determinism:
//   - Branch order is sorted by edge weight then vertex index.
//   - Deadline checks are sparse but do not change correctness, only whether partial result is returned.
//
// Complexity:
//   - Worst-case exponential time.
//   - Space O(n^2) precompute + O(n) search state.
//
// AI-Hints:
//   - Do not clear a valid incumbent on timeout.
//   - Do not mark timed-out results as Optimal.
func branchAndBound(dist matrix.Matrix, opts Options) (*Result, error) {
	if err := validateOptionsStandalone(opts); err != nil {
		return nil, err
	}

	weights, err := copyCompleteWeights(dist, opts.Symmetric)
	if err != nil {
		return nil, err
	}
	n := weights.n
	if err = validateStartVertex(n, opts.StartVertex); err != nil {
		return nil, err
	}

	// Engine initialization (no anonymous closures).
	var engine bbEngine
	engine.n = n
	engine.start = opts.StartVertex
	engine.symmetric = opts.Symmetric
	engine.eps = opts.Eps
	engine.useBound = opts.BoundAlgo != NoBound

	// Deadline setup.
	if opts.TimeLimit > 0 {
		engine.useDeadline = true
		engine.deadline = time.Now().Add(opts.TimeLimit)
	}

	// Attach already validated final solver weights.
	if err = engine.initWeights(weights); err != nil {
		return nil, err
	}
	if err = engine.precomputeMinima(); err != nil {
		return nil, err
	}
	engine.buildNeighborOrder()

	// Search state.
	engine.visited = make([]bool, n)
	engine.path = make([]int, n+1)
	engine.path[0] = engine.start
	engine.visited[engine.start] = true

	if engine.deadlineExpired() {
		engine.stopped = true
		return nil, ErrTimeLimit
	}

	// Optional UB seeding (greatly helps pruning, correctness unaffected).
	engine.seedUB(dist, opts)

	if engine.deadlineExpired() {
		engine.stopped = true
		if engine.foundAny && !math.IsInf(engine.bestCost, 0) {
			_ = CanonicalizeOrientationInPlace(engine.bestTour)
			if err = ValidateTour(engine.bestTour, n, engine.start); err != nil {
				return nil, err
			}
			return engine.result(false, true), ErrTimeLimit
		}
		return nil, ErrTimeLimit
	}

	// Optional: root-only 1-tree (Held–Karp) lower bound to tighten pruning
	// before entering DFS. Safe for symmetric instances; correctness unaffected.
	if opts.BoundAlgo == OneTreeBound && opts.Symmetric {
		cfg := DefaultOneTreeConfig()
		// If we already have a finite incumbent, pass it to the step schedule.
		if !math.IsInf(engine.bestCost, 0) && engine.bestCost > 0 {
			cfg.UB = engine.bestCost
		}

		if lowerBound, _, boundErr := OneTreeLowerBound(dist, engine.start, true, cfg); boundErr == nil {
			// If LB >= UB−eps at the root, the incumbent is optimal; return it.
			if !math.IsInf(engine.bestCost, 0) && lowerBound >= engine.bestCost-engine.eps {
				_ = CanonicalizeOrientationInPlace(engine.bestTour)
				if err = ValidateTour(engine.bestTour, n, engine.start); err != nil {
					return nil, err
				}

				return engine.result(true, false), nil
			}
		}
	}

	// Run DFS.
	engine.dfs(engine.start, 1, 0)

	// Finalization.
	if engine.stopped || engine.deadlineExpired() {
		engine.stopped = true
		if engine.foundAny && !math.IsInf(engine.bestCost, 0) {
			_ = CanonicalizeOrientationInPlace(engine.bestTour)
			if err = ValidateTour(engine.bestTour, n, engine.start); err != nil {
				return nil, err
			}

			return engine.result(false, true), ErrTimeLimit
		}

		return nil, ErrTimeLimit
	}

	if !engine.foundAny && math.IsInf(engine.bestCost, 0) {
		return nil, ErrIncompleteGraph
	}

	_ = CanonicalizeOrientationInPlace(engine.bestTour)
	if err = ValidateTour(engine.bestTour, n, engine.start); err != nil {
		return nil, err
	}

	return engine.result(true, false), nil
}
