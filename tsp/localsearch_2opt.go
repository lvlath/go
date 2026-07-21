// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp - 2-opt local search engine (symmetric 2-opt and asymmetric 2-opt*).
//
// TwoOpt performs deterministic first-improvement 2-opt on a closed tour.
//   - Symmetric case (opts.Symmetric == true): classic 2-opt reverses segment [i..k].
//     Δ = w(a,c) + w(b,d) − w(a,b) − w(c,d), with a=T[i−1], b=T[i], c=T[k], d=T[k+1].
//   - Asymmetric case (opts.Symmetric == false): 2-opt* “tail swap”, no reversal.
//     Replace arcs (a→b),(c→d) with (a→d),(c→b); Δ = w(a,d) + w(c,b) − w(a,b) − w(c,d).
//
// Design:
//   - Deterministic scanning order; no RNG usage (seed reserved for future shuffles).
//   - Strict sentinel errors only (see types.go). No fmt.Errorf in hot paths.
//   - Defensive but allocation-conscious: O(1) per-check; O(n) only on accepted move.
//   - Soft time budget via periodic deadline checks.
//   - Cost stabilized to 1e−9 via round1e9.
//
// Contracts:
//   - dist is a complete n×n final solver matrix validated through copyCompleteWeights.
//   - tour is a *closed* Hamiltonian cycle (len==n+1, tour[0]==tour[n]==opts.StartVertex).
//   - For asymmetric instances, the solver uses 2-opt* (does not reverse segments).
//
// Complexity:
//   - One pass: O(n²) candidate checks; first-improvement restarts after each accepted move.
//   - Each accepted move costs O(1) (symmetric) or O(n) (asymmetric, rebuild successor list).
//   - Overall: O(iter*n²) time typical; O(n) extra space on improvements only.
package tsp

import (
	"errors"
	"math"
	"time"

	"github.com/lvlath/go/matrix"
)

// twoOptSearch runs 2-opt local search and publishes canonical result metadata.
//
// Implementation:
//   - Stage 1: Force the internal algorithm identity to TwoOptOnly.
//   - Stage 2: Run twoOptKernel to validate input and improve the current tour.
//   - Stage 3: Publish localSearchResult as Result without IDs or metric-closure metadata.
//   - Stage 4: Preserve ErrTimeLimit when a valid current tour exists.
//
// Behavior highlights:
//   - Never claims exactness or global optimality.
//   - Returns a non-nil result with ErrTimeLimit when the kernel has a valid current tour.
//   - Does not mutate initTour.
//
// Inputs:
//   - dist: complete final solver matrix.
//   - initTour: closed Hamiltonian cycle.
//   - opts: local-search policy.
//
// Returns:
//   - *Result: improved or timeout-current tour result.
//   - error: nil or sentinel-classified failure.
//
// Errors:
//   - ErrInvalidOptions.
//   - ErrNilTour, ErrInvalidTour.
//   - Matrix validation sentinels.
//   - ErrTimeLimit with non-nil result when timeout happens after a valid current tour exists.
//
// Determinism:
//   - Fixed candidate scan order unless the kernel explicitly supports shuffled neighborhoods.
//
// Complexity:
//   - Time O(iterations*n^2), Space O(n).
//
// Notes:
//   - IDs and metric closure metadata are attached only by SolveMatrix.
//
// AI-Hints:
//   - Do not publish Exact=true or Optimal=true.
//   - Do not drop timeout results when local.hasTour() is true.
func twoOptSearch(dist matrix.Matrix, initTour []int, opts Options) (*Result, error) {
	solverOptions := opts
	solverOptions.Algo = TwoOptOnly

	local, err := twoOptKernel(dist, initTour, solverOptions)
	if err != nil {
		if errors.Is(err, ErrTimeLimit) && local.hasTour() {
			result := publishLocalSearchResult(local, nil, solverOptions, TwoOptOnly, false)
			return result, ErrTimeLimit
		}

		return nil, err
	}
	if !local.hasTour() {
		return nil, ErrInvalidTour
	}

	return publishLocalSearchResult(local, nil, solverOptions, TwoOptOnly, false), nil
}

// twoOptKernel runs deterministic first-improvement 2-opt / 2-opt* and returns
// structured local-search metadata for canonical facades.
//
// Implementation:
//   - Stage 1: Validate options, tour shape, complete weights, and tour invariants.
//   - Stage 2: Copy the input tour and compute baseline cost.
//   - Stage 3: Scan candidate pairs in deterministic order and apply first improvements.
//   - Stage 4: Publish a valid localSearchResult on success, move cap, or timeout.
//
// Behavior highlights:
//   - Symmetric mode reverses one segment.
//   - ATSP mode uses orientation-preserving 2-opt* tail rewiring.
//   - Timeout with a current valid tour returns localSearchResult plus ErrTimeLimit.
//
// Inputs:
//   - dist: complete final distance matrix.
//   - initTour: closed Hamiltonian tour starting and ending at opts.StartVertex.
//   - opts: validated local-search policy.
//
// Returns:
//   - localSearchResult: finalized local-search result when a current tour exists.
//   - error: nil on success or sentinel-classified failure.
//
// Errors:
//   - ErrInvalidOptions, ErrNilTour, ErrInvalidTour, ErrDimensionMismatch.
//   - ErrNilDistanceMatrix, ErrNonSquare, ErrNaNInf, ErrNegativeWeight, ErrIncompleteGraph.
//   - ErrTimeLimit with non-empty localSearchResult when a valid current tour exists.
//
// Determinism:
//   - Fixed increasing i,k scan order.
//   - First-improvement restarts scanning after every accepted move.
//
// Complexity:
//   - Time O(iter*n^2), Space O(n).
//   - Symmetric accepted moves reverse O(k-i); ATSP accepted moves rebuild O(n).
//
// Notes:
//   - iterations counts accepted moves, not candidate evaluations.
//   - TimeLimit==0 disables deadline checks.
//
// AI-Hints:
//   - Do not return nil tour on ErrTimeLimit after a valid current tour exists.
//   - Do not let +Inf into final local-search kernels.
//   - High complexity is required to maintain the ultra-tight inner loop execution speed.
//   - Function calls inside this kernel would severely degrade TSP optimization performance.
//
// nolint:gocyclo
func twoOptKernel(dist matrix.Matrix, initTour []int, opts Options) (localSearchResult, error) {
	if err := validateOptionsStandalone(opts); err != nil {
		return localSearchResult{}, err
	}
	if initTour == nil {
		return localSearchResult{}, ErrNilTour
	}
	if len(initTour) < 2 {
		return localSearchResult{}, ErrInvalidTour
	}

	n := len(initTour) - 1
	if n < 2 { // a closed cycle needs at least two distinct vertices
		return localSearchResult{}, ErrInvalidTour
	}
	weights, err := copyCompleteWeights(dist, opts.Symmetric)
	if err != nil {
		return localSearchResult{}, err
	}
	if weights.n != n {
		return localSearchResult{}, ErrDimensionMismatch
	}

	if err = ValidateTour(initTour, n, opts.StartVertex); err != nil {
		return localSearchResult{}, err
	}

	// Current working tour (copy to keep the input immutable).
	cur := make([]int, n+1)
	copy(cur, initTour)

	// Baseline cost with strict checks (rejects +Inf/NaN on existing edges).
	cost, err := TourCost(dist, cur)
	if err != nil {
		return localSearchResult{}, err
	}

	eps := opts.Eps
	maxIters := opts.TwoOptMaxIters // 0 ⇒ unlimited (until local optimum)

	// Soft deadline (checked sparsely to keep overhead negligible).
	var (
		useDeadline     bool      // whether we enforce a wall-clock time budget
		deadline        time.Time // absolute deadline if enabled
		candidateChecks int       // candidate counter to throttle checks
	)
	if opts.TimeLimit > 0 {
		useDeadline = true
		deadline = localSearchNow().Add(opts.TimeLimit)
	}
	// Check every 2048 candidate events. This preserves throughput in tight loops.
	checkDeadline := func() bool {
		candidateChecks++
		if (candidateChecks & twoOptDeadlineCheckMask) != 0 {
			return false
		}

		return localSearchDeadlineExpired(useDeadline, deadline)
	}

	// Main first-improvement loop: restart scan after every accepted move.
	accepted := 0
	for {
		improved := false // toggled to true exactly when a move is applied

		// Variables reused in inner loops to avoid re-declarations in hot path.
		var (
			a, b, c, d    int     // boundary endpoints around (i,k)
			delta         float64 // candidate improvement (negative is good)
			wab, wad, wcb float64 // baseline / new arcs (ATSP)
			wcd, wac, wbd float64 // baseline / new arcs (symmetric)
			i, k          int     // candidate cut indices, 1 ≤ i < k ≤ n−1
		)

		// Scan all candidate pairs (i,k) with 1 ≤ i < k ≤ n−1.
		// In 2-opt* (ATSP) we skip (i==1 && k==n−1) to avoid creating start→start directly.
		for i = 1; i <= n-2; i++ {
			for k = i + 1; k <= n-1; k++ {
				if checkDeadline() {
					local, finishErr := finishLocalSearchCurrent(cur, cost, accepted, true, n, opts.StartVertex)
					if finishErr != nil {
						return localSearchResult{}, finishErr
					}
					return local, ErrTimeLimit
				}
				if !opts.Symmetric && i == 1 && k == n-1 {
					// Would connect start→start in the middle of the tour; skip.
					continue
				}

				// a=T[i−1], b=T[i], c=T[k], d=T[k+1]
				a = cur[i-1]
				b = cur[i]
				c = cur[k]
				d = cur[k+1]

				if opts.Symmetric {
					// Classic 2-opt reversal on [i..k].
					wab = weights.at(a, b)
					wcd = weights.at(c, d)
					wac = weights.at(a, c)
					wbd = weights.at(b, d)

					// If the new edges do not exist, reject this candidate.
					if math.IsInf(wac, 0) || math.IsInf(wbd, 0) {
						continue
					}
					// Δ = new − old; accept strictly improving (beyond tolerance).
					delta = (wac + wbd) - (wab + wcd)
					if delta < -eps {
						// Apply by in-place reversal of segment [i..k] (O(k−i+1)).
						if err = reverseArcInPlace(cur, i, k); err != nil {
							return localSearchResult{}, err
						}
					} else {
						continue // not improving
					}
				} else {
					// ATSP - 2-opt* (no reversal). Replace (a→b),(c→d) with (a→d),(c→b).
					wab = weights.at(a, b)
					wcd = weights.at(c, d)
					wad = weights.at(a, d)
					wcb = weights.at(c, b)

					if math.IsInf(wad, 0) || math.IsInf(wcb, 0) {
						continue // candidate would introduce missing arcs
					}
					delta = (wad + wcb) - (wab + wcd)
					if delta < -eps {
						// Apply by rewiring successors and rebuilding the sequence (O(n)).
						cur = applyTwoOptStar(cur, opts.StartVertex, i, k)
					} else {
						continue // not improving
					}
				}

				// Update cost and bookkeeping after an accepted move.
				cost += delta
				accepted++
				improved = true

				// Guards.
				if maxIters > 0 && accepted >= maxIters {
					return finishLocalSearchCurrent(cur, cost, accepted, false, n, opts.StartVertex)
				}

				// First-improvement policy: restart scanning from the beginning.
				break
			}
			if improved {
				break
			}
		}

		if !improved {
			// Local optimum under the chosen neighborhood.
			break
		}
	}

	return finishLocalSearchCurrent(cur, cost, accepted, false, n, opts.StartVertex)
}

// applyTwoOptStar applies an asymmetric 2-opt* move to a closed tour without reversing
// the middle segment. It rewires two outgoing arcs and reconstructs the resulting cycle
// from a successor representation.
//
// Notation for closed tour T of length n+1:
//   - 1 <= i < k <= n-1.
//   - a=T[i-1], b=T[i], c=T[k], d=T[k+1].
//   - Remove arcs a->b and c->d.
//   - Add arcs a->d and c->b.
//
// Implementation:
//   - Stage 1: Build succ[v] from the current closed tour.
//   - Stage 2: Rewire succ[a]=d and succ[c]=b.
//   - Stage 3: Follow successors from start for n steps.
//   - Stage 4: Append start as the closing vertex.
//
// Behavior highlights:
//   - Allocates a fresh tour.
//   - Does not reverse tour[i:k].
//   - Intended for ATSP-compatible local search.
//   - Executed only after a move has been accepted by delta evaluation.
//
// Inputs:
//   - tour: closed tour.
//   - start: fixed start vertex.
//   - i: first cut index.
//   - k: second cut index.
//
// Returns:
//   - []int: fresh closed tour after the 2-opt* move.
//
// Errors:
//   - None. Caller must pass a validated accepted move.
//
// Determinism:
//   - Pure successor rewiring and deterministic reconstruction.
//
// Complexity:
//   - Time O(n), Space O(n).
//
// Notes:
//   - Symmetric 2-opt uses segment reversal instead.
//
// AI-Hints:
//   - Do not use reverseArcInPlace for 2-opt*; it changes the wrong arcs for ATSP.
func applyTwoOptStar(tour []int, start, i, k int) []int {
	n := len(tour) - 1

	// Build succ[v] from tour - one pass over n arcs of the cycle.
	succ := make([]int, n)
	var p, u, v int
	for p = 0; p < n; p++ {
		u = tour[p]
		v = tour[p+1]
		succ[u] = v
	}
	a := tour[i-1]
	b := tour[i]
	c := tour[k]
	d := tour[k+1]

	// Rewire successors to encode the 2-opt* swap: (a→d) and (c→b).
	succ[a] = d
	succ[c] = b

	// Rebuild the sequence by chasing successors from the fixed start vertex.
	out := make([]int, n+1)
	cur := start
	var idx int
	for idx = 0; idx < n; idx++ {
		out[idx] = cur
		cur = succ[cur]
	}
	out[n] = start // close the cycle

	return out
}
