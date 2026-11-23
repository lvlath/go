// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp implements deterministic 2-opt search on directed and undirected tours.
// Both modes reverse the selected segment; directed costs additionally account for
// every reversed internal arc, preserving the Hamiltonian-cycle invariant.
//
// For cuts a=T[i-1], b=T[i], c=T[k], d=T[k+1], the new boundary arcs
// are a→c and b→d. The asymmetric delta also includes the reversal cost
// of every original arc T[t]→T[t+1] for i<=t<k.
//
// Design:
//   - Deterministic scanning order; no RNG usage (seed reserved for future shuffles).
//   - Strict sentinel errors only (see types.go). No fmt.Errorf in hot paths.
//   - Symmetric candidate checks are O(1); directed checks inspect reversed arcs.
//   - Soft time budget via periodic deadline checks.
//   - Cost stabilized to 1e−9 via round1e9.
//
// Contracts:
//   - dist is a complete n×n final solver matrix validated through copyCompleteWeights.
//   - tour is a *closed* Hamiltonian cycle (len==n+1, tour[0]==tour[n]==opts.StartVertex).
//   - Directed input retains the original start and validates all reversed arcs.
//
// Complexity:
//   - One pass: O(n²) symmetric or O(n³) directed candidate work.
//   - Every accepted reversal costs O(n) worst-case, using O(1) extra space.
//   - First-improvement restarts after each accepted move.
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
//   - Time O(iterations*n^2) symmetric, O(iterations*n^3) directed; Space O(n).
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

// twoOptKernel runs deterministic first-improvement 2-opt and returns
// structured local-search metadata for canonical facades.
//
// Implementation:
//   - Stage 1: Validate options, tour shape, complete weights, and tour invariants.
//   - Stage 2: Copy the input tour and compute baseline cost.
//   - Stage 3: Scan candidate pairs in deterministic order and apply first improvements.
//   - Stage 4: Verify the selected tour and compute its cost from stored arc weights.
//
// Behavior highlights:
//   - Symmetric mode reverses one segment.
//   - ATSP mode also reverses the segment, but pays for reversed directed internal arcs.
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
//   - Time O(iter*n^2) for symmetric input or O(iter*n^3) for directed input.
//   - Space O(n); directed candidate work includes O(k-i) reversed arcs.
//   - Accepted moves reverse O(k-i) vertices in place.
//
// Notes:
//   - iterations counts accepted moves, not candidate evaluations.
//   - TimeLimit==0 disables deadline checks.
//
// AI-Hints:
//   - Do not return nil tour on ErrTimeLimit after a valid current tour exists.
//   - Do not let +Inf into final local-search kernels.
//   - Directed reversals must include the full internal arc-difference sum.
//   - Two disconnected subtours cannot be published as a Hamiltonian cycle.
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
	if _, err = TourCost(dist, cur); err != nil {
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
			a, b, c, d int     // boundary endpoints around (i,k)
			delta      float64 // candidate improvement (negative is good)
			wab, wcd   float64 // removed boundary arc weights
			wac, wbd   float64 // replacement boundary arc weights
			i, k       int     // candidate cut indices, 1 ≤ i < k ≤ n−1
		)

		// Scan all candidate pairs (i,k) with 1 ≤ i < k ≤ n−1.
		// The fixed start is outside the reversed segment in either mode.
		for i = 1; i <= n-2; i++ {
			for k = i + 1; k <= n-1; k++ {
				if checkDeadline() {
					local, finishErr := finishLocalSearchCurrent(cur, weights, accepted, true, n, opts.StartVertex, opts.Symmetric)
					if finishErr != nil {
						return localSearchResult{}, finishErr
					}
					return local, ErrTimeLimit
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
					// A two-cut tail-swap of a single directed cycle would split it
					// into disjoint subtours. Instead, reverse [i..k] as a genuine
					// Hamiltonian 2-opt move. Unlike the symmetric case, each
					// internal directed arc changes direction and contributes to Δ.
					wab = weights.at(a, b)
					wcd = weights.at(c, d)
					wac = weights.at(a, c)
					wbd = weights.at(b, d)
					if math.IsInf(wac, 0) || math.IsInf(wbd, 0) {
						continue
					}
					delta = (wac + wbd) - (wab + wcd)
					for arc := i; arc < k; arc++ {
						forward := weights.at(cur[arc], cur[arc+1])
						reverse := weights.at(cur[arc+1], cur[arc])
						if math.IsInf(reverse, 0) {
							delta = math.Inf(1)
							break
						}
						delta += reverse - forward
					}
					if delta >= -eps {
						continue
					}
					if err = reverseArcInPlace(cur, i, k); err != nil {
						return localSearchResult{}, err
					}
				}

				// Record the committed improvement. Final cost is recomputed
				// from the published tour to avoid accumulating numerical drift.
				accepted++
				improved = true

				// Guards.
				if maxIters > 0 && accepted >= maxIters {
					return finishLocalSearchCurrent(cur, weights, accepted, false, n, opts.StartVertex, opts.Symmetric)
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

	return finishLocalSearchCurrent(cur, weights, accepted, false, n, opts.StartVertex, opts.Symmetric)
}
