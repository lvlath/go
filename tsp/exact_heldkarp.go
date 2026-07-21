// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp implements the Held-Karp exact dynamic program for TSP and ATSP.
//
// The solver computes an optimal Hamiltonian cycle over a complete finite
// distance matrix. Symmetry is not required, so asymmetric TSP instances are
// valid when the matrix itself satisfies the package numeric contract.
//
// Contracts:
//   - dist is a complete finite square n×n final solver matrix.
//   - diagonal ≈ 0; no NaN/-Inf; negative weights are forbidden.
//   - +Inf off-diagonal is rejected before DP because final TSP kernels consume complete instances.
//   - opts.StartVertex ∈ [0..n−1].
//
// Behavior:
//   - A soft size limit via MaxExactN (default 16) bounds time/space.
//   - If opts.TimeLimit > 0, we periodically check a deadline and return ErrTimeLimit.
//   - Final cost is stabilized to 1e−9 (round1e9) for cross-platform reproducibility.
//
// Complexity:
//   - Time  : O(n²·2ⁿ).
//   - Memory: O(n·2ⁿ) for DP and parent tables.
//
// Returns:
//   - A canonical Result through public facades; the private kernel computes the same optimal tour.
package tsp

import (
	"errors"
	"math"
	"math/bits"
	"time"

	"github.com/lvlath/go/matrix"
)

// MaxExactN bounds problem size for the Held–Karp solver (time/memory guard).
const MaxExactN = 16

// ErrSizeTooLarge signals that n exceeds MaxExactN (pragmatic resource limit).
var ErrSizeTooLarge = errors.New("tsp: exact solver supports at most 16 vertices")

// heldKarp solves a complete TSP or ATSP instance with Held-Karp DP.
//
// Implementation:
//   - Stage 1: Validate standalone options and copy the complete distance matrix.
//   - Stage 2: Enforce MaxExactN and StartVertex.
//   - Stage 3: Run heldKarpDP over the immutable weight buffer.
//   - Stage 4: Publish a detached exact optimal Result.
//
// Behavior highlights:
//   - Supports asymmetric distances.
//   - Requires complete finite matrix input.
//   - Does not run metric closure.
//   - Returns nil + ErrTimeLimit when the DP deadline expires before completion.
//
// Inputs:
//   - dist: complete finite square matrix.
//   - opts: ExactHeldKarp policy with StartVertex, MaxExactN, Eps, and TimeLimit.
//
// Returns:
//   - *Result: exact optimal result on completion.
//   - error: nil or sentinel-classified failure.
//
// Errors:
//   - ErrInvalidOptions.
//   - ErrNilDistanceMatrix, ErrNonSquare, ErrDimensionMismatch.
//   - ErrNaNInf, ErrNegativeWeight, ErrIncompleteGraph.
//   - ErrStartOutOfRange.
//   - ErrSizeTooLarge.
//   - ErrTimeLimit.
//
// Determinism:
//   - DP state iteration and predecessor reconstruction follow fixed scan order.
//
// Complexity:
//   - Time O(n^2 * 2^n), Space O(n * 2^n).
//
// Notes:
//   - This function owns exact/optimal metadata.
//   - It does not attach IDs; facade publication does that.
//
// AI-Hints:
//   - Do not return heuristic fallback on ErrSizeTooLarge.
//   - Do not mark timeout as partial unless an incumbent DP implementation exists.
func heldKarp(dist matrix.Matrix, opts Options) (*Result, error) {
	if err := validateOptionsStandalone(opts); err != nil {
		return nil, err
	}

	weights, err := copyCompleteWeights(dist, false)
	if err != nil {
		return nil, err
	}

	maxExactN := opts.MaxExactN
	if maxExactN == 0 {
		maxExactN = DefaultMaxExactN
	}
	if weights.n > maxExactN {
		return nil, ErrSizeTooLarge
	}

	if err = validateStartVertex(weights.n, opts.StartVertex); err != nil {
		return nil, err
	}

	tour, cost, err := heldKarpDP(weights, opts)
	if err != nil {
		return nil, err
	}

	return &Result{
		Tour:               append([]int(nil), tour...),
		Cost:               round1e9(cost),
		Algorithm:          ExactHeldKarp,
		Exact:              true,
		Optimal:            true,
		TimedOut:           false,
		Symmetric:          opts.Symmetric,
		ApproximationRatio: NoApproximationRatio,
	}, nil
}

// heldKarpDP computes an optimal closed Hamiltonian cycle with Held-Karp DP.
//
// Implementation:
//   - Stage 1: Copy the final complete matrix into a dense weight buffer.
//   - Stage 2: Fill subset DP tables in deterministic subset-size and mask order.
//   - Stage 3: Reconstruct one optimal tour through the parent table.
//   - Stage 4: Return a private route/cost payload for canonical publication.
//
// Behavior highlights:
//   - Supports asymmetric distances.
//   - Returns no partial result on timeout.
//   - Does not attach IDs or facade metadata.
//   - Rounds the final cost through round1e9.
//
// Inputs:
//   - dist: complete finite square matrix.
//   - opts: exact-solver policy with StartVertex, MaxExactN, Eps, and TimeLimit.
//
// Returns:
//   - kernelTour: optimal route/cost payload.
//   - error: nil on success or sentinel-classified failure.
//
// Errors:
//   - ErrInvalidOptions.
//   - ErrNilDistanceMatrix, ErrNonSquare, ErrDimensionMismatch.
//   - ErrNaNInf, ErrNegativeWeight, ErrIncompleteGraph.
//   - ErrStartOutOfRange.
//   - ErrSizeTooLarge.
//   - ErrTimeLimit.
//
// Determinism:
//   - Fixed subset cardinality, mask, endpoint, and predecessor scan order.
//
// Complexity:
//   - Time O(n^2 * 2^n), Space O(n * 2^n).
//
// Notes:
//   - The public wrapper attaches Exact=true and Optimal=true after success.
//
// AI-Hints:
//   - Do not replace this with exhaustive permutation enumeration.
//   - Do not mutate opts or caller-owned matrix data.
//   - The Held-Karp recurrence relation requires tight synchronization between
//     subset bitmasks and state transitions. Fragmenting this function would obscure
//     the mathematical precision of the dynamic programming state updates.
//
// nolint:gocyclo,gosec // Gocyclo: Kept monolithic for raw performance and math layout consistency.
//
//	// Gosec (G115): Integer conversions to uint for bitwise operations are safely bounded
//	   because N is restricted by exact-solver memory limits (N <= 31).
func heldKarpDP(weights weightBuffer, opts Options) ([]int, float64, error) {

	// Use the shared detached weight buffer so exact DP preserves the same
	// numeric and sentinel policy as local search and Branch-and-Bound.
	w := weights.w
	n := weights.n

	// Soft time budget: cheap deadline checks at a low fixed cadence.
	var (
		useDeadline bool
		deadline    time.Time
		step        int
	)
	if opts.TimeLimit > 0 {
		useDeadline = true
		deadline = time.Now().Add(opts.TimeLimit)
	}
	checkDeadline := func() bool {
		// Increment a local counter and check the wall clock every 1024 invocations.
		// This keeps overhead negligible vs. DP work in tight loops.
		step++
		if !useDeadline || (step&1023) != 0 {
			return false
		}
		return time.Now().After(deadline)
	}

	// DP tables in a flat layout to avoid [][] indexing overhead:
	//   dp[mask*n + j]     - min cost to visit the set "mask" and end at j (mask always contains "start"),
	//   parent[mask*n + j] - predecessor of j in the optimal transition into (mask, j).
	totalMasks := 1 << uint(n)
	dp := make([]float64, totalMasks*n)
	parent := make([]int, totalMasks*n)

	// Initialize dp to +Inf and parent to −1.
	for idx := 0; idx < totalMasks*n; idx++ {
		dp[idx] = math.Inf(1)
		parent[idx] = -1
	}

	start := opts.StartVertex
	startBit := 1 << uint(start)
	baseMask := startBit
	dp[baseMask*n+start] = 0 // base state: at start, only start visited

	// Precompute lists of masks by popcount to avoid repeated popcount in hot loops.
	// We only keep masks that include the start bit.
	masksBySize := make([][]int, n+1)
	var mask int
	for mask = 0; mask < totalMasks; mask++ {
		if (mask & startBit) == 0 {
			continue
		}
		ps := bits.OnesCount(uint(mask))
		if ps >= 1 && ps <= n {
			masksBySize[ps] = append(masksBySize[ps], mask)
		}
	}

	// Main DP: grow subset size |mask| from 2..n.
	var (
		size int
		jbit int
		kbit int
		j, k int
		prev int
		wij  float64
	)
	for size = 2; size <= n; size++ {
		for _, mask = range masksBySize[size] {
			// For each possible endpoint j in "mask", j ≠ start:
			for j = 0; j < n; j++ {
				jbit = 1 << uint(j)
				if j == start || (mask&jbit) == 0 {
					continue
				}
				prev = mask ^ jbit // predecessor subset w/o j
				// Relax over all k ∈ prev: dp[mask,j] = min_k dp[prev,k] + w[k→j].
				var best float64
				best = math.Inf(1)
				var argk = -1

				for k = 0; k < n; k++ {
					kbit = 1 << uint(k)
					if (prev & kbit) == 0 {
						continue
					}
					var base = dp[prev*n+k]
					if math.IsInf(base, 1) {
						continue // unreachable state
					}
					wij = w[k*n+j]
					if math.IsInf(wij, 0) {
						continue // no edge k→j
					}
					var cand = base + wij
					if cand < best {
						best = cand
						argk = k
					}
				}
				if argk >= 0 {
					dp[mask*n+j] = best
					parent[mask*n+j] = argk
				}

				if checkDeadline() {
					return nil, 0, ErrTimeLimit
				}
			}
		}
	}

	// Close the tour back to start: choose the best last vertex j and add w[j→start].
	all := totalMasks - 1
	var (
		bestCost = math.Inf(1)
		last     = -1
	)
	for j = 0; j < n; j++ {
		if j == start {
			continue
		}
		var base = dp[all*n+j]
		if math.IsInf(base, 1) {
			continue
		}
		wij = w[j*n+start]
		if math.IsInf(wij, 0) {
			continue
		}
		var total = base + wij
		if total < bestCost {
			bestCost = total
			last = j
		}
	}
	if last < 0 || math.IsInf(bestCost, 1) {
		return nil, 0, ErrIncompleteGraph
	}

	// Reconstruct the optimal tour by walking parents backward from (mask=all, j=last).
	tour := make([]int, n+1)
	tour[0] = start
	tour[n] = start

	mask = all
	cur := last
	for idx := n - 1; idx >= 1; idx-- {
		tour[idx] = cur
		prev = parent[mask*n+cur]
		mask ^= 1 << uint(cur) // remove cur from the subset
		cur = prev
	}

	// Canonicalize direction (fixed start) and enforce final tour invariants.
	err := CanonicalizeOrientationInPlace(tour)
	if err != nil {
		return nil, 0, err
	}
	if err = ValidateTour(tour, n, start); err != nil {
		return nil, 0, err
	}

	return tour, bestCost, nil
}
