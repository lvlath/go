// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp defines shared local-search result and deadline helpers.
// These helpers keep 2-opt and 3-opt timeout behavior, ownership, iteration
// metadata, and canonical facade publication consistent.
package tsp

import (
	"math"
	"time"
)

const (
	// twoOptDeadlineCheckMask checks the 2-opt deadline every 2048 candidate events.
	// The value is a power-of-two-minus-one mask to make the hot-path test cheap.
	twoOptDeadlineCheckMask = 2047

	// threeOptDeadlineCheckMask checks the 3-opt deadline every 4096 candidate events.
	// 3-opt has a larger neighborhood, so the check interval is slightly wider.
	threeOptDeadlineCheckMask = 4095
)

// localSearchNow is the wall-clock source used by local-search deadline checks.
// It is centralized so all local-search kernels use identical timeout semantics.
//
// Implementation:
//   - Stage 1: Default to time.Now in production.
//   - Stage 2: Kernels call localSearchNow only through deadline helpers.
//   - Stage 3: TimeLimit==0 bypasses deadline evaluation during search.
//
// Behavior highlights:
//   - Unexported.
//   - No allocation.
//   - Does not affect deterministic runs when TimeLimit==0.
//
// Inputs:
//   - None.
//
// Returns:
//   - time.Time: current wall-clock value.
//
// Errors:
//   - None.
//
// Determinism:
//   - TimeLimit==0 avoids clock reads after setup.
//   - Deadline-triggered partial results still require a valid tour and cost.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Do not expose this variable publicly.
//   - Do not use it for randomness or tie-breaking.
//
// AI-Hints:
//   - Do not call time.Now directly inside local-search kernels.
//   - Wall-clock timeout tests require controlled scheduling, not elapsed-time guesses.
var localSearchNow = time.Now

// localSearchResult carries the internal result of a local-search kernel.
// It is intentionally private because canonical public callers consume Result,
// while public callers consume canonical Result metadata.
//
// Implementation:
//   - Stage 1: Kernels fill tour, cost, and iteration count after successful finalization.
//   - Stage 2: Timeout paths set timedOut=true and still carry a valid current tour.
//   - Stage 3: solvePreparedMatrix publishes this state into Result.
//
// Behavior highlights:
//   - tour is detached from caller input.
//   - cost is stabilized with round1e9.
//   - iterations counts accepted improving moves, not candidate evaluations.
//
// Inputs:
//   - Produced by twoOptKernel and threeOptKernel.
//
// Returns:
//   - Consumed internally by wrappers and canonical dispatcher.
//
// Errors:
//   - Kernel functions return errors separately.
//
// Determinism:
//   - Mirrors the deterministic scan order and seed policy of the producing kernel.
//
// Complexity:
//   - Access Time O(1), Space O(len(tour)) for stored route.
//
// Notes:
//   - ErrTimeLimit with a non-empty result is governed partial success.
//   - A zero localSearchResult means no valid tour is available.
//
// AI-Hints:
//   - Do not expose localSearchResult in public APIs.
//   - Do not publish timedOut local results as Optimal=true.
type localSearchResult struct {
	// tour is a detached closed Hamiltonian cycle produced by the local-search kernel.
	// It is empty only when no valid current result exists.
	tour []int

	// cost is the stabilized round1e9 cost of tour.
	// It is not used to prove optimality.
	cost float64

	// iterations counts accepted improving moves.
	// It does not count candidate evaluations.
	iterations int

	// timedOut reports that the local-search deadline stopped the kernel.
	// When true and tour is non-empty, the result is a governed partial result.
	timedOut bool
}

// hasTour reports whether the local-search result carries a usable tour snapshot.
//
// Implementation:
//   - Stage 1: Check only len(tour)>0.
//   - Stage 2: Leave structural validation to kernel finalization.
//
// Behavior highlights:
//   - No allocation.
//   - Safe on zero value.
//
// Inputs:
//   - r: local search result.
//
// Returns:
//   - bool: true when r contains a tour.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pure value check.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This helper is intentionally stricter than cost checks; a zero-cost tour can be valid.
//
// AI-Hints:
//   - Do not infer presence from cost because degenerate or zero-cost tours are possible.
func (r localSearchResult) hasTour() bool {
	return len(r.tour) > 0
}

// costClosedTourFromWeights sums the arcs of a validated closed tour snapshot.
// Local-search delta computations select moves, but repeated additions of their
// differences lose low bits when the objective has a large common baseline.
//
// Implementation:
//   - Stage 1: Read every directed arc of the final tour from detached weights.
//   - Stage 2: Accumulate with a pre-addition overflow check.
//   - Stage 3: Apply the established nine-decimal TSP presentation policy.
//
// Behavior highlights:
//   - Does not read the caller's mutable Matrix again.
//   - Uses directed weights even when the requested policy is symmetric.
//
// Inputs:
//   - weights: complete validated row-major snapshot built before local search.
//   - tour: validated Hamiltonian cycle with indices in [0,weights.n).
//
// Returns:
//   - float64: finite cost of exactly the published directed arc order.
//
// Errors:
//   - ErrNaNInf, ErrIncompleteGraph, ErrNegativeWeight on corrupted internal weights.
//   - ErrNumericOverflow if the sum cannot be represented as float64.
//
// Determinism:
//   - Adds arcs in tour order with no parallel or unordered reduction.
//
// Complexity:
//   - Time O(n), Space O(1) after the existing O(n^2) weight preparation.
//
// Notes:
//   - Caller validates the tour before invoking this private numerical helper.
//   - The result can differ from a cost reconstructed by accumulating move deltas.
//
// AI-Hints:
//   - Re-evaluate only during result publication, not for every 2-opt candidate.
//   - Do not use the symmetric boundary-edge shortcut to price an ATSP reversal.
func costClosedTourFromWeights(weights weightBuffer, tour []int) (float64, error) {
	total := 0.0
	for position := 0; position+1 < len(tour); position++ {
		weight := weights.at(tour[position], tour[position+1])
		switch {
		case math.IsNaN(weight) || math.IsInf(weight, -1):
			return 0, ErrNaNInf
		case math.IsInf(weight, 1):
			return 0, ErrIncompleteGraph
		case weight < 0:
			return 0, ErrNegativeWeight
		case total > math.MaxFloat64-weight:
			return 0, ErrNumericOverflow
		}
		total += weight
	}
	return round1e9(total), nil
}

// finishLocalSearchCurrent canonicalizes, validates, and detaches the current local-search tour.
// It is used by both success and timeout paths so partial results obey the same invariant
// as fully successful local-search results.
//
// Implementation:
//   - Stage 1: Canonicalize orientation only when reversed arcs have identical costs.
//   - Stage 2: Validate Hamiltonian cycle invariants.
//   - Stage 3: Recalculate the objective from detached validated arc weights.
//   - Stage 4: Copy the tour and publish its cost, iterations, and timeout state.
//
// Behavior highlights:
//   - Mutates the working tour only for canonical orientation.
//   - Returns a detached tour snapshot.
//   - Uses the same validation for success and partial timeout.
//   - Recomputes cost from the published arc order instead of accumulating deltas.
//
// Inputs:
//   - tour: current closed Hamiltonian tour owned by the kernel.
//   - weights: validated detached n-by-n directed arc costs.
//   - iterations: accepted move count.
//   - timedOut: true when publishing a partial timeout result.
//   - n: vertex count.
//   - start: required start/end vertex.
//   - symmetric: whether reversal preserves the direction-sensitive cost objective.
//
// Returns:
//   - localSearchResult: finalized detached local result.
//   - error: nil or sentinel-classified invariant failure.
//
// Errors:
//   - ErrDimensionMismatch from canonicalization shape failures.
//   - ErrInvalidTour from ValidateTour structural failures.
//   - ErrStartOutOfRange from invalid start.
//   - ErrNumericOverflow if the selected finite arc costs exceed float64 range.
//
// Determinism:
//   - Symmetric orientation uses the fixed neighbor comparison; directed order is unchanged.
//   - Validation scans left-to-right.
//
// Complexity:
//   - Time O(n), Space O(n).
//
// Notes:
//   - Run this only after a valid current tour exists.
//   - Candidate-level rejected moves must not call this helper.
//   - The complete input was already checked by copyCompleteWeights.
//
// AI-Hints:
//   - Do not reverse ATSP results just to choose a lexicographic orientation.
//   - Do not skip this helper on timeout; partial results still need invariants.
//   - Do not publish the working slice directly.
//   - Never publish an incremental cost that can drift from the final arc order.
func finishLocalSearchCurrent(
	tour []int,
	weights weightBuffer,
	iterations int,
	timedOut bool,
	n int,
	start int,
	symmetric bool,
) (localSearchResult, error) {
	if n != weights.n {
		return localSearchResult{}, ErrDimensionMismatch
	}
	if err := canonicalizeTourIfSymmetric(tour, symmetric); err != nil {
		return localSearchResult{}, err
	}
	if err := ValidateTour(tour, n, start); err != nil {
		return localSearchResult{}, err
	}
	cost, err := costClosedTourFromWeights(weights, tour)
	if err != nil {
		return localSearchResult{}, err
	}

	return localSearchResult{
		tour:       append([]int(nil), tour...),
		cost:       cost,
		iterations: iterations,
		timedOut:   timedOut,
	}, nil
}

// publishLocalSearchResult converts a finalized local-search result to Result.
// It is used by top-level local-search-only dispatcher branches.
//
// Implementation:
//   - Stage 1: Detach tour and IDs.
//   - Stage 2: Attach algorithm and facade metadata.
//   - Stage 3: Mark exact/optimal fields according to heuristic local-search semantics.
//
// Behavior highlights:
//   - Exact=false and Optimal=false even on full local optimum.
//   - TimedOut mirrors local.timedOut.
//   - Iterations preserves accepted move count.
//
// Inputs:
//   - local: finalized local-search result.
//   - ids: optional matrix-index ordered vertex IDs.
//   - opts: finalized solver policy.
//   - algorithm: selected top-level algorithm.
//   - metricClosureApplied: whether the facade applied APSP closure.
//
// Returns:
//   - *Result: detached canonical local-search result.
//
// Errors:
//   - None. The caller must pass a finalized localSearchResult.
//
// Determinism:
//   - Preserves local tour order and ID order.
//
// Complexity:
//   - Time O(len(tour)+len(ids)), Space O(len(tour)+len(ids)).
//
// Notes:
//   - This helper does not validate the tour; kernels already finalized local.
//
// AI-Hints:
//   - Do not mark local-search-only results as Exact.
//   - Do not drop TimedOut when returning ErrTimeLimit.
func publishLocalSearchResult(
	local localSearchResult,
	ids []string,
	opts Options,
	algorithm Algorithm,
	metricClosureApplied bool,
) *Result {
	return &Result{
		Tour:                 append([]int(nil), local.tour...),
		Cost:                 local.cost,
		IDs:                  append([]string(nil), ids...),
		Algorithm:            algorithm,
		Exact:                false,
		Optimal:              false,
		TimedOut:             local.timedOut,
		MetricClosureApplied: metricClosureApplied,
		Symmetric:            opts.Symmetric,
		ApproximationRatio:   NoApproximationRatio,
		Iterations:           local.iterations,
	}
}

// attachLocalSearchProgress applies a finalized local-search result to an existing Result.
// It is used by Christofides post-passes because Christofides already published a feasible
// Hamiltonian tour and local search only improves or partially improves that tour.
//
// Implementation:
//   - Stage 1: Ignore empty local results.
//   - Stage 2: Replace Tour and Cost with detached local values.
//   - Stage 3: Add iterations and propagate timeout status.
//
// Behavior highlights:
//   - Preserves approximation metadata unless timed out.
//   - On timeout, Optimal is cleared because the refinement did not complete.
//   - Does not mutate local.tour.
//
// Inputs:
//   - result: existing canonical result to update.
//   - local: finalized local-search result.
//
// Returns:
//   - None.
//
// Errors:
//   - None. Inputs are assumed finalized by kernels.
//
// Determinism:
//   - Preserves local tour order exactly.
//
// Complexity:
//   - Time O(len(local.tour)), Space O(len(local.tour)).
//
// Notes:
//   - A timed-out local refinement still owns a valid tour and cost.
//   - ApproximationRatio remains metadata of the pre-refinement construction.
//
// AI-Hints:
//   - Do not clear approximation metadata here; local search improves a feasible route but does not prove a new ratio.
//   - Do not drop partial local-search progress on ErrTimeLimit.
func attachLocalSearchProgress(result *Result, local localSearchResult) {
	if result == nil || !local.hasTour() {
		return
	}

	result.Tour = append([]int(nil), local.tour...)
	result.Cost = local.cost
	result.Iterations += local.iterations

	if local.timedOut {
		result.TimedOut = true
		result.Optimal = false
	}
}

// localSearchDeadlineExpired reports whether a local-search deadline has passed.
// It centralizes the clock source and keeps timeout behavior consistent between
// 2-opt and 3-opt.
//
// Implementation:
//   - Stage 1: Return false when no deadline is active.
//   - Stage 2: Compare localSearchNow against the fixed deadline.
//
// Behavior highlights:
//   - No allocation.
//   - No hidden side effects besides reading the clock.
//
// Inputs:
//   - useDeadline: whether TimeLimit is active.
//   - deadline: absolute deadline computed at kernel start.
//
// Returns:
//   - bool: true when the deadline has passed.
//
// Errors:
//   - None.
//
// Determinism:
//   - Deterministic when TimeLimit==0 or when tests control localSearchNow.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Kernels throttle calls with power-of-two masks.
//
// AI-Hints:
//   - Do not perform direct time.Now comparisons in local-search kernels.
func localSearchDeadlineExpired(useDeadline bool, deadline time.Time) bool {
	return useDeadline && localSearchNow().After(deadline)
}
