// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw implements machine-certified LB_Keogh pruning for squared-cost search.
// This file separates sound exclusion certificates from ordinary rounded diagnostics.
package dtw

import (
	"context"
	"math"
	"math/bits"
	"unsafe"
)

// searchEnvelope owns a finite radius-aware query envelope and its global extrema.
// Each candidate reads this immutable state; only the search invocation creates it.
type searchEnvelope struct {
	lower   []float64
	upper   []float64
	minimum float64
	maximum float64
}

// optionalEnvelopeBytes estimates every additional backing buffer required for pruning.
// Two float64 envelope vectors survive preparation; two integer monotone deques are
// temporary, but are charged as well so the configured budget covers peak allocation.
//
// Implementation:
//   - Stage 1: compute the constant per-position buffer cost with machine-size types.
//   - Stage 2: multiply by query length using checked 128-bit integer arithmetic.
//   - Stage 3: reject counts that cannot fit a host-sized backing allocation.
//
// Behavior highlights:
//   - Pruning is optional; its workspace never silently reduces the exact scorer budget.
//
// Inputs:
//   - length: positive validated query length, in float64 samples.
//
// Returns:
//   - uint64: conservative extra bytes; bool: true if the size is not representable.
//
// Errors:
//   - None; overflow disables an optional optimization, not the exact search.
//
// Determinism:
//   - Pure integer arithmetic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Goroutine stacks and allocator metadata follow the existing search budget policy.
//
// AI-Hints:
//   - Never allocate uncounted temporary deques under MaxWorkingBytes.
//   - Do not use int multiplication until the product fits the machine address domain.
func optionalEnvelopeBytes(length int) (uint64, bool) {
	const buffersPerCoordinate = 2*unsafe.Sizeof(float64(0)) + 2*unsafe.Sizeof(int(0))
	count, overflow := checkedBytes(uint64(length), uint64(buffersPerCoordinate))
	if overflow {
		return 0, true
	}
	count, carry := bits.Add64(count, uint64(unsafe.Sizeof(searchEnvelope{})), 0)
	return count, carry != 0 || count > uint64(maxIntValue)
}

// prepareSearchEnvelope constructs a finite, radius-aware envelope only when it fits.
// If an optional envelope would exceed MaxWorkingBytes, the caller uses the unchanged
// exhaustive scorer; this choice affects performance, never the ranking or error law.
//
// Implementation:
//   - Stage 1: validate the optional allocation against the already accounted base.
//   - Stage 2: build min/max over each center's admitted Sakoe-Chiba neighborhood.
//   - Stage 3: publish immutable vectors and global extrema to worker readers.
//
// Behavior highlights:
//   - Radius=-1 and radius>=length-1 both mean an unconstrained full-query envelope.
//   - The query has already been normalized when SearchZPopulation was selected.
//
// Inputs:
//   - query: validated finite, immutable scorer-side samples.
//   - radius: -1 or nonnegative; baseBytes: mandatory accounted workspace bytes.
//   - maxBytes: 0 means no explicit workspace limit.
//
// Returns:
//   - *searchEnvelope: immutable optional certificate data, or nil when unavailable.
//
// Errors:
//   - None; memory uncertainty disables pruning and retains the exact scorer.
//
// Determinism:
//   - Increasing-index insertion and eviction; equal samples retain the earliest index.
//
// Complexity:
//   - Time O(L), Space O(L) when enabled; O(1) if disabled by the budget.
//
// Notes:
//   - The caller must not modify query while this function or any worker is running.
//
// AI-Hints:
//   - An absent envelope must NEVER cause a candidate to be skipped.
//   - Do not approximate radius by a global interval for a claimed radius-aware certificate.
func prepareSearchEnvelope(query []float64, radius int, baseBytes, maxBytes uint64) *searchEnvelope {
	extra, overflow := optionalEnvelopeBytes(len(query))
	if overflow {
		return nil
	}
	total, carry := bits.Add64(baseBytes, extra, 0)
	if carry != 0 || total > uint64(maxIntValue) || (maxBytes != 0 && total > maxBytes) {
		return nil
	}

	length := len(query)
	envelope := &searchEnvelope{
		lower:   make([]float64, length),
		upper:   make([]float64, length),
		minimum: query[0],
		maximum: query[0],
	}

	// The two deques hold indices, not floats. Each index enters and leaves
	// at most once, so even the unconstrained radius takes linear time.
	minimumDeque := make([]int, length)
	maximumDeque := make([]int, length)
	minHead, minTail, maxHead, maxTail := 0, 0, 0, 0
	right := 0
	for center := 0; center < length; center++ {
		end := length
		if radius >= 0 && radius < length-1-center {
			end = center + radius + 1
		}
		for right < end {
			value := query[right]
			if value < envelope.minimum {
				envelope.minimum = value
			}
			if value > envelope.maximum {
				envelope.maximum = value
			}
			for minTail > minHead && value < query[minimumDeque[minTail-1]] {
				minTail--
			}
			minimumDeque[minTail] = right
			minTail++
			for maxTail > maxHead && value > query[maximumDeque[maxTail-1]] {
				maxTail--
			}
			maximumDeque[maxTail] = right
			maxTail++
			right++
		}

		start := 0
		if radius >= 0 && center > radius {
			start = center - radius
		}
		for minHead < minTail && minimumDeque[minHead] < start {
			minHead++
		}
		for maxHead < maxTail && maximumDeque[maxHead] < start {
			maxHead++
		}
		envelope.lower[center] = query[minimumDeque[minHead]]
		envelope.upper[center] = query[maximumDeque[maxHead]]
	}
	return envelope
}

// certifiedKeoghPrune proves that a candidate cannot improve an existing finite cutoff.
// It compares a lower envelope cost to the actual rounded scorer through the monotonic
// arithmetic law, rather than assuming that a rounded real-arithmetic bound is safe.
//
// Implementation:
//   - Stage 1: prove that every possible local squared difference is finite by extrema.
//   - Stage 2: prove that a finite admissible diagonal path exists in the scorer model.
//   - Stage 3: accumulate the radius-aware Keogh costs in the scorer's operation order.
//   - Stage 4: prune only if the finite machine lower bound is strictly above cutoff.
//
// Behavior highlights:
//   - Every admitted DTW path visits every candidate index at least once in order.
//   - For each candidate index, envelope deviation cannot exceed any admitted local
//     squared difference because rounded subtraction, squaring, and checked addition
//     are monotone on finite nonnegative operands.
//   - Removing other nonnegative path costs can only decrease a sequentially rounded
//     sum. Therefore the machine accumulated envelope cost is <= every valid path.
//   - Finite diagonal proves the optimal terminal score is representable; the extrema
//     proof prevents pruning from hiding an admitted local-cost overflow error.
//
// Inputs:
//   - ctx: non-nil invocation context, checked cooperatively.
//   - query, candidate: finite, equal-length, immutable scorer-side sequences.
//   - envelope: finite query bounds from the same radius and normalization domain.
//   - cutoff: finite exact kth score; equality is NOT prunable.
//
// Returns:
//   - bool: true only for a certified noncompetitive candidate.
//   - error: context cancellation; uncertain numeric comparisons produce false, nil.
//
// Errors:
//   - context.Canceled/DeadlineExceeded when the context stops during proof.
//
// Determinism:
//   - Ascending-index proof and strict float64 comparisons, without epsilon.
//
// Complexity:
//   - Time O(L), Space O(1) per attempted certificate.
//
// Notes:
//   - The certification is specific to equal-length squared scalar DTW with zero penalty.
//   - Unrepresentable local/diagonal values trigger the ordinary scorer, preserving its
//     candidate-level error semantics rather than turning a numerical failure into a prune.
//
// AI-Hints:
//   - Do not substitute a single Nextafter on a rounded LB_Keogh sum.
//   - Do not remove the finite-diagonal or full-local-domain guard: both preserve errors.
//   - Never prune on lowerBound==cutoff even if the current start seems less competitive.
func certifiedKeoghPrune(ctx context.Context, query, candidate []float64, envelope *searchEnvelope, cutoff float64) (bool, error) {
	if envelope == nil || math.IsInf(cutoff, 1) || math.IsNaN(cutoff) {
		return false, nil
	}
	length := len(query)
	if len(candidate) != length || len(envelope.lower) != length || len(envelope.upper) != length {
		return false, nil
	}

	// Both extreme cross-pairs dominate the absolute difference of any other
	// query/candidate pair. Testing them prevents a certificate from concealing
	// nonfinite local costs in unvisited cells of the original scorer.
	candidateMin, candidateMax := candidate[0], candidate[0]
	for _, value := range candidate[1:] {
		if value < candidateMin {
			candidateMin = value
		}
		if value > candidateMax {
			candidateMax = value
		}
	}
	for _, difference := range [...]float64{
		envelope.minimum - candidateMax,
		envelope.maximum - candidateMin,
	} {
		if !isFiniteSquaredDifference(difference) {
			return false, nil
		}
	}

	// The strict diagonal lies inside every nonnegative-radius band, and inside
	// the unconstrained domain. A finite diagonal proves that an optimal path
	// cannot overflow, so skipping this candidate cannot hide ErrOverflow.
	var diagonal float64
	for index := 0; index < length; index++ {
		if err := ctx.Err(); err != nil {
			return false, err
		}
		difference := query[index] - candidate[index]
		local := difference * difference
		if math.IsNaN(local) || math.IsInf(local, 0) {
			return false, nil
		}
		var overflow bool
		diagonal, overflow = checkedNonnegativeAdd(diagonal, local)
		if overflow {
			return false, nil
		}
	}

	// For each candidate coordinate, its envelope floor is no greater than
	// the cost of any query coordinate the band can legally align to it.
	// Summing one floor per candidate coordinate in chronological order is a
	// machine lower bound on every completed monotone path, including stretches.
	var bound float64
	for index, value := range candidate {
		if err := ctx.Err(); err != nil {
			return false, err
		}
		var difference float64
		if value < envelope.lower[index] {
			difference = envelope.lower[index] - value
		} else if value > envelope.upper[index] {
			difference = envelope.upper[index] - value
		}
		local := difference * difference
		if math.IsNaN(local) || math.IsInf(local, 0) {
			return false, nil
		}
		var overflow bool
		bound, overflow = checkedNonnegativeAdd(bound, local)
		if overflow {
			return false, nil
		}
		if bound > cutoff {
			return true, nil
		}
	}
	return false, nil
}

// isFiniteSquaredDifference checks whether a float64 subtraction can be squared safely.
// The check is used only to gate pruning; failure delegates to the exact scorer.
//
// Implementation:
//   - Stage 1: exclude nonfinite differences.
//   - Stage 2: square using the same binary64 operation as scoreSquaredBand.
//   - Stage 3: accept only a finite result.
//
// Behavior highlights:
//   - Subnormal underflow is accepted as the scorer also rounds the square.
//
// Inputs:
//   - difference: already evaluated float64 subtraction of finite samples.
//
// Returns:
//   - bool: whether the scorer's squared local cost is finite.
//
// Errors:
//   - None; false simply disables the optional certificate.
//
// Determinism:
//   - Pure float64 comparison.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - The checked accumulated sum has a separate finite-diagonal precondition.
//
// AI-Hints:
//   - Do not reject underflowed zero squares: the scorer permits them.
func isFiniteSquaredDifference(difference float64) bool {
	if math.IsInf(difference, 0) || math.IsNaN(difference) {
		return false
	}
	square := difference * difference
	return !math.IsInf(square, 0) && !math.IsNaN(square)
}
