// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp defines tour validation, canonicalization, construction, and cost helpers.
//
// A tour is represented as a closed vertex-index sequence of length n+1:
// tour[0] == tour[n] == start, and every vertex in [0,n) appears exactly once
// before the closing start vertex.
//
// Contracts:
//   - ValidateTour checks closed Hamiltonian cycle shape.
//   - ValidatePermutation checks open permutation shape without a closing vertex.
//   - TourCost computes the sum of directed arc costs in tour order.
//   - Dense and generic cost paths must preserve the same sentinel behavior.
//
// Complexity:
//   - Tour validation is O(n) time and O(n) memory.
//   - Tour cost is O(n) time and O(1) memory, excluding matrix access cost.
//   - Canonical rotation is O(n) time and O(n) memory unless performed in place.
//
// AI-Hints:
//   - Do not use map iteration for tour canonicalization.
//   - Do not treat permutation and closed-tour representations as interchangeable.
//   - Do not let dense and generic cost paths drift in sentinel behavior.
package tsp

import (
	"errors"
	"math"

	"github.com/lvlath/go/matrix"
)

// roundScale controls final cost stabilization precision (1e-9).
// Avoids tiny FP drifts across platforms/opt levels without affecting optimality.
const roundScale = 1e9

// ValidatePermutation checks that perm is a complete permutation of local vertices [0,n).
// It rejects length mismatches, out-of-range vertices, and duplicate vertices before
// downstream tour-cost or local-search code can observe corrupt cyclic state.
//
// Implementation:
//   - Stage 1: Validate the permutation length.
//   - Stage 2: Allocate one O(n) marker slice.
//   - Stage 3: Scan vertices, checking bounds and duplicates.
//   - Stage 4: Mark each seen vertex exactly once.
//
// Behavior highlights:
//   - Does not mutate perm.
//   - Does not accept partial paths.
//   - Uses sentinel errors so callers can preserve API-level validation semantics.
//
// Inputs:
//   - perm: candidate vertex order without the closing duplicate.
//   - n: expected number of local vertices.
//
// Returns:
//   - error: nil when perm contains every vertex in [0,n) exactly once.
//
// Errors:
//   - ErrDimensionMismatch for wrong length.
//   - ErrInvalidVertex for out-of-range vertices.
//   - ErrDuplicateVertex for repeated vertices.
//
// Determinism:
//   - Fixed left-to-right scan.
//
// Complexity:
//   - Time O(n), Space O(n).
//
// Notes:
//   - Closed tours should pass only their first n vertices to this function.
//
// AI-Hints:
//   - Do not replace this with sorting; order must be preserved for tour semantics.
func ValidatePermutation(perm []int, n int) error {
	if len(perm) != n {
		return ErrDimensionMismatch
	}
	if n <= 0 {
		return ErrDimensionMismatch
	}
	seen := make([]bool, n)

	var (
		i int
		v int
	)
	for i = 0; i < n; i++ {
		v = perm[i]
		// Out-of-range element violates the dimension contract.
		if v < 0 || v >= n {
			return ErrDimensionMismatch
		}
		// Duplicate also violates the bijection/dimension contract.
		if seen[v] {
			return ErrDimensionMismatch
		}
		seen[v] = true
	}
	return nil
}

// MakeTourFromPermutation builds a closed Hamiltonian tour from a vertex permutation.
// Steps:
//  1. Validate that perm is a permutation of {0..n-1}.
//  2. Find the index of start in perm; rotate so perm[idx]==start becomes position 0.
//  3. Return a new slice of length n+1 with the closing start at position n.
//
// Contract:
//   - perm is a permutation (ValidatePermutation).
//   - start ∈ [0..n-1] and present in perm.
//   - Returned tour satisfies: len==n+1, tour[0]==tour[n]==start.
//
// Complexity: O(n) time, O(n) space.
func MakeTourFromPermutation(perm []int, n int, start int) ([]int, error) {
	if err := ValidatePermutation(perm, n); err != nil {
		return nil, err
	}
	if start < 0 || start >= n {
		return nil, ErrStartOutOfRange
	}

	// Locate start inside perm.
	var (
		i     int
		pivot = -1
	)
	for i = 0; i < n; i++ {
		if perm[i] == start {
			pivot = i
			break
		}
	}
	if pivot == -1 {
		// The permutation does not contain start - inconsistent input shape.
		return nil, ErrDimensionMismatch
	}

	// Rotate into a fresh [n+1] tour and close with start.
	tour := make([]int, n+1)
	for i = 0; i < n; i++ {
		tour[i] = perm[(pivot+i)%n]
	}
	tour[n] = start
	return tour, nil
}

// ValidateTour verifies a closed Hamiltonian cycle over n vertices.
// Implementation:
//   - Stage 1: reject nil and malformed shape.
//   - Stage 2: verify fixed start/end closure.
//   - Stage 3: scan each non-closing vertex exactly once.
//
// Behavior highlights:
//   - Does not allocate beyond the O(n) seen bitmap.
//   - Does not canonicalize; it only validates.
//
// Inputs:
//   - tour: closed cycle of length n+1.
//   - n: vertex count.
//   - start: required first and last vertex.
//
// Returns:
//   - error: nil if the tour is valid.
//
// Errors:
//   - ErrNilTour for nil slices.
//   - ErrInvalidTour for wrong length, wrong closure, duplicate, or out-of-range vertex.
//   - ErrStartOutOfRange for invalid start.
//
// Determinism:
//   - Fixed left-to-right scan.
//
// Complexity:
//   - Time O(n), Space O(n).
//
// AI-Hints:
//   - Do not use ErrDimensionMismatch for semantic tour violations.
func ValidateTour(tour []int, n int, start int) error {
	if tour == nil {
		return ErrNilTour
	}
	if start < 0 || start >= n {
		return ErrStartOutOfRange
	}
	if n <= 0 || len(tour) != n+1 {
		return ErrInvalidTour
	}
	if tour[0] != start || tour[n] != start {
		return ErrInvalidTour
	}

	seen := make([]bool, n)

	var i, v int
	for i = 0; i < n; i++ {
		v = tour[i]
		if v < 0 || v >= n {
			return ErrInvalidTour
		}
		if seen[v] {
			return ErrInvalidTour
		}
		seen[v] = true
	}

	return nil
}

// RotateTourToStart returns a fresh closed tour whose first and last vertex are start.
// It accepts either a closed tour of length n+1 or a raw Hamiltonian path of length n,
// then rotates the cyclic order without changing orientation.
//
// Implementation:
//   - Stage 1: Validate input shape as closed tour or raw path.
//   - Stage 2: Search start in the first n positions.
//   - Stage 3: Copy vertices from start to end, then prefix to start.
//   - Stage 4: Append the closing start vertex.
//
// Behavior highlights:
//   - Does not mutate the input tour.
//   - Preserves cyclic order and orientation.
//   - Normalizes raw paths into closed tours.
//   - Allocates exactly one output slice.
//
// Inputs:
//   - tour: closed cycle len n+1 or raw path len n.
//   - start: local vertex that must become out[0] and out[n].
//
// Returns:
//   - []int: fresh closed tour rotated to start.
//   - error: nil when start is present and shape is valid.
//
// Errors:
//   - ErrDimensionMismatch for empty or malformed tour shape.
//   - ErrInvalidVertex when start does not appear in the first n vertices.
//
// Determinism:
//   - Uses the first occurrence of start in the permutation prefix.
//
// Complexity:
//   - Time O(n), Space O(n).
//
// Notes:
//   - This helper does not validate full permutation uniqueness.
//
// AI-Hints:
//   - Do not rotate over the duplicated closing vertex; search only the first n entries.
func RotateTourToStart(tour []int, start int) ([]int, error) {
	if len(tour) == 0 {
		return nil, ErrDimensionMismatch
	}

	// Determine n (number of unique vertices).
	var (
		n        int
		isClosed bool
	)
	if tour[0] == tour[len(tour)-1] {
		n = len(tour) - 1
		isClosed = true
	} else {
		n = len(tour)
	}
	if start < 0 || start >= n {
		return nil, ErrStartOutOfRange
	}

	// Find start in the first n entries.
	var (
		i     int
		pivot = -1
	)
	for i = 0; i < n; i++ {
		if tour[i] == start {
			pivot = i
			break
		}
	}
	if pivot == -1 {
		return nil, ErrDimensionMismatch
	}

	// Build rotated copy and close it.
	out := make([]int, n+1)

	for i = 0; i < n; i++ {
		out[i] = tour[(pivot+i)%n]
	}
	out[n] = start

	_ = isClosed // documented above; not used in fast path.
	return out, nil
}

// CanonicalizeOrientationInPlace chooses a deterministic direction for a closed tour
// whose start vertex is already fixed. It compares the right neighbor tour[1] and
// left neighbor tour[n-1], then reverses the interior when the reversed orientation
// is lexicographically preferred.
//
// Implementation:
//   - Stage 1: Validate closed-tour shape.
//   - Stage 2: Keep tours with fewer than three distinct vertices unchanged.
//   - Stage 3: Compare the two neighbors adjacent to the fixed start.
//   - Stage 4: Reverse the interior segment [1,n-1] when needed.
//
// Behavior highlights:
//   - Mutates the supplied tour in place.
//   - Keeps tour[0] and tour[n] fixed.
//   - Produces stable output for the same cyclic order.
//   - Does not recompute cost.
//
// Inputs:
//   - tour: closed tour with len n+1 and tour[0]==tour[n].
//
// Returns:
//   - error: nil after orientation is canonicalized.
//
// Errors:
//   - ErrDimensionMismatch for malformed or non-closed tours.
//
// Determinism:
//   - Fixed neighbor comparison under a fixed start.
//
// Complexity:
//   - Time O(n), Space O(1).
//
// Notes:
//   - The permutation part is assumed already valid.
//
// AI-Hints:
//   - Do not reverse the duplicated closing vertex.
func CanonicalizeOrientationInPlace(tour []int) error {
	if len(tour) < 3 {
		return ErrDimensionMismatch
	}
	var n = len(tour) - 1
	if tour[0] != tour[n] {
		return ErrDimensionMismatch
	}
	// Compare right vs left neighbor of start (indices 1 and n-1).
	if tour[1] > tour[n-1] {
		return reverseArcInPlace(tour, 1, n-1)
	}
	return nil
}

// reverseArcInPlace reverses the inclusive interior segment tour[i:k] of a closed tour.
// It preserves the duplicated closing vertex and is the primitive mutation used by
// 2-opt and related local-search moves.
//
// Implementation:
//   - Stage 1: Validate closed-tour shape.
//   - Stage 2: Validate interior arc bounds.
//   - Stage 3: Swap endpoints inward until the segment is reversed.
//
// Behavior highlights:
//   - Mutates tour in place.
//   - Never moves tour[0] or tour[n].
//   - Keeps the tour closed.
//   - Performs no cost lookup.
//
// Inputs:
//   - tour: closed tour with len n+1 and tour[0]==tour[n].
//   - i: first interior index, requiring 1 <= i.
//   - k: last interior index, requiring i < k <= n-1.
//
// Returns:
//   - error: nil after the segment is reversed.
//
// Errors:
//   - ErrDimensionMismatch for malformed tours.
//   - ErrInvalidVertex for invalid arc indices.
//
// Determinism:
//   - Fixed symmetric swaps.
//
// Complexity:
//   - Time O(k-i), Space O(1).
//
// Notes:
//   - Caller is responsible for recomputing or delta-updating tour cost.
//
// AI-Hints:
//   - Do not allow k==n, because that would move the closing duplicate.
func reverseArcInPlace(tour []int, i, k int) error {
	var n = len(tour) - 1
	if n < 2 {
		return ErrDimensionMismatch
	}
	if tour[0] != tour[n] {
		return ErrDimensionMismatch
	}
	if i < 1 || k > n-1 || i >= k {
		return ErrDimensionMismatch
	}
	for i < k {
		tour[i], tour[k] = tour[k], tour[i]
		i++
		k--
	}
	return nil
}

// TourCost computes the total cost of a closed Hamiltonian tour.
// Implementation:
//   - Stage 1: validate nil matrix and nil/short tour inputs.
//   - Stage 2: choose Dense fast-path or generic matrix.Matrix path.
//   - Stage 3: each path validates edge indices and numeric weights.
//
// Behavior highlights:
//   - Does not mutate the tour or matrix.
//   - Dense and generic paths preserve identical sentinel classification.
//
// Inputs:
//   - dist: square distance matrix.
//   - tour: closed cycle of vertex indices.
//
// Returns:
//   - float64: rounded total cost.
//   - error: sentinel-classified failure.
//
// Errors:
//   - ErrNilDistanceMatrix joined with matrix nil sentinels.
//   - ErrNilTour for nil tour.
//   - ErrInvalidTour for malformed non-nil tour.
//   - ErrNonSquare for bad matrix shape.
//   - ErrNaNInf for NaN/-Inf.
//   - ErrIncompleteGraph for +Inf tour edges.
//   - ErrNegativeWeight for negative finite weights.
//
// Determinism:
//   - Fixed tour order scan from index 0 to len(tour)-2.
//
// Complexity:
//   - Time O(len(tour)), Space O(1).
//
// AI-Hints:
//   - Do not classify NaN as ErrDimensionMismatch.
//   - Keep Dense and generic path error classes equivalent.
func TourCost(dist matrix.Matrix, tour []int) (float64, error) {
	if err := matrix.ValidateNotNil(dist); err != nil {
		return 0, errors.Join(ErrNilDistanceMatrix, err)
	}
	if tour == nil {
		return 0, ErrNilTour
	}
	if len(tour) < 2 {
		return 0, ErrInvalidTour
	}

	if d, ok := dist.(*matrix.Dense); ok {
		return tourCostDense(d, tour)
	}

	return tourCostGeneric(dist, tour)
}

// tourCostDense sums directed edge weights along a closed tour using the *matrix.Dense
// fast path. It centralizes strict edge validation while avoiding interface-call overhead
// for the common dense TSP representation.
//
// Implementation:
//   - Stage 1: Validate non-nil dense matrix and tour shape.
//   - Stage 2: Validate every directed edge endpoint.
//   - Stage 3: Read weights directly from the dense matrix.
//   - Stage 4: Reject NaN, infinity, and negative weights.
//   - Stage 5: Accumulate the total cost.
//
// Behavior highlights:
//   - Does not mutate the tour or matrix.
//   - Uses the same sentinel semantics as generic cost evaluation.
//   - Treats ±Inf as incomplete graph edges.
//   - Keeps dense fast path allocation-free.
//
// Inputs:
//   - d: dense weighted adjacency matrix.
//   - tour: closed tour, normally len n+1.
//
// Returns:
//   - float64: total directed cycle cost.
//   - error: nil when all edges are valid.
//
// Errors:
//   - ErrDimensionMismatch for malformed matrix/tour shape.
//   - ErrInvalidVertex for out-of-range endpoints.
//   - ErrIncompleteGraph for infinite weights.
//   - ErrNaNWeight for NaN weights.
//   - ErrNegativeWeight for negative weights.
//
// Determinism:
//   - Fixed left-to-right edge scan.
//
// Complexity:
//   - Time O(n), Space O(1).
//
// Notes:
//   - This function does not enforce symmetry.
//
// AI-Hints:
//   - Prefer this path for *matrix.Dense to avoid interface overhead.
func tourCostDense(d *matrix.Dense, tour []int) (float64, error) {
	// Shape guard.
	var (
		nr = d.Rows()
		nc = d.Cols()
	)
	if nr != nc || nr <= 0 {
		return 0, ErrNonSquare
	}

	// Main accumulation.
	var (
		sum float64
		i   int
		u   int
		v   int
		w   float64
		err error
		n   = nr
		L   = len(tour) - 1 // last index used as closing
	)

	for i = 0; i < L; i++ {
		u = tour[i]
		v = tour[i+1]

		// Index range checks.
		if u < 0 || u >= n || v < 0 || v >= n {
			return 0, ErrInvalidTour
		}

		// Fetch weight and validate.
		w, err = d.At(u, v)
		if err != nil {
			// Dense.At should only fail on OOB; map to shape sentinel.
			return 0, ErrDimensionMismatch
		}
		if math.IsNaN(w) {
			return 0, errors.Join(ErrNaNInf, matrix.ErrNaNInf)
		}
		if math.IsInf(w, 0) {
			return 0, ErrIncompleteGraph
		}
		if w < 0 {
			return 0, ErrNegativeWeight
		}

		sum += w
	}

	return round1e9(sum), nil
}

// tourCostGeneric sums directed edge weights along a closed tour through the matrix.Matrix
// interface. It mirrors tourCostDense validation semantics while supporting any compatible
// matrix backend.
//
// Implementation:
//   - Stage 1: Validate non-nil matrix and tour shape.
//   - Stage 2: Validate every directed edge endpoint.
//   - Stage 3: Read edge weights through At(i,j).
//   - Stage 4: Reject NaN, infinity, and negative weights.
//   - Stage 5: Accumulate the total cost.
//
// Behavior highlights:
//   - Does not mutate the tour or matrix.
//   - Keeps sentinel errors aligned with the dense fast path.
//   - Avoids hidden allocations.
//   - Supports asymmetric directed costs when the chosen algorithm allows them.
//
// Inputs:
//   - m: matrix backend implementing matrix.Matrix.
//   - tour: closed tour, normally len n+1.
//
// Returns:
//   - float64: total directed cycle cost.
//   - error: nil when all edges are valid.
//
// Errors:
//   - ErrDimensionMismatch for malformed matrix/tour shape.
//   - ErrInvalidVertex for out-of-range endpoints.
//   - ErrIncompleteGraph for infinite weights.
//   - ErrNaNWeight for NaN weights.
//   - ErrNegativeWeight for negative weights.
//
// Determinism:
//   - Fixed left-to-right edge scan.
//
// Complexity:
//   - Time O(n), Space O(1).
//
// Notes:
//   - This path has higher call overhead than tourCostDense.
//
// AI-Hints:
//   - Keep this lean; backend-specific acceleration belongs in dedicated fast paths.
func tourCostGeneric(m matrix.Matrix, tour []int) (float64, error) {
	// Shape guard.
	var (
		nr = m.Rows()
		nc = m.Cols()
	)
	if nr != nc || nr <= 0 {
		return 0, ErrNonSquare
	}

	var (
		sum float64
		i   int
		u   int
		v   int
		w   float64
		err error
		n   = nr
		L   = len(tour) - 1
	)

	for i = 0; i < L; i++ {
		u = tour[i]
		v = tour[i+1]

		if u < 0 || u >= n || v < 0 || v >= n {
			return 0, ErrInvalidTour
		}

		w, err = m.At(u, v)
		if err != nil {
			return 0, ErrDimensionMismatch
		}
		if math.IsNaN(w) {
			return 0, errors.Join(ErrNaNInf, matrix.ErrNaNInf)
		}
		if math.IsInf(w, 0) {
			return 0, ErrIncompleteGraph
		}
		if w < 0 {
			return 0, ErrNegativeWeight
		}

		sum += w
	}

	return round1e9(sum), nil
}

// edgeCost fetches and validates one directed edge weight u->v through matrix.Matrix.
// Local-search deltas use this helper so sentinel error behavior remains centralized
// across 2-opt, 3-opt, and candidate move evaluation.
//
// Implementation:
//   - Stage 1: Validate endpoint bounds against matrix dimensions.
//   - Stage 2: Fetch weight through At(u,v).
//   - Stage 3: Reject NaN, infinity, and negative weights.
//   - Stage 4: Return the validated finite cost.
//
// Behavior highlights:
//   - Does not mutate the matrix.
//   - Works for symmetric and asymmetric matrices.
//   - Preserves strict validation even in local-search fast paths.
//
// Inputs:
//   - m: matrix backend.
//   - u: source local vertex.
//   - v: target local vertex.
//
// Returns:
//   - float64: finite non-negative directed edge weight.
//   - error: nil when the edge is valid.
//
// Errors:
//   - ErrDimensionMismatch for nil or malformed matrix.
//   - ErrInvalidVertex for out-of-range endpoints.
//   - ErrIncompleteGraph for infinite weights.
//   - ErrNaNWeight for NaN weights.
//   - ErrNegativeWeight for negative weights.
//
// Determinism:
//   - Pure indexed edge lookup.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This helper intentionally does not check u==v; algorithm-level validation owns that policy.
//
// AI-Hints:
//   - Do not duplicate weight validation in local-search code; call this helper.
func edgeCost(m matrix.Matrix, u, v int) (float64, error) {
	if err := matrix.ValidateNotNil(m); err != nil {
		return 0, errors.Join(ErrNilDistanceMatrix, err)
	}

	nr := m.Rows()
	nc := m.Cols()

	if nr != nc || nr <= 0 {
		return 0, ErrNonSquare
	}
	if u < 0 || u >= nr || v < 0 || v >= nr {
		return 0, ErrInvalidTour
	}

	w, err := m.At(u, v)
	if err != nil {
		return 0, errors.Join(ErrDimensionMismatch, err)
	}
	if math.IsNaN(w) || math.IsInf(w, -1) {
		return 0, errors.Join(ErrNaNInf, matrix.ErrNaNInf)
	}
	if math.IsInf(w, 1) {
		return 0, ErrIncompleteGraph
	}
	if w < 0 {
		return 0, ErrNegativeWeight
	}

	return w, nil
}

// round1e9 returns x rounded to 1e-9 absolute precision.
// This keeps costs stable across platforms without affecting algorithmic correctness.
//
// Complexity: O(1).
func round1e9(x float64) float64 {
	return math.Round(x*roundScale) / roundScale
}
