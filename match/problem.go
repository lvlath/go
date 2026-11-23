// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package match builds the numerical local perfect-matching problem.
// The engine uses local dense indices while output maps back to original IDs.
package match

import (
	"errors"
	"math"

	"github.com/lvlath/go/matrix"
)

// matchingUnmatched is the historical sentinel for an unpaired local position.
const matchingUnmatched = -1

// matchingProblem is the detached k*k graph consumed by weighted Blossom.
// odd[local] maps each dense local position to its original matrix index.
type matchingProblem struct {
	odd []int
	n   int
	w   []float64
}

// at returns a stored cost using only validated local matching indices.
// Direction is preserved for accepted near-symmetric source weights.
// Implementation:
//   - Stage 1: Index row-major storage at i*n+j.
//
// Behavior highlights:
//   - No matrix.At call, allocation, or source aliasing.
//
// Inputs:
//   - i, j: validated local indices [0,n).
//
// Returns:
//   - float64: unrounded original directional cost.
//
// Errors:
//   - None; invalid internal indices violate caller preconditions.
//
// Determinism:
//   - O(1) direct lookup.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Never pass global original indices directly.
//
// AI-Hints:
//   - Mapping lives in odd, not in the index expression.
func (p matchingProblem) at(i, j int) float64 { return p.w[i*p.n+j] }

// buildMatchingProblem constructs a complete local graph in original subset order.
// The full source domain is checked and copied before subset validation.
// Implementation:
//   - Stage 1: Reject odd cardinality and handle the empty fast path.
//   - Stage 2: Validate and copy all source weights.
//   - Stage 3: Validate subset indices and copy ordered k*k local weights.
//
// Behavior highlights:
//   - Inherited n*n then k*k preparation and direction preservation.
//
// Inputs:
//   - odd: ordered original indices; dist: stable caller-owned source matrix.
//
// Returns:
//   - matchingProblem: detached, complete, ordered local matching graph.
//
// Errors:
//   - ErrOddVertices, ErrInvalidVertices, input domain errors, ErrSizeOverflow.
//
// Determinism:
//   - Matrix row-major scan, subset input order, local row-major copy.
//
// Complexity:
//   - Time O(n^2+k^2), Space O(n^2+k^2).
//
// Notes:
//   - This builder does not check triangle inequality or mutate adjacency.
//
// AI-Hints:
//   - Do not optimize away full-matrix validation as part of extraction.
func buildMatchingProblem(odd []int, dist matrix.Matrix) (matchingProblem, error) {
	k := len(odd)
	if k == 0 {
		return matchingProblem{odd: []int{}, n: 0}, nil
	}
	if k&1 != 0 {
		return matchingProblem{}, errors.Join(ErrOddVertices, ErrInvalidMatching)
	}
	weights, err := copyCompleteWeights(dist, true)
	if err != nil {
		return matchingProblem{}, err
	}
	localOdd, err := validateSubset(odd, weights.n)
	if err != nil {
		return matchingProblem{}, errors.Join(ErrInvalidMatching, err)
	}
	count, err := checkedSquareLength(k)
	if err != nil {
		return matchingProblem{}, err
	}
	problem := matchingProblem{
		odd: localOdd,
		n:   k,
		w:   make([]float64, count),
	}
	for i := 0; i < k; i++ {
		for j := 0; j < k; j++ {
			if i == j {
				continue
			}
			problem.w[i*k+j] = weights.at(localOdd[i], localOdd[j])
		}
	}
	return problem, nil
}

// verifyPerfectMatching checks an entire reciprocal local mate relation.
// It does not trust a completed solver just because it returned no error.
// Implementation:
//   - Stage 1: Verify even cardinality.
//   - Stage 2: Check every index, absence of self-pairs and reciprocity.
//
// Behavior highlights:
//   - Empty matching is valid.
//
// Inputs:
//   - mates: one partner index per local vertex.
//
// Returns:
//   - error: nil only for a complete perfect-matching relation.
//
// Errors:
//   - ErrInvalidMatching on any structural violation.
//
// Determinism:
//   - Ascending local index scan.
//
// Complexity:
//   - Time O(k), Space O(1).
//
// Notes:
//   - This is structural validation, not an independent optimum oracle.
//
// AI-Hints:
//   - Check mate[mate[i]] before ever exporting pairs.
func verifyPerfectMatching(mates []int) error {
	if len(mates)&1 != 0 {
		return ErrInvalidMatching
	}
	for i, j := range mates {
		if j < 0 || j >= len(mates) || i == j {
			return ErrInvalidMatching
		}
		if mates[j] != i {
			return ErrInvalidMatching
		}
	}
	return nil
}

// matchingCost returns the checked raw cost of an entire local mate relation.
// It accumulates only i<mate[i] directional original weights, without rounding.
// Implementation:
//   - Stage 1: Verify reciprocity, shape and all indices.
//   - Stage 2: Sum selected edges in increasing local index order.
//   - Stage 3: Reject any non-finite arithmetic result.
//
// Behavior highlights:
//   - Does not use transformed Blossom profits or display rounding.
//
// Inputs:
//   - problem: detached k*k costs; mates: complete local partners.
//
// Returns:
//   - float64: representable, unrounded matching objective.
//
// Errors:
//   - ErrInvalidMatching for structure, ErrNumericOverflow for finite-sum overflow.
//
// Determinism:
//   - Increasing local i with i<mates[i].
//
// Complexity:
//   - Time O(k), Space O(1).
//
// Notes:
//   - The result is a float64 sequential sum, not exact real arithmetic.
//
// AI-Hints:
//   - Do not replace this with round1e9 or re-read caller matrix data.
func matchingCost(problem matchingProblem, mates []int) (float64, error) {
	if err := verifyPerfectMatching(mates); err != nil {
		return 0, err
	}
	if problem.n != len(mates) || len(problem.odd) != len(mates) {
		return 0, ErrInvalidMatching
	}
	if size, err := checkedSquareLength(problem.n); err != nil || len(problem.w) != size {
		return 0, ErrInvalidMatching
	}
	total := 0.0
	for i, j := range mates {
		if i >= j {
			continue
		}
		weight := problem.at(i, j)
		if math.IsNaN(weight) || math.IsInf(weight, 0) || weight < 0 {
			return 0, ErrInvalidMatching
		}
		total += weight
		if math.IsNaN(total) || math.IsInf(total, 0) {
			return 0, ErrNumericOverflow
		}
	}
	return total, nil
}
