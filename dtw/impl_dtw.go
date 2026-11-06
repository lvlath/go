// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw implements the scalar Dynamic Time Warping adapter.
// This file maintains the accumulated-matrix boundary contract and scalar delegation.
package dtw

import (
	"errors"
	"math"

	"github.com/lvlath/go/matrix"
)

const (
	// emptyPrefixCost is the DP boundary value D(0,0), before either sequence is consumed.
	emptyPrefixCost = 0.0
)

// newAccumulatedCostMatrix allocates the full public accumulated-cost artifact.
// Its first row and column are structurally unreachable apart from D(0,0),
// while interior +Inf cells may represent either window absence or numeric overflow.
//
// Implementation:
//   - Stage 1: allocate a Dense matrix allowing +Inf in the distance domain.
//   - Stage 2: initialize all positions to +Inf in row-major order.
//   - Stage 3: set the empty-prefix origin to finite zero.
//
// Behavior highlights:
//   - The numeric artifact alone does not carry private reachability classes.
//
// Inputs:
//   - rows, cols: positive accumulated-matrix dimensions, including one boundary row/column.
//
// Returns:
//   - *matrix.Dense: allocated and initialized caller-owned accumulated storage.
//   - error: sentinel-preserving allocation or initialization failure.
//
// Errors:
//   - ErrBadInput: matrix allocation or boundary write failure.
//   - ErrNaNInf: an unexpected failure to store the +Inf distance sentinel.
//
// Determinism:
//   - Initializes every cell row-major and updates origin once.
//
// Complexity:
//   - Time O(rows*cols), Space O(rows*cols).
//
// Notes:
//   - Only returned when accumulated storage was explicitly requested by policy.
//   - The internal caller may allocate this for FullMatrix even without publication.
//
// AI-Hints:
//   - Use matrix.WithAllowInfDistances: +Inf is a DP representation state here.
//   - Do not assign reachability by inspecting accumulated cells without class metadata.
func newAccumulatedCostMatrix(rows, cols int) (*matrix.Dense, error) {
	dense, err := matrix.NewPreparedDense(rows, cols, matrix.WithAllowInfDistances())
	if err != nil {
		return nil, errors.Join(ErrBadInput, err)
	}

	values := make([]float64, rows*cols)
	for index := range values {
		values[index] = math.Inf(1)
	}
	if err = dense.Fill(values); err != nil {
		return nil, errors.Join(ErrNaNInf, err)
	}
	if err = dense.Set(0, 0, emptyPrefixCost); err != nil {
		return nil, errors.Join(ErrBadInput, err)
	}
	return dense, nil
}

// compute validates two scalar sequences and runs the shared classified DP kernel.
// It preserves absolute-cost defaults, callback ownership, and the legacy
// option assembly performed by the public Align/DTW facades.
//
// Implementation:
//   - Stage 1: validate nil, empty, and finite-input policy in legacy order.
//   - Stage 2: adapt each zero-based DP cell to the configured local CostFunc.
//   - Stage 3: delegate to the unique classified recurrence and publish its result.
//
// Behavior highlights:
//   - Without LocalCost artifact the callback is evaluated only inside the band.
//   - With LocalCost artifact it is evaluated once for every cell in row-major order.
//
// Inputs:
//   - a, b: non-nil scalar sequences, read-only for the duration of the call.
//   - cfg: finalized options with finite, non-negative penalty and valid memory mode.
//
// Returns:
//   - *Result: detached finite alignment or documented structural no-path state.
//   - error: sentinel-classified input, local, overflow, or witness error.
//
// Errors:
//   - ErrNilInput / ErrEmptyInput / ErrNaNInf: invalid sequences.
//   - ErrNegativeCost / ErrNaNInf: invalid callback return.
//   - ErrOverflow joined with ErrNaNInf: terminal optimal cost is unrepresentable.
//   - ErrNoPath: structural absence when a path was requested.
//
// Determinism:
//   - Scalar callbacks are called row-major over published or admitted cells.
//   - The shared kernel chooses diagonal, up, then left on strict ties.
//
// Complexity:
//   - Time O(n*m), rolling state O(m), path/explicit artifacts O(n*m).
//
// Notes:
//   - Concurrent modification of a or b is unsupported; no input data is mutated.
//   - An arbitrary custom CostFunc must be pure and deterministic for repeatable results.
//
// AI-Hints:
//   - Never call CostFunc again during path reconstruction.
//   - Do not replace the scalar default AbsoluteCost with squared L2.
func compute(a, b []float64, cfg options) (*Result, error) {
	if err := validateSequences(a, b, cfg.validateFinite); err != nil {
		return nil, err
	}

	local := func(i, j int) (float64, error) {
		return cfg.costFunc(a[i], b[j])
	}
	return computeFromCostSource(len(a), len(b), cfg, local)
}

// reversePath changes a private backward witness to the public start-to-end order.
// The operation is an in-place reversal of a newly allocated private path,
// without touching input sequences or any caller-owned slice.
//
// Implementation:
//   - Stage 1: place cursors at the first and last coordinate.
//   - Stage 2: exchange coordinates while the cursors have not crossed.
//   - Stage 3: return with the same allocated backing storage.
//
// Behavior highlights:
//   - Empty and singleton paths remain unchanged.
//
// Inputs:
//   - path: mutable, exclusively owned reverse-ordered coordinate slice.
//
// Returns:
//   - None; path becomes forward-ordered in place.
//
// Errors:
//   - None.
//
// Determinism:
//   - Symmetric left-to-right swaps, independent of path values.
//
// Complexity:
//   - Time O(len(path)), Space O(1).
//
// Notes:
//   - The path must not alias an externally published Result.Path during mutation.
//
// AI-Hints:
//   - Do not allocate a second path slice just to reverse the first.
//   - Reversal is unrelated to tie selection, which occurs only in forward DP.
func reversePath(path Path) {
	for left, right := 0, len(path)-1; left < right; left, right = left+1, right-1 {
		path[left], path[right] = path[right], path[left]
	}
}

// absInt returns the absolute value of a bounded index difference.
// Its caller passes differences of positive DP indices, not arbitrary MinInt,
// so negation is safe under the validated sequence-size contract.
//
// Implementation:
//   - Stage 1: detect a negative index difference.
//   - Stage 2: negate it if needed, otherwise return unchanged.
//   - Stage 3: publish the non-negative magnitude.
//
// Behavior highlights:
//   - Used only for Sakoe-Chiba band admission checks.
//
// Inputs:
//   - value: difference between two admissible DP indices.
//
// Returns:
//   - int: absolute index difference in time-step units.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pure integer arithmetic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - MinInt is outside this private helper's call domain.
//
// AI-Hints:
//   - Window=0 means strict diagonal, not an unbounded band.
//   - Window=-1 is the one supported unconstrained sentinel.
func absInt(value int) int {
	if value < 0 {
		return -value
	}
	return value
}
