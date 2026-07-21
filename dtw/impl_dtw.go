// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw implements the canonical scalar Dynamic Time Warping kernel.
// This file contains the single source of truth for the scalar DTW recurrence.
package dtw

import (
	"errors"
	"math"

	"github.com/lvlath/go/matrix"
)

const (
	// emptyPrefixCost is the DP boundary value D(0,0).
	// It represents aligning two empty prefixes before consuming any samples.
	emptyPrefixCost = 0.0
)

// newAccumulatedCostMatrix creates an accumulated-cost DP matrix initialized to +Inf.
//
// Implementation:
//   - Stage 1: allocate matrix.Dense with +Inf distance policy enabled.
//   - Stage 2: fill every cell with +Inf to represent unreachable prefixes.
//   - Stage 3: set D(0,0)=0 as the empty-prefix base case.
//
// Behavior highlights:
//   - The first row D(0,j>0) remains +Inf.
//   - The first column D(i>0,0) remains +Inf.
//   - This directly fixes the historical boundary corruption where row 0 contained zeroes.
//
// Inputs:
//   - rows: number of DP rows, usually len(a)+1.
//   - cols: number of DP columns, usually len(b)+1.
//
// Returns:
//   - *matrix.Dense: initialized accumulated-cost matrix.
//   - error: sentinel-preserving matrix allocation or write failure.
//
// Errors:
//   - ErrBadInput joined with matrix errors from allocation or base-cell write.
//   - ErrNaNInf joined with matrix errors if +Inf fill is rejected unexpectedly.
//
// Determinism:
//   - Initializes cells in row-major slice order.
//
// Complexity:
//   - Time O(rows*cols), Space O(rows*cols).
//
// Notes:
//   - The returned matrix is owned by the caller of this helper.
//   - matrix.WithAllowInfDistances is required because +Inf is a valid unreachable-state sentinel.
//
// AI-Hints:
//   - Do not replace +Inf boundaries with zeroes; that breaks path reconstruction.
//   - Do not allocate [][]float64 here; matrix.Dense is the package-wide numeric artifact.
func newAccumulatedCostMatrix(rows, cols int) (*matrix.Dense, error) {
	dense, err := matrix.NewPreparedDense(rows, cols, matrix.WithAllowInfDistances())
	if err != nil {
		return nil, errors.Join(ErrBadInput, err)
	}

	values := make([]float64, rows*cols)
	for idx := range values {
		values[idx] = math.Inf(1)
	}

	if err = dense.Fill(values); err != nil {
		return nil, errors.Join(ErrNaNInf, err)
	}

	if rows > 0 && cols > 0 {
		if err = dense.Set(0, 0, emptyPrefixCost); err != nil {
			return nil, errors.Join(ErrBadInput, err)
		}
	}

	return dense, nil
}

// compute executes scalar DTW with finalized options.
//
// Implementation:
//   - Stage 1: validate input sequences and numeric policy.
//   - Stage 2: allocate rolling DP rows and optional matrix artifacts once.
//   - Stage 3: fill the DP recurrence in deterministic row-major order.
//   - Stage 4: classify no-path before attempting backtracking.
//   - Stage 5: optionally reconstruct path and publish Result.
//
// Behavior highlights:
//   - Uses +Inf for unreachable DP states.
//   - Window cells outside |i-j| <= window are unreachable.
//   - Path reconstruction is attempted only when the final distance is finite.
//   - Result.Accumulated is published only when returnAccumulated is true.
//
// Inputs:
//   - a: first scalar sequence.
//   - b: second scalar sequence.
//   - cfg: finalized runtime policy from applyOptions or optionsFromLegacy.
//
// Returns:
//   - *Result: canonical result artifact; may be partial with ErrNoPath when path was requested.
//   - error: sentinel-classified failure.
//
// Errors:
//   - ErrNilInput / ErrEmptyInput / ErrNaNInf from validateSequences.
//   - ErrNegativeCost / ErrNaNInf from validateLocalCost.
//   - ErrNoPath when path was requested and the final state is unreachable.
//   - ErrIncompletePath if a finite final state cannot be reconstructed.
//
// Determinism:
//   - DP loops are row-major: i ascending, j ascending.
//   - Local costs are requested in the same row-major order.
//   - Path tie-break is handled by backtrackDense.
//
// Complexity:
//   - Time O(n*m), where n=len(a), m=len(b).
//   - Space O(m) for rolling rows.
//   - Additional O(n*m) if FullMatrix/path/accumulated/local-cost artifacts are requested.
//
// Notes:
//   - This P0 kernel preserves original sequence orientation.
//   - O(min(n,m)) rolling orientation is intentionally postponed until symmetric-cost policy is explicit.
//
// AI-Hints:
//   - Do not backtrack when Result.Reachable is false.
//   - Do not publish internal accumulated state unless returnAccumulated is true.
func compute(a, b []float64, cfg options) (*Result, error) {
	if err := validateSequences(a, b, cfg.validateFinite); err != nil {
		return nil, err
	}

	n := len(a)
	m := len(b)
	inf := math.Inf(1)

	prevRow := make([]float64, m+1)
	currRow := make([]float64, m+1)

	for j := 1; j <= m; j++ {
		prevRow[j] = inf
	}

	needAccumulated := cfg.memoryMode == FullMatrix || cfg.returnAccumulated || cfg.returnPath

	var accumulated *matrix.Dense
	var err error

	if needAccumulated {
		accumulated, err = newAccumulatedCostMatrix(n+1, m+1)
		if err != nil {
			return nil, err
		}
	}

	var localCostMatrix *matrix.Dense

	if cfg.returnLocalCost {
		localCostMatrix, err = matrix.NewPreparedDense(n, m)
		if err != nil {
			return nil, errors.Join(ErrBadInput, err)
		}
	}

	for i := 1; i <= n; i++ {
		currRow[0] = inf

		for j := 1; j <= m; j++ {
			if cfg.window >= 0 && absInt(i-j) > cfg.window {
				currRow[j] = inf

				if accumulated != nil {
					if err = accumulated.Set(i, j, inf); err != nil {
						return nil, errors.Join(ErrNaNInf, err)
					}
				}

				continue
			}

			localCost, err := cfg.costFunc(a[i-1], b[j-1])
			if err != nil {
				return nil, err
			}

			if err = validateLocalCost(localCost); err != nil {
				return nil, err
			}

			if localCostMatrix != nil {
				if err = localCostMatrix.Set(i-1, j-1, localCost); err != nil {
					return nil, errors.Join(ErrNaNInf, err)
				}
			}

			matchCost := prevRow[j-1]
			insertCost := prevRow[j] + cfg.slopePenalty
			deleteCost := currRow[j-1] + cfg.slopePenalty
			bestPrevCost := min3Stable(matchCost, insertCost, deleteCost)

			currRow[j] = localCost + bestPrevCost

			if accumulated != nil {
				if err = accumulated.Set(i, j, currRow[j]); err != nil {
					return nil, errors.Join(ErrNaNInf, err)
				}
			}
		}

		prevRow, currRow = currRow, prevRow
	}

	distance := prevRow[m]

	res := &Result{
		Distance:     distance,
		Reachable:    !math.IsInf(distance, 1),
		PathTracked:  cfg.returnPath,
		Window:       cfg.window,
		SlopePenalty: cfg.slopePenalty,
		MemoryMode:   cfg.memoryMode,
		LocalCost:    localCostMatrix,
	}

	if cfg.returnAccumulated {
		res.Accumulated = accumulated
	}

	if !res.Reachable {
		if cfg.returnPath {
			return res, ErrNoPath
		}

		return res, nil
	}

	if cfg.returnPath {
		path, err := backtrackDense(accumulated, a, b, cfg)
		if err != nil {
			return res, err
		}

		res.Path = path
	}

	return res, nil
}

// backtrackDense reconstructs one deterministic optimal DTW path.
//
// Implementation:
//   - Stage 1: verify accumulated matrix availability.
//   - Stage 2: walk from D(n,m) to D(0,0).
//   - Stage 3: compare predecessor candidates using the same local cost and slope penalty.
//   - Stage 4: reverse the collected end-to-start path.
//
// Behavior highlights:
//   - Backtracking starts only after compute confirms a finite final distance.
//   - Tie-break order is diagonal, then vertical/up, then horizontal/left.
//   - The returned path is start-to-end and caller-owned.
//
// Inputs:
//   - accumulated: full accumulated-cost matrix.
//   - a: first scalar sequence.
//   - b: second scalar sequence.
//   - cfg: finalized runtime policy.
//
// Returns:
//   - Path: deterministic representative optimal path.
//
// Errors:
//   - ErrPathNeedsMatrix when accumulated is nil.
//   - ErrIncompletePath when predecessor reconstruction fails.
//   - ErrNaNInf / ErrNegativeCost from local-cost validation.
//   - ErrBadInput joined with matrix access errors.
//
// Determinism:
//   - Fixed predecessor order: diagonal -> up -> left.
//   - No map iteration or nondeterministic source is used.
//
// Complexity:
//   - Time O(k), Space O(k), where k is path length and k <= n+m-1.
//
// Notes:
//   - This function reconstructs one path, not all optimal paths.
//
// AI-Hints:
//   - Do not change tie-break order without updating doc.go and regression tests.
//   - Do not call this function for +Inf final distances.
func backtrackDense(accumulated *matrix.Dense, a, b []float64, cfg options) (Path, error) {
	if accumulated == nil {
		return nil, ErrPathNeedsMatrix
	}

	i := len(a)
	j := len(b)
	path := make(Path, 0, i+j)

	for i > 0 || j > 0 {
		if i == 0 || j == 0 {
			return nil, ErrIncompletePath
		}

		path = append(path, Coord{I: i - 1, J: j - 1})

		currentCost, err := accumulated.At(i, j)
		if err != nil {
			return nil, errors.Join(ErrBadInput, err)
		}

		localCost, err := cfg.costFunc(a[i-1], b[j-1])
		if err != nil {
			return nil, err
		}

		if err = validateLocalCost(localCost); err != nil {
			return nil, err
		}

		predecessorTarget := currentCost - localCost

		diagonalCost, err := accumulated.At(i-1, j-1)
		if err != nil {
			return nil, errors.Join(ErrBadInput, err)
		}

		if equalCost(predecessorTarget, diagonalCost, cfg.tieEpsilon) {
			i--
			j--
			continue
		}

		upCost, err := accumulated.At(i-1, j)
		if err != nil {
			return nil, errors.Join(ErrBadInput, err)
		}

		if equalCost(predecessorTarget, upCost+cfg.slopePenalty, cfg.tieEpsilon) {
			i--
			continue
		}

		leftCost, err := accumulated.At(i, j-1)
		if err != nil {
			return nil, errors.Join(ErrBadInput, err)
		}

		if equalCost(predecessorTarget, leftCost+cfg.slopePenalty, cfg.tieEpsilon) {
			j--
			continue
		}

		return nil, ErrIncompletePath
	}

	reversePath(path)

	return path, nil
}

// reversePath reverses a DTW path in place.
//
// Implementation:
//   - Stage 1: use two cursors at both ends.
//   - Stage 2: swap until cursors meet.
//
// Behavior highlights:
//   - Backtracking collects end-to-start order.
//   - Public Result.Path must be start-to-end order.
//
// Inputs:
//   - path: mutable path slice.
//
// Returns:
//   - None; path is modified in place.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pure deterministic slice operation.
//
// Complexity:
//   - Time O(k), Space O(1), where k=len(path).
//
// AI-Hints:
//   - Do not allocate another path here; backtracking already owns this slice.
func reversePath(path Path) {
	for left, right := 0, len(path)-1; left < right; left, right = left+1, right-1 {
		path[left], path[right] = path[right], path[left]
	}
}

// min3Stable returns the smallest predecessor cost.
//
// Implementation:
//   - Stage 1: start with diagonal/match cost.
//   - Stage 2: accept insert cost only when strictly smaller.
//   - Stage 3: accept delete cost only when strictly smaller.
//
// Behavior highlights:
//   - Ties keep the earlier candidate.
//   - Forward distance is unaffected by tie choice, but stable behavior prevents accidental drift.
//
// Inputs:
//   - matchCost: D(i-1,j-1).
//   - insertCost: D(i-1,j)+penalty.
//   - deleteCost: D(i,j-1)+penalty.
//
// Returns:
//   - float64: smallest predecessor cost.
//
// Errors:
//   - None.
//
// Determinism:
//   - Stable strict-less comparison order: match -> insert -> delete.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// AI-Hints:
//   - Keep this stable; do not replace with math.Min chains if tie semantics matter later.
func min3Stable(matchCost, insertCost, deleteCost float64) float64 {
	bestCost := matchCost

	if insertCost < bestCost {
		bestCost = insertCost
	}

	if deleteCost < bestCost {
		bestCost = deleteCost
	}

	return bestCost
}

// absInt returns the absolute value of an int.
//
// Implementation:
//   - Stage 1: check the sign.
//   - Stage 2: negate only negative values.
//
// Behavior highlights:
//   - Used for Sakoe-Chiba band checks.
//
// Inputs:
//   - x: integer value.
//
// Returns:
//   - int: |x|.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pure arithmetic.
//
// Complexity:
//   - Time O(1), Space O(1).
func absInt(x int) int {
	if x < 0 {
		return -x
	}

	return x
}

// equalCost compares accumulated costs for deterministic backtracking.
//
// Implementation:
//   - Stage 1: reject NaN comparisons.
//   - Stage 2: compare matching infinities exactly.
//   - Stage 3: compare finite values using epsilon.
//
// Behavior highlights:
//   - +Inf equals +Inf and -Inf equals -Inf.
//   - NaN never equals anything.
//   - Finite comparison uses absolute tolerance only.
//
// Inputs:
//   - a: first cost.
//   - b: second cost.
//   - epsilon: finite tolerance.
//
// Returns:
//   - bool: true when costs are equal under backtracking comparison rules.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pure numeric comparison.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - No-path cases should be classified before backtracking, so +Inf equality is defensive.
//
// AI-Hints:
//   - Do not use math.Abs(+Inf-+Inf); it produces NaN and breaks equality.
func equalCost(a, b, epsilon float64) bool {
	if math.IsNaN(a) || math.IsNaN(b) {
		return false
	}

	if math.IsInf(a, 1) || math.IsInf(b, 1) {
		return math.IsInf(a, 1) && math.IsInf(b, 1)
	}

	if math.IsInf(a, -1) || math.IsInf(b, -1) {
		return math.IsInf(a, -1) && math.IsInf(b, -1)
	}

	return math.Abs(a-b) <= epsilon
}
