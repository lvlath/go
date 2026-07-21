// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw computes DTW over precomputed local-cost grids.
// This file shares recurrence semantics with scalar DTW while using flat local costs.
package dtw

import (
	"errors"
	"math"

	"github.com/lvlath/go/matrix"
)

// computeFromFlatLocal executes DTW over row-major precomputed local costs.
//
// Implementation:
//   - Stage 1: validate flat cost shape.
//   - Stage 2: allocate rolling rows and optional accumulated matrix.
//   - Stage 3: run the same row-major recurrence as scalar DTW.
//   - Stage 4: classify no-path before backtracking.
//   - Stage 5: publish Result.
//
// Behavior highlights:
//   - localCosts[i*cols+j] is the local cost for aligning i with j.
//   - CostFunc is not used.
//   - Window and SlopePenalty semantics are identical to scalar Align.
//   - Result.LocalCost is attached by the caller facade when requested.
//
// Inputs:
//   - localCosts: row-major local-cost values.
//   - rows: number of sequence-A steps.
//   - cols: number of sequence-B steps.
//   - cfg: finalized DTW policy.
//
// Returns:
//   - *Result: canonical result artifact.
//   - error: sentinel-classified failure.
//
// Errors:
//   - ErrEmptyInput when rows or cols are zero.
//   - ErrBadInput when len(localCosts) != rows*cols.
//   - ErrNaNInf / ErrNegativeCost when localCosts contain invalid values.
//   - ErrNoPath when path tracking is requested and final state is unreachable.
//   - ErrIncompletePath if path reconstruction fails unexpectedly.
//
// Determinism:
//   - DP traversal is row-major.
//   - Backtracking tie-break is diagonal, then vertical/up, then horizontal/left.
//
// Complexity:
//   - Time O(rows*cols), Space O(cols) for distance-only.
//   - Additional O(rows*cols) when FullMatrix/path/accumulated output is required.
//
// Notes:
//   - This kernel intentionally receives a flat slice to keep the hot loop free of Matrix.At calls.
//
// AI-Hints:
//   - Keep recurrence semantics identical to scalar compute.
//   - Do not apply CostFunc here; the local-cost matrix is already the cost source.
func computeFromFlatLocal(localCosts []float64, rows, cols int, cfg options) (*Result, error) {
	if rows == 0 || cols == 0 {
		return nil, ErrEmptyInput
	}

	if len(localCosts) != rows*cols {
		return nil, ErrBadInput
	}

	for _, value := range localCosts {
		if err := validateLocalCost(value); err != nil {
			return nil, err
		}
	}

	inf := math.Inf(1)

	prevRow := make([]float64, cols+1)
	currRow := make([]float64, cols+1)

	for j := 1; j <= cols; j++ {
		prevRow[j] = inf
	}

	needAccumulated := cfg.memoryMode == FullMatrix || cfg.returnAccumulated || cfg.returnPath

	var accumulated *matrix.Dense
	var err error

	if needAccumulated {
		accumulated, err = newAccumulatedCostMatrix(rows+1, cols+1)
		if err != nil {
			return nil, err
		}
	}

	for i := 1; i <= rows; i++ {
		currRow[0] = inf

		for j := 1; j <= cols; j++ {
			if cfg.window >= 0 && absInt(i-j) > cfg.window {
				currRow[j] = inf

				if accumulated != nil {
					if err = accumulated.Set(i, j, inf); err != nil {
						return nil, errors.Join(ErrNaNInf, err)
					}
				}

				continue
			}

			localCost := localCosts[(i-1)*cols+(j-1)]
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

	distance := prevRow[cols]

	res := &Result{
		Distance:     distance,
		Reachable:    !math.IsInf(distance, 1),
		PathTracked:  cfg.returnPath,
		Window:       cfg.window,
		SlopePenalty: cfg.slopePenalty,
		MemoryMode:   cfg.memoryMode,
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
		path, err := backtrackFlatLocal(accumulated, localCosts, rows, cols, cfg)
		if err != nil {
			return res, err
		}

		res.Path = path
	}

	return res, nil
}

// backtrackFlatLocal reconstructs a DTW path over precomputed local costs.
//
// Implementation:
//   - Stage 1: verify accumulated matrix availability.
//   - Stage 2: walk backward from (rows,cols) to (0,0).
//   - Stage 3: use localCosts for predecessor comparisons.
//   - Stage 4: reverse the path into start-to-end order.
//
// Behavior highlights:
//   - Tie-break order is diagonal, then vertical/up, then horizontal/left.
//   - The result is one deterministic representative optimal path.
//
// Inputs:
//   - accumulated: full accumulated-cost matrix.
//   - localCosts: row-major local-cost grid.
//   - rows: number of sequence-A steps.
//   - cols: number of sequence-B steps.
//   - cfg: finalized DTW policy.
//
// Returns:
//   - Path: deterministic start-to-end path.
//
// Errors:
//   - ErrPathNeedsMatrix when accumulated is nil.
//   - ErrIncompletePath when no valid predecessor explains the current cell.
//   - ErrBadInput joined with matrix access errors.
//
// Determinism:
//   - Fixed predecessor order: diagonal -> up -> left.
//
// Complexity:
//   - Time O(k), Space O(k), where k is path length.
//
// AI-Hints:
//   - Do not call CostFunc here; localCosts already define cell costs.
func backtrackFlatLocal(accumulated *matrix.Dense, localCosts []float64, rows, cols int, cfg options) (Path, error) {
	if accumulated == nil {
		return nil, ErrPathNeedsMatrix
	}

	i := rows
	j := cols
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

		localCost := localCosts[(i-1)*cols+(j-1)]
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
