// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix completes deficient singular-vector bases deterministically.
// This file also fixes the externally observable sign convention of economy SVD.
package matrix

import (
	"context"
	"math"
)

// svdCompleteZeroColumns constructs left singular vectors for exactly zero modes.
// The orthogonal complement is selected deterministically from canonical vectors
// e_0, e_1, ... and twice reorthogonalized against every accepted nonzero mode.
//
// Implementation:
//   - Stage 1: visit missing columns in increasing singular-mode order.
//   - Stage 2: try each standard basis candidate, twice removing existing projections.
//   - Stage 3: normalize the first numerically independent residual and mark it complete.
//
// Behavior highlights:
//   - Nonzero singular modes, including those below the rank cutoff, are never replaced.
//   - Candidate residual acceptance scales with sqrt(rows), not an arbitrary 0.5.
//
// Inputs:
//   - ctx: non-nil context, checked before each missing-column completion.
//   - u: private row-major rows-by-cols left singular-vector matrix.
//   - nonzero: length cols; true for every already constructed nonzero vector.
//
// Returns: nil after an orthonormal completion, or an error without publication.
// Errors: ErrNotConverged if no resolvable orthogonal complement exists; ctx.Err() on cancel.
// Determinism: missing columns, candidate e_i, orthogonalization passes, and
// existing vectors are visited in increasing index order.
// Complexity: Time O(rows*cols*cols), Space O(rows) for residual-score caching.
// Notes: Caller controls input ownership; only exact-zero modes are completed.
// AI-Hints: An absolute residual threshold such as 0.5 fails when the missing
// complement has its mass distributed over many row coordinates.
func svdCompleteZeroColumns(ctx context.Context, u []float64, rows, cols int, nonzero []bool) error {
	// An orthonormal basis projects each e_i onto itself by 1-sum(u[i,j]^2).
	// Cache these row residual scores to avoid O(rows) Gram-Schmidt attempts
	// for every missing mode. The threshold is far above binary64 roundoff.
	minimumResidual := svdBasisSafetyMultiplier * math.Sqrt(svdMachineEpsilon*float64(rows))
	minimumResidualSquared := minimumResidual * minimumResidual
	residualSquares := make([]float64, rows)
	for row := 0; row < rows; row++ {
		remaining := 1.0
		for column := 0; column < cols; column++ {
			if nonzero[column] {
				value := u[row*cols+column]
				remaining = math.FMA(-value, value, remaining)
			}
		}
		residualSquares[row] = math.Max(0, remaining)
	}

	for column := 0; column < cols; column++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		if nonzero[column] {
			continue
		}
		// The first numerically independent e_i preserves the canonical
		// coordinate scan. Computing its estimated residual is O(1) per row.
		candidate := -1
		for row := 0; row < rows; row++ {
			if residualSquares[row] > minimumResidualSquared {
				candidate = row
				break
			}
		}
		if candidate < 0 {
			return ErrNotConverged
		}
		for row := 0; row < rows; row++ {
			u[row*cols+column] = 0
		}
		u[candidate*cols+column] = 1

		// Twice-modified Gram-Schmidt removes projections onto all earlier
		// nonzero and completed columns, with fixed, auditable traversal.
		for pass := 0; pass < 2; pass++ {
			for accepted := 0; accepted < cols; accepted++ {
				if !nonzero[accepted] || accepted == column {
					continue
				}
				projection := 0.0
				for row := 0; row < rows; row++ {
					projection = math.FMA(u[row*cols+column], u[row*cols+accepted], projection)
				}
				for row := 0; row < rows; row++ {
					index := row*cols + column
					u[index] = math.FMA(-projection, u[row*cols+accepted], u[index])
				}
			}
		}
		length := svdColumnNorm(u, rows, cols, column)
		if length <= minimumResidual {
			// The row-projection estimate cannot certify a stable basis vector.
			// Do not publish a vector magnified from roundoff-sized residue.
			return ErrNotConverged
		}
		for row := 0; row < rows; row++ {
			u[row*cols+column] /= length
		}
		nonzero[column] = true
		// Update the remaining standard-vector residuals in one row scan.
		// Subsequent completions will never reselect this same pivot direction.
		for row := 0; row < rows; row++ {
			value := u[row*cols+column]
			residualSquares[row] = math.Max(0, math.FMA(-value, value, residualSquares[row]))
		}
	}
	return nil
}

// svdCanonicalSigns makes the maximum-magnitude component of each U column positive.
// When several components have equal magnitudes, the first row wins, and the
// paired row of VT is flipped to preserve the same reconstructed matrix.
//
// Implementation:
//   - Stage 1: scan each left column for its first maximum absolute component.
//   - Stage 2: flip all entries of that column only when the pivot is negative.
//   - Stage 3: apply the same sign change to the corresponding right row.
//
// Inputs: caller-owned finite U and VT row-major buffers with economy shapes.
// Returns: No value; both buffers are modified in place and remain detached.
// Errors: None after validated decomposition and dimensions.
// Determinism: Increasing mode then row; strict > comparison preserves first tie.
// Complexity: Time O((rows+n)*k), Space O(1).
// AI-Hints: Never flip U alone: both singular-vector factors must preserve A.
func svdCanonicalSigns(u, vt []float64, rows, k, n int) {
	for column := 0; column < k; column++ {
		pivot, maximum := 0, 0.0
		for row := 0; row < rows; row++ {
			magnitude := math.Abs(u[row*k+column])
			if magnitude > maximum {
				pivot, maximum = row, magnitude
			}
		}
		if maximum == 0 || !math.Signbit(u[pivot*k+column]) {
			continue
		}
		for row := 0; row < rows; row++ {
			u[row*k+column] = -u[row*k+column]
		}
		for col := 0; col < n; col++ {
			vt[column*n+col] = -vt[column*n+col]
		}
	}
}
