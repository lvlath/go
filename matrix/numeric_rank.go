// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix measures numerical rank and 1-norm conditioning of packed factors.
// This file implements diagnostics using actual triangular solves, never fake ratios.
package matrix

import (
	"context"
	"math"
)

// factorNumericalRank counts diagonals above an explicit relative threshold.
// The threshold is relative to the largest absolute diagonal and equality
// belongs to the deficient class; exactly zero always means singularity.
// Implementation:
//   - Stage 1: collect the largest absolute diagonal using a fixed index order.
//   - Stage 2: compare each diagonal to tolerance*maximum without underflow.
//   - Stage 3: report the number of surviving pivots, never a guessed exact rank.
//
// Errors: No errors; inputs are already validated finite packed factors.
// Complexity: Time O(n), Space O(1).
func factorNumericalRank(packed []float64, n, stride int, tolerance float64) int {
	largest := 0.0
	for j := 0; j < n; j++ {
		if magnitude := math.Abs(packed[j*stride+j]); magnitude > largest {
			largest = magnitude
		}
	}
	if largest == 0 {
		return 0
	}
	rank := 0
	for j := 0; j < n; j++ {
		// Compare dimensionless ratios: tolerance*largest can underflow if
		// the diagonal itself is extremely small, changing the rank law.
		if math.Abs(packed[j*stride+j])/largest > tolerance {
			rank++
		}
	}
	return rank
}

// factorOneNorm measures the induced matrix 1-norm of normalized coefficients.
// It supplies one half of the measured condition number without forming an inverse.
// Implementation:
//   - Stage 1: Visit each column of a row-major matrix.
//   - Stage 2: Sum the absolute values of its rows in a fixed order.
//   - Stage 3: Retain and return the maximum column sum.
//
// Behavior highlights:
//   - The input is normalized, so feasible row counts cannot overflow the sum.
//
// Inputs:
//   - data is finite normalized rows-by-cols data; dimensions are validated.
//
// Returns:
//   - Nonnegative finite column-sum maximum.
//
// Errors:
//   - None for valid normalized inputs.
//
// Determinism:
//   - Column index asc and row index asc.
//
// Complexity:
//   - Time O(rows*cols), Space O(1).
//
// Notes:
//   - Use QRFactor.upperOneNorm for packed QR where the lower panel stores reflectors.
//
// AI-Hints:
//   - Never use the original unnormalized norm if it would overflow prematurely.
func factorOneNorm(data []float64, rows, cols int) float64 {
	largest := 0.0
	for j := 0; j < cols; j++ {
		sum := 0.0
		for i := 0; i < rows; i++ {
			sum += math.Abs(data[i*cols+j])
		}
		if sum > largest {
			largest = sum
		}
	}
	return largest
}

// factorConditionFromBasisSolves measures kappa_1(M)=||M||_1*||M^-1||_1.
// Each inverse column is computed by solving M*x=e_j; no inverse Matrix is
// materialized or published, and the largest resulting column sum is retained.
// Implementation:
//   - Stage 1: allocate one basis vector and one solution vector per call.
//   - Stage 2: solve each identity basis in ascending column order.
//   - Stage 3: multiply the measured 1-norms and check representability.
//
// Behavior highlights: This is an explicit O(n^3) reference diagnostic, not
// a claimed O(n^2) Hager/Higham iteration or diagonal-ratio heuristic.
// Errors: context cancellation, underlying triangular-solve failure, ErrOverflow.
// Determinism: Unit basis column order 0..n-1 and fixed inner addition order.
// Complexity: Time O(n^3) for a triangular solve per basis; Space O(n).
// AI-Hints: Never substitute largest/smallest diagonal for this condition value.
func factorConditionFromBasisSolves(
	ctx context.Context,
	n int,
	oneNorm float64,
	solve func(context.Context, []float64, []float64) error,
) (float64, error) {
	basis := make([]float64, n)
	column := make([]float64, n)
	inverseOneNorm := 0.0
	for j := 0; j < n; j++ {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		basis[j] = 1
		if j > 0 {
			basis[j-1] = 0
		}
		if err := solve(ctx, column, basis); err != nil {
			return 0, err
		}
		columnSum := 0.0
		for i := 0; i < n; i++ {
			columnSum += math.Abs(column[i])
		}
		if err := factorCheckFiniteOutput(columnSum); err != nil {
			return 0, err
		}
		if columnSum > inverseOneNorm {
			inverseOneNorm = columnSum
		}
	}
	condition := oneNorm * inverseOneNorm
	if err := factorCheckFiniteOutput(condition); err != nil {
		return 0, err
	}
	// All induced matrix condition numbers are >=1 mathematically. Correct
	// only the harmless last-bit downward roundoff of the measured product.
	if condition < factorUnity {
		condition = factorUnity
	}
	return condition, nil
}

// factorConditionAccepted checks an already measured 1-norm condition estimate.
// It never substitutes a guessed value for an unknown diagnostic.
// Implementation:
//   - Stage 1: Interpret limit==0 as no rejection but not as no measurement.
//   - Stage 2: Compare a finite condition strictly against a finite positive threshold.
//   - Stage 3: Return ErrIllConditioned only when the strict limit is exceeded.
//
// Behavior highlights:
//   - Equality to the configured limit is accepted.
//
// Inputs:
//   - condition is measured kappa_1>=1; limit is validated 0 or >=1.
//
// Returns:
//   - nil or ErrIllConditioned.
//
// Errors:
//   - ErrIllConditioned on strictly greater measured condition.
//
// Determinism:
//   - Pure one-comparison policy with no data-dependent iteration.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This is a threshold on an estimate, not a proof about kappa_2.
//
// AI-Hints:
//   - Never reject at equality or reinterpret zero as an unknown condition.
func factorConditionAccepted(condition, limit float64) error {
	if limit != 0 && condition > limit {
		return ErrIllConditioned
	}
	return nil
}
