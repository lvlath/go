// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ridge prepares an equivalent, balanced QR representation of the
// augmented Tikhonov problem while preserving its original objective.
package ridge

import (
	"context"
	"math"

	"github.com/lvlath/go/matrix"
)

// prepareQRSystem reorders and equilibrates the canonical augmented problem.
// The transformed system has precisely the same least-squares minimizers:
// it first permutes rows orthogonally and then changes coefficient coordinates
// by a nonsingular positive diagonal matrix.
//
// Implementation:
//   - Stage 1: measure each penalized design column without squaring entries.
//   - Stage 2: place zero-response penalty rows before observation rows, and
//     penalized columns before the unpenalized intercept column.
//   - Stage 3: divide each column by its positive scale and return detached
//     matrices and the corresponding coefficient-coordinate divisors.
//
// Behavior highlights:
//   - Penalty-first rows protect small nonzero slopes from subtractive loss in
//     the Householder Q^T*y application when Lambda is extremely large.
//   - The intercept is always represented and unpenalized; moving its column
//     to the right prevents it from being pivoted ahead of the penalty basis
//     on exact norm ties, which would amplify roundoff into small slopes.
//   - Column equilibration keeps the penalized diagonal near unit magnitude
//     without changing the penalty strength or the normalization by n.
//
// Inputs:
//   - ctx: active non-nil context checked throughout preparation.
//   - design: detached canonical (n+p)-by-q [1,Z;0,wI] or [Z;wI] matrix.
//   - response: detached canonical (n+p)-by-1 [y;0] matrix.
//   - n: positive training-row count; p: positive original feature count.
//   - noIntercept: selects whether the canonical column zero is an intercept.
//
// Returns:
//   - *matrix.Dense: QR input, with penalty rows first and slopes before intercept.
//   - *matrix.Dense: identically row-permuted RHS, including exact zero penalty rows.
//   - []float64: detached column divisors d_j, with theta_j=gamma_j/d_j.
//   - error: nil only when the complete transformed system is finite.
//
// Errors:
//   - ErrOverflow: a required transformed entry or divisor cannot be represented.
//   - ctx.Err(): cancellation while reading or writing the matrix.
//   - Matrix.At/Set and matrix constructor errors are preserved unchanged.
//
// Determinism:
//   - Feature maxima scan canonical rows ascending; the output visits rows then
//     reordered columns ascending. No randomness, maps, or parallel reduction.
//
// Complexity:
//   - Time O((n+p)*q); additional space O((n+p)*q+p), including the QR input
//     and RHS but excluding the separately owned canonical augmented matrices.
//
// Notes:
//   - The solver internally normalizes its matrix by a common maximum; the
//     column transform prevents that global normalization from losing an
//     intercept whose magnitude is tiny relative to sqrt(n*Lambda).
//   - The matrix solve is still ordinary Householder QR of an exactly equivalent
//     augmented Tikhonov objective; this is not a normal-equation solver.
//   - Very small, nonzero source entries can underflow after equilibration.
//     Numerical rank and coefficient representability remain explicit gates.
//
// AI-Hints:
//   - Never reuse the solver's gamma coefficients directly as physical beta;
//     undo each divisor before the training scale and mean correction.
//   - Keep the penalty RHS equal to zero. Putting observed responses in the
//     penalty rows changes the optimizer and defeats the numerical advantage.
//   - *matrix.Dense gives constant-time indexed access; the public Matrix API
//     does not expose a safe writable backing slice for direct assembly.
//
// nolint:gocyclo
func prepareQRSystem(
	ctx context.Context,
	design, response matrix.Matrix,
	n, p int,
	noIntercept bool,
) (*matrix.Dense, *matrix.Dense, []float64, error) {
	// Internal QR columns are always slopes first, then the optional intercept.
	q := p
	canonicalOffset := 0
	if !noIntercept {
		q += interceptColumns
		canonicalOffset = interceptColumns
	}
	rows := n + p

	// Use per-column maximum magnitudes instead of Euclidean column norms:
	// the latter may overflow even when all normalized entries are representable.
	// Every slope column contains one strictly positive penalty entry, so every
	// divisor is positive regardless of source rank and collinearity.
	divisors := make([]float64, q)
	for col := 0; col < p; col++ {
		if col%contextCheckInterval == 0 {
			if err := ctx.Err(); err != nil {
				return nil, nil, nil, err
			}
		}
		maximum := 0.0
		for row := 0; row < rows; row++ {
			if row%contextCheckInterval == 0 {
				if err := ctx.Err(); err != nil {
					return nil, nil, nil, err
				}
			}
			value, err := design.At(row, col+canonicalOffset)
			if err != nil {
				return nil, nil, nil, err
			}
			if magnitude := math.Abs(value); magnitude > maximum {
				maximum = magnitude
			}
		}
		if !finite(maximum) || maximum <= 0 {
			return nil, nil, nil, ErrOverflow
		}
		divisors[col] = maximum
	}
	if !noIntercept {
		// The intercept column consists of n ones and p zeros. Its exact
		// Euclidean length sqrt(n) balances it against normalized slope columns.
		divisors[p] = math.Sqrt(float64(n))
		if !finite(divisors[p]) || divisors[p] <= 0 {
			return nil, nil, nil, ErrOverflow
		}
	}

	// The canonical design is left untouched. A separate owning matrix is
	// materialized because matrix.Matrix does not guarantee a writable raw view.
	balanced, err := matrix.NewDense(rows, q)
	if err != nil {
		return nil, nil, nil, err
	}
	balancedRHS, err := matrix.NewDense(rows, 1)
	if err != nil {
		return nil, nil, nil, err
	}

	for row := 0; row < rows; row++ {
		if row%contextCheckInterval == 0 {
			if err = ctx.Err(); err != nil {
				return nil, nil, nil, err
			}
		}
		// Move exactly p canonical bottom rows to the leading QR rows.
		// This is an orthogonal permutation, preserving the squared loss.
		canonicalRow := row - p
		if row < p {
			canonicalRow = n + row
		}
		for col := 0; col < q; col++ {
			if col%contextCheckInterval == 0 {
				if err = ctx.Err(); err != nil {
					return nil, nil, nil, err
				}
			}
			canonicalCol := col + canonicalOffset
			if col == p && !noIntercept {
				canonicalCol = 0
			}
			value, readErr := design.At(canonicalRow, canonicalCol)
			if readErr != nil {
				return nil, nil, nil, readErr
			}
			coordinate := value / divisors[col]
			if !finite(coordinate) {
				return nil, nil, nil, ErrOverflow
			}
			if err = balanced.Set(row, col, coordinate); err != nil {
				return nil, nil, nil, err
			}
		}
		target, readErr := response.At(canonicalRow, 0)
		if readErr != nil {
			return nil, nil, nil, readErr
		}
		if err = balancedRHS.Set(row, 0, target); err != nil {
			return nil, nil, nil, err
		}
	}
	return balanced, balancedRHS, divisors, nil
}
