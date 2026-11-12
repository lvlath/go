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
//   - Stage 1: derive the slope-first basis and measure positive column divisors.
//   - Stage 2: allocate independent QR design/RHS matrices after the scale scan.
//   - Stage 3: populate the penalty-first row order and slope-first columns,
//     divide each coefficient by its positive scale, and publish only on success.
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
func prepareQRSystem(
	ctx context.Context,
	design, response matrix.Matrix,
	n, p int,
	noIntercept bool,
) (*matrix.Dense, *matrix.Dense, []float64, error) {
	// The QR basis always places regularized slopes before the optional
	// unpenalized intercept, regardless of the canonical design column order.
	q := p
	canonicalOffset := 0
	if !noIntercept {
		q += interceptColumns
		canonicalOffset = interceptColumns
	}
	rows := n + p

	// Column maxima are computed before allocating either QR matrix. The
	// separate reduction retains the original row order and cancellation polls.
	divisors, err := qrColumnDivisors(ctx, design, n, p, q, canonicalOffset, noIntercept)
	if err != nil {
		return nil, nil, nil, err
	}

	// Two owning matrices keep the original augmented system untouched. Only
	// successfully populated matrices can be returned to the QR solve boundary.
	balanced, err := matrix.NewDense(rows, q)
	if err != nil {
		return nil, nil, nil, err
	}
	balancedRHS, err := matrix.NewDense(rows, 1)
	if err != nil {
		return nil, nil, nil, err
	}

	// Apply the row/column permutations and positive column scaling together.
	// Never publish partial matrices on a read, write, or cancellation failure.
	if err = qrPopulateBalanced(ctx, design, response, balanced, balancedRHS,
		divisors, n, p, canonicalOffset, noIntercept); err != nil {
		return nil, nil, nil, err
	}
	return balanced, balancedRHS, divisors, nil
}

// qrColumnDivisors finds the positive scales for the slope-first QR basis.
// It scans the entire canonical design for every penalized feature, ensuring
// that collinear and wide predictors remain identifiable through penalty rows.
//
// Implementation:
//   - Stage 1: visit slope columns in original feature order and poll cancellation.
//   - Stage 2: scan each column's canonical rows for its maximum absolute value.
//   - Stage 3: append the analytically known intercept length when applicable.
//
// Behavior highlights:
//   - Maximum magnitudes avoid the overflow of squaring large column entries.
//   - Each slope has a nonzero penalty diagonal, so its maximum must be positive.
//   - Intercept scaling is sqrt(n) rather than a slope-dependent penalty weight.
//
// Inputs:
//   - ctx: non-nil active context owned by the caller.
//   - design: validated, detached (n+p)-by-q canonical augmented matrix.
//   - n, p, q: observation, feature, and QR column counts, respectively.
//   - canonicalOffset: 1 for an intercept-first canonical design, otherwise 0.
//   - noIntercept: whether the physical intercept is fixed to positive zero.
//
// Returns:
//   - []float64: fresh q-vector of strictly positive QR column divisors.
//   - error: nil after a complete finite scan; nil divisors on any failure.
//
// Errors:
//   - ErrOverflow: a required divisor is nonpositive or nonfinite.
//   - ctx.Err(): cancellation before a feature or during a column scan.
//   - Matrix.At errors: returned without changing their identity.
//
// Determinism:
//   - Slopes ascend by original feature index; each reads rows 0 through n+p-1.
//
// Complexity:
//   - Time O(p*(n+p)), Space O(q) for the detached divisor vector.
//
// Notes:
//   - Does not form Euclidean column norms or modify the canonical matrix.
//   - Positive scaling changes solver coordinates, not ridge regularization.
//
// AI-Hints:
//   - Never substitute a common global maximum for these column-specific scales;
//     a huge penalty would then erase the much smaller intercept basis.
//   - Preserve the scan order when adding specialized *matrix.Dense access;
//     repeated Matrix.At errors and cancellation points are observable.
func qrColumnDivisors(
	ctx context.Context,
	design matrix.Matrix,
	n, p, q, canonicalOffset int,
	noIntercept bool,
) ([]float64, error) {
	rows := n + p
	divisors := make([]float64, q)
	for col := 0; col < p; col++ {
		if col%contextCheckInterval == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		maximum := 0.0
		for row := 0; row < rows; row++ {
			if row%contextCheckInterval == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			value, err := design.At(row, col+canonicalOffset)
			if err != nil {
				return nil, err
			}
			if magnitude := math.Abs(value); magnitude > maximum {
				maximum = magnitude
			}
		}
		if !finite(maximum) || maximum <= 0 {
			return nil, ErrOverflow
		}
		divisors[col] = maximum
	}
	if !noIntercept {
		// The intercept contains n ones and p zero penalty entries, so its
		// exact column length is sqrt(n), independent of feature scaling.
		divisors[p] = math.Sqrt(float64(n))
		if !finite(divisors[p]) || divisors[p] <= 0 {
			return nil, ErrOverflow
		}
	}
	return divisors, nil
}

// qrPopulateBalanced writes the equivalent penalty-first QR design and RHS.
// It applies the same orthogonal row permutation to both matrices and an
// invertible column transformation only to the design, preserving least squares.
//
// Implementation:
//   - Stage 1: map each penalty-first destination row to its canonical row.
//   - Stage 2: map slope-first columns to canonical columns and divide by d_j.
//   - Stage 3: copy the corresponding RHS cell after all design cells in that row.
//
// Behavior highlights:
//   - Original penalty rows appear first and retain exact zero response values.
//   - With an intercept, the original first column moves to the final QR column.
//   - Invalid output arithmetic fails before publishing an incomplete QR system.
//
// Inputs:
//   - ctx: active context, polled at the same row/column boundaries as the scan.
//   - design, response: read-only canonical augmented system, with stable data.
//   - balanced, balancedRHS: caller-owned newly allocated QR output matrices.
//   - divisors: strictly positive q-vector from qrColumnDivisors.
//   - n, p: original observation and predictor counts, both positive.
//   - canonicalOffset: 1 when the intercept precedes predictors, otherwise 0.
//   - noIntercept: select either a pure slope basis or a trailing intercept.
//
// Returns:
//   - error: nil after filling every destination cell, otherwise the first error.
//
// Errors:
//   - ErrOverflow: a divided coefficient is not finite.
//   - ctx.Err(): cancellation during destination row/column traversal.
//   - Matrix.At/Set errors: returned without losing the backend error cause.
//
// Determinism:
//   - Destination rows ascend, then QR columns ascend, then one RHS cell per row.
//
// Complexity:
//   - Time O((n+p)*q), Space O(1) beyond the owning destination matrices.
//
// Notes:
//   - On failure, output matrices may contain partial writes but never escape Fit.
//   - The caller guarantees structural shape; this is not an exported validator.
//
// AI-Hints:
//   - Never scale y or insert observations into penalty RHS rows: doing either
//     changes the minimizer or its diagnostics.
//   - A writable *matrix.Dense backing slice is not part of the public API;
//     retain checked At/Set until a safe owning accessor is explicitly offered.
func qrPopulateBalanced(
	ctx context.Context,
	design, response matrix.Matrix,
	balanced, balancedRHS *matrix.Dense,
	divisors []float64,
	n, p, canonicalOffset int,
	noIntercept bool,
) error {
	rows, q := n+p, len(divisors)
	for row := 0; row < rows; row++ {
		if row%contextCheckInterval == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		// The first p destination rows correspond to the canonical penalty
		// block; subsequent rows recover observations in original order.
		canonicalRow := row - p
		if row < p {
			canonicalRow = n + row
		}
		for col := 0; col < q; col++ {
			if col%contextCheckInterval == 0 {
				if err := ctx.Err(); err != nil {
					return err
				}
			}
			canonicalCol := col + canonicalOffset
			if col == p && !noIntercept {
				canonicalCol = 0
			}
			value, err := design.At(canonicalRow, canonicalCol)
			if err != nil {
				return err
			}
			coordinate := value / divisors[col]
			if !finite(coordinate) {
				return ErrOverflow
			}
			if err = balanced.Set(row, col, coordinate); err != nil {
				return err
			}
		}
		// Keep response writes after design writes for each row. This
		// preserves failure precedence for alternate Matrix backends.
		target, err := response.At(canonicalRow, 0)
		if err != nil {
			return err
		}
		if err = balancedRHS.Set(row, 0, target); err != nil {
			return err
		}
	}
	return nil
}
