// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix provides exponent-aware right-hand-side handling for reusable factors.
// This file keeps finite solutions representable without prematurely scaling away inputs.
package matrix

import (
	"context"
	"errors"
	"math"
	"math/big"
)

const (
	// factorSolveRHSRelativeMinimum marks a coordinate smaller than one
	// binary64 unit of relative precision at its column's maximum magnitude.
	factorSolveRHSRelativeMinimum = 0x1p-53

	// factorSolveWideCondition is the inverse relative spacing of binary64.
	// Once condition estimates reach this magnitude, ordinary rounding may
	// eliminate meaningful coordinates even if the input ratios stay finite.
	factorSolveWideCondition = 0x1p52

	// factorSolveWidePrecision exceeds the full binary64 exponent span plus
	// guard bits. Extended arithmetic is reserved for exceptional RHS columns.
	factorSolveWidePrecision uint = 4096
)

// factorScaledKernel solves the stored magnitude-normalized factor for one RHS.
// Inputs and output are private float64 workspaces; the kernel must not publish data.
type factorScaledKernel func(context.Context, []float64, []float64) error

// factorWideKernel solves the same stored factor with extended-range intermediates.
// The input is the original detached row-major RHS and the output is in physical units.
type factorWideKernel func(context.Context, []float64, int, int, []float64) error

// factorSolveScaledColumns solves independent RHS columns without unsafe global rescaling.
// Each column is normalized by its own maximum input magnitude; recovered
// coefficients use separately tracked binary exponents of both matrix and RHS.
//
// Implementation:
//   - Stage 1: choose an independent positive scale and bounded RHS for each column.
//   - Stage 2: run the existing factor solve, using extended arithmetic only when
//     small RHS coordinates, condition, or an intermediate overflow require it.
//   - Stage 3: restore original coefficients without materializing scale ratios.
//
// Behavior highlights:
//   - A nonzero RHS coordinate too small for the shared float64 scale is never
//     silently replaced with zero; it triggers a private extended-range solve.
//   - A failed fast solve cannot publish partial results and does not alter the factor.
//
// Inputs:
//   - ctx: active context checked between RHS columns and in the solvers.
//   - rhs: detached finite m-by-r row-major right-hand sides.
//   - rows, rhsColumns, unknowns: validated positive dimensions.
//   - matrixScale: finite strictly positive scale of the stored factorization.
//   - condition: measured finite factor condition (not a numerical rank override).
//   - fast, wide: kernels for the same stored factor, never different objectives.
//
// Returns:
//   - []float64: independent n-by-r solution in original physical coordinates.
//
// Errors:
//   - ErrOverflow: the final solution is not representable even on the wide path.
//   - ctx.Err(): cancellation, including inside an extended-range solve.
//   - Other existing factor-kernel errors are propagated without replacement.
//
// Determinism:
//   - RHS column order, row order, and coefficient publication are increasing.
//
// Complexity:
//   - Fast: O(m*r + r*Tsolve) time and O(m+n+n*r) space.
//   - Exceptional columns: O(Tsolve) extended-precision operations with O(m+n)
//     extended-precision workspaces. Exact residual computation remains separate.
//
// Notes:
//   - Extended precision cannot restore information already lost during
//     factor construction or guarantee small coefficient error for ill-conditioned A.
//   - This helper does not modify pivots, rank policy or factor conditioning.
//
// AI-Hints:
//   - Dividing every original RHS by factor.scale can overflow or erase tiny
//     observations before triangular substitution can recover them.
//   - Never combine rhsScale/matrixScale in float64 before multiplying a result.
func factorSolveScaledColumns(
	ctx context.Context,
	rhs []float64,
	rows, rhsColumns, unknowns int,
	matrixScale, condition float64,
	fast factorScaledKernel,
	wide factorWideKernel,
) ([]float64, error) {
	solution := make([]float64, unknowns*rhsColumns)
	normalized := make([]float64, rows)
	calculated := make([]float64, unknowns)
	physical := make([]float64, unknowns)

	for column := 0; column < rhsColumns; column++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		rhsScale, needWide, err := factorNormalizeRHSColumn(ctx, rhs, rows, rhsColumns, column, normalized)
		if err != nil {
			return nil, err
		}
		if condition > factorSolveWideCondition {
			// An all-zero RHS has the exact zero solution at every condition.
			// Avoid expensive extended arithmetic for this common special case.
			for _, coordinate := range normalized {
				if coordinate != 0 {
					needWide = true
					break
				}
			}
		}
		if !needWide {
			err = fast(ctx, normalized, calculated)
			if err != nil && !errors.Is(err, ErrOverflow) {
				return nil, err
			}
			if err == nil {
				for row, value := range calculated {
					physical[row], err = factorRestoreScaled(value, rhsScale, matrixScale)
					if err != nil || (value != 0 && physical[row] == 0) {
						needWide = true
						break
					}
				}
			} else {
				needWide = true
			}
		}
		if needWide {
			if err := wide(ctx, rhs, rhsColumns, column, physical); err != nil {
				return nil, err
			}
		}
		for row, value := range physical {
			solution[row*rhsColumns+column] = value
		}
	}
	return solution, nil
}

// factorNormalizeRHSColumn supplies bounded float64 coordinates for one RHS.
// It detects values whose normalization would lose all or essentially all
// significant bits, selecting an extended-range solve before such a loss.
// Implementation:
//   - Stage 1: inspect the nonnegative maximum magnitude in ascending row order.
//   - Stage 2: divide by that maximum, or use scale one for an all-zero column.
//   - Stage 3: flag exceptional ratios for exact-range processing.
//
// Behavior highlights: a bounded normalized RHS never overflows by construction.
// Inputs: detached finite rhs, validated shape and column, private output workspace.
// Returns: positive RHS scale, extended-range requirement, or context error.
// Errors: ctx.Err() only; caller already validated input finiteness.
// Determinism: rows increase monotonically in both scans.
// Complexity: Time O(rows), Space O(1) beyond caller's output workspace.
// Notes: A tiny ratio may be meaningful in an ill-conditioned coordinate.
// AI-Hints: Testing normalized != 0 is insufficient: sub-ulp terms may still
// disappear when later added to order-one terms during substitution.
func factorNormalizeRHSColumn(
	ctx context.Context,
	rhs []float64,
	rows, rhsColumns, column int,
	normalized []float64,
) (float64, bool, error) {
	scale := 0.0
	for row := 0; row < rows; row++ {
		if row&31 == 0 {
			if err := ctx.Err(); err != nil {
				return 0, false, err
			}
		}
		if magnitude := math.Abs(rhs[row*rhsColumns+column]); magnitude > scale {
			scale = magnitude
		}
	}
	if scale == 0 {
		scale = 1
	}
	needWide := false
	for row := 0; row < rows; row++ {
		if row&31 == 0 {
			if err := ctx.Err(); err != nil {
				return 0, false, err
			}
		}
		value := rhs[row*rhsColumns+column]
		normalized[row] = value / scale
		if value != 0 && math.Abs(normalized[row]) < factorSolveRHSRelativeMinimum {
			needWide = true
		}
	}
	return scale, needWide, nil
}

// factorRestoreScaled reconstructs one physical coefficient from three finite factors.
// The mantissas are multiplied separately, so neither rhsScale/matrixScale nor
// the inverse ratio needs to exist as an intermediate binary64 number.
// Implementation:
//   - Stage 1: decompose solution, RHS scale and matrix scale into binary powers.
//   - Stage 2: form the bounded significand ratio, then combine integer exponents.
//   - Stage 3: round once into the destination exponent range and check finiteness.
//
// Behavior highlights: preserves a representable result despite an extreme scale ratio.
// Inputs: normalized coefficient, positive RHS scale and positive matrix scale.
// Returns: physical coefficient (including subnormal or zero) or ErrOverflow.
// Errors: ErrOverflow if the final float64 result is nonfinite.
// Determinism: fixed scalar binary operations, independent of RHS layout.
// Complexity: Time O(1), Space O(1).
// Notes: The caller handles suspicious zeros through the extended-range path.
// AI-Hints: Multiplying first and then dividing is not equivalent in finite
// arithmetic when the intermediate exponent lies outside binary64.
func factorRestoreScaled(coefficient, rhsScale, matrixScale float64) (float64, error) {
	if coefficient == 0 {
		return coefficient, nil
	}
	coefficientMantissa, coefficientExponent := math.Frexp(coefficient)
	rhsMantissa, rhsExponent := math.Frexp(rhsScale)
	matrixMantissa, matrixExponent := math.Frexp(matrixScale)
	significand := (coefficientMantissa * rhsMantissa) / matrixMantissa
	result := math.Ldexp(significand, coefficientExponent+rhsExponent-matrixExponent)
	if err := factorCheckFiniteOutput(result); err != nil {
		return 0, err
	}
	return result, nil
}

// factorWideFloat converts a finite binary64 number into a private wide value.
// Every source float is represented exactly before subsequent operations.
// Implementation: allocate one independent mantissa with fixed rounding precision.
// Behavior highlights: never changes the original factor or RHS memory.
// Inputs: finite original binary64 input; no NaN or infinity is permitted.
// Returns: separately owned wide scalar.
// Errors: none for validated finite inputs.
// Determinism: IEEE-754 input bit pattern determines the exact initial value.
// Complexity: O(1) fixed-precision allocation and conversion.
// Notes: this is an exceptional-path-only helper.
// AI-Hints: Converting to big.Float after float64 underflow cannot recover data.
func factorWideFloat(value float64) *big.Float {
	return new(big.Float).SetPrec(factorSolveWidePrecision).SetMode(big.ToNearestEven).SetFloat64(value)
}

// factorWideOriginalRHS copies one unscaled source column to wide arithmetic.
// It avoids the information loss inherent in any single float64 RHS divisor.
// Implementation: convert source rows in ascending order into private big.Float values.
// Behavior highlights: no division by matrix or RHS scale has occurred.
// Inputs: active ctx, detached finite RHS, its column count, selected column and row count.
// Returns: independently allocated high-range working vector, or nil on cancellation.
// Errors: ctx.Err() at deterministic row checkpoints.
// Determinism: input row order asc.
// Complexity: Time O(rows), Space O(rows) wide values.
// Notes: constructing a large wide RHS is itself cooperatively cancellable.
// AI-Hints: Keep the original RHS available until the complete SolveResult is built.
func factorWideOriginalRHS(ctx context.Context, rhs []float64, rhsColumns, column, rows int) ([]*big.Float, error) {
	work := make([]*big.Float, rows)
	for row := range work {
		if row&31 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		work[row] = factorWideFloat(rhs[row*rhsColumns+column])
	}
	return work, nil
}

// factorWideRecover writes original-unit coefficients from the stored normalized factor.
// It delays division by matrixScale until the high-range triangular solution exists.
// Implementation:
//   - Stage 1: validate context and positive finite factor scale.
//   - Stage 2: divide each wide coefficient once by its original matrix scale.
//   - Stage 3: round the result into binary64, rejecting only nonfinite outputs.
//
// Behavior highlights: permits cancellation and temporary values beyond MaxFloat64.
// Inputs: ctx, private wide unknowns, factor scale and independent float64 destination.
// Returns: nil after all finite coefficients are ready, otherwise classified error.
// Errors: ErrOverflow, ctx.Err().
// Determinism: increasing coefficient index.
// Complexity: O(n) wide operations, O(1) reused scratch (besides input/output).
// Notes: the caller never publishes a partially recovered solution on error.
// AI-Hints: Do not round the wide solution before applying matrixScale.
func factorWideRecover(ctx context.Context, normalized []*big.Float, matrixScale float64, destination []float64) error {
	denominator := factorWideFloat(matrixScale)
	var quotient big.Float
	quotient.SetPrec(factorSolveWidePrecision)
	for index, value := range normalized {
		if index&31 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		quotient.Quo(value, denominator)
		rounded, _ := quotient.Float64()
		if err := factorCheckFiniteOutput(rounded); err != nil {
			return err
		}
		destination[index] = rounded
	}
	return nil
}

// solveWide solves a row-pivoted LU factor with high-range right-hand-side arithmetic.
// It uses the existing stored L and U; no new factorization or inverse is formed.
// Implementation:
//   - Stage 1: permute original RHS rows into the private forward workspace.
//   - Stage 2: substitute through unit-lower and upper pivoted triangles.
//   - Stage 3: restore the original matrix scale before final float64 rounding.
//
// Behavior highlights: handles true temporary overflow in normalized substitution.
// Inputs: ctx, original flattened RHS, its column index and output workspace.
// Returns: nil on representable completion; otherwise a classified error.
// Errors: ErrOverflow, ErrRankDeficient for a malformed zero pivot, ctx.Err().
// Determinism: increasing forward rows, descending backward rows, fixed permutation.
// Complexity: Time O(n^2) wide operations, Space O(n) wide values.
// Notes: factors and their permutation are never modified.
// AI-Hints: This is a precision fallback, not normal equations or OLS.
func (f *LUFactor) solveWide(ctx context.Context, rhs []float64, rhsColumns, column int, destination []float64) error {
	n := f.n
	work := make([]*big.Float, n)
	for row := 0; row < n; row++ {
		if row&31 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		work[row] = factorWideFloat(rhs[f.perm[row]*rhsColumns+column])
	}
	var coefficient, product big.Float
	coefficient.SetPrec(factorSolveWidePrecision)
	product.SetPrec(factorSolveWidePrecision)
	for row := 0; row < n; row++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		for previous := 0; previous < row; previous++ {
			coefficient.SetFloat64(f.packed[row*n+previous])
			product.Mul(&coefficient, work[previous])
			work[row].Sub(work[row], &product)
		}
	}
	answer := make([]*big.Float, n)
	for row := n - 1; row >= 0; row-- {
		if err := ctx.Err(); err != nil {
			return err
		}
		candidate := new(big.Float).SetPrec(factorSolveWidePrecision).Set(work[row])
		for next := row + 1; next < n; next++ {
			coefficient.SetFloat64(f.packed[row*n+next])
			product.Mul(&coefficient, answer[next])
			candidate.Sub(candidate, &product)
		}
		pivot := f.packed[row*n+row]
		if pivot == 0 {
			return ErrRankDeficient
		}
		coefficient.SetFloat64(pivot)
		answer[row] = candidate.Quo(candidate, &coefficient)
	}
	return factorWideRecover(ctx, answer, f.scale, destination)
}

// solveWide solves the original Householder QR factor with wide RHS arithmetic.
// It applies the same reflector order and the same stored pivoted R as the
// ordinary solve, preserving the accepted least-squares problem.
// Implementation:
//   - Stage 1: copy unscaled RHS and apply compact Q transpose in forward order.
//   - Stage 2: solve pivoted upper R by descending substitution.
//   - Stage 3: undo the column permutation and original factor scale.
//
// Behavior highlights: no normal-equation or pseudoinverse fallback is used.
// Inputs: ctx, raw finite RHS column and private destination of original unknowns.
// Returns: nil for all representable coefficients, otherwise classified error.
// Errors: ErrOverflow, ErrRankDeficient for malformed zero pivot, ctx.Err().
// Determinism: reflector asc, triangular row desc, unpermutation by original index.
// Complexity: Time O(m*n+n^2) wide operations, Space O(m+n) wide values.
// Notes: does not mutate the stored packed reflectors or input RHS.
// AI-Hints: Q^T uses reflectors in forward order; perm is current-to-original.
func (f *QRFactor) solveWide(ctx context.Context, rhs []float64, rhsColumns, column int, destination []float64) error {
	work, err := factorWideOriginalRHS(ctx, rhs, rhsColumns, column, f.m)
	if err != nil {
		return err
	}
	var coefficient, product, dot, update big.Float
	coefficient.SetPrec(factorSolveWidePrecision)
	product.SetPrec(factorSolveWidePrecision)
	dot.SetPrec(factorSolveWidePrecision)
	update.SetPrec(factorSolveWidePrecision)
	for reflector := 0; reflector < f.n; reflector++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		if f.tau[reflector] == 0 {
			continue
		}
		dot.Set(work[reflector])
		for row := reflector + 1; row < f.m; row++ {
			coefficient.SetFloat64(f.packed[row*f.n+reflector])
			product.Mul(&coefficient, work[row])
			dot.Add(&dot, &product)
		}
		coefficient.SetFloat64(f.tau[reflector])
		update.Mul(&coefficient, &dot)
		work[reflector].Sub(work[reflector], &update)
		for row := reflector + 1; row < f.m; row++ {
			coefficient.SetFloat64(f.packed[row*f.n+reflector])
			product.Mul(&coefficient, &update)
			work[row].Sub(work[row], &product)
		}
	}
	answer := make([]*big.Float, f.n)
	for row := f.n - 1; row >= 0; row-- {
		if err := ctx.Err(); err != nil {
			return err
		}
		candidate := new(big.Float).SetPrec(factorSolveWidePrecision).Set(work[row])
		for next := row + 1; next < f.n; next++ {
			coefficient.SetFloat64(f.packed[row*f.n+next])
			product.Mul(&coefficient, answer[next])
			candidate.Sub(candidate, &product)
		}
		diagonal := f.packed[row*f.n+row]
		if diagonal == 0 {
			return ErrRankDeficient
		}
		coefficient.SetFloat64(diagonal)
		answer[row] = candidate.Quo(candidate, &coefficient)
	}
	originalOrder := make([]*big.Float, f.n)
	for pivotIndex, originalIndex := range f.perm {
		originalOrder[originalIndex] = answer[pivotIndex]
	}
	return factorWideRecover(ctx, originalOrder, f.scale, destination)
}

// solveWide solves the accepted symmetric SPD factor using wide triangular passes.
// The private Cholesky factor is never modified, and both substitutions retain
// the original factor scale until coefficients can be rounded safely.
// Implementation:
//   - Stage 1: forward-substitute through the normalized lower triangle.
//   - Stage 2: backward-substitute through its transpose.
//   - Stage 3: divide the wide coefficients by original A scaling.
//
// Behavior highlights: this preserves valid tiny coordinates across extreme scales.
// Inputs: ctx, raw finite RHS column, shape and destination workspace.
// Returns: nil on representable completion, otherwise classified error.
// Errors: ErrOverflow, ErrRankDeficient for malformed zero pivot, ctx.Err().
// Determinism: forward rows asc, backward rows desc.
// Complexity: Time O(n^2) wide operations, Space O(n) wide values.
// Notes: accepted SPD symmetry policy remains exactly as constructed by FactorCholesky.
// AI-Hints: Never square the normalized Cholesky factor or form A^T*A here.
func (f *CholeskyFactor) solveWide(ctx context.Context, rhs []float64, rhsColumns, column int, destination []float64) error {
	work, err := factorWideOriginalRHS(ctx, rhs, rhsColumns, column, f.n)
	if err != nil {
		return err
	}
	var coefficient, product big.Float
	coefficient.SetPrec(factorSolveWidePrecision)
	product.SetPrec(factorSolveWidePrecision)
	for row := 0; row < f.n; row++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		for previous := 0; previous < row; previous++ {
			coefficient.SetFloat64(f.lower[row*f.n+previous])
			product.Mul(&coefficient, work[previous])
			work[row].Sub(work[row], &product)
		}
		diagonal := f.lower[row*f.n+row]
		if diagonal == 0 {
			return ErrRankDeficient
		}
		coefficient.SetFloat64(diagonal)
		work[row].Quo(work[row], &coefficient)
	}
	answer := make([]*big.Float, f.n)
	for row := f.n - 1; row >= 0; row-- {
		if err := ctx.Err(); err != nil {
			return err
		}
		candidate := new(big.Float).SetPrec(factorSolveWidePrecision).Set(work[row])
		for next := row + 1; next < f.n; next++ {
			coefficient.SetFloat64(f.lower[next*f.n+row])
			product.Mul(&coefficient, answer[next])
			candidate.Sub(candidate, &product)
		}
		coefficient.SetFloat64(f.lower[row*f.n+row])
		answer[row] = candidate.Quo(candidate, &coefficient)
	}
	return factorWideRecover(ctx, answer, f.scale, destination)
}
