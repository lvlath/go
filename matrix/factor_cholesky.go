// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix implements a symmetric, positive-definite Cholesky factorization.
// This file enforces explicit symmetry tolerances and immutable accepted systems.
package matrix

import (
	"context"
	"math"
)

// factorSymmetricAverage computes an accepted finite symmetric entry midpoint.
// It avoids both overflow of a+b and premature subnormal division.
// Implementation:
//   - Stage 1: Return an exactly equal pair unchanged.
//   - Stage 2: Add before halving when the sum fits, preserving tie-to-even subnormal rounding.
//   - Stage 3: Otherwise use a nonoverflowing same-sign difference midpoint.
//
// Behavior highlights:
//   - The chosen midpoint is identical on both sides of the stored symmetric pair.
//
// Inputs:
//   - left and right are finite, validated approximate-symmetric coefficients.
//
// Returns:
//   - Finite binary64 midpoint in the original coefficient units.
//
// Errors:
//   - None for finite operands under the accepted symmetry law.
//
// Determinism:
//   - Pure fixed branch on operand signs and sum headroom.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - The caller retains original symmetry tolerances without implicit diagonal shifts.
//
// AI-Hints:
//   - Do not write left/2+right/2 unconditionally: it erases the smallest subnormals.
func factorSymmetricAverage(left, right float64) float64 {
	if left == right {
		return left
	}
	leftMagnitude, rightMagnitude := math.Abs(left), math.Abs(right)
	// Opposite signs cannot overflow a sum. Otherwise compare the larger
	// magnitude against the representable headroom remaining after the
	// smaller operand, avoiding the overflowing addition itself.
	if math.Signbit(left) != math.Signbit(right) ||
		math.Max(leftMagnitude, rightMagnitude) <= math.MaxFloat64-math.Min(leftMagnitude, rightMagnitude) {
		return (left + right) / 2
	}
	return left + (right-left)/2
}

// factorSymmetryThreshold constructs a finite elementwise symmetry allowance.
// An accidentally infinite tolerance must never authorize arbitrary asymmetry.
// Implementation:
//   - Stage 1: Choose the maximum original-coordinate pair magnitude.
//   - Stage 2: Reject an overflowing relative or combined absolute-plus-relative allowance.
//   - Stage 3: Return the finite bound used by the subsequent pairwise comparison.
//
// Behavior highlights:
//   - relTol<=1 needs no quotient-overflow sentinel to check multiplication.
//
// Inputs:
//   - left,right are finite coefficients; tolerances are validated finite>=0.
//
// Returns:
//   - Finite threshold in coefficient units.
//
// Errors:
//   - ErrOverflow if the required tolerance cannot fit finite float64.
//
// Determinism:
//   - Pure operations in fixed left/right argument order.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Difference computation may overflow only when asymmetry truly exceeds any finite bound.
//
// AI-Hints:
//   - Do not replace an overflowing tolerance with +Inf and silently accept the pair.
func factorSymmetryThreshold(left, right, absTol, relTol float64) (float64, error) {
	magnitude := math.Max(math.Abs(left), math.Abs(right))
	// Multiplying a finite magnitude by relTol<=1 cannot overflow;
	// avoiding MaxFloat64/relTol also prevents a needless infinite quotient.
	if relTol > 1 && magnitude > math.MaxFloat64/relTol {
		return 0, ErrOverflow
	}
	relative := relTol * magnitude
	if absTol > math.MaxFloat64-relative {
		return 0, ErrOverflow
	}
	return absTol + relative, nil
}

// factorCholeskyKernel accepts a symmetric SPD matrix under explicit tolerances.
// It materializes the symmetric midpoint of each admissible off-diagonal pair
// and factorizes only this detached accepted system.
// Implementation:
//   - Stage 1: validate square finite input and both finite nonnegative tolerances.
//   - Stage 2: enforce the declared elementwise symmetry inequality, storing
//     one symmetric midpoint for every pair without overflow.
//   - Stage 3: scale the matrix and compute L by ascending-row Cholesky;
//     nonpositive diagonal residuals reject the factor without regularization.
//   - Stage 4: compute kappa_1(acceptedA) through independent basis solves.
//
// Behavior highlights: Exact symmetry (0,0) never changes an input coefficient.
// Inputs:
//   - ctx: non-nil cancellation context checked before each outer factor row.
//   - a: read-only finite square n-by-n input with n>=1.
//   - absTol: nonnegative finite tolerance in the input's coefficient units.
//   - relTol: dimensionless nonnegative finite relative symmetry tolerance.
//
// Returns: Immutable, independently owned factor or nil on any failure.
// Errors: ErrNonSquare, ErrAsymmetry, ErrNotPositiveDefinite, ErrOverflow,
// ErrNaNInf, ErrBadShape, ctx.Err(), and wrapped Matrix.At failures.
// Determinism: Pairs are visited i ascending, j ascending, with fixed dot order.
// Complexity: O(n^3) factorization+condition, O(n^2) retained storage.
// Notes: Residuals are measured against the accepted symmetrized A.
// AI-Hints: Do not add an epsilon to a nonpositive pivot: it silently changes
// an indefinite matrix into a different optimization problem.
func factorCholeskyKernel(ctx context.Context, a Matrix, absTol, relTol float64) (*CholeskyFactor, error) {
	rows, cols, err := factorInputShape(a)
	if err != nil {
		return nil, err
	}
	if rows != cols {
		return nil, ErrNonSquare
	}
	original, err := factorReadMatrix(ctx, a, rows, cols)
	if err != nil {
		return nil, err
	}
	for i := 0; i < rows; i++ {
		if err = ctx.Err(); err != nil {
			return nil, err
		}
		for j := i + 1; j < rows; j++ {
			left, right := original[i*rows+j], original[j*rows+i]
			// Equality satisfies the symmetry inequality without evaluating the
			// tolerance expression. This avoids rejecting exactly symmetric SPD
			// inputs merely because a caller supplied huge finite tolerances.
			if left == right {
				continue
			}
			threshold, thresholdErr := factorSymmetryThreshold(left, right, absTol, relTol)
			if thresholdErr != nil {
				return nil, thresholdErr
			}
			if math.Abs(left-right) > threshold {
				return nil, ErrAsymmetry
			}
			midpoint := factorSymmetricAverage(left, right)
			original[i*rows+j], original[j*rows+i] = midpoint, midpoint
		}
	}
	packed, scale, err := factorScaleAndNormalize(original)
	if err != nil {
		return nil, ErrNotPositiveDefinite
	}
	oneNorm := factorOneNorm(packed, rows, rows)
	lower := make([]float64, len(packed))
	for i := 0; i < rows; i++ {
		if err = ctx.Err(); err != nil {
			return nil, err
		}
		for j := 0; j <= i; j++ {
			value := packed[i*rows+j]
			for k := 0; k < j; k++ {
				value = math.FMA(-lower[i*rows+k], lower[j*rows+k], value)
			}
			if err = factorCheckFiniteOutput(value); err != nil {
				return nil, err
			}
			if i == j {
				if value <= 0 {
					return nil, ErrNotPositiveDefinite
				}
				lower[i*rows+j] = math.Sqrt(value)
			} else {
				lower[i*rows+j] = value / lower[j*rows+j]
			}
			if err = factorCheckFiniteOutput(lower[i*rows+j]); err != nil {
				return nil, err
			}
		}
	}
	factor := &CholeskyFactor{n: rows, accepted: original, lower: lower, scale: scale}
	condition, err := factorConditionFromBasisSolves(ctx, rows, oneNorm, factor.solveUnit)
	if err != nil {
		return nil, err
	}
	factor.condition = condition
	return factor, nil
}

// solveUnit solves the accepted normalized SPD system by two triangular passes.
// The shared immutable L factor avoids a second decomposition for each RHS.
// Implementation:
//   - Stage 1: Forward-solve L*y=rhs with a fresh per-call y workspace.
//   - Stage 2: Backward-solve L^T*dst=y, checking cancellation at row blocks.
//   - Stage 3: Reject nonfinite computed coefficients before returning.
//
// Behavior highlights:
//   - The original accepted matrix and packed L are never modified.
//
// Inputs:
//   - ctx non-nil; rhs and dst are private n-vectors; f is a ready factor.
//
// Returns:
//   - nil with finite dst or an error with no published SolveResult.
//
// Errors:
//   - ErrOverflow or ctx.Err() propagated unchanged.
//
// Determinism:
//   - Forward rows ascend; backward rows descend, with increasing inner columns.
//
// Complexity:
//   - Time O(n^2), Space O(n).
//
// Notes:
//   - The caller scales RHS consistently with normalized L before this method.
//
// AI-Hints:
//   - Never solve against the raw nonsymmetric matrix when a midpoint was accepted.
func (f *CholeskyFactor) solveUnit(ctx context.Context, dst, rhs []float64) error {
	n := f.n
	work := make([]float64, n)
	for i := 0; i < n; i++ {
		if i&31 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		value := rhs[i]
		for j := 0; j < i; j++ {
			value = math.FMA(-f.lower[i*n+j], work[j], value)
		}
		work[i] = value / f.lower[i*n+i]
		if err := factorCheckFiniteOutput(work[i]); err != nil {
			return err
		}
	}
	for i := n - 1; i >= 0; i-- {
		if i&31 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		value := work[i]
		for j := i + 1; j < n; j++ {
			value = math.FMA(-f.lower[j*n+i], dst[j], value)
		}
		dst[i] = value / f.lower[i*n+i]
		if err := factorCheckFiniteOutput(dst[i]); err != nil {
			return err
		}
	}
	return nil
}
