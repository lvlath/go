// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix solves linear systems with immutable reusable factors.
// This file validates right-hand sides and measures exact original-coordinate residuals.
package matrix

import (
	"context"
	"math/big"
)

// factorExactResidualNorms evaluates each original-coordinate Euclidean residual.
// All observations and solved coefficients are interpreted as exact binary64
// numbers. This avoids false overflow during A*x, cancellation loss in A*x-b,
// and premature rounding of an individual tiny residual before forming the norm.
// Implementation:
//   - Stage 1: convert source floats to exact integers on the 2^-1074 lattice.
//   - Stage 2: accumulate each dot product and b subtraction exactly at a
//     shared 2^-2148 scale, then square the exact residual integer.
//   - Stage 3: sum those integer squares, take one 256-bit-precision square root,
//     divide by 2^2148 and round the final L2 norm to finite binary64.
//
// Behavior highlights:
//   - Tiny individual residuals are not discarded before computing their norm.
//   - No m-by-n or m-by-r intermediate residual matrix is allocated.
//
// Inputs:
//   - original: detached row-major m-by-n finite original or accepted SPD A.
//   - solutions: detached row-major n-by-r finite solved coefficients.
//   - rhs: detached row-major m-by-r finite right-hand sides.
//   - ctx: non-nil cancellation context, checked at each row boundary.
//
// Returns: Detached []float64 with one finite L2 norm for each RHS column.
// Errors: ErrOverflow for an unrepresentable final norm; ctx.Err() on cancel.
// Determinism: RHS column -> matrix row -> coefficient column, always ascending.
// Complexity: O(m*n*r) fixed-range exact-integer operations plus O(m*r)
// integer squares; O(r) output and O(1) reusable fixed-size big-number registers.
// Notes: Intermediate rational expressions can exceed 64-bit bounds, but the
// significant bit span is bounded by the IEEE binary64 exponent range and n.
// AI-Hints: Do not round each residual independently before L2 aggregation:
// many sub-ulp residuals can have a representable combined Euclidean norm.
func factorExactResidualNorms(
	ctx context.Context,
	original []float64,
	rows, cols int,
	solutions []float64,
	rhs []float64,
	rhsCols int,
) ([]float64, error) {
	norms := make([]float64, rhsCols)
	// Every finite binary64 value maps exactly to an integer at 2^-1074;
	// a product therefore has the common integer denominator 2^2148.
	denominator := new(big.Int).Lsh(big.NewInt(1), 2*statisticsExactInputShift)
	var coefficient, unknown, product, observation, signedSum big.Int
	var square, sumOfSquares big.Int
	const residualNormPrecision uint = 256
	for k := 0; k < rhsCols; k++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		sumOfSquares.SetInt64(0)
		for i := 0; i < rows; i++ {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			signedSum.SetInt64(0)
			for j := 0; j < cols; j++ {
				statisticsFloatToInteger(&coefficient, original[i*cols+j])
				statisticsFloatToInteger(&unknown, solutions[j*rhsCols+k])
				product.Mul(&coefficient, &unknown)
				signedSum.Add(&signedSum, &product)
			}
			statisticsFloatToInteger(&observation, rhs[i*rhsCols+k])
			observation.Lsh(&observation, statisticsExactInputShift)
			signedSum.Sub(&signedSum, &observation)
			// Squaring signedSum exactly before floating conversion keeps
			// both enormous canceling terms and subnormal residue visible.
			square.Mul(&signedSum, &signedSum)
			sumOfSquares.Add(&sumOfSquares, &square)
		}
		if sumOfSquares.Sign() == 0 {
			continue
		}
		numerator := new(big.Float).SetPrec(residualNormPrecision).SetInt(&sumOfSquares)
		numerator.Sqrt(numerator)
		scale := new(big.Float).SetPrec(residualNormPrecision).SetInt(denominator)
		value, _ := numerator.Quo(numerator, scale).Float64()
		if err := factorCheckFiniteOutput(value); err != nil {
			return nil, err
		}
		norms[k] = value
	}
	return norms, nil
}

// factorBuildSolveResult publishes a complete detached solution and true residual metadata.
// The result boundary is responsible for rejecting any partial or corrupted solve.
// Implementation:
//   - Stage 1: Check cancellation and every solved coefficient is finite.
//   - Stage 2: Compute original-coordinate residual L2 norms without early rounding.
//   - Stage 3: Allocate and copy an owning Dense only after all diagnostics pass.
//
// Behavior highlights:
//   - The caller receives independent solution and metadata storage.
//
// Inputs:
//   - ctx is non-nil; original, solution and RHS are private finite snapshots.
//
// Returns:
//   - A fresh SolveResult with rank=cols and measured condition or nil on error.
//
// Errors:
//   - ErrOverflow, ctx.Err(), and allocation errors.
//
// Determinism:
//   - Solutions preserve RHS column order and residual scan order.
//
// Complexity:
//   - Time O(m*n*r) exact residual work plus O(n*r) output copy.
//
// Notes:
//   - No partial results are published after any canceled or invalid RHS column.
//
// AI-Hints:
//   - Do not expose private work vectors or retain mutable result memory on the factor.
func factorBuildSolveResult(
	ctx context.Context,
	original []float64,
	rows, cols int,
	solutions []float64,
	rhs []float64,
	rhsCols int,
	condition float64,
) (*SolveResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	for _, coefficient := range solutions {
		if err := factorCheckFiniteOutput(coefficient); err != nil {
			return nil, err
		}
	}
	residuals, err := factorExactResidualNorms(ctx, original, rows, cols, solutions, rhs, rhsCols)
	if err != nil {
		return nil, err
	}
	out, err := NewDense(cols, rhsCols)
	if err != nil {
		return nil, err
	}
	copy(out.data, solutions)
	return &SolveResult{Solution: out, Rank: cols, ResidualNorms: residuals, ConditionEstimate: condition}, nil
}

// Solve solves A*X=B using an immutable row-pivoted LU decomposition.
// The factor is safe to share between goroutines, because every call allocates
// its own right-hand-side snapshot and triangular workspace.
// Implementation:
//   - Stage 1: validate ctx, receiver, nonempty B shape, and finite entries.
//   - Stage 2: permute and scale each RHS, solve unit-lower and upper systems.
//   - Stage 3: compute exact original-coordinate L2 residuals and publish X.
//
// Behavior highlights: The same stored permutation is used on every RHS.
// Inputs:
//   - ctx: non-nil cancellation context, checked between RHS columns.
//   - b: read-only n-by-r Matrix, r>=1; caller must not mutate concurrently.
//
// Returns: Detached n-by-r SolveResult, or nil on any error/cancellation.
// Errors: ErrNilMatrix, ErrDimensionMismatch, ErrBadShape, ErrNaNInf,
// ErrOverflow, ctx.Err(), and wrapped Matrix.At errors.
// Determinism: RHS columns ascending; pivoted triangular rows in canonical order.
// Complexity: Time O(n*n*r) arithmetic plus exact residual work; Space O(n*r+n).
// Notes: Factor construction already verified numerical rank and conditioning.
// AI-Hints: Do not use the legacy unpivoted LU(A) output as this factor's storage.
func (f *LUFactor) Solve(ctx context.Context, b Matrix) (*SolveResult, error) {
	if err := factorContextErr(ctx); err != nil {
		return nil, err
	}
	if f == nil {
		return nil, ErrNilMatrix
	}
	rows, columns, err := factorInputShape(b)
	if err != nil {
		return nil, err
	}
	if rows != f.n {
		return nil, ErrDimensionMismatch
	}
	rhs, err := factorReadMatrix(ctx, b, rows, columns)
	if err != nil {
		return nil, err
	}
	solution := make([]float64, f.n*columns)
	work := make([]float64, f.n)
	answer := make([]float64, f.n)
	for k := 0; k < columns; k++ {
		if err = ctx.Err(); err != nil {
			return nil, err
		}
		for i := 0; i < f.n; i++ {
			work[i] = rhs[i*columns+k] / f.scale
			if err = factorCheckFiniteOutput(work[i]); err != nil {
				return nil, err
			}
		}
		if err = f.solveUnit(ctx, answer, work); err != nil {
			return nil, err
		}
		for i := 0; i < f.n; i++ {
			solution[i*columns+k] = answer[i]
		}
	}
	return factorBuildSolveResult(ctx, f.original, f.n, f.n, solution, rhs, columns, f.condition)
}

// Solve computes the full-column-rank least-squares solution using pivoted QR.
// For m>=n, it minimizes the Euclidean residual ||A*X-B||_2 column-wise and
// never silently returns a minimum-norm answer for underdetermined problems.
// Implementation:
//   - Stage 1: check context, receiver, B shape and finite coefficients.
//   - Stage 2: apply compact Q^T to each scaled RHS, back-substitute in R.
//   - Stage 3: undo the original column permutation and measure true residuals.
//
// Behavior highlights: Explicit Q is never allocated; every Solve is race-safe.
// Inputs: ctx non-nil, b read-only m-by-r, with r>=1.
// Returns: Detached n-by-r solution and r measured residuals, or nil on failure.
// Errors: ErrNilMatrix, ErrDimensionMismatch, ErrBadShape, ErrNaNInf,
// ErrOverflow, ctx.Err(), wrapped Matrix.At causes.
// Determinism: RHS columns asc, reflectors k asc, triangular substitution desc.
// Complexity: Time O(m*n*r+n*n*r) plus exact residuals; workspace O(m+n*r).
// Notes: Factor condition is kappa_1(R), not the 2-norm condition of A.
// AI-Hints: Do not multiply by an explicit Q or form A^T*A: both are needless.
func (f *QRFactor) Solve(ctx context.Context, b Matrix) (*SolveResult, error) {
	if err := factorContextErr(ctx); err != nil {
		return nil, err
	}
	if f == nil {
		return nil, ErrNilMatrix
	}
	rows, columns, err := factorInputShape(b)
	if err != nil {
		return nil, err
	}
	if rows != f.m {
		return nil, ErrDimensionMismatch
	}
	rhs, err := factorReadMatrix(ctx, b, rows, columns)
	if err != nil {
		return nil, err
	}
	solution := make([]float64, f.n*columns)
	work := make([]float64, f.m)
	pivoted := make([]float64, f.n)
	for k := 0; k < columns; k++ {
		if err = ctx.Err(); err != nil {
			return nil, err
		}
		for i := 0; i < f.m; i++ {
			work[i] = rhs[i*columns+k] / f.scale
			if err = factorCheckFiniteOutput(work[i]); err != nil {
				return nil, err
			}
		}
		if err = f.applyQTranspose(ctx, work); err != nil {
			return nil, err
		}
		if err = f.solveUpper(ctx, pivoted, work[:f.n]); err != nil {
			return nil, err
		}
		for j := 0; j < f.n; j++ {
			// perm[j] is the original column currently at position j.
			solution[f.perm[j]*columns+k] = pivoted[j]
		}
	}
	return factorBuildSolveResult(ctx, f.original, f.m, f.n, solution, rhs, columns, f.condition)
}

// Solve solves an accepted symmetric SPD system using Cholesky triangular passes.
// The input b is never mutated; the immutable factor supports simultaneous
// solves on different goroutines without locks or shared scratch buffers.
// Implementation:
//   - Stage 1: validate ctx and finite n-by-r RHS snapshot.
//   - Stage 2: scale each RHS, then solve L*y=b and L^T*x=y.
//   - Stage 3: measure residuals against the accepted symmetric A and publish.
//
// Behavior highlights: Approximate symmetry means solving the accepted midpoint A.
// Inputs: ctx non-nil, b read-only n-by-r Matrix, r>=1.
// Returns: Detached n-by-r solution with truthful kappa_1(acceptedA) diagnostic.
// Errors: ErrNilMatrix, ErrDimensionMismatch, ErrBadShape, ErrNaNInf,
// ErrOverflow, ctx.Err(), wrapped Matrix.At errors.
// Determinism: RHS columns asc, forward rows asc, backward rows desc.
// Complexity: Time O(n*n*r) plus exact residuals; memory O(n*r+n).
// Notes: SPD validation was performed at FactorCholesky, not silently repeated.
// AI-Hints: Never interpret nonzero symmetry tolerance as a diagonal regularizer.
func (f *CholeskyFactor) Solve(ctx context.Context, b Matrix) (*SolveResult, error) {
	if err := factorContextErr(ctx); err != nil {
		return nil, err
	}
	if f == nil {
		return nil, ErrNilMatrix
	}
	rows, columns, err := factorInputShape(b)
	if err != nil {
		return nil, err
	}
	if rows != f.n {
		return nil, ErrDimensionMismatch
	}
	rhs, err := factorReadMatrix(ctx, b, rows, columns)
	if err != nil {
		return nil, err
	}
	solution := make([]float64, f.n*columns)
	work := make([]float64, f.n)
	answer := make([]float64, f.n)
	for k := 0; k < columns; k++ {
		if err = ctx.Err(); err != nil {
			return nil, err
		}
		for i := 0; i < f.n; i++ {
			work[i] = rhs[i*columns+k] / f.scale
			if err = factorCheckFiniteOutput(work[i]); err != nil {
				return nil, err
			}
		}
		if err = f.solveUnit(ctx, answer, work); err != nil {
			return nil, err
		}
		for i := 0; i < f.n; i++ {
			solution[i*columns+k] = answer[i]
		}
	}
	return factorBuildSolveResult(ctx, f.accepted, f.n, f.n, solution, rhs, columns, f.condition)
}
