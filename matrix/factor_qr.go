// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix implements pivoted Householder QR factors and preserves legacy QR.
// The compact reusable factor avoids constructing Q in Solve, while legacy QR
// retains its historical square-matrix result orientation and error behavior.
package matrix

import (
	"context"
	"fmt"
	"math"
)

// factorTrailingNorm computes a stable trailing Euclidean column norm.
// Pivot selection must not square unscaled candidates into +Inf or zero.
// Implementation:
//   - Stage 1: Initialize the Blue/Golub scale and sum-of-squares state.
//   - Stage 2: Visit rows from start to rows-1, rescaling when a larger magnitude appears.
//   - Stage 3: Restore the final norm after the sum-of-squares square root.
//
// Behavior highlights:
//   - No full copy, Gram matrix or downdated stale norm is needed.
//
// Inputs:
//   - data is row-major current QR; start and column identify one trailing segment.
//
// Returns:
//   - Nonnegative finite Euclidean norm of that segment.
//
// Errors:
//   - None for already-normalized finite packed values.
//
// Determinism:
//   - Rows ascend and each candidate is evaluated in a fixed column order.
//
// Complexity:
//   - Time O(rows-start), Space O(1).
//
// Notes:
//   - The factor recomputes norms at each pivot for numerical stability.
//
// AI-Hints:
//   - Do not use sqrt(sum(value*value)) for arbitrary large or tiny columns.
func factorTrailingNorm(data []float64, rows, cols, start, column int) float64 {
	scale, sumSquares := 0.0, 1.0
	for i := start; i < rows; i++ {
		magnitude := math.Abs(data[i*cols+column])
		if magnitude == 0 {
			continue
		}
		if magnitude > scale {
			ratio := scale / magnitude
			sumSquares = 1 + sumSquares*ratio*ratio
			scale = magnitude
		} else {
			ratio := magnitude / scale
			sumSquares += ratio * ratio
		}
	}
	return scale * math.Sqrt(sumSquares)
}

// factorQRKernel builds a pivoted economy Householder QR decomposition.
// The factorization is expressed as A[:,perm]/scale=Q*R, with m>=n>=1.
// Implementation:
//   - Stage 1: validate the shape, capture a detached original matrix and
//     normalize it by a single positive input magnitude.
//   - Stage 2: recompute all trailing candidate norms at each pivot, resolving
//     equal norms by original column identity; store each Householder vector
//     under the diagonal and apply it to subsequent columns.
//   - Stage 3: reject deficient numerical rank, measure kappa_1(R) by basis
//     solves, and enforce the optional condition limit before publishing.
//
// Behavior highlights:
//   - Never materializes the full m-by-m Q or the normal-equation Gram matrix.
//   - Recomputing trailing norms every pivot is a stability-first alternative
//     to the DGEQP3 norm-downdate/recompute heuristic, with the same asymptotic
//     O(m*n*n) arithmetic regime and a larger constant factor.
//
// Inputs:
//   - ctx: non-nil context checked at each pivot boundary.
//   - a: finite, read-only tall matrix with m>=n>=1.
//   - opts: validated relative rank and optional condition policy.
//
// Returns: Immutable *QRFactor, nil on all errors and cancellation.
// Errors: ErrBadShape (wide/empty), ErrNaNInf, ErrRankDeficient,
// ErrIllConditioned, ErrOverflow, ctx.Err(), wrapped Matrix.At errors.
// Determinism: k asc; candidates j asc; exact ties by original perm[j].
// Complexity: Time O(m*n*n+n^3); factor + original storage O(m*n).
// Notes: Full rank is mandatory; no least-norm fallback or SVD is performed.
// AI-Hints: Never apply a Householder reflector to previously finalized R
// columns, and never expose the packed below-diagonal vectors as matrix Q.
func factorQRKernel(ctx context.Context, a Matrix, opts FactorOptions) (*QRFactor, error) {
	rows, cols, err := factorInputShape(a)
	if err != nil {
		return nil, err
	}
	if rows < cols {
		return nil, ErrBadShape
	}
	original, err := factorReadMatrix(ctx, a, rows, cols)
	if err != nil {
		return nil, err
	}
	packed, scale, err := factorScaleAndNormalize(original)
	if err != nil {
		return nil, err
	}
	perm := make([]int, cols)
	tau := make([]float64, cols)
	for j := range perm {
		perm[j] = j
	}

	// Each column is selected by its exact (current, floating-point) trailing
	// norm rather than an accumulated downdate that could approach zero early.
	for k := 0; k < cols; k++ {
		if err = ctx.Err(); err != nil {
			return nil, err
		}
		selected := k
		selectedNorm := factorTrailingNorm(packed, rows, cols, k, k)
		for j := k + 1; j < cols; j++ {
			candidate := factorTrailingNorm(packed, rows, cols, k, j)
			if candidate > selectedNorm || (candidate == selectedNorm && perm[j] < perm[selected]) {
				selected, selectedNorm = j, candidate
			}
		}
		if selected != k {
			// The entire column is swapped, including previous reflector
			// application results in the upper R panel. Earlier stored
			// reflectors occupy other columns and remain untouched.
			for i := 0; i < rows; i++ {
				packed[i*cols+k], packed[i*cols+selected] = packed[i*cols+selected], packed[i*cols+k]
			}
			perm[k], perm[selected] = perm[selected], perm[k]
		}

		alpha := packed[k*cols+k]
		tailNorm := factorTrailingNorm(packed, rows, cols, k+1, k)
		if tailNorm == 0 {
			// An already axial column needs no reflection. The diagonal
			// alpha is its signed R pivot; rank is checked after all panels.
			tau[k] = 0
			continue
		}
		beta := -math.Copysign(math.Hypot(alpha, tailNorm), alpha)
		coefficient := (beta - alpha) / beta
		denominator := alpha - beta
		if err = factorCheckFiniteOutput(coefficient); err != nil {
			return nil, err
		}
		tau[k] = coefficient
		packed[k*cols+k] = beta
		for i := k + 1; i < rows; i++ {
			// The implicit leading element of Householder v is exactly 1;
			// only its tail is stored below the diagonal in packed QR.
			packed[i*cols+k] /= denominator
		}

		// H=I-tau*v*v^T is symmetric and orthogonal. Applying H to the
		// trailing panel never creates the full Q, preserving O(m*n) storage.
		for j := k + 1; j < cols; j++ {
			dot := packed[k*cols+j]
			for i := k + 1; i < rows; i++ {
				dot = math.FMA(packed[i*cols+k], packed[i*cols+j], dot)
			}
			update := coefficient * dot
			packed[k*cols+j] -= update
			for i := k + 1; i < rows; i++ {
				packed[i*cols+j] = math.FMA(-packed[i*cols+k], update, packed[i*cols+j])
			}
		}
	}
	if factorNumericalRank(packed, cols, cols, factorRankCutoff(opts, rows, cols)) != cols {
		return nil, ErrRankDeficient
	}

	factor := &QRFactor{m: rows, n: cols, original: original, packed: packed, tau: tau, perm: perm, scale: scale}
	condition, err := factorConditionFromBasisSolves(ctx, cols, factor.upperOneNorm(), factor.solveUpper)
	if err != nil {
		return nil, err
	}
	if err = factorConditionAccepted(condition, opts.ConditionLimit); err != nil {
		return nil, err
	}
	factor.condition = condition
	return factor, nil
}

// upperOneNorm measures the induced 1-norm of the stored triangular R.
// The reflector tails occupy the same packed buffer but must not enter kappa_1(R).
// Implementation:
//   - Stage 1: For each R column, begin an absolute sum at row zero.
//   - Stage 2: Accumulate only rows i<=j above the triangular diagonal.
//   - Stage 3: Return the largest column sum.
//
// Behavior highlights:
//   - Below-diagonal Householder data is intentionally excluded.
//
// Inputs:
//   - f is an immutable successfully constructed QRFactor.
//
// Returns:
//   - Finite positive 1-norm of pivoted R.
//
// Errors:
//   - None for accepted full-rank factors.
//
// Determinism:
//   - Columns and within-column rows are increasing.
//
// Complexity:
//   - Time O(n^2), Space O(1).
//
// Notes:
//   - R is stored in the normalized factor coordinate system.
//
// AI-Hints:
//   - Never use the 1-norm of the entire packed QR buffer as ||R||_1.
func (f *QRFactor) upperOneNorm() float64 {
	maximum := 0.0
	for j := 0; j < f.n; j++ {
		sum := 0.0
		for i := 0; i <= j; i++ {
			sum += math.Abs(f.packed[i*f.n+j])
		}
		if sum > maximum {
			maximum = sum
		}
	}
	return maximum
}

// solveUpper performs deterministic back substitution in pivoted R coordinates.
// It is shared by least-squares solves and basis-column condition evaluation.
// Implementation:
//   - Stage 1: Copy caller RHS into one per-call working vector.
//   - Stage 2: Traverse rows backward, removing already solved upper-triangle terms.
//   - Stage 3: Divide by the nonzero diagonal and validate each computed unknown.
//
// Behavior highlights:
//   - The stored factor remains immutable and concurrency-safe.
//
// Inputs:
//   - ctx is non-nil; dst and rhs are private n-vectors.
//
// Returns:
//   - nil when all dst coefficients are finite, otherwise a classified error.
//
// Errors:
//   - ErrOverflow or ctx.Err() checked at block boundaries.
//
// Determinism:
//   - Descending row order and ascending trailing columns.
//
// Complexity:
//   - Time O(n^2), Space O(n).
//
// Notes:
//   - Its output uses the pivoted column basis; unpermute at public Solve boundary.
//
// AI-Hints:
//   - Never publish dst before all finite and residual postconditions succeed.
func (f *QRFactor) solveUpper(ctx context.Context, dst, rhs []float64) error {
	work := make([]float64, f.n)
	copy(work, rhs)
	for i := f.n - 1; i >= 0; i-- {
		if i&31 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		for j := i + 1; j < f.n; j++ {
			work[i] = math.FMA(-f.packed[i*f.n+j], dst[j], work[i])
		}
		dst[i] = work[i] / f.packed[i*f.n+i]
		if err := factorCheckFiniteOutput(dst[i]); err != nil {
			return err
		}
	}
	return nil
}

// applyQTranspose applies stored Householder reflectors to one RHS vector.
// This computes Q^T*b without materializing a full m-by-m orthogonal matrix.
// Implementation:
//   - Stage 1: Visit compact reflectors from first to last, checking context.
//   - Stage 2: Form the dot product of each implicit unit-leading v and the current RHS.
//   - Stage 3: Subtract tau*v*dot in place, preserving the remaining trailing components.
//
// Behavior highlights:
//   - Zero tau denotes an already axial column requiring no change.
//
// Inputs:
//   - ctx non-nil; work is a private mutable m-vector; f is immutable.
//
// Returns:
//   - nil after updating work, or ctx.Err() on cancellation.
//
// Errors:
//   - Context errors only; reflector data is validated during construction.
//
// Determinism:
//   - Reflector k asc; each tail row asc.
//
// Complexity:
//   - Time O(m*n), Space O(1) auxiliary.
//
// Notes:
//   - Only work[:n] later enters the triangular solve.
//
// AI-Hints:
//   - Do not reverse reflector application order when computing Q^T*b.
func (f *QRFactor) applyQTranspose(ctx context.Context, work []float64) error {
	for k := 0; k < f.n; k++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		if f.tau[k] == 0 {
			continue
		}
		dot := work[k]
		for i := k + 1; i < f.m; i++ {
			dot = math.FMA(f.packed[i*f.n+k], work[i], dot)
		}
		update := f.tau[k] * dot
		work[k] -= update
		for i := k + 1; i < f.m; i++ {
			work[i] = math.FMA(-f.packed[i*f.n+k], update, work[i])
		}
	}
	return nil
}

// Permutation returns the original-column mapping for A[:,perm]=Q*R.
// The returned independent slice records original column IDs in their
// current pivoted order; caller mutation cannot affect subsequent solves.
// Implementation: Copy the private mapping exactly once.
// Behavior highlights: A nil receiver yields nil without panic.
// Inputs: A successfully constructed *QRFactor, or nil.
// Returns: []int of length n detached from private factor state.
// Errors: None.
// Determinism: Current column index increases from zero to n-1.
// Complexity: Time O(n), Space O(n).
// Notes: This zero-based convention is independent of LAPACK's Fortran indexing.
// AI-Hints: Assign solution[perm[j]]=pivotedSolution[j]; never reverse the map.
func (f *QRFactor) Permutation() []int {
	if f == nil {
		return nil
	}
	result := make([]int, len(f.perm))
	copy(result, f.perm)
	return result
}

// ---------- Legacy square QR decomposition (Q^T orientation) ----------

// QR computes a Householder-based factorization such that A ≈ Qᵀ * R.
// Implementation:
//   - Stage 1: Validate m (not nil, square); clone A; init Q to identity.
//   - Stage 2: For k=0..n-1, build a column reflector and apply it to A (forming R) and to Q.
//
// Behavior highlights:
//   - Deterministic column order; fast path uses raw data access; no sign canonicalization inside.
//
// Inputs:
//   - m: square Matrix (n×n).
//
// Returns:
//   - Matrix: Q (accumulated reflectors; note that A ≈ Qᵀ * R, not Q*R).
//   - Matrix: R (upper triangular after reflections).
//
// Errors:
//   - ErrNilMatrix, ErrNonSquare.
//
// Determinism:
//   - Fixed k→{i,j} visitation; stable column-wise accumulation.
//
// Complexity:
//   - Time O(n^3), Space O(n^2).
//
// Notes:
//   - If you need A≈Q*R with diag(R)≥0, post-process via S=diag(sign(R[ii,ii])): use (SQ, SR) with SR≥0.
//
// AI-Hints:
//   - For tall-skinny, prefer blocked/TSQR variants outside this package when cache behavior matters more.
//   - The core mathematical operations (Householder reflections or Gram-Schmidt)
//     are kept in a single monolithic block to ensure optimal CPU cache locality
//     and clear top-to-bottom step alignment with standard mathematical textbooks.
//
// nolint:gocyclo
func QR(m Matrix) (Matrix, Matrix, error) {
	// Validate input non‐nil and square
	if err := ValidateNotNil(m); err != nil {
		return nil, nil, matrixErrorf(opQR, err)
	}
	if err := ValidateSquare(m); err != nil {
		return nil, nil, matrixErrorf(opQR, err)
	}
	n := m.Rows()

	// Prepare working copy A and orthogonal accumulator Q
	Araw := m.Clone()
	Qraw, err := NewDense(n, n)
	if err != nil {
		return nil, nil, matrixErrorf(opQR, err)
	}
	// initialize Q to identity: Q[i,i]=1
	for i := 0; i < n; i++ {
		Qraw.data[i*n+i] = 1.0
	}

	// Detect fast‐path on *Dense for A
	Ad, useFast := Araw.(*Dense)

	// Allocate Householder vector
	v := make([]float64, n)
	buf := make([]float64, n) // reuse: stores A[i,j] or Q[i,j] for i in [k..n)
	// Perform Householder reflections
	var (
		i, j, k    int     // loop indices
		norm, beta float64 // vector norm and β = vᵀv
		alpha, tau float64 // reflection scalar and 2/β factor
		sum, aij   float64 // accumulators and temporary values

	)
	for k = 0; k < n; k++ {
		// 4.1: Compute norm of A[k:n][k]
		norm = NormZero
		if useFast {
			for i = k; i < n; i++ {
				aij = Ad.data[i*n+k]
				norm += aij * aij
			}
		} else {
			for i = k; i < n; i++ {
				aij, err = Araw.At(i, k)
				if err != nil {
					return nil, nil, matrixErrorf(opQR, fmt.Errorf("At(%d,%d): %w", i, k, err))
				}
				norm += aij * aij
			}
		}
		norm = math.Sqrt(norm)
		if norm == NormZero {
			continue // skip zero column
		}

		// 4.2: Compute alpha = -sign(A[k,k]) * norm
		if useFast {
			aij = Ad.data[k*n+k]
		} else {
			aij, err = Araw.At(k, k)
			if err != nil {
				return nil, nil, matrixErrorf(opQR, fmt.Errorf("At(%d,%d): %w", k, k, err))
			}
		}
		alpha = -math.Copysign(norm, aij)

		// 4.3: Build Householder vector v
		for i = 0; i < n; i++ {
			v[i] = 0.0
		}
		if useFast {
			for i = k; i < n; i++ {
				v[i] = Ad.data[i*n+k]
			}
		} else {
			for i = k; i < n; i++ {
				v[i], err = Araw.At(i, k)
				if err != nil {
					return nil, nil, matrixErrorf(opQR, fmt.Errorf("At(%d,%d): %w", i, k, err))
				}
			}
		}
		v[k] -= alpha

		// 4.4: Compute β = vᵀv and τ = 2/β
		beta = NormZero
		for i = k; i < n; i++ {
			beta += v[i] * v[i]
		}
		// Guard: avoid division by zero if v is degenerate (should be rare but must be safe).
		if beta == NormZero {
			continue
		}
		tau = 2.0 / beta

		// 4.5: Apply reflection to A (update R)
		for j = k; j < n; j++ {
			sum = ZeroSum
			if useFast {
				for i = k; i < n; i++ {
					sum += v[i] * Ad.data[i*n+j]
				}
				for i = k; i < n; i++ {
					Ad.data[i*n+j] -= tau * v[i] * sum
				}
			} else {
				// 1) read once into buf + accumulate
				for i = k; i < n; i++ {
					buf[i], err = Araw.At(i, j)
					if err != nil {
						return nil, nil, matrixErrorf(opQR, fmt.Errorf("At(%d,%d): %w", i, j, err))
					}
					sum += v[i] * buf[i]
				}
				// 2) write using buffered values
				for i = k; i < n; i++ {
					aij = buf[i] - tau*v[i]*sum
					if err = Araw.Set(i, j, aij); err != nil {
						return nil, nil, matrixErrorf(opQR, fmt.Errorf("Set(%d,%d): %w", i, j, err))
					}
				}
			}
		}

		// 4.6: Apply reflection to Q
		for j = 0; j < n; j++ {
			sum = ZeroSum
			for i = k; i < n; i++ {
				sum += v[i] * Qraw.data[i*n+j]
			}
			for i = k; i < n; i++ {
				Qraw.data[i*n+j] -= tau * v[i] * sum
			}
		}
	}

	// Finalize R = Araw and return Q, R
	return Qraw, Araw, nil
}
