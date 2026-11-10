// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix implements reusable pivoted LU factors and preserves legacy LU.
// The reusable factor records deterministic original-row pivot identity; the
// historical two-matrix LU remains unpivoted and retains its original semantics.
package matrix

import (
	"context"
	"errors"
	"fmt"
	"math"
)

// factorLUKernel decomposes a finite square matrix using scaled partial pivoting.
// It stores both the packed LU and the original A without retaining caller data.
// Implementation:
//   - Stage 1: copy finite A and normalize its largest absolute coefficient.
//   - Stage 2: choose the largest absolute column pivot; exact ties use the
//     smallest original row ID, then update the Schur complement via FMA.
//   - Stage 3: check relative numerical rank and compute measured kappa_1(A).
//
// Behavior highlights: A zero pivot retains both ErrRankDeficient and ErrSingular.
// Inputs:
//   - ctx: non-nil cancellation context, checked at every pivot boundary.
//   - a: read-only finite n-by-n matrix with n>=1.
//   - opts: already validated by FactorLU, never mutated.
//
// Returns: Detached immutable *LUFactor, nil on any failed stage.
// Errors: ErrNilMatrix, ErrNonSquare, ErrBadShape, ErrNaNInf,
// ErrRankDeficient, ErrSingular, ErrIllConditioned, ErrOverflow and ctx.Err().
// Determinism: k asc, pivot candidate row asc, exact ties by original row.
// Complexity: Time O(n^3) factorization plus O(n^3) condition calculation;
// retained storage O(n^2), temporary workspace O(n).
// Notes: Scaling changes the internal packed units, not the solution of A*x=b.
// AI-Hints: Row swaps must include all columns, including multipliers from
// earlier pivots; swapping only the active trailing block corrupts L.
func factorLUKernel(ctx context.Context, a Matrix, opts FactorOptions) (*LUFactor, error) {
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
	packed, scale, err := factorScaleAndNormalize(original)
	if err != nil {
		return nil, errors.Join(ErrSingular, err)
	}
	perm := make([]int, rows)
	for i := range perm {
		perm[i] = i
	}

	// Retain the induced norm of A/scale before elimination overwrites its
	// entries; conditioning is scale-invariant, unlike the factor coefficients.
	oneNorm := factorOneNorm(packed, rows, rows)
	for k := 0; k < rows; k++ {
		if err = ctx.Err(); err != nil {
			return nil, err
		}
		pivotRow := k
		pivotMagnitude := math.Abs(packed[k*rows+k])
		for i := k + 1; i < rows; i++ {
			magnitude := math.Abs(packed[i*rows+k])
			if magnitude > pivotMagnitude || (magnitude == pivotMagnitude && perm[i] < perm[pivotRow]) {
				pivotRow, pivotMagnitude = i, magnitude
			}
		}
		if pivotMagnitude == 0 {
			return nil, errors.Join(ErrRankDeficient, ErrSingular)
		}
		if pivotRow != k {
			// The prefix contains previously computed L multipliers. Their
			// row identity must follow the same permutation as the trailing U.
			for j := 0; j < rows; j++ {
				packed[k*rows+j], packed[pivotRow*rows+j] = packed[pivotRow*rows+j], packed[k*rows+j]
			}
			perm[k], perm[pivotRow] = perm[pivotRow], perm[k]
		}
		pivot := packed[k*rows+k]
		for i := k + 1; i < rows; i++ {
			multiplier := packed[i*rows+k] / pivot
			packed[i*rows+k] = multiplier
			for j := k + 1; j < rows; j++ {
				// FMA rounds multiplier*upper + old only once; ordinary
				// multiply-then-subtract needlessly loses low-order bits.
				packed[i*rows+j] = math.FMA(-multiplier, packed[k*rows+j], packed[i*rows+j])
			}
		}
	}
	if factorNumericalRank(packed, rows, rows, factorRankCutoff(opts, rows, rows)) != rows {
		return nil, ErrRankDeficient
	}
	factor := &LUFactor{n: rows, original: original, packed: packed, perm: perm, scale: scale}
	condition, err := factorConditionFromBasisSolves(ctx, rows, oneNorm, factor.solveUnit)
	if err != nil {
		return nil, err
	}
	if err = factorConditionAccepted(condition, opts.ConditionLimit); err != nil {
		return nil, err
	}
	factor.condition = condition
	return factor, nil
}

// solveUnit solves the normalized LU system for one private right-hand side.
// It is used both by the public Solve path and the independent basis-column
// solves needed to measure the inverse matrix 1-norm.
// Implementation:
//   - Stage 1: gather the original RHS in the exact stored permutation order.
//   - Stage 2: forward-solve the unit-lower triangular system in ascending rows.
//   - Stage 3: back-solve the upper triangular system in descending rows,
//     checking block cancellation and finiteness of each computed coefficient.
//
// Behavior highlights:
//   - A fresh O(n) workspace prevents any alias with retained factor storage.
//
// Inputs:
//   - ctx: non-nil, checked at forward and backward row-block boundaries.
//   - dst, rhs: private n-vectors, never aliases of f's packed coefficients.
//
// Returns:
//   - nil after complete finite dst assignment; no public result is published here.
//
// Errors:
//   - ErrOverflow for unrepresentable triangular arithmetic; ctx.Err() on cancel.
//
// Determinism:
//   - P-gather in current row order; lower rows ascend, upper rows descend.
//
// Complexity:
//   - Time O(n^2), Space O(n) temporary; no factor data is copied.
//
// Notes:
//   - The public Solve method scales RHS by the same factor scale beforehand.
//
// AI-Hints:
//   - Do not swap only trailing U rows during factorization: earlier lower
//     multipliers must follow the identical P mapping used in this solve.
func (f *LUFactor) solveUnit(ctx context.Context, dst, rhs []float64) error {
	n := f.n
	// Construct the permuted RHS independently of dst, which may alias rhs.
	work := make([]float64, n)
	for i := 0; i < n; i++ {
		if i&31 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		work[i] = rhs[f.perm[i]]
		for j := 0; j < i; j++ {
			work[i] = math.FMA(-f.packed[i*n+j], work[j], work[i])
		}
	}
	for i := n - 1; i >= 0; i-- {
		if i&31 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		for j := i + 1; j < n; j++ {
			work[i] = math.FMA(-f.packed[i*n+j], dst[j], work[i])
		}
		dst[i] = work[i] / f.packed[i*n+i]
		if err := factorCheckFiniteOutput(dst[i]); err != nil {
			return err
		}
	}
	return nil
}

// Permutation returns the original-row mapping used in P*A=L*U.
// At index i, the returned value is the original input row now occupying
// the i-th pivoted row. The detached slice is safe to mutate independently.
// Implementation: copy the private permutation once; no internal alias escapes.
// Behavior highlights: A nil factor returns nil instead of panicking.
// Inputs: The receiver must be a successfully constructed LUFactor for data.
// Returns: A new []int of length n, or nil for a nil receiver.
// Errors: None.
// Determinism: Indices are returned in ascending current row order.
// Complexity: Time O(n), Space O(n).
// Notes: Calling Solve afterward is unaffected by changes to the returned slice.
// AI-Hints: Do not invert this mapping: perm[i] is the original row at i.
func (f *LUFactor) Permutation() []int {
	if f == nil {
		return nil
	}
	result := make([]int, len(f.perm))
	copy(result, f.perm)
	return result
}

// ---------- Legacy non-pivoted matrix decomposition ----------

// LU computes the Doolittle factorization A = L*U with unit diagonal on L (no pivoting).
// Implementation:
//   - Stage 1: Validate m (not nil, square); allocate Dense L,U; set diag(L)=1.
//   - Stage 2: For i=0..n-1, build row i of U and column i of L in fixed order.
//
// Behavior highlights:
//   - Deterministic loops; fast path uses direct flat indexing; zero-pivot guard enforced.
//
// Inputs:
//   - m: square Matrix (n×n).
//
// Returns:
//   - Matrix: L (unit lower triangular).
//   - Matrix: U (upper triangular).
//
// Errors:
//   - ErrNilMatrix, ErrNonSquare, ErrSingular (if U[i,i]==0 during factorization).
//
// Determinism:
//   - Fixed i→{j≥i} for U, then {j>i}→i for L.
//
// Complexity:
//   - Time O(n^3), Space O(n^2).
//
// Notes:
//   - This compatibility kernel uses no pivoting. Its algebraic output is L,U,
//     not the pivoted P,L,U relation implemented by FactorLU.
//
// AI-Hints:
//   - Determinism does not require disabling pivoting; prefer FactorLU and
//     (*LUFactor).Solve for reliable systems with a zero leading diagonal.
//   - Do not attempt to infer the row permutation from these two matrices.
func LU(m Matrix) (Matrix, Matrix, error) {
	// Validate input non‐nil and square
	if err := ValidateNotNil(m); err != nil {
		return nil, nil, matrixErrorf(opLU, err)
	}
	if err := ValidateSquare(m); err != nil {
		return nil, nil, matrixErrorf(opLU, err)
	}

	// Allocate L and U
	n := m.Rows()
	Lraw, err := NewDense(n, n)
	if err != nil {
		return nil, nil, matrixErrorf(opLU, err)
	}
	Uraw, err := NewDense(n, n)
	if err != nil {
		return nil, nil, matrixErrorf(opLU, err)
	}

	// Initialize L diagonal to 1 (unit lower triangular)
	for i := 0; i < n; i++ {
		Lraw.data[i*n+i] = 1.0
	}

	// Detect fast‐path on *Dense
	// mRaw holds the input data if m is *Dense
	mRaw, useFast := m.(*Dense)
	var i, j, k int // loop iterators
	var sum, pivot float64
	// Execute Doolittle decomposition
	if useFast {
		// Fast‐path: operate directly on flat slices
		var baseI, baseJ int
		for i = 0; i < n; i++ {
			// Compute U[i][j] for j >= i
			for j = i; j < n; j++ {
				sum = ZeroSum
				baseI = i * n
				for k = 0; k < i; k++ {
					sum += Lraw.data[baseI+k] * Uraw.data[k*n+j]
				}
				Uraw.data[baseI+j] = mRaw.data[baseI+j] - sum
			}

			// Zero-pivot guard (deterministic singularity detection)
			if Uraw.data[i*n+i] == ZeroPivot {
				return nil, nil, matrixErrorf(opLU, ErrSingular)
			}

			// Compute L[j][i] for j > i
			for j = i + 1; j < n; j++ {
				sum = ZeroSum
				baseJ = j * n
				for k = 0; k < i; k++ {
					sum += Lraw.data[baseJ+k] * Uraw.data[k*n+i]
				}
				pivot = Uraw.data[i*n+i]
				Lraw.data[baseJ+i] = (mRaw.data[baseJ+i] - sum) / pivot
			}
		}
	} else {
		// Fallback: generic interface version
		var a, l, u float64
		for i = 0; i < n; i++ {
			// Compute U[i][j] for j >= i
			for j = i; j < n; j++ {
				sum = ZeroSum
				for k = 0; k < i; k++ {
					l, err = Lraw.At(i, k)
					if err != nil {
						return nil, nil, matrixErrorf(opLU, fmt.Errorf("At(%d,%d): %w", i, k, err))
					}
					u, err = Uraw.At(k, j)
					if err != nil {
						return nil, nil, matrixErrorf(opLU, fmt.Errorf("At(%d,%d): %w", k, j, err))
					}
					sum += l * u
				}
				a, err = m.At(i, j)
				if err != nil {
					return nil, nil, matrixErrorf(opLU, fmt.Errorf("At(%d,%d): %w", i, j, err))
				}
				if err = Uraw.Set(i, j, a-sum); err != nil {
					return nil, nil, matrixErrorf(opLU, fmt.Errorf("Set(%d,%d): %w", i, j, err))
				}
			}

			// Zero-pivot guard (generic path)
			pivot, err = Uraw.At(i, i)
			if err != nil {
				return nil, nil, matrixErrorf(opLU, fmt.Errorf("At(%d,%d): %w", i, i, err))
			}
			if pivot == ZeroPivot {
				return nil, nil, matrixErrorf(opLU, ErrSingular)
			}

			// Compute L[j][i] for j > i
			for j = i + 1; j < n; j++ {
				sum = ZeroSum
				for k = 0; k < i; k++ {
					l, err = Lraw.At(j, k)
					if err != nil {
						return nil, nil, matrixErrorf(opLU, fmt.Errorf("At(%d,%d): %w", j, k, err))
					}
					u, err = Uraw.At(k, i)
					if err != nil {
						return nil, nil, matrixErrorf(opLU, fmt.Errorf("At(%d,%d): %w", k, i, err))
					}
					sum += l * u
				}
				a, err = m.At(j, i)
				if err != nil {
					return nil, nil, matrixErrorf(opLU, fmt.Errorf("At(%d,%d): %w", j, i, err))
				}
				pivot, err = Uraw.At(i, i)
				if err != nil {
					return nil, nil, matrixErrorf(opLU, fmt.Errorf("At(%d,%d): %w", i, i, err))
				}
				if err = Lraw.Set(j, i, (a-sum)/pivot); err != nil {
					return nil, nil, matrixErrorf(opLU, fmt.Errorf("Set(%d,%d): %w", j, i, err))
				}
			}
		}
	}

	// Return L and U
	return Lraw, Uraw, nil
}
