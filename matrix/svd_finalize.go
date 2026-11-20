// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix publishes stable, detached economy SVD artifacts.
// This file handles singular-value ordering, basis mapping, rank, and result ownership.
package matrix

import (
	"context"
	"sort"
)

// svdMode identifies an independently scaled working column after Jacobi sweeps.
// The singular magnitude is rounded to binary64 only at the output boundary.
type svdMode struct {
	index    int
	singular float64
}

// svdBuildResult assembles the rectangular economy decomposition after Jacobi sweeps.
// Final singular values are restored once and ranked without squaring or forming a Gram matrix.
//
// Implementation:
//   - Stage 1: restore values and stably sort modes in descending magnitude.
//   - Stage 2: construct U from nonzero columns and complete exact-zero modes.
//   - Stage 3: map wide duality, canonicalize signs, and publish detached Dense values.
//
// Behavior highlights:
//   - Rank is advisory; no small nonzero singular mode is truncated.
//   - A budget-exhausted caller receives this last finite diagnostic factorization.
//
// Inputs: active ctx, private final Jacobi workspace, original m,n, sweep metadata.
// Returns: Non-aliased U, VT, singular-value slice and diagnostic flags.
// Errors: ErrOverflow for unrepresentable output; ErrNotConverged for impossible
// basis completion; ctx.Err() cancels publication.
// Determinism: Stable descending singular sort; ties retain original column order.
// Complexity: Time O(m*k*k+k*log(k)), Space O((m+n)*k), k=min(m,n).
// Notes: A nonconverged result may not yet have orthogonal singular vectors.
// AI-Hints: Complete only zero working columns; a numeric rank cutoff is not
// permission to replace a nonzero singular vector with a standard basis vector.
func svdBuildResult(
	ctx context.Context,
	state *svdWorkspace,
	m, n, iterations int,
	converged bool,
) (*SVDResult, error) {
	k := state.cols
	modes := make([]svdMode, k)
	for j := 0; j < k; j++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		value, err := svdPhysicalNorm(state.norms[j], state.exponents[j])
		if err != nil {
			return nil, err
		}
		modes[j] = svdMode{index: j, singular: value}
	}
	sort.SliceStable(modes, func(i, j int) bool {
		return modes[i].singular > modes[j].singular
	})

	values := make([]float64, k)
	left := make([]float64, state.rows*k)
	right := make([]float64, k*k)
	nonzero := make([]bool, k)
	for destination, mode := range modes {
		values[destination] = mode.singular
		// Use the normalized column norm, not rounded physical singular value:
		// even a subnormal output must retain the associated nonzero direction.
		length := state.norms[mode.index]
		if length != 0 {
			nonzero[destination] = true
			for row := 0; row < state.rows; row++ {
				left[row*k+destination] = state.data[row*k+mode.index] / length
			}
		}
		for row := 0; row < k; row++ {
			right[row*k+destination] = state.right[row*k+mode.index]
		}
	}
	if err := svdCompleteZeroColumns(ctx, left, state.rows, k, nonzero); err != nil {
		return nil, err
	}

	// For wide original A, the tall decomposition was applied to A^T:
	// A^T = left*Sigma*right^T, hence A = right*Sigma*left^T.
	leftData := make([]float64, m*k)
	rightData := make([]float64, k*n)
	if m >= n {
		copy(leftData, left)
		for mode := 0; mode < k; mode++ {
			for col := 0; col < n; col++ {
				rightData[mode*n+col] = right[col*k+mode]
			}
		}
	} else {
		copy(leftData, right)
		for mode := 0; mode < k; mode++ {
			for col := 0; col < n; col++ {
				rightData[mode*n+col] = left[col*k+mode]
			}
		}
	}
	svdCanonicalSigns(leftData, rightData, m, k, n)

	// Rank selection is relative to the largest nonnegative singular value.
	// Dividing by s[0] avoids overflow in max(m,n)*epsilon*s[0].
	maxDimension := max(m, n)
	cutoff := float64(maxDimension) * svdMachineEpsilon
	rank := 0
	if values[0] > 0 {
		for _, value := range values {
			if value/values[0] > cutoff {
				rank++
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	u, err := NewDense(m, k)
	if err != nil {
		return nil, err
	}
	vt, err := NewDense(k, n)
	if err != nil {
		return nil, err
	}
	copy(u.data, leftData)
	copy(vt.data, rightData)
	return &SVDResult{
		U:              u,
		VT:             vt,
		SingularValues: values,
		Rank:           rank,
		Iterations:     iterations,
		Converged:      converged,
	}, nil
}
