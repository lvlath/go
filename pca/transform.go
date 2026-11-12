// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pca applies the frozen PCA affine preprocessing and orthogonal basis.
// Each call owns its output and temporary buffers for safe concurrent inference.
package pca

import (
	"context"
	"math"

	"github.com/lvlath/go/matrix"
)

// ready reports whether a Model contains a complete fitted parameter layout.
// This guard separates pointer presence from the usable trained-state contract
// before inference methods read any private feature or loading data.
//
// Implementation:
//   - Stage 1: reject nil receivers and nonpositive fitted dimensions.
//   - Stage 2: require d training means and positive-scale storage positions.
//   - Stage 3: require a complete d-by-k private loading buffer.
//
// Behavior highlights:
//   - The method checks the internal layout, not the numerical values of already
//     certified fit parameters, which cannot be mutated through public APIs.
//
// Inputs:
//   - m: possibly nil or zero-valued *Model receiver; no ownership transfer.
//
// Returns:
//   - bool: true only when all private array lengths and dimensions agree.
//
// Errors:
//   - None; inference callers translate false into ErrNilInput.
//
// Determinism:
//   - Fixed short-circuit validation order and pure private field reads.
//
// Complexity:
//   - Time O(1), Space O(1) with no matrix allocations.
//
// Notes:
//   - Only Fit constructs a valid trained Model; invalid zero values are safe.
//
// AI-Hints:
//   - IsNil on FitResult tests pointer presence, not the validity of Model.
//   - Do not expose setters or mutable slices that could invalidate ready.
func (m *Model) ready() bool {
	return m != nil && m.d > 0 && m.k > 0 && len(m.means) == m.d &&
		len(m.scales) == m.d && len(m.loadings) == m.d*m.k
}

// Transform maps new observations to scores in the learned latent space.
// It never centers or standardizes against the inference batch, preserving
// the training-only parameter ownership boundary of the fitted Model.
//
// Implementation:
//   - Stage 1: validate context, fitted receiver, input shape, and all finite cells.
//   - Stage 2: center/scale each row with frozen means/scales and project via Vk.
//   - Stage 3: publish an independently allocated rows-by-k score matrix.
//
// Behavior highlights:
//   - A correctly shaped zero-row input returns a detached 0-by-k output.
//   - Subtraction overflow is retried as separately scaled operands when valid.
//   - math.FMA reduces intermediate rounding in fixed-order score dot products.
//
// Inputs:
//   - ctx: active non-nil context, checked before input access and per row.
//   - x: caller-owned, finite rows-by-d Matrix; rows>=0 and no concurrent mutation.
//
// Returns:
//   - matrix.Matrix: detached rows-by-k scores in fitted orthonormal coordinates.
//   - error: non-nil on failure; a partially computed score matrix is never returned.
//
// Errors:
//   - ErrInvalidOption: nil ctx; ErrNilInput: nil/unfitted model or input matrix.
//   - ErrEmptyInput, ErrShape: invalid feature extent or unexpected dimensions.
//   - ErrNonFinite: supplied NaN or infinity; ErrOverflow: unrepresentable score.
//   - ctx.Err() or preserved Matrix.At/Fill error: cancellation or dependency failure.
//
// Determinism:
//   - All input checks and arithmetic scan rows ascending, then features ascending;
//     score accumulators are addressed in retained component order.
//
// Complexity:
//   - Time O(rows*d*k+rows*d), extra space O(rows*k) for the staging output
//     plus O(rows*k) for the returned Dense; no persistent model workspace.
//
// Notes:
//   - Read-only calls are safe across many goroutines on the same fitted Model.
//   - The input is read twice for error precedence and projection; concurrent
//     mutations can invalidate the two scans and are explicitly unsupported.
//
// AI-Hints:
//   - Do not replace training means with fresh batch means or call FitStandardizer.
//   - Use *matrix.Dense for cheap At operations; never access its private buffer.
func (m *Model) Transform(ctx context.Context, x matrix.Matrix) (matrix.Matrix, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	if !m.ready() {
		return nil, ErrNilInput
	}
	rows, cols, err := checkedShape(x, 0)
	if err != nil {
		return nil, err
	}
	if cols != m.d {
		return nil, ErrShape
	}
	if err = scanFinite(ctx, x, rows, cols); err != nil {
		return nil, err
	}
	out, err := matrix.NewDense(rows, m.k)
	if err != nil {
		return nil, wrapDependency(err)
	}
	values := make([]float64, rows*m.k)
	for i := 0; i < rows; i++ {
		if err = ctx.Err(); err != nil {
			return nil, err
		}
		for col := 0; col < m.d; col++ {
			v, readErr := x.At(i, col)
			if readErr != nil {
				return nil, readErr
			}
			// Subtract before dividing for precision near the training mean.
			// When the subtraction itself overflows, scale the operands first;
			// opposite-sign operands cannot cancel in this fallback branch.
			difference := v - m.means[col]
			centered := difference / m.scales[col]
			if !finite(difference) {
				centered = v/m.scales[col] - m.means[col]/m.scales[col]
			}
			if !finite(centered) {
				return nil, ErrOverflow
			}
			for j := 0; j < m.k; j++ {
				index := i*m.k + j
				// A fused multiply-add preserves more of a nearly cancelling dot.
				values[index] = math.FMA(centered, m.loadings[col*m.k+j], values[index])
				if !finite(values[index]) {
					return nil, ErrOverflow
				}
			}
		}
	}
	if err = out.Fill(values); err != nil {
		return nil, wrapDependency(err)
	}
	return out, nil
}

// InverseTransform restores PCA scores to the original feature coordinate system.
// It reconstructs only the retained subspace and reverses frozen preprocessing;
// missing principal directions are not recovered or estimated from the scores.
//
// Implementation:
//   - Stage 1: validate context, fitted receiver, score width, and finite cells.
//   - Stage 2: reconstruct each feature with Vk^T from one buffered score row.
//   - Stage 3: apply fitted scales/means and publish a detached rows-by-d Dense.
//
// Behavior highlights:
//   - A 0-by-k score matrix becomes a legal detached 0-by-d feature matrix.
//   - Scores are read once per row after the finite scan, not d times per feature.
//   - math.FMA reduces rounding in both dot products and inverse affine scaling.
//
// Inputs:
//   - ctx: active non-nil context, checked before access and for each row.
//   - scores: caller-owned finite rows-by-k matrix; rows>=0, no concurrent writes.
//
// Returns:
//   - matrix.Matrix: detached rows-by-d reconstruction in original physical units.
//   - error: no partial result; the caller owns the complete returned matrix.
//
// Errors:
//   - ErrInvalidOption, ErrNilInput, ErrEmptyInput, ErrShape: admission failures.
//   - ErrNonFinite: score NaN or infinity; ErrOverflow: reconstruction overflow.
//   - ctx.Err() or preserved matrix access error: cancellation/dependency failure.
//
// Determinism:
//   - Each row is buffered in increasing score index; reconstructed features
//     are visited in ascending column order with the same component summation order.
//
// Complexity:
//   - Time O(rows*d*k+rows*k), extra space O(rows*d+k) including the staging
//     values, plus O(rows*d) for the output matrix; no shared workspace.
//
// Notes:
//   - Returned physical-unit errors are not the same as standardized-space SPE.
//   - Use Reconstruct for ordinary end-to-end projection/reconstruction workflows.
//
// AI-Hints:
//   - Never fit scales to scores or estimate a holdout batch mean here.
//   - Do not claim exact inverse recovery when the trained model discarded modes.
func (m *Model) InverseTransform(ctx context.Context, scores matrix.Matrix) (matrix.Matrix, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	if !m.ready() {
		return nil, ErrNilInput
	}
	rows, cols, err := checkedShape(scores, 0)
	if err != nil {
		return nil, err
	}
	if cols != m.k {
		return nil, ErrShape
	}
	if err = scanFinite(ctx, scores, rows, cols); err != nil {
		return nil, err
	}
	if _, err = checkedFloatElements(rows, m.d); err != nil {
		return nil, err
	}
	out, err := matrix.NewDense(rows, m.d)
	if err != nil {
		return nil, wrapDependency(err)
	}
	values := make([]float64, rows*m.d)
	// Buffer each score row once. Generic Matrix.At can be more expensive than
	// a slice load, and repeatedly reading k scores for every feature would
	// cost d*k interface calls per row instead of k.
	scoreRow := make([]float64, m.k)
	for i := 0; i < rows; i++ {
		if err = ctx.Err(); err != nil {
			return nil, err
		}
		for component := range scoreRow {
			value, readErr := scores.At(i, component)
			if readErr != nil {
				return nil, readErr
			}
			scoreRow[component] = value
		}
		for col := 0; col < m.d; col++ {
			var projected float64
			for component, score := range scoreRow {
				// Ascending component order is stable across repeated calls.
				projected = math.FMA(score, m.loadings[col*m.k+component], projected)
			}
			// One fused affine operation restores the original training units.
			value := math.FMA(projected, m.scales[col], m.means[col])
			if !finite(value) {
				return nil, ErrOverflow
			}
			values[i*m.d+col] = value
		}
	}
	if err = out.Fill(values); err != nil {
		return nil, wrapDependency(err)
	}
	return out, nil
}
