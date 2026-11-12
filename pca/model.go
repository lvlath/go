// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pca provides detached access to fitted loadings and composition-only
// reconstruction helpers. Model training and projection kernels remain elsewhere.
package pca

import (
	"context"

	"github.com/lvlath/go/matrix"
)

// Components copies the frozen loading matrix into a new d-by-k Dense.
// The returned matrix can be changed freely without changing the trained Model.
//
// Implementation:
//   - Stage 1: reject nil or unfitted receivers without dereferencing them.
//   - Stage 2: allocate a matrix with the original d-by-k loading dimensions.
//   - Stage 3: copy the private row-major values into independently owned storage.
//
// Behavior highlights:
//   - Loading columns are orthonormal within numerical solver tolerances.
//   - Their signs are canonicalized; repeated eigenvalues may have rotating bases.
//
// Inputs:
//   - m: fitted Model; the receiver is immutable and safe for concurrent readers.
//
// Returns:
//   - matrix.Matrix: detached d-by-k loadings, or nil for an invalid receiver.
//
// Errors:
//   - None in the signature; invalid receivers and allocation construction
//     failures produce nil because this getter has no error return channel.
//
// Determinism:
//   - Copy order is the Model's original row-major feature/component ordering.
//
// Complexity:
//   - Time O(d*k), space O(d*k), including the entire new backing buffer.
//
// Notes:
//   - A matrix returned earlier is not a writable view of Model parameters.
//   - The caller must not assume a fixed basis inside a repeated eigenspace.
//
// AI-Hints:
//   - Compare Vk*Vk^T when verifying a repeated-eigenvalue subspace.
//   - Do not add a mutable loadings getter; use this detached copy.
func (m *Model) Components() matrix.Matrix {
	// A zero-value Model has no fitted dimensions; it must not be dereferenced.
	if m == nil || m.d <= 0 || m.k <= 0 || len(m.loadings) != m.d*m.k {
		return nil
	}

	// The public matrix constructor owns its flat backing slice independently.
	out, err := matrix.NewDense(m.d, m.k)
	if err != nil {
		return nil
	}
	if err = out.Fill(m.loadings); err != nil {
		return nil
	}
	return out
}

// Reconstruct projects a feature batch and restores it in original feature units.
// It is a convenience composition of Transform and InverseTransform, not a fit.
//
// Implementation:
//   - Stage 1: Transform validates x and produces scores using frozen parameters.
//   - Stage 2: InverseTransform reconstructs the retained orthogonal subspace.
//   - Stage 3: return the detached reconstruction; never publish a partial matrix.
//
// Behavior highlights:
//   - With k less than the rank, reconstruction is generally lossy.
//   - Standardized models restore original feature units after projection.
//   - The input is not mutated and no test-batch means are computed.
//
// Inputs:
//   - ctx: active non-nil context, checked at both component operations.
//   - x: caller-owned nnew-by-d finite feature matrix; nnew may equal zero.
//
// Returns:
//   - matrix.Matrix: independently allocated nnew-by-d reconstructed observations.
//   - error: any projection or inverse error, with no partial reconstruction.
//
// Errors:
//   - ErrInvalidOption, ErrNilInput, ErrShape, ErrNonFinite, ErrOverflow:
//     the exact validation or arithmetic failure of the delegated methods.
//   - ctx.Err() and wrapped matrix errors: cancellation or storage failure.
//
// Determinism:
//   - Transform accumulates feature coordinates ascending; InverseTransform
//     accumulates retained components ascending for every row and feature.
//
// Complexity:
//   - Time O(nnew*d*k+nnew*d), space O(nnew*(d+k)+d) including both outputs.
//
// Notes:
//   - The method uses per-call workspace and may run concurrently on one Model.
//   - Residuals in physical feature units need explicit calibration: dimensions
//     with unlike measurement units cannot be squared and summed blindly.
//
// AI-Hints:
//   - Fit and select k on training data; calibrate anomaly limits separately.
//   - This method is not an inverse for discarded dimensions and cannot restore
//     information outside the retained subspace.
func (m *Model) Reconstruct(ctx context.Context, x matrix.Matrix) (matrix.Matrix, error) {
	// Forward projection performs the canonical inference input validation.
	scores, err := m.Transform(ctx, x)
	if err != nil {
		return nil, err
	}

	// Inverse projection uses the same private fitted means, scales, and loadings.
	return m.InverseTransform(ctx, scores)
}
