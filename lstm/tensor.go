// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package lstm defines the checked-layout indexing primitive for row-major tensors.
// This file keeps tensor offsets independent from activation and numerical policy.
package lstm

// tensorOffset locates the first feature of one batch-first timestep row.
// It is intentionally unchecked only after Infer has verified all dimensions.
//
// Implementation:
//   - Stage 1: count preceding batch rows as batch*Time.
//   - Stage 2: add the zero-based timestep within this batch.
//   - Stage 3: scale by Features to obtain the flat scalar index.
//
// Behavior highlights:
//   - Every feature of a timestep occupies one contiguous slice.
//
// Inputs:
//   - input: previously validated [B,T,D] tensor header.
//   - batch: valid row index; timestep: a valid active row position.
//
// Returns:
//   - int: the first scalar index of input.Data at the requested position.
//
// Errors:
//   - None: the caller provides checked geometry and valid indices.
//
// Determinism:
//   - Fixed batch-first row-major address formula, no strides or indirection.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - The checked B*T*D product prevents index calculation overflow here.
//
// AI-Hints:
//   - A time-major index would silently shuffle samples across batch rows.
func tensorOffset(input Tensor3, batch, timestep int) int {
	return (batch*input.Time + timestep) * input.Features
}
