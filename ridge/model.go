// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ridge provides detached access to immutable fitted model parameters.
// Inference and external coefficient access have separate ownership boundaries.
package ridge

// Coefficients returns independently owned physical-unit slopes and intercept.
// It does not expose the optimization slopes used in the regularization penalty.
//
// Implementation:
//   - Stage 1: return the safe zero result for a nil or unfitted Model.
//   - Stage 2: allocate and copy the complete physical slope vector.
//   - Stage 3: return the copied slopes and the immutable scalar intercept.
//
// Behavior highlights:
//   - A model fitted with NoIntercept returns exactly positive zero for intercept.
//   - Caller mutation of returned coefficients cannot alter Model.Predict.
//
// Inputs:
//   - m: a potentially nil fitted Model receiver; no external state is read.
//
// Returns:
//   - []float64: detached slopes in original feature order, nil on nil/unfitted m.
//   - float64: physical intercept, or zero on nil/unfitted m.
//
// Errors:
//   - None; missing or zero-valued receivers return (nil, 0).
//
// Determinism:
//   - Coefficients preserve the original left-to-right feature order.
//
// Complexity:
//   - Time O(p), Space O(p) for a fitted receiver; O(1) otherwise.
//
// Notes:
//   - Concurrent calls on a fitted immutable Model are race-free.
//   - Returned beta is not the same penalty coordinate as theta when standardized.
//
// AI-Hints:
//   - Do not return m.beta directly; that would leak mutable model storage.
//   - Do not reconstruct Penalty from this slice for Standardize=true models.
func (m *Model) Coefficients() ([]float64, float64) {
	if m == nil || !m.fitted {
		return nil, 0
	}
	coefficients := make([]float64, len(m.beta))
	copy(coefficients, m.beta)
	return coefficients, m.intercept
}
