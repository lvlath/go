// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ols exposes immutable fitted regression parameters without aliasing.
// This file owns coefficient access and nil-receiver behavior.
package ols

// Coefficients returns the original-unit slopes and intercept of a fitted Model.
// The coefficient slice is an independent allocation: caller edits cannot
// modify later predictions, other callers' results, or the fitted Model.
//
// Implementation:
//   - Stage 1: reject nil or never-fitted receivers without dereferencing fields.
//   - Stage 2: allocate a new p-element coefficient slice and copy model-owned data.
//   - Stage 3: publish the copy and the scalar intercept in original response units.
//
// Behavior highlights:
//   - Slopes correspond one-to-one with original training columns.
//   - For a NoIntercept model, the returned intercept is exactly positive zero.
//
// Inputs:
//   - m: fitted immutable Model receiver; nil and zero-value receivers are permitted.
//
// Returns:
//   - []float64: detached beta vector length p, or nil for an unfitted receiver.
//   - float64: intercept b, or +0 when receiver is nil/unfitted.
//
// Errors:
//   - None; this fixed API cannot return error, so nil/unfitted is a safe zero result.
//
// Determinism:
//   - Coefficients are copied in ascending original column order.
//
// Complexity:
//   - Time O(p), Space O(p) for the independent returned slice.
//
// Notes:
//   - Read-only calls are safe concurrently with Predict on the same fitted Model.
//   - Do not interpret (nil,0) from an unfitted receiver as a valid zero-factor fit.
//
// AI-Hints:
//   - Returning m.beta directly would break the immutable concurrent-prediction contract.
//   - This method exposes physical coefficients, never QR-normalized theta values.
func (m *Model) Coefficients() ([]float64, float64) {
	if m == nil || !m.fitted {
		return nil, 0
	}
	out := make([]float64, len(m.beta))
	copy(out, m.beta)
	return out, m.intercept
}
