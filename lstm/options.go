// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package lstm defines the explicit output materialization policy for inference.
// This file isolates public options from recurrent mathematics and storage types.
package lstm

// Options controls whether Infer materializes outputs for every valid timestep.
// The zero value publishes only Final and therefore avoids the large sequence buffer.
//
// Implementation:
//   - Stage 1: read ReturnSequence as a value-copy boolean configuration.
//   - Stage 2: Infer allocates [B,T,H] only when ReturnSequence is true.
//   - Stage 3: the same kernel always computes and publishes detached Final states.
//
// Behavior highlights:
//   - This option changes output materialization, not the LSTM recurrence or masking.
//
// Inputs:
//   - ReturnSequence: false for final-state only; true for [B,T,H] sequence output.
//
// Returns:
//   - Result.Sequence: empty zero value when disabled, zero-padded when enabled.
//
// Errors:
//   - No invalid boolean value exists; allocation geometry is validated by Infer.
//
// Determinism:
//   - Neither setting changes gate order, reduction order, or state values.
//
// Complexity:
//   - Optional storage O(B*T*H) floats and O(B*T*H) zero initialization.
//
// Notes:
//   - Prefer InferFinal for final state or InferSequence for all step features.
//
// AI-Hints:
//   - Do not mistake ReturnSequence=false for suppressing Final publication.
type Options struct {
	// ReturnSequence enables the detached [B,T,H] result with exact-zero padding.
	ReturnSequence bool
}
