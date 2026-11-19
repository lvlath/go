// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package lstm defines the public boundary for deterministic recurrent inference.
// This file selects explicit output/state policies and delegates to one numerical kernel.
package lstm

import "context"

// Import constructs an immutable single-layer model from PyTorch-ordered IFGO weights.
// The model owns normalized copies of every matrix element and the combined bias.
//
// Implementation:
//   - Stage 1: validate hidden size, matrix presence, shapes, and checked element counts.
//   - Stage 2: copy and validate the complete row-major weights and combined bias.
//   - Stage 3: normalize negative zero, compute the canonical SHA-256 ID, and publish.
//
// Behavior highlights:
//   - Gate blocks are input i, forget f, candidate g, output o, in that exact order.
//   - Bias must already contain bias_ih_l0 + bias_hh_l0, including float64 rounding.
//   - Neither the passed matrices nor Bias are retained by the returned model.
//
// Inputs:
//   - weights: finite Input [4H,D], Recurrent [4H,H], Bias [4H], and Hidden=H>=1.
//     Matrices are caller-owned and must not be concurrently modified during import.
//
// Returns:
//   - *Model: owned read-only parameters and a stable ModelID; nil on any error.
//
// Errors:
//   - ErrNilInput: missing matrix or required bias storage.
//   - ErrShape: invalid dimensions, counts, or matrix geometry.
//   - ErrNonFinite: NaN or infinity in any weight or bias element.
//   - An underlying Matrix.At error is retained alongside ErrShape.
//
// Determinism:
//   - Input and recurrent matrices are copied in ascending row-then-column order.
//   - ID uses the fixed IFGO ABI, little-endian words, and normalized +0 encoding.
//
// Complexity:
//   - Time O(H*(D+H)), Space O(H*(D+H)) for copied weights and hashing.
//
// Notes:
//   - This is not a checkpoint parser, multi-layer importer, or training API.
//   - A successful model is safe for concurrent calls with read-only inputs.
//
// AI-Hints:
//   - Do not pass Keras/ONNX gate arrays without an explicitly verified conversion.
//   - matrix.Matrix.At is used only during import, never in the recurrent hot loop.
func Import(weights Weights) (*Model, error) {
	return importWeights(weights)
}

// ID returns the canonical fingerprint of the model's normalized parameters.
// This fingerprint binds streaming states to the exact input/recurrent weights and bias.
//
// Implementation:
//   - Stage 1: handle a nil receiver without dereferencing it.
//   - Stage 2: read the immutable import-time ID.
//   - Stage 3: return the ID without exposing parameter storage.
//
// Behavior highlights:
//   - The ID is a lowercase 64-character SHA-256 hexadecimal string.
//
// Inputs:
//   - m: immutable imported model; a nil receiver is permitted.
//
// Returns:
//   - string: canonical ModelID, or an empty string for a nil receiver.
//
// Errors:
//   - None.
//
// Determinism:
//   - Constant-time read of the ID computed at Import; no rehashing or global state.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - ID is not an authenticity signature: SHA-256 does not authenticate the source.
//
// AI-Hints:
//   - Persist a verified ID together with each detached State, not a file name.
func (m *Model) ID() string {
	if m == nil {
		return ""
	}
	return m.id
}

// Infer evaluates one deterministic, single-layer unidirectional LSTM batch.
// It advances only unmasked steps and publishes detached final state and optional outputs.
//
// Implementation:
//   - Stage 1: validate context, tensor geometry, local lengths, initial state, and all data.
//   - Stage 2: allocate private hidden/cell vectors and optional zero-initialized outputs;
//     invoke the sole recurrent kernel in batch, time, gate-row, then feature order.
//   - Stage 3: recheck cancellation, publish fresh [B,H] matrices, and return Result.
//
// Behavior highlights:
//   - With lengths=nil every sample has Time valid steps; otherwise lengths[b] is local.
//   - Masked positions preserve both states and contain exactly +0.0 in Sequence.
//   - Time=0 and lengths[b]=0 preserve initial state, or the zero state when absent.
//   - ReturnSequence=false returns an entirely zero-valued Sequence without its buffer.
//   - Any failed operation returns (nil, error), never a partially advanced result.
//
// Inputs:
//   - ctx: non-nil execution context, cancellation honored before and during computation.
//   - input: caller-owned row-major [B,T,D] float64 tensor, B>=1, T>=0, D=model.D.
//   - lengths: nil or B valid per-sample step counts in [0,T], not global offsets.
//   - initial: nil for zero states or [B,H] Hidden/Cell with exactly matching ModelID.
//   - options: ReturnSequence selects whether [B,T,H] outputs are materialized.
//
// Returns:
//   - *Result: detached Sequence (when requested) and [B,H] Final, always owned by caller.
//   - error: nil on success, otherwise a nil Result and no published intermediate state.
//
// Errors:
//   - ErrInvalidOption: nil context or invalid lengths.
//   - ctx.Err(): cancellation, including before validation and after computation.
//   - ErrNilInput: nil model, required absent tensor data, or absent state matrices.
//   - ErrShape: incompatible tensor/state geometry or checked allocation size overflow.
//   - ErrStateVersion: initial state belongs to a different model.
//   - ErrNonFinite: any NaN/Inf in input, including padding, or in initial state.
//   - ErrOverflow: exact affine result or computed cell/hidden value is unrepresentable.
//
// Determinism:
//   - Batch ascending, then timestep, IFGO gate-row and feature indices ascending.
//   - Each affine uses fixed-order compensated summation and exact-rational rescue.
//
// Complexity:
//   - Time O(B*T*H*(D+H)+B*T*D), Space O(B*H+H+B*T*H if sequence enabled).
//   - Constant-time, fixed-size bookkeeping is excluded; published state uses O(B*H).
//
// Notes:
//   - A model is read-only and concurrent-safe. The caller must not mutate input,
//     initial state, or prior Results concurrently with any Infer reading them.
//   - Across chunks, use Final as the next initial state and supply chunk-local lengths.
//
// AI-Hints:
//   - Do not carry the last valid hidden vector into padded output positions.
//   - Do not interpret the returned hidden features as calibrated predictions:
//     a task-specific trained head and feature normalization are caller concerns.
func (m *Model) Infer(ctx context.Context, input Tensor3, lengths []int, initial *State, options Options) (*Result, error) {
	// Check the context before touching user tensors or the model, preserving precedence.
	if ctx == nil {
		return nil, ErrInvalidOption
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if m == nil {
		return nil, ErrNilInput
	}

	// Validate shape before mask and state; check every supplied sample, even padding.
	if err := validateInput(input, m.d, lengths); err != nil {
		return nil, err
	}
	hidden, cell, err := m.stateVectors(input.Batch, initial)
	if err != nil {
		return nil, err
	}
	if err = scanFiniteContext(ctx, input.Data); err != nil {
		return nil, err
	}

	// The sequence is allocated only on request; untouched masked cells remain +0.
	var sequence Tensor3
	if options.ReturnSequence {
		n, countErr := checkedProduct(input.Batch, input.Time)
		if countErr != nil {
			return nil, countErr
		}
		n, countErr = checkedProduct(n, m.h)
		if countErr != nil {
			return nil, countErr
		}
		sequence = Tensor3{
			Data:     make([]float64, n),
			Batch:    input.Batch,
			Time:     input.Time,
			Features: m.h,
		}
	}

	// Work exclusively on per-call buffers; errors discard every intermediate result.
	if err = m.kernel(ctx, input, lengths, hidden, cell, sequence.Data); err != nil {
		return nil, err
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}

	// Construct independent matrix-backed state only after a successful full pass.
	hiddenMatrix, err := publishMatrix(hidden, input.Batch, m.h)
	if err != nil {
		return nil, err
	}
	cellMatrix, err := publishMatrix(cell, input.Batch, m.h)
	if err != nil {
		return nil, err
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	return &Result{
		Sequence: sequence,
		Final: State{
			Hidden:  hiddenMatrix,
			Cell:    cellMatrix,
			ModelID: m.id,
		},
	}, nil
}

// InferDefault extracts the final recurrent state from a complete unmasked batch.
// It is the minimal-memory starting point for consumers that need an embedding per row.
//
// Implementation:
//   - Stage 1: choose full-length samples, zero initial state, and default options.
//   - Stage 2: delegate every check and recurrent operation to Infer.
//   - Stage 3: return the canonical Result without discarding its metadata.
//
// Behavior highlights:
//   - The zero Options value disables Sequence allocation, not Final publication.
//
// Inputs:
//   - ctx: non-nil execution context.
//   - input: caller-owned finite [B,T,D] tensor with all T steps active.
//
// Returns:
//   - *Result: detached Final [B,H] and empty Sequence, or nil on error.
//
// Errors:
//   - Identical sentinels and precedence to Infer(ctx, input, nil, nil, Options{}).
//
// Determinism:
//   - Delegates unchanged batch, time and gate traversal to Infer.
//
// Complexity:
//   - Time O(B*T*H*(D+H)+B*T*D), Space O(B*H+H).
//
// Notes:
//   - Use Infer when local masks or nonzero initial state are required.
//
// AI-Hints:
//   - This wrapper does not skip validation or silently infer sequence lengths.
func (m *Model) InferDefault(ctx context.Context, input Tensor3) (*Result, error) {
	return m.Infer(ctx, input, nil, nil, Options{})
}

// InferFinal advances a possibly masked batch and publishes only the final states.
// It avoids the [B,T,H] sequence allocation when downstream work needs one state per row.
//
// Implementation:
//   - Stage 1: select ReturnSequence=false while retaining mask and initial state.
//   - Stage 2: delegate validation, inference and ownership handling to Infer.
//   - Stage 3: return the Result containing detached Final matrices.
//
// Behavior highlights:
//   - Zero-length rows preserve their input state without emitting sequence values.
//
// Inputs:
//   - ctx: active context; input: finite [B,T,D] row-major values.
//   - lengths: nil or B local valid-step counts; initial: nil or compatible State.
//
// Returns:
//   - *Result: Final with caller-owned matrices, empty Sequence; nil on error.
//
// Errors:
//   - Identical to Infer(ctx, input, lengths, initial, Options{}).
//
// Determinism:
//   - Exactly the canonical Infer ordering; no second kernel.
//
// Complexity:
//   - Time O(B*T*H*(D+H)+B*T*D), Space O(B*H+H).
//
// Notes:
//   - Unlike InferDefault, this wrapper accepts a previous streaming State.
//
// AI-Hints:
//   - Passing a zero length does not reset a previous cell; it keeps it intact.
func (m *Model) InferFinal(ctx context.Context, input Tensor3, lengths []int, initial *State) (*Result, error) {
	return m.Infer(ctx, input, lengths, initial, Options{})
}

// InferSequence returns a full zero-padded output tensor plus final recurrent states.
// It is intended for temporal feature extraction that needs a feature at each valid step.
//
// Implementation:
//   - Stage 1: explicitly enable ReturnSequence while preserving mask and initial state.
//   - Stage 2: delegate to the single canonical Infer recurrence.
//   - Stage 3: publish the standard detached Result.
//
// Behavior highlights:
//   - Sequence has [B,T,H] geometry even when individual lengths are zero.
//
// Inputs:
//   - ctx: active context; input: finite row-major [B,T,D] tensor.
//   - lengths: nil or one local length per batch row; initial: nil or matching State.
//
// Returns:
//   - *Result: Sequence with exact-zero padded tails and detached Final; nil on error.
//
// Errors:
//   - Identical to Infer(ctx, input, lengths, initial, Options{ReturnSequence:true}).
//
// Determinism:
//   - Exactly the ascending batch, timestep and IFGO ordering of Infer.
//
// Complexity:
//   - Time O(B*T*H*(D+H)+B*T*D), Space O(B*H+H+B*T*H).
//
// Notes:
//   - Padding samples are still checked for NaN and infinity before inference.
//
// AI-Hints:
//   - Do not use Sequence's last padded position as the final state; use Final.
func (m *Model) InferSequence(ctx context.Context, input Tensor3, lengths []int, initial *State) (*Result, error) {
	return m.Infer(ctx, input, lengths, initial, Options{ReturnSequence: true})
}

// InferChunk continues a stream from a previously completed Result.
// It removes State plumbing while preserving the exact canonical recurrence and masks.
//
// Implementation:
//   - Stage 1: check context and require a non-nil prior Result.
//   - Stage 2: pass previous.Final as the new initial State to Infer.
//   - Stage 3: return the detached next Result or nil on error.
//
// Behavior highlights:
//   - A previous result can originate from InferDefault, InferFinal, or InferSequence.
//   - A missing prior result is an error, not an implicit stream reset.
//
// Inputs:
//   - ctx: non-nil context; chunk: caller-owned [B,Tchunk,D] local data.
//   - lengths: nil or B step counts within this chunk, not absolute stream lengths.
//   - previous: non-nil completed Result; its Final is read-only during this call.
//   - options: standard sequence materialization policy for the new chunk.
//
// Returns:
//   - *Result: next detached Final and optional output, or nil on any failure.
//
// Errors:
//   - ErrInvalidOption: nil context; ctx.Err(): pre-canceled context.
//   - ErrNilInput: missing prior Result or nil model, plus Infer's input conditions.
//   - ErrStateVersion, ErrShape, ErrNonFinite, ErrOverflow: delegated from Infer.
//
// Determinism:
//   - Delegates the same IFGO kernel and unmodified previous-state values to Infer.
//
// Complexity:
//   - Time O(B*Tchunk*H*(D+H)+B*Tchunk*D), Space O(B*H+H+optional sequence).
//
// Notes:
//   - For the first chunk, call Infer with initial=nil rather than InferChunk.
//   - Caller mutation of previous.Final during this call is unsupported.
//
// AI-Hints:
//   - Recompute per-row lengths for each chunk; reusing global lengths is incorrect.
func (m *Model) InferChunk(ctx context.Context, chunk Tensor3, lengths []int, previous *Result, options Options) (*Result, error) {
	if ctx == nil {
		return nil, ErrInvalidOption
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if m == nil || previous == nil {
		return nil, ErrNilInput
	}
	return m.Infer(ctx, chunk, lengths, &previous.Final, options)
}
