// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package lstm defines tensor, parameter, state and result carriers.
// This file fixes their shapes, ownership rules, and nilability contract.
package lstm

import "github.com/lvlath/go/matrix"

// Tensor3 stores a batch-major three-dimensional tensor with contiguous time rows.
// Its linear index is (batch*Time+time)*Features+feature, with no strides or views.
//
// Implementation:
//   - Stage 1: Infer checks B>=1, T>=0, D>=1 and the exact B*T*D length.
//   - Stage 2: the kernel reads each valid row in ascending time order.
//   - Stage 3: optional outputs use the same [B,T,H] layout and detached storage.
//
// Behavior highlights:
//   - A zero-time tensor has length zero and may have nil Data.
//   - All data, including masked tails, must be finite before inference starts.
//
// Inputs:
//   - Data: caller-owned float64 values when used as input, read-only during Infer.
//   - Batch: number of independent sequences, B>=1; Time: steps per row, T>=0.
//   - Features: vector width, D>=1 for input, H>=1 for returned Sequence.
//
// Returns:
//   - As Result.Sequence, Data is independently owned by the caller.
//
// Errors:
//   - ErrShape for inconsistent geometry; ErrNilInput for missing nonempty data;
//     ErrNonFinite for nonfinite values in input, including padding.
//
// Determinism:
//   - Ascending batch, then timestep, then feature; no implicit transposition.
//
// Complexity:
//   - Header O(1) space; data occupies O(B*T*Features) floats.
//
// Notes:
//   - Batch-first corresponds to PyTorch LSTM(batch_first=true) output layout.
//
// AI-Hints:
//   - Never interpret Time as a leading dimension or flatten as time-major.
type Tensor3 struct {
	Data     []float64 // Flat row-major values; not a matrix of row pointers.
	Batch    int       // Number of independent batch rows.
	Time     int       // Time slots (valid and padded) per batch row.
	Features int       // Numeric feature width for each time slot.
}

// Weights describes a single unidirectional PyTorch-compatible IFGO LSTM layer.
// Import checks and deeply copies the complete numerical payload into an immutable Model.
//
// Implementation:
//   - Stage 1: verify H>=1, D>=1 and matrices shaped [4H,D] and [4H,H].
//   - Stage 2: read every weight in ascending row-major order and require finiteness.
//   - Stage 3: normalize signed zero, freeze owned buffers, and calculate ModelID.
//
// Behavior highlights:
//   - Four adjacent H-row blocks always represent i, f, g, o in that order.
//   - Bias is one 4H-vector of already-combined input and recurrent biases.
//
// Inputs:
//   - Input: caller-owned matrix of input coefficients [4H,D].
//   - Recurrent: caller-owned recurrent coefficients [4H,H].
//   - Bias: caller-owned, finite already-combined [4H] slice.
//   - Hidden: H>=1, specifying each of the four gate-block widths.
//
// Returns:
//   - Import produces a detached Model and retains none of these inputs.
//
// Errors:
//   - ErrNilInput, ErrShape or ErrNonFinite during Import validation.
//
// Determinism:
//   - Gate blocks and both matrix row-major scans are fixed by the ABI.
//
// Complexity:
//   - Parameter storage O(H*(D+H)); copying and validation have the same order.
//
// Notes:
//   - Projections, bidirectional/multi-layer weights and training are unsupported.
//
// AI-Hints:
//   - Never swap g and o or silently omit bias_hh_l0 when importing from PyTorch.
type Weights struct {
	Input     matrix.Matrix // Input-to-hidden coefficients [4H,D].
	Recurrent matrix.Matrix // Hidden-to-hidden coefficients [4H,H].
	Bias      []float64     // Combined input/recurrent bias for the IFGO rows.
	Hidden    int           // Hidden and cell state width H, strictly positive.
}

// State contains the complete recurrent memory after a valid input prefix.
// Hidden and Cell are independently owned [B,H] matrices tagged with ModelID.
//
// Implementation:
//   - Stage 1: Infer checks the exact ModelID before accessing initial matrices.
//   - Stage 2: both matrices are validated and copied into local row-major vectors.
//   - Stage 3: a successful inference publishes fresh matrices as Final.
//
// Behavior highlights:
//   - A zero-length chunk preserves both states exactly, without gating or resets.
//   - Hidden values are not sufficient to resume a stream: Cell is also necessary.
//
// Inputs:
//   - Hidden: read-only [B,H] hidden output of the previous valid timestep.
//   - Cell: read-only [B,H] persistent cell memory at that timestep.
//   - ModelID: exact canonical model identity; user-provided labels are invalid.
//
// Returns:
//   - Infer returns detached matrices, never aliases a supplied initial State.
//
// Errors:
//   - ErrStateVersion for a foreign ModelID; ErrNilInput, ErrShape and ErrNonFinite
//     for absent, malformed or nonfinite matrices.
//
// Determinism:
//   - Row-major [B,H] traversal and exact version equality.
//
// Complexity:
//   - Two matrices each occupy O(B*H) binary64 cells.
//
// Notes:
//   - Mutating a State during an Infer call is not supported; afterward caller owns it.
//
// AI-Hints:
//   - Do not use the final padded Sequence slot as a substitute for State.Hidden.
type State struct {
	Hidden  matrix.Matrix // Final hidden activations [B,H].
	Cell    matrix.Matrix // Final persistent cell memory [B,H].
	ModelID string        // Lowercase canonical SHA-256 of imported weights.
}

// Result is the canonical inference artifact with optional per-step outputs.
// It contains no partial outputs on failure, and every successful buffer is detached.
//
// Implementation:
//   - Stage 1: Infer validates all inputs before running the recurrent kernel.
//   - Stage 2: the kernel fills optional valid positions, keeping masked ones +0.
//   - Stage 3: fresh final matrices and optional sequence are published atomically.
//
// Behavior highlights:
//   - Sequence is the zero Tensor3 when ReturnSequence=false.
//   - Final is always present and bound to the model, even for Time=0.
//
// Inputs:
//   - Produced only by successful Infer or its zero-math wrappers.
//
// Returns:
//   - Caller owns Sequence.Data, Final.Hidden and Final.Cell independently.
//
// Errors:
//   - Failed calls return nil *Result and an error; no result fields are usable.
//
// Determinism:
//   - Output features are published in ascending batch/time/hidden dimension order.
//
// Complexity:
//   - O(B*H) final state, optionally O(B*T*H) sequence storage.
//
// Notes:
//   - Implements core.Nilable through the nil-safe IsNil method.
//
// AI-Hints:
//   - Do not infer success from an empty Sequence: inspect error and Final.
type Result struct {
	Sequence Tensor3 // Detached [B,T,H] when enabled; otherwise entirely zero-valued.
	Final    State   // Detached hidden/cell matrices [B,H] and exact ModelID.
}

// IsNil reports whether the Result receiver itself is nil.
// This is the panic-free nilability convention used by lvlath algorithm results.
//
// Implementation:
//   - Stage 1: compare the receiver pointer directly with nil.
//   - Stage 2: return the boolean without reading any Result field.
//   - Stage 3: never dereference an absent result.
//
// Behavior highlights:
//   - Safe on a nil *Result, including a typed-nil inside a Nilable interface.
//
// Inputs:
//   - r: possibly nil inference result.
//
// Returns:
//   - bool: true only for a nil pointer, false for any allocated Result.
//
// Errors:
//   - None.
//
// Determinism:
//   - O(1) pointer identity comparison with no mutable global state.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - An allocated Result with zero fields is not itself a nil pointer.
//
// AI-Hints:
//   - Check the error from Infer; IsNil does not validate a Result's contents.
func (r *Result) IsNil() bool {
	return r == nil
}

// Model owns immutable IFGO row-major float64 weights and the canonical fingerprint.
// Only Import constructs a valid Model; a zero-value Model is not initialized.
//
// Implementation:
//   - Stage 1: store checked D/H dimensions and owned slices at Import.
//   - Stage 2: Infer reads those slices without any matrix-interface dispatch.
//   - Stage 3: publication exposes only new outputs and the immutable ID.
//
// Behavior highlights:
//   - Zero mutable global scratch and no in-place model updates.
//
// Inputs:
//   - Created using Import(Weights); do not construct &Model{} as a usable layer.
//
// Returns:
//   - Model.ID and Infer outcomes; parameter buffers are never exposed.
//
// Errors:
//   - ErrNilInput from Infer on a nil receiver; a zero-valued model fails shape checks.
//
// Determinism:
//   - Fixed parameter array order and single canonical recurrent kernel.
//
// Complexity:
//   - Stored weights occupy O(H*(D+H)) floats.
//
// Notes:
//   - Safe for concurrent read-only Infer; no lock is necessary.
//
// AI-Hints:
//   - Do not expose underlying wi/wr/bias slices as mutable views.
type Model struct {
	d, h         int       // Input width and hidden width established at import.
	wi, wr, bias []float64 // Private row-major input/recurrent weights and combined bias.
	id           string    // Canonical SHA-256 over normalized binary64 storage.
}
