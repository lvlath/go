// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package lstm implements checked initial-state ingestion and detached publication.
// This file converts between matrix.Matrix state carriers and private row-major vectors.
package lstm

import (
	"errors"

	"github.com/lvlath/go/matrix"
)

// stateVectors prepares independent, finite hidden and cell workspaces for a batch.
// It preserves an explicit initial state or creates two positive-zero matrices.
//
// Implementation:
//   - Stage 1: check B*H before any workspace allocation.
//   - Stage 2: when initial is present, compare ModelID before inspecting matrices.
//   - Stage 3: validate/copy both [B,H] matrices into separate flat row-major slices.
//
// Behavior highlights:
//   - Both state components are required; hidden alone cannot resume an LSTM.
//   - The nil initial case creates exact zero buffers without reading external data.
//
// Inputs:
//   - m: valid imported Model; batch: input B>=1; initial: optional caller state.
//
// Returns:
//   - []float64, []float64: detached hidden and cell buffers of length B*H.
//   - error: nil only when both buffers are safely initialized.
//
// Errors:
//   - ErrShape: checked product overflow or wrong matrix dimensions.
//   - ErrStateVersion: initial.ModelID differs from m.ID before matrix inspection.
//   - ErrNilInput: one or both initial matrices are absent.
//   - ErrNonFinite: an initial-state scalar contains NaN or infinity.
//   - ErrShape + underlying adapter root: a Matrix.At call fails.
//
// Determinism:
//   - Version equality, then hidden row-major copy, then cell row-major copy.
//
// Complexity:
//   - Time O(B*H), Space O(B*H) for both vectors.
//
// Notes:
//   - Caller mutation during the copy is not supported; copied values are private.
//
// AI-Hints:
//   - Validate ModelID first: accepting a foreign cell silently corrupts the stream.
func (m *Model) stateVectors(batch int, initial *State) ([]float64, []float64, error) {
	count, err := checkedProduct(batch, m.h)
	if err != nil {
		return nil, nil, err
	}
	if initial == nil {
		return make([]float64, count), make([]float64, count), nil
	}
	if initial.ModelID != m.id {
		return nil, nil, ErrStateVersion
	}
	if !matrixPresent(initial.Hidden) || !matrixPresent(initial.Cell) {
		return nil, nil, ErrNilInput
	}

	hidden, err := matrixCopy(initial.Hidden, batch, m.h)
	if err != nil {
		return nil, nil, err
	}
	cell, err := matrixCopy(initial.Cell, batch, m.h)
	if err != nil {
		return nil, nil, err
	}
	return hidden, cell, nil
}

// publishMatrix exports one successfully computed state into detached matrix storage.
// It never returns the mutable per-call kernel workspace to a caller.
//
// Implementation:
//   - Stage 1: allocate a fresh Dense with the validated [B,H] shape.
//   - Stage 2: assign every finite vector value by ascending row-major index.
//   - Stage 3: return the new matrix only after every assignment succeeds.
//
// Behavior highlights:
//   - The result has its own backing memory and may be mutated after inference.
//
// Inputs:
//   - values: private finite B*H vector, not retained by the returned matrix.
//   - batch, hidden: positive, previously checked matrix dimensions.
//
// Returns:
//   - matrix.Matrix: new [B,H] state snapshot on success.
//
// Errors:
//   - ErrShape + lower-level cause: matrix allocation rejects geometry.
//   - ErrOverflow + lower-level cause: an unexpected matrix Set failure.
//
// Determinism:
//   - Sequential index assignment with row=index/H and col=index%H.
//
// Complexity:
//   - Time O(B*H), Space O(B*H) for the newly allocated dense state.
//
// Notes:
//   - Valid, finite internal inputs make these dependency errors unreachable normally.
//
// AI-Hints:
//   - Do not export an alias to hidden/cell storage: later calls must be independent.
func publishMatrix(values []float64, batch, hidden int) (matrix.Matrix, error) {
	result, err := matrix.NewDense(batch, hidden)
	if err != nil {
		return nil, errors.Join(ErrShape, err)
	}
	for index, value := range values {
		if err = result.Set(index/hidden, index%hidden, value); err != nil {
			// Input invariants guarantee valid indices and finite values on this path.
			return nil, errors.Join(ErrOverflow, err)
		}
	}
	return result, nil
}
