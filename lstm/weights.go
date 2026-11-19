// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package lstm implements the immutable weight-import and canonical ID machinery.
// This file reads matrix interfaces only at import and never retains caller storage.
package lstm

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"hash"
	"math"

	"github.com/lvlath/go/matrix"
)

const modelHashPrefix = "lstm/ifgo-f64/v1" // ABI/schema domain separator for canonical SHA-256.

// matrixCopy reads every element of a Matrix into an owned contiguous float64 slice.
// It removes interface dispatch from the recurrent kernel and checks data at the boundary.
//
// Implementation:
//   - Stage 1: reject nil/typed-nil matrices and enforce the exact expected shape.
//   - Stage 2: check rows*cols before allocation.
//   - Stage 3: read ascending row/column cells, require finiteness, normalize -0 to +0.
//
// Behavior highlights:
//   - Works with Dense and custom Matrix implementations without unsafe data access.
//   - Does not rely on Matrix.Clone's concrete type or numeric-policy metadata.
//
// Inputs:
//   - source: caller-owned read-only matrix during this call.
//   - rows, cols: expected positive shape supplied by checked import/state geometry.
//
// Returns:
//   - []float64: detached row-major cells; nil slice with error on failure.
//
// Errors:
//   - ErrNilInput: missing or Nilable-typed-nil matrix.
//   - ErrShape: incompatible dimensions or unchecked element-count overflow.
//   - ErrNonFinite: a source cell is NaN or infinite.
//   - ErrShape and original lower-level error: Matrix.At returns an error.
//
// Determinism:
//   - Always visit row 0..rows-1 and column 0..cols-1, independent of backend.
//
// Complexity:
//   - Time O(rows*cols), Space O(rows*cols).
//
// Notes:
//   - Concurrent mutation of source by its owner is unsupported.
//
// AI-Hints:
//   - Never retain source or expose a Dense backing slice to avoid model aliasing.
func matrixCopy(source matrix.Matrix, rows, cols int) ([]float64, error) {
	if !matrixPresent(source) {
		return nil, ErrNilInput
	}
	if source.Rows() != rows || source.Cols() != cols {
		return nil, ErrShape
	}
	count, err := checkedProduct(rows, cols)
	if err != nil {
		return nil, err
	}
	copyValues := make([]float64, count)
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			value, atErr := source.At(row, col)
			if atErr != nil {
				// Preserve both LSTM classification and the matrix adapter's root cause.
				return nil, errors.Join(ErrShape, atErr)
			}
			if !finite(value) {
				return nil, ErrNonFinite
			}
			if value == 0 {
				value = 0 // Canonical positive zero in model weights and ID.
			}
			copyValues[row*cols+col] = value
		}
	}
	return copyValues, nil
}

// hashWord appends a canonical little-endian unsigned 64-bit word to SHA-256.
// It makes hash input independent of the host architecture's byte order.
//
// Implementation:
//   - Stage 1: reserve an eight-byte local buffer.
//   - Stage 2: encode one word with binary.LittleEndian.
//   - Stage 3: feed bytes to the fixed SHA-256 digest implementation.
//
// Behavior highlights:
//   - Float coefficients are passed as their exact math.Float64bits representation.
//
// Inputs:
//   - digest: live SHA-256 hash state, owned by Import; word: one 64-bit payload.
//
// Returns:
//   - No value; the digest absorbs exactly eight bytes.
//
// Errors:
//   - None for sha256.Hash.Write, which cannot fail.
//
// Determinism:
//   - Exactly eight bytes in little-endian order, irrespective of CPU endian.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This helper must not use decimal string conversion or platform native words.
//
// AI-Hints:
//   - Normalize -0 before calling math.Float64bits for canonical model storage.
func hashWord(digest hash.Hash, word uint64) {
	var encoded [8]byte
	binary.LittleEndian.PutUint64(encoded[:], word)
	_, _ = digest.Write(encoded[:]) // sha256.Hash.Write cannot fail.
}

// modelHash computes the canonical versioned SHA-256 of the imported layer.
// This fingerprint binds state checkpoints to exact finite normalized coefficients.
//
// Implementation:
//   - Stage 1: feed the fixed ASCII ABI prefix and 64-bit little-endian D/H.
//   - Stage 2: append Input, Recurrent and Bias bits in that exact order.
//   - Stage 3: format the final 32-byte digest as lowercase hexadecimal.
//
// Behavior highlights:
//   - Signed int64 shape encoding equals unsigned LE64 because D and H are positive.
//   - Every float64 bit pattern represents canonical storage, including normalized +0.
//
// Inputs:
//   - d,h: already validated positive sizes; all weight slices: finite, detached.
//
// Returns:
//   - string: 64-character lowercase model identity.
//
// Errors:
//   - None: input was checked and SHA-256/binary writes use supported primitives.
//
// Determinism:
//   - Version prefix -> dimensions -> Input -> Recurrent -> Bias, fixed word order.
//
// Complexity:
//   - Time O(H*(D+H)), Space O(1) transient digest plus the 64-byte result.
//
// Notes:
//   - The hash is identity, not a cryptographic authenticity signature.
//
// AI-Hints:
//   - Do not include file names or timestamps; never alter gate or matrix order.
func modelHash(d, h int, inputWeights, recurrentWeights, bias []float64) string {
	digest := sha256.New()
	_, _ = digest.Write([]byte(modelHashPrefix))
	// A validated nonnegative int fits into int64 on both 32- and 64-bit Go.
	// For positive dimensions, signed LE int64 has the exact LE64 bit encoding.
	// This encoding avoids a signed-to-unsigned conversion before hashing.
	_ = binary.Write(digest, binary.LittleEndian, int64(d))
	_ = binary.Write(digest, binary.LittleEndian, int64(h))
	for _, block := range [][]float64{inputWeights, recurrentWeights, bias} {
		for _, value := range block {
			hashWord(digest, math.Float64bits(value))
		}
	}
	return hex.EncodeToString(digest.Sum(nil))
}

// importWeights validates and snapshots a single PyTorch-compatible IFGO layer.
// It is the sole import builder called by the public Import facade.
//
// Implementation:
//   - Stage 1: validate H, matrix presence, input width, exact shapes and counts.
//   - Stage 2: deep-copy finite input/recurrent matrices and combined bias.
//   - Stage 3: canonicalize signed zero, compute ModelID and publish immutable Model.
//
// Behavior highlights:
//   - Missing Bias is ErrNilInput; an allocated wrong-length Bias is ErrShape.
//   - No source slice or matrix pointer is retained after return.
//
// Inputs:
//   - weights: immutable-by-caller during import, IFGO matrices [4H,D]/[4H,H].
//
// Returns:
//   - *Model: immutable owned coefficients and canonical ID; nil on any error.
//
// Errors:
//   - ErrShape: invalid H/D, matrix shapes, bias length or product overflow.
//   - ErrNilInput: missing required weight matrix or combined bias.
//   - ErrNonFinite: any invalid element in the complete numerical payload.
//   - ErrShape joined with underlying Matrix.At read failure.
//
// Determinism:
//   - All validation and copying follows fixed shape/IFGO/row-major ordering.
//
// Complexity:
//   - Time O(H*(D+H)), Space O(H*(D+H)) for all owned storage.
//
// Notes:
//   - Source weights must already be converted from any external training format.
//
// AI-Hints:
//   - Never reorder candidate/output gates or hold references into caller Bias.
func importWeights(weights Weights) (*Model, error) {
	if weights.Hidden < 1 {
		return nil, ErrShape
	}
	rows, err := checkedProduct(gateBlocks, weights.Hidden)
	if err != nil {
		return nil, err
	}
	if !matrixPresent(weights.Input) || !matrixPresent(weights.Recurrent) {
		return nil, ErrNilInput
	}

	inputWidth := weights.Input.Cols()
	if inputWidth < 1 {
		return nil, ErrShape
	}
	if _, err = checkedProduct(rows, inputWidth); err != nil {
		return nil, err
	}
	if _, err = checkedProduct(rows, weights.Hidden); err != nil {
		return nil, err
	}
	if weights.Bias == nil {
		return nil, ErrNilInput
	}
	if weights.Input.Rows() != rows || weights.Recurrent.Rows() != rows ||
		weights.Recurrent.Cols() != weights.Hidden || len(weights.Bias) != rows {
		return nil, ErrShape
	}

	// No imported buffer aliases either source matrix, even for *matrix.Dense.
	inputWeights, err := matrixCopy(weights.Input, rows, inputWidth)
	if err != nil {
		return nil, err
	}
	recurrentWeights, err := matrixCopy(weights.Recurrent, rows, weights.Hidden)
	if err != nil {
		return nil, err
	}
	bias := make([]float64, rows)
	copy(bias, weights.Bias)
	if err = scanFinite(bias); err != nil {
		return nil, err
	}
	for index, value := range bias {
		if value == 0 {
			bias[index] = 0
		}
	}

	// The ID describes precisely the numerical coefficients used by the kernel.
	return &Model{
		d:    inputWidth,
		h:    weights.Hidden,
		wi:   inputWeights,
		wr:   recurrentWeights,
		bias: bias,
		id:   modelHash(inputWidth, weights.Hidden, inputWeights, recurrentWeights, bias),
	}, nil
}
