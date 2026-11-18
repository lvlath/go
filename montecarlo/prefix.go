// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package montecarlo exposes only committed rows of interrupted experiments.
// This file provides a bounded view over one already-accounted owning Dense.
package montecarlo

import "github.com/lvlath/go/matrix"

// prefixMatrix exposes a mutable, completed-prefix matrix.Matrix.
// The private Dense remains owned by the Result for its entire lifetime;
// trailing preallocated rows are not observable through this interface.
// Retaining a short prefix also retains its budgeted full-size Dense allocation.
type prefixMatrix struct {
	owner *matrix.Dense
	rows  int
	cols  int
}

// Rows returns the number of committed path vectors, never the allocated capacity.
// It is nil-receiver-safe for parity with matrix.Dense's dimension accessors.
//
// Returns: committed row count, or zero for a nil receiver.
// Determinism: immutable published prefix shape. Complexity: O(1) time/space.
// AI-Hints: do not expose the full owner.Rows on an incomplete experiment.
func (p *prefixMatrix) Rows() int {
	if p == nil {
		return 0
	}
	return p.rows
}

// Cols returns the fixed sampler output dimension for the reduced prefix.
// It never changes with cancellation, subsequent result mutation or cloning.
//
// Returns: column count, or zero for a nil receiver.
// Determinism: immutable published prefix shape. Complexity: O(1) time/space.
// AI-Hints: dimension remains fixed even if no row was reduced.
func (p *prefixMatrix) Cols() int {
	if p == nil {
		return 0
	}
	return p.cols
}

// At reads one owned, already-committed component from the prefix.
// Invalid rows cannot access the unused tail of the full backing allocation.
//
// Inputs: row and column are zero-based indices within Rows() and Cols().
// Returns: the value or matrix.ErrOutOfRange without exposing hidden storage.
// Determinism: delegates to the original Dense row-major index.
// Complexity: Time O(1), Space O(1).
// AI-Hints: bounds must use the prefix shape, not the allocated owner shape.
func (p *prefixMatrix) At(row, column int) (float64, error) {
	if p == nil || row < 0 || row >= p.rows || column < 0 || column >= p.cols {
		return 0, matrix.ErrOutOfRange
	}
	return p.owner.At(row, column)
}

// Set mutates a caller-owned committed value but never its hidden tail.
// Snapshot moments and quantiles remain detached from any later sample edits.
//
// Inputs: zero-based prefix indices and a value governed by Dense numeric policy.
// Returns: nil or a matrix index/numeric sentinel from the original Dense.
// Determinism: exactly one row-major write. Complexity: O(1) time/space.
// AI-Hints: modifying Samples must never retroactively recompute Result.Mean.
func (p *prefixMatrix) Set(row, column int, value float64) error {
	if p == nil || row < 0 || row >= p.rows || column < 0 || column >= p.cols {
		return matrix.ErrOutOfRange
	}
	return p.owner.Set(row, column, value)
}

// Clone creates an independent owning Dense of the public prefix shape.
// Its new allocation is proportional to Completed*d rather than requested Paths*d.
//
// Implementation:
//   - Stage 1: validate presence and allocate exactly the visible prefix shape.
//   - Stage 2: copy every visible finite cell through matrix.At and Dense.Set.
//   - Stage 3: return an independent Matrix with no shared backing.
//
// Returns: an independent matrix.Matrix; nil for a nil receiver or an unexpected
//
//	dependency allocation/read/write failure (Matrix.Clone has no error return).
//
// Determinism: row-major copy in ascending indices.
// Complexity: Time O(Completed*d), Space O(Completed*d).
// AI-Hints: do not call owner.Clone; that would expose hidden uncommitted rows.
func (p *prefixMatrix) Clone() matrix.Matrix {
	if p == nil {
		return nil
	}
	copyMatrix, err := matrix.NewDense(p.rows, p.cols)
	if err != nil {
		return nil
	}
	for row := 0; row < p.rows; row++ {
		for column := 0; column < p.cols; column++ {
			value, readErr := p.owner.At(row, column)
			if readErr != nil {
				return nil
			}
			if err := copyMatrix.Set(row, column, value); err != nil {
				return nil
			}
		}
	}
	return copyMatrix
}

// IsNil checks pointer presence when prefixMatrix is held as matrix.Matrix.
// Its optional nilability follows the same contract as matrix.Dense.
//
// Returns: true only for a nil receiver, false for any published prefix.
// Determinism: identity comparison. Complexity: O(1) time and space.
// AI-Hints: a nonempty prefix is a separate property from nilability.
func (p *prefixMatrix) IsNil() bool {
	return p == nil
}

var _ matrix.Matrix = (*prefixMatrix)(nil)
