// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp provides shared matrix-to-weight-buffer helpers for TSP kernels.
// The helpers centralize TSP numeric policy so exact solvers, local search,
// Branch-and-Bound, and 1-tree bounds cannot drift in sentinel classification.
package tsp

import (
	"errors"
	"math"

	"github.com/lvlath/go/matrix"
)

// weightBuffer stores a detached dense row-major snapshot of a distance matrix.
// It is intentionally unexported because it is an internal hot-loop representation,
// not a public matrix replacement.
//
// Implementation:
//   - Stage 1: The copy helpers validate matrix shape and TSP numeric policy.
//   - Stage 2: Values are copied in deterministic row-major order.
//   - Stage 3: Kernels read weights through at(u,v) without matrix interface calls.
//
// Behavior highlights:
//   - Detached from the source matrix.
//   - Row-major layout: w[u*n+v].
//   - Does not own IDs, graph metadata, or solver policy.
//
// Inputs:
//   - Built only by copyCompleteWeights or copyClosureReadyWeights.
//
// Returns:
//   - Used internally by kernels.
//
// Errors:
//   - Construction errors are returned by the copy helpers.
//
// Determinism:
//   - Values are copied by increasing row, then increasing column.
//
// Complexity:
//   - Access Time O(1), Space O(n^2).
//
// Notes:
//   - This type is not a matrix.Matrix implementation.
//   - It is a solver-local cache to prevent repeated At calls in hot loops.
//
// AI-Hints:
//   - Do not expose weightBuffer from public APIs.
//   - Do not mutate w after construction; kernels treat it as read-only.
type weightBuffer struct {
	// n is the square matrix order.
	// Every valid index must be in [0,n].
	n int

	// w stores detached row-major weights.
	// w[u*n+v] is the directed distance from u to v.
	w []float64
}

// at returns the detached cost for directed arc u->v.
// The caller must pass indices already validated against the matrix order.
//
// Implementation:
//   - Stage 1: Compute row-major offset u*n+v.
//   - Stage 2: Return the stored float64 without additional validation.
//
// Behavior highlights:
//   - No allocation.
//   - No matrix interface dispatch.
//   - Panics only if an internal kernel passes invalid indices.
//
// Inputs:
//   - u: source vertex index.
//   - v: destination vertex index.
//
// Returns:
//   - float64: cached distance from u to v.
//
// Errors:
//   - None. Index validation belongs to the caller/kernel preconditions.
//
// Determinism:
//   - Pure indexed lookup.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This helper is deliberately small so hot loops remain clear.
//
// AI-Hints:
//   - Do not add bounds checks here that duplicate already validated tour invariants.
//   - Do not use this helper before copyCompleteWeights/copyClosureReadyWeights succeeds.
func (b weightBuffer) at(u, v int) float64 {
	return b.w[u*b.n+v]
}

// copyCompleteWeights validates and copies a final solver distance matrix.
// It rejects +Inf off-diagonal entries because final TSP/ATSP kernels require a
// complete finite Hamiltonian cost model.
//
// Implementation:
//   - Stage 1: Validate TSP matrix shape and numeric policy with complete=true.
//   - Stage 2: Allocate one row-major buffer of n*n values.
//   - Stage 3: Copy all values in deterministic row-major order.
//
// Behavior highlights:
//   - Preserves matrix-level sentinels through errors.Join.
//   - Rejects NaN and -Inf as ErrNaNInf.
//   - Rejects +Inf as ErrIncompleteGraph.
//   - Rejects negative finite weights as ErrNegativeWeight.
//
// Inputs:
//   - dist: matrix.Matrix distance model.
//   - symmetric: whether symmetry is required by the caller.
//
// Returns:
//   - weightBuffer: detached complete finite row-major weights.
//   - error: nil on success or sentinel-classified validation/copy failure.
//
// Errors:
//   - ErrNilDistanceMatrix joined with matrix.ErrNilMatrix.
//   - ErrNonSquare joined with matrix.ErrNonSquare.
//   - ErrDimensionMismatch joined with matrix At/read failures.
//   - ErrNaNInf joined with matrix.ErrNaNInf.
//   - ErrNonZeroDiagonal, ErrNegativeWeight, ErrIncompleteGraph, ErrAsymmetry.
//
// Determinism:
//   - Validation and copy use fixed row-major order.
//   - The first failing value is stable for the same matrix implementation.
//
// Complexity:
//   - Time O(n^2), Space O(n^2).
//   - Validation also scans O(n^2); asymptotic complexity remains O(n^2).
//
// Notes:
//   - Use this for final Held-Karp, 2-opt, 3-opt, and Branch-and-Bound kernels.
//   - Use copyClosureReadyWeights for pre-closure or lower-bound helpers that may accept +Inf.
//
// AI-Hints:
//   - Do not allow +Inf into final solver kernels.
//   - Do not reimplement NaN/Inf/negative checks inside every algorithm file.
func copyCompleteWeights(dist matrix.Matrix, symmetric bool) (weightBuffer, error) {
	return copyWeightsWithMode(dist, symmetric, true)
}

// copyClosureReadyWeights validates and copies a matrix that may still contain +Inf.
// It is intended for pre-closure helpers and lower-bound components where +Inf
// represents a missing edge that can be handled by the algorithm.
//
// Implementation:
//   - Stage 1: Validate TSP matrix shape and numeric policy with complete=false.
//   - Stage 2: Allocate one row-major buffer of n*n values.
//   - Stage 3: Copy all values in deterministic row-major order.
//
// Behavior highlights:
//   - Allows +Inf off-diagonal values.
//   - Rejects NaN, -Inf, negative finite weights, non-zero diagonal, and asymmetry when requested.
//   - Preserves matrix structural sentinels through errors.Join.
//
// Inputs:
//   - dist: matrix.Matrix distance model.
//   - symmetric: whether symmetry is required.
//
// Returns:
//   - weightBuffer: detached row-major weights, possibly containing +Inf off diagonal.
//   - error: nil on success or sentinel-classified failure.
//
// Errors:
//   - Same structural/numeric sentinels as copyCompleteWeights, except +Inf off-diagonal
//     is allowed and therefore does not return ErrIncompleteGraph.
//
// Determinism:
//   - Fixed row-major validation/copy order.
//
// Complexity:
//   - Time O(n^2), Space O(n^2).
//
// Notes:
//   - This helper is not for final tour-cost kernels.
//   - B&B final solving should use copyCompleteWeights unless the public contract is
//     deliberately changed to exact solving over incomplete digraphs.
//
// AI-Hints:
//   - Do not use closure-ready weights to bypass final completeness validation.
//   - Do not collapse +Inf with NaN; +Inf is a missing-edge sentinel in this mode.
func copyClosureReadyWeights(dist matrix.Matrix, symmetric bool) (weightBuffer, error) {
	return copyWeightsWithMode(dist, symmetric, false)
}

// copyWeightsWithMode is the shared implementation behind the two public internal
// copy modes. It keeps sentinel classification identical for every TSP kernel.
//
// Implementation:
//   - Stage 1: Delegate shape, diagonal, symmetry, NaN/-Inf, negative, and optional
//     completeness validation to validateSolverDistanceMatrix.
//   - Stage 2: Allocate a detached row-major buffer.
//   - Stage 3: Copy values and defensively preserve read-time sentinels.
//
// Behavior highlights:
//   - One canonical matrix scan policy for all solver kernels.
//   - Does not mutate the source matrix.
//   - Does not sort or transform weights.
//
// Inputs:
//   - dist: matrix.Matrix distance model.
//   - symmetric: whether symmetry is required.
//   - complete: whether +Inf off-diagonal entries are forbidden.
//
// Returns:
//   - weightBuffer: detached row-major snapshot.
//   - error: nil on success or sentinel-classified failure.
//
// Errors:
//   - Propagates validateSolverDistanceMatrix sentinels.
//   - Returns ErrDimensionMismatch joined with matrix read errors during copy.
//   - Returns ErrNaNInf joined with matrix.ErrNaNInf for defensive NaN/-Inf discovery.
//   - Returns ErrIncompleteGraph for defensive +Inf discovery when complete=true.
//   - Returns ErrNegativeWeight for defensive negative finite discovery.
//
// Determinism:
//   - Fixed row-major copy order.
//
// Complexity:
//   - Time O(n^2), Space O(n^2).
//
// Notes:
//   - Defensive copy-time checks intentionally mirror validation even after validation succeeds.
//   - The duplicate scan is acceptable because it removes per-kernel drift while preserving O(n^2).
//
// AI-Hints:
//   - Do not remove defensive copy-time checks; mutable matrix implementations may change between calls.
//   - Do not convert matrix errors to strings; preserve sentinels with errors.Join.
func copyWeightsWithMode(dist matrix.Matrix, symmetric bool, complete bool) (weightBuffer, error) {
	n, err := validateSolverDistanceMatrix(dist, symmetric, complete, symTol)
	if err != nil {
		return weightBuffer{}, err
	}

	buffer := weightBuffer{
		n: n,
		w: make([]float64, n*n),
	}

	var (
		row     int
		col     int
		value   float64
		readErr error
		offset  int
	)

	for row = 0; row < n; row++ {
		for col = 0; col < n; col++ {
			value, readErr = dist.At(row, col)
			if readErr != nil {
				return weightBuffer{}, errors.Join(ErrDimensionMismatch, readErr)
			}

			if math.IsNaN(value) || math.IsInf(value, -1) {
				return weightBuffer{}, errors.Join(ErrNaNInf, matrix.ErrNaNInf)
			}

			offset = row*n + col

			if row == col {
				if math.IsInf(value, 1) {
					return weightBuffer{}, ErrNonZeroDiagonal
				}
				if math.Abs(value) > symTol {
					return weightBuffer{}, ErrNonZeroDiagonal
				}

				buffer.w[offset] = 0

				continue
			}

			if value < 0 {
				return weightBuffer{}, ErrNegativeWeight
			}
			if complete && math.IsInf(value, 1) {
				return weightBuffer{}, ErrIncompleteGraph
			}

			buffer.w[offset] = value
		}
	}

	return buffer, nil
}
