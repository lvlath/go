// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package match validates its first-version dense numerical input domain.
// These rules are matching-specific and do not change matrix storage policy.
package match

import (
	"errors"
	"math"
	"unsafe"

	"github.com/lvlath/go/matrix"
)

// symTol is the historic whole-matrix symmetry and diagonal tolerance.
// Greedy's partner tie-break also uses this value, independently of Options.Eps.
const symTol = 1e-12

// completeWeights holds a detached row-major copy of all source costs.
// Its n*n layout preserves full-input validation before subset mapping.
type completeWeights struct {
	n int
	w []float64
}

// at reads a previously validated matrix weight without another matrix.At call.
// The caller must supply indices already checked against the copied dimension.
// Implementation:
//   - Stage 1: Address a row-major element.
//
// Behavior highlights:
//   - No source aliasing or interface dispatch.
//
// Inputs:
//   - u, v: original matrix indices in [0,n).
//
// Returns:
//   - float64: directional, unrounded stored cost.
//
// Errors:
//   - None; callers must validate bounds first.
//
// Determinism:
//   - Fixed row-major access.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Near-symmetric directional weights are not averaged.
//
// AI-Hints:
//   - Never use local subset positions as global indices here.
func (b completeWeights) at(u, v int) float64 { return b.w[u*b.n+v] }

// checkedSquareLength verifies that a row-major float64 allocation is addressable.
// It detects integer and byte-size overflow without attempting a real OOM probe.
// Implementation:
//   - Stage 1: Reject negative dimensions.
//   - Stage 2: Check multiplication and float64 byte addressability.
//
// Behavior highlights:
//   - It is a representability check, not a memory budget.
//
// Inputs:
//   - n: square order; may be zero.
//
// Returns:
//   - int: safe n*n element count.
//
// Errors:
//   - ErrSizeOverflow for unrepresentable sizes.
//
// Determinism:
//   - Pure integer arithmetic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Real OOM remains outside recoverable Go input errors.
//
// AI-Hints:
//   - Check before multiplying; wrapped n*n is never trustworthy.
func checkedSquareLength(n int) (int, error) {
	if n < 0 {
		return 0, ErrSizeOverflow
	}
	if n == 0 {
		return 0, nil
	}
	maxInt := math.MaxInt
	if n > maxInt/n {
		return 0, ErrSizeOverflow
	}
	count := n * n
	if count > maxInt/8 {
		return 0, ErrSizeOverflow
	}
	return count, nil
}

// checkBlossomSizes validates all dense edge and node capacities before make.
// It protects k*(k-1)/2, 2*k-1, incident storage and the scan guard math.
// Implementation:
//   - Stage 1: Validate the local non-negative vertex count.
//   - Stage 2: Check square, edge count and doubled node count.
//   - Stage 3: Verify the existing step-limit multipliers fit int.
//
// Behavior highlights:
//   - Does not introduce an arbitrary maximum graph size.
//
// Inputs:
//   - k: local matching order.
//
// Returns:
//   - error: nil when all integer sizes are representable.
//
// Errors:
//   - ErrSizeOverflow on overflow, ErrInvalidMatching on negative k.
//
// Determinism:
//   - Fixed integer tests.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - The solver may still exhaust memory for representable k.
//
// AI-Hints:
//   - A safe n*n multiplication alone does not prove an edge capacity is safe.
func checkBlossomSizes(k int) error {
	if k < 0 {
		return ErrInvalidMatching
	}
	if _, err := checkedSquareLength(k); err != nil {
		return err
	}
	if k == 0 {
		return nil
	}
	maxInt := math.MaxInt
	if k > maxInt/2+1 { // 2*k-1 is checked without computing 2*k.
		return ErrSizeOverflow
	}
	// k*(k-1)/2 is evaluated in a division-first representation.
	a, b := k, k-1
	if a&1 == 0 {
		a /= 2
	} else {
		b /= 2
	}
	if b != 0 && a > maxInt/b {
		return ErrSizeOverflow
	}
	edges := a * b
	nodes := 2*k - 1
	const (
		edgeBytes      = int(unsafe.Sizeof(blossomEdge{}))
		nodeSliceBytes = int(unsafe.Sizeof([]int(nil)))
	)
	if edges > maxInt/32 || nodes > maxInt/8 ||
		edges > maxInt/edgeBytes || nodes > maxInt/nodeSliceBytes {
		return ErrSizeOverflow
	}
	return nil
}

// validateMatchingMatrix checks whole-matrix scalar and symmetry admission.
// It deliberately preserves TSP's original finite complete nonnegative domain.
// Implementation:
//   - Stage 1: Delegate nil and square shape checks to matrix.ValidateSquare.
//   - Stage 2: Scan scalar values row-major, enforcing diagonal and finite policy.
//   - Stage 3: Compare upper-triangle pairs using symTol.
//
// Behavior highlights:
//   - The triangle inequality is not required by perfect matching.
//   - Invalid values outside the requested subset are still rejected.
//
// Inputs:
//   - dist: stable, caller-owned matrix; symmetric requests a full symmetry scan.
//
// Returns:
//   - int: matrix order after successful admission.
//
// Errors:
//   - ErrNilMatrix, ErrNonSquare, ErrInvalidDimensions, ErrNaNInf,
//     ErrNonZeroDiagonal, ErrNegativeWeight, ErrIncompleteGraph, ErrAsymmetry.
//
// Determinism:
//   - Full row-major scalar pass, then ascending upper-triangle symmetry pass.
//
// Complexity:
//   - Time O(n^2), Space O(1).
//
// Notes:
//   - Concurrent mutation of the source during reading is unsupported.
//
// AI-Hints:
//   - Do not switch to subset-only admission in a behavior-preserving move.
func validateMatchingMatrix(dist matrix.Matrix, symmetric bool) (int, error) {
	if err := matrix.ValidateSquare(dist); err != nil {
		switch {
		case errors.Is(err, matrix.ErrNilMatrix):
			return 0, errors.Join(ErrNilMatrix, err)
		case errors.Is(err, matrix.ErrNonSquare):
			return 0, errors.Join(ErrNonSquare, err)
		default:
			return 0, errors.Join(ErrInvalidDimensions, err)
		}
	}
	n := dist.Rows()
	if n < 2 {
		return 0, ErrInvalidDimensions
	}
	if _, err := checkedSquareLength(n); err != nil {
		return 0, err
	}
	for row := 0; row < n; row++ {
		for col := 0; col < n; col++ {
			value, err := dist.At(row, col)
			if err != nil {
				return 0, errors.Join(ErrDimensionMismatch, err)
			}
			if math.IsNaN(value) || math.IsInf(value, -1) {
				return 0, errors.Join(ErrNaNInf, matrix.ErrNaNInf)
			}
			if row == col {
				if math.IsInf(value, 1) || math.Abs(value) > symTol {
					return 0, ErrNonZeroDiagonal
				}
				continue
			}
			if value < 0 {
				return 0, ErrNegativeWeight
			}
			if math.IsInf(value, 1) {
				return 0, ErrIncompleteGraph
			}
		}
	}
	if symmetric {
		for row := 0; row < n; row++ {
			for col := row + 1; col < n; col++ {
				value, err := dist.At(row, col)
				if err != nil {
					return 0, errors.Join(ErrDimensionMismatch, err)
				}
				mirror, err := dist.At(col, row)
				if err != nil {
					return 0, errors.Join(ErrDimensionMismatch, err)
				}
				if math.Abs(value-mirror) > symTol {
					return 0, ErrAsymmetry
				}
			}
		}
	}
	return n, nil
}

// copyCompleteWeights makes the existing full-input snapshot after validation.
// It preserves the copy-stage error causes and accepted directional near-ties.
// Implementation:
//   - Stage 1: Validate the full matrix and safe square allocation length.
//   - Stage 2: Copy values row-major, checking scalar policy once more.
//   - Stage 3: Normalize accepted diagonal values inside the copy only.
//
// Behavior highlights:
//   - No caller memory is retained, and no caller matrix is mutated.
//
// Inputs:
//   - dist: stable source; symmetric selects the full symmetry check.
//
// Returns:
//   - completeWeights: detached full n*n cost buffer.
//
// Errors:
//   - The same input sentinels as validateMatchingMatrix, including read causes.
//
// Determinism:
//   - Fixed three-phase validate, symmetry, copy sequence.
//
// Complexity:
//   - Time O(n^2), Space O(n^2).
//
// Notes:
//   - It is not an atomic snapshot of arbitrary concurrently mutable sources.
//
// AI-Hints:
//   - Do not average almost-equal mirrored weights when copying.
func copyCompleteWeights(dist matrix.Matrix, symmetric bool) (completeWeights, error) {
	n, err := validateMatchingMatrix(dist, symmetric)
	if err != nil {
		return completeWeights{}, err
	}
	count, err := checkedSquareLength(n)
	if err != nil {
		return completeWeights{}, err
	}
	buffer := completeWeights{n: n, w: make([]float64, count)}
	for row := 0; row < n; row++ {
		for col := 0; col < n; col++ {
			value, err := dist.At(row, col)
			if err != nil {
				return completeWeights{}, errors.Join(ErrDimensionMismatch, err)
			}
			if math.IsNaN(value) || math.IsInf(value, -1) {
				return completeWeights{}, errors.Join(ErrNaNInf, matrix.ErrNaNInf)
			}
			if row == col {
				if math.IsInf(value, 1) || math.Abs(value) > symTol {
					return completeWeights{}, ErrNonZeroDiagonal
				}
				buffer.w[row*n+col] = 0
				continue
			}
			if value < 0 {
				return completeWeights{}, ErrNegativeWeight
			}
			if math.IsInf(value, 1) {
				return completeWeights{}, ErrIncompleteGraph
			}
			buffer.w[row*n+col] = value
		}
	}
	return buffer, nil
}

// validateSubset maps selected original vertices without changing their order.
// It preserves the historic whole-matrix-before-subset error precedence.
// Implementation:
//   - Stage 1: Validate range for each original index.
//   - Stage 2: Reject duplicates and copy the ordered subset.
//
// Behavior highlights:
//   - No sorting or mutation of the source slice.
//
// Inputs:
//   - vertices: even-length requested subset; n: original matrix order.
//
// Returns:
//   - []int: detached ordered mapping from local position to original index.
//
// Errors:
//   - ErrInvalidVertices for a duplicate or out-of-range vertex.
//
// Determinism:
//   - Input scan order is the error precedence and tie-breaking source.
//
// Complexity:
//   - Time O(n+k), Space O(n+k).
//
// Notes:
//   - Whole-matrix admission belongs to the caller before this helper.
//
// AI-Hints:
//   - Do not sort this slice, even if canonical output pairs will be sorted.
func validateSubset(vertices []int, n int) ([]int, error) {
	seen := make([]bool, n)
	copyOfVertices := append([]int(nil), vertices...)
	for _, vertex := range copyOfVertices {
		if vertex < 0 || vertex >= n || seen[vertex] {
			return nil, ErrInvalidVertices
		}
		seen[vertex] = true
	}
	return copyOfVertices, nil
}
