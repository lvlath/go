// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package match declares the error protocol for dense perfect matching.
// Sentinels distinguish input validity, numerical arithmetic, and solver state.
package match

import "errors"

var (
	// ErrInvalidOptions reports an unsupported Algorithm or nonpositive/nonfinite Eps.
	// A nil Option callback and an invalid intermediate options state are refused before matrix access.
	//
	// AI-Hints:
	//   - Do not silently default a caller-supplied zero epsilon.
	ErrInvalidOptions = errors.New("match: invalid options")
	// ErrNilMatrix reports a nil matrix under matrix.ValidateSquare and the core.Nilable convention.
	// Empty subsets do not inspect the matrix; nonempty subsets preserve matrix.ErrNilMatrix via errors.Join.
	//
	// AI-Hints:
	//   - Do not use reflection-based typed-nil detection.
	ErrNilMatrix = errors.New("match: nil matrix")
	// ErrNonSquare reports a mismatch between matrix Rows and Cols.
	// The matrix shape is checked before the first indexed read.
	//
	// AI-Hints:
	//   - Do not treat a rectangular source as an induced square matching instance.
	ErrNonSquare = errors.New("match: non-square matrix")
	// ErrInvalidDimensions reports a square but insufficient matrix order for nonempty matching.
	// Nonempty requests require order >=2, while an empty subset succeeds without matrix reads.
	//
	// AI-Hints:
	//   - Do not interpret zero selected vertices as every matrix vertex.
	ErrInvalidDimensions = errors.New("match: invalid dimensions")
	// ErrDimensionMismatch reports a failed matrix.At read or inconsistent matrix backend dimensions.
	// The original At cause is joined for errors.Is inspection; no partial result is returned.
	//
	// AI-Hints:
	//   - The Matrix implementation must present stable dimensions throughout a solve.
	ErrDimensionMismatch = errors.New("match: dimension mismatch or matrix read error")
	// ErrOddVertices reports odd cardinality of the selected original vertices.
	// The parity guard runs before matrix nil/shape checks and before any allocation.
	//
	// AI-Hints:
	//   - Do not drop a last vertex to force a half-matching.
	ErrOddVertices = errors.New("match: odd number of selected vertices")
	// ErrInvalidVertices reports repeated or out-of-range original matrix vertex indices.
	// The validation preserves caller order and rejects duplicates after whole-matrix admission.
	//
	// AI-Hints:
	//   - Indices are global matrix coordinates, not local matching positions.
	ErrInvalidVertices = errors.New("match: invalid selected vertices")
	// ErrInvalidMatching reports an invalid reciprocal mate array or broken Blossom invariant.
	// Complete matching cardinality, reciprocity and self-loop exclusion are enforced before publication.
	//
	// AI-Hints:
	//   - Do not turn an internal invariant failure into a successful Greedy result.
	ErrInvalidMatching = errors.New("match: invalid matching state")
	// ErrNoMatching reports a selected solver unable to publish a complete matching.
	// When caused by an incomplete-graph solver result, ErrIncompleteGraph is also preserved.
	//
	// AI-Hints:
	//   - This error is not an independently verified combinatorial infeasibility proof.
	ErrNoMatching = errors.New("match: no complete matching produced")
	// ErrIncompleteGraph reports positive-infinite off-diagonal input or an incomplete numerical solve.
	// The dense admission domain requires a finite edge between every two distinct vertices.
	//
	// AI-Hints:
	//   - Run a separately justified metric closure before matching sparse topology.
	ErrIncompleteGraph = errors.New("match: incomplete graph or matching")
	// ErrNonSymmetric reports an off-diagonal weight pair outside the absolute structural symTol.
	// Input directions are compared without averaging; Blossom Eps may be stricter later.
	//
	// AI-Hints:
	//   - Do not alter accepted directional source costs to manufacture symmetry.
	ErrNonSymmetric = errors.New("match: asymmetric weights")
	// ErrAsymmetry reports the historical Blossom-compatible alias of ErrNonSymmetric.
	// Both names resolve to the SAME sentinel identity under errors.Is.
	//
	// AI-Hints:
	//   - Do not assign a second independent error value to the alias.
	ErrAsymmetry = ErrNonSymmetric
	// ErrNonZeroDiagonal reports a self-distance exceeding symTol or positive infinity on the diagonal.
	// Diagonal admission is structural even though self-pairs are forbidden.
	//
	// AI-Hints:
	//   - Keep diagonal costs zero in the source matrix.
	ErrNonZeroDiagonal = errors.New("match: non-zero diagonal")
	// ErrNaNInf reports NaN or negative-infinite input in a forbidden position.
	// Positive infinity on an off-diagonal edge is separately ErrIncompleteGraph.
	//
	// AI-Hints:
	//   - Do not conflate invalid input with valid-input arithmetic overflow.
	ErrNaNInf = errors.New("match: non-finite input")
	// ErrNegativeWeight reports a finite negative off-diagonal weight.
	// The migrated solver preserves the original nonnegative-weight domain.
	//
	// AI-Hints:
	//   - Do not assume the signed-weight general-graph problem is supported.
	ErrNegativeWeight = errors.New("match: negative weight")
	// ErrNumericOverflow reports the sum of selected finite weights becoming nonfinite.
	// Result.Cost is accumulated unrounded and never saturates at MaxFloat64.
	//
	// AI-Hints:
	//   - Do not round before summation or substitute transformed Blossom profit.
	ErrNumericOverflow = errors.New("match: matching cost overflows float64")
	// ErrSizeOverflow reports unrepresentable dense counts, edge capacities or blossom node indices.
	// The check is performed before multiplication or allocation; it is not a memory budget.
	//
	// AI-Hints:
	//   - Large but representable allocations can still exhaust process resources.
	ErrSizeOverflow = errors.New("match: allocation size overflow")
)
