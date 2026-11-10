// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix: sentinel error set (unified, consistent).
// This file defines ONLY package-level sentinel errors used across the matrix
// package. All algorithms MUST return these sentinels and tests MUST check them
// via errors.Is. No algorithm should panic on user-triggered error conditions.
// Panics are reserved for programmer errors in private helpers (if any).

package matrix

import "errors"

// NOTE ON NAMING & PREFIXING
// --------------------------
// Every message is prefixed with "matrix: ..." for consistency and to allow
// easy grepping across logs. DO NOT %w wrap these sentinels when returning
// directly; if context is essential, wrap with fmt.Errorf("ctx: %w", ErrX)
// at the outer boundary - callers will still use errors.Is to match.
//
// ERROR PRIORITY:
// shape/index/NaN -> graph nil -> dimension mismatch -> structural violations
// -> export limitations (ErrMatrixNotImplemented).

// SHAPE vs DIMENSIONS (IMPORTANT)
// -------------------------------
// We intentionally KEEP ErrInvalidDimensions and ErrBadShape as DISTINCT classes.
// They are close but not the same:
//   - ErrInvalidDimensions: invalid size parameters (negative, or otherwise impossible sizes).
//   - ErrBadShape: shape is structurally invalid for an operation (e.g., non-square required,
//     invalid view window, incompatible "shape contract" not expressible by a simple mismatch).
//
// Rationale:
//   - Different failure classes matter to callers and to tests.
//   - Avoid "smart" errors.Is coupling between independent sentinels (it becomes brittle fast).

var (
	// ErrNilOption reports a nil Option passed to matrix option assembly.
	//
	// Contract:
	// - gatherOptions(nil) MUST return ErrNilOption.
	// - NewMatrixOptions(nil) MUST return ErrNilOption.
	// - Any public matrix facade that accepts opts ...Option MUST preserve this sentinel.
	//
	// Notes:
	// - Nil options are invalid public inputs, not no-ops.
	// - This sentinel exists because a nil function value would otherwise panic on call.
	ErrNilOption = errors.New("matrix: nil option")

	// ErrInvalidDimensions indicates invalid dimension parameters.
	// Negative sizes are always invalid. Some builders may additionally enforce
	// "non-empty" rules as part of their own contract; if they do, they should
	// still use this class (invalid dimension request).
	ErrInvalidDimensions = errors.New("matrix: invalid dimensions")

	// ErrBadShape indicates that a requested/observed shape violates a structural
	// contract for an operation (beyond a simple operand mismatch).
	ErrBadShape = errors.New("matrix: invalid shape")

	// ErrOutOfRange indicates that an index (row or column) is outside valid bounds.
	// Public indexers (At/Set) MUST return this, not panic.
	ErrOutOfRange = errors.New("matrix: index out of range")

	// ErrDimensionMismatch indicates incompatible dimensions between operands,
	// e.g., Add/Sub different shapes, or Mul where a.Cols != b.Rows.
	ErrDimensionMismatch = errors.New("matrix: dimension mismatch")

	// ErrNonSquare signals that a square matrix was required but the input wasn't.
	ErrNonSquare = errors.New("matrix: matrix is not square")

	// ErrAsymmetry signals that a matrix expected to be symmetric violated symmetry
	// within the configured numeric policy (epsilon).
	ErrAsymmetry = errors.New("matrix: matrix is not symmetric within eps")

	// ErrNonZeroDiagonal signals that a diagonal is required to be ~0 (within eps)
	// but a non-zero entry was observed (common in Laplacian-like checks).
	ErrNonZeroDiagonal = errors.New("matrix: diagonal not zero within eps")

	// ErrNaNInf signals a NaN or ±Inf value was encountered where finite values
	// are required by the numeric policy (ingestion, Set, etc.).
	ErrNaNInf = errors.New("matrix: NaN or Inf encountered")

	// ErrGraphNil indicates that a nil *core.Graph was passed into an adapter.
	ErrGraphNil = errors.New("matrix: graph is nil")

	// ErrNilCallback indicates that a nil callback function was passed.
	ErrNilCallback = errors.New("matrix: nil callback")

	// ErrUnknownVertex indicates that a referenced VertexID is not present
	// in the current vertex index (adjacency/incidence).
	ErrUnknownVertex = errors.New("matrix: unknown vertex id")

	// ErrNilMatrix indicates that a nil Matrix (receiver or argument) was used.
	ErrNilMatrix = errors.New("matrix: nil receiver")

	// ErrMatrixNotImplemented marks an intentionally unsupported operation
	// on the matrix surface (e.g., ToGraph on MetricClosure adjacency).
	ErrMatrixNotImplemented = errors.New("matrix: operation not implemented")

	// ErrMatrixEigenFailed indicates that an eigen/Jacobi routine failed to converge
	// under the given tolerance/iterations.
	ErrMatrixEigenFailed = errors.New("matrix: eigen decomposition failed")

	// ErrSingular is returned when a zero pivot is encountered during inversion/LU
	// in a non-pivoting scheme (intentional for determinism and simplicity).
	ErrSingular = errors.New("matrix: singular matrix")

	// ErrNonBinaryIncidence - non-±1 entry detected for unweighted incidence.
	ErrNonBinaryIncidence = errors.New("matrix: non-binary incidence")

	// ErrInvalidWeight - edge weight is NaN or ±Inf at ingestion stage.
	ErrInvalidWeight = errors.New("matrix: invalid edge weight")

	// ErrNegativeCycle reports that a shortest-path computation detected a negative cycle.
	//
	// Contract:
	//   - FloydWarshall returns ErrNegativeCycle when the final distance diagonal contains
	//     a negative value below the configured tolerance.
	//   - ValidateDistanceMatrix returns ErrNegativeCycle when the input diagonal already
	//     proves a negative self-cycle.
	//
	// Notes:
	//   - Negative finite off-diagonal distances are valid Floyd–Warshall inputs.
	//   - A negative diagonal is not merely bad shape; mathematically it represents a
	//   negative cycle witness.
	ErrNegativeCycle = errors.New("matrix: negative cycle")

	// ErrOverflow distinguishes unrepresentable finite arithmetic results from invalid inputs.
	// Return it when required float64 outputs cannot be represented without infinity.
	// Existing ErrNaNInf remains the classification for non-finite input data.
	ErrOverflow = errors.New("matrix: arithmetic overflow")

	// ErrInvalidOption reports a non-finite or out-of-domain factor option.
	// Nil contexts also use this class; no algorithmic workspace is allocated.
	ErrInvalidOption = errors.New("matrix: invalid factor option")

	// ErrRankDeficient indicates rank loss relative to the selected diagonal cutoff.
	// Exact singular LU pivots also retain ErrSingular via errors.Join.
	ErrRankDeficient = errors.New("matrix: rank deficient")

	// ErrNotPositiveDefinite reports a nonpositive Cholesky diagonal update.
	// Symmetry alone does not imply positivity; no hidden diagonal shift occurs.
	ErrNotPositiveDefinite = errors.New("matrix: matrix is not positive definite")

	// ErrIllConditioned reports a measured 1-norm condition estimate above the limit.
	// Equality is allowed and a zero limit never disables measurement.
	ErrIllConditioned = errors.New("matrix: condition estimate exceeds limit")
)

// BACKWARD-COMPATIBILITY ALIASES (kept to avoid breaking current callers).
// These are maintained to let existing code compile while we migrate internals
// to the unified names above. They are semantically identical sentinels.

// ErrIndexOutOfBounds historically named the same condition as ErrOutOfRange.
// Keep it as an alias so errors.Is(err, ErrIndexOutOfBounds) remains true.
var ErrIndexOutOfBounds = ErrOutOfRange // Deprecated: use ErrOutOfRange.

// ErrNotSymmetric historically named symmetry violation.
// It aliases ErrAsymmetry to preserve errors.Is behavior during migration.
var ErrNotSymmetric = ErrAsymmetry // Deprecated: use ErrAsymmetry.

// ErrEigenFailed historically named the eigen failure sentinel.
// Alias to unified ErrMatrixEigenFailed for non-breaking migration.
var ErrEigenFailed = ErrMatrixEigenFailed // Deprecated: use ErrMatrixEigenFailed.

// ErrNotSquare historically named square-shape violation.
// Alias to unified ErrNonSquare for non-breaking migration.
var ErrNotSquare = ErrNonSquare // Deprecated: use ErrNonSquare.
