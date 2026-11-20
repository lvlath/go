// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix: domain types used by adapters and dense operations.
// This file intentionally contains ONLY domain-facing types (IDs, weights,
// helper keys) and, for now, preserves the public Matrix interface to avoid
// breaking existing code. Errors and options live in dedicated
// files (errors.go, options.go) per the global conventions.
package matrix

// VertexID uniquely identifies a graph vertex (core uses string IDs).
// Determinism relies on lexicographic ordering of these IDs across adapters.
type VertexID string // string-based ID (stable lex order)

// Weight represents an edge weight for adapters/numeric ingestion.
// Contract:
//   - When numeric validation is enabled, values must be finite (no NaN/±Inf).
//   - No sign or range restriction is imposed here beyond finiteness.
type Weight float64 // enforced via ValidateNaNInf policy

// pairKey is an ordered pair (u,v) used to de-duplicate parallel edges under
// "first-edge-wins" policy in directed mode. For undirected mode we normalize
// into {min,max} and still store in pairKey (u=min, v=max). Using ints keeps
// the key compact and hash-friendly.
// Complexity: O(1) to build; used in O(E) scans during ingestion.
type pairKey struct {
	u int // source row index
	v int // destination column index
}

// Matrix represents a two-dimensional mutable array of float64 values.
//
// Zero-shape law:
//   - Matrix implementations may legally have shape 0×0, 0×N, or N×0.
//   - Rows and Cols must always be non-negative.
//   - A zero-sized matrix has an empty element domain; element-wise scans are
//     deterministic no-ops.
//   - At/Set on any coordinate outside the empty domain must return ErrOutOfRange.
//
// Contract:
//   - At/Set MUST return ErrOutOfRange on invalid indices.
//   - Set may additionally return ErrNaNInf when numeric policy rejects a value.
//   - Clone returns a deep copy independent of the original.
//   - Clone of a zero-sized matrix must preserve shape and numeric policy when
//     the concrete implementation supports policy state.
//
// Complexity notes:
//   - Rows/Cols: O(1)
//   - At/Set: O(1)
//   - Clone: expected O(r*c) for dense implementations.
//
// AI-Hints:
//   - Do not special-case zero-sized matrices as nil.
//   - Use ValidateNotNil for presence and explicit shape validators for operations
//     that require non-empty dimensions.
type Matrix interface {
	// Rows returns the number of rows in the matrix.
	// Complexity: O(1).
	Rows() int

	// Cols returns the number of columns in the matrix.
	// Complexity: O(1).
	Cols() int

	// At retrieves the element at position (i, j).
	// Returns ErrOutOfRange if i<0, i>=Rows(), j<0 or j>=Cols().
	// Concrete implementations may return additional sentinel errors per
	// numeric policy.
	// Complexity: O(1).
	At(i, j int) (float64, error)

	// Set assigns the value v at position (i, j).
	// Returns ErrOutOfRange if indices are invalid. Implementations may also
	// return ErrNaNInf when v is not finite, depending on numeric policy.
	// Complexity: O(1).
	Set(i, j int, v float64) error

	// Clone returns a deep copy of the matrix.
	// The returned Matrix is independent of the original.
	// Complexity: O(rows*cols).
	Clone() Matrix
}

// SVDResult holds a detached rectangular economy singular-value decomposition.
// For a finite m-by-n input, k=min(m,n), U is m-by-k, VT is k-by-n, and
// SingularValues has length k in nonincreasing nonnegative order.
//
// Implementation:
//   - Stage 1: sort the rotated singular modes without removing tiny directions.
//   - Stage 2: complete exact-zero left vectors and apply sign canonicalization.
//   - Stage 3: publish independently owned matrices, singular values, and diagnostics.
//
// Behavior highlights:
//   - Converged=true means the normalized column-correlation criterion succeeded.
//   - Rank counts s[i]/s[0] > max(m,n)*2^-52, not the count of nonzero singular values.
//   - On ErrNotConverged, this is a *diagnostic* last finite decomposition:
//     U columns may be nonorthogonal, so do not use it for PCA or solves.
//
// Inputs: Created only by SVD and its wrappers, not by user-provided matrices.
// Returns: Caller owns both Dense matrices and the distinct singular-value slice.
// Errors: Invalid input/cancellation/overflow always returns nil; only sweep
// exhaustion returns a non-nil result plus ErrNotConverged.
// Determinism: Stable descending values; sign pivot uses earliest maximum row.
// Complexity: Result storage O((m+n)*k), in addition to temporary workspace.
// Notes: Input and result memory never alias. Concurrent mutation of caller-owned
// matrices is unsupported; read-only concurrent decompositions are independent.
// AI-Hints: The computed numeric rank is not a request to discard singular vectors;
// check Converged and err before consuming factor matrices in a model.
type SVDResult struct {
	// U holds k orthonormal left singular vectors when Converged is true.
	// The maximum-absolute element of each column is canonically nonnegative.
	U Matrix

	// VT holds the k transposed right singular vectors, in matching order.
	// Each row uses the same canonical sign flip as its corresponding U column.
	VT Matrix

	// SingularValues contains all k nonnegative modes, including tiny nonzero
	// values below the numeric-rank cutoff and trailing exact zeros.
	SingularValues []float64

	// Rank is the number of singular values whose ratio to s[0] exceeds
	// max(m,n)*2^-52; it is zero for an identically zero matrix.
	Rank int

	// Iterations counts completed full lexicographic sweeps. Already orthogonal
	// matrices can converge with zero sweeps, including every diagonal matrix.
	Iterations int

	// Converged is true only when the *last* full correlation scan satisfies
	// the tolerance. A false flag accompanies ErrNotConverged and diagnostic data.
	Converged bool
}

// IsNil reports whether this SVD result pointer is nil, implementing the library's
// nilable-result convention without dereferencing the receiver.
//
// Implementation:
//   - Stage 1: compare the pointer itself with nil.
//   - Stage 2: return the comparison; no fields are inspected.
//
// Behavior highlights: A non-nil diagnostic result is not considered nil-like.
// Inputs: r may be a nil *SVDResult receiver.
// Returns: True exactly for nil receivers; false for all allocated results.
// Errors: None.
// Determinism: Pure O(1) pointer comparison.
// Complexity: Time O(1), Space O(1).
// Notes: IsNil does not imply that an SVD is numerically converged.
// AI-Hints: Always inspect err and Converged separately after the nil check.
func (r *SVDResult) IsNil() bool {
	return r == nil
}

// SolveResult owns the solution matrix and independent residual diagnostics.
// Each residual is the Euclidean norm of the original-coordinate matrix-vector
// difference, evaluated against the accepted matrix used by the factorization.
// Implementation:
//   - Solution is detached from factor storage and supplied right-hand sides.
//   - ResidualNorms are calculated after the complete solution is finite.
//   - ConditionEstimate is retained from the factor's measured 1-norm diagnostic.
//
// Behavior highlights:
//   - Rank is the accepted number of independent columns and equals Solution.Rows().
//   - All published numeric fields are finite; ConditionEstimate is >=1.
//
// Inputs: The result is produced only by a successful Solve invocation.
// Returns: Caller-owned data; no component aliases factor storage or other calls.
// Errors: Failure returns a nil *SolveResult, never a partially populated instance.
// Determinism: RHS columns preserve input order; residuals follow the same order.
// Complexity: Storage O(n*r+r), beyond internal per-call solve workspace.
// Notes: For Cholesky with nonzero symmetry tolerances, residuals use the
// accepted symmetrized matrix rather than the raw asymmetric input.
// AI-Hints: Use ResidualNorms for backward-error diagnostics, not as a bound
// on coefficient error without accounting for conditioning.
type SolveResult struct {
	// Solution contains n rows (unknowns) and r columns (right-hand sides).
	// The returned Matrix is a fresh owning *Dense, writable by the caller.
	Solution Matrix

	// Rank is the numerical column rank selected at factor construction.
	// Only full-column-rank factors can be returned successfully by these APIs.
	Rank int

	// ResidualNorms contains r independent finite Euclidean residual norms.
	// The slice is detached and may be modified without affecting any factor.
	ResidualNorms []float64

	// ConditionEstimate is kappa_1(A) for LU/Cholesky and kappa_1(R) for QR.
	// It is measured by triangular solves against every identity basis column.
	ConditionEstimate float64
}

// IsNil reports whether the solution-result pointer is nil, independently of
// the numerical data it may hold. It implements the library's nilable contract.
//
// Implementation:
//   - Stage 1: compare the receiver pointer to nil without reading any fields.
//   - Stage 2: return the comparison as a pure boolean result.
//
// Behavior highlights:
//   - A non-nil result whose Solution field was modified remains non-nil.
//   - No receiver access can panic for an absent result.
//
// Inputs: r may be a nil *SolveResult pointer.
// Returns: True exactly when r==nil, false for every allocated SolveResult.
// Errors: None.
// Determinism: Pure pointer comparison with no shared state.
// Complexity: Time O(1), Space O(1).
// Notes: This additive method does not alter existing factor Solve contracts.
// AI-Hints: Test errors before inspecting Solution or ResidualNorms;
// IsNil by itself does not certify a caller-constructed result's validity.
func (r *SolveResult) IsNil() bool {
	return r == nil
}

// LUFactor retains a detached, row-pivoted LU decomposition of a square matrix.
// The private packed factors represent P*(A/scale)=L*U, where P uses original
// row indices and scale is a positive magnitude chosen from the original input.
// Implementation:
//   - Factorization writes exclusively into private packed LU storage.
//   - An independent original matrix copy is kept for true residual calculation.
//   - Solve allocates all mutable workspace locally, permitting concurrent callers.
//
// Behavior highlights: Exact pivot ties prefer the smallest original row index.
// Inputs: Created only through FactorLU; do not construct with a struct literal.
// Returns: Immutable reusable factor, safe for concurrent Solve on read-only inputs.
// Errors: Constructors return nil on failure; nil factor methods never panic.
// Determinism: Column-major elimination with fixed row order and index tie-breaks.
// Complexity: Factorization O(n^3); retained storage O(n^2).
// Notes: Returned permutations are copies, not live views.
// AI-Hints: Do not mutate packed factors or reorder the original permutation in Solve.
type LUFactor struct {
	n         int       // Matrix order; at least one for a usable factor.
	original  []float64 // Original row-major A, retained solely for exact residuals.
	packed    []float64 // Row-major unit-lower/upper decomposition of A/scale.
	perm      []int     // P mapping: current row i -> original input row perm[i].
	scale     float64   // Positive original-coordinate scaling magnitude.
	condition float64   // Measured kappa_1, finite and >=1.
}

// QRFactor retains a full-column-rank Householder QR with column pivoting.
// The compact data satisfy A[:,perm]/scale = Q*R; Q is applied implicitly,
// avoiding construction of a dense m-by-m orthogonal matrix during Solve.
// Implementation:
//   - Upper-triangular entries store R; below-diagonal entries store reflectors.
//   - A tau coefficient is retained per reflector.
//   - A separate original input copy is preserved for residual validation.
//
// Behavior highlights: Pivot ties choose the smallest original column index.
// Inputs: Created only by FactorQR for m>=n>=1 matrices of full numerical rank.
// Returns: Immutable reusable factor, safely shared between Solve goroutines.
// Errors: Rank loss returns nil factor and ErrRankDeficient; no SVD fallback.
// Determinism: Fixed Householder order, exact trailing column-norm recomputation.
// Complexity: O(m*n*n) construction; O(m*n) retained packed/storage copies.
// Notes: ConditionEstimate describes kappa_1(R), not kappa_2(A).
// AI-Hints: Never expose reflector storage as the public Q matrix.
type QRFactor struct {
	m, n      int       // Observation count and unknown count, respectively.
	original  []float64 // Original row-major A; never mutated after construction.
	packed    []float64 // R above diagonal, implicit Householder vectors below it.
	tau       []float64 // Reflector strength for each pivoted column.
	perm      []int     // Current column j -> original input column perm[j].
	scale     float64   // Positive magnitude used to normalize input coefficients.
	condition float64   // Measured kappa_1(R) from independent basis solves.
}

// CholeskyFactor retains a detached lower-triangular factor for a symmetric SPD matrix.
// When symmetry tolerances accept a small asymmetry, the canonical system is
// the entrywise symmetrized matrix rather than either raw input triangle.
// Implementation:
//   - The accepted matrix is copied and normalized before decomposition.
//   - The lower triangle satisfies acceptedA/scale = L*L^T.
//   - Per-call Solve workspaces prevent races between simultaneous solves.
//
// Behavior highlights: Positivity is verified during every diagonal update.
// Inputs: Created only by FactorCholesky on finite square data.
// Returns: Immutable reusable factor with true accepted-matrix residuals.
// Errors: ErrNotPositiveDefinite for nonpositive pivots; nil receivers are safe.
// Determinism: Ascending row/column triangular accumulation.
// Complexity: O(n^3) factorization; retained O(n^2) matrix and factor storage.
// Notes: No tolerance silently regularizes or shifts an indefinite matrix.
// AI-Hints: Tolerance for asymmetry never implies tolerance for nonpositive pivots.
type CholeskyFactor struct {
	n         int       // SPD matrix order.
	accepted  []float64 // Symmetrized, original-coordinate A for residual calculations.
	lower     []float64 // Normalized, row-major lower triangular factor.
	scale     float64   // Positive normalization magnitude.
	condition float64   // Measured kappa_1(acceptedA), finite and >=1.
}
