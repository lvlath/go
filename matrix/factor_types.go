// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix defines reusable factorizations and detached linear-system results.
// This file fixes ownership, dimensional, and diagnostic contracts for numerical solvers.
package matrix

// FactorOptions configures numerical rank selection and condition rejection.
// Its zero value uses a dimension-scaled machine-epsilon rank tolerance without
// imposing a hard condition-number limit on otherwise full-rank inputs.
// Implementation:
//   - RankTolerance is validated before any matrix input is inspected.
//   - ConditionLimit is validated and applied to a measured 1-norm condition estimate.
//
// Behavior highlights:
//   - RankTolerance=0 selects 2^-52*max(rows,cols); positive values are relative.
//   - ConditionLimit=0 disables rejection; values >=1 reject estimates strictly above the limit.
//
// Inputs:
//   - RankTolerance: finite, nonnegative, dimensionless relative diagonal cutoff.
//   - ConditionLimit: finite 0 or >=1, dimensionless diagnostic threshold.
//
// Returns: A value-only immutable policy snapshot stored by each factor.
// Errors: ErrInvalidOption for NaN, infinity, negative tolerance, or a limit in (0,1).
// Determinism: No implicit state, randomness, or environment-dependent options.
// Complexity: O(1) validation time and storage.
// Notes: Numerical rank is a scale-relative heuristic, not algebraic rank certification.
// AI-Hints: Never interpret ConditionLimit as an independently certified bound on kappa_2.
type FactorOptions struct {
	// RankTolerance is a relative cutoff for the largest absolute diagonal
	// of the pivoted U or R. Equality to the cutoff counts as rank loss.
	RankTolerance float64

	// ConditionLimit optionally rejects the measured 1-norm condition number.
	// The comparison is strictly greater, so equality remains admissible.
	ConditionLimit float64
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
