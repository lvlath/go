// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package: matrix
//
// Purpose:
//   - Provide a single, canonical source of truth for common validation checks.
//   - Keep kernels and public facades small by delegating guard logic here.
//   - Return sentinel errors directly so call sites can wrap uniformly (if desired).
//
// Determinism & Performance:
//   - All validators are deterministic and side-effect free.
//   - Structural checks are O(1); value scans are O(r*c) with fixed traversal order.
//
// AI-Hints:
//   - Use ValidateBinarySameShape before element-wise kernels (Add/Sub/Hadamard).
//   - Use ValidateMulCompatible before MatMul to fail fast on dimension mismatch.
//   - Keep ValidateGraphAdjacency structural; apply ValidateAllFinite separately when needed.

package matrix

import (
	"fmt"
	"math"

	"github.com/lvlath/go/core"
)

// matrixCheckedElementCount validates a row-major element count before allocation.
// It distinguishes legal zero-shaped matrices from impossible dimensions and
// prevents signed integer overflow from wrapping a positive element product.
// Implementation:
//   - Stage 1: reject negative row or column counts.
//   - Stage 2: handle any zero-element domain without allocating storage.
//   - Stage 3: check the quotient against MaxInt, then compute the product.
//
// Behavior highlights:
//   - The helper is independent of the Matrix backend and allocation policy.
//
// Inputs:
//   - rows, cols: nonnegative shape components; zero is valid.
//
// Returns:
//   - int: exact number of elements, including zero for empty domains.
//   - error: ErrInvalidDimensions for negative or overflowing sizes.
//
// Errors:
//   - ErrInvalidDimensions for an impossible element count.
//
// Determinism: Fixed sign and quotient checks, with no allocation.
// Complexity: Time O(1), Space O(1).
// Notes: A representable element count does not guarantee enough RAM for make.
// AI-Hints:
//   - Never form rows*cols before checking rows <= MaxInt/cols.
//   - This guard protects shape arithmetic; it is not an OOM recovery strategy.
func matrixCheckedElementCount(rows, cols int) (int, error) {
	if rows < 0 || cols < 0 {
		return 0, ErrInvalidDimensions
	}
	if rows == 0 || cols == 0 {
		return 0, nil
	}
	maxInt := int(^uint(0) >> 1)
	if rows > maxInt/cols {
		return 0, ErrInvalidDimensions
	}
	return rows * cols, nil
}

// distanceMatrixAllowsInfByDefault fixes the distance/APSP convention:
//
//   - ordinary Dense matrices reject +Inf by default;
//   - distance matrices allow +Inf as the "no path" sentinel by default.
//
// Callers may still pass WithDisallowInfDistances() to require a complete finite
// distance matrix.
const distanceMatrixAllowsInfByDefault = true

// isNonFinite REPORTS whether v is NaN or ±Inf.
//
// Implementation:
//   - Stage 1: math.IsNaN(v).
//   - Stage 2: math.IsInf(v, 0).
//
// Behavior highlights:
//   - Pure predicate with IEEE-754 semantics.
//
// Inputs:
//   - v: scalar value.
//
// Returns:
//   - bool: true if v is not a finite real number.
//
// Errors:
//   - None.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Treats both +Inf and -Inf as non-finite.
//
// AI-Hints:
//   - Use this to guard tolerances/bounds before starting an O(n²) scan.
func isNonFinite(v float64) bool {
	return math.IsNaN(v) || math.IsInf(v, 0)
}

// isNaNOrNegInf REPORTS whether v is NaN or -Inf (allows +Inf as "no path").
func isNaNOrNegInf(v float64) bool {
	return math.IsNaN(v) || math.IsInf(v, -1)
}

// validateTol NORMALIZES a tolerance to a non-negative finite value.
//
// Implementation:
//   - Stage 1: Reject NaN/±Inf → ErrNaNInf.
//   - Stage 2: If negative, flip sign.
//
// Behavior highlights:
//   - Accepts negative inputs and converts to |tol| to avoid surprising callers.
//
// Inputs:
//   - tol: tolerance value (finite).
//
// Returns:
//   - float64: normalized tol >= 0.
//
// Errors:
//   - ErrNaNInf if tol is NaN/±Inf.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - A tol of 0 enforces exact comparisons.
//
// AI-Hints:
//   - Use validateTol for rtol/atol to make “sign mistakes” non-fatal and predictable.
func validateTol(tol float64) (float64, error) {
	if isNonFinite(tol) {
		return 0, ErrNaNInf
	}
	if tol < 0 {
		tol = -tol
	}

	return tol, nil
}

// validateBounds VALIDATES and NORMALIZES numeric bounds (lo, hi).
//
// Implementation:
//   - Stage 1: Reject NaN/±Inf in either bound → ErrNaNInf.
//   - Stage 2: If lo > hi, swap.
//
// Behavior highlights:
//   - Makes inverted bounds deterministic and non-fatal.
//
// Inputs:
//   - lo, hi: bound endpoints (finite).
//
// Returns:
//   - (lo, hi): normalized so lo <= hi.
//
// Errors:
//   - ErrNaNInf if either bound is NaN/±Inf.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This is a normalization helper; it does not clamp values by itself.
//
// AI-Hints:
//   - Use this before Clip/threshold logic to avoid writing “lo/hi swap” in every call site.
func validateBounds(lo, hi float64) (float64, float64, error) {
	if isNonFinite(lo) || isNonFinite(hi) {
		return 0, 0, ErrNaNInf
	}
	if lo > hi {
		lo, hi = hi, lo
	}

	return lo, hi, nil
}

// ValidateNotNil ENSURES the matrix reference is non-nil.
//
// Implementation:
//   - Stage 1: Check interface value against nil.
//   - Stage 2 (optional): If m implements core.Nilable, consult m.IsNil() to detect typed-nil.
//
// Behavior highlights:
//   - Canonical nil-guard for all composite validators.
//
// Inputs:
//   - m: matrix interface value.
//
// Returns:
//   - nil on success.
//
// Errors:
//   - ErrNilMatrix if m == nil.
//   - ErrNilMatrix if m is a typed-nil inside interface AND implements core.Nilable.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This validates the interface value; it does not validate internal storage.
//
// AI-Hints:
//   - Call this first in any exported operation that dereferences m.
func ValidateNotNil(m Matrix) error {
	// If the matrix is nil, fail with the unified sentinel.
	if m == nil {
		return ErrNilMatrix
	}

	// Optional typed-nil detection without reflect:
	// if the implementation provides core.Nilable, trust its IsNil().
	if n, ok := m.(core.Nilable); ok && n.IsNil() {
		return ErrNilMatrix
	}

	return nil
}

// IsZeroShape reports whether m has an empty element domain.
//
// What:
//   - Returns true for 0×0, 0×N, and N×0 shapes.
//
// Why:
//   - Element-wise scans over such matrices are deterministic no-ops.
//   - Statistics may return correctly-sized metadata without touching elements.
//
// Inputs:
//   - m: matrix already known to be non-nil.
//
// Returns:
//   - bool: true when Rows()==0 or Cols()==0.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// AI-Hints:
//   - Call ValidateNotNil before this helper at public boundaries.
//   - Do not confuse zero-shape with nil.
func IsZeroShape(m Matrix) bool {
	if m == nil {
		return false
	}

	return m.Rows() == 0 || m.Cols() == 0
}

// ValidateNonEmptyShape ensures that m has at least one row and one column.
//
// What:
//   - Rejects nil and zero-sized shapes.
//
// Why:
//   - Some algorithms require at least one actual element, even though the package
//     generally accepts zero-sized matrices structurally.
//
// Errors:
//   - ErrNilMatrix from ValidateNotNil.
//   - ErrInvalidDimensions when Rows()==0 or Cols()==0.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// AI-Hints:
//   - Do not use this in element-wise kernels that can operate as no-ops.
//   - Use it only when the mathematics requires a non-empty element domain.
func ValidateNonEmptyShape(m Matrix) error {
	if err := ValidateNotNil(m); err != nil {
		return err
	}

	if IsZeroShape(m) {
		return ErrInvalidDimensions
	}

	return nil
}

// ValidateSameShape ENSURES matrices a and b have equal dimensions.
//
// Implementation:
//   - Stage 1: Compare Rows.
//   - Stage 2: Compare Cols.
//
// Behavior highlights:
//   - Pure structural check; does not inspect values.
//
// Inputs:
//   - a,b: matrices (must be non-nil).
//
// Returns:
//   - nil on success.
//
// Errors:
//   - ErrDimensionMismatch if shapes differ.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Prefer ValidateBinarySameShape when you also need nil checks.
//
// AI-Hints:
//   - Use this before allocating output buffers to avoid wasted work.
func ValidateSameShape(a, b Matrix) error {
	if a.Rows() != b.Rows() || a.Cols() != b.Cols() {
		return ErrDimensionMismatch
	}

	return nil
}

// ValidateSquare ENSURES m is square (Rows == Cols).
//
// Implementation:
//   - Stage 1: Reject nil.
//   - Stage 2: Compare Rows and Cols.
//
// Behavior highlights:
//   - Structural precondition for spectral / factorization methods.
//
// Inputs:
//   - m: matrix.
//
// Returns:
//   - nil on success.
//
// Errors:
//   - ErrNilMatrix if m is nil.
//   - ErrNonSquare if Rows != Cols.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Use ValidateSquareNonNil when you want an explicit composite name.
//
// AI-Hints:
//   - Apply this before symmetry checks to avoid ambiguous “triangle scans” on non-square matrices.
func ValidateSquare(m Matrix) error {
	if err := ValidateNotNil(m); err != nil {
		return err
	}
	// Check the square condition explicitly.
	if m.Rows() != m.Cols() {
		return ErrNonSquare
		// return ErrDimensionMismatch
	}

	return nil
}

// ValidateVecLen ENSURES vector length matches n.
//
// Implementation:
//   - Stage 1: Reject nil slice.
//   - Stage 2: Check len(x) == n.
//
// Behavior highlights:
//   - Canonical guard for MatVec-like operations.
//
// Inputs:
//   - x: vector slice.
//   - n: expected length.
//
// Returns:
//   - nil on success.
//
// Errors:
//   - ErrNilMatrix if x is nil (generic nil-argument sentinel).
//   - ErrDimensionMismatch if len(x) != n.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Uses ErrNilMatrix as a generic nil-argument sentinel.
//
// AI-Hints:
//   - If you need “nil allowed means zero-vector”, do not use this validator.
func ValidateVecLen(x []float64, n int) error {
	// Disallow nil vectors to avoid subtle bugs in MatVec-like routines.
	if x == nil {
		return ErrNilMatrix
	}
	// Check the exact expected length.
	if len(x) != n {
		return ErrDimensionMismatch
	}

	return nil
}

// ValidateBinarySameShape COMPOSES NotNil(a) → NotNil(b) → SameShape.
//
// Implementation:
//   - Stage 1: ValidateNotNil(a).
//   - Stage 2: ValidateNotNil(b).
//   - Stage 3: ValidateSameShape(a,b).
//
// Behavior highlights:
//   - Standard guard sequence for element-wise operations.
//
// Inputs:
//   - a,b: matrices.
//
// Returns:
//   - nil on success.
//
// Errors:
//   - ErrNilMatrix if any input is nil.
//   - ErrDimensionMismatch if shapes differ.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Keeps wrapping out of validators; callers can wrap with operation context.
//
// AI-Hints:
//   - Use this at public API boundaries; internal kernels may assume non-nil.
func ValidateBinarySameShape(a, b Matrix) error {
	if err := ValidateNotNil(a); err != nil {
		return err
	}
	if err := ValidateNotNil(b); err != nil {
		return err
	}

	return ValidateSameShape(a, b)
}

// ValidateSquareNonNil COMPOSES NotNil → Square.
//
// Implementation:
//   - Stage 1: ValidateNotNil.
//   - Stage 2: ValidateSquare.
//
// Behavior highlights:
//   - Named composite to clarify intent at call sites.
//
// Inputs:
//   - m: matrix.
//
// Returns:
//   - nil on success.
//
// Errors:
//   - ErrNilMatrix, ErrNonSquare.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Equivalent to ValidateSquare but kept for readability in composites.
//
// AI-Hints:
//   - Prefer this in exported graph/spectral APIs where “square required” is part of the contract.
func ValidateSquareNonNil(m Matrix) error {
	return ValidateSquare(m)
}

// ValidateMulCompatible ENSURES a.Cols == b.Rows, inputs non-nil.
//
// Implementation:
//   - Stage 1: ValidateNotNil(a), ValidateNotNil(b).
//   - Stage 2: Compare a.Cols() and b.Rows().
//
// Behavior highlights:
//   - Canonical dimension guard for MatMul.
//
// Inputs:
//   - a,b: matrices.
//
// Returns:
//   - nil on success.
//
// Errors:
//   - ErrNilMatrix if any input is nil.
//   - ErrDimensionMismatch if a.Cols != b.Rows.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This does not check overflow/NaN behavior in multiplication.
//
// AI-Hints:
//   - ValidateMulCompatible early to avoid allocating output on invalid shapes.
func ValidateMulCompatible(a, b Matrix) error {
	if err := ValidateNotNil(a); err != nil {
		return err
	}
	if err := ValidateNotNil(b); err != nil {
		return err
	}
	if a.Cols() != b.Rows() {
		return ErrDimensionMismatch
	}

	return nil
}

// ValidateSymmetric ENSURES A is symmetric within tolerance tol:
// |A[i,j] - A[j,i]| <= tol for all i<j.
//
// Implementation:
//   - Stage 1: ValidateSquare (includes nil guard).
//   - Stage 2: Normalize tol via validateTol.
//   - Stage 3: Scan upper triangle; reject non-finite entries; compare deviations.
//
// Behavior highlights:
//   - Deterministic traversal order and early-exit on first violation.
//
// Inputs:
//   - m: square matrix.
//   - tol: finite tolerance (negative allowed; treated as |tol|).
//
// Returns:
//   - nil on success.
//
// Errors:
//   - ErrNilMatrix, ErrNonSquare from structure checks..
//   - ErrNaNInf if tol is NaN/±Inf or if any compared entry is non-finite.
//   - ErrAsymmetry if symmetry violation exceeds tol.
//
// Determinism:
//   - Deterministic (fixed i→j scan).
//
// Complexity:
//   - Time O(n²), Space O(1).
//
// Notes:
//   - Non-finite entries make symmetry comparisons ill-defined and are rejected.
//
// AI-Hints:
//   - Use tol=0 for strict symmetry on exact constructions (identity, Laplacians with exact weights).
func ValidateSymmetric(m Matrix, tol float64) error {
	// Guard nil first.
	if err := ValidateSquare(m); err != nil {
		return err
	}
	// Normalize tolerance to a non-negative finite value.
	t, err := validateTol(tol)
	if err != nil {
		return err
	}

	n := m.Rows()
	if n <= 1 {
		return nil
	}

	// Scan diagonal
	var i int       // loop counter
	var aii float64 // A[i,i]
	for i = 0; i < n; i++ {
		aii, err = m.At(i, i)
		if err != nil {
			return fmt.Errorf("ValidateSymmetric: At(%d,%d): %w", i, i, err)
		}
		if isNonFinite(aii) {
			return fmt.Errorf("ValidateSymmetric: diagonal[%d,%d]=%v: %w", i, i, aii, ErrNaNInf)
		}
	}
	// Scan the strict upper triangle once, tracking the maximum deviation.
	// Deterministic i→j order ensures reproducible short-circuiting behavior.
	var j int            // loop counter
	var aij, aji float64 // A[i,j] and A[j,i]
	for i = 0; i < n; i++ {
		for j = i + 1; j < n; j++ {
			aij, err = m.At(i, j)
			if err != nil {
				return err
			}
			aji, err = m.At(j, i)
			if err != nil {
				return err
			}

			if isNonFinite(aij) || isNonFinite(aji) {
				return ErrNaNInf
			}

			if math.Abs(aij-aji) > t {
				return ErrAsymmetry
			}
		}
	}

	// At this point, all |A[i,j]-A[j,i]| ≤ tol, so A is symmetric within tol.
	// Callers (e.g., Eigen) can treat (maxOff == 0) as a "diagonal already" shortcut.
	return nil
}

// IsZeroOffDiagonal REPORTS whether max_{i!=j} |A[i,j]| <= tol.
//
// Implementation:
//   - Stage 1: ValidateSquare (includes nil guard).
//   - Stage 2: Normalize tol via validateTol.
//   - Stage 3: Scan all off-diagonal entries; reject non-finite; compare |v| to tol.
//
// Behavior highlights:
//   - Fast early-exit when a single entry exceeds tolerance.
//
// Inputs:
//   - m: square matrix.
//   - tol: finite tolerance (negative allowed; treated as |tol|).
//
// Returns:
//   - bool: true if all off-diagonal entries are within tol.
//
// Errors:
//   - ErrNilMatrix, ErrNonSquare from structure checks.
//   - ErrNaNInf if tol is NaN/±Inf or if any inspected entry is non-finite.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(n²), Space O(1).
//
// Notes:
//   - This is a numeric predicate; it does not mutate the matrix.
//
// AI-Hints:
//   - Use this to short-circuit diagonal-only fast paths (Jacobi, repeated deflation loops).
func IsZeroOffDiagonal(m Matrix, tol float64) (bool, error) {
	if err := ValidateSquare(m); err != nil {
		return false, err
	}

	t, err := validateTol(tol)
	if err != nil {
		return false, err
	}

	n := m.Rows()
	if n <= 1 {
		return true, nil
	}

	var i, j int
	var v float64
	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			if i == j {
				continue
			}
			v, err = m.At(i, j)
			if err != nil {
				return false, err
			}
			if isNonFinite(v) {
				return false, ErrNaNInf
			}
			if math.Abs(v) > t {
				return false, nil
			}
		}
	}

	return true, nil
}

// ValidateGraphAdjacency VALIDATES an adjacency-matrix wrapper structurally.
//
// Implementation:
//   - Stage 1: Reject nil wrapper or nil Mat.
//   - Stage 2: ValidateSquare(Mat).
//   - Stage 3: If VertexIndex is present, ensure its size matches N and indices are in [0,N).
//
// Behavior highlights:
//   - Structural-only: does not inspect matrix values (finite-ness is a separate concern).
//
// Inputs:
//   - am: adjacency matrix wrapper.
//
// Returns:
//   - nil on success.
//
// Errors:
//   - ErrNonSquare if Mat is not square.
//   - ErrDimensionMismatch if index metadata contradicts dimensions.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1) to O(|V|) depending on index metadata scan, Space O(1).
//
// Notes:
//   - Value validation (NaN/Inf) must be done explicitly via ValidateAllFinite if required.
//
// AI-Hints:
//   - Keep this check at graph API boundaries; do not pay O(n²) scans unless the algorithm needs them.
func ValidateGraphAdjacency(am *AdjacencyMatrix) error {
	if am == nil || am.Mat == nil {
		return ErrNilMatrix
	}
	if err := ValidateSquare(am.Mat); err != nil {
		return err
	}

	// If the type exposes VertexIndex, validate its basic consistency without allocating.
	if am.VertexIndex != nil {
		n := am.Mat.Rows()
		if len(am.VertexIndex) != n {
			return ErrDimensionMismatch
		}
		for _, idx := range am.VertexIndex {
			if idx < 0 || idx >= n {
				return ErrDimensionMismatch
			}
		}
	}

	return nil
}

// ValidateAllFinite ENSURES every entry of m is finite.
//
// Implementation:
//   - Stage 1: Reject nil.
//   - Stage 2: Scan all entries in deterministic row-major order.
//
// Behavior highlights:
//   - Explicit value-level validator (opt-in).
//
// Inputs:
//   - m: matrix.
//
// Returns:
//   - nil on success.
//
// Errors:
//   - ErrNilMatrix if m is nil.
//   - ErrNaNInf if any entry is NaN/±Inf.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(r*c), Space O(1).
//
// Notes:
//   - Use only when the algorithm cannot tolerate non-finite values.
//
// AI-Hints:
//   - Apply after raw ingest (Fill) when you want to enforce “finite-only” preconditions.
func ValidateAllFinite(m Matrix) error {
	err := ValidateNotNil(m)
	if err != nil {
		return err
	}

	r, c := m.Rows(), m.Cols()
	var i, j int
	var v float64
	for i = 0; i < r; i++ {
		for j = 0; j < c; j++ {
			v, err = m.At(i, j)
			if err != nil {
				return err
			}
			if isNonFinite(v) {
				return ErrNaNInf
			}
		}
	}

	return nil
}

// ValidateDistanceMatrix VALIDATES the semantic numeric contract of a distance/APSP matrix.
//
// What:
//   - Checks that m is a square distance matrix suitable for Floyd–Warshall-style
//     shortest-path processing.
//
// Why:
//   - Ordinary Dense matrices normally reject infinities.
//   - Distance matrices intentionally use +Inf as "no path".
//   - This validator centralizes that domain exception without weakening NaN/-Inf safety.
//
// Implementation:
//   - Stage 1: assemble options with distance semantics:
//     +Inf is allowed by default, while caller-provided options may override it.
//   - Stage 2: validate non-nil square shape through existing validators.
//   - Stage 3: normalize tolerance through validateTol.
//   - Stage 4: scan entries in row-major order.
//   - Stage 5: reject NaN and -Inf everywhere.
//   - Stage 6: allow +Inf only off-diagonal and only when allowInfDistances is true.
//   - Stage 7: classify diagonal:
//     negative below tolerance => ErrNegativeCycle;
//     positive/non-zero above tolerance => ErrBadShape.
//
// Behavior highlights:
//   - Off-diagonal finite negative distances are allowed.
//   - +Inf off-diagonal means "no path".
//   - +Inf diagonal is invalid.
//   - The validator does not require triangle inequality; Floyd–Warshall produces closure.
//
// Inputs:
//   - m: matrix to validate.
//   - opts: optional policy. WithEpsilon and WithDisallowInfDistances are meaningful.
//
// Returns:
//   - nil when m satisfies the distance-matrix input contract.
//   - sentinel-preserving error otherwise.
//
// Errors:
//   - ErrNilMatrix: nil or typed-nil matrix.
//   - ErrNonSquare: non-square matrix, via ValidateSquareNonNil.
//   - ErrInvalidDimensions: negative shape from a malformed Matrix implementation.
//   - ErrNaNInf: NaN, -Inf, or disallowed +Inf.
//   - ErrBadShape: invalid non-negative diagonal.
//   - ErrNegativeCycle: negative diagonal below tolerance.
//
// Determinism:
//   - Row-major scan order; first failing cell is reported deterministically.
//
// Complexity:
//   - Time O(n²), Space O(1).
//
// Notes:
//   - WithNoValidateNaNInf does not disable this semantic validator.
//   - This function validates distance-domain meaning, not Dense write policy.
//
// AI-Hints:
//   - Do not replace this with ValidateSquareNonNil in FloydWarshall.
//   - Do not treat +Inf as valid on the diagonal.
//   - Do not reject finite negative off-diagonal distances; Floyd–Warshall supports them.
func ValidateDistanceMatrix(m Matrix, opts ...Option) error {
	// Distance matrices use +Inf as "no path" by default.
	//
	// We intentionally do this locally rather than introducing another option
	// assembly helper. The public option system remains the single source of
	// truth; this validator only prepends the distance-domain default.
	distanceOpts := make([]Option, 0, len(opts)+1)
	if distanceMatrixAllowsInfByDefault {
		distanceOpts = append(distanceOpts, WithAllowInfDistances())
	}
	distanceOpts = append(distanceOpts, opts...)

	cfg, err := gatherOptions(distanceOpts...)
	if err != nil {
		return err
	}

	if err = ValidateSquareNonNil(m); err != nil {
		return err
	}

	rows := m.Rows()
	cols := m.Cols()
	if rows < 0 || cols < 0 {
		return ErrInvalidDimensions
	}

	tol, err := validateTol(cfg.eps)
	if err != nil {
		return err
	}

	var i, j int
	var v float64

	for i = 0; i < rows; i++ {
		for j = 0; j < cols; j++ {
			v, err = m.At(i, j)
			if err != nil {
				return fmt.Errorf("ValidateDistanceMatrix: At(%d,%d): %w", i, j, err)
			}

			if isNaNOrNegInf(v) {
				return fmt.Errorf("ValidateDistanceMatrix: invalid distance[%d,%d]=%v: %w", i, j, v, ErrNaNInf)
			}

			if math.IsInf(v, 1) {
				if i == j {
					return fmt.Errorf("ValidateDistanceMatrix: +Inf diagonal[%d,%d]: %w", i, j, ErrBadShape)
				}
				if !cfg.allowInfDistances {
					return fmt.Errorf("ValidateDistanceMatrix: disallowed +Inf distance[%d,%d]: %w", i, j, ErrNaNInf)
				}
				continue
			}

			if i == j {
				if v < -tol {
					return fmt.Errorf("ValidateDistanceMatrix: negative diagonal[%d,%d]=%v: %w", i, j, v, ErrNegativeCycle)
				}
				if math.Abs(v) > tol {
					return fmt.Errorf("ValidateDistanceMatrix: non-zero diagonal[%d,%d]=%v: %w", i, j, v, ErrBadShape)
				}
			}
		}
	}

	return nil
}
