// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix provides universal operations on any Matrix implementation,
// including element-wise addition, subtraction, matrix multiplication,
// transpose, and scalar scaling. All functions perform strict
// fail-fast validation and return clear errors on dimension mismatches.
//
// Purpose:
//   - Declare canonical linear-algebra kernels (signatures) used across the package.
//   - Define operation tags and shared constants for determinism and error reporting.
//
// Notes:
//   - Implementations live in dedicated kernel files (same package) to keep roles clean.
//   - All kernels must use central validators and return plain sentinels or wrapped via matrixErrorf at the facade.
package matrix

import (
	"fmt"
	"math"
)

// NormZero is the additive identity for norm and accumulation operations.
const NormZero = 0.0

// ZeroSum is the initial sum value for forward/backward substitution and similar.
const ZeroSum = 0.0

// ZeroPivot is the sentinel for detecting a zero pivot in LU/Inverse routines.
const ZeroPivot = 0.0

// Operation name constants for unified error wrapping and reducing magic strings.
const (
	opAdd       = "Add"
	opSub       = "Sub"
	opMul       = "Mul"
	opTranspose = "Transpose"
	opScale     = "Scale"
	opEigen     = "Eigen"
	opInverse   = "Inverse"
	opLU        = "LU"
	opQR        = "QR"
	opHadamard  = "Hadamard"
	opMatVec    = "MatVec"
)

// matrixErrorf wraps err with an operation tag, preserving the original error via %w.
// The wrapper keeps a stable "Op: underlying" shape for uniform reporting across facades.
// Use only when err != nil to avoid creating a non-nil wrapper around a nil cause.
//
// Implementation:
//   - Stage 1: Wrap using fmt.Errorf("%s: %w", tag, err) to enable errors.Is/As.
//
// Behavior highlights:
//   - Preserves the underlying sentinel/type for errors.Is/errors.As.
//   - Keeps human-readable operation prefixes (e.g., "Add|Sub", "Transpose").
//
// Inputs:
//   - tag: operation name/label (use package-level op* constants; no magic strings).
//   - err: underlying non-nil error to wrap.
//
// Returns:
//   - error: a non-nil error that formats as "<tag>: <underlying>" and still matches Is/As.
//
// Errors:
//   - None produced here; this function assumes err != nil. Caller responsibility.
//
// Determinism:
//   - Fully deterministic formatting; no data-dependent branches.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Wrapping nil with %w yields a non-nil error that wraps a nil cause; do not do this.
//   - Centralizes formatting so all kernels expose uniform error surfaces.
//
// AI-Hints:
//   - Always gate calls with `if err != nil { return nil, matrixErrorf(tag, err) }`.
//   - Keep `tag` to the canonical constants to simplify log/search pipelines.
func matrixErrorf(tag string, err error) error {
	return fmt.Errorf("%s: %w", tag, err)
}

// addSub computes elementwise out = a + sign*b for sign ∈ {+1, -1}.
// Inputs must have identical shapes. A fresh Dense is allocated; operands are not mutated.
// Internal helper for Add/Sub to share validation, allocation, and fast-path.
//
// Implementation:
//   - Stage 1: ValidateBinarySameShape(a, b). Allocate result Dense(rows, cols).
//   - Stage 2: Fast-path if both are *Dense - single flat loop 0..n-1.
//     Otherwise, fallback At/Set with fixed i→j order.
//
// Behavior highlights:
//   - Deterministic loop orders (flat in fast-path; i→j in fallback).
//   - Single result allocation; no inner-loop temps beyond scalars.
//   - Inputs remain immutable.
//
// Inputs:
//   - a, b: conformable matrices (non-nil; same rows/cols).
//   - sign: +1 for Add, −1 for Sub (callers must enforce).
//   - opTag: opAdd for Add, opSub for Sub (for error wrapping).
//
// Returns:
//   - Matrix: newly allocated Dense with the result.
//   - error : validation/allocation failures wrapped with opAdd/opSub.
//
// Errors:
//   - ErrNilMatrix          (from ValidateBinarySameShape when a or b is nil).
//   - ErrDimensionMismatch  (from ValidateBinarySameShape when shapes differ).
//   - Allocation errors     (from NewDense).
//
// Determinism:
//   - Fast-path: single flat slice walk 0..(r*c−1).
//   - Fallback: fixed nested loops i=0..r−1, j=0..c−1.
//
// Complexity:
//   - Time O(r*c), Space O(r*c) for the new result.
//
// Notes:
//   - Keeping `sign` as a float avoids an extra branch inside the hot loop.
//   - The function is unexported by design; invariants are enforced by Add/Sub.
//
// AI-Hints:
//   - To trigger fast-path, pass concrete *Dense operands (avoid interface wrappers).
//   - If you need in-place add/sub, implement a dedicated kernel; do not modify inputs here.
//   - Prefer batching several add/sub calls at a higher level to amortize allocations.
func addSub(a, b Matrix, sign float64, opTag string) (Matrix, error) {
	// Validate shapes match
	if err := ValidateBinarySameShape(a, b); err != nil {
		return nil, matrixErrorf(opTag, err)
	}

	// Allocate result Dense
	rows, cols := a.Rows(), a.Cols()
	res, err := NewDense(rows, cols)
	if err != nil {
		return nil, matrixErrorf(opTag, err)
	}

	// Fast path: *Dense with *Dense → single flat loop.
	if da, okA := a.(*Dense); okA {
		if db, okB := b.(*Dense); okB {
			// direct element-wise addition on backing slices
			length := rows * cols
			for idx := 0; idx < length; idx++ { // deterministic 0..n-1
				res.data[idx] = da.data[idx] + sign*db.data[idx]
			}

			return res, nil
		}
	}

	// Fallback: interface path with fixed i→j order.
	var i, j int       // loop iterators (deterministic order)
	var av, bv float64 // element temporaries
	for i = 0; i < rows; i++ {
		for j = 0; j < cols; j++ {
			// Read a(i,j).
			av, err = a.At(i, j)
			if err != nil {
				return nil, matrixErrorf(opTag, fmt.Errorf("At(%d,%d): %w", i, j, err))
			}
			// Read b(i,j).
			bv, err = b.At(i, j)
			if err != nil {
				return nil, matrixErrorf(opTag, fmt.Errorf("At(%d,%d): %w", i, j, err))
			}
			// Write result(i,j).
			if err = res.Set(i, j, av+sign*bv); err != nil {
				return nil, matrixErrorf(opTag, fmt.Errorf("Set(%d,%d): %w", i, j, err))
			}
		}
	}

	// Return result
	return res, nil
}

// Add computes the element-wise sum C = A + B and returns a fresh Dense result.
// Implementation:
//   - Stage 1: Validate both operands are non-nil and have identical shapes.
//   - Stage 2: If both are *Dense, run a single flat loop; otherwise fall back to i→j.
//
// Behavior highlights:
//   - Deterministic loop order; no hidden aliasing; one allocation for the result.
//
// Inputs:
//   - A: left matrix operand (any Matrix).
//   - B: right matrix operand (any Matrix) with the same shape as A.
//
// Returns:
//   - Matrix: a new Dense with C[i,j] = A[i,j] + B[i,j].
//
// Errors:
//   - ErrNilMatrix (nil input), ErrDimensionMismatch (shape mismatch).
//
// Determinism:
//   - Flat 0..n-1 for *Dense; i→j for the generic path.
//
// Complexity:
//   - Time O(r*c), Space O(r*c). The fast path is bandwidth-bound.
//
// Notes:
//   - Inputs are never mutated; result is always a freshly allocated Dense.
//
// AI-Hints:
//   - Prefer *Dense inputs for tight loops and contiguous data; hide concrete types
//     (e.g., via wrappers) to force the fallback path in tests or when needed.
func Add(a, b Matrix) (Matrix, error) { return addSub(a, b, +1, opAdd) }

// Sub computes the element-wise difference C = A - B and returns a fresh Dense result.
// Implementation:
//   - Stage 1: Validate both operands are non-nil and have identical shapes.
//   - Stage 2: If both are *Dense, run a single flat loop; otherwise fall back to i→j.
//
// Behavior highlights:
//   - Deterministic loop order; no hidden aliasing; one allocation for the result.
//
// Inputs:
//   - A: left matrix operand (any Matrix).
//   - B: right matrix operand (any Matrix) with the same shape as A.
//
// Returns:
//   - Matrix: a new Dense with C[i,j] = A[i,j] - B[i,j].
//
// Errors:
//   - ErrNilMatrix (nil input), ErrDimensionMismatch (shape mismatch).
//
// Determinism:
//   - Flat 0..n-1 for *Dense; i→j for the generic path.
//
// Complexity:
//   - Time O(r*c), Space O(r*c). The fast path is bandwidth-bound.
//
// Notes:
//   - Inputs are never mutated; result is always a freshly allocated Dense.
//
// AI-Hints:
//   - Prefer *Dense inputs for tight loops and contiguous data; hide concrete types
//     (e.g., via wrappers) to force the fallback path in tests or when needed.
func Sub(a, b Matrix) (Matrix, error) { return addSub(a, b, -1, opSub) }

// Mul performs standard matrix multiplication C = A × B (no aliasing).
// Implementation:
//   - Stage 1: Validate A,B (not nil) and inner dimensions (A.Cols == B.Rows).
//   - Stage 2: If A and B are *Dense, use i→k→j with row-major strides and skip zeros;
//     otherwise use i→j→k with a fixed order and zero-skip on A[i,k].
//
// Behavior highlights:
//   - Deterministic triple loops; no temporary tiles; one allocation for C.
//
// Inputs:
//   - A: left matrix with shape (r × n).
//   - B: right matrix with shape (n × c).
//
// Returns:
//   - Matrix: new Dense C with shape (r × c).
//
// Errors:
//   - ErrNilMatrix (nil input), ErrDimensionMismatch (inner mismatch).
//
// Determinism:
//   - Fixed loop orders (i→k→j for fast path, i→j→k for fallback).
//
// Complexity:
//   - Time O(r*n*c), Space O(r*c). Skipping zero A[i,k] avoids useless multiplies.
//
// Notes:
//   - For extremely sparse workloads consider dedicated sparse kernels outside this package.
//
// AI-Hints:
//   - If you can keep A as *Dense and cache-friendly by rows, you unlock the best path here.
func Mul(a, b Matrix) (Matrix, error) {
	// Validate inputs via canonical validator
	if err := ValidateMulCompatible(a, b); err != nil {
		return nil, matrixErrorf(opMul, err)
	}

	// Allocate result Dense
	aRows, aCols, bCols := a.Rows(), a.Cols(), b.Cols()
	res, err := NewDense(aRows, bCols)
	if err != nil {
		return nil, matrixErrorf(opMul, err)
	}
	var (
		i, j, k         int // loop iterators
		av, bv, current float64
	)
	// Fast-path for two Dense matrices
	if da, okA := a.(*Dense); okA {
		if db, okB := b.(*Dense); okB {
			// row-major multiplication into res.data
			// da.data layout: i*aCols + k
			// db.data layout: k*bCols + j
			var rowOffsetA, rowOffsetB, rowOffsetR int
			for i = 0; i < aRows; i++ {
				rowOffsetA = i * aCols
				rowOffsetR = i * bCols
				for k = 0; k < aCols; k++ {
					av = da.data[rowOffsetA+k]
					if av == 0 {
						continue // skip zero for performance
					}
					rowOffsetB = k * bCols
					for j = 0; j < bCols; j++ {
						res.data[rowOffsetR+j] += av * db.data[rowOffsetB+j]
					}
				}
			}
			return res, nil
		}
	}

	// Fallback: generic interface triple-loop (i-j-k)
	for i = 0; i < aRows; i++ {
		for j = 0; j < bCols; j++ {
			current = ZeroSum
			for k = 0; k < aCols; k++ {
				av, err = a.At(i, k)
				if err != nil {
					return nil, matrixErrorf(opMul, fmt.Errorf("At(%d,%d): %w", i, k, err))
				}
				if av == 0 {
					continue // skip zero for performance
				}
				bv, err = b.At(k, j)
				if err != nil {
					return nil, matrixErrorf(opMul, fmt.Errorf("At(%d,%d): %w", k, j, err))
				}
				current += av * bv // accumulate product
			}
			if err = res.Set(i, j, current); err != nil {
				return nil, matrixErrorf(opMul, fmt.Errorf("Set(%d,%d): %w", i, j, err))
			}
		}
	}

	// Return result
	return res, nil
}

// Transpose returns a new matrix with rows and columns swapped (mᵀ).
// Input is validated non-nil; the original matrix is never mutated.
// Fast-path copies *Dense data via flat indexing; fallback uses At/Set.
//
// Implementation:
//   - Stage 1: ValidateNotNil(m). Allocate Dense(cols, rows).
//   - Stage 2: If m is *Dense, use contiguous slice mapping; else generic i→j loop.
//
// Behavior highlights:
//   - Deterministic copy order (dense: row blocks; generic: i→j).
//   - One allocation for the result; no temporaries proportional to size.
//
// Inputs:
//   - m: non-nil matrix (r×c).
//
// Returns:
//   - Matrix: newly allocated Dense(c×r) with mᵀ.
//   - error : validation/allocation failures wrapped with opTranspose.
//
// Errors:
//   - ErrNilMatrix      (from ValidateNotNil).
//   - Allocation errors (from NewDense).
//
// Determinism:
//   - Fixed traversal orders independent of data values.
//
// Complexity:
//   - Time O(r*c), Space O(r*c) for the returned matrix.
//
// Notes:
//   - For square *Dense matrices, complexity is unchanged; flat indexing still wins cache-wise.
//   - Transpose is a full materialization; if a lazy/view is needed, add a separate type.
//
// AI-Hints:
//   - Keep operands as *Dense to unlock the flat-copy fast-path.
//   - If you only need Aᵀ*x, prefer MatVec on A with indices swapped instead of forming Aᵀ.
//   - Avoid transposing repeatedly in tight loops; hoist and reuse the result where possible.
func Transpose(m Matrix) (Matrix, error) {
	// Validate input non-nil
	if err := ValidateNotNil(m); err != nil {
		return nil, matrixErrorf(opTranspose, err)
	}

	// Allocate result Dense with flipped dimensions
	rows, cols := m.Rows(), m.Cols()
	res, err := NewDense(cols, rows) // dims flipped
	if err != nil {
		return nil, matrixErrorf(opTranspose, err)
	}

	// Fast-path for Dense → Dense
	var i, j int // loop iterators
	if dm, ok := m.(*Dense); ok {
		// data[i*cols + j] → res.data[j*rows + i]
		var baseSrc int
		for i = 0; i < rows; i++ {
			baseSrc = i * cols
			for j = 0; j < cols; j++ {
				res.data[j*rows+i] = dm.data[baseSrc+j]
			}
		}
		return res, nil
	}

	// Fallback: generic interface loop
	var v float64
	for i = 0; i < rows; i++ {
		for j = 0; j < cols; j++ {
			v, err = m.At(i, j)
			if err != nil {
				return nil, matrixErrorf(opTranspose, fmt.Errorf("At(%d,%d): %w", i, j, err))
			}
			if err = res.Set(j, i, v); err != nil {
				return nil, matrixErrorf(opTranspose, fmt.Errorf("Set(%d,%d): %w", j, i, err))
			}
		}
	}

	// Return result
	return res, nil
}

// Scale returns a new matrix whose elements are alpha * m[i,j].
// Input is validated non-nil; the original matrix is never mutated.
// Fast-path multiplies a *Dense backing slice in a single flat loop.
//
// Implementation:
//   - Stage 1: ValidateNotNil(m). Allocate Dense(rows, cols).
//   - Stage 2: If *Dense, flat multiply; else generic i→j At/Set scaling.
//
// Behavior highlights:
//   - Deterministic traversal order (flat or i→j).
//   - Exactly one allocation for the result, no extra buffers.
//
// Inputs:
//   - m     : non-nil matrix (r×c).
//   - alpha : scalar multiplier (any finite float64; NaN/Inf propagate).
//
// Returns:
//   - Matrix: Dense with elements alpha*m[i,j].
//   - error : validation/allocation failures wrapped with opScale.
//
// Errors:
//   - ErrNilMatrix      (from ValidateNotNil).
//   - Allocation errors (from NewDense).
//
// Determinism:
//   - Fixed loop orders independent of values.
//
// Complexity:
//   - Time O(r*c), Space O(r*c).
//
// Notes:
//   - This is an eager materialization; for pipelines, consider fusing scaling into
//     the next kernel (e.g., scale inputs right before Mul) to reduce allocations.
//   - alpha = 0 yields an explicit zero matrix with the same shape.
//
// AI-Hints:
//   - Use *Dense to hit the flat-slice path; keep data contiguous.
//   - Prefer composing `Scale(M, a)` then `Add/ Mul` only if reuse justifies the copy;
//     otherwise fold `alpha` into the consumer kernel to save work.
func Scale(m Matrix, alpha float64) (Matrix, error) {
	// Validate input non-nil
	if err := ValidateNotNil(m); err != nil {
		return nil, matrixErrorf(opScale, err)
	}

	// Allocate result Dense
	rows, cols := m.Rows(), m.Cols()
	res, err := NewDense(rows, cols)
	if err != nil {
		return nil, matrixErrorf(opScale, err)
	}

	// Fast-path for Dense → Dense
	if dm, ok := m.(*Dense); ok {
		n := rows * cols
		for idx := 0; idx < n; idx++ {
			res.data[idx] = dm.data[idx] * alpha
		}
		return res, nil
	}

	// Fallback: generic interface loop
	var i, j int
	var v float64
	for i = 0; i < rows; i++ {
		for j = 0; j < cols; j++ {
			v, err = m.At(i, j)
			if err != nil {
				return nil, matrixErrorf(opScale, fmt.Errorf("At(%d,%d): %w", i, j, err))
			}
			if err = res.Set(i, j, v*alpha); err != nil {
				return nil, matrixErrorf(opScale, fmt.Errorf("Set(%d,%d): %w", i, j, err))
			}
		}
	}

	// Return result
	return res, nil
}

// Hadamard computes the elementwise product (a ⊙ b) with a fresh Dense result.
// Both inputs must be non-nil and have identical shapes; operands are not mutated.
// Uses a single flat loop for *Dense×*Dense and a fixed-order generic fallback.
//
// Implementation:
//   - Stage 1: ValidateBinarySameShape(a, b). Allocate Dense(rows, cols).
//   - Stage 2: Fast-path if both *Dense (flat 0..n-1). Else At/Set with i→j loops.
//
// Behavior highlights:
//   - Bandwidth-bound kernel; contiguous data and flat traversal maximize throughput.
//   - Deterministic loop orders; no data-dependent branches in the hot path.
//
// Inputs:
//   - a, b: conformable matrices (same r×c).
//
// Returns:
//   - Matrix: Dense with a[i,j]*b[i,j].
//   - error : validation/allocation failures wrapped with opHadamard.
//
// Errors:
//   - ErrNilMatrix          (from ValidateBinarySameShape when a or b is nil).
//   - ErrDimensionMismatch  (from ValidateBinarySameShape when shapes differ).
//   - Allocation errors     (from NewDense).
//
// Determinism:
//   - Flat 0..(r*c−1) in fast-path; i→j in fallback; results stable across runs.
//
// Complexity:
//   - Time O(r*c), Space O(r*c).
//
// Notes:
//   - Hadamard ≠ matrix multiplication; it is elementwise. Use Mul for A×B.
//   - Keep shapes small but contiguous to stay cache-friendly.
//
// AI-Hints:
//   - Favor *Dense inputs to avoid interface dispatch and enable tight loops.
//   - If chaining multiple elementwise ops, consider fusing into one pass to reduce memory traffic.
func Hadamard(a, b Matrix) (Matrix, error) {
	// Validate both operands are non-nil and have identical shapes.
	if err := ValidateBinarySameShape(a, b); err != nil {
		return nil, matrixErrorf(opHadamard, err)
	}

	// Allocate the result Dense with the same shape.
	rows, cols := a.Rows(), a.Cols()
	res, err := NewDense(rows, cols)
	if err != nil {
		return nil, matrixErrorf(opHadamard, err)
	}

	// Fast-path: both operands are *Dense → operate on flat slices directly.
	if da, okA := a.(*Dense); okA {
		if db, okB := b.(*Dense); okB {
			var n, idx int                // predeclare loop variables to avoid per-iteration allocations
			n = rows * cols               // total number of elements
			for idx = 0; idx < n; idx++ { // fixed order ensures deterministic accumulation
				res.data[idx] = da.data[idx] * db.data[idx] // element-wise product
			}

			return res, nil // return immediately on fast-path
		}
	}

	// Fallback: generic interface loop using At/Set (bounds-safe, shape already validated).
	var i, j int // loop indices (predeclared)
	var av, bv float64
	for i = 0; i < rows; i++ { // fixed i-outer loop
		for j = 0; j < cols; j++ { // fixed j-inner loop
			av, err = a.At(i, j)
			if err != nil {
				return nil, matrixErrorf(opHadamard, fmt.Errorf("At(%d,%d): %w", i, j, err))
			}
			bv, err = b.At(i, j)
			if err != nil {
				return nil, matrixErrorf(opHadamard, fmt.Errorf("At(%d,%d): %w", i, j, err))
			}
			if err = res.Set(i, j, av*bv); err != nil {
				return nil, matrixErrorf(opHadamard, fmt.Errorf("Set(%d,%d): %w", i, j, err))
			}
		}
	}

	// Return the computed result (Dense implements Matrix).
	return res, nil
}

// MatVec computes y = m * x for a column vector x.
//
// Contract: m non-nil; x non-nil; len(x) == m.Cols().
// Fast-path: *Dense performs one pass per row with flat indexing.
// Determinism: fixed i→j loop order.
// Complexity: Time O(r*c), Space O(r) for y.
//
// AI-Hints:
//   - Use *Dense to keep a single pass per row with flat indexing.
//   - Skipping zero x[j] helps when x is sparse-ish.
func MatVec(m Matrix, x []float64) ([]float64, error) {
	// Validate m is not nil.
	if err := ValidateNotNil(m); err != nil {
		return nil, matrixErrorf(opMatVec, err)
	}
	// Validate x is not nil and match with number of columns
	if err := ValidateVecLen(x, m.Cols()); err != nil {
		return nil, matrixErrorf(opMatVec, err)
	}
	// Prepare result vector y with length rows.
	rows, cols := m.Rows(), m.Cols()
	y := make([]float64, rows) // allocate exactly rows outputs

	// Fast-path: *Dense allows flat, row-major dot-products.
	if d, ok := m.(*Dense); ok {
		var i, j, base int // indices and row base offset
		var acc, xv float64
		for i = 0; i < d.r; i++ { // iterate rows deterministically
			acc = ZeroSum             // reset accumulator per row
			base = i * d.c            // compute flat base offset for row i
			for j = 0; j < d.c; j++ { // iterate columns
				xv = x[j]    // read x(j) once per iteration
				if xv != 0 { // micro-optimization: skip zero multiplications
					acc += d.data[base+j] * xv // accumulate a(i,j)*x(j)
				}
			}
			y[i] = acc // store y(i)
		}

		return y, nil // return on fast-path
	}

	// Fallback: interface-based dot-products via At.
	var i, j int   // loop indices
	var mv float64 // temporary to hold m(i,j)
	var err error
	for i = 0; i < rows; i++ { // iterate rows
		y[i] = ZeroSum             // initialize y(i) to zero
		for j = 0; j < cols; j++ { // iterate columns
			mv, err = m.At(i, j) // read m(i,j)
			if err != nil {
				return nil, matrixErrorf(opMatVec, fmt.Errorf("At(%d,%d): %w", i, j, err))
			}
			y[i] += mv * x[j] // accumulate
		}
	}

	return y, nil // return computed vector
}

// Eigen computes eigenvalues and eigenvectors of a symmetric matrix via Jacobi sweeps.
// Implementation:
//   - Stage 1: Validate symmetric square input within tol (not nil, square, |A[i,j]-A[j,i]| ≤ tol).
//   - Stage 2: Repeatedly pick (p,q) with the largest |A[p,q]| in i→j order and apply a Jacobi rotation.
//
// Behavior highlights:
//   - Stable, deterministic pivot scan; fast path for *Dense updates.
//
// Inputs:
//   - m: symmetric Matrix (within tol); n := m.Rows().
//   - tol: convergence threshold (typ. 1e-9..1e-12 for float64).
//   - maxIter: safety cap on iterations.
//
// Returns:
//   - []float64: eigenvalues (diagonal of the rotated matrix).
//   - Matrix: Q whose columns are eigenvectors.
//
// Errors:
//   - ErrNonSquare (non-square), ErrAsymmetry (not symmetric within tol),
//     ErrMatrixEigenFailed (max off-diagonal ≥ tol after maxIter).
//
// Determinism:
//   - Fixed i→j pivot search and fixed update order produce stable results.
//
// Complexity:
//   - Time O(maxIter * n^3), Space O(n^2).
//
// Notes:
//   - If |A[p,q]| ≤ tol, the rotation is skipped via (c=1,s=0) to avoid numerical blow-ups.
//
// AI-Hints:
//   - Good defaults: tol≈1e-10, maxIter≈100..300 for n≤128;
//   - Precondition by symmetrizing if input comes from numerically noisy ops.
//   - High cyclomatic complexity is intentional to avoid splitting the core mathematical
//     logic (state machine and convergence loop) into fragmented helper functions,
//     which improves both performance (by reducing call overhead) and readability.
//
// nolint:gocyclo
func Eigen(m Matrix, tol float64, maxIter int) ([]float64, Matrix, error) {
	// Validate: notNil; Square; Symmetric;
	if err := ValidateSymmetric(m, tol); err != nil {
		return nil, nil, matrixErrorf(opEigen, err) // unify error wrapping
	}
	// Prepare working copy A and orthogonal accumulator Q
	n := m.Rows()               // n - number of rows (and columns), cols - number of columns
	aRaw := m.Clone()           // aRaw is a working copy of m to avoid modifying the original
	qRaw, err := NewDense(n, n) // qRaw is a newly allocated zero dense matrix
	var i, j int                // loop iterators over rows and columns
	if err != nil {
		return nil, nil, matrixErrorf(opEigen, err)
	}
	// Initialize Q as identity: Q[i,i] = 1
	for i = 0; i < n; i++ {
		qRaw.data[i*n+i] = 1.0 // _ = qRaw.Set(i, i, 1.0)
	}

	// Detect if we can use fast-path on *Dense
	// if aRaw is actually *Dense, then useFast=true
	Adense, useFast := aRaw.(*Dense)

	// Jacobi rotations
	var (
		iter               int     // iteration counter
		base               int     // helper offset into the flat data slice
		p, q               int     // current pivot indices
		maxOff, off        float64 // maxOff - current max |A[p,q]|; off - temporary
		app, aqq           float64 // diagonal entries A[p,p], A[q,q]
		aip, aiq, qip, qiq float64 // temporaries for A[i,p], A[i,q] and Q[i,p], Q[i,q]
		newIP, newIQ       float64 // updated values for A[i,p] and A[i,q]
		apq                float64 // off-diagonal entry A[p,q]
		theta, t           float64 // intermediate rotation parameters
		c, s               float64 // cosine and sine of the rotation angle
	)
	for iter = 0; iter < maxIter; iter++ {
		// J.1: Find pivot (p,q) maximizing |A[p,q]|
		maxOff = NormZero
		if useFast {
			// fast-path: operate directly on data []float64
			for i = 0; i < n; i++ {
				base = i * n
				for j = i + 1; j < n; j++ {
					// off = |A[i,j]|
					off = math.Abs(Adense.data[base+j])
					if off > maxOff {
						maxOff, p, q = off, i, j
					}
				}
			}
		} else {
			// fallback: interface-based path via At
			for i = 0; i < n; i++ {
				for j = i + 1; j < n; j++ {

					off, err = aRaw.At(i, j)
					if err != nil {
						return nil, nil, matrixErrorf(opEigen, fmt.Errorf("At(%d,%d): %w", i, j, err))
					}
					off = math.Abs(off)
					if off > maxOff {
						maxOff, p, q = off, i, j
					}
				}
			}
		}

		// J.2: Check convergence: if maxOff < tol, break
		if maxOff < tol {
			break
		}

		// J.3: Compute rotation parameters from A[p,p], A[q,q], A[p,q]
		if useFast {
			app = Adense.data[p*n+p]
			aqq = Adense.data[q*n+q]
			apq = Adense.data[p*n+q]
		} else {
			app, err = aRaw.At(p, p)
			if err != nil {
				return nil, nil, matrixErrorf(opEigen, fmt.Errorf("At(%d,%d): %w", p, p, err))
			}
			aqq, err = aRaw.At(q, q)
			if err != nil {
				return nil, nil, matrixErrorf(opEigen, fmt.Errorf("At(%d,%d): %w", q, q, err))
			}
			apq, err = aRaw.At(p, q)
			if err != nil {
				return nil, nil, matrixErrorf(opEigen, fmt.Errorf("At(%d,%d): %w", p, q, err))
			}
		}
		// Guard: avoid division by ~zero off-diagonal
		if math.Abs(apq) <= tol {
			// No-op rotation (c=1,s=0) keeps determinism and prevents blow-ups.
			// Continue to next sweep; the pivot search will progress.
			continue
		}
		// θ = (aqq−app)/(2*apq)
		theta = (aqq - app) / (2 * apq)
		// t = sign(θ) / (|θ|+√(θ²+1))
		// t = math.Copysign(1.0/(math.Abs(theta)+math.Sqrt(theta*theta+1)), theta)
		t = math.Copysign(1.0/(math.Abs(theta)+math.Hypot(theta, 1)), theta)

		// c = 1/√(1+t²), s = t*c
		c = 1.0 / math.Sqrt(t*t+1)
		s = t * c

		// J.4: Apply rotation to A
		if useFast {
			// fast-path: update two pairs of elements in data at once
			for i = 0; i < n; i++ {
				if i == p || i == q {
					continue
				}
				// original A[i,p], A[i,q]
				aip = Adense.data[i*n+p]
				aiq = Adense.data[i*n+q]
				// new values
				newIP = c*aip - s*aiq
				newIQ = s*aip + c*aiq
				// assign symmetrically to [i,p] and [p,i], [i,q] and [q,i]
				Adense.data[i*n+p], Adense.data[p*n+i] = newIP, newIP
				Adense.data[i*n+q], Adense.data[q*n+i] = newIQ, newIQ
			}
			// update diagonals and zero out A[p,q], A[q,p]
			Adense.data[p*n+p] = c*c*app - 2*c*s*apq + s*s*aqq
			Adense.data[q*n+q] = s*s*app + 2*c*s*apq + c*c*aqq
			Adense.data[p*n+q], Adense.data[q*n+p] = 0, 0
		} else {
			// fallback via At/Set
			for i = 0; i < n; i++ {
				if i == p || i == q {
					continue
				}
				aip, err = aRaw.At(i, p)
				if err != nil {
					return nil, nil, matrixErrorf(opEigen, fmt.Errorf("At(%d,%d): %w", i, p, err))
				}
				aiq, err = aRaw.At(i, q)
				if err != nil {
					return nil, nil, matrixErrorf(opEigen, fmt.Errorf("At(%d,%d): %w", i, q, err))
				}
				newIP = c*aip - s*aiq
				newIQ = s*aip + c*aiq
				if err = aRaw.Set(i, p, newIP); err != nil {
					return nil, nil, matrixErrorf(opEigen, fmt.Errorf("Set(%d,%d): %w", i, p, err))
				}
				if err = aRaw.Set(p, i, newIP); err != nil {
					return nil, nil, matrixErrorf(opEigen, fmt.Errorf("Set(%d,%d): %w", p, i, err))
				}
				if err = aRaw.Set(i, q, newIQ); err != nil {
					return nil, nil, matrixErrorf(opEigen, fmt.Errorf("Set(%d,%d): %w", i, q, err))
				}
				if err = aRaw.Set(q, i, newIQ); err != nil {
					return nil, nil, matrixErrorf(opEigen, fmt.Errorf("Set(%d,%d): %w", q, i, err))
				}
			}
			if err = aRaw.Set(p, p, c*c*app-2*c*s*apq+s*s*aqq); err != nil {
				return nil, nil, matrixErrorf(opEigen, fmt.Errorf("Set(%d,%d): %w", p, p, err))
			}
			if err = aRaw.Set(q, q, s*s*app+2*c*s*apq+c*c*aqq); err != nil {
				return nil, nil, matrixErrorf(opEigen, fmt.Errorf("Set(%d,%d): %w", q, q, err))
			}
			if err = aRaw.Set(p, q, 0.0); err != nil {
				return nil, nil, matrixErrorf(opEigen, fmt.Errorf("Set(%d,%d): %w", p, q, err))
			}
			if err = aRaw.Set(q, p, 0.0); err != nil {
				return nil, nil, matrixErrorf(opEigen, fmt.Errorf("Set(%d,%d): %w", q, p, err))
			}
		}

		// J.5: Accumulate rotation into Q
		// here qRaw is also expected to be *Dense, but this works anyway
		for i = 0; i < n; i++ {
			qip = qRaw.data[i*n+p] // Q[i,p]
			qiq = qRaw.data[i*n+q] // Q[i,q]
			qRaw.data[i*n+p] = c*qip - s*qiq
			qRaw.data[i*n+q] = s*qip + c*qiq
		}
	}

	// Final convergence check: recompute max off-diagonal using the fastest path available.
	maxOff = NormZero
	if useFast {
		for i = 0; i < n; i++ {
			base = i * n
			for j = i + 1; j < n; j++ {
				off = math.Abs(Adense.data[base+j])
				if off > maxOff {
					maxOff = off
				}
			}
		}
	} else {
		for i = 0; i < n; i++ {
			for j = i + 1; j < n; j++ {
				off, err = aRaw.At(i, j)
				if err != nil {
					return nil, nil, matrixErrorf(opEigen, fmt.Errorf("At(%d,%d): %w", i, j, err))
				}
				off = math.Abs(off)
				if off > maxOff {
					maxOff = off
				}
			}
		}
	}
	if maxOff >= tol {
		return nil, nil, matrixErrorf(opEigen, ErrMatrixEigenFailed)
	}

	// Extract eigenvalues from diagonal of A
	eigs := make([]float64, n)
	if useFast {
		for i = 0; i < n; i++ {
			eigs[i] = Adense.data[i*n+i]
		}
	} else {
		var v float64
		for i = 0; i < n; i++ {
			v, err = aRaw.At(i, i)
			if err != nil {
				return nil, nil, matrixErrorf(opEigen, fmt.Errorf("At(%d,%d): %w", i, i, err))
			}
			eigs[i] = v
		}
	}

	// Return eigenvalues and eigenvectors
	return eigs, qRaw, nil
}

// Inverse computes A^{-1} using Doolittle LU factorization without pivoting (deterministic).
// The input must be non-nil and square. Returns ErrSingular if a zero pivot is detected.
// Produces new Dense matrices; does not mutate the input.
//
// Implementation:
//   - Stage 1: ValidateNotNil(m) and ValidateSquare(m). Factorize via LU(m) → L (unit lower), U (upper).
//     Allocate invDense(n×n) and workspace vectors y, x of length n.
//   - Stage 2: For each canonical basis column e_col:
//   - Forward solve L*y = e_col (top-down).
//   - Backward solve U*x = y    (bottom-up; check nonzero pivots).
//   - Write x into column `col` of invDense.
//     Dense fast-path uses flat indexing; generic fallback uses At/Set.
//
// Behavior highlights:
//   - Fully deterministic loop orders (col↑, forward i↑, backward i↓).
//   - No pivoting by design (stable determinism and reproducibility).
//   - Input m is read-only; factors L and U are freshly allocated by LU.
//
// Inputs:
//   - m: non-nil square matrix (n×n).
//
// Returns:
//   - Matrix: Dense(n×n) containing A^{-1}.
//   - error : validation/factorization/solve failures wrapped with opInverse.
//
// Errors:
//   - ErrNilMatrix         (ValidateNotNil).
//   - ErrNonSquare         (ValidateSquare).
//   - ErrSingular          (detected during backward substitution when U[i,i] == 0).
//   - Propagated LU errors (from LU validation/allocation).
//   - Allocation errors    (from NewDense).
//
// Determinism:
//   - Fixed traversal and no pivoting → identical results for identical inputs.
//
// Complexity:
//   - Time O(n^3): Doolittle LU is O(n^3); solving n RHS via triangular solves is O(n^3).
//   - Space O(n^2): L, U, and invDense are O(n^2); y, x are O(n).
//
// Notes:
//   - Numerical stability: no partial/complete pivoting. Upstream callers should avoid
//     ill-conditioned matrices or apply scaling/preconditioning if stability matters.
//   - For SPD matrices, prefer Cholesky-based inversion outside this package.
//   - If you only need A^{-1}*b, solve via LU once and apply triangular solves (cheaper than forming A^{-1}).
//
// AI-Hints:
//   - Reuse LU(m) if multiple solves are needed; forming A^{-1} is typically a last resort.
//   - Avoid near-singular inputs (tiny U[i,i]); detect upstream and skip inversion when possible.
//   - Keep inputs as *Dense to hit the fast-path inside LU and the triangular solves.
func Inverse(m Matrix) (Matrix, error) {
	// Validate input non‐nil and square
	if err := ValidateNotNil(m); err != nil {
		return nil, matrixErrorf(opInverse, err)
	}
	if err := ValidateSquare(m); err != nil {
		return nil, matrixErrorf(opInverse, err)
	}

	// LU decomposition (Doolittle)
	Lmat, Umat, err := LU(m)
	if err != nil {
		return nil, matrixErrorf(opInverse, err)
	}

	// Prepare result container and scratch arrays
	n := m.Rows()
	invDense, err := NewDense(n, n)
	if err != nil {
		return nil, matrixErrorf(opInverse, err)
	}

	var (
		col, i, k  int // loop iterators
		sum, pivot float64
		y          = make([]float64, n) // forward substitution workspace
		x          = make([]float64, n) // backward substitution workspace
	)
	// Fast‐path: detect *Dense for L, U, and inv
	Ld, okL := Lmat.(*Dense)
	Ud, okU := Umat.(*Dense)
	if okL && okU {
		// row‐major stride
		var baseUI, baseLI int
		for col = 0; col < n; col++ {
			// 4.1 Forward substitution: L*y = e_col
			for i = 0; i < n; i++ {
				sum = ZeroSum
				baseLI = i * n
				for k = 0; k < i; k++ {
					sum += Ld.data[baseLI+k] * y[k]
				}
				if i == col {
					y[i] = 1.0 - sum
				} else {
					y[i] = -sum
				}
			}
			// 4.2 Backward substitution: U*x = y
			for i = n - 1; i >= 0; i-- {
				sum = ZeroSum
				baseUI = i * n
				for k = i + 1; k < n; k++ {
					sum += Ud.data[baseUI+k] * x[k]
				}
				pivot = Ud.data[baseUI+i]
				if pivot == ZeroPivot {
					return nil, matrixErrorf(opInverse, ErrSingular)
				}
				x[i] = (y[i] - sum) / pivot
			}
			// 4.3 Write x into column col of inv
			for i = 0; i < n; i++ {
				invDense.data[i*n+col] = x[i]
			}
		}

		return invDense, nil
	}

	// Fallback: generic interface version
	var v float64
	for col = 0; col < n; col++ {
		// Forward substitution: L*y = e_col
		for i = 0; i < n; i++ {
			sum = ZeroSum
			for k = 0; k < i; k++ {
				v, err = Lmat.At(i, k)
				if err != nil {
					return nil, matrixErrorf(opInverse, fmt.Errorf("At(%d,%d): %w", i, k, err))
				}
				sum += v * y[k]
			}
			if i == col {
				y[i] = 1.0 - sum
			} else {
				y[i] = -sum
			}
		}
		// Backward substitution: U*x = y
		for i = n - 1; i >= 0; i-- {
			sum = ZeroSum
			for k = i + 1; k < n; k++ {
				v, err = Umat.At(i, k)
				if err != nil {
					return nil, matrixErrorf(opInverse, fmt.Errorf("At(%d,%d): %w", i, k, err))
				}
				sum += v * x[k]
			}
			pivot, err = Umat.At(i, i)
			if err != nil {
				return nil, matrixErrorf(opInverse, fmt.Errorf("At(%d,%d): %w", i, i, err))
			}
			if pivot == ZeroPivot {
				return nil, matrixErrorf(opInverse, ErrSingular)
			}
			x[i] = (y[i] - sum) / pivot
		}
		// Write x into column col of inv
		for i = 0; i < n; i++ {
			if err = invDense.Set(i, col, x[i]); err != nil {
				return nil, matrixErrorf(opInverse, fmt.Errorf("Set(%d,%d): %w", i, col, err))
			}
		}
	}

	return invDense, nil
}
