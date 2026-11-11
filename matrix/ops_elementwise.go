// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix owns deterministic elementwise operations used by its
// public algebra and statistics facades.
// This file provides row-transform kernels and finite-data sanitation and
// comparison operations without exposing private storage contracts.
//
// Purpose:
//   - Provide the private row-wise centering/normalization kernels still needed
//     by CenterRows and NormalizeRowsL1, plus the public sanitization and comparison
//     kernels used by Clip, ReplaceInfNaN and AllClose.
//   - Column-centering is owned by the stable statistics implementation; no
//     redundant elementwise column-offset or scaling kernel is retained.
//   - Keep all loops deterministic and cache-friendly with Dense fast-paths.
//
// Design:
//   - All ew* are UNEXPORTED by design (internal micro-kernels).
//   - Public facades in api.go delegate to these kernels when applicable.
//
// Determinism & Performance:
//   - Fixed loop orders (i→j or flat 0..n-1).
//   - Dense fast-path operates on a single flat buffer (row-major).
//   - No hidden allocations beyond the output Dense; O(r*c) time and space.
//
// Zero-shape law:
//   - Element-wise kernels accept 0×0, 0×N, and N×0 matrices.
//   - Output shape always matches input shape.
//   - Empty element domains are deterministic no-ops.
//   - Broadcast vector lengths are still validated against the corresponding dimension.
//
// AI-Hints:
//   - Prefer passing *Dense to unlock the flat-slice fast path.
//   - Keep rowMeans/scale vectors private and aligned to the input row count.
//   - Do not resurrect an independent column-centering kernel; stable column
//     statistics already owns that arithmetic and overflow policy.
package matrix

import "math"

// Operation name constants for unified error wrapping and reducing magic strings.
const (
	opBroadcastSubRows = "broadcastSubRows"
	opScaleRows        = "scaleRows"
	opReplaceInfNaN    = "ReplaceInfNaN"
	opClip             = "Clip"
	opAllClose         = "AllClose"
)

// allCloseElem checks element-wise |a-b| ≤ atol + rtol*|b|, with robust NaN/Inf semantics.
// Implementation:
//   - Stage 1: reject NaN in either operand → not close.
//   - Stage 2: handle ±Inf: only equal-sign infinities are close.
//   - Stage 3: apply finite all-close inequality.
//
// Behavior highlights:
//   - Fixes the subtle IEEE issue: (+Inf - +Inf) yields NaN, which must NOT be treated as “close by accident”.
//
// Complexity:
//   - Time O(1), Space O(1).
func allCloseElem(a, b, rtol, atol float64) bool {
	// NaN is never “close” under our strict default semantics.
	if math.IsNaN(a) || math.IsNaN(b) {
		return false
	}

	// Infinities: only same-sign infinities are close.
	if math.IsInf(a, 0) || math.IsInf(b, 0) {
		if math.IsInf(a, 1) && math.IsInf(b, 1) {
			return true
		}
		if math.IsInf(a, -1) && math.IsInf(b, -1) {
			return true
		}
		return false
	}

	// Finite inequality: |a-b| ≤ atol + rtol*|b|.
	diff := a - b
	if diff < 0 {
		diff = -diff
	}
	absb := b
	if absb < 0 {
		absb = -absb
	}
	return diff <= (atol + rtol*absb)
}

// ewBroadcastSubRows computes out[i,j] = X[i,j] - rowMeans[i].
// Implementation:
//   - Stage 1: Validate X (non-nil) and length(rowMeans)==Rows(X).
//   - Stage 2: Dense fast-path over flat buffer; fallback via At/Set.
//
// Behavior highlights:
//   - Deterministic i→j loops; single allocation for output.
//
// Inputs:
//   - X: input matrix.
//   - rowMeans: length==Rows(X); rows offsets.
//
// Returns:
//   - Matrix: newly allocated Dense with centered values.
//
// Errors:
//   - ErrNilMatrix (nil X), ErrDimensionMismatch (len mismatch), Dense alloc/Set errors.
//
// Determinism:
//   - Fixed row-major traversal.
//
// Complexity:
//   - Time O(rc), Space O(rc).
//
// Notes:
//   - Precompute rowMeans once; reuse across calls.
//
// AI-Hints:
//   - Use for rows-centering and z-scoring.
//   - Pass *Dense to unlock flat fast-path.
func ewBroadcastSubRows(X Matrix, rowMeans []float64) (Matrix, error) {
	// Validate matrix presence.
	if err := ValidateNotNil(X); err != nil {
		return nil, matrixErrorf(opBroadcastSubRows, err)
	}
	// Read shape once.
	r, c := X.Rows(), X.Cols()
	// Check broadcast vector length.
	if len(rowMeans) != r {
		return nil, matrixErrorf(opBroadcastSubRows, ErrDimensionMismatch)
	}
	// Zero-shape law:
	//   - 0×N requires len(rowMeans)==0 and returns a new 0×N Dense.
	//   - N×0 requires len(rowMeans)==N and returns a new N×0 Dense.
	//   - No element reads occur for either case.

	// Allocate result dense.
	out, err := NewDense(r, c)
	if err != nil {
		return nil, matrixErrorf(opBroadcastSubRows, err)
	}

	// Dense fast-path.
	var i, j int
	var rowMean float64 // cache row mean once per row
	if d, ok := X.(*Dense); ok {
		for i = 0; i < r; i++ {
			rowMean = rowMeans[i]
			for j = 0; j < c; j++ { // (i * c) = base offset for row i
				out.data[i*c+j] = d.data[i*c+j] - rowMean
			}
		}

		return out, nil
	}

	// Generic fallback.
	var v float64
	for i = 0; i < r; i++ {
		rowMean = rowMeans[i] // read once per row
		for j = 0; j < c; j++ {
			v, err = X.At(i, j)
			if err != nil {
				return nil, matrixErrorf(opBroadcastSubRows, err)
			}
			// Set() may fail (bounds, policy). Never ignore the error.
			if err = out.Set(i, j, v-rowMean); err != nil {
				return nil, matrixErrorf(opBroadcastSubRows, err)
			}
		}
	}

	return out, nil
}

// ewScaleRows computes out[i,j] = X[i,j] * scale[i].
// Time: O(r*c). Space: O(r*c). Deterministic i→j loops.
//
// AI-Hint: use for L1/L2 row-normalization. O(r*c).
func ewScaleRows(X Matrix, scale []float64) (Matrix, error) {
	// Validate matrix presence.
	if err := ValidateNotNil(X); err != nil {
		return nil, matrixErrorf(opScaleRows, err)
	}
	// Read shape once.
	r, c := X.Rows(), X.Cols()
	// Validate scale length.
	if len(scale) != r {
		return nil, matrixErrorf(opScaleRows, ErrDimensionMismatch)
	}
	// Zero-shape law:
	//   - 0×N requires len(rowMeans)==0 and returns a new 0×N Dense.
	//   - N×0 requires len(rowMeans)==N and returns a new N×0 Dense.
	//   - No element reads occur for either case.

	// Allocate result dense.
	out, err := NewDense(r, c)
	if err != nil {
		return nil, matrixErrorf(opScaleRows, err)
	}

	// Dense fast-path.
	var i, j int
	var v, sf float64
	if d, ok := X.(*Dense); ok {
		for i = 0; i < r; i++ {
			sf = scale[i]           // scale factor for row i
			for j = 0; j < c; j++ { // (i * c) = row base offset
				out.data[i*c+j] = d.data[i*c+j] * sf
			}
		}

		return out, nil
	}

	// Generic fallback.
	for i = 0; i < r; i++ {
		sf = scale[i] // row scale once per row
		for j = 0; j < c; j++ {
			v, err = X.At(i, j)
			if err != nil {
				return nil, matrixErrorf(opScaleRows, err)
			}
			// Set() may fail (bounds, policy). Never ignore the error.
			if err = out.Set(i, j, v*sf); err != nil {
				return nil, matrixErrorf(opScaleRows, err)
			}
		}
	}

	return out, nil
}

// ewReplaceInfNaN copies X replacing any {±Inf, NaN} by val (finite).
// Replace non-finite: out[i,j] = val if X[i,j] is {±Inf, NaN}, else X[i,j].
// Implementation:
//   - Stage 1: Validate X (non-nil) and val is finite (not NaN/Inf).
//   - Stage 2: Flat pass for Dense, otherwise At/Set.
//
// Behavior highlights:
//   - Deterministic; single output allocation.
//
// Inputs:
//   - X: input matrix.
//   - val: finite replacement value.
//
// Returns:
//   - Matrix: newly allocated Dense.
//
// Errors:
//   - ErrNilMatrix (nil X), ErrNaNInf (non-finite val), Dense alloc/Set errors.
//
// Determinism:
//   - Fixed traversal.
//
// Complexity:
//   - Time O(rc), Space O(r*c).
//
// Notes:
//   - Useful as sanitizer pre-stats.
//
// AI-Hints:
//   - Keep val small and domain-appropriate (e.g., 0 or column mean).
func ewReplaceInfNaN(X Matrix, val float64) (Matrix, error) {
	// Validate input matrix.
	if err := ValidateNotNil(X); err != nil {
		return nil, matrixErrorf(opReplaceInfNaN, err)
	}
	// Validate 'val' is finite per numeric policy.
	if isNonFinite(val) {
		return nil, matrixErrorf(opReplaceInfNaN, ErrNaNInf)
	}
	// Read shape and allocate result.
	r, c := X.Rows(), X.Cols()
	out, err := NewDense(r, c)
	if err != nil {
		return nil, matrixErrorf(opReplaceInfNaN, err)
	}

	// Dense fast-path: direct flat slice iteration.
	var v float64
	if d, ok := X.(*Dense); ok {
		n := r * c
		for idx := 0; idx < n; idx++ {
			v = d.data[idx] // read element
			// Replace if not finite.
			if isNonFinite(v) {
				v = val
			}
			out.data[idx] = v // write element
		}

		return out, nil
	}

	// Generic fallback.
	var i, j int
	for i = 0; i < r; i++ {
		for j = 0; j < c; j++ {
			v, err = X.At(i, j)
			if err != nil {
				return nil, matrixErrorf(opReplaceInfNaN, err)
			}
			if isNonFinite(v) {
				v = val
			}
			// Set() may fail (bounds, policy). Never ignore the error.
			if err = out.Set(i, j, v); err != nil {
				return nil, matrixErrorf(opReplaceInfNaN, err)
			}
		}
	}

	return out, nil
}

// Clamp into [lo, hi] (bounds finite). If lo > hi, bounds are swapped.
// Implementation:
//   - Stage 1: Validate X (non-nil); check bounds are finite else ErrNaNInf; normalize bound order.
//   - Stage 2: Flat pass for Dense; fallback via At/Set.
//
// Behavior highlights:
//   - Deterministic; predictable branches.
//
// Inputs:
//   - X: input matrix.
//   - lo, hi: finite bounds; order does not matter (auto-swap).
//
// Returns:
//   - Matrix: newly allocated Dense.
//
// Errors:
//   - ErrNilMatrix, ErrNaNInf (invalid bounds), Dense alloc/Set errors.
//
// Determinism:
//   - Fixed traversal.
//
// Complexity:
//   - Time O(rc), Space O(r*c).
//
// Notes:
//   - Bounds must be finite; if lo > hi, they are swapped (normalized).
//   - Swap prevents surprising failures on inverted bounds.
//
// AI-Hints:
//   - Combine with ReplaceInfNaN for robust pipelines.
func ewClipRange(X Matrix, lo, hi float64) (Matrix, error) {
	// Validate input matrix.
	if err := ValidateNotNil(X); err != nil {
		return nil, matrixErrorf(opClip, err)
	}

	// Normalize and validate bounds using canonical helper.
	var err error
	lo, hi, err = validateBounds(lo, hi)
	if err != nil {
		return nil, matrixErrorf(opClip, err)
	}

	// Read shape and allocate output.
	r, c := X.Rows(), X.Cols()
	out, err := NewDense(r, c)
	if err != nil {
		return nil, matrixErrorf(opClip, err)
	}

	// Dense fast-path: single pass with branchy clamp (predictable).
	var v float64
	if d, ok := X.(*Dense); ok {
		n := r * c
		for idx := 0; idx < n; idx++ {
			v = d.data[idx] // read
			if v < lo {
				v = lo
			} else if v > hi {
				v = hi
			} // clamp into [lo,hi]
			out.data[idx] = v // write
		}

		return out, nil
	}

	// Generic fallback via At/Set.
	var i, j int
	for i = 0; i < r; i++ {
		for j = 0; j < c; j++ {
			v, err = X.At(i, j)
			if err != nil {
				return nil, matrixErrorf(opClip, err)
			}
			if v < lo {
				v = lo
			} else if v > hi {
				v = hi
			}
			// Set() may fail (bounds, policy). Never ignore the error.
			if err = out.Set(i, j, v); err != nil {
				return nil, matrixErrorf(opClip, err)
			}
		}
	}

	return out, nil
}

// AllClose checks element-wise |a-b| ≤ atol + rtol*|b| for identical shapes.
// Element-wise closeness: |a-b| ≤ atol + rtol*|b| for identical shapes.
// Implementation:
//   - Stage 1: Validate rtol/atol finite; normalize negatives to abs.
//   - Stage 2: Validate a,b non-nil and same shape.
//   - Stage 3: Flat dual-pass for (*Dense,Dense); else At-based fallback.
//
// Behavior highlights:
//   - Early exit on first violation; deterministic order.
//
// Inputs:
//   - a,b: matrices with identical shape.
//   - rtol, atol: finite tolerances; negatives are allowed → abs-ed.
//
// Returns:
//   - (ok bool, err error): ok=true if all elements close.
//
// Errors:
//   - ErrNaNInf (invalid tolerances), ErrNilMatrix / ErrDimensionMismatch for inputs.
//
// Determinism:
//   - Fixed traversal; first violation determines false.
//
// Complexity:
//   - Time O(rc), Space O(1) extra.
//
// Notes:
//   - RHS uses |b|; keep in mind asymmetry.
//
// AI-Hints:
//   - Use small but non-zero atol for exact-equality comparisons on float data.
func ewAllClose(a, b Matrix, rtol, atol float64) (bool, error) {
	// Normalize tolerances via canonical helper (reject NaN/Inf; accept negatives as abs()).
	var err error
	rtol, err = validateTol(rtol)
	if err != nil {
		return false, matrixErrorf(opAllClose, err)
	}
	atol, err = validateTol(atol)
	if err != nil {
		return false, matrixErrorf(opAllClose, err)
	}

	// Validate presence and shape equality using central validators.
	if err = ValidateNotNil(a); err != nil {
		return false, matrixErrorf(opAllClose, err)
	}
	if err = ValidateNotNil(b); err != nil {
		return false, matrixErrorf(opAllClose, err)
	}
	if err = ValidateSameShape(a, b); err != nil {
		return false, matrixErrorf(opAllClose, err)
	}

	// Read shape once (O(1)).
	r, c := a.Rows(), a.Cols()
	// Zero-shape law:
	// Identical empty element domains are vacuously close.
	if IsZeroShape(a) {
		return true, nil
	}
	// Dense fast-path: operate over flat slices when both are *Dense.
	if da, okA := a.(*Dense); okA {
		if db, okB := b.(*Dense); okB {
			n := r * c // total number of elements
			for idx := 0; idx < n; idx++ {
				// Compute absolute difference and RHS tolerance bound.
				// Check |a-b| ≤ atol + rtol*|b|.
				if !allCloseElem(da.data[idx], db.data[idx], rtol, atol) {
					return false, nil // early-exit on first violation
				}
			}

			return true, nil // all ok
		}
	}

	// Generic fallback via At (bounds-safe; still deterministic).
	var i, j int
	var av, bv float64
	for i = 0; i < r; i++ {
		for j = 0; j < c; j++ {
			av, err = a.At(i, j) // read a(i,j)
			if err != nil {
				return false, matrixErrorf(opAllClose, err)
			}
			bv, err = b.At(i, j) // read b(i,j)
			if err != nil {
				return false, matrixErrorf(opAllClose, err)
			}
			if !allCloseElem(av, bv, rtol, atol) {
				// Compare to tolerance threshold.
				return false, nil
			}
		}
	}

	return true, nil
}
