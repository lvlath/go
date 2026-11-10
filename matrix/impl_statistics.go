// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package: matrix
//
// Purpose:
//   - Maintain legacy row-centering and L1-normalization kernels.
//   - The stable column-wise statistics and row L2 normalization live in
//     impl_statistics_stable.go, with exact binary cancellation recovery in
//     impl_statistics_exact.go; the public facades delegate there directly.
//
// Exposed API:
//   - CenterRows(X)      -> (Xc, means)         // subtract per-row mean
//   - NormalizeRowsL1(X) -> (Y, norms)          // L1 row normalization (degenerate rows unchanged)
//
// Determinism & Performance:
//   - Fixed i→j traversal for all explicit loops.
//   - Dense fast-paths avoid At/Set and operate on row-major flat buffers.
//
// Zero-shape law:
//   - 0×0, 0×N, and N×0 are valid structural matrices.
//   - CenterRows and NormalizeRowsL1 treat zero-shape inputs as no-op
//     transforms; their returned metadata preserves the expected shape.
//
// AI-Hints:
//   - Prefer passing *Dense to unlock flat-slice fast paths.
//   - Sanitize inputs first (ReplaceInfNaN) if NaN/Inf propagation is undesired in downstream statistics.

package matrix

// Operation name constants for unified error wrapping and reducing magic strings.
const (
	opCenterColumns   = "CenterColumns"
	opCenterRows      = "CenterRows"
	opNormalizeRowsL1 = "NormalizeRowsL1"
	opNormalizeRowsL2 = "NormalizeRowsL2"
	opCovariance      = "Covariance"
	opCorrelation     = "Correlation"
)

// statsDegenerateRowScale is the neutral scale factor for rows whose norm is exactly zero.
//   - A zero-norm row cannot be normalized to unit norm.
//   - The least surprising stable policy is to leave it unchanged.
//   - Using 1 avoids accidental mutation and keeps zero rows zero.
//
// Notes:
//   - This is a package policy constant, not an option.
//   - Making this configurable would split normalization semantics across callers.
//
// AI-Hints:
//   - Do not replace with 0: that would silently erase non-empty rows if a future
//     norm policy classifies a row as degenerate for tolerance reasons.
const statsDegenerateRowScale = 1.0

// centerRows subtracts the per-row mean from every element (row-wise centering).
// Implementation:
//   - Stage 1: Validate X (non-nil) and handle zero-size as a strict no-op.
//   - Stage 2: Compute row means deterministically (Dense fast-path; At fallback).
//   - Stage 3: Apply ewBroadcastSubRows to produce a centered copy.
//
// Behavior highlights:
//   - Zero-size (0×N or N×0): returns (X, zeroMeans, nil) without allocations.
//   - Deterministic i→j traversal; stable results.
//
// Inputs:
//   - X: input matrix (r×c).
//
// Returns:
//   - Matrix: centered copy (r×c) for r>0 && c>0; otherwise X itself (no-op).
//   - []float64: row means (len=r).
//
// Errors:
//   - ErrNilMatrix from validation.
//   - Wrapped At/NewDense/Set errors from fallback paths.
//
// Determinism:
//   - Fixed loop order; no randomness.
//
// Complexity:
//   - Time O(r*c), Space O(r*c) for output (+ O(r) means).
//
// Notes:
//   - Means are Σ_j X[i,j] / c for c>0.
//
// AI-Hints:
//   - Useful for per-sample baselining before distance/similarity computations.
func centerRows(X Matrix) (Matrix, []float64, error) {
	// Stage 1 (Validate): ensure X is present.
	if err := ValidateNotNil(X); err != nil {
		return nil, nil, matrixErrorf(opCenterRows, err)
	}

	// Stage 1 (Zero-size policy): centering is a no-op when there are no elements.
	r, c := X.Rows(), X.Cols()
	means := make([]float64, r) // correct length for row means
	if r == 0 || c == 0 {
		// Zero-shape law:
		//   - 0×N has no rows and therefore no row means.
		//   - N×0 has N row means, all vacuously 0.
		//   - The matrix itself is returned because there are no elements to transform.
		return X, means, nil
	}

	// Stage 2 (Execute): compute mean per row.
	var i, j int
	var s, v float64

	if d, ok := X.(*Dense); ok {
		for i = 0; i < r; i++ {
			s = 0.0
			base := i * c
			for j = 0; j < c; j++ {
				s += d.data[base+j]
			}
			means[i] = s / float64(c)
		}
	} else {
		var err error
		for i = 0; i < r; i++ {
			s = 0.0
			for j = 0; j < c; j++ {
				v, err = X.At(i, j)
				if err != nil {
					return nil, nil, matrixErrorf(opCenterRows, err)
				}
				s += v
			}
			means[i] = s / float64(c)
		}
	} // for c==0, means zeros

	// Stage 3 (Apply): broadcast-subtract row means over columns.
	Xc, err := ewBroadcastSubRows(X, means)
	if err != nil {
		return nil, nil, matrixErrorf(opCenterRows, err)
	}

	// Return centered matrix and means.
	return Xc, means, nil
}

// normalizeRowsL1 scales each row to have L1-norm == 1 when possible.
// Implementation:
//   - Stage 1: Validate X (non-nil) and handle zero-size as a strict no-op.
//   - Stage 2: Compute per-row L1 norms deterministically.
//   - Stage 3: Build row scale factors (1/norm); for norm==0 use scale=1 to keep the row unchanged.
//   - Stage 4: Apply ewScaleRows to produce a normalized copy.
//
// Behavior highlights:
//   - Degenerate rows (norm==0) are left unchanged (stable policy).
//
// Inputs:
//   - X: input matrix (r×c).
//
// Returns:
//   - Matrix: normalized copy (r×c) for r>0 && c>0; otherwise X itself (no-op).
//   - []float64: L1 norms (len=r).
//
// Errors:
//   - ErrNilMatrix from validation.
//   - Wrapped At/NewDense/Set errors from fallback paths.
//
// Determinism:
//   - Fixed i→j traversal; no randomness.
//
// Complexity:
//   - Time O(r*c), Space O(r*c) for output (+ O(r) norms + O(r) scales).
//
// Notes:
//   - L1 norm is Σ_j |x_ij|.
//
// AI-Hints:
//   - Prefer L1 normalization for row-stochastic style preprocessing.
func normalizeRowsL1(X Matrix) (Matrix, []float64, error) {
	// Stage 1 (Validate): ensure X is present.
	if err := ValidateNotNil(X); err != nil {
		return nil, nil, matrixErrorf(opNormalizeRowsL1, err)
	}

	// Stage 1 (Zero-size policy): no elements => no-op.
	r, c := X.Rows(), X.Cols()
	norms := make([]float64, r)
	if r == 0 || c == 0 {
		// Zero-shape law:
		//   - 0×N has no row norms.
		//   - N×0 has N row norms, all vacuously 0.
		//   - The matrix itself is returned because there are no elements to scale.
		return X, norms, nil
	}

	// Stage 2 (Execute): compute L1 norms per row.
	var i, j, base int
	var s, v float64
	var err error

	if d, ok := X.(*Dense); ok {
		for i = 0; i < r; i++ {
			s = 0.0
			base = i * c
			for j = 0; j < c; j++ {
				v = d.data[base+j]
				if v < 0 {
					v = -v // abs
				}
				s += v
			}
			norms[i] = s
		}
	} else {
		for i = 0; i < r; i++ {
			s = 0.0
			for j = 0; j < c; j++ {
				v, err = X.At(i, j)
				if err != nil {
					return nil, nil, matrixErrorf(opNormalizeRowsL1, err)
				}
				if v < 0 {
					v = -v
				}
				s += v
			}
			norms[i] = s
		}
	}

	// Stage 3 (Prepare scales):
	//   - positive finite norm => scale by 1/norm;
	//   - exact zero norm      => leave the row unchanged;
	//   - non-finite norm      => error, because NaN/Inf is invalid data or overflow,
	//                             not a legitimate degenerate row.
	scale := make([]float64, r)
	for i = 0; i < r; i++ {
		scale[i], err = rowNormScale(norms[i])
		if err != nil {
			return nil, nil, matrixErrorf(opNormalizeRowsL1, err)
		}
	}

	// Stage 4 (Apply): scale rows via the canonical ew micro-kernel.
	Y, err := ewScaleRows(X, scale)
	if err != nil {
		return nil, nil, matrixErrorf(opNormalizeRowsL1, err)
	}

	// Return normalized matrix and original norms.
	return Y, norms, nil
}

// ---------- helpers ----------

// rowNormScale converts a computed row norm into a normalization scale.
//
// What:
//   - finite positive norm => 1/norm;
//   - exact zero norm      => statsDegenerateRowScale;
//   - NaN/+Inf/-Inf norm   => ErrNaNInf.
//
// Why:
//   - Zero rows should remain unchanged.
//   - Non-finite norms are not legitimate degeneracy; they indicate invalid
//     input or floating-point overflow and must not be silently treated as a
//     zero/degenerate row.
//
// Errors:
//   - ErrNaNInf when norm is NaN or ±Inf.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// AI-Hints:
//   - Keep this helper shared by L1 and L2 normalization.
//   - Do not inline the norm==0 policy in each normalizer.
func rowNormScale(norm float64) (float64, error) {
	if isNonFinite(norm) {
		return 0, ErrNaNInf
	}
	if norm == 0 {
		return statsDegenerateRowScale, nil
	}

	return 1.0 / norm, nil
}
