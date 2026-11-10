// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix - public API facades.
//
// Purpose:
//   - Provide thin, well-documented entry points for common tasks across the package.
//   - Avoid any logic duplication - each facade delegates to the canonical implementation.
//   - Keep function names explicit and intention-revealing to improve discoverability.
//
// Determinism & Policy:
//   - Facades never change the loop orders or numeric policy of underlying kernels.
//   - APSP expects +Inf for "no edge" and 0 on the diagonal; facades preserve this contract.
//   - Validation is performed in the kernels; facades only compose or forward.
//
// AI-Hints:
//   - Prefer passing *Dense to unlock fast-paths in kernels (flat-slice loops).
//   - Use NewIdentity/NewZeros to build matrices with explicit shape and neutral elements.
//   - For APSP, call APSPInPlace (delegates to FloydWarshall).
//   - For graph export, AdjacencyToGraph mirrors GraphFromAdjacency for discoverability.
package matrix

import (
	"context"
	"errors"
	"math"

	"github.com/lvlath/go/core"
)

const (
	opNewZeros      = "NewZeros"
	opNewIdentity   = "NewIdentity"
	opIdentityLike  = "IdentityLike"
	opZerosLike     = "ZerosLike"
	opRowSums       = "RowSums"
	opColSums       = "ColSums"
	opSymmetrize    = "Symmetrize"
	opMetricClosure = "MetricClosure"
)

// ---------- Constructors & Utilities (O(1) alloc + O(rc) zeroing by runtime) ----------

// NewZeros allocates an r×c zero matrix.
// Implementation:
//   - Stage 1: Delegate allocation to NewDense (same numeric policy).
//   - Stage 2: Return the zeroed matrix.
//
// Behavior highlights:
//   - Backwards compatible: opts are optional.
//
// Inputs:
//   - r,c: shape (>= 0).
//   - opts: numeric-policy options forwarded to NewDense.
//
// Returns:
//   - *Dense: zero matrix.
//
// Errors:
//   - ErrInvalidDimensions: on negative dimensions.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(r*c), Space O(r*c).
//
// Notes:
//   - Use WithAllowInfDistances when you plan to Set(+Inf) into the matrix.
//
// AI-Hints:
//   - For APSP inputs with “no path” sentinels, prefer NewZeros(n,n, WithAllowInfDistances()) and then fill +Inf off-diagonal.
func NewZeros(rows, cols int, opts ...Option) (*Dense, error) {
	// Delegate directly to the strict constructor (single allocation).
	d, err := NewPreparedDense(rows, cols, opts...)
	if err != nil {
		return nil, matrixErrorf(opNewZeros, err)
	}

	return d, nil
}

// NewIdentity allocates an n×n identity matrix (square of n; ones on the diagonal, zeros elsewhere).
// Implementation:
//   - Stage 1: Allocate n×n via NewZeros (policy forwarded).
//   - Stage 2: Set diagonal to 1 using Set() (policy-safe).
//
// Behavior highlights:
//   - Backwards compatible: opts are optional.
//
// Inputs:
//   - n: size (>= 0).
//   - opts: numeric-policy options forwarded to NewZeros/NewDense.
//
// Returns:
//   - matrix.Matrix: identity matrix (typically *Dense).
//
// Errors:
//   - ErrInvalidDimensions: if n < 0.
//   - ErrIndexOutOfRange / ErrNaNInf: only if internal invariants are broken (should not happen).
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(n^2) allocation, Space O(n^2).
//
// Notes:
//   - Identity contains only finite values; allowInfDistances does not change the result.
//
// AI-Hints:
//   - Use as a neutral element for inverses/preconditioning/orthogonalization.
//   - Use identity matrices as stable baselines for algebraic property tests.
func NewIdentity(n int, opts ...Option) (*Dense, error) {
	// Allocate an n×n zero matrix via the constructor.
	I, err := NewZeros(n, n, opts...) // O(1) alloc + O(n^2) zeroing
	if err != nil {
		return nil, matrixErrorf(opNewIdentity, err) // propagate constructor error unchanged
	}
	// Set the diagonal deterministically in a single loop.
	for i := 0; i < n; i++ { // fixed i order guarantees reproducibility
		if err = I.Set(i, i, 1.0); err != nil { // Set is bounds-safe; error is not expected after shape validation
			return nil, matrixErrorf(opNewIdentity, err)
		}
	}

	// Return the identity matrix.
	return I, nil
}

// CloneMatrix returns a structural clone of m.
//
// Behavior highlights:
//   - nil input returns nil to keep this convenience helper panic-free.
//   - Use ValidateNotNil in callers that need nil to be an error.
//
// Complexity:
//   - O(r*c) for dense implementations.
func CloneMatrix(m Matrix) Matrix {
	if m == nil {
		return nil
	}

	return m.Clone()
}

// ZerosLike returns a new zero matrix with the same shape as m.
// Complexity: O(1) alloc + O(rc) zeroing. Handy to preallocate staging buffers.
//
// AI-Hints: Useful for staging buffers or accumulating into fresh containers.
func ZerosLike(m Matrix) (*Dense, error) {
	// Validate early: we read dimensions directly.
	if err := ValidateNotNil(m); err != nil {
		return nil, matrixErrorf(opZerosLike, err)
	}
	// Read shape once and call NewDense with the same dimensions.
	d, err := NewZeros(m.Rows(), m.Cols())
	if err != nil { // errors (if any) bubble up
		return nil, matrixErrorf(opZerosLike, err)
	}

	return d, nil
}

// IdentityLike returns I with dimension = Rows(m); requires square shape.
// Complexity: O(n^2). Validates square via central validator.
//
// AI-Hints: Handy to build projectors or initialize iterative schemes.
func IdentityLike(m Matrix) (*Dense, error) {
	// Ensure the input is square using the centralized validator.
	if err := ValidateSquare(m); err != nil {
		return nil, matrixErrorf(opIdentityLike, err) // wrap with call-site tag
	}
	// Construct the identity of matching dimension.
	return NewIdentity(m.Rows()) // returns (*Dense, error)
}

// ---------- Linear Algebra (facades map 1:1 to kernels; O(rc) unless noted) ----------

// Sum is an alias for Add: element-wise a + b.
// Complexity: O(rc).
//
// AI-Hints: Prefer passing *Dense operands for single flat-loop fast-path.
func Sum(a, b Matrix) (Matrix, error) { return Add(a, b) }

// Diff is an alias for Sub: element-wise a − b.
// Complexity: O(rc).
func Diff(a, b Matrix) (Matrix, error) { return Sub(a, b) }

// Product is an alias for Mul: matrix product a × b.
// Complexity: O(r*n*c).
//
// AI-Hints: Prefer Dense to unlock cache-friendly fast path.
func Product(a, b Matrix) (Matrix, error) { return Mul(a, b) }

// HadamardProd is an alias for Hadamard: element-wise product a ⊙ b.
// Complexity: O(rc).
func HadamardProd(a, b Matrix) (Matrix, error) { return Hadamard(a, b) }

// T is an alias for Transpose: returns mᵀ.
// Complexity: O(rc).
//
// AI-Hints: Good for small helpers and chaining.
func T(m Matrix) (Matrix, error) { return Transpose(m) }

// ScaleBy is an alias for Scale: α*m.
// Complexity: O(rc).
func ScaleBy(m Matrix, alpha float64) (Matrix, error) { return Scale(m, alpha) }

// MatVecMul is an alias for MatVec: y = m*x.
// Complexity: O(rc).
//
// AI-Hints: For repeated calls with same shape, reuse x/y slices outside.
func MatVecMul(m Matrix, x []float64) ([]float64, error) { return MatVec(m, x) }

// EigenSym calls the canonical Jacobi eigen-decomposition (symmetric input).
// Complexity: O(maxIter * n^3). Numeric policy unchanged.
// Note: Under the hood it calls Eigen; symmetric validation lives in kernels.
func EigenSym(m Matrix, tol float64, maxIter int) ([]float64, Matrix, error) {
	// Delegate directly to the kernel. The kernel performs ValidateNotNil/Square/Symmetric.
	return Eigen(m, tol, maxIter)
}

// InverseOf is an alias for Inverse: returns A^{-1} (no pivoting; deterministic).
// Complexity: O(n^3).
func InverseOf(m Matrix) (Matrix, error) { return Inverse(m) }

// LUDecompose is an alias for LU: returns (L, U) with unit diagonal on L.
// Complexity: O(n^3).
func LUDecompose(m Matrix) (Matrix, Matrix, error) { return LU(m) }

// QRDecompose is an alias for QR: returns (Q, R) via Householder reflections.
// Complexity: O(n^3).
func QRDecompose(m Matrix) (Matrix, Matrix, error) { return QR(m) }

// ---------- APSP / Metric Closure (graph kernels; O(n^3)) ----------

// APSPInPlace runs Floyd–Warshall in-place on m (all-pairs shortest paths).
// Thin alias to FloydWarshall; provided for graph-oriented API discoverability.
// Contract: m square; +Inf for “no edge”; diagonal 0. Deterministic k→i→j loop order.
// AI-Hints: For *Dense, the fast path uses a single in-slice triple loop.
func APSPInPlace(m Matrix) error { return FloydWarshall(m) }

// MetricClosure mutates am.Mat into all-pairs shortest-path distances.
//
// Contract:
//   - am must be a valid adjacency wrapper.
//   - If am is already marked as metric-closure, the function is idempotent and
//     only re-validates/runs FloydWarshall on the distance matrix.
//   - Otherwise the stored adjacency encoding is converted to distance encoding:
//     0-as-no-edge   => InitDistancesInPlace;
//     +Inf-as-no-edge => normalize only the diagonal, preserving zero-weight edges.
//
// Behavior highlights:
//   - Preserves P3 zero-weight-edge semantics.
//   - Marks am.opts.metricClose=true after successful conversion.
//   - ToGraph must refuse the result.
//
// Complexity:
//   - O(n²) conversion + O(n³) Floyd–Warshall.
//
// AI-Hints:
//   - Do not call FloydWarshall directly on raw 0/weight adjacency.
//   - Do not use InitDistancesInPlace for +Inf-no-edge weighted adjacency; it would
//     erase real zero-weight off-diagonal edges.
func MetricClosure(am *AdjacencyMatrix) error {
	if err := ValidateGraphAdjacency(am); err != nil {
		return matrixErrorf(opMetricClosure, err)
	}

	d, ok := am.Mat.(*Dense)
	if !ok {
		return matrixErrorf(opMetricClosure, ErrMatrixNotImplemented)
	}

	// If the wrapper is already a distance matrix, FloydWarshall is idempotent
	// for valid APSP distances and still performs validation.
	if !am.opts.metricClose {
		if am.adjacencyUsesInfNoEdge() {
			if err := normalizeInfNoEdgeDistanceDiagonal(d); err != nil {
				return matrixErrorf(opMetricClosure, err)
			}
		} else {
			if err := InitDistancesInPlace(d); err != nil {
				return matrixErrorf(opMetricClosure, err)
			}
		}
	}

	if err := FloydWarshall(d); err != nil {
		return matrixErrorf(opMetricClosure, err)
	}

	am.opts.metricClose = true
	am.opts.allowInfDistances = true

	return nil
}

// BuildMetricClosure constructs a metric-closure adjacency wrapper from g.
//
// Implementation:
//   - Stage 1: force metric-closure distance policy in Options.
//   - Stage 2: delegate to NewAdjacencyMatrix, whose builder performs adjacency
//     construction, distance initialization, Floyd–Warshall, and negative-cycle detection.
//   - Stage 3: persist policy flags on the wrapper for downstream ToGraph refusal.
//
// Notes:
//   - Empty graphs are supported: the result is a valid 0×0 distance matrix.
//   - Floyd–Warshall must run exactly once here; NewAdjacencyMatrix already does it
//     because opts.metricClose=true.
//
// AI-Hints:
//   - Use for TSP/DTW pipelines where you want pairwise shortest-path distances immediately.
//   - Keep in mind the diagonal=0 and +Inf for unreachable pairs policy.
//   - Do not call FloydWarshall again after NewAdjacencyMatrix in this function.
func BuildMetricClosure(g *core.Graph, opts Options) (*AdjacencyMatrix, error) {
	// Stage 1: enforce distance-policy regardless of caller opts.
	// This guarantees:
	//   - “no edge / no path” is represented as +Inf
	//   - the underlying Dense is allowed to store +Inf (via AllowInfDistances).
	opts.metricClose = true
	opts.allowInfDistances = true

	// Stage 2: build adjacency deterministically using the adapter builder.
	am, err := NewAdjacencyMatrix(g, opts)
	if err != nil {
		return nil, err
	}

	// Stage 3: persist policy flags on the wrapper for correct downstream behavior (ToGraph refusal).
	am.opts.metricClose = true
	am.opts.allowInfDistances = true

	// Return the mutated adjacency wrapper.
	return am, nil
}

// ---------- Graph <-> Adjacency helpers (thin; no hidden semantics) ----------

// BuildAdjacency constructs a deterministic adjacency matrix from a core.Graph.
// Thin alias to NewAdjacencyMatrix; exposed in API to improve discoverability.
// Notes: Empty graphs (0 vertices) are supported and produce a valid 0×0 adjacency.
// AI-Hints: Pass Options that match your graph semantics (directed/loops/multi/weighted).
func BuildAdjacency(g *core.Graph, opts Options) (*AdjacencyMatrix, error) {
	return NewAdjacencyMatrix(g, opts)
}

// GraphFromAdjacency exports a core.Graph from AdjacencyMatrix with the given options.
// Thin alias to (*AdjacencyMatrix).ToGraph with identical behavior.
func GraphFromAdjacency(am *AdjacencyMatrix, optFns ...Option) (*core.Graph, error) {
	return am.ToGraph(optFns...)
}

// AdjacencyToGraph is a discoverability alias for GraphFromAdjacency.
// Same semantics; different entry name for API cohesion.
func AdjacencyToGraph(am *AdjacencyMatrix, optFns ...Option) (*core.Graph, error) {
	// Delegate to the canonical adapter export.
	return am.ToGraph(optFns...)
}

// DegreeVector returns per-vertex row sums on an adjacency (loops count as 1).
// Thin alias to (*AdjacencyMatrix).DegreeVector; documented here for API cohesion.
func DegreeVector(am *AdjacencyMatrix) ([]float64, error) {
	return am.DegreeVector()
}

// ---------- Convenience facades (compositions only; no loop duplication) ----------

// Symmetrize returns (m + mᵀ)/2. Deterministic composition: Transpose → Add → Scale.
// Complexity: O(rc).
//
// AI-Hints: Useful in spectral methods (PCA, Laplacians) to repair asymmetry drift.
func Symmetrize(m Matrix) (Matrix, error) {
	// Validate early to avoid nil-deref when reading sizes in downstream kernels
	// Symmetrization is defined for square matrices only: (A + Aᵀ)/2 has the same shape as A iff A is square.
	if err := ValidateSquare(m); err != nil {
		return nil, matrixErrorf(opSymmetrize, err)
	}
	// Transpose first; kernel validates non-nil input.
	mt, err := Transpose(m) // O(rc)
	if err != nil {
		return nil, matrixErrorf(opSymmetrize, err) // wrap with context
	}
	// Add original and transpose; shapes are guaranteed identical.
	sum, err := Add(m, mt) // O(rc)
	if err != nil {
		return nil, matrixErrorf(opSymmetrize, err) // wrap
	}

	// Scale by 0.5 to complete the symmetrization.
	return Scale(sum, 0.5) // O(rc)
}

// RowSums returns vector r where r[i] = sum_j m[i,j].
// Implementation: MatVec(m, ones(cols)). No custom loops.
// Complexity: O(rc).
//
// AI-Hints: Used by Markov/stochastic normalization, degree-like features, DTW bands, etc.
func RowSums(m Matrix) ([]float64, error) {
	// Validate early: we read m.Cols() below.
	if err := ValidateNotNil(m); err != nil {
		return nil, matrixErrorf(opRowSums, err)
	}
	// Build an all-ones vector of length equal to the number of columns.
	cols := m.Cols()              // O(1) read of dimension
	ones := make([]float64, cols) // allocate the vector once
	for j := 0; j < cols; j++ {   // deterministic fill
		ones[j] = 1.0 // neutral element for summation
	}

	// Multiply m by the ones vector to get per-row sums.
	y, err := MatVec(m, ones) // O(rc)
	if err != nil {
		return nil, matrixErrorf(opRowSums, err)
	}

	return y, nil
}

// ColSums returns vector c where c[j] = sum_i m[i,j].
// Implementation: T(m) then MatVec with ones(rows).
// Complexity: O(rc).
//
// AI-Hints: Useful for indegree-like stats, column-normalization, PCA centering.
func ColSums(m Matrix) ([]float64, error) {
	// Transpose m first.
	mt, err := Transpose(m) // O(rc)
	if err != nil {
		return nil, matrixErrorf(opColSums, err) // wrap with context
	}
	// Build an all-ones vector of length equal to the (transposed) number of columns,
	// which equals the original number of rows.
	rows := mt.Cols()             // == m.Rows()
	ones := make([]float64, rows) // allocate the vector once
	for i := 0; i < rows; i++ {   // deterministic fill
		ones[i] = 1.0 // neutral element for summation
	}

	// Multiply to get per-column sums of the original matrix.
	y, err := MatVec(mt, ones) // O(rc)
	if err != nil {
		return nil, matrixErrorf(opColSums, err)
	}

	return y, nil
}

// ---------- Sanitization & numeric compare (thin wrappers → ew*) ----------

// Clip returns a copy of m with elements clamped into [lo, hi].
//
//	out[i,j] = min(max(A[i,j], lo), hi)
//
// Policy:
//   - lo and hi must be finite.
//   - If lo > hi, bounds are swapped by the kernel.
//   - NaN/±Inf bounds are rejected.
//
// Time: O(r*c). Space: O(r*c). Deterministic.
//
// AI-Hints:
//   - Use ReplaceInfNaN before Clip when input data itself may contain non-finite values.
//   - Use Clip to enforce constraints such as probabilities in [0,1].
func Clip(m Matrix, lo, hi float64) (Matrix, error) {
	// Delegate to the private element-wise kernel (centralizes the loop).
	return ewClipRange(m, lo, hi) // errors are already wrapped with "Clip" tag inside
}

// ReplaceInfNaN returns a copy of m where any {±Inf, NaN} are replaced by 'val' (finite).
// Time: O(r*c). Space: O(r*c). Deterministic.
//
// Policy: 'val' must be finite; otherwise ErrNaNInf is returned.
// AI-Hints:
//   - Use ReplaceInfNaN before statistics to avoid NaN propagation.
//   - stabilize downstream stats and ML features.
func ReplaceInfNaN(m Matrix, val float64) (Matrix, error) {
	// Delegate to the private ew* sanitizer (centralizes numeric checks and loops).
	return ewReplaceInfNaN(m, val) // errors are wrapped with "ReplaceInfNaN" tag inside
}

// AllClose checks element-wise |a-b| ≤ atol + rtol*|b| for identical shapes.
// Returns (true,nil) if all elements satisfy the relation; (false,nil) otherwise.
// NaN != anything; +Inf equals +Inf; -Inf equals -Inf. Deterministic.
// Time: O(r*c). Space: O(1). Deterministic.
//
// Policy:
//   - a and b must be non-nil and have identical shapes.
//   - rtol, atol are treated as |rtol|, |atol| (negative values are normalized).
//
// AI-Hints:
//   - AllClose with small atol/rtol is ideal for invariance tests in unit tests.
func AllClose(a, b Matrix, rtol, atol float64) (bool, error) {
	// Normalize tolerances at API boundary for explicit, stable policy.
	rtol = math.Abs(rtol)
	atol = math.Abs(atol)

	return ewAllClose(a, b, rtol, atol)
}

// ---------- Statistics (canonical public facades) ----------

// CenterColumns materializes the deviations of each observation from its column mean.
// The returned column means use a compensated, scale-aware reduction, and the centered
// matrix is independent of the original nonempty input.
// Implementation:
//   - Stage 1: validate input presence, size arithmetic, and finite observations.
//   - Stage 2: establish per-column scales and compensated means, recovering
//     cancellation-sensitive means from exact binary values when necessary.
//   - Stage 3: materialize finite centered coordinates in a new owning Dense.
//
// Behavior highlights:
//   - Empty element domains preserve the established no-op matrix alias policy.
//   - The Dense fast path reads row-major storage without interface dispatch.
//
// Inputs:
//   - X: caller-owned matrix with nonnegative dimensions and finite observations.
//
// Returns:
//   - Matrix: detached centered Dense for nonempty X; otherwise X itself.
//   - []float64: fresh column means of length X.Cols(), including zero-feature cases.
//
// Errors:
//   - ErrNilMatrix: absent matrix; ErrInvalidDimensions: invalid size arithmetic.
//   - ErrNaNInf: nonfinite input; ErrOverflow: unrepresentable mean or centered cell.
//   - Wrapped Matrix.At errors from generic input implementations.
//
// Determinism: observation rows ascending, then original column indices ascending.
// Complexity: Time O(rows*cols), Space O(rows*cols+cols) including the output.
// Notes: Concurrent mutation of X while the computation runs is unsupported.
// AI-Hints:
//   - A rounded mean may not represent the exact center of adjacent huge values;
//     use the returned centered matrix instead of subtracting the means again.
//   - Prefer *Dense for the direct-slice read path without changing the numeric contract.
func CenterColumns(X Matrix) (Matrix, []float64, error) {
	return statisticsCenterColumns(X)
}

// CenterRows returns a centered copy: Xc[i,*] = X[i,*] − mean(X[i,*]) for each row.
// Returns Xc and the row means. O(r*c).
// Implementation: RowSums + divide by cols; then ewBroadcastSubRows.
// Time: O(r*c). Space: O(r*c). Deterministic.
//
// AI-Hints: useful for DTW/row shifts; for stochastic models, NormalizeRows* is better.
func CenterRows(X Matrix) (Matrix, []float64, error) { return centerRows(X) }

// NormalizeRowsL1 returns Y where each row i is scaled to L1-norm 1 when possible.
//
// Degenerate rows:
//   - If a row has exact L1 norm 0, it is left unchanged.
//   - The returned norm for that row is 0.
//
// Zero-shape law:
//   - 0×N and N×0 inputs are valid no-op transforms.
//   - Returned norms length is Rows(X).
//
// Implementation:
//   - compute per-row L1 norms;
//   - convert norms to scale factors through the canonical rowNormScale policy;
//   - apply ewScaleRows.
//
// Time: O(r*c). Space: O(r*c). Deterministic.
//
// AI-Hints: produce row-stochastic matrices for Markov chains.
func NormalizeRowsL1(X Matrix) (Matrix, []float64, error) { return normalizeRowsL1(X) }

// NormalizeRowsL2 normalizes finite matrix rows by their Euclidean lengths.
// It reports the original row norms and preserves all-zero rows unchanged.
// Implementation:
//   - Stage 1: validate the input and each cell, checking shape arithmetic.
//   - Stage 2: accumulate per-row scaled sums of squares without forming x*x.
//   - Stage 3: materialize the normalized Dense and detached norm slice.
//
// Behavior highlights:
//   - A one-element row containing 1e308 has norm 1e308 and normalized value 1.
//   - Empty element domains retain their original matrix identity by no-op policy.
//
// Inputs:
//   - X: read-only finite matrix with nonnegative rows and columns.
//
// Returns:
//   - Matrix: detached normalized rows for nonempty input; otherwise X itself.
//   - []float64: fresh norms, one per row; zero for an all-zero row.
//
// Errors:
//   - ErrNilMatrix, ErrInvalidDimensions or ErrNaNInf for invalid inputs.
//   - ErrOverflow when a required finite norm does not fit float64; preserves
//     the historical ErrNaNInf norm-error classification through errors.Is.
//   - Wrapped Matrix.At errors for generic implementations.
//
// Determinism: increasing row index; each row scans columns in ascending order.
// Complexity: Time O(rows*cols), Space O(rows*cols+rows) including output.
// Notes: Input remains unmodified; concurrent mutation by the caller is unsupported.
// AI-Hints:
//   - Do not calculate sum(x*x) or multiply by 1/norm: both fail for extreme
//     finite magnitudes; the implementation uses scaled ratios instead.
//   - *Dense enables a direct row-major read path without altering error laws.
func NormalizeRowsL2(X Matrix) (Matrix, []float64, error) {
	out, norms, err := statisticsNormalizeRowsL2(X)
	if err != nil && errors.Is(err, ErrOverflow) {
		return nil, nil, matrixErrorf(opNormalizeRowsL2, errors.Join(err, ErrNaNInf))
	}
	if err != nil {
		return nil, nil, matrixErrorf(opNormalizeRowsL2, err)
	}
	return out, norms, nil
}

// Covariance computes the sample covariance of observation columns.
// For rows >= 2 it returns a c-by-c matrix of unbiased second central moments,
// C[j,k] = sum_i ((x[i,j]-mean[j])*(x[i,k]-mean[k]))/(rows-1).
// Implementation:
//   - Stage 1: validate input presence, checked shape, and sample count.
//   - Stage 2: compute compensated, scaled upper-triangle centered products;
//     recover cross-moments exactly when large opposing terms conceal a result.
//   - Stage 3: divide before rescaling and mirror each computed cell bitwise.
//
// Behavior highlights:
//   - Uses O(cols*cols+cols) auxiliary state; no centered input, transpose or Gram copy.
//   - For rows >= 2 and zero columns, returns a 0-by-0 Dense and empty means.
//
// Inputs:
//   - X: finite rows-by-cols observation matrix, rows >= 2, cols >= 0.
//
// Returns:
//   - Matrix: independently owned, bitwise symmetric c-by-c covariance Dense.
//   - []float64: detached column means, length cols.
//
// Errors:
//   - ErrNilMatrix, ErrInvalidDimensions, ErrDimensionMismatch (rows < 2).
//   - ErrNaNInf for invalid observations; ErrOverflow for unrepresentable output.
//   - Wrapped Matrix.At failures from a non-Dense implementation.
//
// Determinism: rows ascending; feature pairs j <= k ascending; one accumulator per pair.
// Complexity: Time O(rows*cols^2) normally; exact cancellation recovery incurs
// additional arbitrary-precision work. Space O(cols^2+cols), excluding result.
// Notes: No partial covariance or means are published on any error.
// AI-Hints:
//   - An intermediate Gram matrix may overflow even if the sample covariance
//     fits float64; never replace scaled products with raw Xc^T*Xc.
//   - Correlation operates independently: it can be representable when covariance is not.
func Covariance(X Matrix) (Matrix, []float64, error) {
	out, means, err := statisticsCovariance(X)
	if err != nil {
		return nil, nil, matrixErrorf(opCovariance, err)
	}
	return out, means, nil
}

// Correlation computes Pearson coefficients directly from centered column moments.
// Unlike division of a materialized covariance, this operation can succeed
// when finite observations have an unrepresentable intermediate covariance.
// Implementation:
//   - Stage 1: validate sample count, dimensions and finite observations.
//   - Stage 2: compute compensated upper-triangle moments and per-column norms.
//   - Stage 3: form coefficients from normalized moments and mirror each pair.
//
// Behavior highlights:
//   - Constant features have an exactly zero row, column and diagonal.
//   - A nonconstant feature has diagonal correlation exactly one.
//
// Inputs:
//   - X: finite rows-by-cols matrix with at least two observation rows.
//
// Returns:
//   - Matrix: detached c-by-c Pearson correlation Dense.
//   - []float64: detached column means; []float64: detached sample deviations.
//
// Errors:
//   - ErrNilMatrix, ErrInvalidDimensions, ErrDimensionMismatch (rows < 2).
//   - ErrNaNInf for nonfinite input, ErrOverflow for unrepresentable mandatory
//     statistics or an out-of-range coefficient beyond the roundoff bound.
//   - Wrapped Matrix.At errors for generic backends.
//
// Determinism: row-major accumulation; upper feature pairs evaluated once.
// Complexity: Time O(rows*cols^2), with exact cancellation rescue when needed;
// Space O(cols^2+cols) beyond the result and detached metadata.
// Notes: Independent inputs and outputs; no factorization or inverse is involved.
// AI-Hints:
//   - Do not divide Covariance outputs by standard deviations: covariance may
//     overflow while dimensionless Pearson correlations remain finite.
//   - *Dense uses direct row-major data access; zero-variance columns stay zero.
func Correlation(X Matrix) (Matrix, []float64, []float64, error) {
	out, means, stds, err := statisticsCorrelation(X)
	if err != nil {
		return nil, nil, nil, matrixErrorf(opCorrelation, err)
	}
	return out, means, stds, nil
}

// FactorLU builds a reusable deterministic partial-pivoting LU factor.
// It solves full-rank square systems via P*A=L*U without mutating the input.
// Implementation:
//   - Stage 1: reject nil/canceled ctx, invalid options, and invalid shape.
//   - Stage 2: delegate to the scaled partial-pivoting factor kernel.
//   - Stage 3: publish only a complete factor with measured kappa_1(A).
//
// Behavior highlights: Exact ties prefer the smallest original row index.
// Inputs: ctx non-nil; a finite square n-by-n Matrix (n>=1), read-only;
// opts finite relative rank tolerance and optional condition limit.
// Returns: Detached immutable *LUFactor or nil on error/cancellation.
// Errors: ErrInvalidOption, ErrNilMatrix, ErrNonSquare, ErrBadShape,
// ErrNaNInf, ErrRankDeficient (possibly with ErrSingular), ErrIllConditioned,
// ErrOverflow, context error, or wrapped Matrix.At error.
// Determinism: k ascending; pivot rows ordered by magnitude then original ID.
// Complexity: O(n^3) time including 1-norm condition evaluation; O(n^2) storage.
// Notes: Does not change the legacy two-matrix LU(A) signature.
// AI-Hints: Use FactorLU+Solve for pivoted systems; old LU(A) is not this API.
func FactorLU(ctx context.Context, a Matrix, opts FactorOptions) (*LUFactor, error) {
	if err := factorContextErr(ctx); err != nil {
		return nil, err
	}
	if err := factorValidatedOptions(opts); err != nil {
		return nil, err
	}
	return factorLUKernel(ctx, a, opts)
}

// FactorQR builds a reusable column-pivoted Householder QR factor.
// It accepts tall full-column-rank matrices for ordinary least-squares solves.
// Implementation:
//   - Stage 1: check context, numeric options, input shape and finite data.
//   - Stage 2: select deterministic column pivots, storing compact reflectors.
//   - Stage 3: reject rank loss and publish measured kappa_1(R) diagnostics.
//
// Behavior highlights: R is triangular in pivoted-column coordinates;
// no full Q or normal-equation Gram matrix is created.
// Inputs: ctx non-nil; a finite m-by-n Matrix with m>=n>=1, read-only;
// opts finite relative rank cutoff and optional condition limit.
// Returns: Detached immutable *QRFactor, or nil on error/cancellation.
// Errors: ErrInvalidOption, ErrNilMatrix, ErrBadShape, ErrNaNInf,
// ErrRankDeficient, ErrIllConditioned, ErrOverflow, context error, At error.
// Determinism: Pivot columns by trailing norm, ties by original column ID.
// Complexity: O(m*n^2+n^3) time including condition evaluation; O(m*n) memory.
// Notes: The existing QR(A) output orientation is not changed by this API.
// AI-Hints: No minimum-norm rank-deficient or wide mode is offered without SVD.
func FactorQR(ctx context.Context, a Matrix, opts FactorOptions) (*QRFactor, error) {
	if err := factorContextErr(ctx); err != nil {
		return nil, err
	}
	if err := factorValidatedOptions(opts); err != nil {
		return nil, err
	}
	return factorQRKernel(ctx, a, opts)
}

// FactorCholesky builds a reusable Cholesky factor of an accepted SPD matrix.
// Its symmetry tolerances allow an explicitly documented symmetric midpoint,
// not a hidden diagonal shift or regularization of an indefinite matrix.
// Implementation:
//   - Stage 1: validate context and both nonnegative finite tolerances.
//   - Stage 2: verify pairwise symmetry and normalize a detached accepted matrix.
//   - Stage 3: compute an SPD lower factor and measured kappa_1(acceptedA).
//
// Behavior highlights: Exact symmetry is required when both tolerances are 0.
// Inputs: ctx non-nil; a finite square n-by-n Matrix, read-only;
// absTol finite >=0 in coefficient units; relTol finite >=0 dimensionless.
// Returns: Immutable *CholeskyFactor or nil on error/cancellation.
// Errors: ErrInvalidOption, ErrNilMatrix, ErrNonSquare, ErrAsymmetry,
// ErrNotPositiveDefinite, ErrOverflow, ErrNaNInf, context and At errors.
// Determinism: Pair scan and factor triangles use ascending row/column indices.
// Complexity: O(n^3) time including condition evaluation; O(n^2) memory.
// Notes: Residuals after Solve refer to the accepted symmetric midpoint matrix.
// AI-Hints: A positive diagonal alone is not a sufficient SPD test.
func FactorCholesky(ctx context.Context, a Matrix, absTol, relTol float64) (*CholeskyFactor, error) {
	if err := factorContextErr(ctx); err != nil {
		return nil, err
	}
	if math.IsNaN(absTol) || math.IsInf(absTol, 0) || absTol < 0 ||
		math.IsNaN(relTol) || math.IsInf(relTol, 0) || relTol < 0 {
		return nil, ErrInvalidOption
	}
	return factorCholeskyKernel(ctx, a, absTol, relTol)
}

// SolveLeastSquares computes a full-column-rank least-squares solution.
// It is a convenience composition of FactorQR and (*QRFactor).Solve with
// identical rank, residual, condition, and cancellation semantics.
// Implementation:
//   - Stage 1: validate context and FactorOptions through FactorQR.
//   - Stage 2: build one column-pivoted Householder factor of the input.
//   - Stage 3: solve each RHS and return a single detached SolveResult.
//
// Behavior highlights: No explicit inverse, pseudo-inverse, or SVD is constructed.
// Inputs: ctx non-nil; a finite m-by-n Matrix (m>=n>=1); b finite m-by-r
// Matrix (r>=1); opts rank cutoff and optional condition limit.
// Returns: Detached n-by-r SolveResult or nil on any failure.
// Errors: Same errors as FactorQR and QRFactor.Solve; context cancellation
// never publishes a partial matrix or residual metadata.
// Determinism: QR pivot order and RHS column order are both fixed.
// Complexity: O(m*n^2+n^3+m*n*r) arithmetic plus exact residual evaluation;
// O(m*n+n*r) factor and solution storage.
// Notes: Fit algorithms requiring minimum-norm wide solutions need SVD instead.
// AI-Hints: Never route deficient matrices silently into normal equations.
func SolveLeastSquares(ctx context.Context, a, b Matrix, opts FactorOptions) (*SolveResult, error) {
	// All public inputs must be checked before the expensive QR factorization:
	// context -> options -> both nils -> both shapes -> both finite scans.
	if err := factorContextErr(ctx); err != nil {
		return nil, err
	}
	if err := factorValidatedOptions(opts); err != nil {
		return nil, err
	}
	if err := ValidateNotNil(a); err != nil {
		return nil, err
	}
	if err := ValidateNotNil(b); err != nil {
		return nil, err
	}
	rows, cols, err := factorInputShape(a)
	if err != nil {
		return nil, err
	}
	bRows, bCols, err := factorInputShape(b)
	if err != nil {
		return nil, err
	}
	if rows < cols {
		return nil, ErrBadShape
	}
	if bRows != rows {
		return nil, ErrDimensionMismatch
	}
	// Validate without duplicating full m*n or m*r matrices; the owning
	// factor constructor and Solve will subsequently take their snapshots.
	if err = factorWalkFinite(ctx, a, rows, cols, nil); err != nil {
		return nil, err
	}
	if err = factorWalkFinite(ctx, b, bRows, bCols, nil); err != nil {
		return nil, err
	}
	factor, err := FactorQR(ctx, a, opts)
	if err != nil {
		return nil, err
	}
	return factor.Solve(ctx, b)
}
