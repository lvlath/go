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

// ---------- Statistics (public surface → internal implementations) ----------

// CenterColumns returns a centered copy: Xc = X − mean(X, by columns) and the column means.
// Returns Xc and the column means (length = Cols(X)).
// Implementation: ColSums + divide by rows to get means; then ewBroadcastSubCols.
// Determinism: fixed loops and pure compositions. O(r*c).
// Time: O(r*c). Space: O(r*c).
//
// AI-Hints: feed means into PCA/Regression; reuse for z-scoring.
func CenterColumns(X Matrix) (Matrix, []float64, error) { return centerColumns(X) }

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

// NormalizeRowsL2 returns Y where each row i is scaled to L2-norm 1 when possible.
//
// Degenerate rows:
//   - If a row has exact L2 norm 0, it is left unchanged.
//   - The returned norm for that row is 0.
//
// Zero-shape law:
//   - 0×N and N×0 inputs are valid no-op transforms.
//   - Returned norms length is Rows(X).
//
// Numeric safety:
//   - Non-finite computed norms are rejected as ErrNaNInf instead of being treated
//     as degenerate rows.
//
// Time: O(r*c). Space: O(r*c). Deterministic.
//
// AI-Hints: common for cosine similarity / spectral features.
func NormalizeRowsL2(X Matrix) (Matrix, []float64, error) { return normalizeRowsL2(X) }

// Covariance computes sample covariance of columns:
//
//	Cov = (Xcᵀ Xc)/(r-1)
//
// Zero-feature law:
//   - If Cols(X)==0, returns a valid 0×0 covariance and empty means.
//
// Sample-size law:
//   - If Cols(X)>0, Rows(X) must be >=2.
//
// Returns Cov and column means.
//
// Time: O(r*c + r*c²). Space: O(r*c + c²).
//
// Notes:
//   - Requires r >= 2 to avoid division by zero; else ErrDimensionMismatch.
//   - Uses CenterColumns then reuses canonical kernels (Transpose/Mul/Scale).
func Covariance(X Matrix) (Matrix, []float64, error) { return covariance(X) }

// Correlation computes Pearson correlation of columns via z-scoring:
//
//	Z = (X - mean) / std
//	Corr = (Zᵀ Z)/(r-1)
//
// Zero-feature law:
//   - If Cols(X)==0, returns valid 0×0 correlation and empty means/stds.
//
// Sample-size law:
//   - If Cols(X)>0, Rows(X) must be >=2.
//
// Degenerate columns:
//   - std==0 columns are mapped to zero columns in Z, producing zero correlation
//     rows/columns for those features.
//
// Time: O(r*c + r*c²). Space: O(r*c + c²).
func Correlation(X Matrix) (Matrix, []float64, []float64, error) { return correlation(X) }
