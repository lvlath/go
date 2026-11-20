// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix provides deterministic, allocation-aware linear algebra
// and graph-by-matrix utilities used across lvlath.
//
// The package enables efficient construction, manipulation, and analysis of
// graph representations in matrix form (adjacency, incidence), plus spectral
// analysis (eigen) and statistics commonly used in graph/ML workflows.
//
// # Philosophy
//
// - Determinism first: loops use fixed orders; outputs are bit-stable across runs.
// - Clear contracts:
//   - Classic adjacency uses 0 as "no edge"; zero-preserving weighted adjacency
//     instead represents absence with +Inf so a real zero-weight edge survives.
//   - Distance/MetricClosure uses +Inf for unreachable/absent distances;
//     diagonal is normalized to 0 for APSP. This does not permit +Inf in SVD.
//
// - Single source of truth: fast-paths on *Dense live in canonical kernels; fallbacks match behavior.
// - No silent policy drift: validation is explicit (dimension/shape/nil/singularity).
//
// # What (major building blocks)
//
//	Core kernels
//	- Add / Sub / Hadamard / Scale / Transpose - element-wise ops and transforms.
//	- Mul / MatVec - matrix×matrix and matrix×vector.
//	- Eigen (Jacobi) - symmetric eigen-decomposition (values + eigenvectors).
//	- LU / QR - compatibility decompositions without pivoting (square matrices).
//	- FactorLU / FactorQR / FactorCholesky - reusable detached numerical factors.
//	- SolveLeastSquares / factor.Solve - full-rank systems with true residual norms.
//	- Inverse - legacy inversion without pivoting; not a replacement for Solve.
//	- FloydWarshall - APSP in-place (k→i→j); +Inf=no edge; diag=0.
//
//	Graph-related
//	- AdjacencyMatrix - dense adjacency with stable vertex order and options (directed/weighted/loops/multi).
//	- NewAdjacencyMatrix / ToGraph - round-trip between core.Graph and adjacency (metric-closure guarded).
//	- MetricClosure / APSP over adjacency - computes all-pairs shortest-path distances in-place.
//	- DegreeVector - per-vertex row-sum; loops count as 1; +Inf/NaN ignored.
//	- Neighbors - stable-order adjacency scan.
//
//	Facades & convenience
//	- T / Sum / Diff / Product / HadamardProd / ScaleBy / MatVecMul - thin aliases to kernels.
//	- RowSums / ColSums - reductions via MatVec (useful for stochastic normalization and features).
//	- NewZeros / NewIdentity / ZerosLike / IdentityLike - intention-revealing constructors.
//	- Symmetrize - (A + Aᵀ)/2 to repair asymmetry for spectral methods.
//	- BuildMetricClosure - adjacency build + APSP + policy flag to refuse edge export.
//	- CenterColumns / CenterRows - mean-centering by columns/rows (PCA/DTW/Regression).
//	- NormalizeRowsL1 / NormalizeRowsL2 - row-stochastic / L2-normalization.
//	- Covariance / Correlation - sample covariance and Pearson correlation (by columns).
//	- AllClose - PR-friendly equality with tolerances.
//	- Clip / ReplaceInfNaN - numeric sanitizers for robust pipelines.
//
// # Economy SVD: mathematical and operational contract
//
//   - SVD, SVDDefault, SVDWithTolerance and SVDWithMaxIterations delegate to one
//     cyclic one-sided Jacobi implementation. No A^T*A or A*A^T Gram matrix is
//     constructed to compute singular values, and no pseudoinverse is returned.
//   - Inputs are nonempty finite m-by-n matrices. k=min(m,n); successful results
//     contain U[m,k], SingularValues[k] and VT[k,n] in descending singular order.
//     Wide matrices are treated through the transpose; no full large orthogonal
//     basis is allocated. Each nonzero working column keeps a separate binary
//     exponent so one large column does not erase a representable tiny mode.
//   - SVDOptions with zero values uses 100 complete lexicographic sweeps and
//     normalized correlation tolerance 8*2^-52. Tolerance is not a rank cutoff.
//     Rank counts singular ratios above max(m,n)*2^-52 and does not remove small
//     nonzero singular values or their vectors from the full economy result.
//   - Exact-zero singular directions receive deterministic twice-modified
//     Gram-Schmidt completion from standard basis candidates. Signs are fixed
//     using the first maximum-absolute coordinate of each public U column and
//     corresponding VT row. Equal singular subspaces are not a unique basis.
//   - Validation precedence is context, input presence and shape, options, then
//     finite cell values. Nil ctx and invalid options produce ErrInvalidOption;
//     nil inputs ErrNilMatrix; empty SVD shapes ErrBadShape; bad data ErrNaNInf;
//     unrepresentable published singular values ErrOverflow. Context cancellation
//     returns nil and the original context error. Generic At failures stay wrapped.
//   - Sweep exhaustion returns ErrNotConverged and a *diagnostic* SVDResult with
//     Converged=false. Such a result is not guaranteed orthogonal. Failure to
//     complete a zero basis may instead return nil with ErrNotConverged. Other
//     errors never publish usable factors. SVDResult.IsNil and SolveResult.IsNil
//     report pointer nilability only, not numerical validity or fitted status.
//   - Inputs are read-only. Returned matrices and singular values are detached;
//     workspaces are per-call. Concurrent read-only calls are supported when the
//     supplied Matrix is not concurrently mutated, including through a View.
//   - A global float64 accuracy guarantee for arbitrarily coupled extreme-scale
//     data is not claimed. An unrepresentable rotation coefficient or a
//     roundoff-sensitive low-rank case can prevent convergence; do not hide this
//     by zeroing small nonzero singular modes or weakening tolerances implicitly.
//
// # Non-goals and package boundaries
//
//   - SVD does not solve least squares or compute a pseudoinverse. FactorQR and
//     SolveLeastSquares remain strict full-column-rank solvers without SVD fallback.
//   - SVD does not fit a PCA model, choose a feature center, manage heldout data,
//     infer time-index masks, or certify downstream application decisions.
//     The reserved pca.ThinSVD selection is not enabled by matrix.SVD alone.
//   - Covariance computes sample cross-feature covariance as its own statistical
//     result; the prohibition on Gram-based SVD does not remove that operation.
//   - A Matrix may represent distances with +Inf under explicit policy, but that
//     numerical domain is not accepted by finite-only SVD or factorization.
//   - Dimension checks prevent integer overflow, not arbitrary runtime OOM.
//
// # Why
//
// Dense adjacency provides O(1) edge queries and predictable throughput.
// Deterministic kernels benefit spectral and optimization methods.
// Graph algorithms may use matrix adapters for explicit dense representations,
// while numerical consumers choose stable covariance, factors, or direct SVD.
// These operations never redefine the independent contracts of graph packages.
//
// # Complexity (typical)
//
//   - Add / Sub / Hadamard / Scale / Transpose: O(rc) time, O(rc) space
//   - Mul:                                      O(rnc) time, O(rc) space
//   - MatVec:                                   O(rc) time, O(r) space
//   - Eigen (max-offdiag Jacobi):              O(rotations*n²) time, O(n²) space
//   - Legacy LU / Inverse / QR:                 O(n³) time, O(n²) space
//   - FactorLU / FactorCholesky:                O(n³) time, O(n²) retained space
//   - FactorQR (m>=n):                          O(m*n²+n³) time, O(m*n) retained space
//   - SVD (k=min(m,n)):                         O((sweeps+1)*max(m,n)*k²) time,
//     O((m+n)*k) total matrix workspace/output
//   - Reusable Solve:                           O(n²*r) triangular solve, plus
//     exact residual computation
//   - FloydWarshall:                            O(n³) time, O(1) extra (in-place)
//   - DegreeVector / RowSums / ColSums:         O(n²) time, O(n) space
//   - Covariance / Correlation:                 O(r*c²) time, O(c²+c) workspace;
//     exact rescue may add big-number work
//
// # Options (adjacency/incidence builders)
//
//   - Directed:     orient edges; Undirected: mirror (loops preserved as configured).
//   - Weighted:     keep weights; otherwise degrade to binary (1).
//   - AllowMulti:   dense adjacency collapses parallel edges using stable last-write-wins;
//     disabled multi mode uses first-edge-wins, never an edge-identity ledger.
//   - AllowLoops:   admit self-loops; DegreeVector counts a loop as 1 if present.
//   - MetricClosure: if a matrix encodes APSP distances, mark as non-exportable to edges.
//
// # Sentinel errors
//
//   - ErrNilMatrix, ErrDimensionMismatch, ErrNotSymmetric, ErrSingular,
//     ErrUnknownVertex, ErrGraphNil, ErrMatrixNotImplemented, ErrEigenFailed,
//     ErrBadShape (invalid shapes like empty adjacency/incidence),
//     ErrOverflow, ErrRankDeficient, ErrNotPositiveDefinite,
//     ErrIllConditioned, ErrInvalidOption, ErrNotConverged,
//     and policy-specific sentinels defined in this package.
//
// # Best practices
//
//   - Prefer *Dense to unlock flat-slice fast paths and minimize interface dispatch.
//   - Use +Inf for “no edge”; keep diagonal 0 for APSP/MetricClosure semantics.
//   - Stabilize tests: repeat runs and compare with AllClose at agreed ε.
//   - CenterColumns, Covariance, Correlation, and NormalizeRowsL2 require finite
//     observations. Their arithmetic overflow is distinct from a nonfinite input.
//   - Sample covariance and correlation require at least two rows, even at zero columns.
//     At two or more rows, zero-feature results have shape 0x0 and empty metadata.
//   - A constant column has zero Pearson correlation with every column, including itself.
//   - For untrusted numeric inputs use ReplaceInfNaN / Clip explicitly before statistics.
//
// # Determinism & Reproducibility
//
// - Loops are ordered i→j (or k→i→j for APSP) and never rely on map iteration.
// - Constructors have single allocation where possible; empty slices are zero-initialized.
// - Facades never change numeric policy of kernels and never reorder loops.
//
// # Performance & AI-Hints
//
// - Reuse buffers (ZerosLike) for hot paths; avoid per-call allocations in MatVec/Mul when shapes repeat.
// - For APSP on *Dense, Floyd–Warshall fast path uses a single in-slice triple loop.
// - IdentityLike/ZerosLike help preallocate deterministic staging matrices.
//
// See also:
//   - package-level docs and tests for usage patterns and guarantees,
//   - docs/MATRICES.md for the canonical matrix, solver and economy SVD reference.
package matrix
