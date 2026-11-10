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
//   - Normal adjacency uses 0 as “no edge”, therefore 0-weight edges are indistinguishable from missing edges.
//   - Distance/MetricClosure uses +Inf as “no edge”; diagonal is forced to 0 for APSP.
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
// # Why
//
// Dense adjacency provides O(1) edge queries and predictable throughput.
// Deterministic kernels benefit spectral and optimization methods.
// Graph workflows (BFS/DFS/Dijkstra/Prim/Kruskal/Flows/DTW/TSP) rely on matrices,
// while analytics (PCA/Regression/HMM/Monte-Carlo/SAX) use centering, scaling, and reductions.
//
// # Complexity (typical)
//
//   - Add / Sub / Hadamard / Scale / Transpose: O(rc) time, O(rc) space
//   - Mul:                                      O(rnc) time, O(rc) space
//   - MatVec:                                   O(rc) time, O(r) space
//   - Eigen (Jacobi):                           O(maxIter*n³) time, O(n²) space
//   - Legacy LU / Inverse / QR:                 O(n³) time, O(n²) space
//   - FactorLU / FactorCholesky:                O(n³) time, O(n²) retained space
//   - FactorQR (m>=n):                          O(m*n²+n³) time, O(m*n) retained space
//   - Reusable Solve:                           O(n²*r) triangular solve, plus
//     exact residual computation
//   - FloydWarshall:                            O(n³) time, O(1) extra (in-place)
//   - DegreeVector / RowSums / ColSums:         O(n²) time, O(n) space
//   - Covariance / Correlation:                 O(r*c²) time, O(c²+c) workspace;
//     exact rescue may add big-number work
//
// # Options (adjacency/incidence builders)
//
// - Directed:     orient edges; Undirected: mirror (loops preserved as configured).
// - Weighted:     keep weights; otherwise degrade to binary (1).
// - AllowMulti:   preserve parallel edges when true; else take first (deterministic).
// - AllowLoops:   admit self-loops; DegreeVector counts a loop as 1 if present.
// - MetricClosure: if a matrix encodes APSP distances, mark as non-exportable to edges.
//
// # Sentinel errors
//
//   - ErrNilMatrix, ErrDimensionMismatch, ErrNotSymmetric, ErrSingular,
//     ErrUnknownVertex, ErrGraphNil, ErrMatrixNotImplemented, ErrEigenFailed,
//     ErrBadShape (invalid shapes like empty adjacency/incidence),
//     ErrOverflow, ErrRankDeficient, ErrNotPositiveDefinite,
//     ErrIllConditioned, ErrInvalidOption,
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
//   - docs/matrix.md for a full tutorial and best practices.
package matrix
