// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pca implements fitted, immutable principal component analysis.
// It discovers a low-dimensional orthogonal subspace in multivariate training
// observations and reuses that frozen subspace for all later inference batches.
//
// Domain and purpose:
//   - Training rows are independent observations; columns are ordered features.
//   - Fit publishes a *FitResult with a privately owned *Model and detached
//     spectral diagnostics, rather than free-floating eigenvectors and scalars.
//   - The covariance-based reference implementation is CovarianceEigen. It uses
//     matrix.Covariance and matrix.EigenSym, without a second hidden solver.
//   - Reconstruction is suitable for compression and residual analysis after
//     domain-specific validation of omitted variance and decision thresholds.
//
// Shape and numeric laws:
//   - Fit requires n>=2 rows and d>=1 columns and scans every input for finiteness.
//   - All fitted data are centered; Standardize additionally fits population
//     standard deviations using stats.RejectConstant and DDOF=0.
//   - Sample covariance always uses n-1, even after population standardization.
//   - Numerical covariance rank cannot exceed min(n-1,d), and explicit k>rank
//     returns ErrRankDeficient rather than padding zero-variance components.
//   - Components=0 selects the detected rank; RankTolerance=0 selects
//     machineEpsilon*max(n,d) against covariance eigenvalues (not singular values).
//   - EigenTolerance=0 selects a relative 1e-12 max-pivot Jacobi criterion.
//     A looser solver stopping option cannot relax PCA's independent eigenpair
//     verification beyond the default accuracy requirement.
//   - MaxRotations=0 selects checked 50*d*d individual Jacobi rotations.
//   - Zero covariance variance returns ErrZeroVariance; materially negative
//     eigenvalues and inaccurate eigenpairs return ErrNumerical, never a clipped
//     or symmetrized substitute for the actual covariance.
//   - Every accepted eigenvector must be orthonormal to all other loading
//     columns and satisfy C*v=lambda*v against the original covariance.
//   - A loose eigen-solver stopping tolerance cannot weaken the independent
//     precision check used before publishing the entire fitted spectrum.
//
// Fitted projection law:
//   - Components returns detached d-by-k loading columns, orthonormal within
//     the validated numerical error budget and ordered by descending eigenvalue.
//   - Transform returns a detached nnew-by-k score matrix using ONLY stored
//     training means and scales. It never fits moments on the new batch.
//   - InverseTransform returns a detached nnew-by-d reconstruction in original
//     units; Reconstruct composes these existing operations without refitting.
//   - Zero-row Transform and InverseTransform inputs are legal when the feature
//     or score width matches the fitted model. Training inputs cannot be empty.
//   - ReconstructionResidual is the training Frobenius norm in centered or
//     centered-and-scaled optimization units; it is not a per-row mean or SPE.
//   - ExplainedVarianceRatio is divided by all accepted nonnegative covariance
//     eigenvalues. Its k entries do not automatically sum to one when truncated.
//
// Determinism:
//   - Training and inference walk rows and original features in increasing order.
//   - Jacobi chooses its maximal off-diagonal pivot under matrix's fixed scan.
//   - Eigenpairs sort in descending order; exact equal eigenvalues retain the
//     solver index; loading signs use the first maximum-absolute coordinate.
//   - Degenerate eigenvalue subspaces admit many valid rotated bases. Cross-kernel
//     tests compare projectors of fully retained subspaces, not raw columns.
//
// Ownership, nil, and concurrency laws:
//   - Fit neither modifies nor retains the training Matrix and returns nil on
//     ANY error. No partially trained model or diagnostic snapshot is published.
//   - Every fitted parameter slice is a private owned copy; public result slices
//     and matrices returned by accessor/inference calls are detached.
//   - A nil *FitResult obeys core.Nilable; a non-nil zero-valued result is present
//     but not evidence of a successful fit.
//   - A nil or zero-valued Model rejects inference through ErrNilInput;
//     Components returns nil because its signature has no error channel.
//   - Concurrent read-only Model inference is safe with per-call workspace.
//     Concurrent input mutation by callers is unsupported; there are no locks
//     because fitted Model state is never mutated.
//
// Error law:
//   - New canonical validation order is: context, options, typed-nil input,
//     shape/checked size, non-finite observation, domain/rank, computation.
//   - Nil context returns ErrInvalidOption; canceled context returns ctx.Err().
//   - ThinSVD is explicitly unsupported: requests return ErrUnsupportedSolver
//     before reading any input and never run CovarianceEigen as a fallback.
//   - ErrNotConverged wraps matrix.ErrMatrixEigenFailed. Other lower-level
//     matrix/stats errors retain their original identity through errors.Is.
//   - ErrOverflow distinguishes unrepresentable finite arithmetic from nonfinite
//     caller input. ErrNumerical distinguishes failed PSD/eigenpair invariants.
//
// Complexity:
//   - Sample covariance is O(n*d*d) time; max-pivot Jacobi is O(rotations*d*d)
//     due to its full off-diagonal search per rotation. Spectral validation
//     additionally costs O(d*d*d), while residual computation is O(n*d*k).
//   - Fitting uses O(n*d+d*d) workspaces plus O(d*k+d) fitted Model storage;
//     each inference call costs O(nnew*d*k) and returns its own output buffer.
//
// Explicit limitations and non-goals:
//   - CovarianceEigen squares the condition number of centered training data.
//     It is not a reliable substitute for direct thin SVD on high-condition data.
//   - matrix.EigenSym has no context argument; cancellation is checked before
//     and after its call but cannot interrupt its internal pivot loop.
//   - The eigenvalue PSD allowance is deliberately strict and scale-aware;
//     it is not a universal proof of covariance formation backward accuracy.
//   - PCA does not fit regression targets, interpret economic factors causally,
//     estimate Nelson-Siegel parameters, calibrate anomaly thresholds, or make
//     trading, medical, or industrial safety decisions by itself.
//
// AI-Hints:
//   - Do not replace corrupted covariance with matrix.Symmetrize, arbitrarily
//     clip negative eigenvalues, or return a last-iteration Jacobi model.
//   - Never fit standardization or centering on a holdout batch; Fit must be
//     isolated to the designated training partition.
//   - Do not compare individual loadings across repeated-eigenvalue solvers;
//     use subspace projectors when a whole eigenspace is retained.
//   - Prefer *matrix.Dense for efficient At access; matrix's private backing
//     slice is not part of the PCA cross-package public API.
package pca
