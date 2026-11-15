// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pca implements fitted, immutable principal component analysis.
// It discovers a low-dimensional orthogonal subspace in multivariate training
// observations and reuses that frozen subspace for all later inference batches.
//
// Domain and purpose:
//   - Rows represent observations and columns identify ordered features.
//     Statistical independence or temporal stationarity is not inferred.
//   - Fit publishes a *FitResult with a privately owned *Model and detached
//     spectral diagnostics, rather than free-floating eigenvectors and scalars.
//   - The covariance-based reference implementation is CovarianceEigen. It uses
//     matrix.Covariance and matrix.EigenSym, without a second hidden solver.
//   - Reconstruction is suitable for compression and residual analysis after
//     domain-specific validation of omitted variance and decision thresholds.
//
// Shape and numeric laws:
//   - Fit requires n>=2 rows and d>=1 columns and scans every input for finiteness.
//   - Matrix.Covariance performs stable centering in prepared training
//     coordinates. Model independently freezes that prepared-space center
//     as a first-row anchor and high/low exact binary mean displacement.
//   - A one-float rounded mean cannot express a midpoint between adjacent
//     large observations; the composite center preserves those scores.
//   - Standardize additionally fits population deviations (DDOF=0) under
//     stats.RejectConstant; any exactly constant feature is rejected.
//   - Sample covariance always uses n-1, even after population standardization.
//   - Numerical covariance rank cannot exceed min(n-1,d), and explicit k>rank
//     returns ErrRankDeficient rather than padding zero-variance components.
//   - Components=0 selects the detected rank; RankTolerance=0 selects
//     machineEpsilon*max(n,d) against covariance eigenvalues (not singular values).
//   - EigenTolerance=0 selects a relative 1e-12 max-pivot Jacobi criterion.
//     A looser solver stopping option cannot relax PCA's independent eigenpair
//     verification beyond the default accuracy requirement.
//   - MaxRotations=0 selects checked 50*d*d individual Jacobi rotations.
//   - Zero total training variance returns ErrZeroVariance. A constant
//     column remains legal without standardization if other columns vary.
//   - Only negative eigenvalues within machineEpsilon*d*lambdaMax may be
//     replaced by zero. Material negatives and inaccurate eigenpairs return
//     ErrNumerical; covariance is never silently symmetrized or repaired.
//   - Covariance entries are scaled by a positive binary power chosen from
//     their maximum magnitude before any infinity norm or Jacobi rotation.
//     The power of two is kept as an exponent, not a potentially infinite float.
//   - All d eigenvectors must pass orthogonality and Cscaled*v=lambdaScaled*v
//     against the original scaled covariance snapshot, before rescaling.
//   - A loose eigen-solver stopping tolerance cannot weaken the independent
//     precision check used before publishing the entire fitted spectrum.
//
// Fitted projection law:
//   - Components returns detached d-by-k loading columns, orthonormal within
//     the validated numerical error budget and ordered by descending eigenvalue.
//   - Transform first applies frozen population scaling (if selected), then
//     subtracts the separate two-part covariance center in prepared units.
//     It never fits moments on the new batch.
//   - InverseTransform restores the covariance center BEFORE undoing the
//     population scaling and returns original units; Reconstruct composes
//     these operations without refitting.
//   - Zero-row Transform and InverseTransform inputs are legal when the feature
//     or score width matches the fitted model. Training inputs cannot be empty.
//   - ReconstructionResidual is the training Frobenius norm in centered or
//     centered-and-scaled optimization units; it is not a per-row mean or SPE.
//   - ExplainedVarianceRatio divides retained eigenvalues by the complete
//     accepted nonnegative covariance trace, including positive modes below
//     the numerical-rank cutoff; even k=Rank need not sum exactly to one.
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
//     Input matrices must remain stable for the entire read, and feature
//     columns must retain their training order. There are no Model locks
//     because fitted state is never mutated.
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
//   - The exact training-center accumulator adds O(n*d) bounded binary-integer
//     operations; no rational state is retained in the fitted Model.
//   - Sample covariance takes O(n*d*d) time; max-pivot Jacobi takes
//     O(R*d*d) for R rotations because each pivot searches O(d*d) cells.
//     Spectral validation is O(d*d*d); train residual is O(n*d*k).
//     With the default maximum R=50*d*d, this is an O(d^4) worst-case
//     Jacobi bound, not a promise of cubic-time Jacobi convergence.
//   - Fitting uses O(n*d+d*d) workspaces plus O(d*k+d) Model storage.
//     Spectral infinity norms, PSD allowances and eigenpair residuals use
//     bounded coordinates; eigenvalues are restored with math.Ldexp.
//     Each inference call costs O(nnew*d*k) and owns its output buffer.
//
// Explicit limitations and non-goals:
//   - CovarianceEigen squares the condition number of centered training data.
//     It is not a reliable substitute for direct thin SVD on high-condition data.
//   - matrix.EigenSym has no context argument; cancellation is checked before
//     and after its call but cannot interrupt its internal pivot loop.
//   - The eigenvalue PSD allowance is deliberately strict and evaluated in
//     scaled coordinates; it is not a universal proof of covariance accuracy.
//   - Training centering is exact for the supplied binary64 prepared inputs;
//     covariance/eigenvectors and projected dot products remain floating-point
//     approximations rather than an exact arithmetic decomposition.
//   - PCA does not fit regression targets, prove i.i.d. sampling, forecast
//     future observations, assign causal economic factor labels, estimate
//     Nelson-Siegel parameters, calibrate anomaly thresholds, or make
//     trading, medical, or industrial safety decisions by itself.
//   - Training ReconstructionResidual measures prepared-space energy;
//     per-row original-unit reconstruction errors are a distinct diagnostic.
//     A standardized-space SPE cannot be inferred from mixed-unit raw
//     residuals without an explicit, frozen scaling policy.
//
// AI-Hints:
//   - Do not replace corrupted covariance with matrix.Symmetrize, arbitrarily
//     clip negative eigenvalues, or return a last-iteration Jacobi model.
//   - Never fit standardization or centering on a holdout batch; Fit must be
//     isolated to the designated training partition.
//   - Do not flatten a two-part covariance center into one float64 mean,
//     or form a raw infinity norm before binary scaling the covariance.
//   - Do not compare individual loadings across repeated-eigenvalue solvers;
//     use subspace projectors when a whole eigenspace is retained.
//   - Prefer *matrix.Dense for efficient At access; matrix's private backing
//     slice is not part of the PCA cross-package public API.
package pca
