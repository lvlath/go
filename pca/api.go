// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pca exposes the canonical fitted PCA facade and narrowly scoped helpers.
// Every helper delegates to Fit so validation, mathematics, and diagnostics agree.
package pca

import (
	"context"

	"github.com/lvlath/go/matrix"
)

// Fit learns a frozen orthogonal projection from observations arranged by rows.
// It centers every feature, optionally uses population-standardized coordinates,
// and factors the sample covariance without substituting another solver.
//
// Implementation:
//   - Stage 1: validate context, complete options, matrix presence, and checked sizes.
//   - Stage 2: scan every input value before domain/rank-dependent computation.
//   - Stage 3: delegate preparation, covariance, spectral validation, and publication.
//
// Behavior highlights:
//   - Components=0 retains the entire detected numerical rank, not all d columns.
//   - Solver=ThinSVD is rejected before inspecting x; no covariance fallback occurs.
//   - Standardize uses training-only population scales and rejects constant features.
//   - EigenTolerance is relative to covariance norm; MaxRotations counts Jacobi rotations.
//
// Inputs:
//   - ctx: non-nil active operation context; cancellation propagates as ctx.Err().
//   - x: caller-owned n-by-d observations; n>=2, d>=1; no concurrent mutation.
//   - opts: copied training and numerical policy; see Options for defaults and units.
//
// Returns:
//   - *FitResult: detached spectral diagnostics and an immutable fitted Model.
//   - error: classified failure, always accompanied by a nil result.
//
// Errors:
//   - ErrInvalidOption: nil context or invalid options, before any matrix access.
//   - ErrUnsupportedSolver: requested ThinSVD without an implemented SVD kernel.
//   - ErrNilInput, ErrEmptyInput, ErrShape: absent or invalid training dimensions.
//   - ErrNonFinite: NaN or infinity in the observations.
//   - ErrZeroVariance: a constant training dataset or forbidden standardization column.
//   - ErrRankDeficient: requested components exceed available numerical rank.
//   - ErrOverflow: checked dimensions or unrepresentable numeric output.
//   - ErrNumerical: corrupted covariance or failed spectral integrity checks.
//   - ErrNotConverged: exhausted Jacobi budget, with the matrix root cause retained.
//   - ctx.Err(), wrapped matrix/stats errors: cancellation and dependency failures.
//
// Determinism:
//   - Observations and columns scan in ascending order; Jacobi selects max pivots
//     in the matrix kernel's fixed scan order; equal eigenvalues keep solver order.
//   - Each retained loading has a positive greatest-absolute coordinate; ties
//     choose the lowest feature index, without claiming a unique repeated basis.
//
// Complexity:
//   - Time O(n*d*d + rotations*d*d + d*d*d + n*d*k), space O(n*d+d*d+n*k)
//     including preparation, covariance/eigen matrices, and residual work.
//
// Notes:
//   - Fit does not retain x or a pointer to any caller-provided buffer.
//   - CovarianceEigen is unsuitable for highly ill-conditioned designs where
//     direct SVD is needed to distinguish small singular directions reliably.
//   - The current matrix eigensolver has no in-kernel cancellation hook.
//
// AI-Hints:
//   - Do not choose Components or RankTolerance using the final holdout batch.
//   - Prefer *matrix.Dense for efficient access, but never use private Dense data.
//   - Do not replace failed covariance or eigensystem checks with Symmetrize or clip.
func Fit(ctx context.Context, x matrix.Matrix, opts Options) (*FitResult, error) {
	// Context precedes options, which precede all input state by contract.
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	if err := checkOptions(opts); err != nil {
		return nil, err
	}

	// Validate all size arithmetic before reading adversarial or very large matrices.
	n, d, err := checkedShape(x, 2)
	if err != nil {
		return nil, err
	}
	if _, err = checkedFloatElements(d, d); err != nil {
		return nil, err
	}
	if _, err = checkedRotations(d, opts.MaxRotations); err != nil {
		return nil, err
	}

	// A non-finite cell outranks a domain/rank failure in the same valid shape.
	if err = scanFinite(ctx, x, n, d); err != nil {
		return nil, err
	}
	if opts.Components > min(n-1, d) {
		return nil, ErrRankDeficient
	}

	// A single training kernel owns all fit mathematics and atomic publication.
	return fitKernel(ctx, x, n, d, opts)
}

// FitDefault trains centered PCA with the default covariance solver and full rank.
// It is precisely Fit(ctx, x, Options{}) and retains all spectral diagnostics.
//
// Implementation:
//   - Stage 1: construct the zero Options value (no standardization).
//   - Stage 2: let Fit validate inputs and run the canonical covariance kernel.
//   - Stage 3: return the original result and error, without adaptation.
//
// Behavior highlights:
//   - The numerical rank cutoff defaults to epsilon*max(n,d).
//   - Zero Components means the determined rank, not a fixed component count.
//
// Inputs:
//   - ctx: non-nil active context; x: finite, read-only n-by-d training matrix.
//
// Returns:
//   - *FitResult: full numerical-rank fitted model; nil on error.
//
// Errors:
//   - Exactly the applicable Fit errors for Options{}, including cancellation.
//
// Determinism:
//   - Identical to Fit(ctx, x, Options{}) for equal input values.
//
// Complexity:
//   - Time O(n*d*d+rotations*d*d+d*d*d), space O(n*d+d*d).
//
// Notes:
//   - The returned components may be fewer than d if the data are rank deficient.
//
// AI-Hints:
//   - Use FitWithComponents to bound downstream latent feature dimensions.
//   - Use Fit with Options for explicit EigenTolerance and MaxRotations control.
func FitDefault(ctx context.Context, x matrix.Matrix) (*FitResult, error) {
	return Fit(ctx, x, Options{})
}

// FitWithComponents trains centered PCA and requests an exact latent dimension.
// A request greater than the detected rank fails instead of inventing zero factors.
//
// Implementation:
//   - Stage 1: copy the supplied component count into Options.
//   - Stage 2: Fit validates k and computes the unstandardized covariance spectrum.
//   - Stage 3: return the fitted model and full-spectrum-based variance ratios.
//
// Behavior highlights:
//   - components=0 means the full numerical rank; negative values are invalid.
//   - Variance ratios are divided by total training variance, not retained variance.
//
// Inputs:
//   - ctx: non-nil context; x: finite n-by-d training matrix, n>=2 and d>=1.
//   - components: number k of retained directions, 0 for detected rank.
//
// Returns:
//   - *FitResult: model with exactly k columns (or detected rank when k=0).
//
// Errors:
//   - ErrInvalidOption: negative components or nil context.
//   - ErrRankDeficient: k exceeds centered numerical rank.
//   - Other Fit errors, with the same precedence and dependency roots.
//
// Determinism:
//   - Exactly the fixed traversal and spectral tie-break order of Fit.
//
// Complexity:
//   - Time O(n*d*d+rotations*d*d+d*d*d+n*d*k); space O(n*d+d*d).
//
// Notes:
//   - The singular-value cutoff of a future direct SVD need not equal this
//     covariance-eigenvalue rank convention.
//
// AI-Hints:
//   - Select k on training or separate validation partitions, never on test data.
//   - Use ReconstructionResidual to assess information lost in training coordinates.
func FitWithComponents(ctx context.Context, x matrix.Matrix, components int) (*FitResult, error) {
	return Fit(ctx, x, Options{Components: components})
}

// FitStandardized trains PCA in population-scaled feature coordinates.
// It freezes training means and scales and keeps the full detected numerical rank.
//
// Implementation:
//   - Stage 1: select Standardize=true and zero retained-components policy.
//   - Stage 2: Fit learns each DDOF=0 scale under stats.RejectConstant.
//   - Stage 3: publish the same canonical Model and spectral diagnostics.
//
// Behavior highlights:
//   - Constant training columns fail; the wrapper never drops them silently.
//   - InverseTransform restores original feature units using training scales.
//
// Inputs:
//   - ctx: active non-nil context; x: finite n-by-d training matrix, n>=2.
//
// Returns:
//   - *FitResult: full-rank projection in standardized optimization coordinates.
//
// Errors:
//   - ErrZeroVariance: any training feature is exactly constant.
//   - Other Fit errors, including wrapped stats sentinels and cancellation.
//
// Determinism:
//   - Training statistics and covariance scans follow the same fixed row/column order.
//
// Complexity:
//   - Time O(n*d*d+rotations*d*d+d*d*d), space O(n*d+d*d).
//
// Notes:
//   - The learned scale is a population standard deviation, not the sample stddev.
//
// AI-Hints:
//   - Do not standardize holdout batches separately; Transform owns that step.
//   - Standardization changes the optimization metric and therefore the PCA basis.
func FitStandardized(ctx context.Context, x matrix.Matrix) (*FitResult, error) {
	return Fit(ctx, x, Options{Standardize: true})
}

// FitStandardizedWithComponents trains population-standardized PCA with k factors.
// It combines explicit latent compression and strict constant-feature rejection.
//
// Implementation:
//   - Stage 1: select Standardize=true and the caller's component count.
//   - Stage 2: delegate all train-only scaling and eigensolving to Fit.
//   - Stage 3: return the canonical result with no modified diagnostics.
//
// Behavior highlights:
//   - components=0 retains full numerical rank; k>rank fails explicitly.
//   - ReconstructionResidual is measured in scaled coordinates, not sensor units.
//
// Inputs:
//   - ctx: non-nil active context; x: finite n-by-d training observations.
//   - components: number of factors k, 0 for the detected rank.
//
// Returns:
//   - *FitResult: fitted model, length-k spectrum, and full-rank diagnostics.
//
// Errors:
//   - ErrInvalidOption or ErrRankDeficient: invalid or unattainable k.
//   - ErrZeroVariance: constant training feature; all other Fit errors are retained.
//
// Determinism:
//   - Identical to Fit with the same Options fields.
//
// Complexity:
//   - Time O(n*d*d+rotations*d*d+d*d*d+n*d*k), space O(n*d+d*d).
//
// Notes:
//   - A retained basis is not an anomaly detector until a threshold is separately calibrated.
//
// AI-Hints:
//   - Compare SPE against validation-only thresholds after fitting; no holdout refit.
//   - Do not describe the retained components as causal or uniquely identified factors.
func FitStandardizedWithComponents(ctx context.Context, x matrix.Matrix, components int) (*FitResult, error) {
	return Fit(ctx, x, Options{Standardize: true, Components: components})
}
