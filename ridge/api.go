// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ridge exposes the canonical regularized least-squares facade and
// policy-specific convenience wrappers. All wrappers delegate to Fit.
package ridge

import (
	"context"

	"github.com/lvlath/go/matrix"
)

// Fit trains a Tikhonov-regularized linear regression using augmented QR.
// It minimizes SSE/(2*n)+Lambda*||theta||^2/2, with no penalty on the intercept.
//
// Implementation:
//   - Stage 1: validate context, options, input presence, and checked dimensions.
//   - Stage 2: validate all observations, build and equilibrate the augmented system, then solve by QR.
//   - Stage 3: restore physical coefficients, verify diagnostics, and publish atomically.
//
// Behavior highlights:
//   - Lambda must be strictly positive; no implicit OLS or pseudoinverse fallback exists.
//   - With NoIntercept, Standardize scales predictors without centering them.
//   - Duplicate and wide predictor matrices are legal when augmented QR has numerical rank.
//   - Penalty-first QR rows and column equilibration preserve the objective and
//     prevent loss of small representable slopes under strong regularization.
//
// Inputs:
//   - ctx: non-nil active context; cancellation propagates as ctx.Err().
//   - x: caller-owned read-only n-by-p observations, n,p>=1; p may exceed n.
//   - y: caller-owned length-n finite responses, in the original target units.
//   - opts: required finite positive Lambda and optional basis/rank policies.
//
// Returns:
//   - *FitResult: independent fitted Model and complete finite diagnostics.
//   - error: non-nil on failure, always accompanied by a nil result.
//
// Errors:
//   - ErrInvalidOption: nil context, invalid Lambda, or invalid RankTolerance.
//   - ErrNilInput, ErrEmptyInput, ErrShape: absent, empty, or incompatible inputs.
//   - ErrNonFinite: NaN or infinity in any supplied observation.
//   - ErrRankDeficient: augmented QR rejects the numerical rank.
//   - ErrOverflow: unrepresentable size, transform, coefficient, or diagnostic.
//   - ctx.Err(), matrix/stats errors: cancellation or retained dependency causes.
//
// Determinism:
//   - All input scans use increasing row and feature indices; QR follows matrix pivot ties.
//   - Model coefficient indices match original feature columns.
//
// Complexity:
//   - Time O((n+p)*q*q+q^3), Space O((n+p)*q+p), q=p or p+1.
//   - The solver also retains O((n+p)*q) internal copies of the QR representation.
//
// Notes:
//   - Neither x nor y is retained; concurrent caller mutation during Fit is unsupported.
//   - The zero Options value is invalid because an explicit Lambda is mandatory.
//
// AI-Hints:
//   - Select ols.Fit explicitly when regularization is not part of the model.
//   - Do not subtract train means at inference: the published beta and b are physical.
func Fit(ctx context.Context, x matrix.Matrix, y []float64, opts Options) (*FitResult, error) {
	if err := validateContext(ctx); err != nil {
		return nil, err
	}
	if err := validateOptions(opts); err != nil {
		return nil, err
	}
	// Dimension and allocation checks precede the data-value scan. This also
	// prevents adversarial Matrix implementations from demanding huge loops.
	n, p, err := validateTrainingShape(x, y)
	if err != nil {
		return nil, err
	}
	if err = checkAugmentedSize(n, p, opts.NoIntercept); err != nil {
		return nil, err
	}
	if err = validateTrainingValues(ctx, x, y, n, p); err != nil {
		return nil, err
	}
	return fitKernel(ctx, x, y, n, p, opts)
}

// FitWithLambda trains a ridge model with an explicit penalty and default basis.
// It fits an unpenalized intercept and leaves predictors unscaled.
//
// Implementation:
//   - Stage 1: assemble Options with the supplied Lambda.
//   - Stage 2: delegate all validation and arithmetic to Fit.
//   - Stage 3: return Fit's complete result without changing diagnostics.
//
// Behavior highlights:
//   - Lambda=0 is rejected; this is not a default or automatic regularization mode.
//
// Inputs:
//   - ctx: active context; x: finite n-by-p predictor matrix; y: finite n-vector.
//   - lambda: strictly positive finite penalty strength in optimization coordinates.
//
// Returns:
//   - *FitResult: fitted affine model or nil on any error.
//
// Errors:
//   - Identical to Fit with Options{Lambda: lambda}.
//
// Determinism:
//   - Exactly the original row/feature and QR pivot ordering of Fit.
//
// Complexity:
//   - Time O((n+p)*(p+1)^2+(p+1)^3), Space O((n+p)*(p+1)).
//
// Notes:
//   - Internal centering of the unscaled design improves affine conditioning.
//
// AI-Hints:
//   - Use Fit when custom RankTolerance or preprocessing is required.
func FitWithLambda(ctx context.Context, x matrix.Matrix, y []float64, lambda float64) (*FitResult, error) {
	return Fit(ctx, x, y, Options{Lambda: lambda})
}

// FitStandardized fits an affine ridge model using training population scales.
// It exposes the common standardized penalty without changing predictor units at inference.
//
// Implementation:
//   - Stage 1: select explicit positive Lambda and Standardize=true.
//   - Stage 2: Fit learns training DDOF=0 scales and solves the augmented system.
//   - Stage 3: publish original-unit beta and intercept via Fit.
//
// Behavior highlights:
//   - The penalty applies to standardized optimization slopes, not physical slopes.
//   - Constant training features use scale=1 and remain in the penalty basis.
//
// Inputs:
//   - ctx: active context; x: finite n-by-p matrix; y: finite length-n responses.
//   - lambda: finite positive regularization strength in standardized coordinates.
//
// Returns:
//   - *FitResult: detached affine model and original-unit residual diagnostics.
//
// Errors:
//   - Identical to Fit with Standardize=true and the supplied Lambda.
//
// Determinism:
//   - Row-major stats accumulation and the QR's deterministic pivot selection.
//
// Complexity:
//   - Time O(n*p+(n+p)*(p+1)^2+(p+1)^3), Space O((n+p)*(p+1)).
//
// Notes:
//   - Population scales are frozen at fit time; held-out data is not refitted.
//
// AI-Hints:
//   - Compare penalties only for models using the same feature coordinate basis.
func FitStandardized(ctx context.Context, x matrix.Matrix, y []float64, lambda float64) (*FitResult, error) {
	return Fit(ctx, x, y, Options{Lambda: lambda, Standardize: true})
}

// FitThroughOrigin fits a ridge model with the physical intercept fixed to zero.
// It preserves the origin constraint by omitting all feature centering.
//
// Implementation:
//   - Stage 1: set Lambda and NoIntercept=true without standardization.
//   - Stage 2: delegate validation and penalized QR computation to Fit.
//   - Stage 3: publish a model whose returned intercept is exactly +0.0.
//
// Behavior highlights:
//   - Predictions at an all-zero feature row are exactly zero when arithmetic succeeds.
//
// Inputs:
//   - ctx: active context; x: finite n-by-p matrix; y: finite length-n responses.
//   - lambda: finite positive penalty strength in original slope coordinates.
//
// Returns:
//   - *FitResult: detached origin-constrained model or nil/error.
//
// Errors:
//   - Identical to Fit with NoIntercept=true and the supplied Lambda.
//
// Determinism:
//   - Same observation and feature order as the canonical Fit.
//
// Complexity:
//   - Time O((n+p)*p^2+p^3), Space O((n+p)*p).
//
// Notes:
//   - Through-origin modeling requires a real physical zero-offset assumption.
//
// AI-Hints:
//   - Do not replace this with post-fit assignment of intercept=0 on an affine model.
func FitThroughOrigin(ctx context.Context, x matrix.Matrix, y []float64, lambda float64) (*FitResult, error) {
	return Fit(ctx, x, y, Options{Lambda: lambda, NoIntercept: true})
}

// FitThroughOriginStandardized fits origin-constrained ridge with scale-only inputs.
// Its training population scales balance the penalty without introducing an intercept.
//
// Implementation:
//   - Stage 1: select NoIntercept and Standardize with a caller-provided Lambda.
//   - Stage 2: Fit learns scales but sets all training coordinate centers to zero.
//   - Stage 3: return original-unit coefficients and the exact-zero intercept.
//
// Behavior highlights:
//   - Unlike an affine fit, no feature means are subtracted in the design.
//
// Inputs:
//   - ctx: active context; x: finite n-by-p matrix; y: finite length-n responses.
//   - lambda: positive finite penalty on scale-only optimization slopes.
//
// Returns:
//   - *FitResult: detached model and full objective diagnostics or nil/error.
//
// Errors:
//   - Identical to Fit with NoIntercept=true, Standardize=true, and Lambda.
//
// Determinism:
//   - Fixed training row order, per-column scales, and QR pivot ties.
//
// Complexity:
//   - Time O(n*p+(n+p)*p^2+p^3), Space O((n+p)*p).
//
// Notes:
//   - The original response is never centered or rescaled.
//
// AI-Hints:
//   - Never call a centered Standardizer.Transform in this origin-constrained mode.
func FitThroughOriginStandardized(ctx context.Context, x matrix.Matrix, y []float64, lambda float64) (*FitResult, error) {
	return Fit(ctx, x, y, Options{Lambda: lambda, NoIntercept: true, Standardize: true})
}
