// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ols defines the canonical training boundary and convenient regression facades.
// This file fixes option selection without introducing a second numerical solver.
package ols

import (
	"context"

	"github.com/lvlath/go/matrix"
)

// Fit trains an ordinary least-squares regression with an explicit full-rank design.
// The resulting affine model uses the original units of x and y, regardless of
// the coordinates chosen for the internal pivoted Householder QR solve.
//
// Implementation:
//   - Stage 1: validate context, rank policy, input presence, sample counts, and finite cells.
//   - Stage 2: build a detached design and delegate exclusively to matrix.SolveLeastSquares.
//   - Stage 3: recover physical coefficients, verify original-coordinate SSE, and publish.
//
// Behavior highlights:
//   - The zero Options value fits an intercept and uses the solver's automatic rank cutoff.
//   - NoIntercept fixes b to positive zero; Standardize then scales but never centers.
//   - Rank deficiency never triggers a pseudoinverse, column dropping, or regularization.
//
// Inputs:
//   - ctx: non-nil active context; cancellation is checked throughout supported stages.
//   - x: caller-owned, read-only n-by-p feature matrix; p>=1 and n>=p+intercept.
//   - y: caller-owned, finite response vector of length n, in physical output units.
//   - opts: value-copy fit policy; RankTolerance is finite, nonnegative, and relative.
//
// Returns:
//   - *FitResult: independently fitted Model, full rank, finite SSE and QR condition estimate.
//   - error: nil on success; otherwise the result is always nil, never partial.
//
// Errors:
//   - ErrInvalidOption: missing context or invalid RankTolerance, before input access.
//   - ErrNilInput, ErrEmptyInput, ErrShape: missing or incompatible observations.
//   - ErrNonFinite: a caller-supplied observation is NaN or infinite.
//   - ErrRankDeficient: QR rejects an insufficiently independent design basis.
//   - ErrOverflow: scaling, coefficient conversion, prediction, or SSE is unrepresentable.
//   - ctx.Err() and wrapped matrix/stats errors: cancellation or dependency failures.
//
// Determinism:
//   - Validation and design scan rows then features ascending; QR pivot ties use matrix's rule.
//   - Coefficients are published in the original feature-column order.
//
// Complexity:
//   - Time O(n*q*q+q*q*q), Space O(n*q), q=p or p+1; n-by-q design is detached.
//
// Notes:
//   - Caller mutation of x or y while Fit runs is unsupported; Fit never retains them.
//   - ConditionEstimate is kappa_1(R) of the transformed QR design, not kappa_2(x).
//   - This method makes no IID inference or statistical significance claims.
//
// AI-Hints:
//   - Do not form, invert, or solve normal equations in this package.
//   - Prefer *matrix.Dense for efficient At/Set, and do not standardize heldout data anew.
func Fit(ctx context.Context, x matrix.Matrix, y []float64, opts Options) (*FitResult, error) {
	n, p, q, err := validateFit(ctx, x, y, opts)
	if err != nil {
		return nil, err
	}
	return fitKernel(ctx, x, y, n, p, q, opts)
}

// FitDefault fits an affine regression with an intercept and unscaled predictors.
// It is a shortcut for Fit(ctx, x, y, Options{}), without different arithmetic.
//
// Implementation:
//   - Stage 1: create the zero-value fit policy.
//   - Stage 2: delegate all validation, QR work, and diagnostics to Fit.
//   - Stage 3: return the same canonical result and error without alteration.
//
// Behavior highlights:
//   - Centering still occurs internally to stabilize the explicit intercept basis.
//
// Inputs:
//   - ctx: active non-nil training context.
//   - x: finite, read-only n-by-p predictors (p>=1, n>=p+1).
//   - y: finite, read-only responses in original physical units, length n.
//
// Returns:
//   - *FitResult: detached full-rank model/diagnostics; nil on any error.
//
// Errors:
//   - Same categories, order, and dependency wrapping as Fit with zero Options.
//
// Determinism:
//   - Exactly the row/column and pivot order of Fit with zero Options.
//
// Complexity:
//   - Time O(n*(p+1)^2+(p+1)^3), Space O(n*(p+1)).
//
// Notes:
//   - This wrapper does not alter rank tolerance or discard diagnostics.
//
// AI-Hints:
//   - Use Fit when explicit numerical rank selection is required.
func FitDefault(ctx context.Context, x matrix.Matrix, y []float64) (*FitResult, error) {
	return Fit(ctx, x, y, Options{})
}

// FitStandardized fits an affine model using population-scaled training predictors.
// It provides the common intercept-enabled fit policy without changing returned units.
//
// Implementation:
//   - Stage 1: select Standardize=true while retaining the default intercept.
//   - Stage 2: let Fit learn DDOF=0 means/scales and solve its centered QR design.
//   - Stage 3: publish original-unit coefficients and diagnostics through Fit.
//
// Behavior highlights:
//   - Exact-constant training columns retain scale=1 and are rejected by QR if dependent.
//
// Inputs:
//   - ctx: active non-nil training context.
//   - x: finite read-only n-by-p training matrix, n>=p+1.
//   - y: finite read-only response vector of length n.
//
// Returns:
//   - *FitResult: full-rank affine model in original units, or nil on error.
//
// Errors:
//   - The exact categories and precedence of Fit with Standardize=true.
//
// Determinism:
//   - Same row-major stats accumulation and pivot-tie ordering as Fit.
//
// Complexity:
//   - Time O(n*(p+1)^2+(p+1)^3), Space O(n*(p+1)).
//
// Notes:
//   - Scales are learned from training data only and never refitted by Predict.
//
// AI-Hints:
//   - This wrapper does not standardize y and is not a regularized regression.
func FitStandardized(ctx context.Context, x matrix.Matrix, y []float64) (*FitResult, error) {
	return Fit(ctx, x, y, Options{Standardize: true})
}

// FitThroughOrigin fits slopes with the physical intercept fixed to exactly zero.
// It is suitable only when the domain truly mandates zero output at zero input.
//
// Implementation:
//   - Stage 1: enable NoIntercept without rescaling.
//   - Stage 2: delegate validation and the p-column QR solve to Fit.
//   - Stage 3: return the original-unit zero-intercept model and diagnostics.
//
// Behavior highlights:
//   - The basis has exactly p columns and receives no hidden centering.
//
// Inputs:
//   - ctx: active non-nil training context.
//   - x: finite read-only n-by-p matrix, n>=p>=1.
//   - y: finite read-only response vector, length n.
//
// Returns:
//   - *FitResult: fitted slopes with intercept=+0, or nil on error.
//
// Errors:
//   - Same categories and error order as Fit with NoIntercept=true.
//
// Determinism:
//   - Training rows and original predictor columns are processed ascending.
//
// Complexity:
//   - Time O(n*p*p+p*p*p), Space O(n*p).
//
// Notes:
//   - This constraint is not synonymous with centering the response.
//
// AI-Hints:
//   - Do not apply this to price-demand models unless the zero-origin law is justified.
func FitThroughOrigin(ctx context.Context, x matrix.Matrix, y []float64) (*FitResult, error) {
	return Fit(ctx, x, y, Options{NoIntercept: true})
}

// FitThroughOriginStandardized fits a scale-only, zero-intercept regression.
// Population scales improve predictor balance without adding an implicit constant.
//
// Implementation:
//   - Stage 1: explicitly select NoIntercept and Standardize together.
//   - Stage 2: delegate train-only scales and the uncentered QR design to Fit.
//   - Stage 3: publish the resulting exactly zero-intercept coefficients.
//
// Behavior highlights:
//   - Original-unit predictions remain y-hat = sum_j x_j*beta_j.
//
// Inputs:
//   - ctx: non-nil training context.
//   - x: finite read-only n-by-p predictors, n>=p>=1.
//   - y: finite read-only response vector of length n.
//
// Returns:
//   - *FitResult: model with intercept=+0, or nil on error.
//
// Errors:
//   - The same sentinels, dependency roots, and precedence as the matching Fit call.
//
// Determinism:
//   - Same ascending row/feature order and QR pivot policy as Fit.
//
// Complexity:
//   - Time O(n*p*p+p*p*p), Space O(n*p).
//
// Notes:
//   - Only the feature scale, never the feature mean, enters the design matrix.
//
// AI-Hints:
//   - Replacing scale-only by Standardizer.Transform here introduces a false intercept.
func FitThroughOriginStandardized(ctx context.Context, x matrix.Matrix, y []float64) (*FitResult, error) {
	return Fit(ctx, x, y, Options{NoIntercept: true, Standardize: true})
}
