// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ols implements ordinary least squares over finite rectangular
// matrices by delegating to matrix's column-pivoted Householder QR solver.
// It publishes immutable, original-unit regression models and explicit
// numerical diagnostics rather than significance tests or pseudo-inverses.
//
// # Domain and mathematical objective
//
// The input matrix X has n observation rows and p predictor columns; y contains
// n responses. Fit minimizes the unweighted sum of squared residuals
//
//	SSE(beta,b) = sum_i (y[i] - b - sum_j X[i,j]*beta[j])^2.
//
// An intercept is enabled by default. With Options.NoIntercept, b is fixed to
// exactly positive zero, not merely omitted from the result. The number of
// unknowns q is p+1 with an intercept, and p without; n>=q and p>=1 are required.
// The design must have full *numerical* column rank according to the matrix QR
// tolerance, otherwise Fit returns ErrRankDeficient and a nil result. An n=q
// full-rank system is admissible despite having no residual degrees of freedom.
//
// # Design matrix and training-only transformations
//
// With an intercept, the design begins with an explicit column of ones. Each
// predictor is centered using its training-column mean, even when Standardize
// is false; Standardize additionally divides the centered values by fitted
// population standard deviations (DDOF=0). With NoIntercept the predictors
// are never centered: Standardize performs only x/scale, keeping b=+0.
//
// Training scales come from stats.FitStandardizer with UnitScaleConstant.
// Exactly constant features therefore receive scale one; they are never
// silently dropped. A constant predictor alongside an intercept has deficient
// design rank; a nonzero constant predictor *without* an intercept can be a
// legitimate single independent column.
//
// For a centered design, a fitted QR slope theta[j] is mapped into physical
// units as beta[j]=theta[j]/scale[j], and the original-unit intercept is
// b=theta[0]-sum_j mean[j]*beta[j]. The origin-constrained design uses b=+0
// and the same slope conversion without the centering correction. Both
// transformations retain their training parameters; Predict does not fit,
// update, center, or rescale its input batch.
//
// # Numerical stability and solver boundaries
//
// matrix.SolveLeastSquares performs column-pivoted Householder QR. This avoids
// the condition-number squaring associated with normal equations X^T*X.
// No production path forms a Gram matrix, calls an inverse, uses a pseudoinverse,
// or substitutes a different solver upon rank loss. Options.RankTolerance is a
// relative pivoted-R cutoff; zero selects matrix's dimension-scaled default.
//
// FitResult.Rank equals q on success. ConditionEstimate inherits matrix's
// measured induced 1-norm condition number of the *transformed, pivoted R*.
// It is neither kappa_2(X) nor a bound on coefficient error. No condition limit
// is imposed by ols; callers must assess high condition estimates explicitly.
// ResidualSumSquares is recomputed from predictions in original physical units,
// using a scaled sum-of-squares accumulation and rejecting unrepresentable
// final SSE. It is not the QR residual norm, a mean square, or R-squared.
//
// IEEE-754 arithmetic is approximate. QR recovery and FMA prediction can yield
// a few ULPs of deviation from an exact mathematical response. Use mixed
// absolute/relative numerical comparisons for coefficients and predictions;
// reserve exact equality for dimensions, presence, error identity, explicit
// zero intercept, and ownership guarantees. No universal correctly-rounded
// least-squares solution or statistical inference guarantee is claimed.
//
// # Validation and error law
//
// New Fit calls validate in the order context -> options -> nil/typed-nil ->
// shape/size -> nonfinite observations -> domain/rank -> computation. Nil
// context gives ErrInvalidOption; canceled context propagates ctx.Err().
// Nil x/y gives ErrNilInput; zero-extent training inputs ErrEmptyInput;
// incompatible response length, insufficient rows, and invalid dimensions
// ErrShape. NaN or infinity in x/y gives ErrNonFinite. Rank loss yields
// ErrRankDeficient. Unrepresentable transformations, coefficients, predictions,
// and SSE yield ErrOverflow. Wrapped dependency errors preserve their original
// errors.Is identities, including matrix/stats and cancellation roots.
//
// Predict validates context, fitted receiver, matrix presence, matching feature
// count, and finite observations before evaluating rows. For a correctly shaped
// 0-by-p inference matrix it returns a non-nil empty slice. A nil Model
// Predict call returns ErrNilInput; a nil Model Coefficients call yields
// (nil,0) without panic. On every Fit or Predict error the result is nil,
// never a partially published model or partial prediction slice.
//
// # Ownership, mutation, and concurrency
//
// X and y are read-only for the duration of Fit and are never retained.
// Successful Model values own private coefficients and train transformation
// arrays. Coefficients allocates a detached slope slice; replacing the public
// FitResult.Model pointer cannot mutate a previously retained Model.
// Predictions allocate independent output slices and read private Model fields
// only, so concurrent calls on the *same fitted Model* are safe when each
// caller's input matrix is stable for the duration of the call. The package
// does not snapshot live Matrix or View inputs; concurrent mutation of inputs
// is unsupported. Exposed FitResult fields are caller-mutable diagnostics,
// not synchronized shared mutable state.
//
// The pointer method (*FitResult).IsNil implements core.Nilable and reports
// pointer presence only. A non-nil zero-value FitResult is present but is not
// evidence of successful training. Always inspect Fit's error before using it.
//
// # Determinism and complexity
//
// Validation visits observations in increasing row order and predictors in
// increasing column order. QR pivots follow matrix's deterministic original
// column-index tie-break. Original-unit coefficients preserve caller feature
// order. Predictions visit rows then columns in ascending order. Repeated
// calls on identical stable input, options, toolchain, and arithmetic platform
// produce reproducible results; concurrent input mutation is excluded.
//
// A q-column design requires O(n*q) storage and is solved in
// O(n*q*q+q*q*q) floating-point arithmetic (including condition measurement).
// Fit additionally performs O(n*p) validation/scaling and O(n*p) residual
// prediction; matrix's exact-binary residual diagnostics may require extra
// multiprecision work. Predict is O(m*p) time and O(m) output space for m
// inference rows. No hidden second module or graph dependency is introduced.
//
// # Non-goals and usage guidance
//
// The package does not perform ridge or lasso regularization, weighted or
// robust regression, imputation, automatic feature selection, SVD-based
// minimum-norm solutions, recursive online updates, cross-validation,
// autocorrelation correction, p-values, F tests, or confidence intervals.
// Econometric data frequently violate IID Gaussian residual assumptions.
// Train/holdout splitting, timestamp validity, feature-order provenance,
// drift monitoring, and downstream error governance belong to the caller.
//
// AI-Hints:
//   - Do not replace QR with normal equations or alter rank tolerance to hide
//     a genuinely redundant feature. Fix the feature schema explicitly.
//   - Never center predictors under NoIntercept: the transformation creates
//     a hidden affine offset even when the returned intercept is set to zero.
//   - Use stats scaling during Fit only; Predict in original units uses the
//     frozen coefficients and must not learn holdout means or variances.
//   - Do not assert exact floating-point recovery from Householder QR; compare
//     with numerical budgets while preserving strict structural assertions.
//   - Interpret SSE and kappa_1(R) as numerical diagnostics, not as evidence
//     of causal effects or statistical significance.
//
// See docs/OLS.md, docs/TUTORIAL.md, and ols/example_test.go for mathematics,
// API usage, realistic calibration scenarios, and regression pitfalls.
package ols
