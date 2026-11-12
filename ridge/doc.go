// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ridge implements deterministic Tikhonov L2-regularized linear regression
// through an explicitly augmented, column-pivoted Householder QR system.
// It supports finite rectangular predictors, including duplicated columns and p>n.
//
// Domain and public carriers:
//   - Fit is the single canonical training facade; convenience facades only select Options.
//   - Options fixes positive Lambda, intercept, scaling, and relative rank policy.
//   - Model owns frozen physical-unit coefficients and training transform metadata.
//   - FitResult publishes a complete Model, SSE, penalty, and normalized objective.
//
// Mathematical contract:
//   - Minimize SSE/(2*n) + Lambda*sum_j(theta_j^2)/2, Lambda finite and >0.
//   - theta represents optimization slopes, not the unpenalized intercept.
//   - With intercept, the augmented design is [1, Z; 0, sqrt(n*Lambda)*I_p].
//   - With NoIntercept, the augmented design is [Z; sqrt(n*Lambda)*I_p].
//   - Both right-hand sides are [y; 0_p]. The penalized column index always
//     skips the intercept when an intercept exists.
//   - Before the QR solve, an orthogonal row permutation moves the penalty
//     rows to the front. A positive diagonal column change places normalized
//     slopes before the intercept. With C the fixed column permutation and
//     D the positive diagonal of column divisors, the QR system is
//     P*A*C*D^-1 with RHS P*yAug. Original coefficients are C*D^-1*gamma.
//   - These two transformations preserve the exact least-squares objective:
//     ||P*(A*C*D^-1*gamma-yAug)||^2 = ||A*C*D^-1*gamma-yAug||^2.
//     They improve strongly regularized fits by preventing Householder RHS
//     cancellation from erasing tiny, representable nonzero slopes.
//   - With Standardize, each training scale is a population DDOF=0 standard
//     deviation; exact-constant features have scale=1 and remain in the basis.
//   - Intercept models center predictors; origin-constrained models use scale-only
//     normalization, so their physical intercept remains exactly +0.0.
//   - FitResult.Penalty is in optimization coordinates. Coefficients and Predict
//     return physical units after beta=theta/scale and intercept correction.
//
// Shape and numeric law:
//   - Fit accepts n,p>=1 and len(y)=n; p may exceed n; all cells must be finite.
//   - Lambda must be strictly positive and finite. Its zero value is NOT OLS.
//   - RankTolerance is a finite nonnegative relative QR cutoff; zero selects
//     matrix's automatic dimension-scaled machine-epsilon policy.
//   - Positive Lambda guarantees algebraic slope regularity, but a very small
//     representable penalty can still fail the selected numerical QR rank test.
//   - RankTolerance governs pivoted QR of the equilibrated augmented matrix;
//     it is a numerical cutoff, not an exact rank certificate for original X.
//   - Very large finite Lambda is not automatically rounded into an infinite
//     weight or a fixed zero coefficient. Float64 representability still limits
//     recoverable slopes; no alternate estimator is selected silently.
//   - The diagonal weight is formed as sqrt(n)*sqrt(Lambda); never multiply
//     n*Lambda first. Impossible dense byte sizes fail before data scans.
//   - The SSE, penalty, objective, and physical model coefficients are each
//     checked for finite float64 representation before result publication.
//   - Predict accepts an exactly dimensioned 0-by-p Matrix and returns an
//     independent non-nil empty result. Other zero or negative feature shapes fail.
//
// Error precedence and cancellation:
//   - Fit: context -> options -> nil/typed-nil -> size/shape -> finite inputs
//     -> transform/augmented rank -> numerical computation and diagnostics.
//   - Predict: context -> model/matrix presence -> size/shape -> finite inputs
//     -> physical-unit calculation. Custom Matrix.At causes are preserved.
//   - ErrInvalidOption, ErrNilInput, ErrEmptyInput, ErrShape, ErrNonFinite,
//     ErrRankDeficient, and ErrOverflow are package-owned sentinel identities.
//   - Check errors using errors.Is; dependency matrix/stats causes are retained
//     when translating failures into corresponding ridge-level sentinels.
//   - A nil context returns ErrInvalidOption; canceled contexts return ctx.Err().
//   - Fit and Predict never return partial results on any error or cancellation.
//
// Ownership, determinism, and concurrency:
//   - Neither Fit nor Predict mutates or retains caller-provided matrices or y.
//   - The returned Model keeps private copies of slopes and training transforms;
//     Coefficients always copies its returned slope slice.
//   - Model has no exported mutators. Concurrent calls to Predict and Coefficients
//     are safe, provided the caller does not mutate input matrices concurrently.
//   - Validation is row-major; learning fitted means uses fixed ascending index
//     order. QR preparation moves the p penalty rows before the n observation
//     rows, and places original-order slopes before the optional intercept.
//     Pivoted QR breaks equal trailing-norm ties by this fixed column order.
//   - Inference uses compensated left-to-right summation. When an intermediate
//     overflow could cancel, a rare exact binary-rational path determines whether
//     the final prediction is representable. This does not change the model.
//
// Complexity and memory:
//   - q=p with NoIntercept and q=p+1 otherwise.
//   - Fit time O((n+p)*q^2+q^3+n*p), with O((n+p)*q+p) storage per
//     canonical/equilibrated design plus O((n+p)*q) matrix solver copies.
//     Preconditioning uses one extra O((n+p)*q) allocation and pass; profiling
//     it separately from Householder factorization is important for wide p.
//   - Predict time O(m*p), output O(m), and O(1) scratch on the ordinary path;
//     the rare exact-arithmetic overflow path uses multiprecision temporary state.
//   - Checked sizes rule out integer arithmetic overflow, not physical RAM exhaustion.
//
// Non-goals:
//   - No automatic Lambda choice, cross-validation, feature dropping, sample weights,
//     elastic-net/L1 penalty, SVD pseudoinverse, multivariate response, Bayesian
//     posterior, iid significance intervals, or hidden OLS/normal-equations fallback.
//   - Model coefficients are statistical estimates, not automatically calibrated
//     financial hedges, sensor safety thresholds, or causal effects.
//
// AI-Hints:
//   - If Lambda=0 is required, call the separate ols package explicitly.
//   - Do not penalize the affine column, standardize y, or center an origin model.
//   - Do not replace the population-scale policy with sample DDOF=1 statistics.
//   - Use errors.Is for every sentinel, never compare error messages.
//   - *matrix.Dense offers constant-time At/Set, but has no public raw-data
//     accessor; do not bypass its private storage or promise an unsafe fast-path.
//   - Do not remove penalty-first equilibration to recover one allocation:
//     huge but finite Lambda can otherwise make the computed slope exactly -0.
//   - The QR right scaling is NOT predictor Standardize: it is undone before
//     physical coefficients and the explicitly scaled-L2 penalty are published.
package ridge
