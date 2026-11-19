// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package arima estimates and forecasts fixed-order, univariate ARIMA(p,d,q)
// processes under a conditional Gaussian innovation model. It separates an
// explicitly specified ARMA law on d-th differences from the historical level
// anchors needed to reconstruct forecasts in the original measurement units.
//
// Domain and supported operations:
//   - Order fixes AR p and MA q in [0,8] and differences d in [0,2]. Order{}
//     denotes nonseasonal white noise, not automatic order identification.
//   - Fit learns physical AR/MA coefficients, differenced-space intercept c and
//     positive innovation variance from a finite, ordered scalar sample.
//   - FitWithDrift, FitWithoutDrift, FitARMA and FitIntegrated are explicit
//     convenience policies over the same Fit kernel; none changes the estimand.
//   - New validates complete physical Parameters and owns an immutable snapshot.
//     ID fingerprints the model method, order and parameters; Parameters copies
//     both coefficient vectors rather than exposing model-owned storage.
//   - Forecast and ForecastNext condition on the history supplied to that call,
//     publishing detached future levels and marginal innovation variances.
//
// Mathematical law and parameter interpretation:
//   - Let y[t]=Delta^d x[t]. The signed recurrence is
//     y[t]=c+sum(phi[i]*y[t-i])+epsilon[t]+sum(theta[j]*epsilon[t-j]).
//   - Drift is c in the differenced equation. For a stationary differenced
//     ARMA component its unconditional mean, when defined, is
//     c/(1-sum(phi)). Therefore c is not generally long-run level growth.
//   - With IncludeDrift=false, Fit fixes c to positive zero; New accepts the
//     caller's explicit c without interpreting any IncludeDrift policy.
//   - AR polynomial 1-sum(phi[i]*z^i) and MA polynomial 1+sum(theta[j]*z^j)
//     must both have roots strictly outside the unit circle. Schur step-down
//     rejects unit-circle boundaries and nonstable/noninvertible coefficients;
//     the optimizer never repairs a candidate by clipping its reflections.
//
// Conditional likelihood law:
//   - Let N=len(values)-d and b=max(p,q). The first b differenced samples are
//     conditioning observations; epsilon[t]=0 for t<b, and these residuals
//     contribute nothing to the likelihood or its variance estimate.
//   - For t>=b, epsilon[t] is the observed y[t] minus c, lagged AR terms and
//     lagged MA innovations, in ascending sample and lag order.
//   - EffectiveN=N-b must exceed p+q+I(IncludeDrift)+1 for Fit admission.
//   - Profile sigmaSquared=SSE/EffectiveN and minimize the negative *total*
//     conditional Gaussian log likelihood. An exactly zero effective SSE is
//     ErrSingular; this method does not admit deterministic zero-noise models.
//   - With p=q=0, fitting the constant/no-constant white-noise law is analytic.
//     Otherwise bounded deterministic L-BFGS uses tanh reflection coordinates,
//     analytic residual derivatives, and an accepted-step objective trajectory.
//   - Diagnostics.Objective[0] is the initial loss. Iterations counts accepted
//     updates; a finite candidate with ErrNotConverged is not a successful fit.
//
// Forecast and uncertainty law:
//   - A Model does not retain a mutable training cursor or forecasting origin.
//     Forecast recomputes residuals from the caller's complete observed prefix
//     using exactly the conditional burn-in and sign conventions of Fit.
//   - Future innovations have conditional expectation zero; known historical
//     innovations continue to enter MA terms until they leave the lag window.
//   - For d=1, integration begins at the last *actual* level x[n-1]. For d=2,
//     integration begins at x[n-1] and the last actual first difference
//     x[n-1]-x[n-2]. Neither integration starts at a fabricated zero origin.
//   - The stationary impulse response psi has psi[0]=1 and
//     psi[j]=theta[j]+sum(phi[i]*psi[j-i]) for valid positive lags. One or two
//     successive cumulative sums produce integrated response Psi for d=1/2.
//   - The marginal horizon-h variance is sigmaSquared*sum(Psi[j]^2), j=0..h-1.
//     The package never allocates an H-by-H forecast covariance matrix and
//     excludes fitted-parameter uncertainty and unmodeled future exogenous risk.
//
// Input, option and numeric laws:
//   - Fit requires finite, non-nil, nonempty, equally spaced observations in
//     time order; the caller establishes cadence and handles missing timestamps.
//   - Forecast requires at least d+max(p,q)+1 observed levels and horizon >=1.
//     The stricter Fit effective-sample threshold does not apply to Forecast.
//   - Public context calls reject nil context with ErrInvalidOption. An already
//     canceled context returns ctx.Err() before inspecting options or data;
//     lengthy loops perform cooperative cancellation checks during computation.
//   - Fit validates options before presence, sample length and finiteness;
//     Forecast validates model/horizon before input presence, length and finiteness.
//     New validates parameter shape, finiteness, variance and roots in that order.
//   - NaN/Inf in caller data is ErrNonFinite. Arithmetic overflow in differences,
//     conditional scores, integration or published variance is ErrOverflow.
//   - ErrInvalidOption, ErrNilInput, ErrEmptyInput, ErrInsufficientData,
//     ErrNonFinite, ErrNonStationary, ErrNonInvertible, ErrSingular,
//     ErrOverflow and ErrNotConverged are classified using errors.Is.
//   - Required allocation-size products are checked before allocation; this is
//     not an operating-system memory-availability or arbitrary-precision guarantee.
//
// Result, ownership, cancellation and concurrency laws:
//   - Input slices are read-only and never retained. Returned ForecastResult
//     Mean/Variance and FitResult.Diagnostics.Objective slices are caller-owned;
//     Model.Parameters returns independently copied AR and MA arrays.
//   - FitResult and ForecastResult implement pointer-only core.Nilable semantics:
//     IsNil is true exactly when the receiver pointer itself is nil.
//   - Models are immutable and may serve independent concurrent read-only
//     Forecast calls. A caller must not mutate any input slice during a call.
//     Forecast uses a separate workspace and does not alter Model or history.
//   - Forecast publishes nil on every error, including cancellation. Fit may
//     return a last-accepted candidate *only* with ErrNotConverged; all other
//     errors, including context cancellation, publish a nil result.
//   - Repeated calls traverse samples, lags, optimizer correction history and
//     future horizons in defined ascending/deterministic order. No RNG, clock,
//     global mutable solver state or network operation affects the result.
//
// Complexity and performance:
//   - New: O(p^2+q^2) time, O(p+q) additional space for Schur checks/copies.
//   - Fit: O((I+1)*N*(p+q+1)^2) time with a bounded line-search factor,
//     O(N*(p+q+1)+I) additional space for residual sensitivities/history;
//     I counts accepted L-BFGS steps and N is the differenced sample count.
//   - Forecast: O((N+H)*(p+q+d+1)) time and O(N+H) additional space for a
//     history of length N and horizon H. Impulse-response variance alone uses
//     O(H*(p+q+d+1)) time and O(H) space, never a dense horizon covariance.
//   - ForecastNext is Forecast with H=1; it still recomputes the supplied
//     history's innovations rather than carrying a mutable streaming cursor.
//
// Non-goals and AI-Hints:
//   - No automatic/seasonal order selection, exogenous regressors, missing-data
//     repair, exact diffuse Kalman MLE, online retraining, predictive-skill
//     certification, parameter-uncertainty intervals or operational risk policy.
//   - Do not confuse conditional Gaussian estimation with an exact diffuse
//     likelihood: its initial residuals, effective sample and estimand differ.
//   - Do not run a model with Difference>0 on data already differenced by series;
//     it would integrate from the wrong physical origins and estimate another law.
//   - Do not interpret Drift as unconditional mean with AR terms, nor integrate
//     future levels without preserving x[n-1] and the last slope for d=2.
//   - Do not treat IsNil, a decreasing objective, or the Model.ID hash as a
//     model-deployment approval or evidence of authenticated provenance.
//   - Validate training/holdout timing independently. A backtest must forecast
//     at each prefix before exposing the next observed value to the caller.
package arima
