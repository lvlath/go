// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gjrgarch provides stationary GJR-GARCH(1,1) conditional-variance
// modeling, strictly causal filtering, expected-variance forecasting, and
// deterministic constrained Gaussian quasi-maximum-likelihood estimation.
//
// Domain and purpose:
//   - Returns represent uniformly sampled stationary observations in a fixed unit
//     system (for example, decimal log returns), not prices or raw volatility.
//   - Parameters holds the conditional mean, positive intercept, asymmetric shock
//     weights, and lagged variance weight.
//   - Model is validated and immutable; State represents a posterior observation,
//     not an uninitialized or predictive state.
//   - Fit estimates a feasible parameter candidate and reports the optimization
//     history; EvaluateGaussianQMLE evaluates an explicit fixed candidate.
//
// Model law:
//
//	  epsilon[t] = return[t] - Mean
//	  h[t+1] = Omega + (Alpha + Gamma*I(epsilon[t]<0))*epsilon[t]^2 + Beta*h[t]
//
//	- The indicator is zero when epsilon[t] is positive, +0, or -0.
//	- All coefficients are finite, Omega>0, Alpha>=0, Beta>=0,
//	  Alpha+Gamma>=0, and rho=Alpha+Beta+Gamma/2<1.
//	- Gamma MAY be negative. Its signed value affects the difference between
//	  positive- and negative-innovation variance responses.
//	- Rho assumes standardized innovations are symmetric with unit variance,
//	  so E[I(z<0)*z^2]=1/2; skewed innovation laws are outside this package.
//	- The unconditional conditional variance is Omega/(1-rho), provided it is
//	  representable in binary64. New rejects unrepresentable long-run values.
//
// Causal index and state law:
//   - InitialState(r[0], h[0]) stores epsilon[0] and h[0] with Steps=1.
//   - Step(state, r[t+1]) first uses state.epsilon[t] and state.h[t] to calculate
//     h[t+1], then stores epsilon[t+1]=r[t+1]-Mean; the new innovation CANNOT
//     affect the current output variance.
//   - Filter(ctx, returns, h[0]) emits a detached sequence h[0]..h[T-1], with
//     the first output exactly equal to the supplied initialVariance.
//   - FilterStationary selects the long-run h[0] explicitly without looking at
//     the observed return series to estimate initial variance.
//   - A zero State, foreign ModelID, nonfinite state, h<=0 or Steps=0 is invalid;
//     a Steps counter at maximum uint64 cannot be advanced.
//   - A model ID is the lowercase SHA-256 hex digest of a policy-versioned
//     parameter bit sequence. Negative zero is canonicalized for identity only.
//
// Facade and nilability law:
//   - New constructs one immutable model; InitialState, Step, Filter and
//     ForecastVariance expose one canonical lagged recurrence. Point and
//     volatility wrappers delegate to these kernels without changing policy.
//   - Fit and FitWithInitial invoke the same projected QMLE pipeline, with an
//     explicit Initial in both cases. EvaluateGaussianQMLE evaluates a fixed
//     candidate and does not iterate, project or continue a previous State.
//   - FitResult and QMLEEvaluation implement IsNil as pointer presence alone.
//     A non-nil FitResult with Converged=false remains a diagnostic artifact.
//   - Model.ID is a policy-bound SHA-256 compatibility tag, not authentication
//     of an externally stored mutable State or its market-data source offset.
//
// Forecast law:
//   - ForecastVariance requires a valid posterior state and horizon H>=1.
//   - Its first value uses the last known innovation exactly. Horizons k>=2
//     use the expected recurrence h[k]=Omega+rho*h[k-1].
//   - ForecastVolatility returns sqrt of the conditional expected future
//     variances, not E[sqrt(h)] or a full predictive return distribution.
//   - ForecastNextVariance returns the first conditional forecast as a scalar,
//     avoiding the horizon-one slice allocation.
//
// Gaussian QMLE and optimizer law:
//   - Loss is the NEGATIVE AVERAGE conditional Gaussian log-likelihood,
//     0.5/T * sum_t [log(2*pi)+log(h[t])+epsilon[t]^2/h[t]].
//   - Every objective evaluation starts with h[0]=Omega/(1-rho) of THAT
//     candidate. The analytic gradient includes dh[0]/dtheta.
//   - VarianceFloor is a feasibility requirement, never a post-hoc clamp.
//     Fit projects Omega>=VarianceFloor and the nonnegative weight constraints.
//   - Fit accepts only an explicit Options.Initial; it NEVER seeds a random or
//     data-derived initial vector. A zero MaxIterations/Tolerance/VarianceFloor
//     uses the documented defaults (200, 1e-7, 1e-12 respectively).
//   - Feasible variance weights satisfy Alpha>=0, Beta>=0, Alpha+Gamma>=0,
//     and rho<=1-64*machine_epsilon during optimization.
//   - The deterministic active-set KKT projection uses matrix.SolveLeastSquares;
//     the BFGS inverse metric starts from identity and restarts after curvature
//     failures. The Armijo line search tries the full step plus at most 20 halvings.
//   - Success requires the normalized projected-gradient stopping condition;
//     a small loss improvement alone never asserts convergence.
//   - Diagnostics.Objective begins with the initial loss and adds only accepted
//     iterates. Diagnostics.Iterations counts accepted updates, not line trials.
//
// Numeric-unit and scale law:
//   - Mean and each return share one physical unit; Omega, h and VarianceFloor
//     have its square; Alpha, Gamma, Beta and rho are dimensionless.
//   - Rescaling returns by a>0 requires Mean -> a*Mean, Omega -> a*a*Omega,
//     h -> a*a*h and VarianceFloor -> a*a*VarianceFloor, with weights unchanged.
//   - Gaussian negative average loss shifts by log(a) while total log-likelihood
//     shifts by -T*log(a); the analytic gradient uses physical parameter units.
//   - Fit normalizes training returns once into a private fixed unit system,
//     not separately for each line-search candidate; public results are restored
//     to the original return units after final acceptance.
//
// Validation, cancellation, and error law:
//   - Fit validates context, options, nil/empty observations, and nonfinite
//     observations before evaluation, in that order.
//   - Nil contexts return ErrInvalidOption; canceled contexts retain ctx.Err().
//   - ErrNonFinite classifies invalid input, ErrDomain invalid mathematical
//     parameter/state-independent domains, and ErrOverflow nonrepresentable
//     arithmetic. The error sentinel protocol is based on errors.Is.
//   - Step rejects invalid State with ErrStateVersion before inspecting a new
//     return; it never mutates caller state on failure.
//   - Filter and ForecastVariance return nil output on failure/cancellation.
//   - Fit returns nil before any validated initial objective. After that point,
//     an unfinished budget, line-search stall, or cancellation may publish the
//     latest valid model with Converged=false and the corresponding error.
//   - A non-nil FitResult is NOT evidence that the solver converged or that
//     the statistical model is an adequate predictor.
//
// Error precedence and partial publication law:
//   - Fit checks context first, then option fields and Initial, then nil and
//     empty input, then all nonfinite observations, and only then numerical work.
//   - State validation (identity, Steps, finite positive h) precedes Step's
//     new return validation. Overflow from valid finite operands is ErrOverflow,
//     not ErrNonFinite or a successful saturated variance.
//   - ErrNotConverged is distinct from fatal candidate/domain errors: the last
//     *validated* candidate may accompany an unfinished budget or stalled search.
//     On late cancellation the same candidate may accompany ctx.Err(); no error
//     can be interpreted as completed optimization without an explicit nil err.
//
// Ownership and concurrency law:
//   - The model is immutable and supports simultaneous read-only operations.
//   - Each State is copied by value; any changing stream State is owned by the
//     caller, who serializes its own updates.
//   - Input slices are read-only during each call; concurrent caller mutation
//     of input storage is unsupported.
//   - Filter/Forecast slices and Diagnostics.Objective are detached and caller
//     owned; FitResult.Clone copies mutable diagnostic slice storage.
//   - No global RNG, wall clock, network, DB, panic, recover, or reflection is
//     involved in any model or optimizer operation.
//
// Determinism:
//   - Observations are scanned in ascending time order; gradient coordinates
//     follow Mean,Omega,Alpha,Gamma,Beta; inequality active sets ascend by mask.
//   - Matrix pivoting follows the dependency's documented stable QR policy.
//   - BFGS and Armijo trial ordering are fixed and independent of goroutine
//     scheduling for independent operations with immutable data.
//
// Complexity:
//   - New, Step, InitialState, ForecastNextVariance: O(1) time and storage.
//   - Filter and its volatility wrappers: O(T) time and O(T) result storage.
//   - ForecastVariance: O(H) time and O(H) result storage.
//   - EvaluateGaussianQMLE: O(T) time, O(1) auxiliary state.
//   - Fit: O(T+I*(21*T+P)) work and O(T+I) state, where I counts accepted
//     iterations and P is bounded active-set work on a fixed five-dimensional
//     feasible region. Runtime also includes matrix's small-system QR residuals.
//
// Non-goals:
//   - No heavy-tail or skewed-innovation likelihood, stochastic volatility,
//     time-varying mean process, volatility of variance, or parameter uncertainty.
//   - No distribution-correct multi-period VaR/ES, position sizing, automated
//     trades, margin threshold calibration, model approval or backtesting.
//   - Empirical usefulness must be established independently via out-of-sample
//     log-score comparisons and standardized residual checks.
//
// AI-Hints:
//   - Never use epsilon[t+1] to compute h[t+1]; this leaks the future observation.
//   - Never clamp h[t] in likelihood: infeasibility must reject the candidate.
//   - Never force Gamma>=0; only Alpha+Gamma must be nonnegative.
//   - Never infer Fit success solely from a non-nil Model: check error and Converged.
//   - For price series, obtain validated log returns first (for example through
//     series.LogReturns), and skip the transform's invalid warmup positions.
package gjrgarch
