// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gjrgarch defines the immutable model, caller-owned posterior state,
// optimization policy, and detached result artifacts for GJR-GARCH(1,1).
package gjrgarch

// Parameters defines the five coefficients of a stationary GJR-GARCH(1,1) process.
// It models the conditional variance of mean-centered returns, not a price process.
//
// Implementation:
//   - Stage 1: New validates fields.
//   - Stage 2: Canonical kernels consume a copied value.
//   - Stage 3: Models retain only validated scalars.
//
// Behavior highlights:
//   - Omega > 0, Alpha >= 0, Beta >= 0, Alpha+Gamma >= 0, and
//     Alpha+Beta+Gamma/2 < 1 are required; Gamma itself may be negative.
//   - All fields must be finite; Mean is an unconditional location parameter.
//
// Inputs:
//   - Mean: return units; Omega: squared-return units.
//   - Alpha, Gamma, Beta: dimensionless innovation and variance weights.
//
// Returns:
//   - New stores an immutable value copy inside Model.
//
// Errors:
//   - New classifies invalid coefficients with ErrNonFinite, ErrDomain, or ErrOverflow.
//
// Determinism:
//   - Field order is Mean, Omega, Alpha, Gamma, Beta throughout ID and gradient.
//
// Complexity:
//   - Space O(1); parameter construction and copies O(1).
//
// Notes:
//   - A Parameters value never contains mutable or aliased storage.
//   - Stationarity assumes symmetric standardized innovations.
//
// AI-Hints:
//   - Do not impose Gamma >= 0: only the negative-shock coefficient must be nonnegative.
//   - Scale Mean by a and Omega by a*a when converting return units.
type Parameters struct {
	// Mean is the constant conditional location mu in the same units as the observed return.
	// Every innovation uses observedReturn-Mean; it does not represent an annualized drift.
	Mean float64
	// Omega is the strictly positive conditional variance intercept, in squared-return units.
	// It controls the stationary level together with the persistence rho.
	Omega float64
	// Alpha weights the prior squared innovation whenever it is nonnegative.
	// Alpha must be nonnegative even if Gamma makes downside response smaller.
	Alpha float64
	// Gamma changes the prior squared-innovation coefficient for strictly negative shocks.
	// Gamma may be negative provided Alpha+Gamma remains nonnegative.
	Gamma float64
	// Beta weights the prior conditional variance h[t] in the next-period recurrence.
	// Beta is nonnegative and enters stationary persistence alongside both shock weights.
	Beta float64
}

// State stores the posterior innovation and the variance assigned to its observation.
// The next variance uses this state's residual, never the next observed return.
//
// Implementation:
//   - Stage 1: InitialState binds the first observed return to a supplied h[0].
//   - Stage 2: Step advances h[t+1] before recording the incoming residual.
//   - Stage 3: The caller owns each independent state value.
//
// Behavior highlights:
//   - Steps >= 1 identifies the number of already-consumed observations.
//   - ModelID prevents using the state with a different model or numeric policy.
//
// Inputs:
//   - ModelID: policy-bound identifier; LastResidual/LastVariance: same-index finite values; Steps>=1.
//
// Returns:
//   - State is a copied value, never a shared live model buffer.
//
// Errors:
//   - Methods reject malformed or cross-model state with ErrStateVersion.
//
// Determinism:
//   - One ascending observation transition per Step call.
//
// Complexity:
//   - Space O(1), update time O(1), no dynamic array retention.
//
// Notes:
//   - The source sequence offset and checkpoint authentication belong to the caller.
//
// AI-Hints:
//   - A zero State is invalid; initialize it explicitly from the first observation.
//   - Copy State values freely, but serialize access to a mutable state variable.
type State struct {
	// ModelID binds this value to the immutable parameters and numerical-policy revision.
	// It rejects accidental cross-model continuation but does not authenticate stored bytes.
	ModelID string
	// LastResidual is epsilon[t]=r[t]-Mean for the latest consumed observation.
	// Its sign determines the asymmetric coefficient at the next transition only.
	LastResidual float64
	// LastVariance is positive finite h[t] assigned to the same observation as LastResidual.
	// It is not the next predicted variance; obtain that through ForecastNextVariance.
	LastVariance float64
	// Steps counts observations incorporated since initialization, starting at one.
	// It is not a timestamp, wall-clock duration or independently authenticated feed cursor.
	Steps uint64
}

// Options configures deterministic, constrained Gaussian QMLE estimation.
// The caller must explicitly provide Initial; there is no random or data-dependent start.
//
// Implementation:
//   - Stage 1: PrepareOptions validates the explicit start and substitutes zero-valued control defaults.
//
// Behavior highlights:
//   - Zero MaxIterations selects 200, zero Tolerance selects 1e-7, and zero
//     VarianceFloor selects 1e-12 squared-return units.
//   - Initial must be a feasible stationary model with Omega >= VarianceFloor.
//   - The optimizer uses a stricter stationarity margin than New.
//
// Inputs:
//   - MaxIterations: nonnegative count; Tolerance: nonnegative; VarianceFloor: squared-return floor; Initial: mandatory Parameters.
//
// Returns:
//   - A validated options value is copied into the fit pipeline.
//
// Errors:
//   - Invalid controls/Initial return ErrInvalidOption (with preserved parameter root where relevant).
//
// Determinism:
//   - Options are assembled before any observation scan and remain fixed within Fit.
//
// Complexity:
//   - Space O(1); assembly O(1).
//
// Notes:
//   - A zero Options struct is invalid because Initial has no default.
//
// AI-Hints:
//   - Never supply the zero Parameters value to request automatic initialization.
//   - Rescale VarianceFloor by a*a when changing the units of returns.
type Options struct {
	// MaxIterations caps accepted projected-BFGS updates; zero selects 200.
	// The initial objective evaluation and line-search trials are not accepted iterations.
	MaxIterations int
	// Tolerance is the nonnegative relative projected-gradient convergence threshold.
	// Zero selects 1e-7; an objective plateau alone cannot certify convergence.
	Tolerance float64
	// VarianceFloor is a hard lower bound on admissible conditional variance, in squared-return units.
	// Zero selects 1e-12; candidates below it are rejected, never repaired by likelihood clipping.
	VarianceFloor float64
	// Initial is the mandatory stationary, floor-feasible starting point for deterministic training.
	// No random, time-based, or fitted-from-data fallback is chosen for zero Initial.
	Initial Parameters
}

// Diagnostics records the path of accepted QMLE candidate points.
// Objective is a detached sequence of negative average Gaussian log-likelihoods.
//
// Implementation:
//   - Stage 1: The optimizer records its valid starting score and appends a score only for an accepted iterate.
//
// Behavior highlights:
//   - Objective[0] is the validated starting candidate; later elements are accepted steps.
//   - Iterations counts accepted moves, not attempted Armijo halvings.
//   - Converged is true only when the projected-gradient condition is satisfied.
//
// Inputs:
//   - Objective: negative average log-likelihoods, beginning at the explicit Initial.
//
// Returns:
//   - Diagnostics.Objective is detached, mutable and caller-owned.
//
// Errors:
//   - Nonconvergence is signaled by the accompanying Fit error, not an error field here.
//
// Determinism:
//   - Objective follows the exact acceptance order and never contains rejected trials.
//
// Complexity:
//   - Storage O(I+1), where I is accepted BFGS iteration count.
//
// Notes:
//   - The caller owns Objective and may mutate it without changing the model.
//   - Exhausted budgets and line-search stalls produce Converged=false.
//
// AI-Hints:
//   - A decreasing Objective history alone does not prove optimizer convergence.
type Diagnostics struct {
	// Iterations is the number of accepted projected-BFGS moves, excluding Armijo trials.
	// A run may exhaust its budget with a valid candidate but without convergence.
	Iterations int
	// Converged is true only after the constrained projected-gradient test succeeds.
	// It is an optimizer status, not a guarantee of predictive suitability or VaR coverage.
	Converged bool
	// Objective records negative average Gaussian quasi-log-likelihood in input return units.
	// Entry zero is Initial; every later entry corresponds to an accepted candidate and is detached.
	Objective []float64
}

// FitResult publishes a validated model and its QMLE diagnostics.
// Nonconverged results are usable candidates, not statistically certified estimates.
//
// Implementation:
//   - Stage 1: Publish restores original return units, validates the Model and copies accepted history.
//
// Behavior highlights:
//   - A non-nil result may accompany ErrNotConverged or late context cancellation.
//   - LogLikelihood is the total Gaussian quasi-log-likelihood for Model's parameters.
//   - Model is immutable; Diagnostics.Objective is caller-owned and detached.
//
// Inputs:
//   - Model: immutable validated candidate; Diagnostics: acceptance history; LogLikelihood: total log density.
//
// Returns:
//   - FitResult is detached except for its safely shared immutable Model pointer.
//
// Errors:
//   - Caller must inspect the error returned by Fit; no Result-owned error exists.
//
// Determinism:
//   - Scalar likelihood and history use the same final accepted candidate.
//
// Complexity:
//   - Storage O(I+1) for diagnostics; IsNil O(1), Clone O(I+1).
//
// Notes:
//   - Converged means optimization criterion met, never empirical risk-model approval.
//
// AI-Hints:
//   - Always inspect both the returned error and Diagnostics.Converged.
//   - Evaluate out-of-sample predictive fit before using variance estimates in risk decisions.
type FitResult struct {
	// Model is the last validated immutable model candidate represented by this result.
	// It can accompany ErrNotConverged or late cancellation and is not automatically approved.
	Model *Model
	// Diagnostics describes the accepted optimization history and constrained convergence status.
	// Its Objective slice is caller-owned and does not share the optimizer workspace.
	Diagnostics Diagnostics
	// LogLikelihood is the total, nonnegated Gaussian quasi-log-likelihood on training data.
	// It equals -T times the final negative average objective up to floating-point rounding.
	LogLikelihood float64
}

// IsNil reports whether the FitResult pointer itself is nil.
// It follows the core.Nilable convention without dereferencing typed-nil receivers.
//
// Implementation:
//   - Stage 1: Compare the receiver pointer directly with nil.
//   - Stage 2: Return the comparison without reading any result fields.
//
// Behavior highlights:
//   - An allocated empty or incomplete FitResult is not a nil pointer.
//
// Inputs:
//   - r: possibly typed-nil FitResult receiver.
//
// Returns:
//   - bool: true exactly when r == nil; an allocated empty result is not nil.
//
// Errors:
//   - None, including on a nil receiver.
//
// Determinism:
//   - Pure O(1) pointer comparison; no shared state or allocation.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Implements core.Nilable-style pointer presence; it does not inspect Model.
//
// AI-Hints:
//   - Do not conflate an allocated incomplete result with a nil pointer.
func (r *FitResult) IsNil() bool { return r == nil }

// Clone returns a detached copy of a FitResult and its mutable diagnostics.
// It shares only the safe immutable Model pointer with the source result.
//
// Implementation:
//   - Stage 1: Handle a nil receiver without dereferencing it.
//   - Stage 2: Copy scalar fields and the immutable Model reference.
//   - Stage 3: Deep-copy Diagnostics.Objective into independently owned storage.
//
// Behavior highlights:
//   - Only the immutable Model pointer is shared.
//
// Inputs:
//   - r: possibly nil source with caller-owned Diagnostics.Objective.
//
// Returns:
//   - *FitResult: nil for a nil receiver, or a distinct value with detached slices.
//
// Errors:
//   - None; no numeric calculations or state validation are performed.
//
// Determinism:
//   - Preserves Objective order and all scalar fields exactly.
//
// Complexity:
//   - Time O(iterations), Space O(iterations).
//
// Notes:
//   - A clone does not produce an authenticated checkpoint or re-fit the model.
//
// AI-Hints:
//   - Do not replace the Objective copy with a slice assignment: it would alias callers.
func (r *FitResult) Clone() *FitResult {
	if r == nil {
		return nil
	}
	clone := *r
	if r.Diagnostics.Objective != nil {
		clone.Diagnostics.Objective = append([]float64(nil), r.Diagnostics.Objective...)
	}
	return &clone
}

// QMLEEvaluation exposes the Gaussian QMLE objective and its analytic gradient.
// It supports independent numerical audits and sensitivity analysis without fitting.
//
// Implementation:
//   - Stage 1: EvaluateGaussianQMLE validates a fixed point and runs the analytic score kernel.
//
// Behavior highlights:
//   - Objective is negative AVERAGE log-likelihood; LogLikelihood is total log-likelihood.
//   - Gradient differentiates Objective in physical parameter coordinates and includes
//     the dependence of h[0] = Omega/(1-rho) on all variance coefficients.
//
// Inputs:
//   - Gradient: five physical-unit derivatives, ordered Mean/Omega/Alpha/Gamma/Beta.
//
// Returns:
//   - An owning, scalar-only result with Observations equal to scored T.
//
// Errors:
//   - Invalid candidates are rejected by EvaluateGaussianQMLE before publication.
//
// Determinism:
//   - Accumulation follows ascending observation order; output gradient order is fixed.
//
// Complexity:
//   - Space O(1) result; computing it costs O(T).
//
// Notes:
//   - The evaluation starts at stationary h[0] and does not continue any previous State.
//
// AI-Hints:
//   - Gradient is not the derivative of total LogLikelihood: that derivative is -T*Gradient.
//   - On an active inequality face use feasible one-sided differences, not central differences.
type QMLEEvaluation struct {
	// Objective is the negative average Gaussian quasi-log-likelihood per observation.
	// Unlike LogLikelihood, it is divided by the number of scored observations.
	Objective float64
	// LogLikelihood is the sum of the ordinary (nonnegated) Gaussian log-density scores.
	// It uses a fresh stationary variance for this fixed parameter evaluation.
	LogLikelihood float64
	// Gradient differentiates Objective, not total LogLikelihood, in physical parameter units.
	// Its five coordinates include the derivative of the stationary initial variance.
	Gradient Parameters
	// Observations is the count of returns included in Objective and LogLikelihood.
	// The evaluation consumes every observation exactly once in ascending input order.
	Observations int
}

// IsNil reports whether an evaluation pointer is nil, including a typed-nil receiver.
// It has no side effects and costs O(1) time and O(1) space.
//
// Implementation:
//   - Stage 1: Compare the evaluation receiver directly with nil.
//
// Behavior highlights:
//   - Zero-value non-nil evaluation is still not a nil pointer.
//
// Inputs:
//   - e: possibly typed-nil QMLEEvaluation receiver.
//
// Returns:
//   - bool: true exactly when e==nil.
//
// Errors:
//   - None, including nil receiver.
//
// Determinism:
//   - No arithmetic or mutable package state is read.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Pointer presence does not validate Objective or Gradient values.
//
// AI-Hints:
//   - Use the error from EvaluateGaussianQMLE to classify evaluation failure.
func (e *QMLEEvaluation) IsNil() bool { return e == nil }

// Model is an immutable, validated GJR-GARCH(1,1) conditional variance model.
// It may safely be shared across goroutines performing independent read-only calls.
//
// Implementation:
//   - Stage 1: New validates stationary coefficients and computes a SHA-256 policy-bound identity.
//
// Behavior highlights:
//   - Private p, id and rho cannot be changed through exported methods.
//
// Inputs:
//   - p: validated Parameters; id: policy-bound identity; rho: symmetric-innovation persistence.
//
// Returns:
//   - *Model: immutable shared inference carrier from New.
//
// Errors:
//   - New rejects invalid input; nil method receivers have documented sentinel/zero behavior.
//
// Determinism:
//   - One stable parameter ordering and fixed SHA-256 byte layout.
//
// Complexity:
//   - Model storage O(1), independent of the observation count.
//
// Notes:
//   - Fields are private to prevent invalidating the stationarity and identity laws.
//   - State values and returned slices are never retained by the model.
//
// AI-Hints:
//   - Use Model.ID to ensure persisted State belongs to exactly this model version.
type Model struct {
	// p contains the validated five-scalar coefficients copied on construction.
	p Parameters
	// id is the deterministic policy-and-parameter digest shared by compatible states.
	id string
	// rho is the validated symmetric-innovation persistence used for expected horizons.
	rho float64
}
