// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gjrgarch defines the immutable model, caller-owned posterior state,
// optimization policy, and detached result artifacts for GJR-GARCH(1,1).
package gjrgarch

// Parameters defines the five coefficients of a stationary GJR-GARCH(1,1) process.
// It models the conditional variance of mean-centered returns, not a price process.
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
// Notes:
//   - A Parameters value never contains mutable or aliased storage.
//   - Stationarity assumes symmetric standardized innovations.
//
// AI-Hints:
//   - Do not impose Gamma >= 0: only the negative-shock coefficient must be nonnegative.
//   - Scale Mean by a and Omega by a*a when converting return units.
type Parameters struct {
	Mean  float64 // Conditional mean mu, in the same units as each return.
	Omega float64 // Positive variance intercept, in squared-return units.
	Alpha float64 // Nonnegative squared-innovation weight for nonnegative shocks.
	Gamma float64 // Signed asymmetry increment for negative innovations.
	Beta  float64 // Nonnegative lagged conditional-variance weight.
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
// AI-Hints:
//   - A zero State is invalid; initialize it explicitly from the first observation.
//   - Copy State values freely, but serialize access to a mutable state variable.
type State struct {
	ModelID      string  // SHA-256 identity of coefficients and numeric policy.
	LastResidual float64 // Epsilon[t] = observedReturn[t] - Mean; must be finite.
	LastVariance float64 // Positive finite h[t] allocated to that same observation.
	Steps        uint64  // Number of observations already consumed, at least one.
}

// Options configures deterministic, constrained Gaussian QMLE estimation.
// The caller must explicitly provide Initial; there is no random or data-dependent start.
//
// Behavior highlights:
//   - Zero MaxIterations selects 200, zero Tolerance selects 1e-7, and zero
//     VarianceFloor selects 1e-12 squared-return units.
//   - Initial must be a feasible stationary model with Omega >= VarianceFloor.
//   - The optimizer uses a stricter stationarity margin than New.
//
// AI-Hints:
//   - Never supply the zero Parameters value to request automatic initialization.
//   - Rescale VarianceFloor by a*a when changing the units of returns.
type Options struct {
	MaxIterations int        // Maximum accepted projected-BFGS iterations; 0 uses the default.
	Tolerance     float64    // Nonnegative projected-gradient stopping tolerance; 0 uses default.
	VarianceFloor float64    // Strictly enforced variance lower bound; 0 uses default.
	Initial       Parameters // Explicit feasible initial coefficients, never inferred from data.
}

// Diagnostics records the path of accepted QMLE candidate points.
// Objective is a detached sequence of negative average Gaussian log-likelihoods.
//
// Behavior highlights:
//   - Objective[0] is the validated starting candidate; later elements are accepted steps.
//   - Iterations counts accepted moves, not attempted Armijo halvings.
//   - Converged is true only when the projected-gradient condition is satisfied.
//
// Notes:
//   - The caller owns Objective and may mutate it without changing the model.
//   - Exhausted budgets and line-search stalls produce Converged=false.
//
// AI-Hints:
//   - A decreasing Objective history alone does not prove optimizer convergence.
type Diagnostics struct {
	Iterations int       // Number of accepted projected-BFGS transitions.
	Converged  bool      // True only after checking the projected-gradient norm.
	Objective  []float64 // Initial and accepted negative average log-likelihoods.
}

// FitResult publishes a validated model and its QMLE diagnostics.
// Nonconverged results are usable candidates, not statistically certified estimates.
//
// Behavior highlights:
//   - A non-nil result may accompany ErrNotConverged or late context cancellation.
//   - LogLikelihood is the total Gaussian quasi-log-likelihood for Model's parameters.
//   - Model is immutable; Diagnostics.Objective is caller-owned and detached.
//
// AI-Hints:
//   - Always inspect both the returned error and Diagnostics.Converged.
//   - Evaluate out-of-sample predictive fit before using variance estimates in risk decisions.
type FitResult struct {
	Model         *Model      // Validated immutable model, non-nil for published candidates.
	Diagnostics   Diagnostics // Detached optimization history and convergence status.
	LogLikelihood float64     // Total Gaussian QMLE at Model, in original return units.
}

// IsNil reports whether the FitResult pointer itself is nil.
// It follows the core.Nilable convention without dereferencing typed-nil receivers.
//
// Implementation:
//   - Stage 1: Compare the receiver pointer directly with nil.
//   - Stage 2: Return the comparison without reading any result fields.
//
// Returns:
//   - bool: true exactly when r == nil; an allocated empty result is not nil.
//
// Determinism:
//   - Pure O(1) pointer comparison; no shared state or allocation.
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
// Returns:
//   - *FitResult: nil for a nil receiver, or a distinct value with detached slices.
//
// Determinism:
//   - Preserves Objective order and all scalar fields exactly.
//
// Complexity:
//   - Time O(iterations), Space O(iterations).
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
// Behavior highlights:
//   - Objective is negative AVERAGE log-likelihood; LogLikelihood is total log-likelihood.
//   - Gradient differentiates Objective in physical parameter coordinates and includes
//     the dependence of h[0] = Omega/(1-rho) on all variance coefficients.
//
// AI-Hints:
//   - Gradient is not the derivative of total LogLikelihood: that derivative is -T*Gradient.
//   - On an active inequality face use feasible one-sided differences, not central differences.
type QMLEEvaluation struct {
	Objective     float64    // Negative mean Gaussian log-likelihood (per observation).
	LogLikelihood float64    // Sum of Gaussian quasi-log-likelihood contributions.
	Gradient      Parameters // Analytic gradient of Objective with respect to Parameters.
	Observations  int        // Number of observations included in the evaluation.
}

// IsNil reports whether an evaluation pointer is nil, including a typed-nil receiver.
// It has no side effects and costs O(1) time and O(1) space.
func (e *QMLEEvaluation) IsNil() bool { return e == nil }

// Model is an immutable, validated GJR-GARCH(1,1) conditional variance model.
// It may safely be shared across goroutines performing independent read-only calls.
//
// Notes:
//   - Fields are private to prevent invalidating the stationarity and identity laws.
//   - State values and returned slices are never retained by the model.
//
// AI-Hints:
//   - Use Model.ID to ensure persisted State belongs to exactly this model version.
type Model struct {
	p   Parameters // Validated parameter value; never modified after New.
	id  string     // Deterministic policy-versioned SHA-256 identity.
	rho float64    // Validated symmetric-innovation variance persistence coefficient.
}
