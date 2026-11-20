// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pacf declares the fixed sample-PACF configuration and result artifact.
// This file establishes centering, shape, nilability, and caller ownership laws.
package pacf

// Options configures one biased-autocovariance PACF computation.
// It fixes the maximum reported prediction order and the whole-window centering
// convention without offering a hidden alternative estimator or lag-dependent divisor.
//
// Implementation:
//   - Stage 1: Compute validates MaxLag against the nonempty input window.
//   - Stage 2: Center selects either a stable fitted global mean or the physical zero.
//   - Stage 3: the same policy is used for all covariances and prediction orders.
//
// Behavior highlights:
//   - Options{} requests uncentered lag zero only; MaxLag=0 is never automatic.
//   - Center=true is not detrending, differencing, or stationarity enforcement.
//
// Inputs:
//   - MaxLag: inclusive nonnegative lag strictly below the number of samples.
//   - Center: true subtracts the one fitted training mean; false uses raw second moments.
//
// Returns:
//   - The options value is copied by Compute; no pointer or mutable policy is retained.
//
// Errors:
//   - Compute returns ErrInvalidOption for negative or out-of-range MaxLag.
//
// Determinism:
//   - The two fields select exactly one deterministic mathematical convention.
//
// Complexity:
//   - Time O(1), Space O(1) for policy storage and selection.
//
// Notes:
//   - New estimators would require explicit new contracts; they do not belong to these fields.
//
// AI-Hints:
//   - Never change the default to centered or infer MaxLag from sample size.
//   - ACF LagAdjusted (n-k divisor) is not compatible with this Toeplitz recursion.
type Options struct {
	// MaxLag is the inclusive highest output lag; zero requests only Values[0].
	// Compute validates 0 <= MaxLag < len(values) before reading observation values.
	MaxLag int

	// Center fits one full-window sample mean if true, otherwise uses exact physical zero.
	// This controls the estimand and must not be inferred from data magnitude.
	Center bool
}

// Result holds the complete partial-autocorrelation and prediction-variance curves.
// The caller owns its slices independently of both the input and subsequent calls.
// Every successful Result describes the same whole-window centered or raw estimator.
//
// Implementation:
//   - Stage 1: covariance preparation supplies gamma[0..MaxLag] at one scale.
//   - Stage 2: Durbin-Levinson fills one coefficient and variance per order.
//   - Stage 3: return the artifact only after all orders and context checks succeed.
//
// Behavior highlights:
//   - len(Values)==len(PredictionVariance)==MaxLag+1; Values[0] is exactly one.
//   - PredictionVariance[m] is the order-m one-step linear prediction error variance.
//   - Count is the complete input window length, not n-m available pairs.
//
// Inputs:
//   - Constructed by Compute; exported fields may subsequently be modified by the owner.
//
// Returns:
//   - Both slices are detached, independently allocated, and contain finite values.
//
// Errors:
//   - An unsuccessful computation returns nil *Result instead of a partial artifact.
//
// Determinism:
//   - Both slices are indexed by ascending order from zero through MaxLag.
//
// Complexity:
//   - Retained space O(MaxLag+1); Clone uses O(MaxLag+1) time and space.
//
// Notes:
//   - An allocated zero-value Result is not a valid Compute output, but it is non-nil.
//   - Concurrent read-only sharing is safe; owners synchronize concurrent mutation.
//
// AI-Hints:
//   - PredictionVariance is not a coefficient confidence interval or held-out forecast MSE.
//   - Never infer the estimator centering mode from Values alone; retain caller policy.
type Result struct {
	// Values contains kappa[0..MaxLag], where kappa[0] is exactly 1 on success.
	// Its backing array belongs to the caller and never aliases PredictionVariance.
	Values []float64

	// PredictionVariance contains V[0..MaxLag] in squared input units, with V[m] > 0.
	// On success V[0] is the biased centered covariance or raw second moment.
	PredictionVariance []float64

	// Count is the original number of observations used by the whole-window estimator.
	// Count equals len(values), even for high-lag estimates with fewer available pairs.
	Count int
}

// IsNil reports whether the receiver pointer is nil, including typed nil interfaces.
// It implements the core.Nilable protocol structurally without importing core.
//
// Implementation:
//   - Stage 1: compare only the receiver pointer to nil.
//   - Stage 2: return the boolean without inspecting slices or dereferencing r.
//
// Behavior highlights:
//   - An allocated empty Result is not classified as nil.
//
// Inputs:
//   - r: possibly nil *Result receiver, including a typed nil in an interface.
//
// Returns:
//   - bool: true exactly when r == nil, including an interface-wrapped nil receiver.
//
// Errors:
//   - None; a nil receiver is a supported input.
//
// Determinism:
//   - One pointer comparison with no shared mutable state.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Nilability is a presence protocol, not a validity or completeness check.
//
// AI-Hints:
//   - Never use reflect, slice length, or zero-value field heuristics for nilability.
func (r *Result) IsNil() bool {
	return r == nil
}

// Clone returns a deep, detached copy of a PACF result without recomputing it.
// It permits safe ownership transfer when one caller intends to modify result curves.
//
// Implementation:
//   - Stage 1: return nil immediately for a nil receiver.
//   - Stage 2: copy scalar metadata and allocate independent copies of present slices.
//   - Stage 3: return the complete copy with exactly the same lag order.
//
// Behavior highlights:
//   - A non-nil empty slice remains non-nil; nil slices remain nil.
//   - Copies of both slices are independent from the source and from one another.
//
// Inputs:
//   - r: nil or read-only during the call; concurrent mutation is unsupported.
//
// Returns:
//   - *Result: nil for a nil receiver, otherwise caller-owned detached data.
//
// Errors:
//   - None; allocations use existing slice lengths and may fail only at runtime limits.
//
// Determinism:
//   - Copies every element by its existing index, without reordering or recomputation.
//
// Complexity:
//   - Time O(L+1), Space O(L+1), where L+1 is the output slice length.
//
// Notes:
//   - The method does not validate manually constructed or manually modified Results.
//
// AI-Hints:
//   - Copying only the struct would alias public slice backing arrays.
func (r *Result) Clone() *Result {
	if r == nil {
		return nil
	}

	cloned := *r
	if r.Values != nil {
		cloned.Values = make([]float64, len(r.Values))
		copy(cloned.Values, r.Values)
	}
	if r.PredictionVariance != nil {
		cloned.PredictionVariance = make([]float64, len(r.PredictionVariance))
		copy(cloned.PredictionVariance, r.PredictionVariance)
	}
	return &cloned
}
