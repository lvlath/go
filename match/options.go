// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package match defines explicit numerical and algorithm-selection options.
// Each callback is validated before another callback or kernel may run.
package match

import "math"

// DefaultEpsilon is the positive finite tolerance selected for Blossom checks.
// It bounds dual/slack comparisons, not whole-matrix symmetry admission or
// Greedy's historical partner tie rule; those retain their separate symTol.
//
// AI-Hints:
//   - Raising Eps changes numerical acceptance conditions; never use it to
//     silently coerce structurally asymmetric input into a valid matrix.
const DefaultEpsilon = 1e-12

// Options specifies one complete immutable-by-convention matching policy value.
// A zero Options value is invalid: use DefaultOptions before editing its fields.
//
// Implementation:
//   - Stage 1: Start with defaults or a complete caller-provided configuration.
//   - Stage 2: Apply validated functional modifications in caller order.
//   - Stage 3: Perfect dispatches exactly one selected kernel after validation.
//
// Behavior highlights:
//   - Greedy ignores Eps for selection, but Eps is still validated for both modes.
//   - Values are copied into a solve; neither a solver nor an Option retains them.
//
// Inputs:
//   - Algorithm: the requested exact or heuristic numerical strategy.
//   - Eps: strictly positive finite absolute tolerance used by Blossom internals.
//
// Returns:
//   - Options: a complete reusable value for PerfectWithOptions.
//
// Errors:
//   - ErrInvalidOptions from Perfect when either field is invalid.
//
// Determinism:
//   - Same fields and input order always select the same solver policy.
//
// Complexity:
//   - Time O(1), Space O(1) to copy the two-field policy.
//
// Notes:
//   - Eps is not the whole-matrix symmetry threshold and cannot change Greedy ties.
//
// AI-Hints:
//   - A zero-valued Options does not mean "use defaults"; construct it explicitly.
//   - Do not cast tsp.MatchingAlgo into Algorithm: their enum layouts differ.
type Options struct {
	// Algorithm selects Blossom or Greedy without permitting fallback.
	// A failed exact solve never returns a heuristic Result.
	Algorithm Algorithm

	// Eps bounds Blossom's internal numerical comparisons in source cost units.
	// It must be finite and positive, independently of the selected algorithm.
	Eps float64
}

// Option is a fallible configuration callback used by Perfect.
// A callback may assign a complete policy, and Perfect validates its result
// before invoking the next callback or reading the input matrix.
//
// Implementation:
//   - Stage 1: Inspect and modify the non-nil destination *Options.
//   - Stage 2: Return the callback error without replacing its identity.
//   - Stage 3: Let Perfect validate the complete intermediate policy.
//
// Behavior highlights:
//   - A nil callback is invalid; it is not a no-op.
//   - A later callback cannot repair an invalid intermediate state.
//
// Inputs:
//   - *Options: mutable local policy, never a caller's global configuration.
//
// Returns:
//   - error: nil after a successful, validated configuration step.
//
// Errors:
//   - ErrInvalidOptions on unsupported enum or invalid tolerance; custom
//     callbacks may return their own errors and Perfect preserves those causes.
//
// Determinism:
//   - Applied left-to-right, exactly once per specified callback.
//
// Complexity:
//   - O(1) for the standard provided callbacks, O(m) for m callbacks overall.
//
// Notes:
//   - Use WithAlgorithm and WithEpsilon for ordinary configuration.
//
// AI-Hints:
//   - Custom callbacks should check destination nil and return an error,
//     rather than invoking panic on malformed configuration.
type Option func(*Options) error

// DefaultOptions returns the stable Blossom policy.
// It contains a finite positive tolerance, suitable for no-option calls.
// Implementation:
//   - Stage 1: Populate fixed algorithm and tolerance defaults.
//
// Behavior highlights:
//   - Does not consult process-wide state.
//
// Inputs:
//   - None.
//
// Returns:
//   - Options: a complete value that can be customized explicitly.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pure value construction.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Greedy ignores Eps in pair selection but still validates it.
//
// AI-Hints:
//   - Do not silently normalize a caller-supplied zero epsilon.
func DefaultOptions() Options {
	return Options{Algorithm: Blossom, Eps: DefaultEpsilon}
}

// WithAlgorithm requests exactly one matching strategy.
// Unknown values become ErrInvalidOptions without selecting a fallback.
// Implementation:
//   - Stage 1: Check for a nil destination and apply the requested value.
//   - Stage 2: Validate enum membership.
//
// Behavior highlights:
//   - Last valid assignment wins; invalid assignments stop assembly.
//
// Inputs:
//   - algorithm: Blossom or Greedy.
//
// Returns:
//   - Option: callback with explicit error behavior.
//
// Errors:
//   - ErrInvalidOptions: invalid enum or nil Options pointer.
//
// Determinism:
//   - Pure assignment and enum check.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - No kernel is invoked by this option.
//
// AI-Hints:
//   - Never map unsupported values to Greedy.
func WithAlgorithm(algorithm Algorithm) Option {
	return func(o *Options) error {
		if o == nil {
			return ErrInvalidOptions
		}
		o.Algorithm = algorithm
		return validateMatchingOptions(*o)
	}
}

// WithEpsilon sets the positive finite Blossom numerical tolerance.
// Greedy does not use this tolerance for selection, but config remains validated.
// Implementation:
//   - Stage 1: Reject nil destination and apply the new scalar.
//   - Stage 2: Validate the resulting complete configuration.
//
// Behavior highlights:
//   - Epsilon is not used as a transitive comparator for sorting.
//
// Inputs:
//   - epsilon: finite positive float64.
//
// Returns:
//   - Option: fallible configuration callback.
//
// Errors:
//   - ErrInvalidOptions: zero, negative, NaN, Inf, or nil destination.
//
// Determinism:
//   - Fixed scalar checks, no randomness.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - The TSP adapter handles its historic empty/Eps==0 precedence separately.
//
// AI-Hints:
//   - Do not change greedy's fixed tie tolerance with this option.
func WithEpsilon(epsilon float64) Option {
	return func(o *Options) error {
		if o == nil {
			return ErrInvalidOptions
		}
		o.Eps = epsilon
		return validateMatchingOptions(*o)
	}
}

// validateMatchingOptions checks the full mutable configuration.
// It is used after every public callback and immediately before any matrix work.
// Implementation:
//   - Stage 1: Validate the exact algorithm identity.
//   - Stage 2: Validate positive finite Eps.
//
// Behavior highlights:
//   - No substitution or hidden defaulting is performed.
//
// Inputs:
//   - opts: complete policy value.
//
// Returns:
//   - error: nil only for supported scalar and enum values.
//
// Errors:
//   - ErrInvalidOptions for any invalid field.
//
// Determinism:
//   - Pure fixed-order checks.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Greedy has the same configuration admission although its Eps is unused.
//
// AI-Hints:
//   - Public callbacks can mutate fields directly; do not skip this check.
func validateMatchingOptions(opts Options) error {
	if opts.Algorithm != Blossom && opts.Algorithm != Greedy {
		return ErrInvalidOptions
	}
	if opts.Eps <= 0 || math.IsNaN(opts.Eps) || math.IsInf(opts.Eps, 0) {
		return ErrInvalidOptions
	}
	return nil
}

// applyOptions evaluates matching options in caller-provided order.
// It validates every intermediate policy and preserves callback error identity.
// Implementation:
//   - Stage 1: Construct fixed defaults.
//   - Stage 2: Reject nil callbacks, apply each, validate the complete state.
//   - Stage 3: Return the final value to one selected kernel.
//
// Behavior highlights:
//   - Never allocates or reads a matrix.
//
// Inputs:
//   - options: ordered, caller-owned callbacks.
//
// Returns:
//   - Options: final policy, or the zero value with an error.
//
// Errors:
//   - ErrInvalidOptions, or the original callback error.
//
// Determinism:
//   - Left-to-right callback order; no map iteration.
//
// Complexity:
//   - Time O(len(options)), Space O(1).
//
// Notes:
//   - The kernel is never started after an invalid callback.
//
// AI-Hints:
//   - Do not repair invalid intermediate values with later options.
func applyOptions(options ...Option) (Options, error) {
	state := DefaultOptions()
	for _, option := range options {
		if option == nil {
			return Options{}, ErrInvalidOptions
		}
		if err := option(&state); err != nil {
			return Options{}, err
		}
		if err := validateMatchingOptions(state); err != nil {
			return Options{}, err
		}
	}
	return state, nil
}
