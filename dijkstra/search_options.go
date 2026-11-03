// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dijkstra defines the independent policy for target-directed shortest paths.
// This file assembles search options without changing legacy SSSP configuration.
package dijkstra

import (
	"context"
	"math"
)

// SearchOptions describes the policy of one target-directed search.
// The witness is mandatory for reachable targets, unlike optional SSSP tracking.
// Fields are copied into each invocation; callers must not mutate configuration
// concurrently with an executing custom SearchOption.
//
// Implementation:
//   - Stage 1: Start with DefaultSearchOptions.
//   - Stage 2: Apply each SearchOption with immediate scalar validation.
//   - Stage 3: Enforce supported method-specific completeness before graph allocation.
//
// Behavior highlights:
//   - Method is a concrete working algorithm, not an implicit heuristic selector.
//   - MaxDistance is an inclusive original-cost bound, not an edge threshold.
//   - InfEdgeThreshold excludes edges with weights >= the threshold.
//
// Inputs:
//   - Method: implemented search algorithm; initially SearchDijkstra.
//   - Context: non-nil cancellation source owned by the caller.
//   - MaxDistance: finite non-negative inclusive bound or +Inf.
//   - InfEdgeThreshold: finite positive edge-wall threshold or +Inf.
//   - ToTarget: required by A* methods; otherwise must be nil.
//   - FromSource: required by balanced bidirectional A*; otherwise must be nil.
//
// Returns:
//   - SearchOptions: a copied configuration value; no graph data is retained.
//
// Errors:
//   - ErrInvalidSearchMethod, ErrNilContext, ErrBadMaxDistance, ErrBadInfEdgeThreshold,
//     ErrNilHeuristic or ErrInvalidSearchOptions during configuration validation.
//
// Determinism:
//   - Options are applied in caller order; repeated valid setters use last-writer-wins.
//
// Complexity:
//   - Time O(1), Space O(1) for a value copy, excluding option callbacks.
//
// Notes:
//   - The zero value is not a valid complete configuration; use DefaultSearchOptions.
//   - Balanced methods require both frozen callbacks; ordinary A* uses ToTarget only.
//
// AI-Hints:
//   - Do not reinterpret the +Inf policy sentinel as a legal graph edge weight.
//   - Do not introduce TrackPaths: a target result always includes its witness.
type SearchOptions struct {
	// Method selects the concrete target kernel; unsupported values are errors.
	Method SearchMethod

	// Context controls cooperative cancellation and must be non-nil.
	Context context.Context

	// MaxDistance is an inclusive original-cost bound, or +Inf when disabled.
	MaxDistance float64

	// InfEdgeThreshold blocks valid finite edges with weight >= this bound.
	InfEdgeThreshold float64

	// ToTarget estimates remaining original cost; nil for nonheuristic methods.
	ToTarget Heuristic

	// FromSource estimates original cost from the source, for balanced A* only.
	FromSource Heuristic
}

// SearchOption changes an in-construction target-search configuration.
// The assembler always checks the resulting scalar state after callbacks return.
// Implementation:
//   - Stage 1: Receive a private mutable SearchOptions value.
//   - Stage 2: Validate the requested payload and apply the change.
//   - Stage 3: Return any failure without publishing the partially changed value.
//
// Behavior highlights:
//   - Custom callbacks may return their own errors, which are preserved.
//
// Inputs:
//   - *SearchOptions: assembler-owned mutable configuration, not caller storage.
//
// Returns:
//   - error: nil on valid mutation; otherwise the originating error.
//
// Errors:
//   - ErrNilOption for a nil function, reported by applySearchOptions.
//
// Determinism:
//   - Callback execution is sequential, in the supplied order.
//
// Complexity:
//   - Depends on the callback; built-in setters are O(1) time and space.
//
// Notes:
//   - Callbacks must not retain or mutate their argument asynchronously.
//
// AI-Hints:
//   - Do not use callbacks to alter graph topology or to select hidden kernels.
type SearchOption func(*SearchOptions) error

// DefaultSearchOptions returns the canonical target-directed policy.
// The zero-heuristic, single-frontier search is selected without a cutoff or wall.
// Implementation:
//   - Stage 1: Select the implemented target Dijkstra method.
//   - Stage 2: Set a background context and unbounded scalar policies.
//   - Stage 3: Return the detached value.
//
// Behavior highlights:
//   - Both infinities are option-domain values, never valid edge weights.
//
// Inputs:
//   - None.
//
// Returns:
//   - SearchOptions: a complete policy with no external mutable state.
//
// Errors:
//   - None.
//
// Determinism:
//   - Identical values for all calls except the opaque, immutable background context.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Returning this value does not allocate traversal state.
//
// AI-Hints:
//   - Do not replace +Inf defaults with a large finite approximation.
func DefaultSearchOptions() SearchOptions {
	return SearchOptions{
		Method:           SearchDijkstra,
		Context:          context.Background(),
		MaxDistance:      math.Inf(1),
		InfEdgeThreshold: math.Inf(1),
	}
}

// WithSearchMethod selects the target-search algorithm explicitly.
// Unsupported methods are rejected by the option assembler, not silently run as Dijkstra.
// Implementation:
//   - Stage 1: Capture the requested method.
//   - Stage 2: Validate its availability when the option is applied.
//   - Stage 3: Assign the method only on success.
//
// Behavior highlights:
//   - No heuristic or other option implicitly changes the method.
//
// Inputs:
//   - method: a currently implemented SearchMethod constant.
//
// Returns:
//   - SearchOption: an O(1) method setter.
//
// Errors:
//   - ErrInvalidSearchMethod for unsupported method values.
//
// Determinism:
//   - The final valid method obeys last-writer-wins option ordering.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Methods are published only when their kernels are operational.
//
// AI-Hints:
//   - Never treat an unknown enum value as SearchDijkstra.
func WithSearchMethod(method SearchMethod) SearchOption {
	return func(config *SearchOptions) error {
		if err := validateSearchMethod(method); err != nil {
			return err
		}
		config.Method = method
		return nil
	}
}

// WithSearchContext configures cooperative search cancellation.
// A nil context is rejected rather than silently replaced with Background.
// Implementation:
//   - Stage 1: Capture the caller-owned context.
//   - Stage 2: Reject nil when the option is applied.
//   - Stage 3: Assign the context without retaining it in PathResult.
//
// Behavior highlights:
//   - Search observes cancellation during preparation, expansion and publication.
//
// Inputs:
//   - ctx: a non-nil context.Context with caller-owned lifetime.
//
// Returns:
//   - SearchOption: an O(1) context setter.
//
// Errors:
//   - ErrNilContext when ctx is nil.
//
// Determinism:
//   - Configuration order is fixed; cancellation itself depends on caller timing.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - A synchronous core.Clone or user callback cannot be forcibly interrupted.
//
// AI-Hints:
//   - Never replace explicit nil with context.Background in the setter.
func WithSearchContext(ctx context.Context) SearchOption {
	return func(config *SearchOptions) error {
		if ctx == nil {
			return ErrNilContext
		}
		config.Context = ctx
		return nil
	}
}

// WithSearchMaxDistance sets the inclusive original-cost cutoff.
// Candidate admission uses the exact represented-binary sum in the target kernel.
// Implementation:
//   - Stage 1: Validate the supplied scalar with the legacy shared validator.
//   - Stage 2: Assign the value to the target configuration.
//   - Stage 3: Leave graph weights and wall policy unchanged.
//
// Behavior highlights:
//   - Zero allows zero-cost routes; +Inf disables the cutoff.
//
// Inputs:
//   - limit: finite non-negative cost or +Inf, in original edge-cost units.
//
// Returns:
//   - SearchOption: an O(1) inclusive cutoff setter.
//
// Errors:
//   - ErrBadMaxDistance for NaN, -Inf or negative values.
//
// Determinism:
//   - Identical scalar payload yields identical policy.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Legacy SSSP uses rounded-candidate cutoff; target Search uses exact comparisons.
//
// AI-Hints:
//   - Do not compare the cutoff with reduced A* priority keys.
func WithSearchMaxDistance(limit float64) SearchOption {
	return func(config *SearchOptions) error {
		if err := validateMaxDistance(limit); err != nil {
			return err
		}
		config.MaxDistance = limit
		return nil
	}
}

// WithSearchInfEdgeThreshold configures an inclusive per-edge wall.
// Finite edges at or above the threshold are excluded after global weight validation.
// Implementation:
//   - Stage 1: Apply the canonical scalar-domain validator.
//   - Stage 2: Assign a positive threshold or +Inf.
//   - Stage 3: Leave total-path cutoff unchanged.
//
// Behavior highlights:
//   - Even excluded edges must have finite non-negative weights.
//
// Inputs:
//   - threshold: finite positive edge weight or +Inf in original units.
//
// Returns:
//   - SearchOption: an O(1) edge-wall setter.
//
// Errors:
//   - ErrBadInfEdgeThreshold for invalid values.
//
// Determinism:
//   - The same threshold excludes the same effective edge IDs.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Policy affects arcs only and does not alter the private cloned topology.
//
// AI-Hints:
//   - Do not apply walls before classifyWeight; invalid input is never hidden.
func WithSearchInfEdgeThreshold(threshold float64) SearchOption {
	return func(config *SearchOptions) error {
		if err := validateInfEdgeThreshold(threshold); err != nil {
			return err
		}
		config.InfEdgeThreshold = threshold
		return nil
	}
}

// WithTraversalPolicy copies two scalar traversal limits from legacy Options.
// TrackPaths is deliberately ignored because target search publishes a path witness.
// Implementation:
//   - Stage 1: Capture the provided Options value without retaining a pointer.
//   - Stage 2: Validate both scalars before changing either destination field.
//   - Stage 3: Copy MaxDistance and InfEdgeThreshold atomically as configuration.
//
// Behavior highlights:
//   - This is a projection, not an alias of the caller's Options.
//
// Inputs:
//   - policy: detached legacy Options value; TrackPaths has no target meaning.
//
// Returns:
//   - SearchOption: an O(1) policy projection.
//
// Errors:
//   - ErrBadMaxDistance or ErrBadInfEdgeThreshold in validator order.
//
// Determinism:
//   - Last valid scalar setter wins in the caller's option sequence.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Using Options{} without initialized limits is invalid.
//
// AI-Hints:
//   - Do not propagate TrackPaths into target results or skip scalar validation.
func WithTraversalPolicy(policy Options) SearchOption {
	return func(config *SearchOptions) error {
		if err := validateMaxDistance(policy.MaxDistance); err != nil {
			return err
		}
		if err := validateInfEdgeThreshold(policy.InfEdgeThreshold); err != nil {
			return err
		}
		config.MaxDistance = policy.MaxDistance
		config.InfEdgeThreshold = policy.InfEdgeThreshold
		return nil
	}
}

// WithHeuristic supplies a consistent target potential without choosing a method.
// The callback is validated eagerly for nil and evaluated once per vertex at runtime.
// Implementation:
//   - Stage 1: Capture the caller-owned function without invoking it.
//   - Stage 2: Reject nil when the option is applied.
//   - Stage 3: Assign the callback for final method-specific completeness validation.
//
// Behavior highlights:
//   - A later valid setter may replace a previous callback; no implicit method switch.
//
// Inputs:
//   - toTarget: non-nil callback producing finite original-cost estimates.
//
// Returns:
//   - SearchOption: a deterministic O(1) callback setter.
//
// Errors:
//   - ErrNilHeuristic for an explicitly nil callback.
//
// Determinism:
//   - The setter does not call the heuristic; graph preparation defines call order.
//
// Complexity:
//   - Time O(1), Space O(1) for setup; runtime callback work is separate.
//
// Notes:
//   - Explicitly nil callbacks are invalid even when a later option would repair them.
//
// AI-Hints:
//   - Do not invoke callbacks before the coherent topology snapshot is validated.
//   - Do not silently set Method=SearchAStar here; the caller must select it.
func WithHeuristic(toTarget Heuristic) SearchOption {
	return func(config *SearchOptions) error {
		if toTarget == nil {
			return ErrNilHeuristic
		}
		config.ToTarget = toTarget
		return nil
	}
}

// WithBalancedHeuristics supplies the two consistent directional estimates.
// The pair is installed atomically and never implicitly selects a search method.
// Implementation:
//   - Stage 1: Capture the target-directed and source-directed callback values.
//   - Stage 2: Reject either missing callback before modifying the configuration.
//   - Stage 3: Install both functions for final method-specific validation.
//
// Behavior highlights:
//   - ToTarget must satisfy hT(target)=0 and forward-edge consistency.
//   - FromSource must satisfy hS(source)=0 and reverse-edge consistency.
//
// Inputs:
//   - toTarget: non-nil estimate of remaining original cost to the target.
//   - fromSource: non-nil estimate of original cost from the source.
//
// Returns:
//   - SearchOption: an O(1) setter that stores two caller-owned functions.
//
// Errors:
//   - ErrNilHeuristic if either callback is nil when the option is applied.
//
// Determinism:
//   - An invalid pair leaves both fields unchanged; valid settings obey option order.
//
// Complexity:
//   - Time O(1), Space O(1) for configuration; evaluation occurs later.
//
// Notes:
//   - The functions must remain safe for synchronous invocation; no global cache exists.
//
// AI-Hints:
//   - Do not turn a missing callback into an implicit zero heuristic.
//   - Do not silently change Method to SearchBidirectionalAStar from a setter.
func WithBalancedHeuristics(toTarget, fromSource Heuristic) SearchOption {
	return func(config *SearchOptions) error {
		if toTarget == nil || fromSource == nil {
			return ErrNilHeuristic
		}
		config.ToTarget = toTarget
		config.FromSource = fromSource
		return nil
	}
}

// validateSearchMethod accepts only methods backed by an implemented kernel.
// This check prevents publishing a mode whose result would be a fabricated fallback.
// Implementation:
//   - Stage 1: Compare the method against supported public values.
//   - Stage 2: Return a sentinel for any other enum representation.
//   - Stage 3: Leave the input unchanged.
//
// Behavior highlights:
//   - A valid zero value selects target Dijkstra.
//
// Inputs:
//   - method: candidate SearchMethod value.
//
// Returns:
//   - error: nil for the implemented method.
//
// Errors:
//   - ErrInvalidSearchMethod for unsupported values.
//
// Determinism:
//   - Exact enum comparison, independent of graph state.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Extend cases only in the same change that implements their kernels.
//
// AI-Hints:
//   - Never silently map unsupported methods to the default.
func validateSearchMethod(method SearchMethod) error {
	switch method {
	case SearchDijkstra, SearchBidirectional, SearchAStar, SearchBidirectionalAStar:
		return nil
	default:
		return ErrInvalidSearchMethod
	}
}

// validateSearchScalars checks all local configuration invariants.
// The check runs after every custom option, not only after built-in setters.
// Implementation:
//   - Stage 1: Validate the supported method and context.
//   - Stage 2: Apply shared scalar-domain validators in documented order.
//   - Stage 3: Report the first failure without applying any normalization.
//
// Behavior highlights:
//   - Invalid scalar state cannot be repaired by a later callback.
//
// Inputs:
//   - config: current assembler-owned search policy value.
//
// Returns:
//   - error: nil only for valid local fields.
//
// Errors:
//   - ErrInvalidSearchMethod, ErrNilContext, ErrBadMaxDistance, ErrBadInfEdgeThreshold.
//
// Determinism:
//   - Field order: Method, Context, MaxDistance, InfEdgeThreshold.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Cross-field heuristic completeness is checked after the complete option list.
//
// AI-Hints:
//   - Do not change legacy applyOptions to reuse this assembly model.
func validateSearchScalars(config SearchOptions) error {
	if err := validateSearchMethod(config.Method); err != nil {
		return err
	}
	if config.Context == nil {
		return ErrNilContext
	}
	if err := validateMaxDistance(config.MaxDistance); err != nil {
		return err
	}
	return validateInfEdgeThreshold(config.InfEdgeThreshold)
}

// validateSearchConfiguration checks final method/heuristic compatibility.
// Local scalar integrity is checked after every option; callback completeness
// is delayed until all options have been applied, so their order remains ergonomic.
// Implementation:
//   - Stage 1: Inspect the selected supported method.
//   - Stage 2: Reject callbacks for algorithms that cannot consume them.
//   - Stage 3: Require only ToTarget for A* and both callbacks for balanced A*.
//
// Behavior highlights:
//   - No heuristic is silently ignored and no missing one triggers zero fallback.
//
// Inputs:
//   - config: fully assembled local search policy with valid scalar fields.
//
// Returns:
//   - error: nil only for a complete, method-compatible configuration.
//
// Errors:
//   - ErrInvalidSearchOptions for an inapplicable callback.
//   - ErrNilHeuristic when an A* method lacks a required callback.
//   - ErrInvalidSearchMethod for a method without a working kernel.
//
// Determinism:
//   - The selected method controls one fixed, documented compatibility branch.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - An option may select A* before a later option supplies its callback.
//
// AI-Hints:
//   - Never enforce A* completeness after each option: that rejects valid
//     WithSearchMethod(SearchAStar), WithHeuristic(h) sequences.
func validateSearchConfiguration(config SearchOptions) error {
	switch config.Method {
	case SearchDijkstra, SearchBidirectional:
		if config.ToTarget != nil || config.FromSource != nil {
			return ErrInvalidSearchOptions
		}
	case SearchAStar:
		if config.FromSource != nil {
			return ErrInvalidSearchOptions
		}
		if config.ToTarget == nil {
			return ErrNilHeuristic
		}
	case SearchBidirectionalAStar:
		if config.ToTarget == nil || config.FromSource == nil {
			return ErrNilHeuristic
		}
	default:
		return ErrInvalidSearchMethod
	}
	return nil
}

// applySearchOptions assembles and validates one target-search configuration.
// All option callbacks run before graph cloning or search workspace allocation.
// Implementation:
//   - Stage 1: Allocate default scalar configuration on the stack.
//   - Stage 2: Invoke options sequentially, rejecting nil and callback failures.
//   - Stage 3: Revalidate local invariants after every callback, then method completeness.
//
// Behavior highlights:
//   - Failed configuration never reaches a kernel or publishes partial state.
//
// Inputs:
//   - opts: ordered functional options; caller retains ownership of the slice.
//
// Returns:
//   - SearchOptions: finalized detached configuration, or zero value on error.
//
// Errors:
//   - ErrNilOption, scalar/method/heuristic sentinels and custom callback errors.
//
// Determinism:
//   - Exact caller order; first error wins; valid later options override earlier ones.
//
// Complexity:
//   - Time O(n) plus callback work, Space O(1), with n supplied options.
//
// Notes:
//   - This assembly is intentionally independent of legacy applyOptions.
//
// AI-Hints:
//   - Never defer validation of an invalid scalar until the final option.
//   - Do not silently recover from an error in a user-defined callback.
func applySearchOptions(opts ...SearchOption) (SearchOptions, error) {
	config := DefaultSearchOptions()
	for _, option := range opts {
		if option == nil {
			return SearchOptions{}, ErrNilOption
		}
		if err := option(&config); err != nil {
			return SearchOptions{}, err
		}
		if err := validateSearchScalars(config); err != nil {
			return SearchOptions{}, err
		}
	}
	if err := validateSearchScalars(config); err != nil {
		return SearchOptions{}, err
	}
	if err := validateSearchConfiguration(config); err != nil {
		return SearchOptions{}, err
	}
	return config, nil
}
