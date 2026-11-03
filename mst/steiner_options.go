// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mst defines admission and storage policies for terminal-covering networks.
// The options are independent of the compatibility-sensitive ordinary MST Options.
package mst

import "context"

// SteinerPathStorage selects the live witness state retained between KMB phases.
// Both modes compute the same SSSP objective, differing only in retained state and calls.
// Implementation:
//   - Stage 1: Select retained or recomputed predecessor maps.
//   - Stage 2: Use the chosen policy for closure and original-path expansion.
//
// Behavior highlights:
//   - No fallback to another mode occurs under memory pressure.
//
// Inputs:
//   - The value is accepted only through validated SteinerOptions.
//
// Returns:
//   - A storage policy interpreted by SteinerTree.
//
// Errors:
//   - ErrInvalidOption for unsupported values.
//
// Determinism:
//   - Explicit values choose deterministic execution schedules.
//
// Complexity:
//   - Retain O(kV) history; recompute O(V) live SSSP state.
//
// Notes:
//   - This enum does not select a different graph optimization objective.
//
// AI-Hints:
//   - Never ignore PathStorage or silently choose a cheaper runtime mode.
type SteinerPathStorage uint8

const (
	// SteinerRecomputePaths performs source-grouped tracking searches after closure selection.
	SteinerRecomputePaths SteinerPathStorage = iota
	// SteinerRetainPaths keeps terminal-source distances and predecessor maps for expansion.
	SteinerRetainPaths
)

// SteinerOptions configures one nonnegative-weight graph Steiner construction.
// Its limits are logical input counts, not a cap on process allocations or RSS.
// Implementation:
//   - Stage 1: Resolve context and memory policy.
//   - Stage 2: Validate caller caps and compare them against input counts.
//   - Stage 3: Delegate to the KMB kernel with immutable scalar policy.
//
// Behavior highlights:
//   - A zero explicit cap means no caller cap; int overflow guards remain mandatory.
//
// Inputs:
//   - Context: a non-nil context; PathStorage: one of the declared modes.
//   - MaxVertices/MaxEdges/MaxTerminals: nonnegative upper bounds on original input counts.
//
// Returns:
//   - SteinerOptions: a standalone policy value, not a graph or workspace.
//
// Errors:
//   - ErrInvalidOption on invalid context, mode, or cap.
//
// Determinism:
//   - Options are applied in caller order and valid later setters win.
//
// Complexity:
//   - Time O(p) to assemble p options; Space O(1).
//
// Notes:
//   - This type is separate from legacy Options to preserve old final-only validation.
//
// AI-Hints:
//   - Do not claim MaxEdges limits allocations made by core or dijkstra.
type SteinerOptions struct {
	// Context allows phase-cooperative cancellation; nil contexts are invalid.
	Context context.Context
	// PathStorage fixes the SSSP witness retention strategy.
	PathStorage SteinerPathStorage
	// MaxVertices bounds the original graph vertex count when positive.
	MaxVertices int
	// MaxEdges bounds the original graph edge catalog, including loops and islands.
	MaxEdges int
	// MaxTerminals bounds the caller's terminal-list length when positive.
	MaxTerminals int
}

// SteinerOption mutates one SteinerOptions value and reports configuration errors.
// Custom callbacks are checked after each invocation before touching graph topology.
// Implementation:
//   - Stage 1: Apply a caller-supplied policy mutation.
//   - Stage 2: Revalidate scalar fields in applySteinerOptions.
//
// Behavior highlights:
//   - Nil callbacks and nil contexts are errors rather than no-ops.
//
// Inputs:
//   - *SteinerOptions: non-nil assembler-owned policy pointer.
//
// Returns:
//   - error: nil for valid policy changes.
//
// Errors:
//   - ErrInvalidOption, ErrNilOption, or a preserved custom callback error.
//
// Determinism:
//   - Sequential application in caller-specified order.
//
// Complexity:
//   - Time and Space determined by the callback; built-ins use O(1).
//
// Notes:
//   - Custom callbacks must not retain the ephemeral configuration pointer.
//
// AI-Hints:
//   - Do not rewrite existing mst.Option to use this new assembler.
type SteinerOption func(*SteinerOptions) error

// DefaultSteinerOptions provides the explicit KMB execution defaults.
// It selects recomputation and leaves optional input-count caps unset.
// Implementation:
//   - Stage 1: Supply context.Background and recompute witness storage.
//   - Stage 2: Leave each logical cap at zero.
//
// Behavior highlights:
//   - No finite cutoff, wall, or edge-filter policy is introduced.
//
// Inputs:
//   - None.
//
// Returns:
//   - SteinerOptions: independent default configuration.
//
// Errors:
//   - None.
//
// Determinism:
//   - Fixed scalar values and a fresh background context.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Zero caps do not disable checked arithmetic on sizes.
//
// AI-Hints:
//   - Do not change this default to retain because a benchmark favors one graph.
func DefaultSteinerOptions() SteinerOptions {
	return SteinerOptions{Context: context.Background(), PathStorage: SteinerRecomputePaths}
}

// validateSteinerOptions classifies a complete scalar policy configuration.
// It is called after every custom callback and after the complete option list.
// Implementation:
//   - Stage 1: Reject a nil context and unsupported memory mode.
//   - Stage 2: Reject negative logical caps.
//   - Stage 3: Return success without consulting a graph.
//
// Behavior highlights:
//   - Zero caps are legal; no default limits are injected.
//
// Inputs:
//   - options: assembler-owned mutable policy snapshot.
//
// Returns:
//   - error: nil for a supported configuration.
//
// Errors:
//   - ErrInvalidOption when scalar values are invalid.
//
// Determinism:
//   - Checks in fixed context, mode, then cap order.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Caller count admission occurs only after options succeed.
//
// AI-Hints:
//   - Never silently replace a custom nil Context with Background.
func validateSteinerOptions(options SteinerOptions) error {
	if options.Context == nil {
		return ErrInvalidOption
	}
	if options.PathStorage != SteinerRecomputePaths && options.PathStorage != SteinerRetainPaths {
		return ErrInvalidOption
	}
	if options.MaxVertices < 0 || options.MaxEdges < 0 || options.MaxTerminals < 0 {
		return ErrInvalidOption
	}
	return nil
}

// applySteinerOptions assembles one validated KMB policy without graph allocations.
// It stops at the first callback error or invalid complete scalar state.
// Implementation:
//   - Stage 1: Initialize defaults and reject nil callbacks.
//   - Stage 2: Apply every callback and validate scalars immediately.
//   - Stage 3: Return a final owned policy value.
//
// Behavior highlights:
//   - Last valid setter wins; invalid intermediate states cannot be repaired later.
//
// Inputs:
//   - opts: ordered caller-owned callbacks, not retained.
//
// Returns:
//   - SteinerOptions: validated configuration; zero value on error.
//
// Errors:
//   - ErrNilOption, ErrInvalidOption, or the original callback error.
//
// Determinism:
//   - Exact caller ordering and first-failure precedence.
//
// Complexity:
//   - Time O(p), Space O(1) excluding callback work.
//
// Notes:
//   - Legacy ApplyOptions deliberately retains its final-only compatibility policy.
//
// AI-Hints:
//   - Do not delegate new options to legacy ApplyOptions or change its validation order.
func applySteinerOptions(opts ...SteinerOption) (SteinerOptions, error) {
	options := DefaultSteinerOptions()
	for _, option := range opts {
		if option == nil {
			return SteinerOptions{}, ErrNilOption
		}
		if err := option(&options); err != nil {
			return SteinerOptions{}, err
		}
		if err := validateSteinerOptions(options); err != nil {
			return SteinerOptions{}, err
		}
	}
	if err := validateSteinerOptions(options); err != nil {
		return SteinerOptions{}, err
	}
	return options, nil
}

// WithSteinerContext configures cooperative cancellation for KMB phases.
// It rejects nil rather than replacing the caller's explicit value with a default.
// Implementation:
//   - Stage 1: Capture the caller's context.
//   - Stage 2: Validate it when the callback is applied.
//   - Stage 3: Store the same context without retaining it in the public result.
//
// Behavior highlights:
//   - Opaque legacy SSSP calls cannot be interrupted internally.
//
// Inputs:
//   - ctx: non-nil caller-owned context.Context.
//
// Returns:
//   - SteinerOption: valid setter or error-returning callback.
//
// Errors:
//   - ErrInvalidOption for nil ctx.
//
// Determinism:
//   - Fixed assignment; cancellation timing is caller-controlled.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - The context lifetime remains owned by the caller.
//
// AI-Hints:
//   - Never launch an abandoned background SSSP to simulate early cancellation.
func WithSteinerContext(ctx context.Context) SteinerOption {
	return func(options *SteinerOptions) error {
		if ctx == nil {
			return ErrInvalidOption
		}
		options.Context = ctx
		return nil
	}
}

// WithSteinerPathStorage chooses one implemented KMB witness-memory strategy.
// It does not change shortest-path weights, termination, or output meaning.
// Implementation:
//   - Stage 1: Check the enumerated mode.
//   - Stage 2: Store the requested mode in the private options snapshot.
//
// Behavior highlights:
//   - Both modes use the same public SSSP numeric semantics.
//
// Inputs:
//   - mode: SteinerRecomputePaths or SteinerRetainPaths.
//
// Returns:
//   - SteinerOption: validated policy setter.
//
// Errors:
//   - ErrInvalidOption for unsupported mode.
//
// Determinism:
//   - Pure enum check and assignment.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - No automatic switch under memory pressure occurs.
//
// AI-Hints:
//   - Keep both modes operational; an ignored mode is a broken public contract.
func WithSteinerPathStorage(mode SteinerPathStorage) SteinerOption {
	return func(options *SteinerOptions) error {
		if mode != SteinerRecomputePaths && mode != SteinerRetainPaths {
			return ErrInvalidOption
		}
		options.PathStorage = mode
		return nil
	}
}

// WithSteinerMaxVertices limits admission by the original graph's vertex count.
// It leaves algorithm complexity unchanged for graphs admitted under that limit.
// Implementation:
//   - Stage 1: Reject negative caps.
//   - Stage 2: Assign a positive limit or zero for no caller cap.
//
// Behavior highlights:
//   - A positive cap applies before cloning.
//
// Inputs:
//   - limit: nonnegative maximum original vertices.
//
// Returns:
//   - SteinerOption: scalar limit setter.
//
// Errors:
//   - ErrInvalidOption for a negative limit.
//
// Determinism:
//   - Pure scalar assignment.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - An admitted graph may still require more live memory than its vertex count suggests.
//
// AI-Hints:
//   - Do not interpret the vertex limit as a heap/RSS bound.
func WithSteinerMaxVertices(limit int) SteinerOption {
	return func(options *SteinerOptions) error {
		if limit < 0 {
			return ErrInvalidOption
		}
		options.MaxVertices = limit
		return nil
	}
}

// WithSteinerMaxEdges limits admission by the original graph's edge catalog count.
// Self-loops and disconnected islands are included in the measurement.
// Implementation:
//   - Stage 1: Reject negative caps.
//   - Stage 2: Store the scalar cap without touching graph topology.
//
// Behavior highlights:
//   - A zero cap is unset, not a request for an empty graph.
//
// Inputs:
//   - limit: nonnegative maximum input edges.
//
// Returns:
//   - SteinerOption: scalar limit setter.
//
// Errors:
//   - ErrInvalidOption for negative limits.
//
// Determinism:
//   - Pure scalar assignment.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Counts original edges, not mirrored adjacency arcs.
//
// AI-Hints:
//   - Never exclude loops or islands from a MaxEdges admission check.
func WithSteinerMaxEdges(limit int) SteinerOption {
	return func(options *SteinerOptions) error {
		if limit < 0 {
			return ErrInvalidOption
		}
		options.MaxEdges = limit
		return nil
	}
}

// WithSteinerMaxTerminals bounds the input terminal-set length.
// It prevents accidentally requesting an excessively large closure under an explicit cap.
// Implementation:
//   - Stage 1: Reject negative caps.
//   - Stage 2: Store the nonnegative caller threshold.
//
// Behavior highlights:
//   - A positive limit applies before sorting and closure allocation.
//
// Inputs:
//   - limit: nonnegative maximum number of requested terminal IDs.
//
// Returns:
//   - SteinerOption: scalar limit setter.
//
// Errors:
//   - ErrInvalidOption for a negative limit.
//
// Determinism:
//   - Fixed scalar assignment.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Membership and duplicate checks are separate validation steps.
//
// AI-Hints:
//   - Do not use MaxTerminals as an estimate of total SSSP allocations.
func WithSteinerMaxTerminals(limit int) SteinerOption {
	return func(options *SteinerOptions) error {
		if limit < 0 {
			return ErrInvalidOption
		}
		options.MaxTerminals = limit
		return nil
	}
}
