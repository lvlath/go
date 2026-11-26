// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ema provides exact affine exponential moving average smoothing.
// This file declares the reachable error categories; text is not a protocol.
package ema

import "errors"

var (
	// ErrNilContext classifies a missing context at Compute or Push entry.
	// It is a stable sentinel for errors.Is; its message is not a machine-readable protocol.
	//
	// Implementation:
	//   - Stage 1: detect the documented boundary failure.
	//   - Stage 2: return ErrNilContext without publishing partial output or state.
	//
	// Behavior highlights:
	//   - Context validation precedes options, receiver checks, and data inspection.
	//
	// Inputs:
	//   - Returned by the public operation whose admission rule failed.
	//
	// Returns:
	//   - error: this package-level identity, potentially wrapped by caller code.
	//
	// Errors:
	//   - a missing context at Compute or Push entry.
	//
	// Determinism:
	//   - Classification follows the public operation precedence, not message text.
	//
	// Complexity:
	//   - O(1) classification at a detected failure; owning validation costs are documented on the operation.
	//
	// Notes:
	//   - Supply a non-nil Context; an already canceled one returns ctx.Err() instead.
	//   - Sentinel variables must not be reassigned by callers.
	//
	// AI-Hints:
	//   - Use errors.Is rather than equality on wrapped errors or substring matching.
	ErrNilContext = errors.New("ema: nil context")

	// ErrNilInput classifies a nil *Stream receiver in Push or Restore.
	// It is a stable sentinel for errors.Is; its message is not a machine-readable protocol.
	//
	// Implementation:
	//   - Stage 1: detect the documented boundary failure.
	//   - Stage 2: return ErrNilInput without publishing partial output or state.
	//
	// Behavior highlights:
	//   - Push validates context first; Restore has no context parameter.
	//
	// Inputs:
	//   - Returned by the public operation whose admission rule failed.
	//
	// Returns:
	//   - error: this package-level identity, potentially wrapped by caller code.
	//
	// Errors:
	//   - a nil *Stream receiver in Push or Restore.
	//
	// Determinism:
	//   - Classification follows the public operation precedence, not message text.
	//
	// Complexity:
	//   - O(1) classification at a detected failure; owning validation costs are documented on the operation.
	//
	// Notes:
	//   - A nil input slice is not this error: Compute treats it as a successful empty history.
	//   - Sentinel variables must not be reassigned by callers.
	//
	// AI-Hints:
	//   - Use errors.Is rather than equality on wrapped errors or substring matching.
	ErrNilInput = errors.New("ema: nil stream")

	// ErrInvalidOption classifies Period outside [1,MaxInt-1] or an unknown Seed.
	// It is a stable sentinel for errors.Is; its message is not a machine-readable protocol.
	//
	// Implementation:
	//   - Stage 1: detect the documented boundary failure.
	//   - Stage 2: return ErrInvalidOption without publishing partial output or state.
	//
	// Behavior highlights:
	//   - New and Compute validate options before allocation; wrappers delegate without defaults.
	//
	// Inputs:
	//   - Returned by the public operation whose admission rule failed.
	//
	// Returns:
	//   - error: this package-level identity, potentially wrapped by caller code.
	//
	// Errors:
	//   - Period outside [1,MaxInt-1] or an unknown Seed.
	//
	// Determinism:
	//   - Classification follows the public operation precedence, not message text.
	//
	// Complexity:
	//   - O(1) classification at a detected failure; owning validation costs are documented on the operation.
	//
	// Notes:
	//   - Use an explicit period; selecting SeedSMA as the zero enum does not validate Options{}.
	//   - Sentinel variables must not be reassigned by callers.
	//
	// AI-Hints:
	//   - Use errors.Is rather than equality on wrapped errors or substring matching.
	ErrInvalidOption = errors.New("ema: invalid option")

	// ErrNonFinite classifies a NaN or infinite observation.
	// It is a stable sentinel for errors.Is; its message is not a machine-readable protocol.
	//
	// Implementation:
	//   - Stage 1: detect the documented boundary failure.
	//   - Stage 2: return ErrNonFinite without publishing partial output or state.
	//
	// Behavior highlights:
	//   - Input validation accepts all finite signs and subnormals; no partially computed batch is returned.
	//
	// Inputs:
	//   - Returned by the public operation whose admission rule failed.
	//
	// Returns:
	//   - error: this package-level identity, potentially wrapped by caller code.
	//
	// Errors:
	//   - a NaN or infinite observation.
	//
	// Determinism:
	//   - Classification follows the public operation precedence, not message text.
	//
	// Complexity:
	//   - O(1) classification at a detected failure; owning validation costs are documented on the operation.
	//
	// Notes:
	//   - Malformed Snapshot.Current is ErrStateVersion instead because it is a state admission failure.
	//   - Sentinel variables must not be reassigned by callers.
	//
	// AI-Hints:
	//   - Use errors.Is rather than equality on wrapped errors or substring matching.
	ErrNonFinite = errors.New("ema: non-finite observation")

	// ErrOverflow classifies an exhausted uint64 Count or unaddressable batch payload.
	// It is a stable sentinel for errors.Is; its message is not a machine-readable protocol.
	//
	// Implementation:
	//   - Stage 1: detect the documented boundary failure.
	//   - Stage 2: return ErrOverflow without publishing partial output or state.
	//
	// Behavior highlights:
	//   - Push checks Count before computing; Compute checks payload capacity before allocation.
	//
	// Inputs:
	//   - Returned by the public operation whose admission rule failed.
	//
	// Returns:
	//   - error: this package-level identity, potentially wrapped by caller code.
	//
	// Errors:
	//   - an exhausted uint64 Count or unaddressable batch payload.
	//
	// Determinism:
	//   - Classification follows the public operation precedence, not message text.
	//
	// Complexity:
	//   - O(1) classification at a detected failure; owning validation costs are documented on the operation.
	//
	// Notes:
	//   - Exact convex EMA arithmetic with admitted finite inputs cannot overflow its final binary64 value.
	//   - Sentinel variables must not be reassigned by callers.
	//
	// AI-Hints:
	//   - Use errors.Is rather than equality on wrapped errors or substring matching.
	ErrOverflow = errors.New("ema: overflow")

	// ErrStateVersion classifies an uninitialized stream or malformed/incompatible snapshot.
	// It is a stable sentinel for errors.Is; its message is not a machine-readable protocol.
	//
	// Implementation:
	//   - Stage 1: detect the documented boundary failure.
	//   - Stage 2: return ErrStateVersion without publishing partial output or state.
	//
	// Behavior highlights:
	//   - Restore checks schema, Options, phase, canonical numbers, and exact seed bounds before mutation.
	//
	// Inputs:
	//   - Returned by the public operation whose admission rule failed.
	//
	// Returns:
	//   - error: this package-level identity, potentially wrapped by caller code.
	//
	// Errors:
	//   - an uninitialized stream or malformed/incompatible snapshot.
	//
	// Determinism:
	//   - Classification follows the public operation precedence, not message text.
	//
	// Complexity:
	//   - O(1) classification at a detected failure; owning validation costs are documented on the operation.
	//
	// Notes:
	//   - A valid snapshot is not authenticated provenance; protect external persistence separately.
	//   - Sentinel variables must not be reassigned by callers.
	//
	// AI-Hints:
	//   - Use errors.Is rather than equality on wrapped errors or substring matching.
	ErrStateVersion = errors.New("ema: invalid or incompatible state")
)
