// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package atr provides exact True Range and Wilder Average True Range computations.
// This file defines the reachable stable error categories.
package atr

import "errors"

var (
	// ErrNilContext classifies failures where the context interface is nil.
	// Compute or Push rejects nil context before reading inputs.
	//
	// Implementation:
	//   - Validate the relevant API stage; return this stable category.
	//
	// Behavior highlights:
	//   - Classify with errors.Is; the message is not a protocol.
	//
	// Inputs:
	//   - A public API call subject to the condition above.
	//
	// Returns:
	//   - An error category; batch results are nil and failed pushes return Value{}.
	//
	// Errors:
	//   - This sentinel is the originating category, not a wrapper that hides another error.
	//
	// Determinism:
	//   - Sequential validation has the precedence documented by Compute and Push.
	//
	// Complexity:
	//   - Category access is O(1) time and space.
	//
	// Notes:
	//   - Do not reassign exported sentinel variables.
	//
	// AI-Hints:
	//   - Do not compare error text or replace this sentinel with a newly constructed error.
	ErrNilContext = errors.New("atr: the context interface is nil")

	// ErrNilInput classifies failures where a required stream or a nonempty batch channel is nil.
	// Push/Restore reject nil receivers; mixed nil/nonempty channels fail the shape stage.
	//
	// Implementation:
	//   - Validate the relevant API stage; return this stable category.
	//
	// Behavior highlights:
	//   - Classify with errors.Is; the message is not a protocol.
	//
	// Inputs:
	//   - A public API call subject to the condition above.
	//
	// Returns:
	//   - An error category; batch results are nil and failed pushes return Value{}.
	//
	// Errors:
	//   - This sentinel is the originating category, not a wrapper that hides another error.
	//
	// Determinism:
	//   - Sequential validation has the precedence documented by Compute and Push.
	//
	// Complexity:
	//   - Category access is O(1) time and space.
	//
	// Notes:
	//   - Do not reassign exported sentinel variables.
	//
	// AI-Hints:
	//   - Do not compare error text or replace this sentinel with a newly constructed error.
	ErrNilInput = errors.New("atr: a required stream or a nonempty batch channel is nil")

	// ErrInvalidOption classifies failures where Period is less than one.
	// New and Compute reject invalid options without allocating state.
	//
	// Implementation:
	//   - Validate the relevant API stage; return this stable category.
	//
	// Behavior highlights:
	//   - Classify with errors.Is; the message is not a protocol.
	//
	// Inputs:
	//   - A public API call subject to the condition above.
	//
	// Returns:
	//   - An error category; batch results are nil and failed pushes return Value{}.
	//
	// Errors:
	//   - This sentinel is the originating category, not a wrapper that hides another error.
	//
	// Determinism:
	//   - Sequential validation has the precedence documented by Compute and Push.
	//
	// Complexity:
	//   - Category access is O(1) time and space.
	//
	// Notes:
	//   - Do not reassign exported sentinel variables.
	//
	// AI-Hints:
	//   - Do not compare error text or replace this sentinel with a newly constructed error.
	ErrInvalidOption = errors.New("atr: Period is less than one")

	// ErrShape classifies failures where nonnil batch channels have unequal lengths.
	// Compute rejects incompatible high, low, and close lengths.
	//
	// Implementation:
	//   - Validate the relevant API stage; return this stable category.
	//
	// Behavior highlights:
	//   - Classify with errors.Is; the message is not a protocol.
	//
	// Inputs:
	//   - A public API call subject to the condition above.
	//
	// Returns:
	//   - An error category; batch results are nil and failed pushes return Value{}.
	//
	// Errors:
	//   - This sentinel is the originating category, not a wrapper that hides another error.
	//
	// Determinism:
	//   - Sequential validation has the precedence documented by Compute and Push.
	//
	// Complexity:
	//   - Category access is O(1) time and space.
	//
	// Notes:
	//   - Do not reassign exported sentinel variables.
	//
	// AI-Hints:
	//   - Do not compare error text or replace this sentinel with a newly constructed error.
	ErrShape = errors.New("atr: nonnil batch channels have unequal lengths")

	// ErrNonFinite classifies failures where a bar contains NaN or infinity.
	// finite-domain validation precedes candle-geometry validation at each index.
	//
	// Implementation:
	//   - Validate the relevant API stage; return this stable category.
	//
	// Behavior highlights:
	//   - Classify with errors.Is; the message is not a protocol.
	//
	// Inputs:
	//   - A public API call subject to the condition above.
	//
	// Returns:
	//   - An error category; batch results are nil and failed pushes return Value{}.
	//
	// Errors:
	//   - This sentinel is the originating category, not a wrapper that hides another error.
	//
	// Determinism:
	//   - Sequential validation has the precedence documented by Compute and Push.
	//
	// Complexity:
	//   - Category access is O(1) time and space.
	//
	// Notes:
	//   - Do not reassign exported sentinel variables.
	//
	// AI-Hints:
	//   - Do not compare error text or replace this sentinel with a newly constructed error.
	ErrNonFinite = errors.New("atr: a bar contains NaN or infinity")

	// ErrBar classifies failures where low <= close <= high does not hold.
	// invalid candle geometry is rejected before any range arithmetic.
	//
	// Implementation:
	//   - Validate the relevant API stage; return this stable category.
	//
	// Behavior highlights:
	//   - Classify with errors.Is; the message is not a protocol.
	//
	// Inputs:
	//   - A public API call subject to the condition above.
	//
	// Returns:
	//   - An error category; batch results are nil and failed pushes return Value{}.
	//
	// Errors:
	//   - This sentinel is the originating category, not a wrapper that hides another error.
	//
	// Determinism:
	//   - Sequential validation has the precedence documented by Compute and Push.
	//
	// Complexity:
	//   - Category access is O(1) time and space.
	//
	// Notes:
	//   - Do not reassign exported sentinel variables.
	//
	// AI-Hints:
	//   - Do not compare error text or replace this sentinel with a newly constructed error.
	ErrBar = errors.New("atr: low <= close <= high does not hold")

	// ErrOverflow classifies failures where a required range, count, or output byte size is unrepresentable.
	// no partial result or mutated stream is published.
	//
	// Implementation:
	//   - Validate the relevant API stage; return this stable category.
	//
	// Behavior highlights:
	//   - Classify with errors.Is; the message is not a protocol.
	//
	// Inputs:
	//   - A public API call subject to the condition above.
	//
	// Returns:
	//   - An error category; batch results are nil and failed pushes return Value{}.
	//
	// Errors:
	//   - This sentinel is the originating category, not a wrapper that hides another error.
	//
	// Determinism:
	//   - Sequential validation has the precedence documented by Compute and Push.
	//
	// Complexity:
	//   - Category access is O(1) time and space.
	//
	// Notes:
	//   - Do not reassign exported sentinel variables.
	//
	// AI-Hints:
	//   - Do not compare error text or replace this sentinel with a newly constructed error.
	ErrOverflow = errors.New("atr: a required range, count, or output byte size is unrepresentable")

	// ErrStateVersion classifies failures where a stream is uninitialized or a snapshot violates its ABI.
	// Restore validates metadata, options, scalar domains, phase, and exact seed before commit.
	//
	// Implementation:
	//   - Validate the relevant API stage; return this stable category.
	//
	// Behavior highlights:
	//   - Classify with errors.Is; the message is not a protocol.
	//
	// Inputs:
	//   - A public API call subject to the condition above.
	//
	// Returns:
	//   - An error category; batch results are nil and failed pushes return Value{}.
	//
	// Errors:
	//   - This sentinel is the originating category, not a wrapper that hides another error.
	//
	// Determinism:
	//   - Sequential validation has the precedence documented by Compute and Push.
	//
	// Complexity:
	//   - Category access is O(1) time and space.
	//
	// Notes:
	//   - Do not reassign exported sentinel variables.
	//
	// AI-Hints:
	//   - Do not compare error text or replace this sentinel with a newly constructed error.
	ErrStateVersion = errors.New("atr: a stream is uninitialized or a snapshot violates its ABI")
)
