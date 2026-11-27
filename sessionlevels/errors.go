// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package sessionlevels provides previous completed session High/Low/Close levels without look-ahead bias.
// This file declares immutable error identities for errors.Is classification.
package sessionlevels

import "errors"

var (
	// ErrNilContext reports a missing context interface at the entry of Compute.
	// The algorithm must not read price inputs when context ownership is unspecified.
	//
	// Implementation:
	//   - Stage 1: the canonical facade validates the required input category.
	//   - Stage 2: on failure it returns this sentinel or wraps it with %w.
	//   - Stage 3: the caller receives nil Result and an errors.Is-compatible error.
	// Behavior highlights:
	//   - Context inspection runs before all price and session checks.
	// Inputs:
	//   - Classification depends on the corresponding input to Compute.
	// Returns:
	//   - An error identity, not a price or partially usable Result.
	// Errors:
	//   - The sentinel itself is the stable public category; text is diagnostic.
	// Determinism:
	//   - Classification follows Compute's documented input-validation priority.
	// Complexity:
	//   - Sentinel creation O(1); errors.Is traversal depends on wrapper-chain depth.
	// Notes:
	//   - Distinguish the sentinel from context.Canceled and context.DeadlineExceeded.
	// AI-Hints:
	//   - Pass context.Background() when no cancellation policy is required.
	ErrNilContext = errors.New("sessionlevels: nil context")

	// ErrNilInput reports a missing high, low, or closing-price slice.
	// A nil price channel is rejected even if its length would be zero.
	//
	// Implementation:
	//   - Stage 1: the canonical facade validates the required input category.
	//   - Stage 2: on failure it returns this sentinel or wraps it with %w.
	//   - Stage 3: the caller receives nil Result and an errors.Is-compatible error.
	// Behavior highlights:
	//   - Non-nil zero-length channels are valid when sessions is empty.
	// Inputs:
	//   - Classification depends on the corresponding input to Compute.
	// Returns:
	//   - An error identity, not a price or partially usable Result.
	// Errors:
	//   - The sentinel itself is the stable public category; text is diagnostic.
	// Determinism:
	//   - Classification follows Compute's documented input-validation priority.
	// Complexity:
	//   - Sentinel creation O(1); errors.Is traversal depends on wrapper-chain depth.
	// Notes:
	//   - Distinguish the sentinel from context.Canceled and context.DeadlineExceeded.
	// AI-Hints:
	//   - Allocate []float64{} explicitly for an accepted empty history.
	ErrNilInput = errors.New("sessionlevels: nil price input")

	// ErrShape reports different lengths of the three price channels.
	// A result requires one coherent aligned observation at each bar index.
	//
	// Implementation:
	//   - Stage 1: the canonical facade validates the required input category.
	//   - Stage 2: on failure it returns this sentinel or wraps it with %w.
	//   - Stage 3: the caller receives nil Result and an errors.Is-compatible error.
	// Behavior highlights:
	//   - Shape validation follows nil-channel validation and precedes session validation.
	// Inputs:
	//   - Classification depends on the corresponding input to Compute.
	// Returns:
	//   - An error identity, not a price or partially usable Result.
	// Errors:
	//   - The sentinel itself is the stable public category; text is diagnostic.
	// Determinism:
	//   - Classification follows Compute's documented input-validation priority.
	// Complexity:
	//   - Sentinel creation O(1); errors.Is traversal depends on wrapper-chain depth.
	// Notes:
	//   - Distinguish the sentinel from context.Canceled and context.DeadlineExceeded.
	// AI-Hints:
	//   - Correct upstream alignment rather than truncating a mismatched channel.
	ErrShape = errors.New("sessionlevels: mismatched price lengths")

	// ErrSession reports an invalid session partition or an early incomplete interval.
	// The intervals must be strictly adjacent, nonempty and cover all bars.
	//
	// Implementation:
	//   - Stage 1: the canonical facade validates the required input category.
	//   - Stage 2: on failure it returns this sentinel or wraps it with %w.
	//   - Stage 3: the caller receives nil Result and an errors.Is-compatible error.
	// Behavior highlights:
	//   - Indices are checked before the kernel performs any slice access.
	// Inputs:
	//   - Classification depends on the corresponding input to Compute.
	// Returns:
	//   - An error identity, not a price or partially usable Result.
	// Errors:
	//   - The sentinel itself is the stable public category; text is diagnostic.
	// Determinism:
	//   - Classification follows Compute's documented input-validation priority.
	// Complexity:
	//   - Sentinel creation O(1); errors.Is traversal depends on wrapper-chain depth.
	// Notes:
	//   - Distinguish the sentinel from context.Canceled and context.DeadlineExceeded.
	// AI-Hints:
	//   - Complete=false is allowed only for the final interval of a live cut.
	ErrSession = errors.New("sessionlevels: invalid session partition")

	// ErrBar reports an invalid finite low/close/high relation.
	// Each admitted candle must satisfy low <= close <= high.
	//
	// Implementation:
	//   - Stage 1: the canonical facade validates the required input category.
	//   - Stage 2: on failure it returns this sentinel or wraps it with %w.
	//   - Stage 3: the caller receives nil Result and an errors.Is-compatible error.
	// Behavior highlights:
	//   - A later NaN or Inf takes precedence over the first finite bad candle.
	// Inputs:
	//   - Classification depends on the corresponding input to Compute.
	// Returns:
	//   - An error identity, not a price or partially usable Result.
	// Errors:
	//   - The sentinel itself is the stable public category; text is diagnostic.
	// Determinism:
	//   - Classification follows Compute's documented input-validation priority.
	// Complexity:
	//   - Sentinel creation O(1); errors.Is traversal depends on wrapper-chain depth.
	// Notes:
	//   - Distinguish the sentinel from context.Canceled and context.DeadlineExceeded.
	// AI-Hints:
	//   - A signed negative price is valid; never clamp it to zero.
	ErrBar = errors.New("sessionlevels: invalid candle geometry")

	// ErrNonFinite reports a NaN or infinity in any price channel.
	// An undefined observation cannot participate in extrema selection.
	//
	// Implementation:
	//   - Stage 1: the canonical facade validates the required input category.
	//   - Stage 2: on failure it returns this sentinel or wraps it with %w.
	//   - Stage 3: the caller receives nil Result and an errors.Is-compatible error.
	// Behavior highlights:
	//   - The entire input, including warmup and incomplete sessions, is validated.
	// Inputs:
	//   - Classification depends on the corresponding input to Compute.
	// Returns:
	//   - An error identity, not a price or partially usable Result.
	// Errors:
	//   - The sentinel itself is the stable public category; text is diagnostic.
	// Determinism:
	//   - Classification follows Compute's documented input-validation priority.
	// Complexity:
	//   - Sentinel creation O(1); errors.Is traversal depends on wrapper-chain depth.
	// Notes:
	//   - Distinguish the sentinel from context.Canceled and context.DeadlineExceeded.
	// AI-Hints:
	//   - Do not silently replace NaN with zero or ignore a masked bar.
	ErrNonFinite = errors.New("sessionlevels: non-finite price")
)
