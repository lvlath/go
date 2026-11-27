// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package vwap defines stable sentinel classifications for public failures.
// This file keeps diagnostic wording separate from errors.Is identity and precedence.
package vwap

import "errors"

var (
	// ErrNilContext reports a missing context before any input or receiver inspection.
	// It distinguishes an absent cancellation boundary from cancellation of an existing context.
	//
	// Implementation:
	//   - Stage 1: Compute and Push call validateContext first.
	//   - Stage 2: A nil context returns this sentinel immediately.
	//   - Stage 3: No observation, result allocation, or stream mutation follows.
	//
	// Behavior highlights:
	//   - Classification uses errors.Is; diagnostic text is not a machine protocol.
	//
	// Inputs:
	//   - Compute, ComputeDefault, or Push with ctx == nil.
	//
	// Returns:
	//   - A stable error identity; batch failures return nil and Push failures return Value{}.
	//
	// Errors:
	//   - This symbol is a sentinel, not an operation that returns a second error.
	//
	// Determinism:
	//   - Context presence precedes shape, anchor, numeric, and receiver checks.
	//
	// Complexity:
	//   - O(1) identity comparison; the originating operation documents validation cost.
	//
	// Notes:
	//   - A non-nil context whose Err is non-nil returns that context error instead.
	//
	// AI-Hints:
	//   - Pass context.Background() explicitly when cancellation is not needed.
	ErrNilContext = errors.New("vwap: nil context")

	// ErrNilInput reports a nil Stream or a nil channel mixed with nonempty batch channels.
	// It does not reject a valid empty history merely because its slices are nil.
	//
	// Implementation:
	//   - Stage 1: Push/Restore validate the stream pointer before reading its fields.
	//   - Stage 2: Compute identifies mixed nil/nonempty shapes during shape validation.
	//   - Stage 3: Return the sentinel alone for a nil stream, or joined with ErrShape for that batch mismatch.
	//
	// Behavior highlights:
	//   - Classification uses errors.Is; diagnostic text is not a machine protocol.
	//
	// Inputs:
	//   - A nil Push/Restore receiver, or nil input channel alongside a nonempty channel.
	//
	// Returns:
	//   - A stable error identity; batch failures return nil and Push failures return Value{}.
	//
	// Errors:
	//   - This symbol is a sentinel, not an operation that returns a second error.
	//
	// Determinism:
	//   - Batch shape validation precedes anchor and numeric input access.
	//
	// Complexity:
	//   - O(1) identity comparison; the originating operation documents validation cost.
	//
	// Notes:
	//   - Reset(nil) is a no-op; Snapshot(nil) returns Snapshot{} instead of an error.
	//
	// AI-Hints:
	//   - Use errors.Is for both ErrNilInput and ErrShape on a mixed nil/nonempty batch.
	ErrNilInput = errors.New("vwap: nil input")

	// ErrShape reports unequal lengths of the price, volume, and reset channels.
	// The aligned output contract requires one price, one volume, and one reset flag per row.
	//
	// Implementation:
	//   - Stage 1: Compute checks the three lengths after context validation.
	//   - Stage 2: Return this identity, joined with ErrNilInput for a mixed nil/nonempty shape.
	//   - Stage 3: Do not inspect reset[0], input rows, or allocate result channels after rejection.
	//
	// Behavior highlights:
	//   - Classification uses errors.Is; diagnostic text is not a machine protocol.
	//
	// Inputs:
	//   - Different lengths among price, volume, and reset.
	//
	// Returns:
	//   - A stable error identity; batch failures return nil and Push failures return Value{}.
	//
	// Errors:
	//   - This symbol is a sentinel, not an operation that returns a second error.
	//
	// Determinism:
	//   - Shape precedes the first-anchor check and every row-domain check.
	//
	// Complexity:
	//   - O(1) identity comparison; the originating operation documents validation cost.
	//
	// Notes:
	//   - Three empty channels, including nil slices, form a valid empty history.
	//
	// AI-Hints:
	//   - Do not truncate to the shortest input or infer missing reset flags.
	ErrShape = errors.New("vwap: incompatible channel lengths")

	// ErrAnchor reports an absent explicit first reset in a nonempty computation history.
	// A successful reset marks the start of a segment before its first observation is included.
	//
	// Implementation:
	//   - Stage 1: Compute checks reset[0] only after equal nonempty shapes are established.
	//   - Stage 2: Push requires reset=true when no successful observation has started the stream.
	//   - Stage 3: Return without changing Started, Count, exact sums, or the cached value.
	//
	// Behavior highlights:
	//   - Classification uses errors.Is; diagnostic text is not a machine protocol.
	//
	// Inputs:
	//   - Nonempty Compute with reset[0]==false, or the first Push after New/Reset with reset=false.
	//
	// Returns:
	//   - A stable error identity; batch failures return nil and Push failures return Value{}.
	//
	// Errors:
	//   - This symbol is a sentinel, not an operation that returns a second error.
	//
	// Determinism:
	//   - Anchor validation precedes finite/domain validation at the first row.
	//
	// Complexity:
	//   - O(1) identity comparison; the originating operation documents validation cost.
	//
	// Notes:
	//   - An unsuccessful initial Push never establishes an anchor.
	//
	// AI-Hints:
	//   - Choose the anchor using information already available; a true flag does not prove causality.
	ErrAnchor = errors.New("vwap: first observation requires reset")

	// ErrNonFinite reports NaN or either infinity in a price or volume input.
	// Every observation must have finite numerical inputs even when its volume is zero.
	//
	// Implementation:
	//   - Stage 1: Inspect price and volume before checking finite negative volume.
	//   - Stage 2: Batch scans rows in ascending input order.
	//   - Stage 3: Reject the whole batch or leave the streaming candidate uncommitted.
	//
	// Behavior highlights:
	//   - Classification uses errors.Is; diagnostic text is not a machine protocol.
	//
	// Inputs:
	//   - NaN, +Inf, or -Inf in price or volume.
	//
	// Returns:
	//   - A stable error identity; batch failures return nil and Push failures return Value{}.
	//
	// Errors:
	//   - This symbol is a sentinel, not an operation that returns a second error.
	//
	// Determinism:
	//   - Within each row finite checks precede ErrVolume; batch order is by row index.
	//
	// Complexity:
	//   - O(1) identity comparison; the originating operation documents validation cost.
	//
	// Notes:
	//   - A later non-finite row does not override an earlier finite negative-volume error.
	//
	// AI-Hints:
	//   - Do not replace invalid quotes by zero or skip price validation on zero-volume rows.
	ErrNonFinite = errors.New("vwap: non-finite input")

	// ErrVolume reports a finite negative observation weight, not a signed price.
	// Nonnegative weights are essential to the convexity bound of the exact mean.
	//
	// Implementation:
	//   - Stage 1: Validate both numerical inputs as finite.
	//   - Stage 2: Check volume<0 without restricting the price sign.
	//   - Stage 3: Reject before a reset or a count increment can commit.
	//
	// Behavior highlights:
	//   - Classification uses errors.Is; diagnostic text is not a machine protocol.
	//
	// Inputs:
	//   - Finite volume<0 in Compute or Push.
	//
	// Returns:
	//   - A stable error identity; batch failures return nil and Push failures return Value{}.
	//
	// Errors:
	//   - This symbol is a sentinel, not an operation that returns a second error.
	//
	// Determinism:
	//   - The first invalid row wins; a non-finite value in that same row wins over ErrVolume.
	//
	// Complexity:
	//   - O(1) identity comparison; the originating operation documents validation cost.
	//
	// Notes:
	//   - Both signs of zero volume are accepted; negative finite prices are valid.
	//
	// AI-Hints:
	//   - Never infer short-sale direction from a negative weight; represent signed prices separately.
	ErrVolume = errors.New("vwap: negative volume")

	// ErrOverflow reports an exhausted segment counter or unrepresentable output-size arithmetic.
	// Huge exact products and cumulative volumes do not cause this error by themselves.
	//
	// Implementation:
	//   - Stage 1: Compute bounds aggregate output payload bytes before allocating channels.
	//   - Stage 2: Push checks Count==MaxUint64 when reset is false.
	//   - Stage 3: Return without publishing output or modifying an active segment.
	//
	// Behavior highlights:
	//   - Classification uses errors.Is; diagnostic text is not a machine protocol.
	//
	// Inputs:
	//   - A non-reset Push at MaxUint64, or a batch length that overflows output-byte arithmetic.
	//
	// Returns:
	//   - A stable error identity; batch failures return nil and Push failures return Value{}.
	//
	// Errors:
	//   - This symbol is a sentinel, not an operation that returns a second error.
	//
	// Determinism:
	//   - Domain validation precedes count exhaustion; no float overflow fallback changes the mean.
	//
	// Complexity:
	//   - O(1) identity comparison; the originating operation documents validation cost.
	//
	// Notes:
	//   - A valid reset at MaxUint64 starts a new segment; invalid input still prevents that reset.
	//
	// AI-Hints:
	//   - Do not clamp sums or retry a rejected observation without deciding the intended segment boundary.
	ErrOverflow = errors.New("vwap: counter or allocation-size overflow")

	// ErrStateVersion reports an unconstructed Stream or a structurally invalid continuation snapshot.
	// Its scope includes every schema, lexical, phase, and magnitude check, not only Version.
	//
	// Implementation:
	//   - Stage 1: Reject the zero-value Stream before interpreting a supplied snapshot or observation.
	//   - Stage 2: Restore checks version, algorithm, options, count/start phase, ASCII strings, and exact bounds.
	//   - Stage 3: Publish a detached candidate only after the complete state is admissible.
	//
	// Behavior highlights:
	//   - Classification uses errors.Is; diagnostic text is not a machine protocol.
	//
	// Inputs:
	//   - Push/Restore on Stream{}, or Restore with any invalid Snapshot field or cross-field invariant.
	//
	// Returns:
	//   - A stable error identity; batch failures return nil and Push failures return Value{}.
	//
	// Errors:
	//   - This symbol is a sentinel, not an operation that returns a second error.
	//
	// Determinism:
	//   - Restore admission has fixed field order and no partial assignment to the destination.
	//
	// Complexity:
	//   - O(1) identity comparison; the originating operation documents validation cost.
	//
	// Notes:
	//   - Structural validity is not authentication and does not establish historical trade provenance.
	//
	// AI-Hints:
	//   - Construct with New/NewDefault before Restore and authenticate externally stored checkpoints separately.
	ErrStateVersion = errors.New("vwap: incompatible or invalid state")
)
