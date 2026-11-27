// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package vwap defines canonical calculation/construction boundaries and default facades.
// This file delegates every numerical transition to the same exact state machine.
package vwap

import (
	"context"
	"math"
)

// Compute returns exact volume-weighted prefix means within explicitly reset segments.
// Each result is the nearest-even rounding of exact sums of the supplied binary64 inputs.
//
// Implementation:
//   - Stage 1: validate context, the empty option domain, shape, anchor, and rows.
//   - Stage 2: allocate owning channels and execute one ordered numerical pass.
//   - Stage 3: check context immediately before publishing the complete result.
//
// Behavior highlights:
//   - Zero cumulative volume is masked; huge finite price-volume products remain exact.
//
// Inputs:
//   - ctx: non-nil context, polled at least every 1024 rows in each linear pass.
//   - price, volume: read-only equal-length finite channels; volume must be nonnegative.
//   - reset: equal-length flags; reset[0] must be true for nonempty input.
//   - opts: Options{}, the only configuration value in v1.
//
// Returns:
//   - A non-nil Result with owning aligned channels, including non-nil empty channels.
//
// Errors:
//   - ErrNilContext, ctx.Err(), ErrShape, ErrNilInput, ErrAnchor, ErrNonFinite,
//     ErrVolume, or ErrOverflow; every error returns a nil Result.
//
// Determinism:
//   - Validation follows increasing row indices; each row checks finiteness before volume.
//   - Numerical output is bit-identical to the same successful sequence of Push calls.
//
// Complexity:
//   - Time O(n*L), Space O(n+L), L=67 fixed limbs; validation plus one numerical pass.
//
// Notes:
//   - Shape precedes anchor; a mixed nil/nonempty shape matches both shape and nil errors.
//   - Byte-size guards do not promise recovery from system memory exhaustion.
//
// AI-Hints:
//   - Never infer session boundaries or retrospective pivot anchors from these arrays.
//   - Caller inputs must remain unchanged until the call returns.
func Compute(ctx context.Context, price, volume []float64, reset []bool, opts Options) (*Result, error) {
	// Stage 1: enforce context precedence before any slice is inspected.
	if err := validateContext(ctx); err != nil {
		return nil, err
	}
	// Options{} has exactly one value, so configuration admission has no failure branch.
	// Shape and first-anchor checks protect every subsequent aligned slice access.
	if err := validateShape(price, volume, reset); err != nil {
		return nil, err
	}
	// Validate all rows before allocating outputs; numerical work never hides a later input error.
	if err := validateHistory(ctx, price, volume); err != nil {
		return nil, err
	}
	// Stage 2: check byte-size arithmetic before creating detached output channels.
	n := len(price)
	// The logical channel payload is one binary64 (8 bytes) plus one bool (1 byte).
	// Fixed slice headers and allocator overhead are not a promise of available RAM.
	const outputBytesPerRow = 8 + 1
	if n > math.MaxInt/outputBytesPerRow {
		return nil, ErrOverflow
	}
	// make with length zero intentionally returns non-nil empty slices on successful empty input.
	result := &Result{Values: make([]float64, n), Valid: make([]bool, n)}
	// This local state shares only the canonical transition, not caller memory or a live Stream.
	state := Stream{initialized: true, options: opts}
	// Increasing row order fixes both reset boundaries and prefix publication order.
	for i := range price {
		if i&pollMask == 0 {
			// Poll during long calculations; cancellation discards all locally accumulated output.
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		// n fits int, so a batch segment cannot exhaust the larger uint64 Count domain.
		value := state.accept(price[i], volume[i], reset[i])
		result.Values[i], result.Valid[i] = value.Value, value.Valid
	}
	// Stage 3: never publish a partially computed Result after observed cancellation.
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return result, nil
}

// New constructs an empty single-owner VWAP stream with the complete configuration.
// The first accepted observation must explicitly pass reset=true to Push.
//
// Implementation:
//   - Stage 1: admit the single empty Options value.
//   - Stage 2: initialize the construction marker and copy Options.
//   - Stage 3: return an empty stream with zero count and exact sums.
//
// Behavior highlights:
//   - There is no implicit first anchor and no retained history buffer.
//
// Inputs:
//   - opts: Options{}, the complete exact-weighted v1 configuration.
//
// Returns:
//   - A new independent *Stream and nil error.
//
// Errors:
//   - None are reachable for the v1 empty Options domain.
//
// Determinism:
//   - Every new stream has the same canonical empty Snapshot.
//
// Complexity:
//   - Time O(L), Space O(L), L=67 fixed limbs; one stream object allocation.
//
// Notes:
//   - Callers handle the constructor error uniformly even though Options{} has no invalid value.
//
// AI-Hints:
//   - Do not use Stream{} as an alternative constructor.
//   - Do not add a fake invalid option to exercise ErrInvalidOption.
func New(opts Options) (*Stream, error) {
	// The initialized marker is the sole distinction between a fresh stream and Stream{}.
	return &Stream{initialized: true, options: opts}, nil
}

// ComputeDefault calculates owning segmented VWAP channels with the empty configuration.
// It removes only Options{} boilerplate and still requires every reset flag explicitly.
//
// Implementation:
//   - Stage 1: select the complete zero-value configuration.
//   - Stage 2: delegate validation, allocation, and exact accumulation to Compute.
//   - Stage 3: return its canonical Result and error without modification.
//
// Behavior highlights:
//   - No anchor is synthesized; a nonempty reset slice must still begin with true.
//
// Inputs:
//   - ctx: non-nil context, observed only by Compute.
//   - price, volume: equal-length read-only channels, finite prices and nonnegative volumes.
//   - reset: caller-supplied segment starts, applied before their corresponding rows.
//
// Returns:
//   - *Result: detached aligned Values and Valid, or nil on any error.
//
// Errors:
//   - Exactly Compute's categories and precedence, including ErrAnchor and ctx.Err().
//
// Determinism:
//   - Identical output bits, validation order, and context polls to Compute with Options{}.
//
// Complexity:
//   - Time O(n*L), Space O(n+L), L=67; the wrapper adds no buffers or passes.
//
// Notes:
//   - Session and anchored workflows differ only in the explicit reset sequence.
//
// AI-Hints:
//   - Default does not mean a default period, price proxy, session calendar, or first anchor.
//   - Keep this wrapper as direct delegation so cancellation and ownership cannot drift.
func ComputeDefault(ctx context.Context, price, volume []float64, reset []bool) (*Result, error) {
	// Reuse the canonical boundary, including its single publication check.
	return Compute(ctx, price, volume, reset, Options{})
}

// NewDefault constructs an empty exact VWAP stream with the zero-value configuration.
// Its first accepted observation still needs Push(ctx, price, volume, true).
//
// Implementation:
//   - Stage 1: select Options{}, the complete v1 configuration.
//   - Stage 2: delegate construction exclusively to New.
//   - Stage 3: return the same stream/error pair without touching its state.
//
// Behavior highlights:
//   - The resulting stream has Started=false, Count=0, and both exact sums zero.
//
// Inputs:
//   - None; prices, volumes, and causal resets are supplied to Push by the owner.
//
// Returns:
//   - *Stream: a new independent, single-owner stream and nil error.
//
// Errors:
//   - Identical to New(Options{}); no configuration error is reachable in v1.
//
// Determinism:
//   - The empty Snapshot has the same public field values as New(Options{}).Snapshot().
//
// Complexity:
//   - Time O(L), Space O(L), L=67; no allocation beyond New's stream object.
//
// Notes:
//   - Reset and Restore retain the same rules as streams constructed through New.
//
// AI-Hints:
//   - Do not add a hidden priming Push here: first-anchor validation belongs to the caller.
//   - Prefer New when explicitly retaining the Options carrier alongside pipeline metadata.
func NewDefault() (*Stream, error) {
	// Preserve one constructor and one construction marker policy.
	return New(Options{})
}
