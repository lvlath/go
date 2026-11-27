// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package sessionlevels provides previous completed session High/Low/Close levels without look-ahead bias.
// This file owns the sole public computation facade and atomic publication boundary.
package sessionlevels

import "context"

// Compute returns the preceding completed session's high, low, and final close at every bar.
// No price from a session contributes to levels published inside that same session.
//
// Implementation:
//   - Stage 1: validate context, nil inputs, shape, sessions, finiteness, then candle domains.
//   - Stage 2: allocate detached output channels and run the scalar-state kernel.
//   - Stage 3: check context immediately before publishing the complete result.
//
// Behavior highlights:
//   - The first session is masked with positive-zero payloads, even if it is complete.
//   - The last session may be incomplete and still receives its predecessor's levels.
//   - Non-nil empty prices with an empty sessions slice produce non-nil empty channels.
//
// Inputs:
//   - ctx: non-nil usable context; cancellation is polled at most 1024 iterations apart per scan.
//   - high, low, closePrices: non-nil equal-length finite price channels with low <= close <= high.
//   - sessions: ordered nonempty intervals partitioning [0,n); only the last may be incomplete.
//   - opts: reserved zero-value Options{}; no session duration or execution policy is inferred.
//
// Returns:
//   - *Result: owning, input-length output channels, or nil on every error.
//   - error: a sentinel-compatible validation failure or ctx.Err() unchanged.
//
// Errors:
//   - ErrNilContext, then an already observed context error, then ErrNilInput and ErrShape.
//   - ErrSession before price validation, including nonempty sessions for empty prices.
//   - ErrNonFinite anywhere in the input before the first finite ErrBar.
//   - Non-nil empty prices with no sessions succeed; any nil price slice is ErrNilInput.
//
// Determinism:
//   - Increasing session and bar order; earliest equal extrema; exact final-close selection.
//   - A racing context cancellation and input validation failure after entry have no prescribed winner.
//
// Complexity:
//   - Time O(n+s), including validation and zero initialization; output space O(n).
//   - Auxiliary space O(1), with no per-session arrays or allocations in the kernel loops.
//
// Notes:
//   - Inputs are read-only and not retained. Concurrent input mutation is unsupported.
//   - Independent calls may share stable inputs. There is no Stream or calendar integration.
//   - Typed-nil or misbehaving custom contexts and runtime memory exhaustion are not recovered.
//   - Allocation cannot be interrupted; cancellation just after the final poll may follow success.
//
// AI-Hints:
//   - Do not shadow Go's close builtin with a closing-price parameter.
//   - Do not publish a current candidate, replace the final close with an extremum, or infer a calendar.
func Compute(ctx context.Context, high, low, closePrices []float64, sessions []Session, _ Options) (*Result, error) {
	if ctx == nil {
		return nil, ErrNilContext
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	// Check input ownership/shape before consulting the session manifest.
	if err := validateShape(high, low, closePrices); err != nil {
		return nil, err
	}
	// Reject malformed partitions before indexing any bar or publishing a level.
	if err := validateSessions(ctx, sessions, len(high)); err != nil {
		return nil, err
	}
	// Validate all prices, including masked and incomplete-session bars; a later
	// non-finite price takes precedence over an earlier finite malformed candle.
	if err := validatePrices(ctx, high, low, closePrices); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	// Only after whole-batch admission do we allocate detached output buffers.
	// The one shared kernel keeps developing extrema private until session finality.
	result, err := compute(ctx, high, low, closePrices, sessions)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return result, nil
}

// ComputeDefault calculates previous-session levels without an explicit Options argument.
// This convenience entry point preserves the same segmentation, numerical policy, and result contract as Compute.
//
// Implementation:
//   - Stage 1: construct the only supported Options{} value without copying input data.
//   - Stage 2: delegate all validation, cancellation checks, and arithmetic to Compute.
//   - Stage 3: return the original result or error without postprocessing.
//
// Behavior highlights:
//   - Does not infer calendar boundaries or change the allowed final incomplete session.
//
// Inputs:
//   - ctx: non-nil active context with the same cancellation requirements as Compute.
//   - high, low, closePrices: caller-owned price slices in matching units and index order.
//   - sessions: caller-owned partition of [0,n) with all nonfinal sessions completed.
//
// Returns:
//   - *Result: detached per-bar channels; nil on error.
//   - error: the unaltered Compute failure, including context cancellation.
//
// Errors:
//   - ErrNilContext, ErrNilInput, ErrShape, ErrSession, ErrNonFinite, ErrBar or ctx.Err().
//
// Determinism:
//   - Identical to Compute with Options{} including exact binary64 tie selection.
//
// Complexity:
//   - Time O(n+s), Space O(n) outputs plus O(1) auxiliary state; no wrapper allocations.
//
// Notes:
//   - The input slices are never modified or retained; caller mutation during execution is unsupported.
//
// AI-Hints:
//   - Choose this shortcut when the caller already has a validated session manifest.
//   - Do not infer a default 24-hour or 390-bar session from the absence of Options.
func ComputeDefault(ctx context.Context, high, low, closePrices []float64, sessions []Session) (*Result, error) {
	return Compute(ctx, high, low, closePrices, sessions, Options{})
}

// ComputeDaily publishes previous-trading-day levels using an explicit day-session manifest.
// A trading day can cross UTC midnight, be shortened, or exclude overnight bars; this wrapper never derives boundaries.
//
// Implementation:
//   - Stage 1: accept daySessions as already resolved by the caller's exchange-calendar adapter.
//   - Stage 2: delegate all validation and levels to canonical Compute.
//   - Stage 3: return the same complete Result or sentinel-compatible error.
//
// Behavior highlights:
//   - Every day is one nonempty Session; a final live day can be incomplete.
//   - There are no timezone, holiday, DST, or fixed bar-count assumptions.
//
// Inputs:
//   - ctx: non-nil active context; cancellation follows Compute.
//   - high, low, closePrices: aligned, read-only daily intraday observations.
//   - daySessions: complete partition of the observed bars, formed by an external calendar.
//
// Returns:
//   - *Result: detached PDH/PDL/PDC channels and readiness mask.
//   - error: the unchanged canonical error; nil result on failure.
//
// Errors:
//   - ErrNilContext, ErrNilInput, ErrShape, ErrSession, ErrNonFinite, ErrBar or ctx.Err().
//
// Determinism:
//   - Precisely Compute's ascending session and bar order, including signed-zero tie rules.
//
// Complexity:
//   - Time O(n+s), Space O(n) outputs and O(1) auxiliary state; no wrapper allocations.
//
// Notes:
//   - Data availability, corporate-action revisions and the session-policy identity remain caller responsibilities.
//
// AI-Hints:
//   - Supply a genuine exchange-local trading-day manifest; not a UTC-midnight grouping.
//   - A breached PDH after a finalized bar is a feature, never proof of a possible order fill.
func ComputeDaily(ctx context.Context, high, low, closePrices []float64, daySessions []Session) (*Result, error) {
	return Compute(ctx, high, low, closePrices, daySessions, Options{})
}

// ComputeWeekly publishes previous-completed-trading-week levels using a weekly manifest.
// Calendar-aligned weeks may contain different numbers of trading days without changing the algorithm.
//
// Implementation:
//   - Stage 1: accept independently prepared trading-week Session descriptors.
//   - Stage 2: delegate admission and previous-session selection to canonical Compute.
//   - Stage 3: return its detached Result and unchanged error identity.
//
// Behavior highlights:
//   - A partial final week inherits its preceding full week's extrema and final close.
//   - Week segmentation is not inferred from elapsed seconds or a fixed five-day count.
//
// Inputs:
//   - ctx: non-nil active context; cancellation follows Compute.
//   - high, low, closePrices: aligned, read-only observations, potentially aggregated to daily bars.
//   - weekSessions: caller-prepared adjacent trading-week intervals covering [0,n).
//
// Returns:
//   - *Result: detached PWH/PWL/previous-week-close channels and readiness mask.
//   - error: the unmodified Compute error, with nil result for any failure.
//
// Errors:
//   - ErrNilContext, ErrNilInput, ErrShape, ErrSession, ErrNonFinite, ErrBar or ctx.Err().
//
// Determinism:
//   - Same increasing-index selection and strict-comparison ties as canonical Compute.
//
// Complexity:
//   - Time O(n+s), Space O(n) outputs with O(1) auxiliary state and no wrapper allocations.
//
// Notes:
//   - The caller must not merge instruments or venues with incompatible calendars in one input.
//
// AI-Hints:
//   - A four-session holiday week is a valid week; a zero-volume missing day is not a fabricated bar.
//   - Construct weekly boundaries upstream, then use Valid to distinguish warmup from true zero levels.
func ComputeWeekly(ctx context.Context, high, low, closePrices []float64, weekSessions []Session) (*Result, error) {
	return Compute(ctx, high, low, closePrices, weekSessions, Options{})
}
