// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package atr provides exact True Range and Wilder Average True Range computations.
// This file provides bounded transactional streaming state and the shared transition.
package atr

import "context"

// Stream retains one exact Wilder history in bounded storage.
// Create it with New; its zero value is intentionally not initialized.
//
// Implementation:
//   - Stage 1: Retain immutable Options and exact integer period.
//   - Stage 2: Store previous close, count, ATR, and a fixed seed accumulator.
//   - Stage 3: Apply each Push to a local candidate before committing.
//
// Behavior highlights:
//   - No retained bars or period-sized buffer; Push uses no internal heap allocation.
//
// Inputs:
//   - Construct through New with a positive Period.
//
// Returns:
//   - Mutable single-owner state; independent streams can run concurrently.
//
// Errors:
//   - Push/Restore: ErrNilInput for nil receivers; ErrStateVersion for zero values.
//
// Determinism:
//   - Only successfully accepted bars advance the history.
//
// Complexity:
//   - Space O(W), W=34 fixed 64-bit limbs; no growth with Count or Period.
//
// Notes:
//   - Concurrent Push, Reset, Snapshot, or Restore on the same stream is unsupported.
//   - Nil/zero Reset is a no-op and nil/zero Snapshot returns Snapshot{}.
//
// AI-Hints:
//   - Never share one stream between goroutines without external synchronization.
type Stream struct {
	// options Immutable constructor policy; Restore must match it exactly.
	options Options
	// period Validated positive integer coefficient; zero marks an unconstructed Stream.
	period uint64
	// count Successful observation count; errors never increment it.
	count uint64
	// previousClose Last accepted close, not the close of a failed or canceled input.
	previousClose float64
	// current Positive zero during warmup; rounded nonnegative ATR after readiness.
	current float64
	// seed Exact rounded-TR sum during warmup; cleared at the first ready value.
	seed exactSum
}

// Push transactionally accepts one candle and returns its TR and ATR.
// The first accepted candle has no previous close, including after Reset.
//
// Implementation:
//   - Stage 1: Validate context, receiver, finite prices, and candle geometry.
//   - Stage 2: Copy the stream and calculate one canonical candidate transition.
//   - Stage 3: Check context immediately before assigning the candidate to the receiver.
//
// Behavior highlights:
//   - Warmup and mature updates have zero internal heap allocations.
//   - A rejected candle never becomes the previous close of the next step.
//
// Inputs:
//   - ctx: a non-nil, non-canceled context.
//   - high, low, closePrice: finite values with low <= closePrice <= high; negative prices are valid.
//
// Returns:
//   - A detached Value on success; Value{} on every error.
//
// Errors:
//   - ErrNilContext or ctx.Err(): initial/final context check.
//   - ErrNilInput or ErrStateVersion: nil or uninitialized receiver, after context validation.
//   - ErrNonFinite, ErrBar: finite-domain and geometry validation.
//   - ErrOverflow: Count is MaxUint64 or required TR rounds to infinity.
//
// Determinism:
//   - Uses exactly the prior successful close and sequentially rounded ATR.
//
// Complexity:
//   - Time O(W), Space O(W) local candidate, W=34 fixed; O(1) in history length and Period.
//
// Notes:
//   - All failures leave every stream field unchanged.
//   - A custom context may itself allocate or misbehave; no recovery wraps its methods.
//
// AI-Hints:
//   - Do not commit PreviousClose, Count, or seed before the final context check.
func (s *Stream) Push(ctx context.Context, high, low, closePrice float64) (Value, error) {
	// Cancellation/presence is checked before inspecting even a nil receiver.
	if err := validateContext(ctx); err != nil {
		return Value{}, err
	}
	// A nil stream cannot be dereferenced or implicitly constructed.
	if s == nil {
		return Value{}, ErrNilInput
	}
	// A zero struct is not initialized by a coincidentally valid candle.
	if s.period == 0 {
		return Value{}, ErrStateVersion
	}
	// Finite-domain and geometry failures must precede range arithmetic.
	if err := validateBar(high, low, closePrice); err != nil {
		return Value{}, err
	}
	// The fixed array is copied by value; no candidate field aliases live seed storage.
	candidate := *s
	// Only the private candidate accumulates a seed or advances a previous close.
	value, err := candidate.advance(high, low, closePrice)
	if err != nil {
		return Value{}, err
	}
	// A late cancellation discards even a fully computed seed-completion candidate.
	if err = ctx.Err(); err != nil {
		return Value{}, err
	}
	// Commit all interdependent fields together after every fallible stage.
	*s = candidate
	return value, nil
}

// advance applies the only numerical state transition used by batch and streaming APIs.
// The receiver is a private batch state or an unpublished candidate, never an exposed mutation.
//
// Implementation:
//   - Stage 1: Reject an exhausted count and compute the mandatory true range.
//   - Stage 2: Select direct P1 output, exact seed accumulation, or the mature Wilder update.
//   - Stage 3: Set ATR, advance Count, and retain the current close only after numerical success.
//
// Behavior highlights:
//   - TR is published even during warmup; the seed is discarded immediately at readiness.
//
// Inputs:
//   - s: initialized private state with validated phase invariants.
//   - high, low, closePrice: a finite geometry-validated candle in the stream unit.
//
// Returns:
//   - Value: current TR plus +0/false warmup or a ready ATR; Value{} on failure.
//
// Errors:
//   - ErrOverflow: exhausted uint64 count or an unrepresentable required TR.
//
// Determinism:
//   - One accepted candle produces one transition; P1 assigns ATR directly from TR.
//
// Complexity:
//   - Time O(W), Space O(W) fixed arithmetic; W=34 and no retained observation buffer.
//
// Notes:
//   - Only Push performs the final cancellation gate and commits the candidate to live state.
//
// AI-Hints:
//   - Do not update the live PreviousClose before the caller commits this candidate.
//   - Readiness is count-based, not inferred from nonzero ATR or nonzero seed.
func (s *Stream) advance(high, low, closePrice float64) (Value, error) {
	// Check before count+1 as well as before the eventual counter increment.
	if s.count == ^uint64(0) {
		return Value{}, ErrOverflow
	}
	// Count zero disables previous-close access in the mathematical envelope.
	tr, err := trueRange(high, low, s.previousClose, s.count != 0)
	if err != nil {
		return Value{}, err
	}
	// TR is already a valid measurement, independently of ATR readiness.
	value := Value{TR: tr}
	switch {
	// P1 is direct assignment, not a numerically fragile weight-one recurrence.
	case s.period == 1:
		s.current = tr
		value.Valid = true
	// During warmup the integer seed, rather than a rounded average, is retained.
	case s.count < s.period:
		// Add exactly one already-rounded TR measured in subnormal units.
		s.seed.addProduct(tr, 1)
		// The accepted candidate completes exactly P observations at this boundary.
		if s.count+1 == s.period {
			// Divide only after the exact seed is complete; round once.
			s.current = s.seed.roundedQuotient(s.period)
			// Mature checkpoints have no residual seed and must serialize it as zero.
			s.seed = exactSum{}
			value.Valid = true
		}
	// The previous published ATR is the exact binary64 operand of the next affine step.
	default:
		s.current = wilder(s.current, tr, s.period)
		value.Valid = true
	}
	// Before readiness current remains +0; afterward it is the new rounded ATR.
	value.ATR = s.current
	// Numerical success is the only event that consumes an observation count.
	s.count++
	// Subsequent gaps start from this accepted candle, never from a rejected input.
	s.previousClose = closePrice
	return value, nil
}

// Reset clears the accepted history while preserving Options.
// The next successful Push is again a genuine first-bar calculation.
//
// Implementation:
//   - Stage 1: Ignore nil and uninitialized receivers.
//   - Stage 2: Replace all history fields with zero values while retaining configuration.
//
// Behavior highlights:
//   - Discards previous close, warmup sum, current ATR, and count together.
//
// Inputs:
//   - s: a single-owner stream, or nil/zero receiver.
//
// Returns:
//   - No value; configuration is unchanged.
//
// Errors:
//   - None.
//
// Determinism:
//   - Equivalent to a new stream with the same Options.
//
// Complexity:
//   - Time O(W), Space O(W), W=34 fixed; no allocation.
//
// Notes:
//   - Nil and zero-value receivers are safe no-ops, not implicit construction.
//
// AI-Hints:
//   - Never retain the old PreviousClose when starting a new history.
func (s *Stream) Reset() {
	// Reset is deliberately harmless for nil and unconstructed streams.
	if s == nil || s.period == 0 {
		return
	}
	// Clear the seed, count, current ATR, and previous close while preserving policy.
	*s = Stream{options: s.options, period: s.period}
}
