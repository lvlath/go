// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package atr provides exact True Range and Wilder Average True Range computations.
// This file centralizes context, shape, finite-domain, and candle validation.
package atr

import (
	"context"
	"math"
)

// The polling interval and output payload accounting are fixed operational limits.
const (
	// contextStride caps bars between cancellation polls in each complete batch pass.
	contextStride = 1024
	// resultBytesPerBar accounts for two binary64 values and one boolean per input bar.
	resultBytesPerBar = 2*8 + 1
)

// validateContext rejects an absent context before calling its cancellation method.
// The original cancellation error is preserved rather than wrapped or reclassified.
//
// Implementation:
//   - Stage 1: Test interface presence without inspecting its dynamic representation.
//   - Stage 2: Ask the valid context for its current error.
//   - Stage 3: Return the detected failure before prices or options are accessed.
//
// Behavior highlights:
//   - No reflection, hidden background context, deadline inference, or allocation.
//
// Inputs:
//   - ctx: any correctly implemented Context; an absent interface is rejected.
//
// Returns:
//   - error: nil when active, ErrNilContext when absent, otherwise ctx.Err().
//
// Errors:
//   - ErrNilContext: nil interface; ctx.Err(): observed cancellation or deadline.
//
// Determinism:
//   - Exactly one Err call after the nil check.
//
// Complexity:
//   - Time O(1), Space O(1), excluding custom Context method costs.
//
// Notes:
//   - A typed-nil broken Context implementation is not repaired by the library.
//
// AI-Hints:
//   - Use context.Background explicitly when cancellation is not needed.
func validateContext(ctx context.Context) error {
	// Presence is checked before calling a method through the context interface.
	if ctx == nil {
		return ErrNilContext
	}
	// Return the original context error identity without another allocation.
	return ctx.Err()
}

// validateOptions converts an admissible observation count into an exact integer weight.
// The positive int domain is preserved without converting Period to float64.
//
// Implementation:
//   - Stage 1: Reject Period below one.
//   - Stage 2: Convert the validated positive int to uint64.
//   - Stage 3: Return the exact count used for readiness and the Wilder denominator.
//
// Behavior highlights:
//   - MaxInt remains valid; ATR never needs the EMA denominator Period+1.
//
// Inputs:
//   - opts: complete value configuration; Period counts accepted bars.
//
// Returns:
//   - uint64: positive exact period, or zero with an error.
//
// Errors:
//   - ErrInvalidOption: Period is zero or negative.
//
// Determinism:
//   - One ordered domain check, followed by a lossless conversion.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Options{} is invalid even though named default wrappers are available.
//
// AI-Hints:
//   - Do not compute a rounded alpha here; retain exact integer coefficients.
func validateOptions(opts Options) (uint64, error) {
	// Zero is not a request for an implicit default.
	if opts.Period < 1 {
		return 0, ErrInvalidOption
	}
	// A positive int is representable in uint64 on both supported word widths.
	return uint64(opts.Period), nil
}

// validateShape validates the three channel lengths without reading candle values.
// An absent nonempty input channel is distinct from an allocated but mismatched channel.
//
// Implementation:
//   - Stage 1: Detect whether any channel contains observations.
//   - Stage 2: Reject mixed nil/nonempty inputs, then unequal lengths.
//   - Stage 3: Accept equal lengths, including every nil/empty zero-length combination.
//
// Behavior highlights:
//   - No channel is padded, truncated, reordered, or allocated.
//
// Inputs:
//   - high, low, closePrices: caller-owned read-only price slices.
//
// Returns:
//   - error: nil for equal-length channels.
//
// Errors:
//   - ErrNilInput: nil channel alongside nonempty data; ErrShape: other length mismatch.
//
// Determinism:
//   - Presence classification precedes length classification.
//
// Complexity:
//   - Time O(1), Space O(1); only slice headers are inspected.
//
// Notes:
//   - A non-nil empty slice and a nil slice both represent an empty history.
//
// AI-Hints:
//   - Do not treat mixed nil/nonempty inputs as a successful empty batch.
func validateShape(high, low, closePrices []float64) error {
	// Length, not nilness, determines whether an observation is present.
	nonempty := len(high) != 0 || len(low) != 0 || len(closePrices) != 0
	// A missing companion channel has its own stable input category.
	if nonempty && (high == nil || low == nil || closePrices == nil) {
		return ErrNilInput
	}
	// Never shorten the longer channels to hide a shape mismatch.
	if len(high) != len(low) || len(high) != len(closePrices) {
		return ErrShape
	}
	return nil
}

// finite recognizes the finite binary64 price domain without a sign restriction.
// Both signed zeros and all subnormal values are legitimate represented prices.
//
// Implementation:
//   - Stage 1: Reject NaN.
//   - Stage 2: Reject either infinity.
//   - Stage 3: Accept every other represented float64 value.
//
// Behavior highlights:
//   - Negative prices are data, not validation failures.
//
// Inputs:
//   - x: any float64 bit pattern.
//
// Returns:
//   - bool: true exactly for finite values.
//
// Errors:
//   - None; callers map a false result to their boundary-specific error.
//
// Determinism:
//   - Pure scalar predicates, with no floating arithmetic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Snapshot failures use ErrStateVersion rather than the input ErrNonFinite.
//
// AI-Hints:
//   - Do not share a positive-price validator with this signed candle domain.
func finite(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) }

// validateBar checks finite prices before testing exact candle geometry.
// No range subtraction occurs until the entire current candle passes this boundary.
//
// Implementation:
//   - Stage 1: Reject any non-finite high, low, or closePrice.
//   - Stage 2: Check low <= high and closePrice in the closed interval [low, high].
//   - Stage 3: Return success without altering any supplied price.
//
// Behavior highlights:
//   - Flat bars, boundary closes, signed zeros, and negative prices are accepted.
//
// Inputs:
//   - high, low, closePrice: represented prices in one common unit.
//
// Returns:
//   - error: nil for a finite valid candle.
//
// Errors:
//   - ErrNonFinite: a non-finite price; ErrBar: finite but inconsistent geometry.
//
// Determinism:
//   - Finite-domain failure wins over geometry at the same input index.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - No epsilon repair or clamp changes the observation.
//
// AI-Hints:
//   - NaN comparisons alone cannot enforce geometry; preserve the finite check first.
func validateBar(high, low, closePrice float64) error {
	// NaN does not participate in a meaningful interval comparison.
	if !finite(high) || !finite(low) || !finite(closePrice) {
		return ErrNonFinite
	}
	// Inclusive boundaries accept flat candles and endpoint closes.
	if low > high || closePrice < low || closePrice > high {
		return ErrBar
	}
	return nil
}

// validateHistory checks every candle before the numerical batch pass starts.
// This prevents an earlier overflowing TR from hiding a later invalid input candle.
//
// Implementation:
//   - Stage 1: Traverse source indexes in ascending order.
//   - Stage 2: Poll context at each contextStride boundary, then validate the current bar.
//   - Stage 3: Stop on the first detected failure or accept the complete domain.
//
// Behavior highlights:
//   - One O(n) admission pass; no output or stream state is published.
//
// Inputs:
//   - ctx: already validated non-nil context.
//   - high, low, closePrices: already shape-validated read-only equal-length channels.
//
// Returns:
//   - error: nil only when all observed admission checks succeed.
//
// Errors:
//   - ctx.Err(): periodic cancellation; ErrNonFinite or ErrBar: current candle.
//
// Determinism:
//   - At each index, finite checks precede geometry; no whole-array non-finite sweep.
//
// Complexity:
//   - Time O(n), Space O(1); at most contextStride bars between cancellation polls.
//
// Notes:
//   - Numerical results remain causal even though batch admission scans the full history.
//
// AI-Hints:
//   - Do not merge this pass with arithmetic and thereby change error precedence.
func validateHistory(ctx context.Context, high, low, closePrices []float64) error {
	// The first invalid index wins; no arithmetic is performed in this pass.
	for i := range high {
		// Poll even at index zero, then at fixed bounded intervals.
		if i%contextStride == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		// Shape was checked before these parallel indexed reads.
		if err := validateBar(high[i], low[i], closePrices[i]); err != nil {
			return err
		}
	}
	return nil
}

// checkOutputSize guards aggregate output byte arithmetic before slice allocation.
// It checks representable allocation accounting, not the amount of free physical memory.
//
// Implementation:
//   - Stage 1: Derive the largest supported signed-int byte count.
//   - Stage 2: Compare n with that bound divided by the per-bar output payload.
//   - Stage 3: Reject arithmetic overflow before multiplying lengths or allocating channels.
//
// Behavior highlights:
//   - The division-based guard cannot itself overflow.
//
// Inputs:
//   - n: nonnegative input length obtained from a real Go slice.
//
// Returns:
//   - error: nil when aggregate payload accounting fits int.
//
// Errors:
//   - ErrOverflow: two float64 channels plus one boolean channel exceed the byte bound.
//
// Determinism:
//   - A fixed platform-width comparison; no runtime memory probing.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Allocator headers and process out-of-memory termination are not recoverable errors here.
//
// AI-Hints:
//   - Do not call recover around make or confuse this guard with a user-configured budget.
func checkOutputSize(n int) error {
	// Compare before multiplication so byte accounting cannot wrap.
	if n > int(^uint(0)>>1)/resultBytesPerBar {
		return ErrOverflow
	}
	return nil
}
