// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package atr provides exact True Range and Wilder Average True Range computations.
// This file exposes the canonical constructors and owning batch facade.
package atr

import "context"

// DefaultPeriod is the fourteen-observation policy selected only by named default wrappers.
// It does not make Options{} valid and does not turn observations into calendar days.
//
// Implementation:
//   - Stage 1: Select ComputeDefault or NewDefault explicitly.
//   - Stage 2: Pass Period=DefaultPeriod to the canonical facade.
//   - Stage 3: Use the unchanged Wilder seed and recurrence.
//
// Behavior highlights:
//   - Calling Compute or New still requires a positive explicit Period.
//
// Inputs:
//   - No input; this constant expresses the named wrapper policy.
//
// Returns:
//   - int: fourteen accepted observations.
//
// Errors:
//   - None; canonical context, shape, and data checks still apply.
//
// Determinism:
//   - A compile-time count, not a data-dependent period estimate.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - A default count is not a claim that fourteen observations suit every feed.
//
// AI-Hints:
//   - Use ComputePeriod or NewPeriod when the observation horizon is part of your model.
const DefaultPeriod = 14

// New constructs an empty Wilder stream with explicit configuration.
// There is no synthetic close or implicit warmup before the first successful Push.
//
// Implementation:
//   - Stage 1: Validate the full Options value.
//   - Stage 2: Convert the validated positive period to an exact uint64.
//   - Stage 3: Return initialized zero-history state.
//
// Behavior highlights:
//   - Accepts every positive int period, including MaxInt.
//
// Inputs:
//   - opts: the Wilder observation count, copied by value.
//
// Returns:
//   - An independent *Stream on success; nil on error.
//
// Errors:
//   - ErrInvalidOption: Period < 1.
//
// Determinism:
//   - No hidden defaults, wall clock, or random state.
//
// Complexity:
//   - Time O(W), Space O(W), W=34 fixed; one bounded stream object.
//
// Notes:
//   - Use one owner or external synchronization for each returned stream.
//
// AI-Hints:
//   - Do not require a Period-sized buffer or borrow EMA initialization.
func New(opts Options) (*Stream, error) {
	// Preserve the exact count and validate before construction or allocation.
	period, err := validateOptions(opts)
	if err != nil {
		return nil, err
	}
	// Zeroed history carries no previous-close observation.
	return &Stream{options: opts, period: period}, nil
}

// Compute calculates aligned True Range and exact Wilder ATR for a history.
// It validates the full candle domain before one canonical numerical pass.
//
// Implementation:
//   - Stage 1: Check context, options, mixed nil/shape, then each candle in index order.
//   - Stage 2: Check output byte size, allocate owning channels, and apply the same transition as Push.
//   - Stage 3: Check context immediately before returning the complete Result.
//
// Behavior highlights:
//   - Empty input succeeds with non-nil empty channels; n < Period is valid warmup.
//   - Only ATR is masked; the first TR is RN64(high[0]-low[0]) without a fabricated close.
//
// Inputs:
//   - ctx: non-nil context; cancellation is polled at least every 1024 bars in each pass.
//   - high, low, closePrices: equal-length, already ordered finite candle prices; low[i] <= closePrices[i] <= high[i].
//   - opts: an explicit positive Period.
//
// Returns:
//   - A fully caller-owned *Result on success; nil on every failure.
//
// Errors:
//   - ErrNilContext or ctx.Err(): initial, periodic, or final context check.
//   - ErrInvalidOption: invalid period, before shape/data inspection.
//   - ErrNilInput: a nil channel accompanies any nonempty channel; otherwise unequal lengths give ErrShape.
//   - ErrNonFinite then ErrBar: per-index finite-domain and geometry checks.
//   - ErrOverflow: required TR or aggregate output byte count is unrepresentable.
//
// Determinism:
//   - Ascending index order; bit-identical to sequential successful Push and restored continuation.
//
// Complexity:
//   - Time O(n*W), Space O(n+W), W=34 fixed; two linear passes and one numerical transition per bar.
//
// Notes:
//   - Caller inputs must not be mutated concurrently; results never alias them.
//   - Concurrent cancellation and a newly found input error have no fixed relative order.
//   - Process out-of-memory termination is not converted into a recoverable error.
//
// AI-Hints:
//   - Never erase warmup TR, switch to alpha-rounded EMA, or normalize prices by Close here.
func Compute(ctx context.Context, high, low, closePrices []float64, opts Options) (*Result, error) {
	// Context always has precedence over options, shape, and price inspection.
	if err := validateContext(ctx); err != nil {
		return nil, err
	}
	// Preserve the exact count and validate before construction or allocation.
	period, err := validateOptions(opts)
	if err != nil {
		return nil, err
	}
	// Establish aligned channel lengths before indexing any candle.
	if err := validateShape(high, low, closePrices); err != nil {
		return nil, err
	}
	// Admit the whole domain before computing any potentially overflowing TR.
	if err := validateHistory(ctx, high, low, closePrices); err != nil {
		return nil, err
	}
	// Protect aggregate byte arithmetic; this is not a physical-memory guarantee.
	if err := checkOutputSize(len(high)); err != nil {
		return nil, err
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	// This stack-owned stream shares the exact transition used by Push.
	stream := Stream{options: opts, period: period}
	// Separate owning arrays preserve input indexes and independent output channels.
	result := &Result{
		TR: make([]float64, len(high)), ATR: make([]float64, len(high)),
		Valid: make([]bool, len(high)),
	}
	// Perform the sole numerical pass in chronological input order.
	for i := range high {
		// Long numerical histories remain cancelable at the same polling cadence.
		if i%contextStride == 0 {
			if err = ctx.Err(); err != nil {
				return nil, err
			}
		}
		// A private transition may fail without exposing any partial Result.
		value, stepErr := stream.advance(high[i], low[i], closePrices[i])
		if stepErr != nil {
			return nil, stepErr
		}
		// Preserve TR during warmup; Valid describes only the ATR payload.
		result.TR[i], result.ATR[i], result.Valid[i] = value.TR, value.ATR, value.Valid
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	// Publication transfers all output ownership to the caller.
	return result, nil
}

// ComputePeriod calculates TR and Wilder ATR with an explicit observation count.
// It is exactly Compute(ctx, high, low, closePrices, Options{Period: period}).
//
// Implementation:
//   - Stage 1: Form the documented period configuration.
//   - Stage 2: Delegate to the canonical facade without pre-validation or fallback.
//   - Stage 3: Return its result and error unchanged.
//
// Behavior highlights:
//   - No alternative kernel, loss of TR, mask changes, or synthetic seed.
//
// Inputs:
//   - ctx: non-nil context; high, low, closePrices: aligned finite signed candle prices.
//   - period: integer count in [1, MaxInt]; no implicit default.
//
// Returns:
//   - *Result: all three detached channels, or nil on any error.
//
// Errors:
//   - Same errors and precedence as Compute; an invalid period is never repaired.
//
// Determinism:
//   - Exactly the canonical facade order and output bits.
//
// Complexity:
//   - Time O(n*W), Space O(n+W), W=34; the wrapper adds O(1) work.
//
// Notes:
//   - Do not use a shorter valid-only slice: output indexes match the original bars.
//
// AI-Hints:
//   - Use the canonical facade when passing a stored Options value.
//   - Snapshot Algorithm and Version are unchanged by the convenience entry point.
func ComputePeriod(ctx context.Context, high, low, closePrices []float64, period int) (*Result, error) {
	// Configuration adaptation delegates all validation and arithmetic unchanged.
	return Compute(ctx, high, low, closePrices, Options{Period: period})
}

// ComputeDefault calculates TR and Wilder ATR with the explicitly named fourteen-bar policy.
// It is exactly Compute with Options{Period: DefaultPeriod}, retaining all output channels.
//
// Implementation:
//   - Stage 1: Form the documented period configuration.
//   - Stage 2: Delegate to the canonical facade without pre-validation or fallback.
//   - Stage 3: Return its result and error unchanged.
//
// Behavior highlights:
//   - No alternative kernel, loss of TR, mask changes, or synthetic seed.
//
// Inputs:
//   - ctx: non-nil context; high, low, closePrices: aligned finite signed candle prices.
//
// Returns:
//   - *Result: complete owning TR/ATR/Valid channels, or nil on error.
//
// Errors:
//   - Same errors and precedence as Compute with a known valid period.
//
// Determinism:
//   - Exactly the canonical facade order and output bits.
//
// Complexity:
//   - Time O(n*W), Space O(n+W), W=34; no extra output allocation.
//
// Notes:
//   - Default names a count, not a recommended trading horizon or an automatic calendar.
//
// AI-Hints:
//   - Use the canonical facade when passing a stored Options value.
//   - Snapshot Algorithm and Version are unchanged by the convenience entry point.
func ComputeDefault(ctx context.Context, high, low, closePrices []float64) (*Result, error) {
	// Selecting this function is the explicit opt-in to the named default count.
	return Compute(ctx, high, low, closePrices, Options{Period: DefaultPeriod})
}

// NewPeriod constructs an empty stream with an explicit observation count.
// It is exactly New(Options{Period: period}) without a second initialization path.
//
// Implementation:
//   - Stage 1: Form the documented period configuration.
//   - Stage 2: Delegate to the canonical facade without pre-validation or fallback.
//   - Stage 3: Return its result and error unchanged.
//
// Behavior highlights:
//   - No alternative kernel, loss of TR, mask changes, or synthetic seed.
//
// Inputs:
//   - period: integer count in [1, MaxInt], copied into the stream configuration.
//
// Returns:
//   - *Stream: independent bounded single-owner state, or nil on error.
//
// Errors:
//   - ErrInvalidOption: period<1, as classified by New.
//
// Determinism:
//   - Exactly the canonical facade order and output bits.
//
// Complexity:
//   - Time O(W), Space O(W), W=34; one bounded object.
//
// Notes:
//   - Every returned stream starts without a previous close, including period one.
//
// AI-Hints:
//   - Use the canonical facade when passing a stored Options value.
//   - Snapshot Algorithm and Version are unchanged by the convenience entry point.
func NewPeriod(period int) (*Stream, error) {
	// New alone owns configuration validation and first-bar initialization.
	return New(Options{Period: period})
}

// NewDefault constructs an empty stream using the named fourteen-bar Wilder policy.
// It is exactly New(Options{Period: DefaultPeriod}); no historical close is synthesized.
//
// Implementation:
//   - Stage 1: Form the documented period configuration.
//   - Stage 2: Delegate to the canonical facade without pre-validation or fallback.
//   - Stage 3: Return its result and error unchanged.
//
// Behavior highlights:
//   - No alternative kernel, loss of TR, mask changes, or synthetic seed.
//
// Inputs:
//   - No parameters; use NewPeriod or New for an explicit alternative period.
//
// Returns:
//   - *Stream and error: the same return surface as New; a valid default gives nil error.
//
// Errors:
//   - None for the fixed valid configuration; process allocation failure is not recovered.
//
// Determinism:
//   - Exactly the canonical facade order and output bits.
//
// Complexity:
//   - Time O(W), Space O(W), W=34; one bounded object.
//
// Notes:
//   - New(Options{}) remains invalid; this wrapper does not change zero-value semantics.
//
// AI-Hints:
//   - Use the canonical facade when passing a stored Options value.
//   - Snapshot Algorithm and Version are unchanged by the convenience entry point.
func NewDefault() (*Stream, error) {
	// Reuse the canonical constructor rather than manufacturing separate state.
	return New(Options{Period: DefaultPeriod})
}
