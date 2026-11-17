// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package rsi exposes the canonical Wilder RSI computation and narrow convenience facades.
// This file normalizes caller policy, validates complete histories, and delegates all mathematics to one kernel.
package rsi

// New creates an independent Wilder RSI stream with a frozen smoothing period.
// It prepares a configuration fingerprint once; subsequent Push calls never hash or allocate.
//
// Implementation:
//   - Stage 1: normalize Period, rejecting an invalid or unrepresentable seed window.
//   - Stage 2: compute the versioned configuration digest from the normalized period.
//   - Stage 3: publish a fresh stream with zero accepted prices and empty recurrence state.
//
// Behavior highlights:
//   - Options{Period: 0} selects 14 changes, not 14 closing prices.
//   - Streams initialized with Period=0 and Period=14 have identical configuration hashes.
//
// Inputs:
//   - opts: value-owned configuration; Period is an integer number of price differences.
//
// Returns:
//   - *Stream: independent initialized mutable state, or nil when construction fails.
//
// Errors:
//   - ErrInvalidOption: negative Period or an overflow of Period+1.
//
// Determinism:
//   - Configuration hashing uses a fixed algorithm identifier and little-endian period bytes.
//
// Complexity:
//   - Time O(1), Space O(1), including the new stream and its configuration digest.
//
// Notes:
//   - The caller owns the stream; a single stream must not be mutated concurrently.
//   - New never touches price data and never performs a warmup step.
//
// AI-Hints:
//   - Construct a separate stream per independently ordered instrument/feed.
//   - Do not create a stream per tick: reuse the initialized stream and Reset on real gaps.
func New(opts Options) (*Stream, error) {
	period, err := normalizeOptions(opts)
	if err != nil {
		return nil, err
	}

	return &Stream{period: period, config: configHash(period)}, nil
}

// NewDefault creates a stream with Wilder's standard 14-change smoothing period.
// It is exactly New(Options{}), without a separate initialization or update policy.
//
// Implementation:
//   - Stage 1: select the zero-value options.
//   - Stage 2: delegate initialization and hashing to New.
//
// Behavior highlights:
//   - The first valid Push result requires 15 accepted closes.
//
// Returns:
//   - *Stream: a freshly initialized stream; nil on error.
//
// Errors:
//   - Identical to New(Options{}); no new failure categories are introduced.
//
// Determinism:
//   - Exactly the same initial state as New(Options{Period: 14}).
//
// Complexity:
//   - Time O(1), Space O(1), including the stream allocation.
//
// Notes:
//   - Use NewPeriod when the feature pipeline uses a nondefault period.
//
// AI-Hints:
//   - A default stream is not already warmed up; honor the valid flag from Push.
func NewDefault() (*Stream, error) {
	return New(Options{})
}

// NewPeriod creates a stream for an explicitly requested number of price changes.
// It allows period-driven feature pipelines to construct a stream without assembling Options.
//
// Implementation:
//   - Stage 1: place period into Options without changing its meaning.
//   - Stage 2: delegate all validation, hashing, and initialization to New.
//
// Behavior highlights:
//   - Period=0 retains the canonical default of 14; negative values are rejected.
//
// Inputs:
//   - period: number of consecutive price differences in the initial seed.
//
// Returns:
//   - *Stream: an initialized stream, or nil on error.
//
// Errors:
//   - ErrInvalidOption: the forwarded period is invalid.
//
// Determinism:
//   - Bit-for-bit equivalent initialization to New(Options{Period: period}).
//
// Complexity:
//   - Time O(1), Space O(1), including the stream allocation.
//
// Notes:
//   - Each market instrument requires independent recurrence state.
//
// AI-Hints:
//   - Never share one Stream across instruments, even when periods are equal.
func NewPeriod(period int) (*Stream, error) {
	return New(Options{Period: period})
}

// Compute calculates aligned Wilder RSI samples for an ordered closing-price history.
// It returns a detached value/mask pair, retaining no reference to the input slice.
//
// Implementation:
//   - Stage 1: normalize options, check nil/empty shape, then scan every price for domain errors.
//   - Stage 2: allocate the two result slices and feed prices in index order to stepWilder.
//   - Stage 3: publish the completed Result only after every transition succeeds.
//
// Behavior highlights:
//   - Values[i] is +0 with Valid[i]=false for all i before the first P changes.
//   - A short but nonempty valid history succeeds, even if all samples are still warming up.
//   - First valid sample is at zero-based index P; zero RSI with Valid=true is meaningful.
//
// Inputs:
//   - closes: non-nil, nonempty, caller-owned read-only slice of strictly positive finite prices.
//   - opts: value-owned smoothing policy; Period=0 normalizes to 14.
//
// Returns:
//   - *Result: detached Values and Valid slices of exactly len(closes), or nil on any error.
//
// Errors:
//   - ErrInvalidOption: option validation precedes any access to closes.
//   - ErrNilInput, ErrEmptyInput: missing history or an allocated zero-length history.
//   - ErrNonFinite, ErrDomain: first invalid price in ascending index order.
//   - ErrOverflow: unrepresentable transition or output; no partial Result is returned.
//
// Determinism:
//   - Prices are validated and processed in ascending index order, using the same stepWilder as Push.
//   - There is no sorting, gap detection, parallel reduction, or alternative batch-only formula.
//
// Complexity:
//   - Time O(n), Space O(n) for detached outputs plus O(1) recurrence state.
//   - The complete input scan precedes all result allocations and arithmetic.
//
// Notes:
//   - Concurrent modification of closes during this call is unsupported.
//   - RSI has no timestamp semantics; callers must supply a clean, ordered series.
//
// AI-Hints:
//   - Always check Valid before consuming Values, especially for short histories.
//   - Never clamp numeric failures or substitute missing closes with fabricated prices.
func Compute(closes []float64, opts Options) (*Result, error) {
	period, err := normalizeOptions(opts)
	if err != nil {
		return nil, err
	}
	if closes == nil {
		return nil, ErrNilInput
	}
	if len(closes) == 0 {
		return nil, ErrEmptyInput
	}

	// Validate the entire caller-owned history before allocating output. The first
	// invalid sample wins, irrespective of whether it falls inside warmup.
	for _, price := range closes {
		if err := validatePrice(price); err != nil {
			return nil, err
		}
	}

	// The shared step kernel owns all numerical decisions. Unlike a streaming
	// feed, a failed batch computation never publishes its partially filled slices.
	out := &Result{Values: make([]float64, len(closes)), Valid: make([]bool, len(closes))}
	var state wilderState
	for i, price := range closes {
		next, value, valid, err := stepWilder(state, price, period)
		if err != nil {
			return nil, err
		}
		state = next
		out.Values[i], out.Valid[i] = value, valid
	}

	return out, nil
}

// ComputeDefault computes the standard 14-period RSI history with an aligned validity mask.
// It reduces boilerplate when a caller intentionally uses the canonical default period.
//
// Implementation:
//   - Stage 1: choose zero-value Options for the default.
//   - Stage 2: delegate all validation and arithmetic to Compute.
//   - Stage 3: return exactly the canonical Result and error.
//
// Behavior highlights:
//   - Histories of fewer than 15 valid prices return all-invalid masks without error.
//
// Inputs:
//   - closes: caller-owned, read-only, finite positive closing prices.
//
// Returns:
//   - *Result: detached aligned RSI and mask, or nil on failure.
//
// Errors:
//   - Identical to Compute(closes, Options{}), including the same precedence.
//
// Determinism:
//   - Exactly the same index order and output bits as the canonical default call.
//
// Complexity:
//   - Time O(n), Space O(n), including the detached result allocation.
//
// Notes:
//   - It is a projection of Compute, not an independently optimized calculation.
//
// AI-Hints:
//   - Do not interpret invalid warmup zeros as actual RSI measurements.
func ComputeDefault(closes []float64) (*Result, error) {
	return Compute(closes, Options{})
}

// ComputePeriod computes a complete RSI series with an explicit smoothing period.
// It makes a period-parametrized factor pipeline concise without introducing new mathematics.
//
// Implementation:
//   - Stage 1: set Period directly in the canonical Options value.
//   - Stage 2: delegate validation, recurrence, and publication to Compute.
//
// Behavior highlights:
//   - Period=0 still means 14, matching every other entry point.
//
// Inputs:
//   - closes: caller-owned, read-only positive finite prices.
//   - period: requested number of seed changes, subject to canonical validation.
//
// Returns:
//   - *Result: detached value and validity arrays, or nil on any failure.
//
// Errors:
//   - Identical to Compute(closes, Options{Period: period}).
//
// Determinism:
//   - Exactly the same sample order and output bits as Compute with these options.
//
// Complexity:
//   - Time O(n), Space O(n), including both result slices.
//
// Notes:
//   - Use separate calls for each period; different periods never share smoothing state.
//
// AI-Hints:
//   - Period counts differences, so P periods need at least P+1 prices.
func ComputePeriod(closes []float64, period int) (*Result, error) {
	return Compute(closes, Options{Period: period})
}

// Latest returns the RSI observation at the last index of a complete price history.
// It retains the validity flag, so a history shorter than warmup is not misreported as RSI=0.
//
// Implementation:
//   - Stage 1: invoke Compute for the complete validation and canonical recurrence.
//   - Stage 2: project the final aligned value and validity bit from the Result.
//   - Stage 3: return only that observation, discarding the temporary batch artifact.
//
// Behavior highlights:
//   - Short, valid histories return (0, false, nil); invalid histories return an error.
//   - No earlier samples are skipped or replaced, even when only the last value is requested.
//
// Inputs:
//   - closes: caller-owned, read-only, nonempty finite positive closes in temporal order.
//   - opts: canonical smoothing period policy.
//
// Returns:
//   - value: last RSI when valid, otherwise zero.
//   - valid: whether at least Period changes have been consumed.
//   - error: the exact classification from Compute, with zero/false on failure.
//
// Errors:
//   - Identical to Compute, including option/shape/price precedence and ErrOverflow.
//
// Determinism:
//   - Exactly the last index of the canonical Result; no alternate numerical path.
//
// Complexity:
//   - Time O(n), Space O(n) transient, because this wrapper delegates to Compute.
//
// Notes:
//   - For long-running online pipelines use New/Push, which retains only O(1) state.
//   - Latest is a convenience projection, not an O(1)-memory historical scanner.
//
// AI-Hints:
//   - Never assume valid=true merely because closes is nonempty.
//   - Use Stream.Push when zero-allocation tick processing is required.
func Latest(closes []float64, opts Options) (value float64, valid bool, err error) {
	result, err := Compute(closes, opts)
	if err != nil {
		return 0, false, err
	}
	last := len(result.Values) - 1
	return result.Values[last], result.Valid[last], nil
}
