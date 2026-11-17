// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package macd defines the canonical batch and streaming construction boundaries.
// This file provides explicit-period conveniences without duplicating the EMA kernel.
package macd

// DefaultOptions returns the conventional 12/26/9 MACD configuration explicitly.
// It is equivalent to the zero Options value after New or Compute normalization.
//
// Implementation:
//   - Stage 1: choose the fixed default period constants.
//   - Stage 2: construct an independent value (no mutable shared defaults).
//   - Stage 3: return it for caller inspection or modification.
//
// Behavior highlights:
//   - Modifying the returned Options does not affect another call or stream.
//
// Returns:
//   - Options: Fast=12, Slow=26, Signal=9, all explicitly positive.
//
// Determinism:
//   - The constants are fixed for algorithm version macd/sma-ema/v1.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Options{} also selects these defaults; this helper is for self-describing callers.
//
// AI-Hints:
//   - Do not mistake a zero individual field for an EMA of period zero.
func DefaultOptions() Options {
	return Options{Fast: defaultFastPeriod, Slow: defaultSlowPeriod, Signal: defaultSignalPeriod}
}

// New creates a streaming SMA-seeded MACD engine for one price history.
// It normalizes and freezes the three EMA periods before any observations arrive.
//
// Implementation:
//   - Stage 1: replace zero-valued periods and validate ordering/index bounds.
//   - Stage 2: allocate exactly one scalar state object, without a history buffer.
//   - Stage 3: return an initialized Stream whose count and EMA seeds are zero.
//
// Behavior highlights:
//   - Fast/slow seed from their first Fast/Slow prices, respectively; signal
//     seeds from its first Signal valid MACD Line observations.
//   - Reset preserves these normalized periods and discards accepted observations.
//
// Inputs:
//   - opts: three periods, with individual zero fields selecting 12/26/9 defaults.
//
// Returns:
//   - *Stream: independently owned mutable state; nil upon an error.
//   - error: nil on success, otherwise ErrInvalidOption.
//
// Errors:
//   - ErrInvalidOption: invalid Fast/Slow ordering, negative period, or index overflow.
//
// Determinism:
//   - The stream consumes future Push calls in their invocation order.
//
// Complexity:
//   - Time O(1), Space O(1), one stream allocation when it escapes.
//
// Notes:
//   - A Stream is single-owner and not safe for concurrent mutation; create one per feed.
//
// AI-Hints:
//   - Do not share a Stream between instruments or timeframes even when periods match.
func New(opts Options) (*Stream, error) {
	normalized, err := normalizeOptions(opts)
	if err != nil {
		return nil, err
	}
	return &Stream{opts: normalized, initialized: true}, nil
}

// NewDefault constructs a stream using the conventional 12/26/9 periods.
// It forwards to New without adding a different initialization convention.
//
// Implementation:
//   - Stage 1: select Options{} and its documented defaults.
//   - Stage 2: delegate all validation and construction to New.
//   - Stage 3: return the same pointer/error pair.
//
// Returns:
//   - *Stream: an empty, caller-owned stream; nil on failure.
//   - error: the unmodified result of New.
//
// Determinism:
//   - Exactly the same seed positions and arithmetic order as New(Options{}).
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Not an alternate EMA algorithm or reduced warmup policy.
//
// AI-Hints:
//   - Use New when a signal horizon different from 9 is explicitly required.
func NewDefault() (*Stream, error) {
	return New(Options{})
}

// NewWithPeriods constructs a stream from three explicit observation horizons.
// It is a convenience for callers with period integers already stored separately.
//
// Implementation:
//   - Stage 1: assemble an Options value in Fast/Slow/Signal order.
//   - Stage 2: delegate all validation, normalization and state creation to New.
//   - Stage 3: return the unchanged result/error.
//
// Inputs:
//   - fast: fast price EMA length, with 0 selecting 12.
//   - slow: slow price EMA length, with 0 selecting 26; must exceed fast after defaults.
//   - signal: count of valid MACD Line samples in the signal EMA seed, 0 selecting 9.
//
// Returns:
//   - *Stream: independent live stream, or nil on invalid periods.
//   - error: ErrInvalidOption for the same conditions as New.
//
// Determinism:
//   - Identical to New(Options{Fast:fast, Slow:slow, Signal:signal}).
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Distinct instruments require distinct Stream values even with equal periods.
//
// AI-Hints:
//   - Do not pass an interval in seconds instead of the number of close observations.
func NewWithPeriods(fast, slow, signal int) (*Stream, error) {
	return New(Options{Fast: fast, Slow: slow, Signal: signal})
}

// Compute calculates a complete, length-preserving MACD Result for validated closes.
// Every observation uses Stream.Push so the batch output is bitwise identical to streaming.
//
// Implementation:
//   - Stage 1: validate periods, nil/empty slice, every finite price, then positivity.
//   - Stage 2: initialize a private Stream and allocate six independent output slices.
//   - Stage 3: Push each close in original order and publish three aligned Series.
//   - Stage 4: return the whole Result only if every sample succeeded.
//
// Behavior highlights:
//   - SMA seeds are calculated independently from the beginning of their sequences.
//   - Short nonempty input succeeds; not-yet-ready fields are +0 with Valid=false.
//   - Histogram validity always equals signal validity, even for a true zero histogram.
//
// Inputs:
//   - closes: non-nil, nonempty, read-only []float64 of finite strictly positive closes.
//   - opts: Fast/Slow/Signal periods with individual zero defaults of 12/26/9.
//
// Returns:
//   - *Result: three separate caller-owned Series, each of input length, with no aliases.
//   - error: nil on success; otherwise the Result pointer is always nil.
//
// Errors:
//   - ErrInvalidOption: periods or first-signal index invalid (checked first).
//   - ErrNilInput: a nil close slice; ErrEmptyInput: an allocated empty slice.
//   - ErrNonFinite: any NaN/Inf price (globally ahead of ErrDomain).
//   - ErrDomain: a nonpositive finite price when all input prices are finite.
//   - ErrOverflow: an unrepresentable count, EMA, MACD line or histogram during processing.
//
// Determinism:
//   - Scan for nonfinite values, then nonpositive values, both in index order;
//     Push each price in ascending original index order.
//   - All line, signal and histogram bytes match a fresh stream at each index.
//
// Complexity:
//   - Time O(n), Space O(n) for six slices of length n plus O(1) recurrence state.
//
// Notes:
//   - Concurrent mutation of close during Compute is unsupported; no input is retained.
//   - This is not a trade decision engine, probability estimate or feed-gap detector.
//
// AI-Hints:
//   - Do not shift or truncate the leading warmup observations: align masks to closes.
//   - Do not replace the single Push kernel with an independently vectorized EMA formula.
func Compute(closes []float64, opts Options) (*Result, error) {
	normalized, err := normalizeOptions(opts)
	if err != nil {
		return nil, err
	}
	if closes == nil {
		return nil, ErrNilInput
	}
	if len(closes) == 0 {
		return nil, ErrEmptyInput
	}

	// Nonfinite values take precedence over domain violations across the entire
	// batch, irrespective of their indices. Both scans finish before allocation.
	for _, price := range closes {
		if !finite(price) {
			return nil, ErrNonFinite
		}
	}
	for _, price := range closes {
		if price <= 0 {
			return nil, ErrDomain
		}
	}

	stream := Stream{opts: normalized, initialized: true}
	length := len(closes)
	result := &Result{
		Line:      Series{Values: make([]float64, length), Valid: make([]bool, length)},
		Signal:    Series{Values: make([]float64, length), Valid: make([]bool, length)},
		Histogram: Series{Values: make([]float64, length), Valid: make([]bool, length)},
	}

	// The one canonical step yields readiness and values together. A later error
	// discards all arrays rather than returning a misleading partially filled result.
	for index, price := range closes {
		point, pushErr := stream.Push(price)
		if pushErr != nil {
			return nil, pushErr
		}
		result.Line.Values[index], result.Line.Valid[index] = point.Line, point.LineValid
		result.Signal.Values[index], result.Signal.Valid[index] = point.Signal, point.SignalValid
		result.Histogram.Values[index], result.Histogram.Valid[index] = point.Histogram, point.SignalValid
	}
	return result, nil
}

// ComputeDefault calculates a full batch using the conventional 12/26/9 periods.
// It is precisely Compute(closes, Options{}) and exposes the same masks and errors.
//
// Implementation:
//   - Stage 1: select the default Options policy.
//   - Stage 2: delegate validation, recurrence and ownership to Compute.
//   - Stage 3: return its complete Result or nil on error.
//
// Inputs:
//   - closes: non-nil, nonempty finite positive prices; input is never retained.
//
// Returns:
//   - *Result: three detached aligned Series; nil on any error.
//   - error: the exact errors and precedence of Compute with default periods.
//
// Determinism:
//   - Bitwise identical to Compute(closes, Options{}).
//
// Complexity:
//   - Time O(n), Space O(n).
//
// Notes:
//   - A short history is not an error and may yield only invalid readiness masks.
//
// AI-Hints:
//   - Signal first becomes valid at index 33 for the default 12/26/9 configuration.
func ComputeDefault(closes []float64) (*Result, error) {
	return Compute(closes, Options{})
}

// ComputeWithPeriods computes MACD with three explicitly supplied horizons.
// It reduces caller boilerplate when an application already stores periods as integers.
//
// Implementation:
//   - Stage 1: construct Options using all three supplied periods.
//   - Stage 2: delegate validation and the sole recurrence to Compute.
//   - Stage 3: publish the unchanged Result/error pair.
//
// Inputs:
//   - closes: read-only non-nil finite-positive price history in index order.
//   - fast: fast EMA period, or 0 for the default 12.
//   - slow: slow EMA period, or 0 for the default 26.
//   - signal: signal EMA period, or 0 for the default 9.
//
// Returns:
//   - *Result: detached length-preserving Line/Signal/Histogram; nil on failure.
//   - error: exactly the categories, ordering and behavior of Compute.
//
// Determinism:
//   - Bitwise identical to Compute(closes, Options{fast,slow,signal}).
//
// Complexity:
//   - Time O(n), Space O(n).
//
// Notes:
//   - Both Fast and Slow must be seeded from the SAME first closes index.
//
// AI-Hints:
//   - Mixing periods from different bar cadences is a caller data-preparation error.
func ComputeWithPeriods(closes []float64, fast, slow, signal int) (*Result, error) {
	return Compute(closes, Options{Fast: fast, Slow: slow, Signal: signal})
}
