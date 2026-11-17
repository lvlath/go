// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mfi exposes canonical batch/stream entry points and transparent conveniences.
// Every wrapper delegates to Compute or New; no alternative numerical recurrence exists.
package mfi

// New creates a single-owner stream with a fixed transition window.
// Ring storage is allocated only after period validation, so an invalid period
// cannot trigger an oversized allocation or change another stream's state.
//
// Implementation:
//   - Stage 1: normalize Options.Period and enforce MaxPeriod.
//   - Stage 2: allocate one positive and one negative float64 ring.
//   - Stage 3: return the zero-observation engine with no retained caller memory.
//
// Behavior highlights:
//   - A zero period selects DefaultPeriod; a nonzero period is exact.
//   - The returned engine must be owned by a single goroutine.
//
// Inputs:
//   - opts: value-copy window policy, with Period in [0, MaxPeriod].
//
// Returns:
//   - *Stream: independent engine; nil whenever error is non-nil.
//
// Errors:
//   - ErrInvalidOption: a negative or too-large period, checked before allocation.
//
// Determinism:
//   - The period and initial zero-state depend only on opts.
//
// Complexity:
//   - Time O(Period), Space O(Period), including zeroed ring allocation.
//
// Notes:
//   - Runtime memory exhaustion is not converted to an error; MaxPeriod bounds request size.
//
// AI-Hints:
//   - Share snapshots, not the mutable stream itself, between independent workers.
func New(opts Options) (*Stream, error) {
	period, err := normalizedPeriod(opts)
	if err != nil {
		return nil, err
	}
	return &Stream{
		period: period,
		pos:    make([]float64, period),
		neg:    make([]float64, period),
	}, nil
}

// NewDefault creates a Stream with the standard fourteen-transition window.
// It is an ergonomic alias for New(Options{}) without an alternative policy.
//
// Implementation:
//   - Stage 1: select the zero Options value.
//   - Stage 2: delegate validation and ring allocation to New.
//   - Stage 3: return New's stream or error unchanged.
//
// Behavior highlights:
//   - Every invocation allocates a new independent ring; no shared global stream state.
//
// Returns:
//   - *Stream: a new independent stream, or nil on allocation-policy error.
//
// Errors:
//   - Identical to New(Options{}).
//
// Determinism:
//   - Identical initial state to New(Options{Period: DefaultPeriod}).
//
// Complexity:
//   - Time O(DefaultPeriod), Space O(DefaultPeriod).
//
// Notes:
//   - The owner must still reset explicitly on a discontinuity in bar timestamps.
//
// AI-Hints:
//   - NewDefault does not infer a period from the amount of historical data.
func NewDefault() (*Stream, error) {
	return New(Options{})
}

// NewPeriod creates a Stream for exactly the requested window length.
// It keeps explicit-period callers out of Options literals without weakening checks.
//
// Implementation:
//   - Stage 1: place period in the canonical Options carrier.
//   - Stage 2: delegate allocation and validation to New.
//   - Stage 3: return New's result unchanged.
//
// Behavior highlights:
//   - A zero argument selects the default but never means an unlimited window.
//
// Inputs:
//   - period: number of transitions, 0 for DefaultPeriod, or [1, MaxPeriod].
//
// Returns:
//   - *Stream: ready engine, or nil when the policy is invalid.
//
// Errors:
//   - ErrInvalidOption: period is negative or larger than MaxPeriod.
//
// Determinism:
//   - Identical to New(Options{Period: period}).
//
// Complexity:
//   - Time O(effective Period), Space O(effective Period).
//
// Notes:
//   - The period is fixed throughout the stream's lifetime, including Restore.
//
// AI-Hints:
//   - Use Restore only between engines with the same effective period.
func NewPeriod(period int) (*Stream, error) {
	return New(Options{Period: period})
}

// Compute computes a fully detached Money Flow Index series over five aligned columns.
// It uses Stream.Push for every bar in input order, including zero-volume bars,
// guaranteeing the same numerical recurrence and validity mask as streaming.
//
// Implementation:
//   - Stage 1: validate options, then nil/empty/shape before element access.
//   - Stage 2: create one stream, then push the corresponding OHLCV bar per index.
//   - Stage 3: publish detached values/mask only after all observations succeed.
//
// Behavior highlights:
//   - The first bar seeds previousTypical without contributing money flow.
//   - The first valid index is Period, after Period+1 valid observations.
//   - Earlier Values are 0 and their Valid entries are false, never synthetic MFI.
//
// Inputs:
//   - bars: caller-owned, read-only, equal-length non-nil OHLCV slices of length n>0.
//   - opts: window period, zero for DefaultPeriod and at most MaxPeriod.
//
// Returns:
//   - *Result: n detached Values and n detached Valid flags, or nil on any error.
//
// Errors:
//   - ErrInvalidOption: invalid period, before reading any input.
//   - ErrNilInput, ErrEmptyInput, ErrShape: invalid column presence or lengths.
//   - ErrNonFinite, ErrDomain: malformed individual OHLCV bars.
//   - ErrOverflow: nonrepresentable money-flow arithmetic or count.
//
// Determinism:
//   - Scans indices ascending; every transition uses the same Push order and rebase policy.
//
// Complexity:
//   - Amortized Time O(n+Period), Space O(n+Period), including owned results/rings.
//
// Notes:
//   - No timestamps are inspected. Inputs must not be concurrently mutated.
//   - A rejected input returns no partial result or exposure of intermediate buffers.
//
// AI-Hints:
//   - Never skip bars with zero volume or unchanged typical price when preparing columns.
//   - Use Valid rather than comparing MFI with zero to identify warmed-up results.
func Compute(bars Bars, opts Options) (*Result, error) {
	period, err := normalizedPeriod(opts)
	if err != nil {
		return nil, err
	}
	count, err := validateShape(bars)
	if err != nil {
		return nil, err
	}

	// The canonical Stream performs all numeric validation and money-flow work.
	stream, err := New(Options{Period: period})
	if err != nil {
		return nil, err
	}
	result := &Result{Values: make([]float64, count), Valid: make([]bool, count)}
	for i := 0; i < count; i++ {
		bar := Bar{
			Open: bars.Open[i], High: bars.High[i], Low: bars.Low[i],
			Close: bars.Close[i], Volume: bars.Volume[i],
		}
		value, valid, pushErr := stream.Push(bar)
		if pushErr != nil {
			return nil, pushErr
		}
		result.Values[i], result.Valid[i] = value, valid
	}
	return result, nil
}

// ComputeDefault calculates MFI using the standard fourteen-transition window.
// It preserves the complete Result, including the warmup validity mask.
//
// Implementation:
//   - Stage 1: select the zero Options value.
//   - Stage 2: delegate the full sequence to Compute.
//   - Stage 3: return the detached canonical Result unchanged.
//
// Behavior highlights:
//   - The default uses 14 transitions; it never shortens the period based on history length.
//
// Inputs:
//   - bars: read-only, aligned non-nil OHLCV columns.
//
// Returns:
//   - *Result: detached MFI series and mask, or nil on any failure.
//
// Errors:
//   - Identical to Compute(bars, Options{}).
//
// Determinism:
//   - Identical to Compute with Period=DefaultPeriod.
//
// Complexity:
//   - Amortized Time O(n+DefaultPeriod), Space O(n+DefaultPeriod).
//
// Notes:
//   - The wrapper never infers an alternative shorter warmup for short input.
//
// AI-Hints:
//   - A non-nil Result with an all-false Valid mask means insufficient warmup, not error.
func ComputeDefault(bars Bars) (*Result, error) {
	return Compute(bars, Options{})
}

// ComputePeriod calculates MFI using an explicit number of price transitions.
// It is a direct wrapper around Compute and does not alter numeric precision.
//
// Implementation:
//   - Stage 1: encode period in Options.
//   - Stage 2: delegate validation and computation to Compute.
//   - Stage 3: return the canonical result or error unchanged.
//
// Behavior highlights:
//   - The window contains the last P transitions, including zero-volume and flat-price bars.
//
// Inputs:
//   - bars: read-only, aligned non-nil OHLCV columns.
//   - period: 0 for DefaultPeriod, or integer in [1, MaxPeriod].
//
// Returns:
//   - *Result: detached series and validity mask, or nil on error.
//
// Errors:
//   - Identical to Compute with Options{Period: period}.
//
// Determinism:
//   - Identical to the corresponding Compute invocation.
//
// Complexity:
//   - Amortized Time O(n+Period), Space O(n+Period).
//
// Notes:
//   - Period counts transitions rather than nonzero-volume contributions.
//
// AI-Hints:
//   - A P-transition MFI first becomes valid after P+1 observations.
func ComputePeriod(bars Bars, period int) (*Result, error) {
	return Compute(bars, Options{Period: period})
}

// ComputeCandles adapts a chronological slice of Bar structs into the canonical columns.
// It is convenient when market-data decoders already yield row-oriented OHLCV bars.
//
// Implementation:
//   - Stage 1: validate options and distinguish nil from empty row slices.
//   - Stage 2: transpose each row to five owned columns without reordering observations.
//   - Stage 3: delegate all candle validation and numerical work to Compute.
//
// Behavior highlights:
//   - No bar is filtered or repaired, including invalid, flat, or zero-volume bars.
//   - This adapter trades O(n) temporary memory for simpler row-oriented input.
//
// Inputs:
//   - candles: read-only non-nil chronological []Bar, containing at least one bar.
//   - opts: same normalized period policy as Compute.
//
// Returns:
//   - *Result: detached values and validity mask; nil on error.
//
// Errors:
//   - ErrInvalidOption, ErrNilInput, ErrEmptyInput: pre-transposition checks.
//   - ErrNonFinite, ErrDomain, ErrOverflow: forwarded unmodified from Compute.
//
// Determinism:
//   - Preserves the exact []Bar index order; classification is delegated to Compute.
//
// Complexity:
//   - Amortized Time O(n+Period), Space O(n+Period) beyond caller-owned candles.
//
// Notes:
//   - For very large histories, pass five existing columns to Compute to avoid copying.
//   - Concurrent mutation of candles during this call is unsupported.
//
// AI-Hints:
//   - Never sort by timestamp here: timestamp quality and gap handling are caller duties.
func ComputeCandles(candles []Bar, opts Options) (*Result, error) {
	if _, err := normalizedPeriod(opts); err != nil {
		return nil, err
	}
	if candles == nil {
		return nil, ErrNilInput
	}
	if len(candles) == 0 {
		return nil, ErrEmptyInput
	}

	columns := Bars{
		Open: make([]float64, len(candles)), High: make([]float64, len(candles)),
		Low: make([]float64, len(candles)), Close: make([]float64, len(candles)),
		Volume: make([]float64, len(candles)),
	}
	for i, bar := range candles {
		columns.Open[i], columns.High[i], columns.Low[i] = bar.Open, bar.High, bar.Low
		columns.Close[i], columns.Volume[i] = bar.Close, bar.Volume
	}
	return Compute(columns, opts)
}

// Latest returns the final MFI observation and whether its window is warmed up.
// It projects the last element of the canonical batch Result rather than running
// a second, possibly numerically different last-window algorithm.
//
// Implementation:
//   - Stage 1: call Compute with its complete validation and recurrence.
//   - Stage 2: select the last detached result entry and its Valid flag.
//   - Stage 3: return that pair, or zero/false with the original error.
//
// Behavior highlights:
//   - Invalid warmup returns value=0 and valid=false with nil error; a failure also returns zeros but with an error.
//
// Inputs:
//   - bars: read-only aligned nonempty OHLCV series.
//   - opts: MFI period, using the same default and limit as Compute.
//
// Returns:
//   - value: final MFI, or 0 if invalid/warming up or on error.
//   - valid: false until Period+1 observations are available, or on error.
//   - error: nil on successful computation, even when the final value is warming up.
//
// Errors:
//   - Identical to Compute; failure never returns a usable scalar result.
//
// Determinism:
//   - Bitwise identical to the final Values entry produced by Compute.
//
// Complexity:
//   - Amortized Time O(n+Period), Space O(n+Period), including the batch Result.
//
// Notes:
//   - For an allocation-free live latest reading, retain a Stream and call Push.
//
// AI-Hints:
//   - Do not confuse invalid zero with a valid bearish MFI of exactly zero.
func Latest(bars Bars, opts Options) (value float64, valid bool, err error) {
	result, err := Compute(bars, opts)
	if err != nil {
		return 0, false, err
	}
	last := len(result.Values) - 1
	return result.Values[last], result.Valid[last], nil
}
