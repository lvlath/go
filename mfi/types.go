// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mfi defines public OHLCV carriers, detached batch results and checkpoint ABI.
// The structures deliberately contain no market timestamp or trading-policy metadata.
package mfi

// Options selects the number of typical-price transitions retained by MFI.
// The zero value selects DefaultPeriod; every nonzero period must be positive.
//
// Implementation:
//   - Stage 1: canonical facades normalize zero to DefaultPeriod.
//   - Stage 2: reject negative or oversized periods before ring memory allocation.
//   - Stage 3: freeze the effective period in the Stream constructor.
//
// Behavior highlights:
//   - Period counts every transition, including flat and zero-volume bars.
//   - MaxPeriod bounds the memory required by each Stream.
//
// Inputs:
//   - Period: integer window length, zero for default or [1, MaxPeriod].
//
// Returns:
//   - The struct itself carries configuration; an invalid value is rejected at New/Compute.
//
// Determinism:
//   - The normalized period completely determines the window and scheduled rebase cadence.
//
// Complexity:
//   - Storing and passing Options costs O(1) time and memory.
//
// Notes:
//   - Options is a value carrier; callers can reuse it without aliasing.
//
// AI-Hints:
//   - Do not reinterpret Period as a count of nonzero money-flow contributions.
type Options struct {
	// Period fixes the rolling transition window, with zero meaning DefaultPeriod.
	Period int
}

// Bar records one OHLC candle and the corresponding nonnegative traded volume.
// Prices must be finite positive values with Low <= Open,Close <= High.
//
// Implementation:
//   - Stage 1: Push validates finiteness and low/high price ordering.
//   - Stage 2: the flow kernel derives typical price from High, Low and Close.
//   - Stage 3: one signed or zero contribution advances the ring after success.
//
// Behavior highlights:
//   - Open participates in geometric validation, not typical-price computation.
//   - Volume is in one consistent base unit throughout a series.
//
// Inputs:
//   - Open, High, Low, Close: finite positive prices in identical quote units.
//   - Volume: finite, nonnegative base-unit quantity; zero is a valid observation.
//
// Returns:
//   - One Bar value is consumed by Push and is not retained or mutated.
//
// Determinism:
//   - Typical price uses High, Low and Close; comparisons use exact float64 ordering.
//
// Complexity:
//   - This value occupies O(1) memory and is copied by value by Push.
//
// Notes:
//   - The caller synchronizes timestamps and resets on discontinuities.
//
// AI-Hints:
//   - Never synthesize a missing candle from a zero-volume bar.
type Bar struct {
	// Open is the opening price used to validate OHLC geometry.
	Open float64
	// High is the maximum price and an operand of typical price.
	High float64
	// Low is the minimum price and an operand of typical price.
	Low float64
	// Close is the closing price and an operand of typical price.
	Close float64
	// Volume is nonnegative traded quantity in one consistent base unit.
	Volume float64
}

// Bars holds five parallel read-only OHLCV columns for batch MFI calculation.
// All columns must be non-nil and have equal nonzero length n.
//
// Implementation:
//   - Stage 1: Compute validates presence, lengths and effective period.
//   - Stage 2: it builds Bar values in ascending index order without changing columns.
//   - Stage 3: the canonical stream computes detached output at each index.
//
// Behavior highlights:
//   - Index i addresses one complete candle across all five columns.
//   - The package never modifies or retains caller-owned underlying arrays.
//
// Inputs:
//   - Open, High, Low, Close, Volume: aligned, chronological []float64 columns.
//
// Returns:
//   - Five caller-owned aligned columns are accepted as read-only input.
//
// Determinism:
//   - Compute consumes bars in their given ascending index order.
//
// Complexity:
//   - The struct is O(1); Compute uses O(n+Period) memory for its own state/result.
//
// Notes:
//   - Do not mutate columns concurrently with Compute.
//
// AI-Hints:
//   - Nil columns and non-nil empty columns have different sentinel errors.
type Bars struct {
	// Open contains opening prices of all candles.
	Open []float64
	// High contains high prices of all candles.
	High []float64
	// Low contains low prices of all candles.
	Low []float64
	// Close contains close prices of all candles.
	Close []float64
	// Volume contains nonnegative base-unit volumes of all candles.
	Volume []float64
}

// Result carries a detached MFI series and its independent warmup validity mask.
// On successful Compute, both slices have length equal to the input bar count.
//
// Implementation:
//   - Stage 1: Allocate equally sized Values and Valid slices.
//   - Stage 2: Publish each successful Stream.Push output in input order.
//   - Stage 3: Return both independently owned slices only after batch success.
//
// Behavior highlights:
//   - Values[i]==0 and Valid[i]==false before the first complete window.
//   - For every Valid[i]==true, Values[i] is a finite value in [0,100].
//
// Inputs:
//   - This result is produced by Compute and additive batch facades, not a mutable Stream view.
//
// Returns:
//   - Values: caller-owned MFI values, including zero warmup placeholders.
//   - Valid: caller-owned flags; false for indices [0, Period).
//
// Errors:
//   - Compute returns nil *Result on every error; no partial Values/Valid arrays escape.
//
// Determinism:
//   - Positional correspondence with the original input columns is exact.
//
// Complexity:
//   - Stored size O(n); IsNil costs O(1).
//
// Notes:
//   - Values and Valid do not alias input memory or a mutable Stream.
//   - A nil *Result is returned on Compute error; no partial Result is exposed.
//
// AI-Hints:
//   - Never infer validity from Values[i]==0: valid bearish windows also produce zero.
type Result struct {
	// Values stores computed MFI values; entries before warmup are exactly zero.
	Values []float64
	// Valid indicates which Values entries have a complete transition window.
	Valid []bool
}

// IsNil reports whether the Result pointer itself is nil, including in interfaces.
// It fulfills the library's nilable result convention without inspecting slices.
//
// Implementation:
//   - Stage 1: compare the receiver with nil.
//   - Stage 2: return that comparison directly, including for typed-nil interfaces.
//   - Stage 3: do not allocate, dereference the receiver or mutate its state.
//
// Behavior highlights:
//   - Slice allocation, validity flags and zero-valued fields do not influence nilness.
//
// Inputs:
//   - r: possibly typed-nil *Result receiver; no dereference or reflection is performed.
//
// Returns:
//   - bool: true only for a nil *Result; an allocated zero-value Result is not nil.
//
// Determinism:
//   - Depends exclusively on receiver nilness, not slice contents.
//
// Complexity:
//   - Time O(1), Space O(1), zero allocation.
//
// Notes:
//   - Calling this method through an interface holding (*Result)(nil) is safe.
//
// AI-Hints:
//   - Do not replace this check with nil-slice tests or reflection-based nil detection.
func (r *Result) IsNil() bool {
	return r == nil
}

// Snapshot is a detached, versioned carrier for the complete Stream recurrence.
// It is a compatibility format, not a cryptographically authenticated checkpoint.
//
// Implementation:
//   - Stage 1: Snapshot exports algorithm/configuration identity and Count.
//   - Stage 2: It copies compensators, previous typical price and ring slots.
//   - Stage 3: Restore validates the carrier before copying into engine-owned memory.
//
// Behavior highlights:
//   - FloatState has layout [previous,sumPos,sumNeg,corrPos,corrNeg,posRing,negRing].
//   - IntState has layout [used,head,extraRebaseUsed]; two rings each have Period slots.
//   - Count is the number of successfully processed bars, including warmup.
//
// Inputs:
//   - Created by Stream.Snapshot and supplied read-only to Stream.Restore.
//
// Returns:
//   - A detached value whose slice fields belong to the caller after Snapshot.
//
// Errors:
//   - Restore rejects incompatible or corrupted state with ErrStateVersion.
//
// Determinism:
//   - Ring slots remain in physical order; head identifies the next overwritten slot.
//
// Complexity:
//   - Snapshot and Restore each take O(Period) time and O(Period) copied data.
//
// Notes:
//   - Config is SHA-256 over a public algorithm ID and period, NOT a state MAC.
//   - Authenticate the entire serialized snapshot externally before trusting it.
//
// AI-Hints:
//   - Preserve the exact float64 bits and array order during serialization; do not rebase on import.
type Snapshot struct {
	// Algorithm identifies the exact checkpoint recurrence and binary layout.
	Algorithm string
	// Config is a lowercase hex compatibility hash of Algorithm and the LE64 period.
	Config string
	// Count records accepted bars since construction or last Reset.
	Count uint64
	// FloatState contains fixed prefix fields and both detached ring arrays.
	FloatState []float64
	// IntState contains used, head and the unscheduled-rebase epoch flag.
	IntState []uint64
}
