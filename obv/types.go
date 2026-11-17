// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package obv defines the stable public input, result and checkpoint carriers.
// This file distinguishes immutable observations from a single-owner mutable stream.
package obv

// Bar represents one chronological OHLCV observation with base-asset volume.
// All prices must be finite and positive; volume must be finite and nonnegative.
//
// Behavior highlights:
//   - Low <= Open,Close <= High, and High >= Low are mandatory.
//   - The first accepted bar's Volume is validated but never accumulated.
//
// Inputs:
//   - Open, High, Low, Close: one price scale, in consistent quotation units.
//   - Volume: traded base units, not an implicitly converted quote notional.
//
// Determinism:
//   - Exact binary64 Close ordering is used; no epsilon or candle normalization.
//
// Complexity:
//   - O(1) storage and copying.
//
// AI-Hints:
//   - Do not infer base volume from price times quote volume inside this package.
type Bar struct {
	// Open is the first reported trade price in the candle's quote units.
	// It must be inside [Low, High], even though OBV does not add it to the sum.
	Open float64

	// High is the greatest reported price during the candle.
	// It must be finite, positive and no lower than Open, Close or Low.
	High float64

	// Low is the least reported price during the candle.
	// It must be finite, positive and no greater than Open, Close or High.
	Low float64

	// Close determines the sign of this candle's volume contribution.
	// Only the immediately preceding accepted Close is used for comparison.
	Close float64

	// Volume is the finite, nonnegative amount traded in the caller's base units.
	// Neither first-bar Volume nor Volume on an unchanged Close changes OBV.
	Volume float64
}

// Bars groups five aligned chronological observation columns for Compute.
// Input slices are borrowed read-only and are never retained by the returned Result.
//
// Behavior highlights:
//   - Every column must be non-nil, nonempty and have identical length.
//   - Shape validation is completed before indexing any column.
//
// Inputs:
//   - Open, High, Low, Close, Volume: aligned caller-owned binary64 slices.
//
// Determinism:
//   - Observation index is the sole processing order; no timestamps or sorting.
//
// Complexity:
//   - O(1) slice header storage; caller owns O(n) element storage.
//
// AI-Hints:
//   - Never mutate or reorder a column while Compute is active.
type Bars struct {
	// Open contains one opening price per observation, in chronological order.
	Open []float64

	// High contains the high price of each corresponding observation.
	High []float64

	// Low contains the low price of each corresponding observation.
	Low []float64

	// Close contains the only prices used for OBV transition comparisons.
	Close []float64

	// Volume contains the amounts added or subtracted after the seed candle.
	Volume []float64
}

// Result contains the complete detached OBV series and its validity mask.
// A successful computation always has len(Values)==len(Valid)>0 and every mask bit true.
//
// Behavior highlights:
//   - Values[0] is positive zero regardless of the first bar's volume.
//   - All values are finite; no partial Result is returned after any error.
//
// Returns:
//   - Values: caller-owned cumulative base-volume units in chronological order.
//   - Valid: caller-owned mask marking every accepted observation true.
//
// Determinism:
//   - Values are published in original observation order.
//
// Complexity:
//   - O(n) owned storage after successful Compute/ComputeCandles.
//
// Notes:
//   - Mutating the returned slices is safe and does not affect future calls.
//
// AI-Hints:
//   - Do not treat a zero Value as invalid; consult the Valid flag.
type Result struct {
	// Values contains finite compensated OBV totals in input-index order.
	// The first element is always positive zero for a successful nonempty batch.
	Values []float64

	// Valid has the same positive length as Values, and every entry is true.
	// It does not indicate statistical significance or a trading signal.
	Valid []bool
}

// IsNil classifies the receiver pointer without dereferencing it.
// It implements the nilable result convention shared by lvlath algorithm packages.
//
// Implementation:
//   - Stage 1: compare the receiver pointer to nil.
//   - Stage 2: report whether the object itself is absent.
//
// Returns:
//   - bool: true exactly for a nil *Result, not for an allocated empty Result.
//
// Errors:
//   - None; a nil receiver cannot panic.
//
// Determinism:
//   - Exact pointer-nil comparison, independent of slice contents.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// AI-Hints:
//   - An allocated Result{} is not evidence that Compute succeeded.
func (r *Result) IsNil() bool { return r == nil }

// Snapshot is a versioned and detached representation of one Stream's recurrence.
// It carries both compensated operands, not just their rounded displayed sum.
//
// Behavior highlights:
//   - Algorithm is "obv/neumaier/v1" and Config is SHA-256 of that ASCII text.
//   - FloatState is [previousClose, sum, compensation] with exactly three elements.
//   - IntState has length zero; Count includes the first seed-only candle.
//
// Returns:
//   - Detached caller-owned slices; mutations never alter the originating Stream.
//
// Determinism:
//   - Fixed field order and binary64 bit representation are retained on round-trip.
//
// Complexity:
//   - O(1) fixed-length checkpoint storage.
//
// Notes:
//   - Config is not an authentication tag; protect persisted state separately.
//
// AI-Hints:
//   - Never replace FloatState with [previousClose, sum+compensation] on failover.
type Snapshot struct {
	// Algorithm identifies the exact Neumaier recurrence and snapshot layout.
	// The v1 identifier is "obv/neumaier/v1".
	Algorithm string

	// Config is the lowercase SHA-256 hex digest of Algorithm's fixed ASCII ID.
	// It establishes compatibility only, not payload authenticity or integrity.
	Config string

	// Count is the number of successfully accepted candles, including the seed.
	// A restored count of MaxUint64 rejects the next Push without mutation.
	Count uint64

	// FloatState contains [previousClose, sum, compensation], in that order.
	// All three values must be finite; the effective sum must also be finite.
	FloatState []float64

	// IntState has exactly zero elements in v1, with no hidden ring cursor.
	// The slice is detached from Stream, like FloatState.
	IntState []uint64
}

// Stream owns the mutable compensated recurrence and last accepted close.
// Its zero value is ready for Push, and it must be owned by a single goroutine.
//
// Behavior highlights:
//   - Previous close is meaningful only after Count becomes positive.
//   - Sum and compensation must both be preserved for bit-identical continuation.
//
// Determinism:
//   - Only accepted Push calls change these four fields, in input order.
//
// Complexity:
//   - O(1) fixed-size state, with no ring buffer or heap-backed recurrence.
//
// Notes:
//   - Concurrent calls to Push/Reset/Restore/Snapshot on one stream are unsupported.
//
// AI-Hints:
//   - Use Snapshot/Restore to transfer state between independent execution workers.
type Stream struct {
	// previousClose is the last accepted Close; it is meaningful when count > 0.
	previousClose float64

	// sum is the rounded high component of the signed-volume accumulation.
	sum float64

	// compensation preserves lower-order Neumaier residuals across transitions.
	compensation float64

	// count includes every accepted candle, not just price-changing transitions.
	count uint64
}
