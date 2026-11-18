// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ad defines the observation, result, checkpoint, and stream carriers.
// This file owns their public invariants and the single-owner state layout.
package ad

// Bar is one OHLCV candle whose close location determines signed money flow.
// All prices must be finite and strictly positive; base volume must be finite and nonnegative.
//
// Implementation:
//   - Stage 1: a caller constructs an unmodified OHLCV observation.
//   - Stage 2: Push or a batch facade validates every field, including Open.
//   - Stage 3: the kernel uses the validated close location and Volume only.
//
// Behavior highlights:
//   - Low <= Open,Close <= High; High == Low is valid only for a flat candle.
//   - Unlike OBV, A/D uses this bar's volume even for the first observation.
//
// Inputs:
//   - Open, High, Low, Close: prices in the same positive quote-price scale.
//   - Volume: traded base units for this candle, not quote notional.
//
// Errors:
//   - ErrNonFinite or ErrDomain is returned by the consuming facade for an invalid Bar.
//
// Determinism:
//   - Candle fields are inspected without sorting, interpolation, or epsilon comparisons.
//
// Complexity:
//   - Time O(1), Space O(1) to copy the value; no owned slice memory.
//
// Notes:
//   - A timestamp and interval policy are deliberately not part of Bar.
//
// AI-Hints:
//   - Do not supply quote turnover as Volume unless that is the intended unit.
//   - Do not omit Open validation merely because the multiplier ignores Open.
type Bar struct {
	// Open is the first observed quote price in the candle's interval.
	// It is checked against [Low, High] even though it does not enter A/D math.
	Open float64

	// High is the maximal quoted price in the candle, including Open and Close.
	// Positive, finite values with High >= Low are required.
	High float64

	// Low is the minimal quoted price in the candle, including Open and Close.
	// It must be positive, finite and no greater than High.
	Low float64

	// Close determines the money-flow multiplier through its relative range position.
	// It must be inside [Low, High]; invalid closes are rejected, never clipped.
	Close float64

	// Volume is the nonnegative finite base-asset volume in this interval.
	// A zero volume is valid; the observation still advances Stream.Count.
	Volume float64
}

// Bars represents a chronological collection of five column-aligned OHLCV series.
// Every slice must be non-nil and have the same strictly positive length.
//
// Implementation:
//   - Stage 1: Compute checks all slice headers before reading a single element.
//   - Stage 2: all rows are validated without changing or copying input data.
//   - Stage 3: cumulative values are published in the original column index order.
//
// Behavior highlights:
//   - An empty but non-nil Open column is ErrEmptyInput; differing lengths are ErrShape.
//
// Inputs:
//   - Open, High, Low, Close, Volume: borrowed, read-only, equal-length binary64 slices.
//
// Determinism:
//   - Index 0 is always the first processed observation; no timestamps are inferred.
//
// Complexity:
//   - Time O(1), Space O(1) for this header; Compute is O(n) time and space.
//
// Notes:
//   - Input must not be modified concurrently with Compute; no input is retained.
//
// AI-Hints:
//   - For existing row-oriented feeds use ComputeCandles to avoid column transposition.
type Bars struct {
	// Open contains the opening prices in the caller's chronological order.
	Open []float64

	// High contains each candle's highest price at the corresponding index.
	High []float64

	// Low contains each candle's lowest price at the corresponding index.
	Low []float64

	// Close contains each candle's final price at the corresponding index.
	Close []float64

	// Volume contains the traded base volume at the corresponding index.
	Volume []float64
}

// Result is the canonical detached Chaikin A/D output artifact.
// Its two parallel fields have equal positive length and describe the same indices.
//
// Implementation:
//   - Stage 1: a batch facade validates every supplied candle.
//   - Stage 2: the canonical recurrence accumulates each candle from initial zero.
//   - Stage 3: the facade returns owned Values and Valid slices only on full success.
//
// Behavior highlights:
//   - Values[0] includes the first candle's contribution and may be nonzero.
//   - Successful computation marks every Valid[i] true; there is no warmup period.
//
// Returns:
//   - Values: finite, caller-owned cumulative base-volume quantities.
//   - Valid: caller-owned mask with the same length, all true on success.
//
// Errors:
//   - On any Compute failure, the Result pointer is nil, not a partial artifact.
//
// Determinism:
//   - Values[i] is published in exact input order after the i-th accepted bar.
//
// Complexity:
//   - Time O(1) to access fields; Space O(n) owned by each successful Result.
//
// Notes:
//   - The slices may be modified by their owner without changing a Stream or input.
//   - A/D is an indicator, not an assertion about counterparties or trade intent.
//
// AI-Hints:
//   - A valid value of zero is not a missing observation; consult Valid explicitly.
//   - Do not assume the entire cumulative output is bounded by any single Volume.
type Result struct {
	// Values contains each finite compensated cumulative balance, including bar zero.
	// Its length is the exact input length after a successful batch call.
	Values []float64

	// Valid identifies successfully computed observations and is true for every index.
	// Its presence provides consistent result semantics across indicator packages.
	Valid []bool
}

// IsNil reports whether the Result receiver itself is nil, including a typed-nil interface.
// It implements core.Nilable without reading any result fields or allocating memory.
//
// Implementation:
//   - Stage 1: compare the receiver directly against nil.
//   - Stage 2: return the comparison, without accessing Values or Valid.
//
// Behavior highlights:
//   - A non-nil Result{} is not nil, even if it has no computed observations.
//
// Inputs:
//   - r: possibly nil result receiver.
//
// Returns:
//   - bool: true exactly when r == nil.
//
// Errors:
//   - None; a typed-nil receiver is safe.
//
// Determinism:
//   - Pointer identity only; no reflection or hidden mutable state.
//
// Complexity:
//   - Time O(1), Space O(1), zero allocations.
//
// Notes:
//   - IsNil does not certify that an allocated Result was produced by Compute.
//
// AI-Hints:
//   - Do not use reflect-based typed-nil detection around Result.
func (r *Result) IsNil() bool { return r == nil }

// Snapshot is the portable, detached state carrier for checkpoint replication.
// It preserves both binary64 Neumaier operands, which are required for exact continuation.
//
// Implementation:
//   - Stage 1: Snapshot records an algorithm identity and fixed compatibility digest.
//   - Stage 2: it copies Count, sum, and compensation into the versioned field layout.
//   - Stage 3: Restore checks the full schema and numeric domain before installation.
//
// Behavior highlights:
//   - Algorithm is "ad/chaikin-neumaier/v1"; Config is SHA-256 of that ASCII string.
//   - FloatState is exactly [sum, compensation]; IntState has length zero.
//   - Count is the number of successful pushes, not the last input index.
//
// Returns:
//   - Value fields and separately owned FloatState; no mutable Stream memory is aliased.
//
// Errors:
//   - Invalid schema, incompatible Config or unrepresentable fields yield ErrStateVersion on Restore.
//
// Determinism:
//   - Fixed field order; continuation preserves each accepted float64 coordinate.
//
// Complexity:
//   - Time O(1), Space O(1) for two floating coordinates and metadata.
//
// Notes:
//   - Config is a compatibility fingerprint, not a signature, checksum of payload, or MAC.
//   - Authenticate persisted/remote snapshots outside this package if needed.
//
// AI-Hints:
//   - Never replace the pair with only sum+compensation: future cancellation can differ.
//   - Unlike OBV, AD does not persist previous Close.
type Snapshot struct {
	// Algorithm names the recurrence and its snapshot ABI revision exactly.
	Algorithm string

	// Config is the lowercase SHA-256 hexadecimal digest of Algorithm's fixed ID.
	// It identifies a compatible layout, not the integrity of arbitrary serialized data.
	Config string

	// Count is the total number of previously accepted candles, including the first.
	// Restoring MaxUint64 is allowed, but another Push returns ErrOverflow.
	Count uint64

	// FloatState stores [sum, compensation] in this precise order and length two.
	// Both must be finite and their effective total must be representable.
	FloatState []float64

	// IntState is a zero-length slice in snapshot ABI v1; no cursor is stored.
	// Restore rejects every nonempty IntState, including one fabricated by a caller.
	IntState []uint64
}

// Stream maintains the signed base-volume balance with Neumaier compensation.
// Its zero value and New() have identical state and accept the first bar immediately.
//
// Implementation:
//   - Stage 1: Push validates a new OHLCV candle without mutating the stream.
//   - Stage 2: a scalar candidate applies the compensated canonical recurrence.
//   - Stage 3: success replaces all scalar fields atomically from the owner's perspective.
//
// Behavior highlights:
//   - No previous price is needed; every accepted candle contributes independently.
//   - Count tracks accepted candles, including flat and zero-volume observations.
//
// Determinism:
//   - Exactly one accepted transition per Push, in caller-supplied order.
//
// Complexity:
//   - Time O(1), Space O(1) per Push; no heap-backed history or ring buffer.
//
// Notes:
//   - One Stream is owned by one goroutine; concurrent access during mutation is unsupported.
//   - The package takes no locks and provides no cross-call linearizable snapshot.
//
// AI-Hints:
//   - Reset explicitly when a timeline gap should create a new accumulation origin.
//   - Move state between owners using Snapshot and Restore, not by sharing a Stream.
type Stream struct {
	// sum is the rounded high component of the signed cumulative volume.
	sum float64

	// compensation retains low-order rounding residuals across accepted bars.
	compensation float64

	// count includes all accepted OHLCV bars, even when their contribution is zero.
	count uint64
}
