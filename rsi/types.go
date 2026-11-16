// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package rsi defines the public Wilder RSI configuration and result artifacts.
// This file specifies shape, ownership, nilability, and portable stream-checkpoint fields.
package rsi

// Options selects the number of closing-price changes used by Wilder's seed and smoothing.
// Its only numeric option is intentionally explicit; it does not configure trading thresholds.
//
// Implementation:
//   - Stage 1: New or Compute copies this value and normalizes its Period.
//   - Stage 2: the canonical kernel uses the resulting period for seed and recurrence.
//
// Behavior highlights:
//   - Period=0 selects 14; a positive Period=P requires P+1 closes for the first valid RSI.
//   - A negative period or an overflowing P+1 is rejected, never silently shortened.
//
// Inputs:
//   - Period: count of changes (not price samples), in integral time steps.
//
// Returns:
//   - Used as a value-only policy; no pointers or aliased memory are retained.
//
// Errors:
//   - ErrInvalidOption is returned by New/Compute when Period cannot be normalized.
//
// Determinism:
//   - Equal normalized periods produce equal configuration fingerprints and warmup phases.
//
// Complexity:
//   - Space O(1); normalization costs O(1) time.
//
// Notes:
//   - The zero value is a valid default configuration, not an invalid period.
//
// AI-Hints:
//   - Do not use the number of bars as the seed-divisor: P changes require P+1 closes.
//   - Choose independent Options for each cross-sectional factor period.
type Options struct {
	// Period is the number of price changes in the initial arithmetic means.
	// Zero chooses 14; positive values use their exact period and negative values are rejected.
	Period int
}

// Result owns a completed, index-aligned Wilder RSI history and its validity mask.
// It separates warmup absence from the legitimate numeric RSI value zero.
//
// Implementation:
//   - Stage 1: Compute fully validates the input and allocates both output slices.
//   - Stage 2: the shared kernel fills each index in caller-supplied price order.
//   - Stage 3: Compute publishes both slices only after every price succeeds.
//
// Behavior highlights:
//   - len(Values)=len(Valid)=len(input), including histories shorter than warmup.
//   - Before zero-based index P, Values are +0 and Valid entries are false.
//   - Every entry with Valid=true has a finite RSI in [0,100].
//
// Inputs:
//   - Produced only by successful Compute or its batch convenience wrappers.
//
// Returns:
//   - Both slices are detached caller-owned memory; the package retains no aliases.
//
// Errors:
//   - Compute returns nil Result on every error, never a usable partial artifact.
//
// Determinism:
//   - Output index i corresponds exactly to input close[i].
//
// Complexity:
//   - Space O(n) for n closing prices; IsNil is O(1).
//
// Notes:
//   - A caller can mutate Values/Valid without changing any Stream state or past input.
//   - Nil *Result is a valid optional interface value recognized by IsNil.
//
// AI-Hints:
//   - Check Valid[i] before interpreting Values[i], including when Values[i]==0.
//   - Do not interpret RSI as an automatic trade recommendation.
type Result struct {
	// Values contains the computed RSI at each input index; warmup positions are +0.
	// These are not price values, percentages of change, or buy/sell decisions.
	Values []float64

	// Valid distinguishes evaluated observations from the warmup prefix.
	// A true entry denotes a complete Wilder seed and a finite computed RSI.
	Valid []bool
}

// IsNil reports whether the Result receiver is nil, including inside a nilable interface.
// The method implements lvlath's explicit nil-state convention without inspecting result data.
//
// Implementation:
//   - Stage 1: compare the pointer receiver with nil.
//   - Stage 2: return that boolean without dereferencing any field.
//
// Behavior highlights:
//   - A non-nil Result containing nil slices is not a nil Result.
//
// Inputs:
//   - r: potentially nil receiver, including a typed-nil in an interface.
//
// Returns:
//   - bool: true exactly when r==nil.
//
// Errors:
//   - None; this method is nil-safe.
//
// Determinism:
//   - Pure pointer comparison with no global or mutable state.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This method intentionally does not validate Result fields or allocate.
//
// AI-Hints:
//   - An interface holding (*Result)(nil) is not equal to nil; prefer IsNil.
func (r *Result) IsNil() bool { return r == nil }

// Snapshot is the versioned, detached transport shape for one Wilder RSI stream.
// It preserves the exact scalar recurrence state without replaying historic prices.
//
// Implementation:
//   - Stage 1: Stream.Snapshot copies the scalar state into fresh slices.
//   - Stage 2: callers may encode or authenticate the entire snapshot externally.
//   - Stage 3: Stream.Restore validates every field before atomically replacing state.
//
// Behavior highlights:
//   - Algorithm is the exact string "rsi/wilder/v1".
//   - Config is lowercase hex SHA-256 of Algorithm UTF-8 bytes followed by LE uint64(P).
//   - FloatState has exactly three entries: [previous, avgGain, avgLoss].
//   - IntState has length zero, reserved for a future ABI version.
//   - Count is the number of accepted closes; P seed changes complete at Count=P+1.
//
// Inputs:
//   - Restore accepts a caller-owned Snapshot with finite, phase-compatible state.
//
// Returns:
//   - Snapshot values own their FloatState/IntState slices and are never aliased by Stream.
//
// Errors:
//   - ErrStateVersion from Restore for incompatible schema/configuration or invalid state.
//
// Determinism:
//   - Restored binary64 bits continue the same ordered recurrence on the same runtime.
//
// Complexity:
//   - Time O(1), Space O(1) for fixed-length snapshot fields.
//
// Notes:
//   - Config identifies the algorithm and period; it does NOT authenticate state bytes.
//   - External serialization must preserve binary64 bits, and integrity needs an external MAC.
//   - Snapshot/Restore require exclusive access to the mutable stream.
//
// AI-Hints:
//   - Reject a mismatched Config; do not reinterpret averages under another period.
//   - Never treat the configuration SHA-256 as an HMAC or a tamper-proof checkpoint seal.
type Snapshot struct {
	// Algorithm identifies the fixed Wilder recurrence and the state-schema version.
	Algorithm string

	// Config binds the normalized period to Algorithm through a lowercase SHA-256 digest.
	// It does not cover Count, FloatState, or IntState.
	Config string

	// Count is the total number of successfully accepted price samples, including baseline.
	Count uint64

	// FloatState stores [previous, average gain, average loss] in this exact order.
	// All values must be finite; previous is positive when Count>0, and means are nonnegative.
	FloatState []float64

	// IntState is reserved for versioned expansion and must have length zero for v1.
	IntState []uint64
}
