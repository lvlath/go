// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package atr provides exact True Range and Wilder Average True Range computations.
// This file defines the public carriers and the nilable result contract.
package atr

// Options selects the observation count for Wilder smoothing.
// There is one smoothing law and no implicit default period.
//
// Implementation:
//   - Stage 1: Validate Period >= 1 before allocating.
//   - Stage 2: Copy the configuration into each Stream.
//   - Stage 3: Persist and compare the complete configuration in snapshots.
//
// Behavior highlights:
//   - Period one has no warmup and returns ATR == TR bit for bit.
//
// Inputs:
//   - Period: an integer observation count in [1, MaxInt], not calendar time.
//
// Returns:
//   - A value configuration with no shared mutable storage.
//
// Errors:
//   - ErrInvalidOption: New or Compute receives Period < 1.
//
// Determinism:
//   - The exact integer Period is used, never a pre-rounded alpha.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - A larger period changes readiness and smoothing, not retained history size.
//
// AI-Hints:
//   - Do not substitute the EMA weights 2/(P+1) or impose Period < MaxInt.
type Options struct {
	// Period is the number P of accepted bars used for exact seed and Wilder weights.
	// Require 1 <= Period <= MaxInt; Options{} is invalid and never selects 14 implicitly.
	Period int
}

// Result owns aligned True Range, Wilder ATR, and ATR-readiness channels.
// TR is present during warmup; only the ATR channel is masked.
//
// Implementation:
//   - Stage 1: Validate the complete input.
//   - Stage 2: Allocate detached channels and apply the common transition.
//   - Stage 3: Publish after the final context check.
//
// Behavior highlights:
//   - Every channel has input length; unavailable ATR is exactly +0.
//
// Inputs:
//   - Produced by Compute from already ordered finite candles.
//
// Returns:
//   - Caller-owned slices with no alias to inputs or to another result.
//   - Empty input returns a non-nil Result with non-nil empty slices.
//
// Errors:
//   - Compute returns nil on every failure, never a partial Result.
//
// Determinism:
//   - Index i corresponds to input bar i; values match sequential Push bit for bit.
//
// Complexity:
//   - Time O(1) per indexed read; Space O(n) for the returned channels.
//
// Notes:
//   - Copying Result copies slice headers, not the backing arrays.
//   - A valid ATR of zero is a measurement, not missing data.
//
// AI-Hints:
//   - Never apply Valid to TR or infer readiness from a nonzero ATR.
type Result struct {
	// TR contains a finite, nonnegative true range at every input index.
	// Its length equals ATR and Valid; it owns independent storage and is never warmup-masked.
	TR []float64
	// ATR contains Wilder averages, with +0 before index Period-1.
	// A ready +0 remains a valid measurement; copying Result does not copy this backing array.
	ATR []float64
	// Valid describes ATR readiness only; TR is never masked.
	// It becomes true at Period-1 and remains true; empty histories have a non-nil empty mask.
	Valid []bool
}

// IsNil reports whether the result pointer itself is nil.
// Allocated empty and all-warmup results remain non-nil.
//
// Implementation:
//   - Stage 1: Compare the receiver with nil without dereferencing it.
//
// Behavior highlights:
//   - Implements the structural interface { IsNil() bool }.
//
// Inputs:
//   - r: any *Result, including a typed nil in an interface.
//
// Returns:
//   - true exactly when r == nil.
//
// Errors:
//   - None.
//
// Determinism:
//   - Independent of slice contents and readiness.
//
// Complexity:
//   - Time O(1), Space O(1), no allocations.
//
// Notes:
//   - Safe on a nil receiver; this is not a shape validator.
//
// AI-Hints:
//   - Do not implement nilability as len(r.ATR) == 0.
func (r *Result) IsNil() bool {
	// Pointer presence is independent of channel lengths, values, and readiness.
	return r == nil
}

// Value is the detached output of one successful streaming step.
// The numeric ranges use the same price units as the supplied candle.
//
// Implementation:
//   - Stage 1: Compute TR from the current candle and the last accepted close.
//   - Stage 2: Update exact seed or Wilder recurrence.
//   - Stage 3: Publish after successful transactional validation.
//
// Behavior highlights:
//   - Valid refers only to ATR; TR is valid even when Valid is false.
//
// Inputs:
//   - Produced by a successful Stream.Push.
//
// Returns:
//   - Finite nonnegative TR and ATR; unavailable ATR is +0.
//
// Errors:
//   - A failed Push returns Value{} together with a non-nil error.
//
// Determinism:
//   - The value matches the same index in Compute.
//
// Complexity:
//   - Time O(1) to copy; Space O(1).
//
// Notes:
//   - No reference into mutable stream storage is retained.
//
// AI-Hints:
//   - Check error before using TR and check Valid before using ATR.
type Value struct {
	// TR is the current true range, valid on every successful Push.
	// Always inspect the returned error first; Value{} also accompanies failures.
	TR float64
	// ATR is +0 during warmup and the Wilder average thereafter.
	// Valid, not the numeric value or its sign, determines availability.
	ATR float64
	// Valid indicates that Period bars have been successfully accepted.
	// A rejected or canceled Push does not contribute to this count.
	Valid bool
}

// Snapshot is the complete continuation state of one Wilder stream.
// It preserves the exact pre-seed sum rather than an approximate floating mean.
//
// Implementation:
//   - Stage 1: Copy metadata, counters, and scalar state.
//   - Stage 2: Encode seed units as canonical decimal ASCII.
//   - Stage 3: Validate all fields transactionally when restoring.
//
// Behavior highlights:
//   - Version is 1 and Algorithm is atr/wilder-rn/v1.
//
// Inputs:
//   - A value returned by Stream.Snapshot, persisted losslessly by the caller.
//
// Returns:
//   - A detached value; SeedSum is a nonnegative integer in units of 2^-1074.
//
// Errors:
//   - Restore returns ErrStateVersion for inconsistent or noncanonical state.
//
// Determinism:
//   - Exact seed, PreviousClose, and Current reproduce the same continuation bits.
//
// Complexity:
//   - State has O(W+D) bounded size, W=34 limbs, D<=660 decimal bytes.
//
// Notes:
//   - Count zero requires zero PreviousClose, +0 Current, and SeedSum="0".
//   - Before Count=Period, Current=+0; at and after readiness SeedSum="0".
//   - This is not authentication or a proof that historical candles existed.
//
// AI-Hints:
//   - Preserve uint64 Count and float64 bits in external serialization.
//   - Do not replace SeedSum with a decimal ATR estimate or fabricate PreviousClose.
type Snapshot struct {
	// Version is the snapshot schema version, exactly 1.
	Version uint32
	// Algorithm identifies the exact arithmetic continuation law.
	Algorithm string
	// Options must exactly match the receiving Stream configuration.
	Options Options
	// Count is the number of successfully accepted bars.
	// Persist it as an integer; an untyped JSON float can lose counts above 2^53.
	Count uint64
	// PreviousClose retains the most recent accepted close, or zero at Count zero.
	// Restore permits either zero sign; New and Reset produce positive zero.
	PreviousClose float64
	// Current is +0 during warmup and a finite nonnegative ATR thereafter.
	// Restore preserves either zero sign in a mature externally supplied checkpoint.
	Current float64
	// SeedSum is canonical nonnegative decimal ASCII, at most 660 bytes.
	// It counts exact 2^-1074 units during warmup and is exactly "0" after readiness.
	SeedSum string
}
