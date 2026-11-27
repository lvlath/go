// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package vwap defines owning results and the public exact continuation carriers.
// This file documents field units, readiness, ownership, and nil-safe result behavior.
package vwap

// Options is the complete configuration carrier for the v1 VWAP algorithm.
// It has no fields: every representable value selects the same exact arithmetic.
//
// Implementation:
//   - Stage 1: admit the single empty configuration value.
//   - Stage 2: retain it in a constructed stream.
//   - Stage 3: include it in checkpoints and require equality on restoration.
//
// Behavior highlights:
//   - There is no implicit price proxy, calendar, window, or anchor selection.
//
// Inputs:
//   - Supply Options{} to Compute or New.
//
// Returns:
//   - A comparable, immutable configuration value.
//
// Errors:
//   - None in v1; ErrInvalidOption is retained for API compatibility and has no reachable v1 condition.
//
// Determinism:
//   - All Options values are equal and choose the same kernel.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - The empty value is copied into every checkpoint and compared on restoration.
//
// AI-Hints:
//   - Do not assume the empty configuration selects a calendar or an implicit first anchor.
//   - Choose the input price and causal reset schedule outside this package.
type Options struct{}

// Result owns the aligned batch VWAP values and positive-volume validity mask.
// A false mask has a strict +0 payload; a valid zero is still a measurement.
//
// Implementation:
//   - Stage 1: validate context, shape, anchor, and all input rows.
//   - Stage 2: traverse the canonical state transition in input order.
//   - Stage 3: publish detached channels only after the final context check.
//
// Behavior highlights:
//   - Successful empty input has a non-nil Result and non-nil empty channels.
//
// Inputs:
//   - Created by Compute from already ordered observations.
//
// Returns:
//   - Values and Valid have the same length as each input channel.
//
// Errors:
//   - Compute returns nil, error rather than a partially usable Result.
//
// Determinism:
//   - Values match sequential Push calls bit for bit, including signed underflow.
//
// Complexity:
//   - Storage O(n); IsNil is O(1).
//
// Notes:
//   - The caller owns both slices. Copying Result is a shallow slice-header copy.
//
// AI-Hints:
//   - Never infer readiness from Values[i] != 0; inspect Valid[i].
//   - Output mutations do not modify caller inputs or other Compute results.
type Result struct {
	// Values owns one prefix mean per input observation, in the supplied price units.
	// Invalid entries are exactly +0; valid entries may be zero, negative, or positive.
	// Mutating this slice cannot affect inputs, another result, or any Stream.
	Values []float64
	// Valid owns the positive-cumulative-volume mask, aligned one-to-one with Values.
	// An accepted zero-volume row retains readiness unless that row resets the segment.
	Valid []bool
}

// Value is the output of one successfully accepted streaming observation.
// It separates an undefined pre-volume state from a legitimate zero average.
//
// Implementation:
//   - Stage 1: apply an explicit reset to a temporary state when requested.
//   - Stage 2: add the observation exactly and round only the final quotient.
//   - Stage 3: publish state and Value together after the final context check.
//
// Behavior highlights:
//   - A zero-volume row retains the previous Value within the same segment.
//
// Inputs:
//   - Produced by Stream.Push; it contains no borrowed memory.
//
// Returns:
//   - Value=+0 and Valid=false before the first positive segment volume.
//
// Errors:
//   - A failed Push returns Value{} and leaves the stream unchanged.
//
// Determinism:
//   - Exact zero is +0; negative nonzero underflow may round to -0.
//
// Complexity:
//   - Time O(1), Space O(1) to copy this carrier.
//
// Notes:
//   - Count includes accepted zero-volume observations, although Value may not change.
//
// AI-Hints:
//   - Do not treat a valid zero as missing data.
//   - Do not substitute an invalid quote merely because its volume is zero.
type Value struct {
	// Value is the rounded current prefix mean, or exactly +0 while undefined.
	// A negative nonzero exact mean may round to -0 without becoming invalid.
	Value float64
	// Valid distinguishes positive cumulative volume from an undefined 0/0 mean.
	// Inspect this flag instead of treating a numeric zero as missing information.
	Valid bool
}

// Snapshot is the detached, exact continuation format for a constructed stream.
// It records integer sums of supplied binary64 inputs, not rounded running averages.
//
// Implementation:
//   - Stage 1: capture the configuration, phase, and accepted segment count.
//   - Stage 2: serialize exact sums using canonical decimal ASCII integers.
//   - Stage 3: Restore validates every field before replacing live state.
//
// Behavior highlights:
//   - Version is 1 and Algorithm is "vwap/exact-weighted/v1".
//
// Inputs:
//   - Restore accepts at most 1300 bytes in each sum field.
//
// Returns:
//   - VolumeSum uses units 2^-1074; WeightedSum uses units 2^-2148.
//
// Errors:
//   - Restore returns ErrStateVersion for any invalid field or invariant.
//
// Determinism:
//   - Grammar is 0|-?[1-9][0-9]*, with a nonnegative VolumeSum.
//
// Complexity:
//   - Fixed bounded storage; decimal conversion costs O(D*L), D<=1300, L=67.
//
// Notes:
//   - This format is not authentication or evidence of historical provenance.
//
// AI-Hints:
//   - Persist Count losslessly as uint64; generic JSON numbers can lose its bits.
//   - Never replace exact sums by a saved rounded VWAP.
type Snapshot struct {
	// Version must equal 1; a nil or unconstructed stream instead returns Snapshot{}.
	Version uint32
	// Algorithm must equal "vwap/exact-weighted/v1"; it identifies numerical semantics.
	Algorithm string
	// Options must equal the complete configuration retained by the destination New.
	Options Options
	// Count counts accepted rows, including zero volume, since the last explicit reset.
	// Preserve this uint64 losslessly; untyped JSON-to-float64 decoding is unsuitable.
	Count uint64
	// Started must equal Count != 0; it does not authenticate the anchor's availability.
	Started bool
	// VolumeSum is a canonical nonnegative ASCII integer in units of 2^-1074.
	// It is "0" before any positive volume and cannot exceed Count*MaxFloat64Units.
	VolumeSum string
	// WeightedSum is a canonical signed ASCII integer in units of 2^-2148.
	// Its magnitude cannot exceed VolumeSum*MaxFloat64Units. Both strings are <=1300 bytes.
	WeightedSum string
}

// IsNil reports pointer absence without interpreting the result's channels.
// It implements the structural nilable-result convention without reflection.
//
// Implementation:
//   - Stage 1: inspect the receiver pointer.
//   - Stage 2: compare it with nil.
//   - Stage 3: return the comparison unchanged.
//
// Behavior highlights:
//   - A non-nil empty Result is not nil.
//
// Inputs:
//   - r may be nil.
//
// Returns:
//   - true exactly when r == nil.
//
// Errors:
//   - None.
//
// Determinism:
//   - A pure pointer comparison.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - No channel is read and no state is mutated.
//
// AI-Hints:
//   - Do not redefine nilness as empty, invalid, or not warmed up.
//   - A typed nil pointer held by an interface still reports true.
func (r *Result) IsNil() bool {
	return r == nil
}
