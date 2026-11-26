// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ema provides exact affine exponential moving average smoothing.
// This file defines the public value carriers and their ownership contracts.
package ema

// Seed selects how the first valid exponential average is initialized.
// It changes readiness and seeding, but never the subsequent affine recurrence.
//
// Implementation:
//   - Stage 1: validate the enum in New or Compute.
//   - Stage 2: collect an exact SMA seed or select the first observation.
//   - Stage 3: apply the common correctly rounded recurrence thereafter.
//
// Behavior highlights:
//   - SeedSMA is the zero enum value; there is no implicit period default.
//
// Inputs:
//   - Options.Seed: one of SeedSMA or SeedFirst.
//
// Returns:
//   - A policy stored unchanged in snapshots.
//
// Errors:
//   - ErrInvalidOption: an unknown enum is rejected during construction.
//
// Determinism:
//   - Initialization consumes observations in ascending input order.
//
// Complexity:
//   - Time O(1), Space O(1) for the policy itself.
//
// Notes:
//   - The policy cannot be changed during Restore.
//
// AI-Hints:
//   - Never substitute first-observation initialization for SMA warmup.
//   - Select the mode explicitly when comparing another library's output.
type Seed uint8

const (
	// SeedSMA selects exact arithmetic-mean initialization over the first P observations.
	// The selected phase policy is preserved unchanged by Snapshot and Restore.
	//
	// Implementation:
	//   - Stage 1: select this enum in Options or a named facade wrapper.
	//   - Stage 2: accumulate P exact observations before the first measurement.
	//   - Stage 3: use the same exact affine recurrence for every later measurement.
	//
	// Behavior highlights:
	//   - First valid index is P-1; preceding payloads are +0 with false masks.
	//
	// Inputs:
	//   - Options.Seed: use with an explicit valid Period.
	//
	// Returns:
	//   - An immutable seed-policy enum; it carries no observation state.
	//
	// Errors:
	//   - New and Compute still reject an invalid Period with ErrInvalidOption.
	//
	// Determinism:
	//   - Initialization consumes the original accepted prefix in order.
	//
	// Complexity:
	//   - Policy selection is O(1) time/space; state retains W=34 limbs at most.
	//
	// Notes:
	//   - P=1 selects x directly under either policy, with canonical +0 output.
	//
	// AI-Hints:
	//   - Do not compare two EMA histories without matching their seed policy.
	SeedSMA Seed = iota
	// SeedFirst selects immediate initialization from the first observation.
	// The selected phase policy is preserved unchanged by Snapshot and Restore.
	//
	// Implementation:
	//   - Stage 1: select this enum in Options or a named facade wrapper.
	//   - Stage 2: publish the first finite input immediately.
	//   - Stage 3: use the same exact affine recurrence for every later measurement.
	//
	// Behavior highlights:
	//   - First valid index is zero; no P-sample warmup is fabricated.
	//
	// Inputs:
	//   - Options.Seed: use with an explicit valid Period.
	//
	// Returns:
	//   - An immutable seed-policy enum; it carries no observation state.
	//
	// Errors:
	//   - New and Compute still reject an invalid Period with ErrInvalidOption.
	//
	// Determinism:
	//   - Initialization consumes the original accepted prefix in order.
	//
	// Complexity:
	//   - Policy selection is O(1) time/space; state retains W=34 limbs at most.
	//
	// Notes:
	//   - P=1 selects x directly under either policy, with canonical +0 output.
	//
	// AI-Hints:
	//   - Do not compare two EMA histories without matching their seed policy.
	SeedFirst
)

// Options configures the period and initialization of one independent history.
// Construction copies this value; subsequent caller mutation has no effect.
//
// Implementation:
//   - Stage 1: require 1 <= Period < MaxInt and a recognized Seed.
//   - Stage 2: retain exact integer weights Period-1, 2, and Period+1.
//   - Stage 3: include the complete configuration in every snapshot.
//
// Behavior highlights:
//   - There are no defaults: Options{} is invalid.
//
// Inputs:
//   - Period: count of observations, not a calendar duration.
//   - Seed: the explicit initialization policy.
//
// Returns:
//   - A copied configuration, never a mutable alias held by Stream.
//
// Errors:
//   - ErrInvalidOption: invalid period, overflowing Period+1, or unknown seed.
//
// Determinism:
//   - Integer weights are never approximated by a rounded alpha.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Very large periods may be valid on 64-bit but not 32-bit targets.
//
// AI-Hints:
//   - Do not convert Period to float64 to compute the canonical weights.
//   - Timestamp validation and gap handling belong to the caller.
type Options struct {
	// Period is the explicit observation count P in [1, MaxInt-1].
	// It selects exact weights P-1, 2, P+1; it is not a duration or retained window size.
	Period int
	// Seed selects the first valid index: P-1 for SeedSMA, zero for SeedFirst.
	// The zero enum selects SeedSMA, but it never supplies a missing Period.
	Seed Seed
}

// Result owns the smoothed history and its explicitly aligned readiness mask.
// Each successful Compute call allocates its own channels, neither of which aliases x.
//
// Implementation:
//   - Stage 1: validate context, options, and all observations.
//   - Stage 2: allocate independent output channels and traverse the history.
//   - Stage 3: publish only after the final context check succeeds.
//
// Behavior highlights:
//   - Invalid positions contain positive zero; a valid zero remains valid.
//
// Inputs:
//   - Produced by Compute from a finite caller-owned history.
//
// Returns:
//   - Empty input produces a non-nil Result with non-nil empty slices.
//
// Errors:
//   - Compute returns nil on every error, never a partial Result.
//
// Determinism:
//   - Values match successive Push calls bit for bit after zero normalization.
//
// Complexity:
//   - Space O(n); querying a channel entry takes O(1) time.
//
// Notes:
//   - The caller may mutate the returned slices after Compute returns.
//   - Copying Result by value copies slice headers, not arrays; such copies share storage.
//
// AI-Hints:
//   - Check Valid instead of testing whether Values[i] equals zero.
//   - Keep the original index alignment when joining other features.
type Result struct {
	// Values preserves input indexes and units; every element is finite.
	// Unavailable entries are exactly +0; non-nil empty slices represent an empty input.
	Values []float64
	// Valid distinguishes unfinished warmup from actual measurements, including a valid zero.
	// Its length equals len(Values) on successful Compute output; callers must preserve it.
	Valid []bool
}

// IsNil reports only whether the Result pointer itself is nil.
// An allocated empty result and an allocated all-warmup result are not nil.
//
// Implementation:
//   - Stage 1: compare the receiver with nil without dereferencing it.
//   - Stage 2: return that comparison without inspecting the slices.
//
// Behavior highlights:
//   - Satisfies the structural interface { IsNil() bool }.
//
// Inputs:
//   - r: any *Result, including a typed nil pointer.
//
// Returns:
//   - bool: true exactly when r == nil.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pointer-presence semantics are independent of slice contents.
//
// Complexity:
//   - Time O(1), Space O(1), no allocations.
//
// Notes:
//   - This method is not a readiness or shape validator.
//
// AI-Hints:
//   - Never implement this contract as len(r.Values) == 0.
//   - A typed nil inside an interface still calls this safe method.
func (r *Result) IsNil() bool {
	// Pointer presence is independent of readiness, shape, and slice contents.
	return r == nil
}

// Value is the detached output of one successful streaming transition.
// Readiness records seed completion and is independent of the numeric payload.
//
// Implementation:
//   - Stage 1: prepare a private candidate state.
//   - Stage 2: compute the exact seed or affine step.
//   - Stage 3: return this value only after committing the candidate.
//
// Behavior highlights:
//   - Warmup outputs and failed Push calls both return Value{}; inspect error.
//
// Inputs:
//   - Produced by Push, not supplied as mutable stream state.
//
// Returns:
//   - Value contains finite canonical output whenever Valid is true.
//
// Errors:
//   - Push reports errors separately and returns Value{} on failure.
//
// Determinism:
//   - Equal accepted prefixes produce equal values and masks.
//
// Complexity:
//   - Time O(1), Space O(1) to copy.
//
// Notes:
//   - A Value contains no reference to the owning Stream.
//
// AI-Hints:
//   - Do not treat a failed Push as an accepted warmup observation.
//   - Forward readiness together with the numeric channel.
type Value struct {
	// Value is the current rounded EMA in input units, or +0 when unavailable.
	// This scalar owns no reference to the live stream.
	Value float64
	// Valid is true once the chosen seed is complete, even when Value equals zero.
	// Failed Push calls return Value{} and a non-nil error; they accept no observation.
	Valid bool
}

// Snapshot is the complete versioned continuation state of one Stream.
// It preserves a bounded exact seed sum rather than a lossy intermediate mean.
//
// Implementation:
//   - Stage 1: copy schema identity, configuration, and successful Count.
//   - Stage 2: encode a warmup sum as signed decimal units of 2^-1074.
//   - Stage 3: Restore validates every field before publishing a candidate.
//
// Behavior highlights:
//   - Version is 1 and Algorithm is "ema/affine-rn/v1".
//   - SeedSum follows 0|-?[1-9][0-9]* and is at most 660 ASCII bytes.
//
// Inputs:
//   - Restore accepts only a snapshot with the receiver's exact Options.
//
// Returns:
//   - A detached value; its immutable SeedSum string never aliases live limbs.
//
// Errors:
//   - ErrStateVersion: malformed schema, configuration, number, or phase.
//
// Determinism:
//   - SeedSum is canonical decimal; all published zeros are positive.
//
// Complexity:
//   - Space O(1) under fixed 660-byte and 2176-bit bounds.
//
// Notes:
//   - Persist float64 bits and uint64 Count losslessly; authenticate externally.
//   - A structurally valid snapshot does not prove historical provenance.
//
// AI-Hints:
//   - Never store Count in a generic floating-point JSON number container.
//   - Do not replace SeedSum with a rounded sum or alter Algorithm in place.
type Snapshot struct {
	// Version is exactly 1 on a constructed stream; zero denotes an empty invalid record.
	Version uint32
	// Algorithm identifies the exact arithmetic: "ema/affine-rn/v1".
	// Relabeling a different rounding rule cannot convert it into this continuation format.
	Algorithm string
	// Options is a detached copy and must exactly match the receiving Stream configuration.
	// Editing this public record never changes the originating stream.
	Options Options
	// Count is the number of successful accepts, including every warmup observation.
	// MaxUint64 can be restored, but another Push then fails until Reset.
	Count uint64
	// Current is +0 for Count zero and during SMA warmup; otherwise it is the last finite EMA.
	// Persist its binary64 value losslessly, including the canonical positive-zero sign.
	Current float64
	// SeedSum is canonical signed decimal integer units of 2^-1074, at most 660 ASCII bytes.
	// During SMA warmup its magnitude is <=Count*MaxBinary64Units; otherwise it is "0".
	SeedSum string
}
