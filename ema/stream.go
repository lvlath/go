// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ema provides exact affine exponential moving average smoothing.
// This file owns transactional streaming state and its allocation-free transition.
package ema

import (
	"context"
	"math"
)

// Stream holds one bounded exact exponential-smoothing history with a single owner.
// Its zero value is intentionally uninitialized; construct it with New.
//
// Implementation:
//   - Stage 1: store validated options and exact unsigned period.
//   - Stage 2: maintain a fixed 34-limb seed sum until readiness.
//   - Stage 3: retain only the rounded current value after readiness.
//
// Behavior highlights:
//   - Push and Restore publish all state atomically with respect to errors.
//
// Inputs:
//   - One finite observation per successful Push; no timestamp or window storage.
//
// Returns:
//   - Push yields Value; Snapshot yields a detached continuation record.
//
// Errors:
//   - Nil receivers: ErrNilInput; uninitialized receivers: ErrStateVersion.
//
// Determinism:
//   - The accepted observation order completely determines the state.
//
// Complexity:
//   - Space O(W), W=34 words, independent of period and history length.
//
// Notes:
//   - Concurrent calls on one Stream require external synchronization.
//   - Copying a constructed Stream makes an independent value-state fork.
//
// AI-Hints:
//   - Do not share one stream among different instruments or sampling cadences.
//   - Do not describe transactionality as lock-free concurrent thread safety.
type Stream struct {
	// options is copied at construction and remains unchanged through Reset/Restore.
	options Options
	// period is the exact validated unsigned coefficient source, never a rounded alpha.
	period uint64
	// count records successful accepts only; Push checks capacity before incrementing.
	count uint64
	// current is +0 until readiness, then the previous rounded output for the next step.
	current float64
	// seed stores exact units during SMA warmup and is fully cleared at maturity.
	seed exactSum
	// initialized distinguishes constructed state from a deliberately unusable zero value.
	initialized bool
}

// Push accepts one observation and returns the next explicitly masked EMA value.
// Every error leaves all receiver fields unchanged and returns Value{}.
//
// Implementation:
//   - Stage 1: validate context, receiver, finite input, and counter capacity.
//   - Stage 2: advance a private value copy with the common exact transition.
//   - Stage 3: recheck context immediately before assigning the entire candidate.
//
// Behavior highlights:
//   - Period one selects x directly, with either zero sign canonicalized to +0.
//
// Inputs:
//   - ctx: non-nil context; it must support allocation-free Err for zero-allocation use.
//   - x: finite binary64 observation; negative values are allowed.
//
// Returns:
//   - Value: readiness and current EMA; +0/false during SMA warmup.
//
// Errors:
//   - ErrNilContext or ctx.Err(): before validation or at final commit check.
//   - ErrNilInput, ErrStateVersion: invalid receiver, after context validation.
//   - ErrNonFinite: invalid observation, before the counter check.
//   - ErrOverflow: Count is MaxUint64, before candidate computation.
//
// Determinism:
//   - The transition rounds once to nearest-even, using exact integer weights.
//
// Complexity:
//   - Time O(W), Space O(W), W=34 fixed limbs; no internal heap allocation.
//
// Notes:
//   - Context cancellation concurrent with validation returns the first observed error.
//   - Cancellation after the final check cannot retract a completed commit.
//
// AI-Hints:
//   - Never advance Count or the seed in place before all checks succeed.
//   - An application context can itself allocate; the package cannot prevent that.
func (s *Stream) Push(ctx context.Context, x float64) (Value, error) {
	// Stage 1: context wins even when the receiver or observation is invalid.
	if err := checkContext(ctx); err != nil {
		return Value{}, err
	}
	// Reject nil before dereferencing any private field.
	if s == nil {
		return Value{}, ErrNilInput
	}
	// Zero-value streams must be constructed explicitly rather than guessed from Options.
	if !s.initialized {
		return Value{}, ErrStateVersion
	}
	// Observation classification precedes count capacity and any candidate work.
	if !finite(x) {
		return Value{}, ErrNonFinite
	}
	// Prevent count wrap before the private transition increments it.
	if s.count == math.MaxUint64 {
		return Value{}, ErrOverflow
	}
	// Stage 2: the complete accumulator is a value copy, so errors cannot leak mutations.
	candidate := *s
	// This is the same exact transition used by the batch facade.
	result := candidate.advance(x)
	// Stage 3: cancellation discards the candidate, even at the seed-completion step.
	if err := ctx.Err(); err != nil {
		return Value{}, err
	}
	// Publish every field together under exclusive ownership; this is not synchronization.
	*s = candidate
	// Return the committed measurement; a valid zero is not a warmup marker.
	return result, nil
}

// advance performs the single numerical transition shared by Compute and Push.
// It mutates only caller-owned private state; public transaction checks surround it.
//
// Implementation:
//   - Stage 1: increment the previously capacity-checked accepted count.
//   - Stage 2: select P1, SMA accumulation/completion, First initialization, or mature recurrence.
//   - Stage 3: return readiness together with the rounded current value.
//
// Behavior highlights:
//   - Exactly P accepted SMA observations precede the first valid output.
//   - The period-one identity bypasses both accumulation and recurrence.
//
// Inputs:
//   - s: initialized private stream with count<MaxUint64; not concurrently accessed.
//   - x: finite represented observation admitted by the public facade.
//
// Returns:
//   - Value: +0/false in incomplete SMA warmup; otherwise canonical finite current/true.
//
// Errors:
//   - None; the caller enforces context, receiver, finiteness, and count capacity.
//
// Determinism:
//   - One ascending accepted observation at a time; exact seed then one rounding per recurrence.
//
// Complexity:
//   - Time O(W), Space O(W) temporary arithmetic, W=34; no heap allocation.
//
// Notes:
//   - Push calls this on a copy; Compute owns a nonescaping local stream.
//   - Compute has at most len(x)<=MaxInt accepts, so its local count cannot wrap.
//
// AI-Hints:
//   - Never move the count increment into the public Push before the final context check.
//   - Never call roundUnits for an unfinished seed or retain seed limbs after maturity.
func (s *Stream) advance(x float64) Value {
	// The facade proved count capacity; every accepted warmup sample also counts.
	s.count++
	// Seed policy changes readiness, not the mature affine mathematics.
	switch {
	case s.period == 1:
		// Select x directly so a tiny current sample survives arbitrarily large old values.
		s.current = canonicalZero(x)
	case s.options.Seed == SeedSMA && s.count <= s.period:
		// The exact sum is order-independent; no running-mean rounding occurs.
		s.seed.add(binaryUnits(x))
		// Before the Pth accepted sample, expose only an unavailable measurement.
		if s.count < s.period {
			return Value{}
		}
		// The first measurement is the exact accumulated sum divided by P, rounded once.
		s.current = roundUnits(&s.seed, s.period)
		// Finished seeds must not remain in the continuation state.
		s.seed = exactSum{}
	case s.count == 1:
		// Only SeedFirst can reach this branch for a period greater than one.
		s.current = canonicalZero(x)
	default:
		// Mature histories use the same bounded exact affine recurrence in both seed modes.
		s.current = affine(s.current, x, s.period)
	}
	// All branches reaching here have a genuine measurement, including numerical zero.
	return Value{Value: s.current, Valid: true}
}

// Reset discards the accepted history but preserves the construction options.
// The next successful Push behaves exactly like the first Push on a new Stream.
//
// Implementation:
//   - Stage 1: accept nil and uninitialized receivers as harmless no-ops.
//   - Stage 2: replace all state with its initialized empty counterpart.
//
// Behavior highlights:
//   - Count, Current, and the entire seed accumulator are cleared together.
//
// Inputs:
//   - s: a stream under exclusive caller ownership, or nil.
//
// Returns:
//   - None.
//
// Errors:
//   - None.
//
// Determinism:
//   - Replaying an accepted history after Reset reproduces its previous outputs.
//
// Complexity:
//   - Time O(W), Space O(1) auxiliary; no allocations.
//
// Notes:
//   - Reset does not initialize a zero-value Stream.
//
// AI-Hints:
//   - Reset is an explicit history boundary, not a missing-bar correction.
//   - Synchronize externally against Push, Restore, and Snapshot.
func (s *Stream) Reset() {
	// Reset is a no-op for invalid receivers and never constructs one implicitly.
	if s == nil || !s.initialized {
		return
	}
	// Rebuild only the immutable construction fields; all history fields become zero.
	*s = Stream{options: s.options, period: s.period, initialized: true}
}
