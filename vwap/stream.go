// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package vwap defines a single-owner transactional anchored state machine.
// This file publishes a detached candidate only after validation and the final context check.
package vwap

import "context"

// maxSegmentCount is the largest accepted row count before a non-reset Push fails.
// Explicit reset starts a new counter before the current valid row is accepted.
const maxSegmentCount = ^uint64(0)

// Stream is a single-owner, fixed-storage processor for explicitly reset segments.
// Construct it with New; its zero value is intentionally not a usable processor.
//
// Implementation:
//   - Stage 1: validate context, receiver, first anchor, row, and count.
//   - Stage 2: apply the exact transition to a value copy of all state.
//   - Stage 3: check context and publish the complete candidate.
//
// Behavior highlights:
//   - Invalid reset rows cannot erase a previously accepted segment.
//
// Inputs:
//   - Finite signed prices, finite nonnegative volumes, and caller-owned resets.
//
// Returns:
//   - Push produces the same bits and masks as Compute.
//
// Errors:
//   - Nil receivers give ErrNilInput; unconstructed receivers give ErrStateVersion.
//
// Determinism:
//   - Exact integer sums and one nearest-even rounding per published observation.
//
// Complexity:
//   - O(L) time and O(L) space per Push, L=67 fixed limbs; O(1) in history length.
//
// Notes:
//   - No same-instance concurrent calls are supported. Independent instances are safe.
//   - State contains no slice or pointer aliases; a constructed value copy is detached.
//
// AI-Hints:
//   - Use external synchronization for shared access, including Snapshot.
//   - Reset is not a trading-session detector; supply a causal reset policy yourself.
type Stream struct {
	// initialized distinguishes New's constructed state from an unusable zero value.
	initialized bool
	// options is copied by value and survives Reset; Restore requires equality.
	options Options
	// count includes accepted zero-volume rows but excludes all rejected rows.
	count uint64
	// started records a successful explicit first anchor, including a zero-volume row.
	started bool
	// sums stores fixed-size values, so copying a Stream cannot alias arithmetic state.
	sums accumulator
	// current caches the rounded mean and retains its exact bits on zero-volume rows.
	current Value
}

// Push accepts one observation and returns its exact-volume weighted prefix mean.
// The reset, count, sums, and output are committed together or not at all.
//
// Implementation:
//   - Stage 1: check context, construction, anchor, finite domains, and count.
//   - Stage 2: reset and accumulate on a detached value copy.
//   - Stage 3: poll context immediately before assigning the complete new state.
//
// Behavior highlights:
//   - A fresh or explicitly Reset stream requires reset=true on its first success.
//
// Inputs:
//   - ctx: non-nil context; cancellation is returned unchanged.
//   - price: any finite binary64 value, including negative values.
//   - volume: finite nonnegative binary64 weight; either zero sign is accepted.
//   - reset: discard the preceding segment before including this accepted row.
//
// Returns:
//   - Value with a positive-volume mask; Value{} on every error.
//
// Errors:
//   - ErrNilContext or ctx.Err() precedes receiver inspection.
//   - ErrNilInput, ErrStateVersion, ErrAnchor, ErrNonFinite, ErrVolume, ErrOverflow.
//
// Determinism:
//   - Input order determines prefixes; reset=false cannot revise an earlier anchor.
//
// Complexity:
//   - Time O(L), Space O(L), L=67 fixed limbs; no internal heap allocations.
//
// Notes:
//   - A cancellation after the final successful poll is outside this publication boundary.
//   - A custom context implementation may allocate independently of this method.
//
// AI-Hints:
//   - Do not mutate live sums before validating a requested reset row.
//   - A zero-volume row still counts and still requires a finite price.
func (s *Stream) Push(ctx context.Context, price, volume float64, reset bool) (Value, error) {
	// Stage 1: context wins over receiver, anchor, and observation-domain failures.
	if err := validateContext(ctx); err != nil {
		return Value{}, err
	}
	if err := s.validateReceiver(); err != nil {
		return Value{}, err
	}
	// A previously rejected row cannot establish the mandatory first anchor.
	if !s.started && !reset {
		return Value{}, ErrAnchor
	}
	// Validate even a requested reset before touching the active segment.
	if err := validateRow(price, volume); err != nil {
		return Value{}, err
	}
	// A legal explicit reset creates headroom; otherwise zero-volume rows count too.
	if !reset && s.count == maxSegmentCount {
		return Value{}, ErrOverflow
	}
	// Stage 2: every state field is value-owned, so this candidate has no mutable aliases.
	next := *s
	value := next.accept(price, volume, reset)
	// Stage 3: a cancellation here discards both the candidate reset and its observation.
	if err := ctx.Err(); err != nil {
		return Value{}, err
	}
	// The successful final poll is the commit boundary, not a concurrency lock.
	*s = next
	return value, nil
}

// validateReceiver checks whether a stream is a constructed processor.
// Nil and zero-value receivers are distinct invalid states with stable error identities.
//
// Implementation:
//   - Stage 1: test receiver pointer presence.
//   - Stage 2: test the private construction marker.
//   - Stage 3: admit constructed receivers without reading numerical inputs.
//
// Behavior highlights:
//   - This method is shared by Push and Restore and never changes the receiver.
//
// Inputs:
//   - s: possibly nil Stream pointer.
//
// Returns:
//   - error: nil only for a constructed stream.
//
// Errors:
//   - ErrNilInput for nil; ErrStateVersion for an unconstructed Stream{}.
//
// Determinism:
//   - Input order and integer comparisons determine the result; no shared mutable state is read.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Push validates context before this check; Restore has no context parameter.
//
// AI-Hints:
//   - Reset on a zero value is deliberately not an alternative constructor.
func (s *Stream) validateReceiver() error {
	// Never read the construction marker through a nil receiver.
	if s == nil {
		return ErrNilInput
	}
	// A zero-value Stream is not silently promoted to a constructed stream.
	if !s.initialized {
		return ErrStateVersion
	}
	return nil
}

// accept executes the single canonical accepted-observation transition.
// Compute and Push share this operation so their exact sums and outputs cannot drift.
//
// Implementation:
//   - Stage 1: when requested, clear the preceding segment before accepting the current row.
//   - Stage 2: mark the segment started and increment its checked row count.
//   - Stage 3: for positive volume update exact sums and the rounded cache; otherwise retain the cache.
//
// Behavior highlights:
//   - A zero-volume reset produces a started masked segment with Count=1.
//
// Inputs:
//   - s: exclusively owned local state or a detached Push candidate.
//   - price, volume: validated finite inputs with volume>=0.
//   - reset: explicit caller decision; a fresh state must be anchored.
//
// Returns:
//   - Value: the post-row cache with readiness determined by positive cumulative volume.
//
// Errors:
//   - No error return; the documented private preconditions are established by the caller.
//
// Determinism:
//   - Input order and integer comparisons determine the result; no shared mutable state is read.
//
// Complexity:
//   - Time O(L), Space O(L), L<=67; zero-volume continuation takes constant scalar work.
//
// Notes:
//   - The caller establishes count headroom or reset before entering this unchecked kernel.
//
// AI-Hints:
//   - Do not move reset after inclusion or skip Count for zero-volume rows.
func (s *Stream) accept(price, volume float64, reset bool) Value {
	// Reset precedes this observation and also clears a formerly ready cache.
	if reset {
		s.count = 0
		s.sums = accumulator{}
		s.current = Value{}
	}
	// All accepted observations, including zero weights, advance the segment phase/count.
	s.started = true
	s.count++
	// Either signed zero leaves the mean bits unchanged; a reset may have just cleared them.
	if volume != 0 {
		s.sums.include(price, volume)
		s.current = s.sums.value()
	}
	return s.current
}

// Reset clears accepted history while retaining construction and configuration.
// The next successful observation must explicitly anchor a new segment again.
//
// Implementation:
//   - Stage 1: ignore nil or unconstructed receivers.
//   - Stage 2: construct an empty state with the current Options.
//   - Stage 3: replace the complete receiver.
//
// Behavior highlights:
//   - Reset(nil) is a safe no-op and does not construct a zero-value stream.
//
// Inputs:
//   - s: a stream owned exclusively by the caller.
//
// Returns:
//   - No result; a constructed stream becomes equivalent to New(options).
//
// Errors:
//   - None.
//
// Determinism:
//   - Clears both exact sums, Count, Started, and cached output.
//
// Complexity:
//   - Time O(L), Space O(1) auxiliary storage, L=67 fixed limbs.
//
// Notes:
//   - To reset and accept a row transactionally, use Push with reset=true instead.
//
// AI-Hints:
//   - Do not call Reset before validating a row intended for a transactional reset.
//   - Reset cannot undo observations already published to external consumers.
func (s *Stream) Reset() {
	// Reset is nil-safe but never acts as a constructor for the zero value.
	if s == nil || !s.initialized {
		return
	}
	// Preserve configuration while atomically replacing all historical fields for the owner.
	*s = Stream{initialized: true, options: s.options}
}
