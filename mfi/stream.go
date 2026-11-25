// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mfi defines transactional streaming state and explicit gap recovery.
// A Stream has a single owner and performs no allocations on successful Push.
package mfi

import "math"

// Stream is the stateful, single-owner Money Flow Index rolling engine.
// Its two sign rings, Neumaier accumulators, previous typical price, counters,
// cursor and drift flag form one logical state committed together after success.
//
// Implementation:
//   - Stage 1: New validates configuration and allocates two Period-sized rings.
//   - Stage 2: Push validates and projects the next state without modifying slots.
//   - Stage 3: commit after all numeric checks; Reset and Restore have separate gates.
//
// Behavior highlights:
//   - head points to the next slot to replace; used never exceeds period.
//   - The first bar only establishes previous typical price, with Count=1.
//   - A rebase is scheduled every Period transitions; at most one extra per epoch.
//
// Inputs:
//   - Created only via New/NewDefault/NewPeriod; feed one valid Bar at a time.
//
// Returns:
//   - Push publishes one (value,valid,error) triple for each accepted bar.
//
// Errors:
//   - Push returns ErrNilInput, ErrStateVersion, ErrNonFinite, ErrDomain or ErrOverflow.
//   - Restore returns ErrNilInput or ErrStateVersion for invalid state.
//
// Determinism:
//   - Slots are filled in arrival order, rotating from index zero modulo period.
//
// Complexity:
//   - Persistent Space O(Period); Push amortized Time O(1), worst-case O(Period).
//
// Notes:
//   - No locks are held; concurrent operations on one Stream are unsupported.
//   - External timestamp gaps must be detected by the caller and trigger Reset.
//
// AI-Hints:
//   - Never drop unchanged or zero-volume transitions, including during warmup.
//   - Do not copy Stream by value: use detached Snapshot to clone the state.
type Stream struct {
	period             int
	pos, neg           []float64
	positive, negative compensated
	previous           float64
	count              uint64
	head, used         int
	extraRebaseUsed    bool
}

// ready identifies a properly constructed engine, not a zero-value Stream.
// The fast check is required before any access to the ring or modulo by period.
//
// Implementation:
//   - Stage 1: reject nil or zero-value receivers.
//   - Stage 2: reject an invalid effective period.
//   - Stage 3: require both ring slice lengths to match that period.
//
// Behavior highlights:
//   - This is a fast constructor-state check, not a full Snapshot consistency audit.
//
// Returns:
//   - bool: true only when receiver, period and both slice lengths are initialized.
//
// Determinism:
//   - Checks only local structural metadata, without scans or side effects.
//
// Complexity:
//   - Time O(1), Space O(1), zero allocation.
//
// Notes:
//   - Only New/Restore create or install full valid states; the field set is unexported.
//
// AI-Hints:
//   - Call ready before the first array access or period-based modulus operation.
func (s *Stream) ready() bool {
	return s != nil && s.period > 0 && s.period <= MaxPeriod &&
		len(s.pos) == s.period && len(s.neg) == s.period
}

// Push processes one OHLCV bar as a fully transactional stream update.
// A valid first bar seeds the previous typical price; every later bar advances
// the ring exactly once, even when volume is zero or typical price is unchanged.
//
// Implementation:
//   - Stage 1: validate receiver, full OHLCV domain, count headroom and typical price.
//   - Stage 2: classify a signed flow and project ring eviction/insertion and rebases.
//   - Stage 3: verify effective sums and MFI, then publish all fields and one ring slot.
//
// Behavior highlights:
//   - Numeric errors never modify previous typical, head, used, rings or compensators.
//   - Until Period transitions exist, the return is value=0 and valid=false.
//   - Once warmed up, the returned MFI is finite and within [0,100].
//
// Inputs:
//   - bar: copied OHLCV value with finite positive prices and finite volume>=0.
//
// Returns:
//   - value: MFI when valid; otherwise zero on warmup or error.
//   - valid: true exactly after at least Period accepted transitions.
//   - err: nil on success; false validity can still be successful warmup.
//
// Errors:
//   - ErrNilInput: nil Stream receiver.
//   - ErrStateVersion: uninitialized or structurally invalid engine.
//   - ErrNonFinite, ErrDomain: invalid caller-supplied bar.
//   - ErrOverflow: count exhaustion, flow/sum overflow or unresolved negative drift.
//
// Determinism:
//   - Exact float64 comparisons; eviction before insertion; ordered rebase every Period.
//
// Complexity:
//   - Amortized Time O(1), worst Time O(Period), Space O(1) extra, zero allocation.
//
// Notes:
//   - Not safe for concurrent access or concurrent mutation of a source bar buffer.
//   - A successful return does not say the market signal is actionable.
//
// AI-Hints:
//   - Classify using current typical price, but never multiply on a zero-volume bar.
//   - Check valid before consuming value as an indicator, especially at value zero.
func (s *Stream) Push(bar Bar) (value float64, valid bool, err error) {
	if s == nil {
		return 0, false, ErrNilInput
	}
	if !s.ready() {
		return 0, false, ErrStateVersion
	}
	if err := validateBar(bar); err != nil {
		return 0, false, err
	}
	if s.count == math.MaxUint64 {
		return 0, false, ErrOverflow
	}
	// The exact convex mean of validated positive finite prices is representable.
	current := typical(bar)

	// The first price fixes the comparison baseline without introducing a transition.
	if s.count == 0 {
		s.previous, s.count = current, 1
		return 0, false, nil
	}

	// Every subsequent valid candle creates one candidate ring entry, including zeros.
	posFlow, negFlow, err := classify(s.previous, current, bar.Volume)
	if err != nil {
		return 0, false, err
	}
	pos, neg, extraRebase, err := s.projected(posFlow, negFlow)
	if err != nil {
		return 0, false, err
	}
	positiveSum, err := pos.value()
	if err != nil {
		return 0, false, err
	}
	negativeSum, err := neg.value()
	if err != nil {
		return 0, false, err
	}
	value, err = ratio(positiveSum, negativeSum)
	if err != nil {
		return 0, false, err
	}

	// All failures have been eliminated. Update linked state without any allocation.
	s.pos[s.head], s.neg[s.head] = posFlow, negFlow
	s.head++
	if s.head == s.period {
		s.head = 0
	}
	if s.used < s.period {
		s.used++
	}
	s.positive, s.negative = pos, neg
	s.extraRebaseUsed = extraRebase
	s.previous = current
	s.count++
	if s.used < s.period {
		return 0, false, nil
	}
	return value, true, nil
}

// Reset drops the historical transition window without reallocating its storage.
// It is the caller's explicit response to an OHLCV timestamp gap or new session.
//
// Implementation:
//   - Stage 1: make a nil receiver a safe no-op.
//   - Stage 2: clear both ring allocations and accumulated corrections.
//   - Stage 3: reset cursor, previous price, Count and rebase epoch.
//
// Behavior highlights:
//   - Reset preserves allocated rings and configured period, but removes all numerical history.
//   - Nil or zero-value receivers remain safe without manufacturing initialized state.
//
// Inputs:
//   - s: nil-safe receiver; an initialized stream retains its period and ring capacity.
//
// Returns:
//   - No result; a valid engine becomes an empty engine with the same period.
//
// Determinism:
//   - All existing slots are zeroed in physical index order.
//
// Complexity:
//   - Time O(Period), Space O(1) extra, zero allocation.
//
// Notes:
//   - Reset on a zero-value Stream remains harmless but does not initialize it.
//   - Concurrent Push or Restore during Reset is unsupported.
//
// AI-Hints:
//   - Reset intentionally requires a fresh Period+1-bar warmup after every gap.
func (s *Stream) Reset() {
	if s == nil {
		return
	}
	clear(s.pos)
	clear(s.neg)
	s.positive, s.negative = compensated{}, compensated{}
	s.previous, s.count, s.head, s.used = 0, 0, 0, 0
	s.extraRebaseUsed = false
}
