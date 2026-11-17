// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mfi defines the compensated rolling window and deterministic rebasing.
// Projections are computed without changing any committed ring slot or accumulator.
package mfi

import "math"

// compensated retains a Neumaier sum and the discarded low-order residual.
// The effective value is sum+correction, checked for finiteness by value.
// The two fields must be copied together when staging or restoring a transition.
//
// Behavior highlights:
//   - Both halves are stored and snapshotted, not just their rounded sum.
//
// Determinism:
//   - Updates always use the canonical Neumaier branch and caller-specified order.
//
// Complexity:
//   - Persistent state O(1) per sign; update Time O(1), Space O(1).
//
// Notes:
//   - A compensated sum reduces drift, but does not make large sums unbounded or exact.
//
// AI-Hints:
//   - Copy both sum and correction when staging a candidate transition or checkpoint.
type compensated struct {
	sum, correction float64
}

// add applies one signed update to a local Neumaier accumulator.
// Eviction uses a negative update while insertion uses a nonnegative contribution;
// the caller must not publish partial updates if either addition overflows.
//
// Implementation:
//   - Stage 1: form and validate the new floating sum without mutating the accumulator.
//   - Stage 2: recover low-order cancellation according to Neumaier operand magnitudes.
//   - Stage 3: validate and publish both new halves only after full success.
//
// Behavior highlights:
//   - Works for nonnegative insertion and negated eviction values in a fixed order.
//
// Inputs:
//   - value: finite signed update, negative for an expired ring contribution.
//
// Returns:
//   - error: nil for committed local accumulator update; ErrOverflow otherwise.
//
// Errors:
//   - ErrOverflow: new sum or compensating correction becomes nonfinite.
//
// Determinism:
//   - Magnitude comparison selects the same correction branch for the same float64 operands.
//
// Complexity:
//   - Time O(1), Space O(1), zero allocation.
//
// Notes:
//   - The receiver is a local candidate during Push; no ring mutation occurs here.
//
// AI-Hints:
//   - Do not simplify Neumaier updates to sum += new-old; it changes drift behavior.
func (acc *compensated) add(value float64) error {
	candidate := acc.sum + value
	if !finite(candidate) {
		return ErrOverflow
	}

	// Correct for low-order bits lost in the addition, using the larger operand.
	var lost float64
	if math.Abs(acc.sum) >= math.Abs(value) {
		lost = (acc.sum - candidate) + value
	} else {
		lost = (value - candidate) + acc.sum
	}
	correction := acc.correction + lost
	if !finite(correction) {
		return ErrOverflow
	}

	// Both halves are representable; atomically publish the local accumulator update.
	acc.sum, acc.correction = candidate, correction
	return nil
}

// value recovers the effective compensated sum, rejecting any final overflow.
// The sign is not clipped: a negative result must trigger rebase or ErrOverflow.
//
// Implementation:
//   - Stage 1: combine sum and correction into the effective value.
//   - Stage 2: reject a nonfinite effective result.
//   - Stage 3: return it without changing either accumulator component.
//
// Behavior highlights:
//   - A negative finite result is returned unchanged for rebase/error handling.
//
// Returns:
//   - float64: effective compensated sum on success; zero on failure.
//   - error: ErrOverflow for a nonfinite effective sum.
//
// Errors:
//   - ErrOverflow: sum+correction cannot be published as finite float64.
//
// Determinism:
//   - The binary64 addition order is fixed: sum followed by correction.
//
// Complexity:
//   - Time O(1), Space O(1), zero allocation.
//
// Notes:
//   - This helper does not clamp negatives; projected/ratio enforce their contracts.
//
// AI-Hints:
//   - Check the returned error and sign before treating the value as a money-flow total.
func (acc compensated) value() (float64, error) {
	result := acc.sum + acc.correction
	if !finite(result) {
		return 0, ErrOverflow
	}
	return result, nil
}

// rebaseVirtual replays the candidate ring after a pending overwrite.
// The logical traversal always starts at the oldest occupied transition and
// uses the new pair at head without actually modifying that physical slot.
//
// Implementation:
//   - Stage 1: select the oldest slot of the would-be ring.
//   - Stage 2: apply each positive/negative pair to new Neumaier accumulators.
//   - Stage 3: return both candidates only after the entire replay succeeds.
//
// Behavior highlights:
//   - The pending overwritten slot is read as the new contribution only; no live slot changes.
//
// Inputs:
//   - head: next physical slot to be overwritten in [0, Period).
//   - used: candidate occupied slot count in [1, Period].
//   - positive, negative: mutually exclusive finite contributions of the pending transition.
//
// Returns:
//   - compensated, compensated: candidate positive/negative accumulators, or zero on error.
//   - error: ErrOverflow if a required Neumaier step cannot be represented.
//
// Errors:
//   - ErrOverflow: an intermediate candidate aggregate or correction overflows.
//
// Determinism:
//   - Starts after the overwritten head on a full ring, otherwise at slot zero; walks used slots.
//
// Complexity:
//   - Time O(Period), Space O(1), zero allocation.
//
// Notes:
//   - This path is O(Period) but is called only at scheduled or bounded extra rebase points.
//
// AI-Hints:
//   - Do not begin at physical index zero after wrap; logical age matters for replay parity.
func (s *Stream) rebaseVirtual(head, used int, positive, negative float64) (compensated, compensated, error) {
	var pos, neg compensated
	index := 0
	if used == s.period {
		// The replaced old head expires, so its successor becomes oldest.
		index = head + 1
		if index == s.period {
			index = 0
		}
	}

	for i := 0; i < used; i++ {
		posFlow, negFlow := s.pos[index], s.neg[index]
		if index == head {
			posFlow, negFlow = positive, negative
		}
		if err := pos.add(posFlow); err != nil {
			return compensated{}, compensated{}, err
		}
		if err := neg.add(negFlow); err != nil {
			return compensated{}, compensated{}, err
		}
		index++
		if index == s.period {
			index = 0
		}
	}
	return pos, neg, nil
}

// projected computes the post-transition sums and drift-epoch flag without commit.
// A scheduled rebase deliberately avoids the intermediate subtract/add path:
// a fully representable replacement window must not fail solely because the
// soon-to-be-discarded accumulator is close to float64's arithmetic limits.
//
// Implementation:
//   - Stage 1: determine occupancy; head==Period-1 marks the next epoch boundary.
//   - Stage 2: scheduled: replay candidate ring; otherwise evict then insert locally.
//   - Stage 3: check effective sums, allowing one negative-drift rebase per epoch.
//
// Behavior highlights:
//   - Scheduled replay skips a potentially overflowing intermediate eviction/addition.
//   - A second unscheduled negative drift within the same epoch returns ErrOverflow.
//
// Inputs:
//   - posFlow, negFlow: current transition contributions, each finite and nonnegative.
//
// Returns:
//   - Two compensated candidates and the epoch extra-rebase flag, or ErrOverflow.
//
// Errors:
//   - ErrOverflow: incremental/rebase arithmetic is not representable or drift limit is reached.
//
// Determinism:
//   - With Count>0, head==(Count-1)%Period, so head==Period-1 schedules every Period transitions.
//   - Eviction precedes insertion; scheduled replay walks oldest-to-newest.
//
// Complexity:
//   - Time O(1) amortized, O(Period) at a scheduled/extra rebase; Space O(1).
//
// Notes:
//   - Neither rings nor committed scalar state change during candidate projection.
//
// AI-Hints:
//   - A second unscheduled negative-drift event in one epoch returns ErrOverflow.
//   - Do not clamp negative sums or commit slots before this projection succeeds.
//   - Preserve head==(Count-1)%Period in Restore; the schedule relies on that invariant.
func (s *Stream) projected(posFlow, negFlow float64) (compensated, compensated, bool, error) {
	used := s.used
	if used < s.period {
		used++
	}

	// After the seed bar, head == (count-1) mod period. The next
	// transition completes a scheduled epoch exactly when head would wrap.
	// This cursor condition avoids unnecessary wide-integer division in Push.
	if s.head == s.period-1 {
		pos, neg, err := s.rebaseVirtual(s.head, used, posFlow, negFlow)
		return pos, neg, false, err
	}

	pos, neg := s.positive, s.negative
	if s.used == s.period {
		if err := pos.add(-s.pos[s.head]); err != nil {
			return compensated{}, compensated{}, false, err
		}
		if err := neg.add(-s.neg[s.head]); err != nil {
			return compensated{}, compensated{}, false, err
		}
	}
	if err := pos.add(posFlow); err != nil {
		return compensated{}, compensated{}, false, err
	}
	if err := neg.add(negFlow); err != nil {
		return compensated{}, compensated{}, false, err
	}

	posEffective, err := pos.value()
	if err != nil {
		return compensated{}, compensated{}, false, err
	}
	negEffective, err := neg.value()
	if err != nil {
		return compensated{}, compensated{}, false, err
	}
	if posEffective < 0 || negEffective < 0 {
		if s.extraRebaseUsed {
			return compensated{}, compensated{}, false, ErrOverflow
		}
		// One honest reconstruction is allowed; a negative replay result remains an error.
		pos, neg, err := s.rebaseVirtual(s.head, used, posFlow, negFlow)
		return pos, neg, true, err
	}
	return pos, neg, s.extraRebaseUsed, nil
}
