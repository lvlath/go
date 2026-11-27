// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package sessionlevels provides previous completed session High/Low/Close levels without look-ahead bias.
// This file validates shape, partition geometry, finiteness, and candle domains.
package sessionlevels

import (
	"context"
	"fmt"
	"math"
)

// contextPollInterval limits the bar/descriptor iterations between context polls.
// A power of two allows a zero-allocation bit-mask check in all linear scans.
// This is a per-scan bound, not a limit on runtime allocation or callback latency.
const contextPollInterval = 1024

// validateShape admits only three present and equally sized price channels.
// It deliberately classifies nil before shape to preserve the public error order;
// present zero-length channels remain valid for the empty-input law.
// Time O(1), auxiliary space O(1), with no allocation on success.
func validateShape(high, low, closePrices []float64) error {
	if high == nil || low == nil || closePrices == nil {
		return ErrNilInput
	}
	if len(high) != len(low) || len(high) != len(closePrices) {
		return ErrShape
	}
	return nil
}

// validateSessions proves that sessions partition [0,n) without gaps or overlaps.
// Every index is checked before the kernel can read it or subtract one from End.
// The expected next Start is exactly the preceding End; nonempty intervals
// therefore imply strict monotonicity without sorting or supplemental storage.
// A nonfinal incomplete interval fails before any price is scanned.
// Time O(s), auxiliary space O(1); successful validation allocates nothing.
func validateSessions(ctx context.Context, sessions []Session, n int) error {
	if n == 0 {
		if len(sessions) != 0 {
			return ErrSession
		}
		return nil
	}
	if len(sessions) == 0 {
		return ErrSession
	}

	// At every iteration nextStart is the only admissible inclusive boundary.
	nextStart := 0
	last := len(sessions) - 1
	for k, session := range sessions {
		if k&(contextPollInterval-1) == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if session.Start < 0 || session.Start != nextStart ||
			session.End <= session.Start || session.End > n {
			return fmt.Errorf("%w: interval %d has invalid bounds or adjacency", ErrSession, k)
		}
		// Only the rightmost session can be a partial live/replay cut.
		if k < last && !session.Complete {
			return fmt.Errorf("%w: interval %d is incomplete before the last interval", ErrSession, k)
		}
		nextStart = session.End
	}
	if nextStart != n {
		return fmt.Errorf("%w: final boundary %d does not equal input length %d", ErrSession, nextStart, n)
	}
	return nil
}

// validatePrices checks finiteness and candle ordering in one increasing-index scan.
// Every row is examined, including first-session warmup and incomplete sessions.
// It saves the first finite geometry failure, but keeps scanning: a later
// NaN/Inf has global precedence without requiring a second traversal.
// Time O(n), auxiliary space O(1); only rejection diagnostics allocate.
func validatePrices(ctx context.Context, high, low, closePrices []float64) error {
	// A negative index means no finite candle-domain failure has been seen.
	badBar := -1
	for i, h := range high {
		if i&(contextPollInterval-1) == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		l, c := low[i], closePrices[i]
		if !isFinite(h) || !isFinite(l) || !isFinite(c) {
			return fmt.Errorf("%w: bar %d", ErrNonFinite, i)
		}
		// Capture the earliest bad finite candle, never an arbitrary later one.
		if badBar < 0 && (l > h || c < l || c > h) {
			badBar = i
		}
	}
	if badBar >= 0 {
		return fmt.Errorf("%w: bar %d", ErrBar, badBar)
	}
	return nil
}

// isFinite classifies one binary64 value without changing either zero sign.
// Comparisons in the kernel are safe only after this predicate admitted every input.
func isFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}
