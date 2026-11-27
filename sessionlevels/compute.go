// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package sessionlevels provides previous completed session High/Low/Close levels without look-ahead bias.
// This file implements the causal scalar-state kernel and detached output allocation.
package sessionlevels

import "context"

// compute performs the single canonical causal previous-session aggregation.
// Its precondition is a complete admission pass over stable prices and sessions.
// Each output allocation uses an existing slice length; independent buffers avoid
// multiplied packed-allocation arithmetic and accidental cross-channel aliasing.
// An unready row remains the initial positive zero and false mask.
// Time O(n+s), output space O(n), auxiliary scalar space O(1).
func compute(ctx context.Context, high, low, closePrices []float64, sessions []Session) (*Result, error) {
	n := len(high)
	result := &Result{
		PreviousHigh:  make([]float64, n),
		PreviousLow:   make([]float64, n),
		PreviousClose: make([]float64, n),
		Valid:         make([]bool, n),
	}

	// The saved triple belongs exclusively to a session strictly before the one
	// currently being written. hasPrevious makes warmup independent of numeric zero.
	var previousHigh, previousLow, previousClose float64
	hasPrevious := false
	for _, session := range sessions {
		// Seed from an observed value, never zero: all-negative prices are valid.
		currentHigh, currentLow := high[session.Start], low[session.Start]
		for i := session.Start; i < session.End; i++ {
			if i&(contextPollInterval-1) == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			// Read only the preceding committed summary for the entire session.
			if hasPrevious {
				result.PreviousHigh[i] = previousHigh
				result.PreviousLow[i] = previousLow
				result.PreviousClose[i] = previousClose
				result.Valid[i] = true
			}
			// Strict comparisons retain the earliest equal extremum's bits.
			// Current-session observations affect only the private candidate, not
			// previously written output; strict ties preserve earliest-value bits.
			if high[i] > currentHigh {
				currentHigh = high[i]
			}
			if low[i] < currentLow {
				currentLow = low[i]
			}
		}
		// The private candidate cannot affect rows already written above.
		// An incomplete final candidate is discarded, never committed.
		if session.Complete {
			previousHigh, previousLow = currentHigh, currentLow
			previousClose = closePrices[session.End-1]
			hasPrevious = true
		}
	}
	return result, nil
}
