// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package sessionlevels_test exercises only the public sessionlevels facade.
// This file defines the independent direct-range oracle and medium differential cases.
package sessionlevels_test

import (
	"testing"

	"github.com/lvlath/go/sessionlevels"
)

// oracle expects an admitted fixture and independently re-extracts the prior range
// for every output row. It shares neither state transitions nor helpers with production.
// Reverse traversal with non-strict comparisons selects the earliest equal extremum.
// Time is O(n*s + sum(m[j]*m[j-1])), hence O(n^2) worst case, not generally O(n*s).
func oracle(f fixture) *sessionlevels.Result {
	n := len(f.high)
	out := &sessionlevels.Result{
		PreviousHigh: make([]float64, n), PreviousLow: make([]float64, n),
		PreviousClose: make([]float64, n), Valid: make([]bool, n),
	}
	for i := 0; i < n; i++ {
		for k, session := range f.sessions {
			if i < session.Start || i >= session.End {
				continue
			}
			if k == 0 {
				break
			}
			previous := f.sessions[k-1]
			h, l := f.high[previous.End-1], f.low[previous.End-1]
			for j := previous.End - 1; j >= previous.Start; j-- {
				if f.high[j] >= h {
					h = f.high[j]
				}
				if f.low[j] <= l {
					l = f.low[j]
				}
			}
			out.PreviousHigh[i], out.PreviousLow[i] = h, l
			out.PreviousClose[i], out.Valid[i] = f.closePrices[previous.End-1], true
			break
		}
	}
	return out
}

func TestSessionLevels_IndependentOracle(t *testing.T) {
	for seed := int64(1); seed <= 100; seed++ {
		f := generatedFixture(seed, 1+int(seed)%79)
		requireResult(t, mustCompute(t, f), oracle(f))
	}
}

func TestSessionLevels_CalendarIndependentMedium(t *testing.T) {
	for _, f := range []fixture{dailyFixture(), weeklyFixture(), generatedFixture(11, 257)} {
		requireResult(t, mustCompute(t, f), oracle(f))
	}
}
