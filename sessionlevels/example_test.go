// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package sessionlevels_test demonstrates market-data engineering workflows over
// explicitly segmented historical bars. Prices are reproducible engineered case studies,
// not independently audited exchange records or simulated executable orders.
package sessionlevels_test

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/lvlath/go/sessionlevels"
)

// ExampleComputeDaily_nyseOpeningRiskGate demonstrates a prior-day reference for an
// exchange-local regular-hours opening risk monitor using finalized minute candles.
//
// Scenario:
//   - An upstream market-data adapter supplies 390 prior-session minute bars and
//     three finalized minutes of an incomplete next regular session.
//   - A risk feature must distinguish the first close above prior day high from
//     a later close that remains above it; the reference cannot move intraday.
//   - The adapter, not this package, handles NYSE holidays, auctions and half-days.
//
// Process:
//   - ComputeDaily consumes the adapter's exact half-open session manifest.
//   - Validate readiness before inspecting PDH, PDL and last prior close (PDC).
//   - Compare finalized closes with the fixed threshold; report an observation,
//     not an assumed intrabar fill or profitable execution.
//
// Complexity:
//   - O(n+s) time, O(n) output storage, O(1) auxiliary state.
func ExampleComputeDaily_nyseOpeningRiskGate() {
	f := dailyFixture()
	r, err := sessionlevels.ComputeDaily(context.Background(), f.high, f.low, f.closePrices, f.sessions)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	start := f.sessions[1].Start
	fmt.Printf("reference PDH=%.0f PDL=%.0f PDC=%.0f ready=%t\n",
		r.PreviousHigh[start], r.PreviousLow[start], r.PreviousClose[start], r.Valid[start])
	for i := start; i < len(f.closePrices); i++ {
		crossed := r.Valid[i] && f.closePrices[i] > r.PreviousHigh[i] &&
			f.closePrices[i-1] <= r.PreviousHigh[i]
		fmt.Printf("minute=%d close=%.0f PDH=%.0f first-cross=%t\n",
			i-start, f.closePrices[i], r.PreviousHigh[i], crossed)
	}
	// Output:
	// reference PDH=103 PDL=97 PDC=101 ready=true
	// minute=0 close=102 PDH=103 first-cross=false
	// minute=1 close=104 PDH=103 first-cross=true
	// minute=2 close=105 PDH=103 first-cross=false
}

// ExampleComputeWeekly_holidayAdjustedTreasuryWeek demonstrates variable-length
// macro-market trading weeks represented by five daily bars per normal week.
//
// Scenario:
//   - A rates-risk dashboard receives four trading days in the first week, five
//     in the next, and a two-day live cut of the current week.
//   - Holidays and special trading schedules are already resolved upstream.
//   - Comparing the current week to its own developing extremes would leak data.
//
// Process:
//   - ComputeWeekly uses three explicit week intervals rather than seven-day bins.
//   - Read the completed first-week range on the second week and the completed
//     second-week range on every available bar in the final incomplete week.
//
// Complexity:
//   - O(n+s) time and O(n) output storage; no calendar or quote dependencies.
func ExampleComputeWeekly_holidayAdjustedTreasuryWeek() {
	f := weeklyFixture()
	r, err := sessionlevels.ComputeWeekly(context.Background(), f.high, f.low, f.closePrices, f.sessions)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("week-2 prior-high=%.0f prior-low=%.0f prior-close=%.0f\n",
		r.PreviousHigh[4], r.PreviousLow[4], r.PreviousClose[4])
	fmt.Printf("week-3 live-bars=%d complete=%t prior-high=%.0f prior-low=%.0f prior-close=%.0f\n",
		f.sessions[2].End-f.sessions[2].Start, f.sessions[2].Complete,
		r.PreviousHigh[10], r.PreviousLow[10], r.PreviousClose[10])
	// Output:
	// week-2 prior-high=115 prior-low=100 prior-close=113
	// week-3 live-bars=2 complete=false prior-high=121 prior-low=110 prior-close=120
}

// ExampleComputeDefault_liveReplayAudit demonstrates a causal market-data replay
// where an incomplete session grows by one finalized bar at a time.
//
// Scenario:
//   - The replay runner repeatedly cuts an immutable input at the observation
//     boundary; the final Session.End is clipped and Complete remains false.
//   - A very high late trade can change a developing session's own high, but
//     cannot change the published reference from the preceding completed day.
//
// Process:
//   - Rebuild a well-formed manifest for each cut and invoke ComputeDefault.
//   - Mutate a future high while preserving candle geometry, then verify the
//     entire current session's result remains bit-identical to its earlier version.
//
// Complexity:
//   - Each independent batch costs O(n+s) time and O(n) returned storage.
func ExampleComputeDefault_liveReplayAudit() {
	full := dailyFixture()
	for end := 391; end <= 393; end++ {
		f := cutFixture(full, end)
		r, err := sessionlevels.ComputeDefault(context.Background(), f.high, f.low, f.closePrices, f.sessions)
		if err != nil {
			fmt.Println("error:", err)
			return
		}
		last := end - 1
		fmt.Printf("cut=%d complete=%t stable-PDH=%.0f stable-PDC=%.0f\n",
			end, f.sessions[len(f.sessions)-1].Complete, r.PreviousHigh[last], r.PreviousClose[last])
	}
	f := dailyFixture()
	base, err := sessionlevels.ComputeDefault(context.Background(), f.high, f.low, f.closePrices, f.sessions)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	f.high[len(f.high)-1] = 1000
	changed, err := sessionlevels.ComputeDefault(context.Background(), f.high, f.low, f.closePrices, f.sessions)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	stable := true
	for i := 390; i < len(f.high); i++ {
		stable = stable && math.Float64bits(base.PreviousHigh[i]) == math.Float64bits(changed.PreviousHigh[i]) &&
			math.Float64bits(base.PreviousLow[i]) == math.Float64bits(changed.PreviousLow[i]) &&
			math.Float64bits(base.PreviousClose[i]) == math.Float64bits(changed.PreviousClose[i]) &&
			base.Valid[i] == changed.Valid[i]
	}
	fmt.Println("future-high-mutation-invariant:", stable)
	// Output:
	// cut=391 complete=false stable-PDH=103 stable-PDC=101
	// cut=392 complete=false stable-PDH=103 stable-PDC=101
	// cut=393 complete=false stable-PDH=103 stable-PDC=101
	// future-high-mutation-invariant: true
}

// ExampleCompute_signedCalendarOvernightSpread demonstrates a futures spread that
// legitimately trades below zero across an exchange-local overnight session.
//
// Scenario:
//   - Synchronized spread OHLC observations are signed: negative prices are
//     ordinary finite numbers, not missing-data markers.
//   - An upstream exchange calendar has already grouped the overnight bars;
//     a UTC-midnight split would be an unrelated and potentially wrong policy.
//   - A risk service needs the prior spread high, low and exact final close.
//
// Process:
//   - Keep the negative bars unmodified and pass explicit index sessions.
//   - Use Valid rather than sign or zero to determine whether levels are ready.
//
// Complexity:
//   - O(n+s) time and O(n) output memory, with no price arithmetic.
func ExampleCompute_signedCalendarOvernightSpread() {
	f := signedFixture()
	r, err := sessionlevels.Compute(context.Background(), f.high, f.low, f.closePrices, f.sessions, sessionlevels.Options{})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("spread prior-high=%.0f prior-low=%.0f final-close=%.0f ready=%t\n",
		r.PreviousHigh[2], r.PreviousLow[2], r.PreviousClose[2], r.Valid[2])
	// Output:
	// spread prior-high=-2 prior-low=-10 final-close=-4 ready=true
}

// ExampleCompute_feedIntegrityGate demonstrates a hard admission boundary in a
// risk-computation pipeline before any previous-session levels are released.
//
// Scenario:
//   - A damaged market-data manifest overlaps two sessions, while a later
//     closing-price observation contains NaN from an upstream feed failure.
//   - The structural error must be rejected before price processing. Once the
//     manifest is repaired, non-finiteness has precedence over an earlier
//     finite candle-domain failure; no partial Result can leak downstream.
//
// Process:
//   - Identify failure classes with errors.Is rather than diagnostic strings.
//   - Keep the entire successful result isolated from any rejected attempt.
//
// Complexity:
//   - Rejection takes O(s) for structural errors, O(n+s) for numeric errors;
//     no output buffers are published on either failure.
func ExampleCompute_feedIntegrityGate() {
	f := goldenFixture()
	f.sessions[1].Start = 1
	result, err := sessionlevels.ComputeDefault(context.Background(), f.high, f.low, f.closePrices, f.sessions)
	fmt.Printf("overlap rejected=%t no-result=%t\n", errors.Is(err, sessionlevels.ErrSession), result == nil)
	f.sessions[1].Start = 2
	f.closePrices[0] = 10 // An invalid finite candle is remembered, not returned early.
	f.closePrices[3] = math.NaN()
	result, err = sessionlevels.ComputeDefault(context.Background(), f.high, f.low, f.closePrices, f.sessions)
	fmt.Printf("nonfinite prioritized=%t no-result=%t\n", errors.Is(err, sessionlevels.ErrNonFinite), result == nil)
	// Output:
	// overlap rejected=true no-result=true
	// nonfinite prioritized=true no-result=true
}
