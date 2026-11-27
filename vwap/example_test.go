// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package vwap_test demonstrates complete operational pipelines with explicit observation meaning.
// This file uses frozen vendor-demo bars and separately labeled engineered execution/recovery tapes.
// AI-Hints: no network, current clock, exchange calendar, live order, or profit promise is involved.
package vwap_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"

	"github.com/lvlath/go/vwap"
)

// ExampleComputeDefault_vendorBarBenchmark audits a fixed closing-hour bar-price benchmark.
// An analyst compares the close-volume prefix with each finalized close, without claiming tick VWAP.
//
// Implementation:
//   - Stage 1: select twelve frozen IBM provider-demo bars, ascending by their source label.
//   - Stage 2: declare a 15:00 anchor and compute exact weighting of the supplied closes.
//   - Stage 3: report the interval, observed volume, readiness, and bar-proxy limitation.
//
// Behavior highlights:
//   - The 15:00 anchor is an explicit reporting-window choice, not a discovered price extremum.
//
// Inputs:
//   - vendorIBMClosingHour: 2026-08-07 US/Eastern demo bars, with provenance in the helper.
//
// Returns:
//   - A deterministic closing-hour benchmark in USD/share for this archived bar subset.
//
// Errors:
//   - Calculation failures are handled before any output is consumed.
//
// Determinism:
//   - Fixed literals and chronological order; no fresh provider query during go test.
//
// Complexity:
//   - Time O(n*L), Space O(n+L), n=12, L=67, including channel preparation.
//
// Notes:
//   - The adapter has already admitted finalized historical bars; bar availability is not inferred here.
//
// AI-Hints:
//   - Keep vendor demo provenance visible; these are not certified exchange tick executions.
//   - A close-volume or HLC3-volume series cannot reconstruct missing within-bar trade prices.
func ExampleComputeDefault_vendorBarBenchmark() {
	// Build caller-owned channels; choosing Close instead of HLC3 is a visible policy.
	bars := vendorIBMClosingHour()
	price, volume, reset := make([]float64, len(bars)), make([]float64, len(bars)), make([]bool, len(bars))
	var observedVolume float64
	for i, bar := range bars {
		price[i], volume[i], reset[i] = bar.price, bar.volume, i == 0
		observedVolume += bar.volume // The fixed integer share totals are exactly representable.
	}
	result, err := vwap.ComputeDefault(context.Background(), price, volume, reset)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	// Readiness is a separate channel even though every bar in this fixture has positive volume.
	last := len(bars) - 1
	fmt.Printf("IBM 2026-08-07 %s..%s US/Eastern: bars=%d shares=%.0f\n", bars[0].label, bars[last].label, len(bars), observedVolume)
	fmt.Printf("close-volume benchmark=%.6f ready=%t last-close=%.4f\n", result.Values[last], result.Valid[last], price[last])
	fmt.Println("basis=vendor-demo bar proxy; tick VWAP is not available")

	// Output:
	// IBM 2026-08-07 15:00..15:55 US/Eastern: bars=12 shares=1085345
	// close-volume benchmark=236.873615 ready=true last-close=237.2100
	// basis=vendor-demo bar proxy; tick VWAP is not available
}

// ExampleCompute_executionQuality compares a buy-fill tape with its contemporaneous market tape.
// This engineered execution audit keeps the benchmark window, currency, symbol, and units aligned.
//
// Implementation:
//   - Stage 1: prepare the complete market window and separate child-order fills.
//   - Stage 2: calculate each tape's VWAP with the same explicit first-anchor convention.
//   - Stage 3: report a buy-side adverse-price difference and basis points against a positive benchmark.
//
// Behavior highlights:
//   - Market trade volume and executed order quantity never share an accumulator.
//
// Inputs:
//   - Twelve market observations and six fills in a fixed synthetic one-minute test-exchange window.
//
// Returns:
//   - A descriptive execution-price comparison, not implementation shortfall or net P&L.
//
// Errors:
//   - Both algorithm errors and unavailable/nonpositive benchmark results are checked.
//
// Determinism:
//   - Fixed ordered trade tapes with no current prices, routing, or trading actions.
//
// Complexity:
//   - Time O((n+m)*L), Space O(n+m+L), L=67; only final means are compared.
//
// Notes:
//   - Fees, spread crossing, market impact attribution, and unexecuted quantity are not estimated.
//
// AI-Hints:
//   - A rounded-mean difference is an application metric, not a new exact-sum field of Result.
//   - Do not divide by a signed/zero spread benchmark or compare mismatched trading windows.
func ExampleCompute_executionQuality() {
	market, fills := executionFixture()
	ctx := context.Background()
	// Compute the market benchmark and the order's realized mean independently.
	benchmark, err := vwap.Compute(ctx, market.price, market.volume, market.reset, vwap.Options{})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	execution, err := vwap.ComputeDefault(ctx, fills.price, fills.volume, fills.reset)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	marketLast, fillLast := len(market.price)-1, len(fills.price)-1
	if !benchmark.Valid[marketLast] || !execution.Valid[fillLast] || benchmark.Values[marketLast] <= 0 {
		fmt.Println("benchmark unavailable for positive-price basis points")
		return
	}
	// For a buy, a positive difference is adverse; fees and causality attribution stay outside VWAP.
	marketMean, fillMean := benchmark.Values[marketLast], execution.Values[fillLast]
	const basisPoints = 10000.0
	adverse := fillMean - marketMean
	var filled float64
	for _, quantity := range fills.volume {
		filled += quantity
	}
	fmt.Printf("test-exchange window=10:00:00..10:01:00 market-trades=%d fills=%d shares=%.0f\n", len(market.price), len(fills.price), filled)
	fmt.Printf("market=%.6f buy-fill=%.6f adverse=%.6f USD/share\n", marketMean, fillMean, adverse)
	fmt.Printf("buy adverse-price difference=%.3f bps; fees excluded\n", adverse/marketMean*basisPoints)

	// Output:
	// test-exchange window=10:00:00..10:01:00 market-trades=12 fills=6 shares=800
	// market=100.209016 buy-fill=100.215625 adverse=0.006609 USD/share
	// buy adverse-price difference=0.659 bps; fees excluded
}

// ExampleStream_Push_availableAtAnchor starts a news-relative benchmark when the news is observable.
// A delayed application event cannot retroactively reset already published pre-delivery prefixes.
//
// Implementation:
//   - Stage 1: receive an engineered event with a separate available-at timestamp.
//   - Stage 2: request reset on the first admitted row at/after that timestamp, committing the latch only on success.
//   - Stage 3: retain a masked zero-volume anchor until the first subsequent positive weight.
//
// Behavior highlights:
//   - Available-at anchoring is not an event-time retrospective AVWAP or a confirmed-pivot backfill.
//
// Inputs:
//   - Twelve engineered observations in admission order; no actual central-bank announcement is asserted.
//
// Returns:
//   - Observable reset points and the final post-delivery VWAP/segment count.
//
// Errors:
//   - A failed Push stops the example before the anchor latch or downstream value is published.
//
// Determinism:
//   - Fixed UTC timestamps; no timers, sleeps, current clock, or calendar lookups.
//
// Complexity:
//   - Time O(n*L), Space O(L) processor state excluding the explicit demonstration tape.
//
// Notes:
//   - The application, not vwap, determines availability and observation admission order.
//
// AI-Hints:
//   - Never set the latch before a successful reset row: rejection must leave the reset pending.
//   - Do not silently use event time when the signal was delivered later.
func ExampleStream_Push_availableAtAnchor() {
	availableAt, rows := newsAvailabilityFixture()
	stream, err := vwap.NewDefault()
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	anchored := false
	var current vwap.Value
	for i, row := range rows {
		// This decision uses only the event available now and the current admitted observation.
		newsReset := !anchored && !row.availableAt.Before(availableAt)
		reset := i == 0 || newsReset
		current, err = stream.Push(context.Background(), row.price, row.volume, reset)
		if err != nil {
			fmt.Println("error:", err)
			return
		}
		// Only a successful accepted reset consumes the application's pending anchor.
		if newsReset {
			anchored = true
			fmt.Printf("available-at anchor=%s ready=%t\n", row.availableAt.Format("15:04:05.000"), current.Valid)
		}
	}
	fmt.Printf("post-delivery rows=%d VWAP=%.6f ready=%t\n", stream.Snapshot().Count, current.Value, current.Valid)
	fmt.Println("earlier published observations were not rewritten")

	// Output:
	// available-at anchor=14:00:01.200 ready=false
	// post-delivery rows=7 VWAP=100.777500 ready=true
	// earlier published observations were not rewritten
}

// ExampleStream_Restore_failoverCursor demonstrates exact continuation and duplicate admission control.
// The source cursor is persisted alongside Snapshot because Count is not a transport sequence number.
//
// Implementation:
//   - Stage 1: checkpoint after two zero-volume observations and round-trip an application envelope through JSON.
//   - Stage 2: restore a fresh processor, skip an already committed source sequence, and reject a corrupt reset atomically.
//   - Stage 3: accept the valid retry and compare final exact state against uninterrupted replay.
//
// Behavior highlights:
//   - The rejected reset does not lose the active segment or advance the external cursor.
//
// Inputs:
//   - A twelve-record engineered source tape plus one duplicate and one deliberately corrupt reset attempt.
//
// Returns:
//   - Restart readiness, duplicate/rejection counts, and exact continuation equivalence.
//
// Errors:
//   - Construction, JSON, Restore, and Push errors are all checked; ErrNonFinite is classified with errors.Is.
//
// Determinism:
//   - Fixed sequence ordering and complete Snapshot equality; no wall-clock race is used.
//
// Complexity:
//   - Time O(n*L+D*L), Space O(L+D), excluding the fixed input fixture; D<=1300.
//
// Notes:
//   - The in-memory JSON round trip models persistence; durable atomic storage and authentication remain external.
//
// AI-Hints:
//   - Commit cursor and checkpoint together in storage; do not advance the cursor after a failed Push.
//   - Exactly-once delivery is not a property of Snapshot alone, and Count resets between segments.
func ExampleStream_Restore_failoverCursor() {
	rows := replayFixture()
	stream, err := vwap.NewDefault()
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	const checkpointRows = 2
	var current vwap.Value
	for _, row := range rows[:checkpointRows] {
		current, err = stream.Push(context.Background(), row.price, row.volume, row.reset)
		if err != nil {
			fmt.Println("error:", err)
			return
		}
	}
	checkpoint := restartEnvelope{State: stream.Snapshot(), LastSequence: rows[checkpointRows-1].sequence}
	encoded, err := json.Marshal(checkpoint)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	var loaded restartEnvelope
	if err := json.Unmarshal(encoded, &loaded); err != nil {
		fmt.Println("error:", err)
		return
	}
	resumed, err := vwap.NewDefault()
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	if err := resumed.Restore(loaded.State); err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("checkpoint cursor=%d rows=%d ready=%t\n", loaded.LastSequence, loaded.State.Count, current.Valid)
	duplicates, rejected := 0, 0
	const corruptSequence = 8
	// Start one observation before the saved cursor to model an at-least-once redelivery.
	for _, row := range rows[checkpointRows-1:] {
		if row.sequence <= loaded.LastSequence {
			duplicates++
			continue
		}
		if row.sequence == corruptSequence {
			before := resumed.Snapshot()
			_, rejectedErr := resumed.Push(context.Background(), math.NaN(), row.volume, true)
			if !errors.Is(rejectedErr, vwap.ErrNonFinite) || resumed.Snapshot() != before {
				fmt.Println("reset rollback invariant failed")
				return
			}
			rejected++
		}
		current, err = resumed.Push(context.Background(), row.price, row.volume, row.reset)
		if err != nil {
			fmt.Println("error:", err)
			return
		}
		// This assignment models the successful-admission boundary, not durable atomic storage.
		loaded.LastSequence = row.sequence
	}
	// Continue the original processor without duplicate/corruption to audit exact restart parity.
	for _, row := range rows[checkpointRows:] {
		if _, err := stream.Push(context.Background(), row.price, row.volume, row.reset); err != nil {
			fmt.Println("error:", err)
			return
		}
	}
	fmt.Printf("duplicates=%d rejected-resets=%d cursor=%d rows=%d\n", duplicates, rejected, loaded.LastSequence, resumed.Snapshot().Count)
	fmt.Printf("VWAP=%.6f exact-continuation=%t\n", current.Value, resumed.Snapshot() == stream.Snapshot())

	// Output:
	// checkpoint cursor=2 rows=2 ready=false
	// duplicates=1 rejected-resets=1 cursor=12 rows=12
	// VWAP=100.964286 exact-continuation=true
}

// ExampleComputeDefault_matchedSpreadBasis calculates a signed basis from synchronized paired executions.
// The weight is completed matched quantity, never an arbitrary sum of independent leg volumes.
//
// Implementation:
//   - Stage 1: derive each paired spread from two synchronized engineered leg prices.
//   - Stage 2: supply paired quantity and an explicit first anchor to the canonical default facade.
//   - Stage 3: consume the readiness mask and report the signed price-unit benchmark.
//
// Behavior highlights:
//   - Negative spreads are valid; the first zero-volume observation remains undefined.
//
// Inputs:
//   - Twelve engineered paired observations, all in one compatible price and matched-quantity unit.
//
// Returns:
//   - A signed spread basis and the count of defined observation prefixes.
//
// Errors:
//   - Nonfinite differences or invalid volumes would be rejected through the normal public facade.
//
// Determinism:
//   - Input order is fixed; all fixture leg differences are exactly representable integers.
//
// Complexity:
//   - Time O(n*L), Space O(n+L), L=67, including derived caller-owned channels.
//
// Notes:
//   - This is not the difference of independently weighted leg VWAPs or an execution-price forecast.
//
// AI-Hints:
//   - Do not use percentage/basis-point division when the benchmark can be zero or negative.
//   - Never substitute differences of unsynchronized leg highs/lows for paired spread observations.
func ExampleComputeDefault_matchedSpreadBasis() {
	rows := matchedSpreadFixture()
	price, volume, reset := make([]float64, len(rows)), make([]float64, len(rows)), make([]bool, len(rows))
	var matched float64
	for i, row := range rows {
		price[i], volume[i], reset[i] = row.near-row.far, row.matched, i == 0
		matched += row.matched
	}
	result, err := vwap.ComputeDefault(context.Background(), price, volume, reset)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	ready := 0
	for _, valid := range result.Valid {
		if valid {
			ready++
		}
	}
	fmt.Printf("paired observations=%d matched quantity=%.0f ready-prefixes=%d\n", len(rows), matched, ready)
	fmt.Printf("first-ready=%t final signed basis=%.6f price units\n", result.Valid[0], result.Values[len(rows)-1])
	fmt.Println("matched-pair weights; no percentage denominator")

	// Output:
	// paired observations=12 matched quantity=28 ready-prefixes=11
	// first-ready=false final signed basis=0.142857 price units
	// matched-pair weights; no percentage denominator
}
