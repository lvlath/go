// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package vwap_test checks the mathematics underlying every executable scenario.
// This file supplements printed Output checks with bitwise independent-oracle comparisons.
// AI-Hints: a plausible formatted value is not enough; keep exact prefix parity assertions.
package vwap_test

import (
	"context"
	"testing"

	"github.com/lvlath/go/vwap"
)

// TestExamples_VendorAndExecutionOracles checks every prefix of the two distinct price meanings.
// Source-demo closes remain a bar proxy, while the engineered trade tapes model individual executions.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: build the fixed ordered vendor and separate market/fill histories.
//   - Stage 3: compare all prefixes with independent rational arithmetic.
//
// Behavior highlights:
//   - This test checks sourced-bar and engineered-execution scenario mathematics without altering production visibility.
//
// Inputs:
//   - The Go testing harness and fixed/procedural cases declared in code.
//
// Returns:
//   - A test or fuzz verdict; no public package state or external data is modified.
//
// Errors:
//   - Unexpected public errors, sentinel classes, masks, or output bits fail the owning test.
//
// Determinism:
//   - Case order, fixed seeds, and logical cancellation are explicit; fuzz inputs select reproducible cases.
//
// Complexity:
//   - The sum of the exercised public calls and independent oracle work; no performance claim is inferred from test duration.
//
// Notes:
//   - Runtime allocation counters and race instrumentation are checked separately from numerical equality.
//
// AI-Hints:
//   - Formatted sample output is insufficient without exact numerical assertions.
func TestExamples_VendorAndExecutionOracles(t *testing.T) {
	bars := vendorIBMClosingHour()
	price, volume, reset := make([]float64, len(bars)), make([]float64, len(bars)), make([]bool, len(bars))
	var total float64
	for i, bar := range bars {
		price[i], volume[i], reset[i] = bar.price, bar.volume, i == 0
		total += bar.volume
		if i > 0 && bar.label <= bars[i-1].label {
			t.Fatal("frozen source labels must remain strictly ascending")
		}
	}
	if len(bars) != 12 || total != 1085345 {
		t.Fatal("frozen vendor sample was silently changed")
	}
	verifyParity(t, price, volume, reset)
	market, fills := executionFixture()
	verifyParity(t, market.price, market.volume, market.reset)
	verifyParity(t, fills.price, fills.volume, fills.reset)
}

// TestExamples_AvailableAtAndMatchedSpreadOracles independently fixes the intended reset schedule.
// The delayed-news fixture must reset at row five, not retroactively at the event-time row two.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: construct the explicit available-at reset and paired-price input histories.
//   - Stage 3: require correct masked anchor transitions and independent prefix parity.
//
// Behavior highlights:
//   - This test checks causal event admission and synchronized-pair weighting scenarios without altering production visibility.
//
// Inputs:
//   - The Go testing harness and fixed/procedural cases declared in code.
//
// Returns:
//   - A test or fuzz verdict; no public package state or external data is modified.
//
// Errors:
//   - Unexpected public errors, sentinel classes, masks, or output bits fail the owning test.
//
// Determinism:
//   - Case order, fixed seeds, and logical cancellation are explicit; fuzz inputs select reproducible cases.
//
// Complexity:
//   - The sum of the exercised public calls and independent oracle work; no performance claim is inferred from test duration.
//
// Notes:
//   - Runtime allocation counters and race instrumentation are checked separately from numerical equality.
//
// AI-Hints:
//   - Do not substitute event time for information delivery or unmatched leg volumes for paired size.
func TestExamples_AvailableAtAndMatchedSpreadOracles(t *testing.T) {
	_, rows := newsAvailabilityFixture()
	price, volume, reset := make([]float64, len(rows)), make([]float64, len(rows)), make([]bool, len(rows))
	for i, row := range rows {
		price[i], volume[i] = row.price, row.volume
	}
	reset[0], reset[5] = true, true
	verifyParity(t, price, volume, reset)
	result := mustCompute(t, price, volume, reset)
	if result.Valid[5] || !result.Valid[6] {
		t.Fatal("zero-volume available-at anchor must erase old readiness and then warm up")
	}
	pairs := matchedSpreadFixture()
	for i, row := range pairs {
		price[i], volume[i], reset[i] = row.near-row.far, row.matched, i == 0
	}
	verifyParity(t, price, volume, reset)
}

// TestExamples_ReplayCursorBoundary confirms the full source tape is a legitimate canonical history.
// It also proves a failed reset at the same retry boundary leaves exact continuation intact.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: replay the engineered source records with a rejected reset at a fixed boundary.
//   - Stage 3: require rollback and exact oracle continuation after retry.
//
// Behavior highlights:
//   - This test checks the numerical tape behind failover and retry examples without altering production visibility.
//
// Inputs:
//   - The Go testing harness and fixed/procedural cases declared in code.
//
// Returns:
//   - A test or fuzz verdict; no public package state or external data is modified.
//
// Errors:
//   - Unexpected public errors, sentinel classes, masks, or output bits fail the owning test.
//
// Determinism:
//   - Case order, fixed seeds, and logical cancellation are explicit; fuzz inputs select reproducible cases.
//
// Complexity:
//   - The sum of the exercised public calls and independent oracle work; no performance claim is inferred from test duration.
//
// Notes:
//   - Runtime allocation counters and race instrumentation are checked separately from numerical equality.
//
// AI-Hints:
//   - A source cursor belongs in the application envelope, not in segment Count.
func TestExamples_ReplayCursorBoundary(t *testing.T) {
	rows := replayFixture()
	price, volume, reset := make([]float64, len(rows)), make([]float64, len(rows)), make([]bool, len(rows))
	for i, row := range rows {
		price[i], volume[i], reset[i] = row.price, row.volume, row.reset
	}
	verifyParity(t, price, volume, reset)
	stream := newStream(t)
	for _, row := range rows[:7] {
		mustPush(t, stream, row.price, row.volume, row.reset)
	}
	before := stream.Snapshot()
	value, err := stream.Push(context.Background(), rows[7].price, -1, true)
	requireError(t, err, vwap.ErrVolume)
	requireValue(t, value, vwap.Value{})
	requireUnchanged(t, stream, before)
	want := rationalOracle(price, volume, reset)
	for i := 7; i < len(rows); i++ {
		requireValue(t, mustPush(t, stream, price[i], volume[i], reset[i]), want[i])
	}
}
