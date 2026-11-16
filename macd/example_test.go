// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package macd_test presents five deterministic operational MACD pipelines.
// These offline price fixtures resemble exchange-bar/tick replays but are not live
// or historical exchange quotes; they never implement order execution or risk policy.
package macd_test

import (
	"errors"
	"fmt"
	"math"

	"github.com/lvlath/go/macd"
)

// ExampleComputeDefault_crossAssetScreen illustrates monitoring a small set of
// liquid-equity-style price histories without assuming all assets have equal momentum.
//
// Scenario:
//   - A nightly research job receives aligned closing bars for three instruments.
//   - It computes the canonical 12/26/9 feature on each bar history and reports
//     the final signed histogram as a descriptive momentum-change diagnostic.
//   - All fixture values are synthetic, reproducible exchange-like closes.
//
// Process:
//   - Generate complete positive bar series in the same observation cadence.
//   - Require the latest Signal and Histogram to be ready, not merely nonzero.
//   - Report the sign and readiness of the latest batch feature per instrument.
//
// AI-Hints:
//   - This is not a strategy or a comparable risk-adjusted cross-asset score;
//     histograms are expressed in each instrument's own price units.
func ExampleComputeDefault_crossAssetScreen() {
	instruments := []struct {
		name                    string
		base, linear, curvature float64
	}{
		{"infrastructure", 96.5, 0.28, 0.013},
		{"utilities", 54.3, -0.05, 0.002},
		{"semiconductors", 212.0, 0.45, -0.018},
	}
	for _, instrument := range instruments {
		// The fixed 56-bar replay makes the 34-bar default signal warmup explicit.
		closes := make([]float64, 56)
		for index := range closes {
			x := float64(index)
			closes[index] = instrument.base + instrument.linear*x + instrument.curvature*x*x + float64(index%5-2)*0.12
		}
		result, err := macd.ComputeDefault(closes)
		if err != nil {
			fmt.Println("screening error:", err)
			return
		}
		last := len(closes) - 1
		if !result.Histogram.Valid[last] || !result.Signal.Valid[last] {
			fmt.Println("history not ready:", instrument.name)
			return
		}
		direction := "falling"
		if result.Histogram.Values[last] >= 0 {
			direction = "rising"
		}
		fmt.Printf("%s: %d bars, momentum-change=%s, histogram=%+.3f\n",
			instrument.name, len(closes), direction, result.Histogram.Values[last])
	}

	// Output:
	// infrastructure: 56 bars, momentum-change=rising, histogram=+0.656
	// utilities: 56 bars, momentum-change=rising, histogram=+0.097
	// semiconductors: 56 bars, momentum-change=falling, histogram=-0.920
}

// ExampleStream_Push_intradayMomentumMonitor demonstrates streaming analytics over
// incoming mid-price events, reporting crossings only after signal warmup completes.
//
// Scenario:
//   - A market-data inspection service batches neither ticks nor rolling state.
//   - Each arriving positive mid-price advances a dedicated MACD(3,7,4) stream.
//   - A sign change of the histogram emits a RESEARCH alert, never an order.
//
// Process:
//   - Feed a price path with expansion, drawdown and recovery phases.
//   - Track valid histogram values and compare only consecutive ready points.
//   - Print the exact crossing index and sign of the resulting momentum spread.
//
// AI-Hints:
//   - MACD consumes observation order, not timestamps; a real feed must enforce
//     event-time ordering, missing-message policy and instrument isolation itself.
func ExampleStream_Push_intradayMomentumMonitor() {
	stream, err := macd.NewWithPeriods(3, 7, 4)
	if err != nil {
		fmt.Println("stream setup:", err)
		return
	}
	prices := []float64{
		102.04, 102.10, 102.16, 102.23, 102.29, 102.36, 102.42, 102.50,
		102.57, 102.65, 102.69, 102.63, 102.54, 102.42, 102.28, 102.12,
		101.95, 101.83, 101.74, 101.66, 101.60, 101.62, 101.70, 101.82,
		101.98, 102.15, 102.32, 102.46, 102.61, 102.72, 102.79, 102.75,
		102.64, 102.49, 102.31, 102.15, 101.97, 101.85, 101.77, 101.81,
	}
	ready := 0
	crossings := 0
	previous := 0.0
	wasReady := false
	for index, closeD := range prices {
		point, pushErr := stream.Push(closeD)
		if pushErr != nil {
			fmt.Println("tick rejected:", pushErr)
			return
		}
		if !point.SignalValid {
			continue
		}
		ready++
		if wasReady && ((previous < 0 && point.Histogram >= 0) || (previous > 0 && point.Histogram <= 0)) {
			crossings++
			fmt.Printf("momentum review at event=%d, histogram=%+.4f\n", index, point.Histogram)
		}
		wasReady = true
		previous = point.Histogram
	}
	fmt.Printf("processed=%d ready=%d crossings=%d\n", len(prices), ready, crossings)

	// Output:
	// momentum review at event=10, histogram=-0.0003
	// momentum review at event=21, histogram=+0.0140
	// momentum review at event=31, histogram=-0.0104
	// processed=40 ready=31 crossings=3
}

// ExampleStream_Restore_warmupFailover demonstrates a standby stream recovering
// in the signal SMA warmup phase without replaying full historical prices.
//
// Scenario:
//   - A stream processor checkpoints after 29 accepted exchange-style closes.
//   - The slow price EMA is ready (26), but the signal EMA is still warming (9).
//   - A standby restores the detached snapshot and continues on the next bar.
//
// Process:
//   - Feed an explicit, deterministic 60-bar history into the primary.
//   - Snapshot and restore exactly at a difficult mid-warmup boundary.
//   - Compare each continuation Point at the float64-bit level and report progress.
//
// AI-Hints:
//   - Restore verifies structure/configuration, not provenance; authenticate
//     persisted snapshots separately and resume at the correct next input offset.
func ExampleStream_Restore_warmupFailover() {
	closes := make([]float64, 60)
	for index := range closes {
		x := float64(index)
		closes[index] = 186.25 + 0.04*x + 0.0009*x*x + float64(index%7-3)*0.17
	}
	primary, err := macd.NewDefault()
	if err != nil {
		fmt.Println("primary setup:", err)
		return
	}
	const checkpointAt = 29
	for _, closeD := range closes[:checkpointAt] {
		if _, pushErr := primary.Push(closeD); pushErr != nil {
			fmt.Println("initial feed:", pushErr)
			return
		}
	}
	checkpoint := primary.Snapshot()
	standby, err := macd.NewDefault()
	if err != nil {
		fmt.Println("standby setup:", err)
		return
	}
	if err = standby.Restore(checkpoint); err != nil {
		fmt.Println("restore:", err)
		return
	}
	lineReady := checkpoint.IntState[1] == 26
	signalWarming := checkpoint.IntState[2] < 9
	matching := true
	for _, closeD := range closes[checkpointAt:] {
		leader, leaderErr := primary.Push(closeD)
		follower, followerErr := standby.Push(closeD)
		if leaderErr != nil || followerErr != nil {
			fmt.Println("continuation error")
			return
		}
		matching = matching && leader.LineValid == follower.LineValid &&
			leader.SignalValid == follower.SignalValid &&
			math.Float64bits(leader.Line) == math.Float64bits(follower.Line) &&
			math.Float64bits(leader.Signal) == math.Float64bits(follower.Signal) &&
			math.Float64bits(leader.Histogram) == math.Float64bits(follower.Histogram)
	}
	fmt.Printf("checkpoint=%d line-ready=%t signal-warming=%t replay-bitwise=%t final-count=%d\n",
		checkpoint.Count, lineReady, signalWarming, matching, standby.Snapshot().Count)

	// Output:
	// checkpoint=29 line-ready=true signal-warming=true replay-bitwise=true final-count=60
}

// ExampleCompute_featureReadinessGate demonstrates a leak-resistant feature
// extraction stage that preserves original observation indices during warmup.
//
// Scenario:
//   - A model-building pipeline needs MACD histogram features from 48 bars.
//   - Its response labels are aligned to the same bar indices.
//   - The pipeline must not misinterpret invalid warmup zeros as observations.
//
// Process:
//   - Compute default MACD over a complete, fixed-cadence positive history.
//   - Select only indices with both a ready histogram and an available next-bar label.
//   - Count usable samples while preserving each original index in the payload.
//
// AI-Hints:
//   - Training/validation splits must be chronological outside MACD; a
//     future-bar label is unavailable at live inference time.
func ExampleCompute_featureReadinessGate() {
	const totalBars = 48
	closes := make([]float64, totalBars)
	for index := range closes {
		closes[index] = 43.8 + 0.055*float64(index) + float64(index%6-2)*0.08
	}
	result, err := macd.Compute(closes, macd.Options{})
	if err != nil {
		fmt.Println("feature computation:", err)
		return
	}
	usable := 0
	positiveLabels := 0
	firstIndex := -1
	lastIndex := -1
	for index := 0; index+1 < len(closes); index++ {
		if !result.Histogram.Valid[index] {
			continue
		}
		// A next-close direction label is an offline training feature only.
		upNext := closes[index+1] > closes[index]
		if upNext {
			positiveLabels++
		}
		if firstIndex == -1 {
			firstIndex = index
		}
		lastIndex = index
		usable++
	}
	fmt.Printf("first histogram index=%d, aligned training rows=%d, positive next-bar labels=%d, last label index=%d\n",
		firstIndex, usable, positiveLabels, lastIndex)

	// Output:
	// first histogram index=33, aligned training rows=14, positive next-bar labels=12, last label index=46
}

// ExampleStream_Push_badMarketDataIsolation illustrates transaction-safe handling
// of corrupt event values without silently advancing the recurrence or phase count.
//
// Scenario:
//   - A market-data gateway receives a nonfinite price, zero, and negative value
//     during otherwise valid streaming ticks.
//   - It rejects each invalid event, records the category, and preserves EMA state.
//   - Its application-level incident workflow decides whether to halt or resync.
//
// Process:
//   - Seed a MACD(3,8,4) stream with accepted closes.
//   - Reject three distinct error classes (nonfinite and positive-domain).
//   - Replay an accepted tick against a clean reference stream to prove continuity.
//
// AI-Hints:
//   - Rejection is NOT automatic gap repair: in live trading the caller must
//     reconcile sequence numbers and timestamps before resuming any decisions.
func ExampleStream_Push_badMarketDataIsolation() {
	feed := []float64{58.2, 58.3, 58.1, 58.6, 58.9, 59.0, 58.7, 58.8, 59.1, 59.4}
	stream, err := macd.NewWithPeriods(3, 8, 4)
	if err != nil {
		fmt.Println("setup:", err)
		return
	}
	reference, err := macd.NewWithPeriods(3, 8, 4)
	if err != nil {
		fmt.Println("reference setup:", err)
		return
	}
	for _, price := range feed {
		if _, err = stream.Push(price); err != nil {
			fmt.Println("accepted feed error:", err)
			return
		}
		if _, err = reference.Push(price); err != nil {
			fmt.Println("reference feed error:", err)
			return
		}
	}
	before := stream.Snapshot()
	nonfinite, domain := 0, 0
	for _, corrupted := range []float64{math.NaN(), math.Inf(1), 0, -20.0} {
		_, pushErr := stream.Push(corrupted)
		if errors.Is(pushErr, macd.ErrNonFinite) {
			nonfinite++
		}
		if errors.Is(pushErr, macd.ErrDomain) {
			domain++
		}
	}
	after := stream.Snapshot()
	unchanged := before.Count == after.Count
	for index := range before.FloatState {
		unchanged = unchanged && math.Float64bits(before.FloatState[index]) == math.Float64bits(after.FloatState[index])
		unchanged = unchanged && before.IntState[index] == after.IntState[index]
	}
	actual, actualErr := stream.Push(59.7)
	want, wantErr := reference.Push(59.7)
	if actualErr != nil || wantErr != nil {
		fmt.Println("resume error")
		return
	}
	matching := actual.LineValid == want.LineValid && actual.SignalValid == want.SignalValid &&
		math.Float64bits(actual.Line) == math.Float64bits(want.Line) &&
		math.Float64bits(actual.Signal) == math.Float64bits(want.Signal) &&
		math.Float64bits(actual.Histogram) == math.Float64bits(want.Histogram)
	fmt.Printf("rejected nonfinite=%d domain=%d unchanged=%t resumed-bitwise=%t count=%d\n",
		nonfinite, domain, unchanged, matching, stream.Snapshot().Count)

	// Output:
	// rejected nonfinite=2 domain=2 unchanged=true resumed-bitwise=true count=11
}
