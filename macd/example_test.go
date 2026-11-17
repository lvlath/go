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
	"time"

	"github.com/lvlath/go/macd"
)

// ExampleComputeDefault_crossAssetScreen illustrates monitoring a small set of
// liquid-equity-style price histories without assuming all assets have equal momentum.
//
// Scenario:
//   - A nightly research job receives aligned closing bars for three instruments.
//   - It computes the canonical 12/26/9 feature on each bar history and reports
//     both the raw signed histogram and a scale-comparable basis-point measure.
//   - All fixture values are synthetic, reproducible exchange-like closes.
//
// Process:
//   - Generate complete positive bar series in the same observation cadence.
//   - Require the latest Signal and Histogram to be ready, not merely nonzero.
//   - Report signed histogram in price units and relative basis points.
//
// AI-Hints:
//   - Histograms in price units cannot be directly compared across instruments.
//     Dividing by close produces a dimensionless quantity, but it is NOT a
//     volatility- or liquidity-adjusted risk score or a trading instruction.
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
		// Express the spread relative to its own closing price before comparing
		// instruments that trade at different nominal price levels. This remains a
		// descriptive, non-volatility-adjusted statistic, not a risk-normalized score.
		histogramBps := 10000 * result.Histogram.Values[last] / closes[last]
		fmt.Printf("%s: %d bars, momentum-change=%s, histogram=%+.3f (%+.1f bp of close)\n",
			instrument.name, len(closes), direction, result.Histogram.Values[last], histogramBps)
	}

	// Output:
	// infrastructure: 56 bars, momentum-change=rising, histogram=+0.656 (+43.4 bp of close)
	// utilities: 56 bars, momentum-change=rising, histogram=+0.097 (+16.9 bp of close)
	// semiconductors: 56 bars, momentum-change=falling, histogram=-0.920 (-50.5 bp of close)
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

// ExampleStream_Push_reconciledMinuteBars demonstrates exactly-once feed admission
// around the MACD recurrence rather than assuming that every received bar is valid.
// It models a production-style minute-bar inspection gateway with explicit sequence
// and timestamp identities, OHLC shape validation, and correction-first recovery.
//
// Scenario:
//   - A deterministic, synthetic 42-minute price replay arrives in event order.
//   - A malformed OHLC record is quarantined and corrected at the same sequence.
//   - A premature future sequence and a duplicate are independently rejected.
//   - Only complete admitted bars advance a single-owner MACD(3,8,4) stream.
//
// Implementation:
//   - Stage 1: construct minute bars with monotone identities and valid OHLC bounds.
//   - Stage 2: classify broken shape and sequence values before Stream.Push.
//   - Stage 3: replay corrected records without inventing or skipping a time slot.
//   - Stage 4: verify each published Point against a complete canonical batch.
//
// Behavior highlights:
//   - A rejected record never reaches Push, so the accepted Count remains intact.
//   - Consecutive minute continuity and duplicate control belong to the feed layer.
//   - SignalValid is checked explicitly; a zero histogram can still be valid.
//
// Determinism:
//   - The synthetic bar values, UTC event identities, and checking order are fixed.
//
// Complexity:
//   - Time O(n), Space O(n) for this diagnostic replay and stored Point witnesses.
//   - Each successful Stream.Push is O(1) time and O(1) state in the library.
//
// Notes:
//   - The stream does not validate OHLC, volumes, timestamps or exchange revisions.
//   - Market signals here are descriptive only and are never used to place orders.
//
// AI-Hints:
//   - Never silently discard an invalid minute and feed the next minute as adjacent:
//     resolve its sequence first, or explicitly reset/rebuild the downstream stream.
//   - Keep the provenance of corrected bars outside Snapshot; the config hash
//     authenticates neither event data nor the checkpoint's external cursor.
func ExampleStream_Push_reconciledMinuteBars() {
	// One dedicated accumulator represents one instrument and one fixed bar cadence.
	stream, err := macd.NewWithPeriods(3, 8, 4)
	if err != nil {
		fmt.Println("setup:", err)
		return
	}

	// The fixture resembles a one-minute change in short-term order-flow pressure;
	// it deliberately uses fixed synthetic values rather than mutable exchange data.
	closes := []float64{
		102.04, 102.10, 102.16, 102.23, 102.29, 102.36, 102.42, 102.50,
		102.57, 102.65, 102.69, 102.63, 102.54, 102.42, 102.28, 102.12,
		101.95, 101.83, 101.74, 101.66, 101.60, 101.62, 101.70, 101.82,
		101.98, 102.15, 102.32, 102.46, 102.61, 102.72, 102.79, 102.75,
		102.64, 102.49, 102.31, 102.15, 101.97, 101.85, 101.77, 101.81,
		101.84, 101.93,
	}
	start := time.Date(2026, time.November, 16, 14, 30, 0, 0, time.UTC)
	type minuteBar struct {
		sequence          uint64
		timestamp         time.Time
		open, high        float64
		low, closingPrice float64
		trades            int
	}
	badShape := errors.New("invalid OHLC bar")
	badOrder := errors.New("unexpected bar identity")

	// Admission is a caller-owned transaction: validate the source record before
	// invoking the kernel and publish the cursor only after a successful Push.
	var lastSequence uint64
	points := make([]macd.Point, 0, len(closes))
	admit := func(bar minuteBar) error {
		if bar.sequence != lastSequence+1 ||
			!bar.timestamp.Equal(start.Add(time.Duration(lastSequence)*time.Minute)) {
			return badOrder
		}
		if bar.trades <= 0 || !finitePositiveOHLC(bar.open, bar.high, bar.low, bar.closingPrice) {
			return badShape
		}
		point, pushErr := stream.Push(bar.closingPrice)
		if pushErr != nil {
			return pushErr
		}
		lastSequence = bar.sequence
		points = append(points, point)
		return nil
	}

	shapeRejected, orderRejected := 0, 0
	for index, price := range closes {
		opening := price
		if index > 0 {
			opening = closes[index-1]
		}
		bar := minuteBar{
			sequence:     lastSequence + 1,
			timestamp:    start.Add(time.Duration(index) * time.Minute),
			open:         opening,
			high:         math.Max(opening, price) + 0.08,
			low:          math.Min(opening, price) - 0.07,
			closingPrice: price,
			trades:       700 + (index%7)*55,
		}

		// First try the malformed record. The authoritative correction retains
		// the SAME sequence and timestamp, so no time slot goes missing.
		if index == 17 {
			corrupt := bar
			corrupt.high = corrupt.low - 0.01
			if rejectErr := admit(corrupt); !errors.Is(rejectErr, badShape) {
				fmt.Println("wrong OHLC rejection:", rejectErr)
				return
			}
			shapeRejected++
		}

		// An early next-minute event is not a license to skip this minute.
		if index == 24 {
			premature := bar
			premature.sequence++
			if rejectErr := admit(premature); !errors.Is(rejectErr, badOrder) {
				fmt.Println("wrong order rejection:", rejectErr)
				return
			}
			orderRejected++
		}

		if applyErr := admit(bar); applyErr != nil {
			fmt.Println("valid bar rejected:", applyErr)
			return
		}

		// A transport retry of an already committed record cannot advance EMA.
		if index == 30 {
			if rejectErr := admit(bar); !errors.Is(rejectErr, badOrder) {
				fmt.Println("duplicate not rejected:", rejectErr)
				return
			}
			orderRejected++
		}
	}

	// Batch and streaming go through the same canonical recurrence, but this
	// comparison additionally checks the end-to-end identity/quality gate's shape.
	batch, err := macd.ComputeWithPeriods(closes, 3, 8, 4)
	if err != nil {
		fmt.Println("batch check:", err)
		return
	}
	bitwise := true
	ready := 0
	for index, point := range points {
		bitwise = bitwise &&
			point.LineValid == batch.Line.Valid[index] &&
			point.SignalValid == batch.Signal.Valid[index] &&
			math.Float64bits(point.Line) == math.Float64bits(batch.Line.Values[index]) &&
			math.Float64bits(point.Signal) == math.Float64bits(batch.Signal.Values[index]) &&
			math.Float64bits(point.Histogram) == math.Float64bits(batch.Histogram.Values[index])
		if point.SignalValid {
			ready++
		}
	}
	fmt.Printf("accepted=%d signal-ready=%d malformed=%d out-of-order=%d bitwise=%t count=%d\n",
		len(points), ready, shapeRejected, orderRejected, bitwise, stream.Snapshot().Count)

	// Output:
	// accepted=42 signal-ready=32 malformed=1 out-of-order=2 bitwise=true count=42
}

// finitePositiveOHLC validates the candle-level price shape independently
// of MACD's scalar close validation; it does not repair corrupt market bars.
// Time O(1), Space O(1), no allocation.
func finitePositiveOHLC(open, high, low, closePrice float64) bool {
	return !math.IsNaN(open) && !math.IsInf(open, 0) &&
		!math.IsNaN(high) && !math.IsInf(high, 0) &&
		!math.IsNaN(low) && !math.IsInf(low, 0) &&
		!math.IsNaN(closePrice) && !math.IsInf(closePrice, 0) &&
		open > 0 && low > 0 && high >= math.Max(open, closePrice) &&
		low <= math.Min(open, closePrice)
}
