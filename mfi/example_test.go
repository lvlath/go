// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Five reproducible market-data operations use only the exported mfi API.
// All fixtures are synthetic, with realistic OHLC geometry and base-unit volumes;
// they illustrate engineering workflows, not claims about live market conditions.
package mfi_test

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"time"

	"github.com/lvlath/go/mfi"
)

// exampleMarketBars builds reproducible multi-instrument OHLCV fixtures.
// Every price is positive, every candle has realistic intrabar spread, and
// volume may be zero without indicating that the candle is absent.
func exampleMarketBars(instrument, count int) []mfi.Bar {
	rows := make([]mfi.Bar, count)
	base := 75 + float64(instrument)*3.75
	previousClose := base
	for i := range rows {
		drift := float64((instrument%5)-2) * 0.045 * float64(i)
		cycle := float64(((i*11+instrument*7)%17)-8) * 0.13
		closePrice := base + drift + cycle
		openPrice := previousClose
		spread := 0.12 + float64((i+instrument)%4)*0.035
		volume := float64(140 + (i*37+instrument*29)%410)
		if (i+instrument)%19 == 0 {
			volume = 0 // A real recorded bar with no matched volume.
		}
		rows[i] = mfi.Bar{
			Open:   openPrice,
			High:   math.Max(openPrice, closePrice) + spread,
			Low:    math.Min(openPrice, closePrice) - spread,
			Close:  closePrice,
			Volume: volume,
		}
		previousClose = closePrice
	}
	return rows
}

// exampleColumns converts the audited row fixture into the public column facade.
// A caller with existing market-data columns would skip this example-only adapter.
func exampleColumns(rows []mfi.Bar) mfi.Bars {
	columns := mfi.Bars{
		Open: make([]float64, len(rows)), High: make([]float64, len(rows)),
		Low: make([]float64, len(rows)), Close: make([]float64, len(rows)),
		Volume: make([]float64, len(rows)),
	}
	for i, row := range rows {
		columns.Open[i], columns.High[i], columns.Low[i] = row.Open, row.High, row.Low
		columns.Close[i], columns.Volume[i] = row.Close, row.Volume
	}
	return columns
}

// ExampleCompute_crossAssetLiquidityReport demonstrates a deterministic
// daily cross-asset MFI screening pipeline over 50 independent instruments.
//
// Scenario:
//   - A research service receives 64 aligned, validated candles per instrument.
//   - MFI(14) is a numeric feature for reviewing price-volume divergence, not a
//     trader or whale-identification policy inside the library.
//
// Process:
//   - Build reproducible OHLCV records and submit independently via ComputeDefault.
//   - Evaluate a downstream review criterion on the final *valid* observation.
//   - Report inventory, zero-volume bar retention and review queue size.
//
// Operational value:
//   - Batch processing can feed cross-asset monitoring and observability with
//     consistent units and repeatable results, without cross-instrument state leaks.
func ExampleCompute_crossAssetLiquidityReport() {
	const assets = 50
	const candlesPerAsset = 64
	review, ready, zeroVolume := 0, 0, 0
	for instrument := 0; instrument < assets; instrument++ {
		rows := exampleMarketBars(instrument, candlesPerAsset)
		for _, row := range rows {
			if row.Volume == 0 {
				zeroVolume++
			}
		}
		result, err := mfi.ComputeDefault(exampleColumns(rows))
		if err != nil {
			fmt.Println("compute failure:", err)
			return
		}
		last := len(result.Values) - 1
		if !result.Valid[last] {
			continue
		}
		ready++
		// Policy belongs to the application: rising close with weak money flow
		// is merely queued for research, never interpreted as a trading signal.
		if rows[last].Close > rows[0].Close && result.Values[last] < 45 {
			review++
		}
	}
	fmt.Printf("assets=%d ready=%d zero-volume-bars=%d review-queue=%d\n", assets, ready, zeroVolume, review)

	// Output:
	// assets=50 ready=50 zero-volume-bars=165 review-queue=4
}

// ExampleStream_Push_tightRangePressure demonstrates streaming quality monitoring
// for a liquid pair whose prices stay in a narrow range while volumes fluctuate.
//
// Scenario:
//   - An exchange gateway receives validated one-minute OHLCV candles.
//   - A downstream analytics policy flags high MFI inside a tight recent price band
//     for analyst review; MFI itself never identifies institutional participants.
//
// Process:
//   - Push 42 consecutive candles to one fixed-window engine.
//   - Track the last eight closes and calculate the range separately from MFI.
//   - Publish the count of warmed-up, numerically high-MFI windows.
func ExampleStream_Push_tightRangePressure() {
	stream, err := mfi.NewPeriod(14)
	if err != nil {
		fmt.Println(err)
		return
	}
	rows := exampleMarketBars(4, 42)
	highPressure := 0
	validWindows := 0
	for i, bar := range rows {
		value, valid, pushErr := stream.Push(bar)
		if pushErr != nil {
			fmt.Println("stream failure:", pushErr)
			return
		}
		if !valid {
			continue
		}
		validWindows++
		start := i - 7
		if start < 0 {
			start = 0
		}
		low, high := rows[start].Close, rows[start].Close
		for _, candle := range rows[start : i+1] {
			low = math.Min(low, candle.Close)
			high = math.Max(high, candle.Close)
		}
		if value >= 65 && (high-low)/bar.Close <= 0.025 {
			highPressure++
		}
	}
	fmt.Printf("valid-windows=%d tight-range-high-MFI=%d state-bars=%d\n",
		validWindows, highPressure, stream.Snapshot().Count)

	// Output:
	// valid-windows=28 tight-range-high-MFI=13 state-bars=42
}

// ExampleStream_Snapshot_authenticatedFailover shows recovery of a replicated
// stream using externally authenticated checkpoint bytes, not the Config hash.
//
// Scenario:
//   - A primary market-data worker checkpoints its live MFI ring after 29 bars.
//   - A standby verifies the entire serialized state before Restore and resumes
//     without recomputing the retained price and volume history.
//
// Process:
//   - Serialize the detached Snapshot and apply a demonstration HMAC.
//   - Reject one tampered payload before it reaches Restore.
//   - Decode valid bytes and prove bitwise identity for 19 resumed observations.
//
// Operational value:
//   - Restoring Neumaier corrections avoids numeric divergence after failover.
//   - Production keys must come from secret management, not from this fixture.
func ExampleStream_Snapshot_authenticatedFailover() {
	rows := exampleMarketBars(11, 48)
	primary, err := mfi.NewPeriod(14)
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, bar := range rows[:29] {
		if _, _, err = primary.Push(bar); err != nil {
			fmt.Println(err)
			return
		}
	}

	checkpoint := primary.Snapshot()
	payload, err := json.Marshal(checkpoint)
	if err != nil {
		fmt.Println(err)
		return
	}
	key := []byte("example-only-key-loaded-from-a-real-secret-store")
	authenticate := func(raw []byte) []byte {
		mac := hmac.New(sha256.New, key)
		_, _ = mac.Write(raw)
		return mac.Sum(nil)
	}
	signature := authenticate(payload)
	altered := append([]byte(nil), payload...)
	altered[len(altered)-2] ^= 1
	tamperRejected := !hmac.Equal(signature, authenticate(altered))

	standby, err := mfi.NewPeriod(14)
	if err != nil {
		fmt.Println(err)
		return
	}
	if !hmac.Equal(signature, authenticate(payload)) {
		fmt.Println("invalid checkpoint authentication")
		return
	}
	var decoded mfi.Snapshot
	if err = json.Unmarshal(payload, &decoded); err != nil {
		fmt.Println("checkpoint decode failure:", err)
		return
	}
	if err = standby.Restore(decoded); err != nil {
		fmt.Println("restore failure:", err)
		return
	}

	identical := true
	for _, bar := range rows[29:] {
		left, readyLeft, leftErr := primary.Push(bar)
		right, readyRight, rightErr := standby.Push(bar)
		if leftErr != nil || rightErr != nil || readyLeft != readyRight ||
			math.Float64bits(left) != math.Float64bits(right) {
			identical = false
			break
		}
	}
	fmt.Printf("checkpoint-bars=%d tamper-rejected=%t resumed-bitwise=%t\n",
		checkpoint.Count, tamperRejected, identical)

	// Output:
	// checkpoint-bars=29 tamper-rejected=true resumed-bitwise=true
}

// ExampleStream_Reset_marketDataGap demonstrates an explicit gap boundary
// between exchange-data sessions, rather than silently bridging missing candles.
//
// Scenario:
//   - A minute-bar consumer receives eight continuous bars, then a feed outage.
//   - The next eleven bars resume after a gap large enough to invalidate history.
//
// Process:
//   - Compare timestamps before each Push, Reset on every missing interval.
//   - Count invalid post-reset warmup observations and the first valid output.
//   - Preserve the exact period and ring allocation across the reset.
func ExampleStream_Reset_marketDataGap() {
	const period = 5
	stream, err := mfi.NewPeriod(period)
	if err != nil {
		fmt.Println(err)
		return
	}
	rows := exampleMarketBars(7, 19)
	start := time.Date(2026, 11, 17, 9, 0, 0, 0, time.UTC)
	previous := time.Time{}
	resets, warmupAfterGap := 0, 0
	var finalValid bool
	for i, bar := range rows {
		minute := i
		if i >= 8 {
			minute += 25 // The next bar arrives after a missing 25-minute interval.
		}
		stamp := start.Add(time.Duration(minute) * time.Minute)
		if !previous.IsZero() && stamp.Sub(previous) != time.Minute {
			stream.Reset()
			resets++
		}
		_, finalValid, err = stream.Push(bar)
		if err != nil {
			fmt.Println("invalid candle:", err)
			return
		}
		if resets > 0 && !finalValid {
			warmupAfterGap++
		}
		previous = stamp
	}
	fmt.Printf("gaps=%d post-gap-warmup=%d final-valid=%t\n", resets, warmupAfterGap, finalValid)

	// Output:
	// gaps=1 post-gap-warmup=5 final-valid=true
}

// ExampleComputeCandles_replayReconciliation illustrates why ingestion must not
// silently drop zero-volume bars when comparing archival and replayed streams.
//
// Scenario:
//   - An archive contains 36 OHLCV bars, one of which has zero volume but a
//     legitimate price transition.
//   - A faulty ETL copy omits that bar and produces a different rolling history.
//
// Process:
//   - Compute canonical MFI on both row-oriented inputs with the same period.
//   - Compare their last valid scores and report the missing observation.
//   - Treat the discrepancy as a data-integrity issue, not a trade recommendation.
func ExampleComputeCandles_replayReconciliation() {
	archive := exampleMarketBars(3, 36)
	archive[29].Volume = 0
	faulty := append([]mfi.Bar(nil), archive[:29]...)
	faulty = append(faulty, archive[30:]...)
	correct, err := mfi.ComputeCandles(archive, mfi.Options{Period: 14})
	if err != nil {
		fmt.Println(err)
		return
	}
	observed, err := mfi.ComputeCandles(faulty, mfi.Options{Period: 14})
	if err != nil {
		fmt.Println(err)
		return
	}
	lastGood, lastBad := len(correct.Values)-1, len(observed.Values)-1
	diverged := math.Float64bits(correct.Values[lastGood]) != math.Float64bits(observed.Values[lastBad])
	fmt.Printf("archive-bars=%d etl-bars=%d complete=%t latest-diverged=%t\n",
		len(archive), len(faulty), correct.Valid[lastGood] && observed.Valid[lastBad], diverged)

	// Output:
	// archive-bars=36 etl-bars=35 complete=true latest-diverged=true
}
