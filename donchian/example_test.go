// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package donchian_test demonstrates fixed-data monitoring and ingestion pipelines.
// This file uses sourced daily observations and explicitly labeled engineered stress data.
// AI-Hints: all examples are offline; an observed signal or next open is never a fill promise.
package donchian_test

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/lvlath/go/donchian"
)

// ExampleComputeIncludeCurrent_marketRangeReport builds a finalized-session range report for IBM.
// A chart service consumes the current 20-observation channel and a guarded close position.
//
// Implementation:
//   - Stage 1: load 58 frozen vendor observations and detach their high/low channels.
//   - Stage 2: compute the inclusive 20-observation range through the named facade.
//   - Stage 3: check readiness and width before displaying the close within the channel.
//
// Behavior highlights:
//   - The latest observed session is 2026-11-20, not an automatically refreshed quote.
//   - The report uses the current bar intentionally; it does not test a prior-range breakout.
//
// Inputs:
//   - The Alpha Vantage IBM demo snapshot in ibmDailyCandles; prices remain in supplied units.
//
// Returns:
//   - Two deterministic report lines, with rounding applied only for display.
//
// Errors:
//   - All Compute errors are checked; an unusable width suppresses normalization.
//
// Determinism:
//   - Frozen observations in ascending session order; no network or clock is consulted.
//
// Complexity:
//   - O(n) copying and channel computation; O(n) outputs plus O(min(20,n)) workspace.
//
// Notes:
//   - The vendor response is not independently exchange-audited; fixture provenance is
//     recorded beside all 58 original OHLC rows in test_helper_test.go.
//   - Real deployments must separately govern adjustments, missing sessions, and bar finality.
//
// AI-Hints:
//   - Do not infer prior-bar thresholds from an inclusive overlay.
//   - A valid midpoint does not make width division safe for all finite inputs.
func ExampleComputeIncludeCurrent_marketRangeReport() {
	const period = 20 // Application-selected display window; not a library default.
	const percentageScale = 100.0
	// The reusable fixture preserves dates and closes outside the numerical API.
	candles := ibmDailyCandles()
	high, low := candleChannels(candles)
	// The function name makes current-bar inclusion explicit at the call site.
	channel, err := donchian.ComputeIncludeCurrent(context.Background(), high, low, period)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	// A payload may be read only after checking that its full window exists.
	last := len(candles) - 1
	if !channel.Valid[last] {
		fmt.Println("range report: insufficient observations")
		return
	}
	fmt.Printf("%s IBM current20 U=%.4f M=%.4f L=%.4f\n",
		candles[last].date, channel.Upper[last], channel.Middle[last], channel.Lower[last])
	// Width is an application-derived quantity; finite boundaries alone do not guarantee it.
	width := channel.Upper[last] - channel.Lower[last]
	if width == 0 || math.IsInf(width, 0) {
		fmt.Println("close-position: undefined width")
		return
	}
	// The admitted example close is inside its candle and hence inside this inclusive channel.
	position := (candles[last].close - channel.Lower[last]) / width
	fmt.Printf("width=%.4f close-position=%.2f%%\n", width, percentageScale*position)

	// Output:
	// 2026-11-20 IBM current20 U=259.8700 M=241.3068 L=222.7437
	// width=37.1263 close-position=4.57%
}

// ExampleComputeExcludeCurrent_twentySessionAudit separates prior-range events from later observations.
// A replay auditor records 20-session boundary crossings without fabricating executions.
//
// Implementation:
//   - Stage 1: load the fixed IBM daily vendor snapshot in original session order.
//   - Stage 2: build a previous-20-observation channel, then compare each ready close.
//   - Stage 3: report subsequent supplied opens only after recording each event.
//
// Behavior highlights:
//   - Both upward and downward crossings are reported; warmup rows produce no event.
//   - A later open is retrospective audit evidence, never an input to the event decision.
//
// Inputs:
//   - The 58 original OHLC records returned by ibmDailyCandles; Period=20.
//
// Returns:
//   - Dated boundary events, subsequent observed quotes, and an event count; no orders.
//
// Errors:
//   - Compute failures stop the replay; unavailable next observations are explicitly labeled.
//
// Determinism:
//   - Fixed rows, ascending index, and strict close > upper or close < lower comparisons.
//
// Complexity:
//   - O(n) total time including fixture extraction and consumption; O(n) result storage.
//
// Notes:
//   - This is an event audit, not a full Turtle system, trade simulator, or profitability claim.
//   - Daily open availability does not establish liquidity, a fill, slippage, or order eligibility.
//
// AI-Hints:
//   - Do not include the tested bar in the channel or execute at its already known close.
//   - Do not tune the period against these printed events; it is an explicit example policy.
func ExampleComputeExcludeCurrent_twentySessionAudit() {
	const period = 20 // Explicit previous-session reference, not twenty calendar days.
	candles := ibmDailyCandles()
	high, low := candleChannels(candles)
	channel, err := donchian.ComputeExcludeCurrent(context.Background(), high, low, period)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	events := 0
	for i, candle := range candles {
		// Do not interpret the zero-filled warmup prefix as an established range.
		if !channel.Valid[i] {
			continue
		}
		// Determine the event using only the current close and the preceding window.
		direction, threshold := "", 0.0
		switch {
		case candle.close > channel.Upper[i]:
			direction, threshold = "above", channel.Upper[i]
		case candle.close < channel.Lower[i]:
			direction, threshold = "below", channel.Lower[i]
		default:
			continue
		}
		events++
		fmt.Printf("%s %s prior20 close=%.4f threshold=%.4f\n", candle.date, direction, candle.close, threshold)
		// This is an audit annotation AFTER the decision, not a same-close fill assumption.
		if i+1 < len(candles) {
			fmt.Printf("  later observation: %s open=%.4f (not a fill)\n", candles[i+1].date, candles[i+1].open)
		} else {
			fmt.Println("  later observation: unavailable")
		}
	}
	fmt.Printf("events=%d; orders=0\n", events)

	// Output:
	// 2026-10-20 above prior20 close=241.9200 threshold=238.3800
	//   later observation: 2026-10-21 open=241.9150 (not a fill)
	// 2026-10-21 above prior20 close=251.4600 threshold=242.5250
	//   later observation: 2026-10-22 open=241.2450 (not a fill)
	// 2026-10-22 above prior20 close=252.1100 threshold=251.7500
	//   later observation: 2026-10-23 open=249.7300 (not a fill)
	// 2026-10-29 above prior20 close=258.2800 threshold=256.3000
	//   later observation: 2026-10-30 open=258.2800 (not a fill)
	// 2026-11-17 below prior20 close=231.0900 threshold=236.0000
	//   later observation: 2026-11-18 open=230.4500 (not a fill)
	// 2026-11-18 below prior20 close=228.5300 threshold=230.6400
	//   later observation: 2026-11-19 open=229.2200 (not a fill)
	// 2026-11-19 below prior20 close=223.8800 threshold=228.4000
	//   later observation: 2026-11-20 open=224.5950 (not a fill)
	// events=7; orders=0
}

// ExampleCompute_multiPeriodRangeMonitor compares recent and long-horizon ranges without price-level division.
// A monitoring job marks compression of the prior 10-observation range inside the prior 50.
//
// Implementation:
//   - Stage 1: extract the same fixed vendor price channels for both horizons.
//   - Stage 2: call canonical Compute with explicit exclude-current options for each period.
//   - Stage 3: join readiness masks and compare guarded fast/slow range widths.
//
// Behavior highlights:
//   - The ratio is dimensionless and does not divide by a possibly negative middle price.
//   - A threshold of 0.5 is an illustrative monitoring policy, not a fitted trading rule.
//
// Inputs:
//   - The frozen 58-row IBM snapshot, fast Period=10 and slow Period=50.
//
// Returns:
//   - One line per jointly ready session and counts of eligible/compressed observations.
//
// Errors:
//   - Both calls are checked independently; zero or infinite widths suppress the ratio.
//
// Determinism:
//   - Identical observation alignment, fixed threshold, and chronological reporting.
//
// Complexity:
//   - Two O(n) computations plus O(n) consumption; O(n) outputs for each horizon.
//
// Notes:
//   - The fast window is a subset of the slow window at the same row.
//   - This feature measures range compression, not future direction or volatility forecasts.
//
// AI-Hints:
//   - Do not compare rows from different sessions or ignore either readiness mask.
//   - Do not retrofit a threshold to obtain an attractive example outcome.
func ExampleCompute_multiPeriodRangeMonitor() {
	const fastPeriod = 10
	const slowPeriod = 50
	const compressionRatio = 0.5 // A declared demonstration threshold, not a forecast.
	candles := ibmDailyCandles()
	high, low := candleChannels(candles)
	// Canonical Options are useful when policy is stored or assembled by the caller.
	fast, err := donchian.Compute(context.Background(), high, low,
		donchian.Options{Period: fastPeriod, ExcludeCurrent: true})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	slow, err := donchian.Compute(context.Background(), high, low,
		donchian.Options{Period: slowPeriod, ExcludeCurrent: true})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	eligible, compressed := 0, 0
	for i, candle := range candles {
		// Equal slice lengths do not imply equal readiness for different periods.
		if !fast.Valid[i] || !slow.Valid[i] {
			continue
		}
		fastWidth := fast.Upper[i] - fast.Lower[i]
		slowWidth := slow.Upper[i] - slow.Lower[i]
		// Keep a zero-width long channel distinct from a small nonzero denominator.
		if slowWidth == 0 || math.IsInf(slowWidth, 0) || math.IsInf(fastWidth, 0) {
			fmt.Printf("%s ratio=undefined\n", candle.date)
			continue
		}
		ratio := fastWidth / slowWidth
		isCompressed := ratio <= compressionRatio
		eligible++
		if isCompressed {
			compressed++
		}
		fmt.Printf("%s fast10/slow50=%.3f compressed=%t\n", candle.date, ratio, isCompressed)
	}
	fmt.Printf("eligible=%d compressed=%d\n", eligible, compressed)

	// Output:
	// 2026-11-11 fast10/slow50=0.464 compressed=true
	// 2026-11-12 fast10/slow50=0.464 compressed=true
	// 2026-11-13 fast10/slow50=0.464 compressed=true
	// 2026-11-16 fast10/slow50=0.464 compressed=true
	// 2026-11-17 fast10/slow50=0.434 compressed=true
	// 2026-11-18 fast10/slow50=0.664 compressed=false
	// 2026-11-19 fast10/slow50=0.660 compressed=false
	// 2026-11-20 fast10/slow50=0.631 compressed=false
	// eligible=8 compressed=5
}

// ExampleComputeExcludeCurrent_negativeSpreadMonitor monitors a signed two-leg mark spread.
// A risk dashboard compares the latest sampled spread close against its prior range.
//
// Implementation:
//   - Stage 1: subtract simultaneous leg marks, then aggregate sampled spread extrema.
//   - Stage 2: calculate the previous 12-bar channel using the explicit exclude facade.
//   - Stage 3: normalize a positive overshoot by the previous width and flag the configured limit.
//
// Behavior highlights:
//   - All prices may be negative; no log or division by a negative price level is used.
//   - The sample bar extrema describe observed synchronized marks, not all intrabar trades.
//
// Inputs:
//   - Fourteen engineered bars with three synchronized two-leg observations each;
//     these stress observations are explicitly synthetic, not vendor or exchange history.
//
// Returns:
//   - Two ready risk-monitor rows with a dimensionless width-normalized overshoot.
//
// Errors:
//   - Compute errors stop the report; zero or unrepresentable width suppresses division.
//
// Determinism:
//   - Literal marks, fixed sample order, exact quarter-dollar arithmetic, and fixed policy.
//
// Complexity:
//   - O(n*k) aggregation for k=3 samples, then O(n) computation; O(n) storage.
//
// Notes:
//   - A threshold of 0.5 previous widths is an illustrative application alert limit.
//   - This is mark monitoring, not an executable bid/ask spread or an arbitrage guarantee.
//
// AI-Hints:
//   - Do not subtract independently aggregated leg highs to obtain a spread high.
//   - Keep sample synchronization and executable quote semantics outside donchian.
func ExampleComputeExcludeCurrent_negativeSpreadMonitor() {
	const period = 12
	const minimumOvershoot = 0.5 // Application alert policy in prior-channel widths.
	// The helper builds extrema from synchronized spreads, not from unrelated leg extrema.
	high, low, closes := syntheticSpreadBars()
	channel, err := donchian.ComputeExcludeCurrent(context.Background(), high, low, period)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	for i, closePrice := range closes {
		if !channel.Valid[i] {
			continue
		}
		width := channel.Upper[i] - channel.Lower[i]
		if width == 0 || math.IsInf(width, 0) {
			fmt.Printf("sampled-bar=%d overshoot=undefined\n", i)
			continue
		}
		// Finite quarter-dollar fixture values make this application subtraction safe.
		overshoot := (closePrice - channel.Upper[i]) / width
		alert := closePrice > channel.Upper[i] && overshoot >= minimumOvershoot
		fmt.Printf("sampled-bar=%d close=%.2f prior=[%.2f,%.2f] excess/width=%.2f alert=%t\n",
			i, closePrice, channel.Lower[i], channel.Upper[i], overshoot, alert)
	}

	// Output:
	// sampled-bar=12 close=-1.00 prior=[-2.25,-1.75] excess/width=1.50 alert=true
	// sampled-bar=13 close=-1.25 prior=[-2.25,-1.00] excess/width=-0.20 alert=false
}

// ExampleCompute_feedQuarantine keeps a last-good result when a new batch is malformed.
// An ingestion worker rejects a deliberately corrupted observation before publishing a replacement.
//
// Implementation:
//   - Stage 1: compute and retain the accepted prefix of the fixed vendor history.
//   - Stage 2: inject a NaN into a separate candidate batch and classify its rejection.
//   - Stage 3: restore that supplied observation, recompute, and publish only on success.
//
// Behavior highlights:
//   - The corruption is deliberately engineered; it was not present in the vendor response.
//   - Even the last excluded bar must be finite; its exclusion is not validation immunity.
//
// Inputs:
//   - The same frozen IBM records; a caller-owned copy receives the injected NaN.
//
// Returns:
//   - A quarantine report and a successful replacement report, without partial artifacts.
//
// Errors:
//   - ErrNonFinite is recognized through errors.Is; every other computation error is checked.
//
// Determinism:
//   - No goroutine, timeout, network retry, or current-time assumption is needed.
//
// Complexity:
//   - Three O(n) batch computations and O(n) owning buffers; no hidden stream state.
//
// Notes:
//   - The example is a single-owner publication pattern, not a concurrent cache implementation.
//   - A real service retains acquisition/revision identifiers separately from this numerical result.
//
// AI-Hints:
//   - Never replace the last-good artifact on error or interpret nil as an empty success.
//   - Do not invent Snapshot/Restore or salvage partial rows from a rejected batch.
func ExampleCompute_feedQuarantine() {
	const period = 20
	candles := ibmDailyCandles()
	high, low := candleChannels(candles)
	last := len(candles) - 1
	opts := donchian.Options{Period: period, ExcludeCurrent: true}
	// The previous accepted artifact is entirely owned by this single ingestion worker.
	published, err := donchian.Compute(context.Background(), high[:last], low[:last], opts)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	// Corrupt a copy: the original sourced observations remain available for repair.
	candidateHigh := append([]float64(nil), high...)
	candidateHigh[last] = math.NaN()
	candidate, candidateErr := donchian.Compute(context.Background(), candidateHigh, low, opts)
	if candidateErr == nil {
		published = candidate
	}
	fmt.Printf("quarantined=%t candidate-nil=%t retained-rows=%d\n",
		errors.Is(candidateErr, donchian.ErrNonFinite), candidate == nil, len(published.Valid))
	// Replace the bad observation with the frozen supplied value, then recompute explicitly.
	candidateHigh[last] = high[last]
	repaired, err := donchian.Compute(context.Background(), candidateHigh, low, opts)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	// Publication occurs only after success; no package state is mutated or restored.
	published = repaired
	if !published.Valid[last] {
		fmt.Println("repaired batch: insufficient observations")
		return
	}
	fmt.Printf("repaired-rows=%d date=%s ready=%t prior20=[%.4f,%.4f]\n",
		len(published.Valid), candles[last].date, published.Valid[last], published.Lower[last], published.Upper[last])

	// Output:
	// quarantined=true candidate-nil=true retained-rows=57
	// repaired-rows=58 date=2026-11-20 ready=true prior20=[222.7437,259.8700]
}
