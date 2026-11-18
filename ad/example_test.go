// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ad_test demonstrates practical financial-data pipelines with explicit units.
// All OHLCV fixtures are deterministic illustrative data, not current market quotes.
package ad_test

import (
	"errors"
	"fmt"
	"math"

	"github.com/lvlath/go/ad"
)

// ExampleCompute demonstrates an intraday liquidity-pressure summary from complete OHLCV columns.
//
// Scenario:
//   - A market-data service produces sixteen consecutive five-minute candles for a single stock.
//   - Its research pipeline needs the signed base-share flow balance, not a trading decision.
//   - Mid-range closes cancel their own volume while top/bottom closes add signed volume.
//
// Process:
//   - Validate and process all sixteen aligned bars with the canonical batch facade.
//   - Read the first bar, midpoint-of-session and final balances with the validity mask.
//   - Keep raw close-price movement separate from the price-position-weighted volume balance.
//
// Complexity:
//   - O(n) time and O(n) detached result storage for n candles.
//
// AI-Hints:
//   - Do not interpret a positive A/D total as proof of an identified buyer's activity.
func ExampleCompute() {
	candles := ad.Bars{
		Open:   []float64{100, 101, 102, 102, 103, 104, 104, 103, 104, 105, 106, 107, 107, 108, 109, 109},
		High:   []float64{103, 104, 105, 105, 106, 107, 107, 106, 107, 108, 109, 110, 110, 111, 112, 112},
		Low:    []float64{99, 100, 101, 101, 102, 103, 103, 102, 103, 104, 105, 106, 106, 107, 108, 108},
		Close:  []float64{102, 103, 102, 101, 106, 106, 104, 103, 107, 105, 108, 110, 107, 111, 109, 112},
		Volume: []float64{1000, 800, 1200, 1500, 900, 1100, 700, 600, 1300, 1500, 1000, 800, 400, 1200, 1400, 900},
	}
	result, err := ad.Compute(candles)
	if err != nil {
		fmt.Println("market data rejected:", err)
		return
	}
	last := len(result.Values) - 1
	fmt.Printf("candles=%d first=%.0f mid=%.0f final=%.0f valid=%v\n",
		len(result.Values), result.Values[0], result.Values[7], result.Values[last], result.Valid[last])
	fmt.Printf("close move=%.0f shares balance=%.0f\n", candles.Close[last]-candles.Close[0], result.Values[last])

	// Output:
	// candles=16 first=500 mid=-400 final=2650 valid=true
	// close move=10 shares balance=2650
}

// illustrativeBTCBars returns fixed five-minute BTC/base-volume candles for feed demonstrations.
// These are engineered examples, not a representation of any specific exchange's history.
func illustrativeBTCBars() []ad.Bar {
	return []ad.Bar{
		{Open: 92100, High: 92140, Low: 92100, Close: 92140, Volume: 14},
		{Open: 92120, High: 92140, Low: 92100, Close: 92110, Volume: 20},
		{Open: 92140, High: 92160, Low: 92120, Close: 92160, Volume: 13},
		{Open: 92140, High: 92160, Low: 92120, Close: 92120, Volume: 12},
		{Open: 92150, High: 92180, Low: 92140, Close: 92170, Volume: 22},
		{Open: 92180, High: 92200, Low: 92160, Close: 92180, Volume: 15},
		{Open: 92190, High: 92220, Low: 92180, Close: 92220, Volume: 17},
		{Open: 92200, High: 92220, Low: 92180, Close: 92190, Volume: 16},
		{Open: 92180, High: 92200, Low: 92160, Close: 92160, Volume: 18},
		{Open: 92160, High: 92180, Low: 92140, Close: 92160, Volume: 11},
		{Open: 92150, High: 92180, Low: 92140, Close: 92180, Volume: 24},
		{Open: 92170, High: 92200, Low: 92160, Close: 92190, Volume: 13},
		{Open: 92200, High: 92220, Low: 92180, Close: 92220, Volume: 18},
		{Open: 92210, High: 92240, Low: 92200, Close: 92200, Volume: 21},
		{Open: 92200, High: 92200, Low: 92200, Close: 92200, Volume: 9},
		{Open: 92200, High: 92240, Low: 92200, Close: 92230, Volume: 15},
		{Open: 92220, High: 92240, Low: 92200, Close: 92200, Volume: 19},
		{Open: 92200, High: 92240, Low: 92200, Close: 92240, Volume: 28},
	}
}

// ExampleStream_Push demonstrates allocation-free online accumulation from an intraday feed.
//
// Scenario:
//   - A gateway receives five-minute crypto OHLCV intervals as immutable Bar records.
//   - The service publishes an aggregate six-bar (30-minute) view without retaining raw bars.
//   - A flat candle still advances the service's observation count and preserves its balance.
//
// Process:
//   - Push each candle in input order, validating before every state transition.
//   - Print cumulative base-volume pressure after each six-candle period.
//   - Treat failures as rejected ingestion events, never as valid numeric zeros.
//
// Complexity:
//   - O(1) time, O(1) memory, zero heap allocations per successful Push.
//
// AI-Hints:
//   - The returned balance is in BTC base-volume units; it is not USD turnover.
func ExampleStream_Push() {
	stream := ad.New()
	for index, bar := range illustrativeBTCBars() {
		balance, valid, err := stream.Push(bar)
		if err != nil || !valid {
			fmt.Println("ingestion rejected")
			return
		}
		if (index+1)%6 == 0 {
			fmt.Printf("half-hour %d: cumulative=%.1f BTC count=%d\n",
				(index+1)/6, balance, stream.Snapshot().Count)
		}
	}

	// Output:
	// half-hour 1: cumulative=16.0 BTC count=6
	// half-hour 2: cumulative=37.5 BTC count=12
	// half-hour 3: cumulative=51.0 BTC count=18
}

// ExampleStream_Restore demonstrates standby reconstruction of a live data-processing service.
//
// Scenario:
//   - A primary market-data worker has processed twelve of eighteen candles when it fails over.
//   - A standby must continue with the same compensated low bits, not only the displayed total.
//   - The checkpoint crosses a process boundary by value, with no shared Stream pointer.
//
// Process:
//   - Ingest twelve candles on primary, then export a fixed-schema detached Snapshot.
//   - Restore that checkpoint on a new independent worker.
//   - Replay the remaining six candles on both paths and compare exact binary64 outputs.
//
// Complexity:
//   - Checkpoint export/restore O(1), six continuation pushes O(6), constant state.
//
// AI-Hints:
//   - Persist the pair [sum, compensation] and Count; Config is not a payload signature.
func ExampleStream_Restore() {
	candles := illustrativeBTCBars()
	primary := ad.New()
	for _, bar := range candles[:12] {
		if _, _, err := primary.Push(bar); err != nil {
			fmt.Println("primary error:", err)
			return
		}
	}

	checkpoint := primary.Snapshot()
	standby := ad.New()
	if err := standby.Restore(checkpoint); err != nil {
		fmt.Println("restore error:", err)
		return
	}
	parity := true
	var final float64
	for _, bar := range candles[12:] {
		left, leftValid, leftErr := primary.Push(bar)
		right, rightValid, rightErr := standby.Push(bar)
		if leftErr != nil || rightErr != nil || !leftValid || !rightValid {
			fmt.Println("continuation error")
			return
		}
		parity = parity && math.Float64bits(left) == math.Float64bits(right)
		final = right
	}
	fmt.Printf("failover parity=%v checkpoint=%d final=%0.1f count=%d fields=%d\n",
		parity, checkpoint.Count, final, standby.Snapshot().Count, len(checkpoint.FloatState))

	// Output:
	// failover parity=true checkpoint=12 final=51.0 count=18 fields=2
}

// ExampleComputeCandles demonstrates a cross-asset disagreement screen built outside A/D.
//
// Scenario:
//   - A research service screens one hundred illustrative assets over twenty daily candles.
//   - It selects assets whose latest close did not rise while their A/D balance is positive.
//   - A/D is one screening feature, not a buyer identity, whale detector, or trading signal.
//
// Process:
//   - Build each asset's deterministic but varied daily OHLCV rows.
//   - Call ComputeCandles independently for each asset and consume its terminal value.
//   - Keep the screen outside the kernel and publish symbol order deterministically.
//
// Complexity:
//   - O(assets*days) processing and O(days) transient result memory per asset.
//
// AI-Hints:
//   - Avoid ranking absolute A/D across assets without normalizing volume units elsewhere.
func ExampleComputeCandles() {
	const assets, days = 100, 20
	candidates := make([]string, 0, assets)
	for asset := 0; asset < assets; asset++ {
		candles := make([]ad.Bar, days)
		base := 120 + float64(asset)
		for day := 0; day < days; day++ {
			// Five groups generate downward, upward and flat close trends.
			// The close position within each day's range varies independently.
			basePrice := base
			switch asset % 5 {
			case 0, 1:
				basePrice -= float64(day)
			case 2, 3:
				basePrice += float64(day)
			}
			closeOffset := 4.0
			if asset%5 == 1 || asset%5 == 3 {
				closeOffset = 0
			}
			candles[day] = ad.Bar{
				Open: basePrice + 2, High: basePrice + 4, Low: basePrice,
				Close: basePrice + closeOffset, Volume: float64(250 + (asset%7)*10 + day),
			}
		}

		series, err := ad.ComputeCandles(candles)
		if err != nil {
			fmt.Println("screening input error:", err)
			return
		}
		latest := len(series.Values) - 1
		if candles[latest].Close <= candles[0].Close && series.Valid[latest] && series.Values[latest] > 0 {
			candidates = append(candidates, fmt.Sprintf("SIM-%03d", asset))
		}
	}
	fmt.Printf("universe=%d evaluated=%d candidates=%d first=%v\n",
		assets, days, len(candidates), candidates[:4])

	// Output:
	// universe=100 evaluated=20 candidates=40 first=[SIM-000 SIM-004 SIM-005 SIM-009]
}

// ExampleLatest demonstrates an end-of-day data-quality gate before report publication.
//
// Scenario:
//   - A reporting process stores a validated day of OHLCV arrays and needs only final A/D.
//   - A later vendor correction contains a Close outside High; reporting must fail closed.
//   - Errors are classified by errors.Is and are not treated as zero accumulated volume.
//
// Process:
//   - Call Latest on independently owned columns, then consume its valid terminal value.
//   - Present an invalid corrected candle in a separate batch and reject the entire report.
//   - Preserve input provenance instead of clipping the invalid close automatically.
//
// Complexity:
//   - Latest is an O(n) time, O(n) space projection of Compute.
//
// AI-Hints:
//   - If the caller needs only live terminal values, Stream.Push avoids batch allocations.
func ExampleLatest() {
	candles := ad.Bars{
		Open:   []float64{70, 71, 71, 72, 72, 71, 72, 73, 73, 74, 74, 73},
		High:   []float64{72, 73, 73, 74, 74, 73, 74, 75, 75, 76, 76, 75},
		Low:    []float64{68, 69, 69, 70, 70, 69, 70, 71, 71, 72, 72, 71},
		Close:  []float64{72, 72, 70, 74, 71, 69, 73, 75, 72, 76, 73, 75},
		Volume: []float64{200, 220, 180, 210, 190, 150, 250, 260, 230, 280, 170, 240},
	}
	last, valid, err := ad.Latest(candles)
	if err != nil {
		fmt.Println("daily report rejected:", err)
		return
	}
	fmt.Printf("daily A/D=%.0f shares valid=%v\n", last, valid)

	// Simulate a separately received, malformed vendor update. The library
	// must reject it as a domain error, not silently move High or Close.
	candles.Close[11] = 80
	last, valid, err = ad.Latest(candles)
	fmt.Printf("vendor correction rejected=%v valid=%v usable-value=%.0f\n",
		errors.Is(err, ad.ErrDomain), valid, last)

	// Output:
	// daily A/D=890 shares valid=true
	// vendor correction rejected=true valid=false usable-value=0
}

// ExampleCompute_invalidOHLC demonstrates an atomic batch rejection by sentinel.
// A malformed close is not clipped to High and no partial Result is returned.
func ExampleCompute_invalidOHLC() {
	bars := ad.Bars{
		Open: []float64{100}, High: []float64{102}, Low: []float64{99},
		Close: []float64{103}, Volume: []float64{500},
	}
	result, err := ad.Compute(bars)
	fmt.Println(result == nil, errors.Is(err, ad.ErrDomain))
	// Output:
	// true true
}
