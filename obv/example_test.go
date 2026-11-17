// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package obv_test demonstrates operational OBV pipelines using the public facade.
// Fixtures are deterministic engineering examples, not empirical trading recommendations.
package obv_test

import (
	"errors"
	"fmt"
	"math"

	"github.com/lvlath/go/obv"
)

// ExampleCompute_accumulationWatch models a market-surveillance desk observing
// a narrow price corridor alongside increasingly one-sided traded base volume.
//
// Scenario:
//   - A 12-bar session oscillates between two nearby closing prices.
//   - Aggressive up-close bars carry materially more volume than down-close bars.
//   - The observer flags positive cumulative volume without claiming that OBV
//     identifies who bought, or predicts the next price move.
//
// Process:
//   - Build aligned OHLCV columns using the session's base-volume units.
//   - Validate and calculate the entire session through canonical Compute.
//   - Compare the last balance with zero while checking price containment.
func ExampleCompute_accumulationWatch() {
	closes := []float64{100.20, 100.45, 100.28, 100.52, 100.31, 100.56, 100.33, 100.60, 100.40, 100.63, 100.46, 100.65}
	baseVolumes := []float64{650, 1200, 330, 1550, 400, 1800, 410, 1950, 450, 2100, 470, 2400}
	// The opening price follows the previous close; each candle has a
	// plausible intrabar range rather than a synthetic zero-range shortcut.
	// All five columns retain identical chronological indexing and volume units.
	opens := make([]float64, len(closes))
	highs := make([]float64, len(closes))
	lows := make([]float64, len(closes))
	previousClose := closes[0]
	for index, closingPrice := range closes {
		opens[index] = previousClose
		highs[index] = math.Max(previousClose, closingPrice) + 0.15
		lows[index] = math.Min(previousClose, closingPrice) - 0.15
		previousClose = closingPrice
	}
	result, err := obv.Compute(obv.Bars{
		Open: opens, High: highs, Low: lows, Close: closes, Volume: baseVolumes,
	})
	if err != nil {
		fmt.Println("invalid session:", err)
		return
	}
	maxClose, minClose := closes[0], closes[0]
	for _, closingPrice := range closes {
		if closingPrice > maxClose {
			maxClose = closingPrice
		}
		if closingPrice < minClose {
			minClose = closingPrice
		}
	}
	// This threshold belongs to the monitoring desk, not the OBV recurrence.
	const maximumCorridorWidth = 0.50
	last := result.Values[len(result.Values)-1]
	watchCandidate := maxClose-minClose <= maximumCorridorWidth && last > 0
	fmt.Printf("bars=%d corridor=%.2f obv=%.0f watch-candidate=%t\n", len(result.Values), maxClose-minClose, last, watchCandidate)
	// Output: bars=12 corridor=0.45 obv=8940 watch-candidate=true
}

// ExampleComputeCandles_futuresBasket screens one hundred independent contract
// histories for price breakouts that are also supported by positive signed volume.
//
// Scenario:
//   - A portfolio gateway receives sixteen complete candles per contract,
//     with exchange contract counts normalized upstream to each instrument's
//     consistent base-volume unit before calling the indicator.
//   - Four fixed feed regimes represent upward moves with strong up-bar volume,
//     upward moves dominated by down-bar volume, range-bound accumulation, and
//     bearish declines. Each regime appears in twenty-five contract histories.
//   - A close strictly above every preceding candle high is a price breakout.
//     Whether a positive OBV supports that breakout is a caller-defined screen,
//     not a forecast or trading recommendation embedded in the indicator.
//
// Process:
//   - Construct deterministic, geometrically valid row-oriented OHLCV records.
//   - Run ComputeCandles independently per contract, preserving volume units.
//   - Count breakout confirmations separately from unsupported price advances.
//
// Notes:
//   - These are purpose-built synthetic fixtures, not exchange-reported prices.
//   - OBV does not identify who traded or supply a forecast or trade decision.
func ExampleComputeCandles_futuresBasket() {
	const (
		contractCount   = 100
		barsPerContract = 16
	)

	confirmed, unconfirmed, breakouts := 0, 0, 0
	for contract := 0; contract < contractCount; contract++ {
		candles := make([]obv.Bar, barsPerContract)
		closingPrice := 2400.0 + float64(contract)*0.25

		for index := range candles {
			openingPrice := closingPrice
			baseVolume := 5000.0 // A seed candle never contributes its normalized base volume.

			if index > 0 {
				switch contract % 4 {
				case 0: // Rising prices with heavier volume on up closes.
					if index%4 == 0 {
						closingPrice -= 0.30
						baseVolume = 200
					} else {
						closingPrice += 0.65
						baseVolume = 1200
					}
				case 1: // Rising prices with heavier volume on down closes.
					if index%4 == 0 {
						closingPrice -= 0.30
						baseVolume = 3200
					} else {
						closingPrice += 0.65
						baseVolume = 150
					}
				case 2: // Sub-breakout range with positive cumulative volume.
					if index%2 == 0 {
						closingPrice -= 0.25
						baseVolume = 100
					} else {
						closingPrice += 0.25
						baseVolume = 950
					}
				default: // Falling prices and negative cumulative volume.
					if index%4 == 0 {
						closingPrice += 0.30
						baseVolume = 100
					} else {
						closingPrice -= 0.65
						baseVolume = 700
					}
				}
			}

			candles[index] = obv.Bar{
				Open: openingPrice, High: math.Max(openingPrice, closingPrice) + 0.50,
				Low:   math.Min(openingPrice, closingPrice) - 0.50,
				Close: closingPrice, Volume: baseVolume,
			}
		}

		result, err := obv.ComputeCandles(candles)
		if err != nil {
			fmt.Println("invalid contract feed:", err)
			return
		}
		// The resistance estimate excludes the candidate bar: using its own
		// High would introduce same-bar information into the breakout threshold.
		last := len(candles) - 1
		priorHigh := candles[0].High
		for _, candle := range candles[1:last] {
			if candle.High > priorHigh {
				priorHigh = candle.High
			}
		}
		if candles[last].Close <= priorHigh {
			continue // A strong OBV alone is not a price breakout.
		}
		breakouts++
		if result.Valid[last] && result.Values[last] > 0 {
			confirmed++
		} else {
			unconfirmed++
		}
	}
	fmt.Printf("contracts=%d breakouts=%d confirmed=%d unconfirmed=%d\n",
		contractCount, breakouts, confirmed, unconfirmed)
	// Output: contracts=100 breakouts=50 confirmed=25 unconfirmed=25
}

// ExampleStream_failoverRecovery models an exchange analytics service moving
// live computation from a primary process to a standby after a checkpoint.
//
// Scenario:
//   - An extremely large up-volume is followed by a one-unit contribution.
//   - During a replica handoff, a raw OBV scalar is insufficient for preserving
//     that small unit against a later equal-magnitude downward volume.
//
// Process:
//   - Push chronological bars, transport a versioned detached Snapshot.
//   - Restore on standby and continue with the next candle.
//   - Verify exact residual preservation; external storage must authenticate state.
func ExampleStream_failoverRecovery() {
	primary := obv.New()
	for _, bar := range []obv.Bar{
		{Open: 100, High: 100, Low: 100, Close: 100, Volume: 500},
		{Open: 101, High: 101, Low: 101, Close: 101, Volume: 1e16},
		{Open: 102, High: 102, Low: 102, Close: 102, Volume: 1},
	} {
		if _, _, err := primary.Push(bar); err != nil {
			fmt.Println("primary error:", err)
			return
		}
	}
	checkpoint := primary.Snapshot()
	standby := obv.New()
	// Authenticate the checkpoint and its external feed offset before Restore
	// in production; Config alone establishes neither authenticity nor freshness.
	if err := standby.Restore(checkpoint); err != nil {
		fmt.Println("checkpoint error:", err)
		return
	}
	continuation := obv.Bar{Open: 101, High: 101, Low: 101, Close: 101, Volume: 1e16}
	sourceValue, sourceValid, sourceErr := primary.Push(continuation)
	if sourceErr != nil || !sourceValid {
		fmt.Println("source continuation error:", sourceErr)
		return
	}
	value, valid, err := standby.Push(continuation)
	identical := valid && math.Float64bits(sourceValue) == math.Float64bits(value)
	fmt.Printf("restored-at=%d parity=%t value=%.0f count=%d err=%v\n",
		checkpoint.Count, identical, value, standby.Snapshot().Count, err)
	// Output: restored-at=3 parity=true value=1 count=4 err=<nil>
}

// ExampleStream_rejectCorruptFeed models a validation boundary around a
// fault-tolerant market-data gateway, without silently dropping bad OHLC geometry.
//
// Scenario:
//   - The gateway has an accepted seed and a first positive-volume advance.
//   - A late candle contains a plausible Close but impossible Open > High.
//   - Rejection leaves the exact last committed observation in the stream.
//
// Process:
//   - Save a checkpoint, reject the malformed candle with errors.Is.
//   - Replay a corrected observation and inspect cumulative volume and Count.
func ExampleStream_rejectCorruptFeed() {
	stream := obv.New()
	for _, bar := range []obv.Bar{{100, 100, 100, 100, 700}, {101, 101, 101, 101, 250}} {
		if _, _, err := stream.Push(bar); err != nil {
			fmt.Println("feed error:", err)
			return
		}
	}
	before := stream.Snapshot()
	_, _, rejected := stream.Push(obv.Bar{Open: 110, High: 104, Low: 99, Close: 103, Volume: 900})
	afterReject := stream.Snapshot()
	rolledBack := before.Algorithm == afterReject.Algorithm &&
		before.Config == afterReject.Config && before.Count == afterReject.Count
	for index := range before.FloatState {
		if math.Float64bits(before.FloatState[index]) != math.Float64bits(afterReject.FloatState[index]) {
			rolledBack = false
		}
	}
	value, valid, err := stream.Push(obv.Bar{Open: 102, High: 104, Low: 100, Close: 103, Volume: 900})
	fmt.Printf("domain=%t rolled-back=%t resumed=%t obv=%.0f count=%d err=%v\n",
		errors.Is(rejected, obv.ErrDomain), rolledBack, valid, value, stream.Snapshot().Count, err)
	// Output: domain=true rolled-back=true resumed=true obv=1150 count=3 err=<nil>
}

// ExampleStream_sessionBreak demonstrates that OBV is anchored to a chosen
// contiguous observation epoch, not to hidden timestamp or trading-calendar rules.
//
// Scenario:
//   - An upstream market-data service reports a session continuity break.
//   - Carrying last session's OBV into the next one would create false signed flow.
//   - The gateway explicitly resets and reseeds instead of inferring a gap.
//
// Process:
//   - Build a first session, reset at the authoritative session boundary.
//   - Feed the next epoch including a zero-volume closing-price change.
//   - Observe the new seed's zero balance and subsequent signed contribution.
func ExampleStream_sessionBreak() {
	stream := obv.New()
	for _, closingPrice := range []float64{99.0, 100.0, 101.0, 100.5} {
		_, _, err := stream.Push(obv.Bar{
			Open: closingPrice, High: closingPrice,
			Low: closingPrice, Close: closingPrice, Volume: 250,
		})
		if err != nil {
			fmt.Println("prior session error:", err)
			return
		}
	}
	previousSession := stream.Snapshot().Count
	stream.Reset()
	var first, final float64
	for index, bar := range []obv.Bar{
		{Open: 104, High: 104, Low: 104, Close: 104, Volume: math.MaxFloat64},
		{Open: 105, High: 105, Low: 105, Close: 105, Volume: 0},
		{Open: 103, High: 103, Low: 103, Close: 103, Volume: 180},
	} {
		value, _, err := stream.Push(bar)
		if err != nil {
			fmt.Println("session error:", err)
			return
		}
		if index == 0 {
			first = value
		}
		final = value
	}
	fmt.Printf("prior-bars=%d new-seed=%.0f new-balance=%.0f new-bars=%d\n", previousSession, first, final, stream.Snapshot().Count)
	// Output: prior-bars=4 new-seed=0 new-balance=-180 new-bars=3
}
