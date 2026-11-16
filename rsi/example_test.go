// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package rsi_test demonstrates production-shaped pipelines using only the public RSI API.
// Price fixtures are illustrative, deterministically ordered observations, not asserted historical exchange prints.
package rsi_test

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/gob"
	"errors"
	"fmt"
	"math"

	"github.com/lvlath/go/rsi"
)

// ExampleCompute runs a multi-period factor extraction over three liquid instruments.
//
// Scenario:
//   - A portfolio-risk service processes independently validated daily closing series
//     for a digital asset, a second digital asset, and a large equity-market ETF.
//   - It needs Wilder momentum features at 7, 14, and 28 changes for each instrument;
//     feature output is data for a downstream model, never a trading recommendation.
//
// Process:
//   - Supply realistic-scale, fixed fixture closes in chronological order.
//   - Compute an independent aligned Result for each instrument/period combination.
//   - Consume the last entry only when its corresponding validity bit is true.
//
// AI-Hints:
//   - Do not share recurrence state across instruments or periods.
//   - Treat the numbers as oscillator features, not buy/sell decisions.
func ExampleCompute() {
	universe := []struct {
		symbol string
		closes []float64
	}{
		{"BTC-USD", []float64{
			64040, 64215, 64100, 64590, 65010, 64800, 64680, 65220, 65540,
			65420, 65040, 64770, 64990, 65480, 65810, 66100, 65940, 65720,
			66360, 66620, 66210, 66090, 66580, 67020, 66910, 67280, 66890,
			67410, 67780, 67520, 67890, 68140, 67920, 68310, 68520,
		}},
		{"ETH-USD", []float64{
			3330, 3342, 3324, 3310, 3328, 3336, 3304, 3289, 3298,
			3317, 3325, 3302, 3280, 3268, 3275, 3290, 3282, 3260,
			3247, 3269, 3278, 3259, 3244, 3238, 3252, 3249, 3230,
			3242, 3236, 3225, 3231, 3220, 3208, 3217, 3211,
		}},
		{"SPY", []float64{
			541.20, 541.85, 542.10, 541.75, 542.60, 543.20, 542.95,
			543.80, 544.05, 543.70, 544.40, 544.15, 545.00, 545.40,
			544.90, 545.65, 546.20, 545.80, 546.55, 547.10, 546.85,
			547.40, 547.15, 547.95, 548.45, 548.05, 548.85, 549.20,
			548.75, 549.60, 549.40, 550.05, 550.40, 550.10, 550.80,
		}},
	}

	periods := []int{7, 14, 28}
	for _, asset := range universe {
		fmt.Print(asset.symbol)
		for _, period := range periods {
			result, err := rsi.ComputePeriod(asset.closes, period)
			if err != nil {
				fmt.Println(" calculation failed:", err)
				return
			}
			last := len(result.Values) - 1
			if !result.Valid[last] {
				fmt.Printf(" RSI(%d)=warming", period)
				continue
			}
			fmt.Printf(" RSI(%d)=%.2f", period, result.Values[last])
		}
		fmt.Println()
	}

	// Output:
	// BTC-USD RSI(7)=75.43 RSI(14)=72.65 RSI(28)=71.16
	// ETH-USD RSI(7)=33.16 RSI(14)=35.40 RSI(28)=36.83
	// SPY RSI(7)=77.72 RSI(14)=77.42 RSI(28)=77.52
}

// ExampleCompute_downtrend runs a collateral-mark deterioration audit with a rebound.
//
// Scenario:
//   - A fixed-price marks feed contains a prolonged decline followed by a partial recovery.
//   - The monitoring service stores oscillator features alongside independent price moves.
//   - A previously falling-only mean has RSI=0; later gains can lift it without resetting.
//
// Process:
//   - Run a 7-change Wilder calculation over fixed, validated collateral marks.
//   - Compare the initial valid sample and the latest valid sample, not warmup zeros.
//
// AI-Hints:
//   - RSI=0 is a valid feature when there are no positive differences; it is not an error.
func ExampleCompute_downtrend() {
	collateralMarks := []float64{
		840, 838, 837, 832, 828, 824, 819, 816, 813, 809, 805,
		803, 797, 792, 788, 790, 795, 793, 800, 802, 799,
	}

	const period = 7
	result, err := rsi.ComputePeriod(collateralMarks, period)
	if err != nil {
		fmt.Println("calculation failed:", err)
		return
	}
	last := len(result.Values) - 1
	fmt.Printf("marks=%d firstValid=%t initial=%.2f latest=%.2f latestValid=%t\n",
		len(collateralMarks), result.Valid[period], result.Values[period],
		result.Values[last], result.Valid[last],
	)

	// Output:
	// marks=21 firstValid=true initial=0.00 latest=40.99 latestValid=true
}

// ExampleCompute_flatMarket distinguishes a flat numeric oscillator from source freshness.
//
// Scenario:
//   - A settlement-mark reader sees many identical positive FX reference prices.
//   - Wilder RSI correctly returns 50, because neither gains nor losses are observed.
//   - The upstream service must separately check timestamps and data-source liveness;
//     RSI cannot determine whether the market is quiet or the feed has stopped.
//
// Process:
//   - Compute over a full valid flat history and count unchanged observations.
//   - Publish both oscillator readiness and feed-level repetition diagnostics.
//
// AI-Hints:
//   - Never turn flat-market RSI=50 into a claim that a feed is healthy.
func ExampleCompute_flatMarket() {
	fixes := []float64{
		1.0852, 1.0852, 1.0852, 1.0852, 1.0852, 1.0852,
		1.0852, 1.0852, 1.0852, 1.0852, 1.0852, 1.0852,
		1.0852, 1.0852, 1.0852, 1.0852, 1.0852, 1.0852,
	}
	result, err := rsi.ComputeDefault(fixes)
	if err != nil {
		fmt.Println("calculation failed:", err)
		return
	}

	unchanged := 0
	for i := 1; i < len(fixes); i++ {
		if fixes[i] == fixes[i-1] {
			unchanged++
		}
	}
	last := len(fixes) - 1
	fmt.Printf("fx valid=%t rsi=%.2f repeated=%d sourceFreshness=external\n",
		result.Valid[last], result.Values[last], unchanged,
	)

	// Output:
	// fx valid=true rsi=50.00 repeated=17 sourceFreshness=external
}

// ExampleStream_Snapshot demonstrates authenticated hot failover for a quote-analytics worker.
//
// Scenario:
//   - A primary worker checkpoints an already-warmed Wilder stream.
//   - The checkpoint is encoded and authenticated before transfer to a standby worker.
//   - The standby verifies the payload, decodes the original binary64 state, then
//     continues without replaying every prior close or changing the smoothing period.
//
// Process:
//   - Encode the detached Snapshot using a deterministic Go gob representation.
//   - Authenticate the *entire serialized payload* with an application-owned HMAC.
//   - Verify the MAC before Restore and compare subsequent RSI bits on both workers.
//
// Security notes:
//   - The literal key is for a reproducible example only; production keys belong in KMS.
//   - Snapshot.Config binds the period but is not integrity protection for mutable state.
//
// AI-Hints:
//   - Never authenticate only Config: Count and FloatState are outside that digest.
//   - Use a binary64-preserving encoding and verify the external MAC before Restore.
func ExampleStream_Snapshot() {
	primary, err := rsi.NewPeriod(7)
	if err != nil {
		fmt.Println("primary init failed:", err)
		return
	}
	past := []float64{
		64120.0, 64140.5, 64125.0, 64189.0, 64175.5, 64220.0,
		64211.5, 64280.0, 64257.0, 64309.0, 64340.0, 64295.0,
		64312.0, 64380.5, 64371.0, 64420.0,
	}
	for _, close := range past {
		if _, _, err := primary.Push(close); err != nil {
			fmt.Println("primary price failed:", err)
			return
		}
	}

	checkpoint := primary.Snapshot()
	var wire bytes.Buffer
	if err := gob.NewEncoder(&wire).Encode(checkpoint); err != nil {
		fmt.Println("checkpoint encode failed:", err)
		return
	}

	// Production systems source this secret from key management and rotate it.
	secret := []byte("example-only-checkpoint-hmac-key")
	sign := hmac.New(sha256.New, secret)
	if _, err := sign.Write(wire.Bytes()); err != nil {
		fmt.Println("checkpoint signature failed:", err)
		return
	}
	tag := sign.Sum(nil)

	verify := hmac.New(sha256.New, secret)
	if _, err := verify.Write(wire.Bytes()); err != nil {
		fmt.Println("verification failed:", err)
		return
	}
	authenticated := hmac.Equal(tag, verify.Sum(nil))
	if !authenticated {
		fmt.Println("checkpoint rejected")
		return
	}

	// A single-bit corruption of the same transmitted bytes cannot reuse the tag.
	tampered := append([]byte(nil), wire.Bytes()...)
	tampered[len(tampered)-1] ^= 1
	verifyBad := hmac.New(sha256.New, secret)
	if _, err := verifyBad.Write(tampered); err != nil {
		fmt.Println("tamper check failed:", err)
		return
	}
	tamperRejected := !hmac.Equal(tag, verifyBad.Sum(nil))

	var restored rsi.Snapshot
	if err := gob.NewDecoder(bytes.NewReader(wire.Bytes())).Decode(&restored); err != nil {
		fmt.Println("checkpoint decode failed:", err)
		return
	}
	standby, err := rsi.NewPeriod(7)
	if err != nil {
		fmt.Println("standby init failed:", err)
		return
	}
	if err := standby.Restore(restored); err != nil {
		fmt.Println("checkpoint restore failed:", err)
		return
	}

	continuation := []float64{64402.0, 64485.0, 64430.0, 64490.0, 64511.0, 64474.0}
	bitwise := true
	for _, close := range continuation {
		live, liveValid, liveErr := primary.Push(close)
		failover, failoverValid, failoverErr := standby.Push(close)
		if liveErr != nil || failoverErr != nil {
			fmt.Println("continuation failed:", liveErr, failoverErr)
			return
		}
		if math.Float64bits(live) != math.Float64bits(failover) || liveValid != failoverValid {
			bitwise = false
		}
	}
	fmt.Printf("checkpoint count=%d authenticated=%t tamperRejected=%t bitwise=%t\n",
		checkpoint.Count, authenticated, tamperRejected, bitwise,
	)

	// Output:
	// checkpoint count=16 authenticated=true tamperRejected=true bitwise=true
}

// ExampleStream_Push processes a deterministic level-one order-book feed with sequence gaps.
//
// Scenario:
//   - An exchange gateway receives best-bid/best-ask quotes for one BTC-USD stream.
//   - An invalid zero quote must not corrupt prior recurrence state.
//   - A missing exchange sequence requires an explicit Reset, because RSI does
//     not know timestamps and must not interpolate unseen bars or ticks.
//
// Process:
//   - Derive a representable midpoint for each ordered level-one event.
//   - Push into a preallocated state object, classify rejects via errors.Is.
//   - Reset only on a caller-detected sequence discontinuity and honor warmup.
//
// Performance notes:
//   - The Stream.Push call itself has zero heap allocations after New.
//   - Event construction, transport decoding, and diagnostic formatting are external.
//
// AI-Hints:
//   - Never share a single Stream between independent instruments.
//   - Treat a rejected price and a missing sequence as different operational events.
func ExampleStream_Push() {
	type quote struct {
		sequence uint64
		bid, ask float64
	}
	quotes := []quote{
		{1001, 64120.00, 64120.10}, {1002, 64121.10, 64121.20},
		{1003, 64119.90, 64120.00}, {1004, 64125.30, 64125.40},
		{1005, 64127.20, 64127.30}, {1006, 64124.70, 64124.80},
		{1007, 64129.40, 64129.50}, {1008, 64130.10, 64130.20},
		{1009, 64128.20, 64128.30}, {1010, 0, 0},
		{1011, 64134.00, 64134.10}, {1012, 64132.80, 64132.90},
		{1013, 64135.40, 64135.50}, {1014, 64137.10, 64137.20},
		{1017, 64136.80, 64136.90}, // Missing 1015, 1016: caller must reset.
		{1018, 64138.30, 64138.40}, {1019, 64138.10, 64138.20},
		{1020, 64140.20, 64140.30}, {1021, 64139.70, 64139.80},
		{1022, 64141.20, 64141.30}, {1023, 64142.50, 64142.60},
		{1024, 64144.10, 64144.20}, {1025, 64143.00, 64143.10},
		{1026, 64146.10, 64146.20}, {1027, 64148.80, 64148.90},
		{1028, 64147.50, 64147.60},
	}
	stream, err := rsi.NewPeriod(7)
	if err != nil {
		fmt.Println("stream init failed:", err)
		return
	}
	var prevSequence uint64
	accepted, rejected, resets, readyCount := 0, 0, 0, 0
	var latest float64
	for _, event := range quotes {
		if prevSequence != 0 && event.sequence != prevSequence+1 {
			stream.Reset()
			resets++
		}
		prevSequence = event.sequence

		// The midpoint is computed without summing two potentially large quotes.
		midpoint := event.bid + (event.ask-event.bid)/2
		value, valid, pushErr := stream.Push(midpoint)
		if pushErr != nil {
			if errors.Is(pushErr, rsi.ErrDomain) {
				rejected++
				continue
			}
			fmt.Println("unexpected input error:", pushErr)
			return
		}
		accepted++
		if valid {
			latest = value
			readyCount++
		}
	}
	fmt.Printf("gateway accepted=%d rejected=%d resets=%d ready=%d lastRSI=%.2f\n",
		accepted, rejected, resets, readyCount, latest,
	)

	// Output:
	// gateway accepted=25 rejected=1 resets=1 ready=11 lastRSI=78.99
}

// ExampleLatest distinguishes short-history onboarding from a fully seeded factor.
// It shows a terminal observation API that still preserves readiness and validation.
//
// Scenario:
//   - A monitoring service onboards one newly listed instrument beside an established one.
//   - A model may omit an unready factor, but must not invent an RSI=0 feature.
//
// AI-Hints:
//   - Latest delegates to full Compute and allocates O(n) output memory; use Push online.
func ExampleLatest() {
	universe := []struct {
		symbol string
		closes []float64
	}{
		{"NEW", []float64{12.1, 12.4, 12.2, 12.5}},
		{"ESTABLISHED", []float64{
			102.1, 102.5, 102.3, 102.8, 103.2, 103.0, 103.4, 103.1,
			103.8, 103.5, 103.9, 104.2, 103.7, 104.0, 104.4, 104.1,
			104.5, 104.8,
		}},
	}
	for _, asset := range universe {
		value, valid, err := rsi.Latest(asset.closes, rsi.Options{})
		if err != nil {
			fmt.Println("factor failed:", err)
			return
		}
		if !valid {
			fmt.Printf("%s factor=unavailable (warming)\n", asset.symbol)
			continue
		}
		fmt.Printf("%s factor=%.2f valid=%t\n", asset.symbol, value, valid)
	}

	// Output:
	// NEW factor=unavailable (warming)
	// ESTABLISHED factor=71.79 valid=true
}
