// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package atr_test demonstrates source admission, range review, and exact restart pipelines.
// All scenarios are built from Go literals; no external data or file loading is required.
package atr_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/lvlath/go/atr"
)

// ExampleComputeDefault_historicalRangeAudit builds a daily range report from fixed Go inputs.
// It preserves warmup TR and separates the session's own range from the prior-close gap.
//
// Implementation:
//   - Stage 1: construct twenty dated HLC records from caller-owned Go literals.
//   - Stage 2: compute ATR(14) and retain the source dates alongside aligned results.
//   - Stage 3: report readiness, the last observation, and an explanatory gap witness.
//
// Behavior highlights:
//   - The fixed regression history demonstrates a pipeline, not a verified market tape.
//
// Inputs:
//   - dailyBars: twenty chronological, code-defined records in consistent price units.
//
// Returns:
//   - A deterministic report suitable for checking a feature-ingestion pipeline.
//
// Errors:
//   - Canonical computation errors are printed before returning; input construction cannot fail.
//
// Determinism:
//   - Input dates and numeric values are pinned; no clock or network is consulted.
//
// Complexity:
//   - Time O(n*W), Space O(n+W), including detached input and output columns.
//
// Notes:
//   - Only Valid gates ATR; TR remains a useful observation during warmup.
//
// AI-Hints:
//   - Keep the source adjustment basis consistent; do not mix adjusted close with raw H/L.
func ExampleComputeDefault_historicalRangeAudit() {
	// Build independent Go inputs and retain their session labels outside the indicator.
	bars := dailyBars()
	high, low, closed := dailyChannels(bars)
	// The name explicitly selects fourteen observations, not fourteen calendar days.
	result, err := atr.ComputeDefault(context.Background(), high, low, closed)
	if err != nil {
		fmt.Println(err)
		return
	}
	firstReady := atr.DefaultPeriod - 1
	last := len(bars) - 1
	fmt.Printf("daily bars=%d first-ready=%s warmup=%d\n", len(bars), bars[firstReady].date, firstReady)
	fmt.Printf("%s TR=%.6f ATR14=%.6f ready=%v\n", bars[last].date, result.TR[last], result.ATR[last], result.Valid[last])
	// This session traded below the preceding close: TR exceeds its own H-L range.
	const gapIndex = 16
	fmt.Printf("%s candle-range=%.6f prior-close-range=%.6f\n", bars[gapIndex].date, high[gapIndex]-low[gapIndex], result.TR[gapIndex])
	// Output:
	// daily bars=20 first-ready=2026-11-12 warmup=13
	// 2026-11-20 TR=9.610000 ATR14=6.904444 ready=true
	// 2026-11-17 candle-range=10.358500 prior-close-range=11.870000
}

// ExampleStream_Push_priorBaselineMonitor measures expansion against a previously ready ATR.
// The current shock does not increase its own denominator or review threshold.
//
// Implementation:
//   - Stage 1: construct one single-owner default stream and a local prior Value.
//   - Stage 2: accept each closed source bar, then compare TR to the prior ready baseline.
//   - Stage 3: publish review events and only then advance the local prior baseline.
//
// Behavior highlights:
//   - The multiplier is application review policy, not an ATR option or trade instruction.
//
// Inputs:
//   - The same code-defined twenty-session daily history; dimensionless reviewMultiplier=1.5.
//
// Returns:
//   - Two dated range-review events and the number of accepted observations.
//
// Errors:
//   - Stop before advancing the local baseline if construction or Push fails.
//
// Determinism:
//   - Only the accepted prefix influences each comparison.
//
// Complexity:
//   - Indicator work O(n*W), indicator state O(W); the Go scenario owns O(n) input data.
//
// Notes:
//   - Formatting and input construction may allocate; zero-allocation claims concern Push only.
//
// AI-Hints:
//   - A valid zero baseline needs its own policy; do not divide by it or silently mask it.
func ExampleStream_Push_priorBaselineMonitor() {
	bars := dailyBars()
	stream, err := atr.NewDefault()
	if err != nil {
		fmt.Println(err)
		return
	}
	const reviewMultiplier = 1.5
	ctx := context.Background()
	var previous atr.Value
	for _, bar := range bars {
		// Push accepts the numeric observation, not its timestamp or availability history.
		value, err := stream.Push(ctx, bar.high, bar.low, bar.close)
		if err != nil {
			fmt.Println(err)
			return
		}
		// This fixture is far from overflow; production policy must bound its own products.
		if previous.Valid && previous.ATR > 0 && value.TR > reviewMultiplier*previous.ATR {
			fmt.Printf("review %s TR=%.6f prior-ATR=%.6f\n", bar.date, value.TR, previous.ATR)
		}
		previous = value
	}
	fmt.Printf("accepted=%d\n", stream.Snapshot().Count)
	// Output:
	// review 2026-11-17 TR=11.870000 prior-ATR=6.187774
	// review 2026-11-19 TR=10.026300 prior-ATR=6.440172
	// accepted=20
}

// ExampleStream_Restore_hotFailover resumes an unfinished seed without replaying accepted events.
// The checkpoint envelope keeps source identity and the first unconsumed record beside the state.
//
// Implementation:
//   - Stage 1: accept five bars on the primary and serialize a typed checkpoint envelope.
//   - Stage 2: validate source identity/cursor and restore into an equally configured standby.
//   - Stage 3: resume the suffix and compare every output bit with the uninterrupted primary.
//
// Behavior highlights:
//   - Count=5 precedes ATR14 readiness; SeedSum, not a rounded seed mean, preserves information.
//
// Inputs:
//   - A named in-code scenario revision and typed in-memory JSON preserving integer Count.
//
// Returns:
//   - The handoff count, cursor, compared suffix length, and exact continuation parity.
//
// Errors:
//   - Serialization, construction, Restore, and Push failures stop publication.
//
// Determinism:
//   - Both nodes accept exactly the same suffix once, in the original source order.
//
// Complexity:
//   - Time O(n*W+D*W); two fixed indicator states plus fixture/JSON storage.
//
// Notes:
//   - Durable storage must atomically save the envelope; JSON is not authentication.
//
// AI-Hints:
//   - Do not convert Count through map[string]any float64 or persist a mismatched cursor.
func ExampleStream_Restore_hotFailover() {
	bars := dailyBars()
	primary, err := atr.NewDefault()
	if err != nil {
		fmt.Println(err)
		return
	}
	const handoff = 5
	const instrument = "example/daily/price-units"
	const revision = "hlc-literals/v1"
	ctx := context.Background()
	for _, bar := range bars[:handoff] {
		if _, err := primary.Push(ctx, bar.high, bar.low, bar.close); err != nil {
			fmt.Println(err)
			return
		}
	}
	// Save state and source cursor together; a database transaction belongs to the service.
	payload, err := json.Marshal(replayEnvelope{Instrument: instrument, Revision: revision, NextIndex: handoff, State: primary.Snapshot()})
	if err != nil {
		fmt.Println(err)
		return
	}
	var checkpoint replayEnvelope
	if err := json.Unmarshal(payload, &checkpoint); err != nil {
		fmt.Println(err)
		return
	}
	if checkpoint.Instrument != instrument || checkpoint.Revision != revision || checkpoint.NextIndex != handoff || checkpoint.State.Count != uint64(handoff) {
		fmt.Println("source checkpoint mismatch")
		return
	}
	standby, err := atr.NewPeriod(atr.DefaultPeriod)
	if err != nil {
		fmt.Println(err)
		return
	}
	if err := standby.Restore(checkpoint.State); err != nil {
		fmt.Println(err)
		return
	}
	matching := true
	for _, bar := range bars[checkpoint.NextIndex:] {
		left, err := primary.Push(ctx, bar.high, bar.low, bar.close)
		if err != nil {
			fmt.Println(err)
			return
		}
		right, err := standby.Push(ctx, bar.high, bar.low, bar.close)
		if err != nil {
			fmt.Println(err)
			return
		}
		matching = matching && sameValueBits(left, right)
	}
	fmt.Printf("handoff=%d seed-retained=%v next=%s\n", checkpoint.State.Count, checkpoint.State.SeedSum != "0", bars[checkpoint.NextIndex].date)
	fmt.Printf("compared=%d bitwise-equal=%v accepted=%d\n", len(bars)-handoff, matching, standby.Snapshot().Count)
	// Output:
	// handoff=5 seed-retained=true next=2026-11-02
	// compared=15 bitwise-equal=true accepted=20
}

// ExampleStream_Push_rejectedCandleRetry quarantines a malformed feed row before retrying it.
// A rejected event must change neither the indicator state nor the application's source cursor.
//
// Implementation:
//   - Stage 1: build a canonical batch reference and a separate online stream.
//   - Stage 2: inject one out-of-range close, classify ErrBar, and verify unchanged state.
//   - Stage 3: submit the corrected record once and compare the complete replay.
//
// Behavior highlights:
//   - An invalid feed row is not silently clamped to its high or dropped from the history.
//
// Inputs:
//   - Twenty Go-literal bars, with one deliberately corrupted transport payload.
//
// Returns:
//   - A dated quarantine witness and a count/parity check after successful retry.
//
// Errors:
//   - Only the explicitly injected ErrBar is handled as expected; other failures stop the run.
//
// Determinism:
//   - The corrected source record occupies its original index exactly once.
//
// Complexity:
//   - Time O(n*W); O(n+W) storage including the batch audit reference.
//
// Notes:
//   - Correcting an already accepted historical bar instead requires earlier-checkpoint replay.
//
// AI-Hints:
//   - Never advance the source offset on a failed Push merely because a Value was returned.
func ExampleStream_Push_rejectedCandleRetry() {
	bars := dailyBars()
	high, low, closed := dailyChannels(bars)
	ctx := context.Background()
	reference, err := atr.ComputeDefault(ctx, high, low, closed)
	if err != nil {
		fmt.Println(err)
		return
	}
	stream, err := atr.NewDefault()
	if err != nil {
		fmt.Println(err)
		return
	}
	const damagedIndex = 14
	matching := true
	for i, bar := range bars {
		if i == damagedIndex {
			before := stream.Snapshot()
			bad, rejected := stream.Push(ctx, bar.high, bar.low, bar.high+1)
			if !errors.Is(rejected, atr.ErrBar) || bad != (atr.Value{}) || stream.Snapshot() != before {
				fmt.Println("unexpected quarantine result")
				return
			}
			fmt.Printf("quarantine=%s state-unchanged=true retry-index=%d\n", bar.date, i)
		}
		value, err := stream.Push(ctx, bar.high, bar.low, bar.close)
		if err != nil {
			fmt.Println(err)
			return
		}
		matching = matching && sameValueBits(value, atr.Value{TR: reference.TR[i], ATR: reference.ATR[i], Valid: reference.Valid[i]})
	}
	fmt.Printf("accepted=%d batch-parity=%v\n", stream.Snapshot().Count, matching)
	// Output:
	// quarantine=2026-11-13 state-unchanged=true retry-index=14
	// accepted=20 batch-parity=true
}

// ExampleComputePeriod_signedSpreadReview measures a signed spread without a positive-price fiction.
// Sixteen engineered observations cross zero and include both close-to-close gaps and a separate flat bar.
//
// Implementation:
//   - Stage 1: retain signed finite HLC in the same spread-price units.
//   - Stage 2: compute an explicitly selected four-observation Wilder average.
//   - Stage 3: distinguish a gap range from a flat candle with genuinely zero True Range.
//
// Behavior highlights:
//   - These are synthetic stress observations, not claimed historical futures quotations.
//
// Inputs:
//   - Fixed signed spread candles; Period=4 is explicit application policy.
//
// Returns:
//   - First readiness, the flat-bar gap witness, and the final smoothed range.
//
// Errors:
//   - Every source candle still obeys low<=close<=high; invalid geometry is not allowed.
//
// Determinism:
//   - Literal binary-exact integer prices and fixed order produce stable printed output.
//
// Complexity:
//   - Time O(n*W), Space O(n+W).
//
// Notes:
//   - ATR is a nonnegative range in the input units, not a signed percentage of close.
//
// AI-Hints:
//   - A zero or negative close is valid; it must not become a normalization denominator.
func ExampleComputePeriod_signedSpreadReview() {
	high := []float64{-1, 1, 0, 2, 3, 4, 2, 1, 0, -1, -2, 0, 2, 3, 1, 0}
	low := []float64{-3, -2, -2, 0, 1, 2, 0, -1, -2, -3, -4, -2, 0, 1, -1, 0}
	closed := []float64{-2, 0, -1, 1, 2, 3, 1, 0, -1, -2, -3, -1, 1, 2, 0, 0}
	result, err := atr.ComputePeriod(context.Background(), high, low, closed, 4)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("signed-bars=%d first-ready=3 seed=%.6f\n", len(high), result.ATR[3])
	// Index 11 contains a gap from -3 to the current [-2,0] interval.
	fmt.Printf("gap-index=11 TR=%.6f candle-range=%.6f\n", result.TR[11], high[11]-low[11])
	// The last candle is flat and repeats the previous close: zero TR is a real observation.
	fmt.Printf("flat-tail TR=%.6f ATR=%.6f ready=%v\n", result.TR[15], result.ATR[15], result.Valid[15])
	// Output:
	// signed-bars=16 first-ready=3 seed=2.500000
	// gap-index=11 TR=3.000000 candle-range=2.000000
	// flat-tail TR=0.000000 ATR=1.906680 ready=true
}

// ExampleComputePeriod_zeroRangeIsReady separates unavailable warmup from a valid zero measurement.
// A sixteen-bar unchanged source is not missing data merely because its prices do not move.
//
// Implementation:
//   - Stage 1: retain all sixteen flat source observations without filtering duplicates.
//   - Stage 2: compute ATR14 and a period-one range-only view through the same facade.
//   - Stage 3: publish readiness and explicitly abstain from range normalization at zero.
//
// Behavior highlights:
//   - Repeated observed prices are different from duplicate delivery of the same event.
//
// Inputs:
//   - A deliberately constant price fixture, not a claim about an observed instrument.
//
// Returns:
//   - Warmup/ready distinctions and confirmation that period one preserves every TR bit.
//
// Errors:
//   - Both complete computations are checked before the result is consumed.
//
// Determinism:
//   - All generated payload zeros are canonical positive zero.
//
// Complexity:
//   - Time O(n*W), Space O(n+W) for the two owning results.
//
// Notes:
//   - The consumer chooses an explicit no-normalization policy; the package never divides by ATR.
//
// AI-Hints:
//   - Read Valid, not ATR!=0; a short history can be successful but still unready.
func ExampleComputePeriod_zeroRangeIsReady() {
	prices := make([]float64, 16)
	for i := range prices {
		prices[i] = 5
	}
	ctx := context.Background()
	smoothed, err := atr.ComputeDefault(ctx, prices, prices, prices)
	if err != nil {
		fmt.Println(err)
		return
	}
	ranges, err := atr.ComputePeriod(ctx, prices, prices, prices, 1)
	if err != nil {
		fmt.Println(err)
		return
	}
	identity := true
	for i := range ranges.TR {
		identity = identity && ranges.Valid[i] && sameValueBits(atr.Value{TR: ranges.TR[i], ATR: ranges.ATR[i], Valid: true}, atr.Value{TR: ranges.TR[i], ATR: ranges.TR[i], Valid: true})
	}
	fmt.Printf("warmup=%v first-ready=%v ATR=%.1f\n", smoothed.Valid[12], smoothed.Valid[13], smoothed.ATR[13])
	fmt.Printf("range-only-identity=%v normalize=%v\n", identity, smoothed.Valid[15] && smoothed.ATR[15] > 0)
	// Output:
	// warmup=false first-ready=true ATR=0.0
	// range-only-identity=true normalize=false
}
