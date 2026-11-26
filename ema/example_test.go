// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ema_test demonstrates ingestion, readiness, incident review, and replay workflows.
// Data are deterministic synthetic fixtures with explicit units, not claimed market records.
// AI-Hints: keep source ordering, event identity, and raw-limit checks outside the EMA kernel.
package ema_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"

	"github.com/lvlath/go/ema"
)

// ExampleCompute_keltnerCenterline builds a closed-bar feature table without consulting the forming bar.
// The centerline and readiness retain the source index alignment for a later ATR join.
//
// Implementation:
//   - Stage 1: select the 24 closed observations from a caller-validated one-minute grid.
//   - Stage 2: compute an explicitly SMA-seeded EMA(20).
//   - Stage 3: consume only ready rows and retain their original timestamps.
//
// Behavior highlights:
//   - Setup, canonical public operations, and meaningful result consumption are all executable.
//
// Inputs:
//   - 25 illustrative closes, one explicitly unfinished tail bar, and a 20-observation period.
//
// Returns:
//   - Five usable centerline rows; two tail rows and their residuals are printed.
//
// Errors:
//   - Every library or codec error is checked before dependent state is used.
//
// Determinism:
//   - Fixed synthetic data and exact index/sequence traversal; output uses no wall clock.
//
// Complexity:
//   - Time O(n*W), Space O(n+W) for n closed bars and W=34.
//
// Notes:
//   - ATR and channel widths are intentionally not fabricated by this example.
//
// AI-Hints:
//   - Do not trim warmup values without making the same selection on every aligned feature.
func ExampleCompute_keltnerCenterline() {
	const period = 20
	const closedBars = 24
	const sessionStartMinute = 9*60 + 30
	// The upstream feed supplies an ordered one-minute grid with one still-forming tail bar.
	closes := []float64{
		184.25, 184.75, 183.50, 185.00, 185.25, 184.50, 186.00, 186.25,
		185.75, 187.00, 186.50, 187.25, 188.00, 187.50, 188.25, 189.00,
		188.50, 189.25, 190.00, 189.50, 190.25, 191.00, 190.75, 192.00,
		195.50,
	}
	// Bar closure is caller policy: the unfinished observation is explicitly excluded.
	result, err := ema.ComputeSMA(context.Background(), closes[:closedBars], period)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	// Build only usable feature rows; never confuse warmup +0 with a real zero baseline.
	featureRows, firstReady := 0, -1
	for i, ready := range result.Valid {
		if !ready {
			continue
		}
		if firstReady < 0 {
			firstReady = i
		}
		featureRows++
		// Reporting the tail keeps the example compact while processing all admitted bars.
		if i >= closedBars-2 {
			minute := sessionStartMinute + i
			fmt.Printf("bar=%d time=%02d:%02d close=%.2f center=%.6f residual=%.6f\n",
				i, minute/60, minute%60, closes[i], result.Values[i], closes[i]-result.Values[i])
		}
	}
	// Original indexes can now be joined with separately validated ATR readiness masks.
	fmt.Printf("closed=%d excluded-forming=%d first-ready=%d feature-rows=%d\n", closedBars, len(closes)-closedBars, firstReady, featureRows)
	// Output:
	// bar=22 time=09:52 close=190.75 center=187.807062 residual=2.942938
	// bar=23 time=09:53 close=192.00 center=188.206389 residual=3.793611
	// closed=24 excluded-forming=1 first-ready=19 feature-rows=5
}

// ExampleStream_Push_tickNoise filters a quote replay after explicit feed-quality admission.
// Only successfully accepted events advance the source checkpoint and the EMA state.
//
// Implementation:
//   - Stage 1: construct the replay fixture and an immediately ready EMA(20).
//   - Stage 2: reject duplicates/crossed quotes, require consecutive sequences, and push midpoints.
//   - Stage 3: report accepted count, latest baseline, and unsmoothed residual outside the loop.
//
// Behavior highlights:
//   - Setup, canonical public operations, and meaningful result consumption are all executable.
//
// Inputs:
//   - 18 quote events with sequence IDs, positive bid/ask prices, one duplicate, and one repair.
//
// Returns:
//   - 16 accepted observations and a deterministic latest-baseline diagnostic.
//
// Errors:
//   - Every library or codec error is checked before dependent state is used.
//
// Determinism:
//   - Fixed synthetic data and exact index/sequence traversal; output uses no wall clock.
//
// Complexity:
//   - Time O(n*W); stream Space O(W), with O(n) replay-fixture storage outside the hot path.
//
// Notes:
//   - First seeding is deliberate; it is not a shortcut for an unfinished SMA seed.
//   - Midpoint construction and quote admission are application rules, not EMA features.
//
// AI-Hints:
//   - Never replay an acknowledged sequence into the filter merely because its numeric price is unchanged.
func ExampleStream_Push_tickNoise() {
	const period = 20
	const initialSequence = uint64(9000)
	// Fixture allocation and stream construction occur before the processing loop.
	ticks := quoteReplay()
	stream, err := ema.NewFirst(period)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	ctx := context.Background()
	lastSequence := initialSequence
	accepted, duplicates, quarantined := 0, 0, 0
	var latest ema.Value
	var lastMidpoint float64
	for _, tick := range ticks {
		// Replayed events must not contribute a second observation to exponential memory.
		if tick.sequence <= lastSequence {
			duplicates++
			continue
		}
		// This application requires positive, finite, non-crossed quotes; EMA itself is signed.
		if math.IsNaN(tick.bid) || math.IsNaN(tick.ask) || math.IsInf(tick.bid, 0) || math.IsInf(tick.ask, 0) || tick.bid <= 0 || tick.ask <= 0 || tick.bid > tick.ask {
			quarantined++
			continue
		}
		// A gap cannot be silently compressed into a regular observation cadence.
		if tick.sequence != lastSequence+1 {
			fmt.Println("error: source gap requires reconciliation")
			return
		}
		// Midpoint construction is an explicit upstream transform of this modest-price fixture.
		midpoint := tick.bid/2 + tick.ask/2
		latest, err = stream.Push(ctx, midpoint)
		if err != nil {
			fmt.Println("error:", err)
			return
		}
		// Acknowledge only a committed observation; no Snapshot or formatting occurs on success.
		lastSequence, lastMidpoint = tick.sequence, midpoint
		accepted++
	}
	// Allocation tests measure Push separately; this reporting step is outside that boundary.
	fmt.Printf("feed=%d accepted=%d duplicates=%d quarantined=%d last-sequence=%d\n", len(ticks), accepted, duplicates, quarantined, lastSequence)
	fmt.Printf("ready=%t midpoint=%.6f baseline=%.6f residual=%.6f\n", latest.Valid, lastMidpoint, latest.Value, lastMidpoint-latest.Value)
	// Output:
	// feed=18 accepted=16 duplicates=1 quarantined=1 last-sequence=9016
	// ready=true midpoint=62106.000000 baseline=62105.294956 residual=0.705044
}

// ExampleStream_Restore_warmupFailover resumes a partially seeded replica with an application replay checkpoint.
// Every resumed output matches uninterrupted primary processing, not just the final scalar.
//
// Implementation:
//   - Stage 1: checkpoint EMA(14) after five accepted values, together with source identity.
//   - Stage 2: decode a typed record and restore a same-options replica.
//   - Stage 3: skip one duplicated event and verify all resumed values and final state.
//
// Behavior highlights:
//   - Setup, canonical public operations, and meaningful result consumption are all executable.
//
// Inputs:
//   - 20 illustrative closes; explicit instrument and source-sequence metadata.
//
// Returns:
//   - Exact seed retention, resumed readiness at sequence 1014, and bit-identical continuation.
//
// Errors:
//   - Every library or codec error is checked before dependent state is used.
//
// Determinism:
//   - Fixed synthetic data and exact index/sequence traversal; output uses no wall clock.
//
// Complexity:
//   - Time O(n*W+D*W), Space O(n+D+W), D<=660; serialization is outside Push.
//
// Notes:
//   - JSON preserves typed scalar values but does not authenticate persistence.
//   - Production storage must commit the source checkpoint and EMA snapshot as one record.
//
// AI-Hints:
//   - Persisting EMA state without the last accepted event ID can double-count data after failover.
func ExampleStream_Restore_warmupFailover() {
	const period = 14
	const checkpointCount = 5
	const firstSequence = uint64(1001)
	const instrument = "DEMO/CLOSE"
	closes := failoverHistory()
	ctx := context.Background()
	primary, err := ema.NewSMA(period)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	// The primary stops after five accepted samples: no SMA result is ready yet.
	for _, value := range closes[:checkpointCount] {
		if _, err = primary.Push(ctx, value); err != nil {
			fmt.Println("error:", err)
			return
		}
	}
	// Store source identity and numerical state in one application-owned checkpoint record.
	envelope := checkpointEnvelope{Instrument: instrument, LastSequence: firstSequence + checkpointCount - 1, EMA: primary.Snapshot()}
	wire, err := json.Marshal(envelope)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	// Typed decoding preserves uint64 counts; JSON transport alone is not authentication.
	var persisted checkpointEnvelope
	if err = json.Unmarshal(wire, &persisted); err != nil {
		fmt.Println("error:", err)
		return
	}
	// The storage owner must also authenticate the record; these are coherence checks only.
	if persisted.Instrument != instrument || persisted.LastSequence < firstSequence || persisted.EMA.Count != persisted.LastSequence-firstSequence+1 {
		fmt.Println("error: incoherent replay checkpoint")
		return
	}
	replica, err := ema.NewSMA(period)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	if err = replica.Restore(persisted.EMA); err != nil {
		fmt.Println("error:", err)
		return
	}
	lastSequence := persisted.LastSequence
	duplicates, resumed, readyCount := 0, 0, 0
	firstReadySequence := uint64(0)
	identical := true
	var latest ema.Value
	// Replay deliberately starts one event before the next required sequence.
	for i := checkpointCount - 1; i < len(closes); i++ {
		sequence := firstSequence + uint64(i)
		if sequence <= lastSequence {
			duplicates++
			continue
		}
		if sequence != lastSequence+1 {
			fmt.Println("error: replay gap requires reconciliation")
			return
		}
		latest, err = replica.Push(ctx, closes[i])
		if err != nil {
			fmt.Println("error:", err)
			return
		}
		// The uninterrupted primary is a parity witness, not part of replica state.
		expected, primaryErr := primary.Push(ctx, closes[i])
		if primaryErr != nil {
			fmt.Println("error:", primaryErr)
			return
		}
		identical = identical && latest.Valid == expected.Valid && math.Float64bits(latest.Value) == math.Float64bits(expected.Value)
		lastSequence = sequence
		resumed++
		if latest.Valid {
			if readyCount == 0 {
				firstReadySequence = sequence
			}
			readyCount++
		}
	}
	// Final state equality includes the cleared seed, count, and complete construction options.
	identical = identical && primary.Snapshot() == replica.Snapshot()
	fmt.Printf("checkpoint count=%d current=%.0f exact-seed=%t\n", persisted.EMA.Count, persisted.EMA.Current, persisted.EMA.SeedSum != "0")
	fmt.Printf("replay duplicates=%d resumed=%d first-ready-sequence=%d ready-count=%d\n", duplicates, resumed, firstReadySequence, readyCount)
	fmt.Printf("accepted=%d baseline=%.6f uninterrupted-parity=%t\n", replica.Snapshot().Count, latest.Value, identical)
	// Output:
	// checkpoint count=5 current=0 exact-seed=true
	// replay duplicates=1 resumed=15 first-ready-sequence=1014 ready-count=7
	// accepted=20 baseline=202.422179 uninterrupted-parity=true
}

// ExampleComputeFirst_latencyIncidentReview constructs a previous-only baseline for service incident triage.
// Sustained deviations become human-review candidates rather than automatic control actions.
//
// Implementation:
//   - Stage 1: provide 24 completed one-second mean-latency windows in milliseconds.
//   - Stage 2: compute an explicitly First-seeded EMA(9).
//   - Stage 3: compare each window with the preceding baseline and report a three-window excess run.
//
// Behavior highlights:
//   - Setup, canonical public operations, and meaningful result consumption are all executable.
//
// Inputs:
//   - Illustrative window means; explicit 8 ms excess and three-consecutive-window review policy.
//
// Returns:
//   - One review candidate and the final mean-latency baseline.
//
// Errors:
//   - Every library or codec error is checked before dependent state is used.
//
// Determinism:
//   - Fixed synthetic data and exact index/sequence traversal; output uses no wall clock.
//
// Complexity:
//   - Time O(n*W), Space O(n+W) for the batch and O(1) consumer state.
//
// Notes:
//   - The thresholds are illustrative application policy, not recommended operational limits.
//   - Smoothing window means does not compute a pooled percentile or a confidence interval.
//
// AI-Hints:
//   - Use baseline[i-1], not baseline[i], when the assessment must exclude the current observation.
func ExampleComputeFirst_latencyIncidentReview() {
	const period = 9
	const excessMilliseconds = 8.0
	const consecutiveWindows = 3
	// Each input is a completed one-second window's mean request latency in milliseconds.
	latency := []float64{
		12, 11.5, 12.5, 11.75, 12, 12.25, 11.5, 12, 12.5, 12, 12.25, 11.75,
		21, 23, 24, 25, 23, 22, 14, 13, 12.5, 12.25, 12, 11.75,
	}
	baseline, err := ema.ComputeFirst(context.Background(), latency, period)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	streak, reviewPoints := 0, 0
	// Compare against the previous baseline so the assessed sample does not dilute its reference.
	for i := 1; i < len(latency); i++ {
		if !baseline.Valid[i-1] {
			continue
		}
		excess := latency[i] - baseline.Values[i-1]
		if excess >= excessMilliseconds {
			streak++
		} else {
			streak = 0
		}
		// One notification per run avoids treating every continuing deviation as a new incident.
		if streak == consecutiveWindows {
			reviewPoints++
			fmt.Printf("review-window=%d previous-baseline-ms=%.6f excess-ms=%.6f\n", i, baseline.Values[i-1], excess)
		}
	}
	// EMA tracks window means, not a pooled request quantile or a confidence interval.
	fmt.Printf("windows=%d review-points=%d final-baseline-ms=%.6f\n", len(latency), reviewPoints, baseline.Values[len(latency)-1])
	// Output:
	// review-window=14 previous-baseline-ms=15.650134 excess-ms=8.349866
	// windows=24 review-points=1 final-baseline-ms=14.382114
}

// ExampleStream_Push_rejectedObservation repairs a telemetry event before advancing its accepted-history state.
// Corrupt and canceled attempts leave the same source position available for a correct retry.
//
// Implementation:
//   - Stage 1: construct an SMA-seeded load dashboard with an independent raw review limit.
//   - Stage 2: demonstrate nonfinite and canceled attempts, then submit the corrected event.
//   - Stage 3: compare final output with uninterrupted batch processing and report raw excursions.
//
// Behavior highlights:
//   - Setup, canonical public operations, and meaningful result consumption are all executable.
//
// Inputs:
//   - 16 illustrative one-second kW readings; a corrupt event and canceled retry at one index.
//
// Returns:
//   - Unchanged rollback snapshots, 16 accepted readings, and two raw-limit review points.
//
// Errors:
//   - Every library or codec error is checked before dependent state is used.
//
// Determinism:
//   - Fixed synthetic data and exact index/sequence traversal; output uses no wall clock.
//
// Complexity:
//   - Time O(n*W), Space O(n+W) including the batch parity witness; the stream itself stays O(W).
//
// Notes:
//   - A smoothed dashboard is not a safety interlock and must not hide raw excursions.
//   - The repaired event is explicitly supplied; missing observations are never replaced by zero.
//
// AI-Hints:
//   - Do not increment a source checkpoint after Push returns an error.
func ExampleStream_Push_rejectedObservation() {
	const period = 5
	const repairIndex = 7
	const rawReviewLimit = 50.0
	// Ordered one-second equipment-load readings, in kW; this is an advisory dashboard fixture.
	load := []float64{42, 42.5, 43, 42.75, 44, 46, 48, 49.5, 51, 50.5, 49.75, 48.5, 47.75, 46.5, 45.5, 44.75}
	stream, err := ema.NewSMA(period)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	ctx := context.Background()
	rejected, canceledCount, rawReviewCount := 0, 0, 0
	unchanged := true
	var latest ema.Value
	for i, reading := range load {
		if i == repairIndex {
			// The corrupt event and a canceled retry both refer to the same uncommitted index.
			before := stream.Snapshot()
			failed, badErr := stream.Push(ctx, math.NaN())
			if !errors.Is(badErr, ema.ErrNonFinite) || failed != (ema.Value{}) {
				fmt.Println("error: corrupt input was not rejected")
				return
			}
			rejected++
			unchanged = unchanged && before == stream.Snapshot()
			canceled, cancel := context.WithCancel(ctx)
			cancel()
			failed, canceledErr := stream.Push(canceled, reading)
			if !errors.Is(canceledErr, context.Canceled) || failed != (ema.Value{}) {
				fmt.Println("error: canceled retry was not rejected")
				return
			}
			canceledCount++
			unchanged = unchanged && before == stream.Snapshot()
		}
		// The upstream repair supplies the actual observation; no missing event is replaced by zero.
		latest, err = stream.Push(ctx, reading)
		if err != nil {
			fmt.Println("error:", err)
			return
		}
		// This independent raw review policy is not weakened by smoothing or warmup.
		if reading > rawReviewLimit {
			rawReviewCount++
		}
	}
	// Batch comparison confirms that unsuccessful attempts never became real observations.
	batch, err := ema.ComputeSMA(ctx, load, period)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	identical := math.Float64bits(latest.Value) == math.Float64bits(batch.Values[len(load)-1]) && latest.Valid == batch.Valid[len(load)-1]
	fmt.Printf("nonfinite=%d canceled=%d rollback-preserved=%t accepted=%d\n", rejected, canceledCount, unchanged, stream.Snapshot().Count)
	fmt.Printf("raw-review-points=%d baseline-kW=%.6f ready=%t batch-parity=%t\n", rawReviewCount, latest.Value, latest.Valid, identical)
	// Output:
	// nonfinite=1 canceled=1 rollback-preserved=true accepted=16
	// raw-review-points=2 baseline-kW=46.304033 ready=true batch-parity=true
}

// ExampleCompute_extremeCancellation exposes exact seed cancellation on a deliberately adversarial fixture.
// This focused numerical law complements, rather than replaces, the application workflows.
//
// Implementation:
//   - Stage 1: create two canceling extremes around a unit residual.
//   - Stage 2: compute a three-observation exact SMA seed.
//   - Stage 3: print the nearest-even one-third value and its true readiness.
//
// Behavior highlights:
//   - Setup, canonical public operations, and meaningful result consumption are all executable.
//
// Inputs:
//   - [1e308,1,-1e308], an artificial dynamic-range diagnostic, not market data.
//
// Returns:
//   - A valid 0.33333333333333331 seed rather than a lost residual.
//
// Errors:
//   - Every library or codec error is checked before dependent state is used.
//
// Determinism:
//   - Fixed synthetic data and exact index/sequence traversal; output uses no wall clock.
//
// Complexity:
//   - Time O(3*W), Space O(3+W), W=34.
//
// Notes:
//   - The extreme values are numerical stress inputs with no claimed physical provenance.
//
// AI-Hints:
//   - Do not replace exact seeding with a naive float64 sum or an online rounded mean.
func ExampleCompute_extremeCancellation() {
	// This adversarial fixture is deliberately extreme; it is not claimed to be measured telemetry.
	residuals := []float64{1e308, 1, -1e308}
	result, err := ema.ComputeSMA(context.Background(), residuals, 3)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("seed=%.17g ready=%t\n", result.Values[2], result.Valid[2])
	// Output:
	// seed=0.33333333333333331 ready=true
}
