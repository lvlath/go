// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Five executable, reproducible OHLCV operations use only the public mfi API.
// Fixtures are synthetic but preserve realistic candle geometry, volume units,
// zero-trade intervals and independent market-data feed identities.
package mfi_test

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/lvlath/go/mfi"
)

// exampleMarketBars builds independent instrument-level OHLCV histories.
// Price and base-asset volume units are consistent within each generated series;
// a zero-volume candle remains a legitimate point in the ordered source feed.
func exampleMarketBars(instrument, count int) []mfi.Bar {
	rows := make([]mfi.Bar, count)
	base := 75 + float64(instrument)*3.75
	previousClose := base
	for i := range rows {
		// Independent drift, cyclical pricing and varying spreads model a feed with
		// micro-trends, reversals and intermittent periods without matched trades.
		drift := float64((instrument%5)-2) * 0.045 * float64(i)
		cycle := float64(((i*11+instrument*7)%17)-8) * 0.13
		closePrice := base + drift + cycle
		openPrice := previousClose
		spread := 0.12 + float64((i+instrument)%4)*0.035
		volume := float64(140 + (i*37+instrument*29)%410)
		if (i+instrument)%19 == 0 {
			volume = 0
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

// exampleColumns adapts row-oriented fixtures to the canonical column facade.
// Unlike a live zero-copy Bars source, this adapter owns each output column.
func exampleColumns(rows []mfi.Bar) mfi.Bars {
	columns := mfi.Bars{
		Open:   make([]float64, len(rows)),
		High:   make([]float64, len(rows)),
		Low:    make([]float64, len(rows)),
		Close:  make([]float64, len(rows)),
		Volume: make([]float64, len(rows)),
	}
	for i, row := range rows {
		columns.Open[i], columns.High[i], columns.Low[i] = row.Open, row.High, row.Low
		columns.Close[i], columns.Volume[i] = row.Close, row.Volume
	}
	return columns
}

// ExampleCompute_crossAssetLiquidityReport demonstrates a universe-scale
// price/volume divergence audit over fifty independent instrument histories.
//
// Scenario:
//   - A research data service receives 64 complete one-minute OHLCV bars for
//     each of 50 instruments, with volume expressed in each asset's base unit.
//   - The service identifies prices rising while the corresponding MFI feature
//     declines over the same ten-bar interval, then ranks observations for review.
//
// Implementation:
//   - Stage 1: build independently synchronized candle histories per instrument.
//   - Stage 2: Compute MFI(14) and check both comparison points' validity masks.
//   - Stage 3: rank the largest opposing feature changes with stable ID tie-breaks.
//
// Behavior highlights:
//   - No cross-asset flow sums are taken: MFI is dimensionless within each feed.
//   - The downstream review policy is illustrative, not an MFI trading signal.
//
// Determinism:
//   - Instruments are scanned ascending; ranking ties are resolved by instrument ID.
//
// Complexity:
//   - O(A*N + A*log A) time, O(N + A) application memory, with A=50, N=64.
//
// AI-Hints:
//   - Compare values at identical historical indices and never use invalid warmup zeros.
func ExampleCompute_crossAssetLiquidityReport() {
	const (
		assets          = 50
		candlesPerAsset = 64
		lookbackBars    = 10
	)
	type reviewItem struct {
		id                 int
		closeChangePercent float64
		mfiChangePoints    float64
	}
	queue := make([]reviewItem, 0, assets)
	ready, zeroVolume := 0, 0

	for instrument := 0; instrument < assets; instrument++ {
		bars := exampleMarketBars(instrument, candlesPerAsset)
		for _, bar := range bars {
			if bar.Volume == 0 {
				zeroVolume++
			}
		}
		result, err := mfi.ComputeDefault(exampleColumns(bars))
		if err != nil {
			fmt.Println("batch validation:", err)
			return
		}
		last := len(bars) - 1
		earlier := last - lookbackBars
		if !result.Valid[last] || !result.Valid[earlier] {
			continue // An incomplete window is not a numeric zero.
		}
		ready++
		priceChange := 100 * (bars[last].Close/bars[earlier].Close - 1)
		flowChange := result.Values[last] - result.Values[earlier]
		if priceChange > 0 && flowChange < 0 {
			queue = append(queue, reviewItem{
				id:                 instrument,
				closeChangePercent: priceChange,
				mfiChangePoints:    flowChange,
			})
		}
	}

	// The reviewing service owns ranking policy; the MFI package does not.
	sort.Slice(queue, func(i, j int) bool {
		if queue[i].mfiChangePoints == queue[j].mfiChangePoints {
			return queue[i].id < queue[j].id
		}
		return queue[i].mfiChangePoints < queue[j].mfiChangePoints
	})
	fmt.Printf("assets=%d ready=%d zero-volume=%d divergence-reviews=%d\n",
		assets, ready, zeroVolume, len(queue))
	for i := 0; i < len(queue) && i < 3; i++ {
		item := queue[i]
		fmt.Printf("review-%d asset-%02d close=%+.2f%% mfi=%+.2f pts\n",
			i+1, item.id, item.closeChangePercent, item.mfiChangePoints)
	}

	// Output:
	// assets=50 ready=50 zero-volume=165 divergence-reviews=13
	// review-1 asset-05 close=+0.16% mfi=-9.32 pts
	// review-2 asset-15 close=+0.11% mfi=-6.65 pts
	// review-3 asset-44 close=+0.79% mfi=-5.85 pts
}

// ExampleStream_Push_tightRangePressure demonstrates a causal market-data
// monitoring pipeline that compares live features with an audited archive.
//
// Scenario:
//   - A one-minute OHLCV gateway maintains an MFI(14) feature under a fixed
//     instrument identity and externally supplied contiguous timestamps.
//   - An observability layer reviews high MFI only when the last eight closes
//     remain inside an independently measured 2.5% range.
//
// Implementation:
//   - Stage 1: compute an archive reference and construct the live stream.
//   - Stage 2: push each bar once, assert live/archive bit-level consistency.
//   - Stage 3: record valid compressed-range events and report the first bar index.
//
// Behavior highlights:
//   - A zero-volume candle moves the window without contributing monetary flow.
//   - Feature comparison and alert thresholds are application policies, not mfi APIs.
//
// Determinism:
//   - All 42 observations are evaluated in arrival order, including warmup entries.
//
// Complexity:
//   - O(N*8 + N) time and O(N) example memory; Stream state is O(Period).
//
// AI-Hints:
//   - Do not compare a warmup zero with a valid numerical value or replay an old bar.
func ExampleStream_Push_tightRangePressure() {
	const period = 14
	rows := exampleMarketBars(4, 42)
	archive, err := mfi.ComputeCandles(rows, mfi.Options{Period: period})
	if err != nil {
		fmt.Println("archive:", err)
		return
	}
	stream, err := mfi.NewPeriod(period)
	if err != nil {
		fmt.Println("stream:", err)
		return
	}

	validWindows, pressureCount, firstPressure := 0, 0, -1
	parity := true
	for i, bar := range rows {
		value, valid, pushErr := stream.Push(bar)
		if pushErr != nil {
			fmt.Println("bar rejected:", pushErr)
			return
		}
		if valid != archive.Valid[i] || math.Float64bits(value) != math.Float64bits(archive.Values[i]) {
			parity = false
		}
		if !valid {
			continue
		}
		validWindows++

		start := i - 7
		if start < 0 {
			start = 0
		}
		minClose, maxClose := rows[start].Close, rows[start].Close
		for _, previous := range rows[start : i+1] {
			minClose = math.Min(minClose, previous.Close)
			maxClose = math.Max(maxClose, previous.Close)
		}
		if value >= 65 && (maxClose-minClose)/bar.Close <= 0.025 {
			pressureCount++
			if firstPressure < 0 {
				firstPressure = i
			}
		}
	}
	fmt.Printf("valid=%d compressed-high-flow=%d first-bar=%d batch-parity=%t count=%d\n",
		validWindows, pressureCount, firstPressure, parity, stream.Snapshot().Count)

	// Output:
	// valid=28 compressed-high-flow=13 first-bar=15 batch-parity=true count=42
}

// ExampleStream_Snapshot_authenticatedFailover demonstrates market-feed
// failover with a signed checkpoint envelope and a monotone source offset.
//
// Scenario:
//   - A primary worker accepts 29 ordered bars for one one-minute instrument.
//   - A standby must verify the source identity, last accepted sequence and
//     full Neumaier ring state before processing the next broker message.
//
// Implementation:
//   - Stage 1: checkpoint the primary and serialize feed metadata together with state.
//   - Stage 2: authenticate the entire payload, rejecting a changed feed identity.
//   - Stage 3: restore the standby and resume with exactly the next source sequence.
//
// Behavior highlights:
//   - Snapshot.Config is a configuration fingerprint, not an authenticity proof.
//   - The broker offset and HMAC key are external to mfi, never stored in Stream.
//
// Determinism:
//   - Both streams consume the identical post-checkpoint sequence in fixed order.
//
// Complexity:
//   - O(P + R) for snapshot and R resumed bars, amortized over the fixed ring.
//
// AI-Hints:
//   - Authenticate both feed identity and source offset; a valid ring alone
//     does not prevent accidental duplicate consumption after restart.
func ExampleStream_Snapshot_authenticatedFailover() {
	const feedID = "SOL-USD:1m"
	const initialSequence uint64 = 870_200
	const checkpointBars = 29

	type marketEvent struct {
		sequence uint64
		bar      mfi.Bar
	}
	events := make([]marketEvent, 48)
	sequence := initialSequence
	for i, bar := range exampleMarketBars(11, len(events)) {
		events[i] = marketEvent{sequence: sequence, bar: bar}
		sequence++
	}
	primary, err := mfi.NewPeriod(14)
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, event := range events[:checkpointBars] {
		if _, _, err = primary.Push(event.bar); err != nil {
			fmt.Println(err)
			return
		}
	}

	// Transport has its own identity and monotone sequence. Both are signed.
	type checkpointEnvelope struct {
		Feed         string       `json:"feed"`
		LastSequence uint64       `json:"last_sequence"`
		State        mfi.Snapshot `json:"state"`
	}
	envelope := checkpointEnvelope{
		Feed:         feedID,
		LastSequence: events[checkpointBars-1].sequence,
		State:        primary.Snapshot(),
	}
	payload, err := json.Marshal(envelope)
	if err != nil {
		fmt.Println(err)
		return
	}

	// The fixed demonstration key is a fixture, never a production credential.
	key := []byte("example-only-key-not-for-deployment")
	macFor := func(message []byte) []byte {
		mac := hmac.New(sha256.New, key)
		_, _ = mac.Write(message)
		return mac.Sum(nil)
	}
	signature := macFor(payload)
	altered := bytes.Replace(payload, []byte(feedID), []byte("DOT-USD:1m"), 1)
	tamperingRejected := !hmac.Equal(signature, macFor(altered))

	if !hmac.Equal(signature, macFor(payload)) {
		fmt.Println("checkpoint rejected")
		return
	}
	var recovered checkpointEnvelope
	if err = json.Unmarshal(payload, &recovered); err != nil {
		fmt.Println(err)
		return
	}
	if recovered.Feed != feedID || recovered.LastSequence != events[checkpointBars-1].sequence {
		fmt.Println("unexpected feed or sequence")
		return
	}
	standby, err := mfi.NewPeriod(14)
	if err != nil {
		fmt.Println(err)
		return
	}
	if err = standby.Restore(recovered.State); err != nil {
		fmt.Println(err)
		return
	}

	// A broker can resume at recovered.LastSequence+1 without replaying a bar.
	expectedSequence := recovered.LastSequence + 1
	identical := true
	for _, event := range events[checkpointBars:] {
		if event.sequence != expectedSequence {
			identical = false
			break
		}
		left, leftReady, leftErr := primary.Push(event.bar)
		right, rightReady, rightErr := standby.Push(event.bar)
		if leftErr != nil || rightErr != nil || leftReady != rightReady ||
			math.Float64bits(left) != math.Float64bits(right) {
			identical = false
			break
		}
		expectedSequence++
	}
	fmt.Printf("feed=%s committed=%d tampering-rejected=%t resumed-bars=%d bitwise=%t\n",
		recovered.Feed, recovered.LastSequence, tamperingRejected,
		len(events)-checkpointBars, identical)

	// Output:
	// feed=SOL-USD:1m committed=870228 tampering-rejected=true resumed-bars=19 bitwise=true
}

// ExampleStream_Reset_marketDataGap demonstrates the caller-owned timestamp
// boundary that prevents MFI windows spanning missing market-data intervals.
//
// Scenario:
//   - A one-minute feed pauses after eight candles and resumes 26 minutes later.
//   - The period-five feature must be reseeded rather than quietly connecting
//     the new market to the previous uninterrupted transition window.
//
// Implementation:
//   - Stage 1: maintain two streams, one gap-aware and one intentionally naive.
//   - Stage 2: Reset the gap-aware engine whenever cadence is discontinuous.
//   - Stage 3: report the new warmup and demonstrate the naive engine's false readiness.
//
// Behavior highlights:
//   - The exact timestamp policy is external to mfi; no synthetic bars are added.
//
// Determinism:
//   - Synthetic UTC minute timestamps are fixed and bars are consumed in order.
//
// Complexity:
//   - O(N) stream work amortized plus O(Period) per explicit Reset.
//
// AI-Hints:
//   - A valid numerical MFI over a missing interval is not proof of a valid time grid.
func ExampleStream_Reset_marketDataGap() {
	const period = 5
	correct, err := mfi.NewPeriod(period)
	if err != nil {
		fmt.Println(err)
		return
	}
	naive, err := mfi.NewPeriod(period)
	if err != nil {
		fmt.Println(err)
		return
	}

	rows := exampleMarketBars(7, 19)
	origin := time.Date(2026, 11, 17, 9, 0, 0, 0, time.UTC)
	previous := time.Time{}
	resets, discardedWarmup := 0, 0
	var atFirstGapNaive, finalReady bool
	for i, bar := range rows {
		minute := i
		if i >= 8 {
			minute += 25
		}
		stamp := origin.Add(time.Duration(minute) * time.Minute)
		if !previous.IsZero() && stamp.Sub(previous) != time.Minute {
			correct.Reset()
			resets++
		}
		_, finalReady, err = correct.Push(bar)
		if err != nil {
			fmt.Println(err)
			return
		}
		_, naiveReady, naiveErr := naive.Push(bar)
		if naiveErr != nil {
			fmt.Println(naiveErr)
			return
		}
		if i == 8 {
			atFirstGapNaive = naiveReady
		}
		if resets > 0 && !finalReady {
			discardedWarmup++
		}
		previous = stamp
	}
	fmt.Printf("gaps=%d post-gap-warmup=%d naive-ready-at-gap=%t final-valid=%t\n",
		resets, discardedWarmup, atFirstGapNaive, finalReady)

	// Output:
	// gaps=1 post-gap-warmup=5 naive-ready-at-gap=true final-valid=true
}

// ExampleComputeCandles_replayReconciliation demonstrates a sequence-aware
// data-quality audit against a replay feed that dropped a zero-volume candle.
//
// Scenario:
//   - An archival source contains 36 sequential OHLCV events, including a
//     no-trade bar with a legitimate price and a unique source sequence number.
//   - An ETL process silently drops that bar, shifting transition-window age
//     and potentially changing MFI for the following shared event timestamps.
//
// Implementation:
//   - Stage 1: construct complete and intentionally defective event sequences.
//   - Stage 2: compare MFI on matching event identities, not array endpoints.
//   - Stage 3: report the first missing sequence and affected subsequent outputs.
//
// Behavior highlights:
//   - Zero volume does not mean absent input; subsequent price comparisons change.
//   - Missing sequence detection is the primary integrity gate, not score equality.
//
// Determinism:
//   - Event sequences and aligned comparisons follow monotonically increasing IDs.
//
// Complexity:
//   - O(N) batch work amortized and O(N) application-side comparison storage.
//
// AI-Hints:
//   - Never compare only last-array-index scores when replay sources lost an event.
func ExampleComputeCandles_replayReconciliation() {
	const missingIndex = 29
	const firstSequence uint64 = 310_000
	type archivedEvent struct {
		sequence uint64
		bar      mfi.Bar
	}
	complete := make([]archivedEvent, 36)
	sequence := firstSequence
	for i, bar := range exampleMarketBars(3, len(complete)) {
		if i == missingIndex {
			bar.Volume = 0
		}
		complete[i] = archivedEvent{sequence: sequence, bar: bar}
		sequence++
	}
	broken := append([]archivedEvent(nil), complete[:missingIndex]...)
	broken = append(broken, complete[missingIndex+1:]...)

	archiveBars := make([]mfi.Bar, len(complete))
	brokenBars := make([]mfi.Bar, len(broken))
	for i, event := range complete {
		archiveBars[i] = event.bar
	}
	for i, event := range broken {
		brokenBars[i] = event.bar
	}
	archiveResult, err := mfi.ComputeCandles(archiveBars, mfi.Options{Period: 14})
	if err != nil {
		fmt.Println(err)
		return
	}
	replayResult, err := mfi.ComputeCandles(brokenBars, mfi.Options{Period: 14})
	if err != nil {
		fmt.Println(err)
		return
	}

	// The ingest layer detects a hole in source sequence IDs before publication.
	expectedSequence := firstSequence
	firstGap := uint64(0)
	for _, event := range broken {
		if event.sequence != expectedSequence {
			firstGap = expectedSequence
			break
		}
		expectedSequence++
	}
	mismatches := 0
	for replayIndex, event := range broken {
		archiveIndex := replayIndex
		if event.sequence > complete[missingIndex].sequence {
			archiveIndex++
		}
		if archiveResult.Valid[archiveIndex] != replayResult.Valid[replayIndex] ||
			archiveResult.Valid[archiveIndex] && math.Float64bits(archiveResult.Values[archiveIndex]) !=
				math.Float64bits(replayResult.Values[replayIndex]) {
			mismatches++
		}
	}
	fmt.Printf("archive=%d replay=%d missing-sequence=%d dropped-zero=%t aligned-mismatches=%d\n",
		len(complete), len(broken), firstGap, complete[missingIndex].bar.Volume == 0, mismatches)

	// Output:
	// archive=36 replay=35 missing-sequence=310029 dropped-zero=true aligned-mismatches=5
}
