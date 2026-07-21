// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package main demonstrates production-style Dynamic Time Warping scenarios.
//
// The examples are intentionally deterministic and use fixed datasets so the output
// can be used as executable documentation and regression material.
package main

import (
	"errors"
	"fmt"
	"math"

	"github.com/lvlath/go/dtw"
)

// Candle represents one exchange candle.
//
// Fields:
//   - OpenTime and CloseTime identify the candle interval.
//   - OpenPrice, HighPrice, LowPrice, and ClosePrice store OHLC market data.
//
// Notes:
//   - The DTW example below intentionally aligns normalized close-to-close returns,
//     not raw prices. Raw price levels often hide the temporal shape that DTW is
//     supposed to compare.
type Candle struct {
	OpenTime   string  // timestamp of candle start (ISO 8601)
	CloseTime  string  // timestamp of candle close (ISO 8601)
	OpenPrice  float64 // opening price
	HighPrice  float64 // highest price
	LowPrice   float64 // lowest price
	ClosePrice float64 // closing price
}

// ExampleDTW_signalAlignment scans a crypto candle stream for a known breakout pattern.
//
// Scenario:
//
//	A market-monitoring service keeps a reference breakout pattern learned from a
//	prior ETH/USDT move. The live history is longer than the pattern and contains
//	quiet accumulation, ignition, expansion, cooling, and pullback phases. Direct
//	index-by-index comparison is too brittle because the same structure may form
//	one or two candles faster or slower.
//
//	The example converts OHLC candles into close-to-close return basis points.
//	This makes the comparison price-level invariant: a 0.04% impulse at 1918 and
//	a 0.04% impulse at 2100 describe the same directional shape.
//
// Playground: https://go.dev/play/p/96eGjJQxauW
//
// Implementation:
//   - Stage 1: Define a deterministic 26-candle reference market window.
//   - Stage 2: Define an 11-candle breakout pattern with a different nominal price level.
//   - Stage 3: Convert both datasets to close-return basis points.
//   - Stage 4: Slide the pattern length across the reference history.
//   - Stage 5: Run canonical dtw.Align with path tracking for each candidate segment.
//   - Stage 6: Keep the lowest reachable DTW distance and inspect the selected path.
//   - Stage 7: Print deterministic anchors that explain where the pattern was found.
//
// Behavior highlights:
//   - Uses canonical dtw.Align, not the legacy tuple-only wrapper.
//   - Uses WithWindow(2) to allow small timing drift without arbitrary warping.
//   - Uses WithSlopePenalty(0.05) to make repeated return frames measurable.
//   - Treats ErrNoPath as a candidate-window rejection, not as a process crash.
//   - Uses caller-defined loop order, so the first minimum is selected deterministically.
//
// Inputs:
//   - None. The candle datasets are fixed in the example.
//
// Returns:
//   - None. The function prints the selected match and path summary.
//
// Errors:
//   - Prints unexpected option, numeric, or path-surface errors and returns early.
//   - Skips candidate segments that are unreachable under the active window policy.
//
// Determinism:
//   - Stable for the same candle data, window radius, slope penalty, and dtw tie-break law.
//
// Complexity:
//   - Let R be reference candle count and P be pattern candle count.
//   - The sliding scan evaluates R-P+1 candidate windows.
//   - Each DTW run compares P-1 return samples against P-1 return samples.
//   - Total time is O((R-P+1) * P²), and each path-tracked run uses O(P²) memory.
//
// Notes:
//   - The path coordinates refer to return-signal indexes, not raw candle indexes.
//   - A path with no repeated coordinates means this matched segment has the same
//     phase count as the pattern; repeated coordinates would reveal stretched phases.
//
// AI-Hints:
//   - Do not compare raw market prices when the goal is shape matching.
//   - Do not treat +Inf as a magic number; use math.IsInf or ErrNoPath classification.
//   - Do not index dtw.Coord like an array; use Coord.I and Coord.J.
func ExampleDTW_signalAlignment() {
	refCandles := []Candle{
		{OpenTime: "2023-07-01 12:01:00.000000", CloseTime: "2023-07-01 12:01:59.999000", OpenPrice: 1916.31, HighPrice: 1916.70, LowPrice: 1916.20, ClosePrice: 1916.60},
		{OpenTime: "2023-07-01 12:02:00.000000", CloseTime: "2023-07-01 12:02:59.999000", OpenPrice: 1916.60, HighPrice: 1916.90, LowPrice: 1916.50, ClosePrice: 1916.80},
		{OpenTime: "2023-07-01 12:03:00.000000", CloseTime: "2023-07-01 12:03:59.999000", OpenPrice: 1916.80, HighPrice: 1916.85, LowPrice: 1916.10, ClosePrice: 1916.24},
		{OpenTime: "2023-07-01 12:04:00.000000", CloseTime: "2023-07-01 12:04:59.999000", OpenPrice: 1916.24, HighPrice: 1916.42, LowPrice: 1916.18, ClosePrice: 1916.33},
		{OpenTime: "2023-07-01 12:05:00.000000", CloseTime: "2023-07-01 12:05:59.999000", OpenPrice: 1916.33, HighPrice: 1916.60, LowPrice: 1916.20, ClosePrice: 1916.53},
		{OpenTime: "2023-07-01 12:06:00.000000", CloseTime: "2023-07-01 12:06:59.999000", OpenPrice: 1916.53, HighPrice: 1916.56, LowPrice: 1916.20, ClosePrice: 1916.34},
		{OpenTime: "2023-07-01 12:07:00.000000", CloseTime: "2023-07-01 12:07:59.999000", OpenPrice: 1916.34, HighPrice: 1917.12, LowPrice: 1916.29, ClosePrice: 1917.06},
		{OpenTime: "2023-07-01 12:08:00.000000", CloseTime: "2023-07-01 12:08:59.999000", OpenPrice: 1917.06, HighPrice: 1917.25, LowPrice: 1916.81, ClosePrice: 1917.20},
		{OpenTime: "2023-07-01 12:09:00.000000", CloseTime: "2023-07-01 12:09:59.999000", OpenPrice: 1917.20, HighPrice: 1917.22, LowPrice: 1916.18, ClosePrice: 1916.82},
		{OpenTime: "2023-07-01 12:10:00.000000", CloseTime: "2023-07-01 12:10:59.999000", OpenPrice: 1916.82, HighPrice: 1917.70, LowPrice: 1916.51, ClosePrice: 1917.63},
		{OpenTime: "2023-07-01 12:11:00.000000", CloseTime: "2023-07-01 12:11:59.999000", OpenPrice: 1917.63, HighPrice: 1917.70, LowPrice: 1917.00, ClosePrice: 1917.56},
		{OpenTime: "2023-07-01 12:12:00.000000", CloseTime: "2023-07-01 12:12:59.999000", OpenPrice: 1917.56, HighPrice: 1917.62, LowPrice: 1916.87, ClosePrice: 1917.39},
		{OpenTime: "2023-07-01 12:13:00.000000", CloseTime: "2023-07-01 12:13:59.999000", OpenPrice: 1917.39, HighPrice: 1917.70, LowPrice: 1917.30, ClosePrice: 1917.58},
		{OpenTime: "2023-07-01 12:14:00.000000", CloseTime: "2023-07-01 12:14:59.999000", OpenPrice: 1917.58, HighPrice: 1918.50, LowPrice: 1917.50, ClosePrice: 1918.40},
		{OpenTime: "2023-07-01 12:15:00.000000", CloseTime: "2023-07-01 12:15:59.999000", OpenPrice: 1918.40, HighPrice: 1918.75, LowPrice: 1918.10, ClosePrice: 1918.60},
		{OpenTime: "2023-07-01 12:16:00.000000", CloseTime: "2023-07-01 12:16:59.999000", OpenPrice: 1918.60, HighPrice: 1918.70, LowPrice: 1917.70, ClosePrice: 1917.81},
		{OpenTime: "2023-07-01 12:17:00.000000", CloseTime: "2023-07-01 12:17:59.999000", OpenPrice: 1917.81, HighPrice: 1917.90, LowPrice: 1916.30, ClosePrice: 1916.39},
		{OpenTime: "2023-07-01 12:18:00.000000", CloseTime: "2023-07-01 12:18:59.999000", OpenPrice: 1916.39, HighPrice: 1916.45, LowPrice: 1916.00, ClosePrice: 1916.32},
		{OpenTime: "2023-07-01 12:19:00.000000", CloseTime: "2023-07-01 12:19:59.999000", OpenPrice: 1916.32, HighPrice: 1917.05, LowPrice: 1915.84, ClosePrice: 1917.00},
		{OpenTime: "2023-07-01 12:20:00.000000", CloseTime: "2023-07-01 12:20:59.999000", OpenPrice: 1917.00, HighPrice: 1918.28, LowPrice: 1916.90, ClosePrice: 1918.20},
		{OpenTime: "2023-07-01 12:21:00.000000", CloseTime: "2023-07-01 12:21:59.999000", OpenPrice: 1918.20, HighPrice: 1919.85, LowPrice: 1918.10, ClosePrice: 1919.76},
		{OpenTime: "2023-07-01 12:22:00.000000", CloseTime: "2023-07-01 12:22:59.999000", OpenPrice: 1919.76, HighPrice: 1921.70, LowPrice: 1919.60, ClosePrice: 1921.62},
		{OpenTime: "2023-07-01 12:23:00.000000", CloseTime: "2023-07-01 12:23:59.999000", OpenPrice: 1921.62, HighPrice: 1923.10, LowPrice: 1920.90, ClosePrice: 1922.94},
		{OpenTime: "2023-07-01 12:24:00.000000", CloseTime: "2023-07-01 12:24:59.999000", OpenPrice: 1922.94, HighPrice: 1923.50, LowPrice: 1922.70, ClosePrice: 1923.30},
		{OpenTime: "2023-07-01 12:25:00.000000", CloseTime: "2023-07-01 12:25:59.999000", OpenPrice: 1923.30, HighPrice: 1923.35, LowPrice: 1922.80, ClosePrice: 1923.10},
		{OpenTime: "2023-07-01 12:26:00.000000", CloseTime: "2023-07-01 12:26:59.999000", OpenPrice: 1923.10, HighPrice: 1924.20, LowPrice: 1923.00, ClosePrice: 1924.00},
	}

	pattern := []Candle{
		{OpenTime: "template T+00", CloseTime: "template T+00 close", OpenPrice: 2100.00, HighPrice: 2100.20, LowPrice: 2099.80, ClosePrice: 2100.00},
		{OpenTime: "template T+01", CloseTime: "template T+01 close", OpenPrice: 2100.00, HighPrice: 2101.00, LowPrice: 2099.90, ClosePrice: 2100.90},
		{OpenTime: "template T+02", CloseTime: "template T+02 close", OpenPrice: 2100.90, HighPrice: 2101.30, LowPrice: 2100.70, ClosePrice: 2101.10},
		{OpenTime: "template T+03", CloseTime: "template T+03 close", OpenPrice: 2101.10, HighPrice: 2101.20, LowPrice: 2100.10, ClosePrice: 2100.25},
		{OpenTime: "template T+04", CloseTime: "template T+04 close", OpenPrice: 2100.25, HighPrice: 2100.30, LowPrice: 2098.60, ClosePrice: 2098.70},
		{OpenTime: "template T+05", CloseTime: "template T+05 close", OpenPrice: 2098.70, HighPrice: 2099.00, LowPrice: 2098.50, ClosePrice: 2098.65},
		{OpenTime: "template T+06", CloseTime: "template T+06 close", OpenPrice: 2098.65, HighPrice: 2099.55, LowPrice: 2098.50, ClosePrice: 2099.40},
		{OpenTime: "template T+07", CloseTime: "template T+07 close", OpenPrice: 2099.40, HighPrice: 2100.85, LowPrice: 2099.20, ClosePrice: 2100.70},
		{OpenTime: "template T+08", CloseTime: "template T+08 close", OpenPrice: 2100.70, HighPrice: 2102.60, LowPrice: 2100.50, ClosePrice: 2102.45},
		{OpenTime: "template T+09", CloseTime: "template T+09 close", OpenPrice: 2102.45, HighPrice: 2104.70, LowPrice: 2102.30, ClosePrice: 2104.50},
		{OpenTime: "template T+10", CloseTime: "template T+10 close", OpenPrice: 2104.50, HighPrice: 2106.00, LowPrice: 2104.20, ClosePrice: 2105.90},
	}

	const (
		windowRadius = 2
		slopePenalty = 0.05
		matchCutoff  = 1.50
	)

	patternSignal := closeReturnBPS(pattern)
	patternCandleCount := len(pattern)

	bestDistance := math.Inf(1)
	bestIndex := -1
	var bestPath dtw.Path

	for start := 0; start+patternCandleCount <= len(refCandles); start++ {
		candidateSignal := closeReturnBPS(refCandles[start : start+patternCandleCount])

		res, err := dtw.Align(
			candidateSignal,
			patternSignal,
			dtw.WithWindow(windowRadius),
			dtw.WithSlopePenalty(slopePenalty),
			dtw.WithReturnPath(true),
		)
		if errors.Is(err, dtw.ErrNoPath) {
			continue
		}
		if err != nil {
			fmt.Printf("align candidate %d: %v\n", start, err)
			return
		}
		if res == nil || !res.Reachable {
			continue
		}

		path, err := res.PathOrError()
		if err != nil {
			fmt.Printf("candidate path %d: %v\n", start, err)
			return
		}

		if res.Distance < bestDistance {
			bestDistance = res.Distance
			bestIndex = start
			bestPath = path
		}
	}

	if bestIndex < 0 {
		fmt.Println("no reachable candidate")
		return
	}

	// stretchSteps counts vertical or horizontal DTW path moves.
	//   - A stretch step means one side advanced while the other side was held.
	//   - In a market-pattern scenario, stretch steps reveal delayed or prolonged phases.
	stretchSteps := 0
	for i := 1; i < len(bestPath); i++ {
		if bestPath[i].I == bestPath[i-1].I || bestPath[i].J == bestPath[i-1].J {
			stretchSteps++
		}
	}

	bestEnd := bestIndex + patternCandleCount - 1

	fmt.Printf("best-window=index=%d start=%s end=%s\n", bestIndex, refCandles[bestIndex].OpenTime, refCandles[bestEnd].CloseTime)
	fmt.Printf("distance=%.2f path-steps=%d stretch-steps=%d\n", bestDistance, len(bestPath), stretchSteps)
	fmt.Printf("anchors first=%v ignition=%v expansion=%v final=%v\n", bestPath[0], bestPath[3], bestPath[7], bestPath[len(bestPath)-1])

	if bestDistance <= matchCutoff {
		fmt.Println("decision=pattern match")
	} else {
		fmt.Println("decision=manual review")
	}

	// Output:
	// best-window=index=12 start=2023-07-01 12:13:00.000000 end=2023-07-01 12:23:59.999000
	// distance=0.89 path-steps=10 stretch-steps=0
	// anchors first={0 0} ignition={3 3} expansion={7 7} final={9 9}
	// decision=pattern match
}

// closeReturnBPS converts candle close prices into close-to-close basis-point returns.
//
// Implementation:
//   - Stage 1: Return an empty signal when fewer than two candles are provided.
//   - Stage 2: Compute each close-to-close relative move in basis points.
//   - Stage 3: Return a shape-oriented signal suitable for scalar DTW.
//
// Notes:
//   - The returned signal has len(candles)-1 samples.
//   - Basis points keep the example readable while avoiding raw-price level bias.
func closeReturnBPS(candles []Candle) []float64 {
	if len(candles) < 2 {
		return nil
	}

	series := make([]float64, len(candles)-1)
	for i := 1; i < len(candles); i++ {
		previousClose := candles[i-1].ClosePrice
		currentClose := candles[i].ClosePrice
		series[i-1] = ((currentClose - previousClose) / previousClose) * 10000
	}

	return series
}
