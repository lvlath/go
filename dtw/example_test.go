// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw_test demonstrates practical DTW workflows.
// Examples use deterministic data that resembles production analytical inputs.
package dtw_test

import (
	"fmt"

	"github.com/lvlath/go/dtw"
	"github.com/lvlath/go/matrix"
)

// ExampleAlignMatrix_cryptoBreakoutOHLC demonstrates matrix-backed DTW for OHLC market-pattern matching.
//
// Scenario:
//   - A trading system stores a reference breakout as normalized OHLC candle vectors.
//   - A live BTC/USDT-like window contains the same price action, but stretched across more candles.
//   - Direct candle-by-candle comparison would punish timing drift and miss the pattern.
//   - DTW aligns the reference and live windows while preserving chronological order.
//
// Options:
//   - WithWindow(4) keeps alignment close to the diagonal and prevents unrealistic jumps.
//   - WithSlopePenalty(0.15) makes repeated candles visible in the distance.
//   - WithReturnPath(true) exposes which reference candles explain which live candles.
//   - WithReturnAccumulated(true) returns the DP surface for later diagnostics.
//   - WithReturnLocalCost(true) returns the OHLC squared-L2 cost matrix.
//
// Use case:
//
//	Signal filtering before a trade trigger: reject live windows that are too far from
//	the known breakout structure, but accept the same structure when the market forms it
//	slower or faster than the template.
//
// Complexity:
//   - Local-cost construction is O(n*m*d), where d=4 OHLC features.
//   - DTW is O(n*m) time.
//   - Returning path and accumulated matrix uses O(n*m) memory.
func ExampleAlignMatrix_cryptoBreakoutOHLC() {
	// The reference pattern is indexed around 100.00 so values remain readable.
	// Each row is one candle: open, high, low, close.
	breakoutTemplate, err := matrix.NewPreparedDense(12, 4)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	err = breakoutTemplate.Fill([]float64{
		100.00, 100.40, 99.70, 100.10,
		100.10, 100.60, 99.90, 100.35,
		100.35, 101.00, 100.10, 100.85,
		100.85, 102.20, 100.70, 101.90,
		101.90, 104.30, 101.70, 103.80,
		103.80, 106.10, 103.20, 105.40,
		105.40, 106.40, 104.60, 105.90,
		105.90, 106.00, 104.80, 105.20,
		105.20, 105.60, 103.90, 104.30,
		104.30, 104.80, 103.20, 103.60,
		103.60, 104.10, 102.90, 103.20,
		103.20, 103.70, 102.70, 103.10,
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	// The observed window has the same breakout structure, but accumulation and
	// cooling phases take more candles. This is exactly where DTW is useful:
	// the shape is similar, while time is not perfectly synchronized.
	liveWindow, err := matrix.NewPreparedDense(15, 4)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	err = liveWindow.Fill([]float64{
		100.00, 100.30, 99.80, 100.05,
		100.05, 100.50, 99.90, 100.20,
		100.20, 100.80, 100.00, 100.55,
		100.55, 101.10, 100.30, 100.90,
		100.90, 102.40, 100.70, 102.00,
		102.00, 103.80, 101.70, 103.40,
		103.40, 105.50, 103.00, 104.80,
		104.80, 106.20, 104.30, 105.70,
		105.70, 106.50, 104.90, 105.80,
		105.80, 106.00, 104.70, 105.10,
		105.10, 105.50, 103.80, 104.40,
		104.40, 104.90, 103.30, 103.80,
		103.80, 104.20, 103.00, 103.30,
		103.30, 103.80, 102.80, 103.00,
		103.00, 103.50, 102.50, 102.90,
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	res, err := dtw.AlignMatrix(
		breakoutTemplate,
		liveWindow,
		dtw.WithWindow(4),
		dtw.WithSlopePenalty(0.15),
		dtw.WithReturnPath(true),
		dtw.WithReturnAccumulated(true),
		dtw.WithReturnLocalCost(true),
	)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	// The path anchors explain the market story:
	//   entry     - accumulation starts at the same structural point;
	//   ignition  - live breakout arrives later than the template;
	//   peak      - peak impulse is aligned despite candle drift;
	//   exit      - both patterns finish at the post-breakout pullback.
	fmt.Printf("distance=%.3f path-steps=%d\n", res.Distance, len(res.Path))
	fmt.Printf(
		"anchors entry=%v ignition=%v peak=%v exit=%v\n",
		res.Path[0],
		res.Path[4],
		res.Path[7],
		res.Path[len(res.Path)-1],
	)
	fmt.Printf(
		"matrices accumulated=%dx%d local=%dx%d\n",
		res.Accumulated.Rows(),
		res.Accumulated.Cols(),
		res.LocalCost.Rows(),
		res.LocalCost.Cols(),
	)

	if res.Distance < 4.0 {
		fmt.Println("trade-filter=similar stretched breakout")
	} else {
		fmt.Println("trade-filter=reject pattern")
	}

	// Output:
	// distance=3.233 path-steps=15
	// anchors entry={0 0} ignition={3 4} peak={6 7} exit={11 14}
	// matrices accumulated=13x16 local=12x15
	// trade-filter=similar stretched breakout
}

// ExampleAlignCostMatrix_voiceCommandAcousticCost demonstrates DTW over a precomputed acoustic cost surface.
//
// Scenario:
//   - A speech model has already converted audio frames into pairwise acoustic costs.
//   - Rows represent expected command frames for a phrase similar to "unlock the vault".
//   - Columns represent a spoken attempt where the final word is stretched.
//   - DTW aligns the phrase without recomputing MFCCs, embeddings, or phoneme logits.
//
// Options:
//   - AlignCostMatrix consumes finite non-negative frame-to-frame costs directly.
//   - WithWindow(4) allows realistic pronunciation drift but blocks arbitrary jumps.
//   - WithSlopePenalty(0.12) makes stretched frames measurable instead of free.
//   - WithReturnPath(true) exposes which spoken frames map to which expected frames.
//   - WithReturnLocalCost(true) returns a detached cost-surface snapshot.
//
// Use case:
//
//	Command verification, wake-word matching, or call-center phrase detection where
//	an upstream neural/acoustic model supplies the local frame costs and DTW supplies
//	the monotone temporal alignment.
//
// Complexity:
//   - Cost materialization is O(n*m).
//   - DTW is O(n*m) time.
//   - Path reconstruction uses O(n*m) accumulated storage.
func ExampleAlignCostMatrix_voiceCommandAcousticCost() {
	localAcousticCost, err := matrix.NewPreparedDense(10, 14)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	// The low-cost valley follows the expected spoken phrase. The tail has extra
	// low-cost columns because the speaker holds the final word longer than the template.
	err = localAcousticCost.Fill([]float64{
		0.03, 0.73, 1.31, 1.80, 2.38, 2.41, 2.90, 3.48, 4.06, 4.00, 4.58, 5.16, 5.10, 5.13,
		0.73, 0.02, 0.70, 1.28, 1.86, 1.80, 2.38, 2.96, 3.45, 3.48, 4.06, 4.55, 4.58, 4.61,
		1.31, 0.70, 0.04, 0.76, 1.25, 1.28, 1.86, 2.35, 2.93, 2.96, 3.45, 4.03, 4.06, 4.00,
		1.80, 1.28, 0.76, 0.01, 0.73, 0.76, 1.25, 1.83, 2.41, 2.35, 2.93, 3.51, 3.45, 3.48,
		2.38, 1.86, 1.25, 0.73, 0.02, 0.02, 0.73, 1.31, 1.80, 1.83, 2.41, 2.90, 2.93, 2.96,
		2.96, 2.35, 1.83, 1.31, 0.70, 0.73, 0.03, 0.70, 1.28, 1.31, 1.80, 2.38, 2.41, 2.35,
		3.45, 2.93, 2.41, 1.80, 1.28, 1.31, 0.70, 0.02, 0.76, 0.70, 1.28, 1.86, 1.80, 1.83,
		4.03, 3.51, 2.90, 2.38, 1.86, 1.80, 1.28, 0.76, 0.03, 0.04, 0.76, 1.25, 1.28, 1.31,
		4.61, 4.00, 3.48, 2.96, 2.35, 2.38, 1.86, 1.25, 0.73, 0.76, 0.02, 0.73, 0.76, 0.70,
		5.10, 4.58, 4.06, 3.45, 2.93, 2.96, 2.35, 1.83, 1.31, 1.25, 0.73, 0.02, 0.02, 0.03,
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	res, err := dtw.AlignCostMatrix(
		localAcousticCost,
		dtw.WithWindow(4),
		dtw.WithSlopePenalty(0.12),
		dtw.WithReturnPath(true),
		dtw.WithReturnLocalCost(true),
	)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	stretchSteps := 0
	for i := 1; i < len(res.Path); i++ {
		if res.Path[i].I == res.Path[i-1].I || res.Path[i].J == res.Path[i-1].J {
			stretchSteps++
		}
	}

	fmt.Printf("command distance=%.2f\n", res.Distance)
	fmt.Printf("path-steps=%d stretch-steps=%d\n", len(res.Path), stretchSteps)
	fmt.Printf("anchors start=%v vault-core=%v end=%v\n", res.Path[0], res.Path[5], res.Path[len(res.Path)-1])
	fmt.Printf("local-shape=%dx%d\n", res.LocalCost.Rows(), res.LocalCost.Cols())

	if res.Distance < 1.0 && stretchSteps <= 4 {
		fmt.Println("decision=accepted with stretched pronunciation")
	} else {
		fmt.Println("decision=manual review")
	}

	// Output:
	// command distance=0.83
	// path-steps=14 stretch-steps=4
	// anchors start={0 0} vault-core={4 5} end={9 13}
	// local-shape=10x14
	// decision=accepted with stretched pronunciation
}

// ExampleAlignMatrix_vibrationSignature demonstrates multivariate DTW for machine-condition monitoring.
//
// Scenario:
//   - A production line stores a reference vibration signature for a healthy spindle impact.
//   - Each time step contains three normalized features: RMS energy, spectral centroid drift,
//     and impulse-kurtosis proxy.
//   - The observed event has the same shape but a slower rise and an extra peak frame.
//   - DTW aligns the physical event phases instead of comparing sample indexes blindly.
//
// Options:
//   - AlignMatrix computes squared L2 row costs from the feature vectors.
//   - WithWindow(4) keeps phase matching local and prevents impossible jumps.
//   - WithSlopePenalty(0.015) records stretching without over-penalizing sensor delay.
//   - WithReturnPath(true) exposes the phase alignment.
//   - WithReturnLocalCost(true) exposes the feature-level cost surface.
//
// Use case:
//
//	Predictive maintenance can accept a delayed-but-normal impact signature while
//	escalating genuinely different vibration shapes for inspection.
//
// Complexity:
//   - Local-cost construction is O(n*m*d), where d=3 sensor features.
//   - DTW is O(n*m) time.
//   - Returned path and local matrix are O(n+m) and O(n*m) memory respectively.
func ExampleAlignMatrix_vibrationSignature() {
	referenceSignature, err := matrix.NewPreparedDense(14, 3)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	err = referenceSignature.Fill([]float64{
		0.05, 0.10, 0.02,
		0.08, 0.13, 0.03,
		0.12, 0.18, 0.04,
		0.24, 0.32, 0.08,
		0.45, 0.55, 0.16,
		0.72, 0.80, 0.30,
		0.94, 0.88, 0.42,
		0.88, 0.70, 0.36,
		0.62, 0.48, 0.24,
		0.38, 0.31, 0.15,
		0.22, 0.22, 0.10,
		0.14, 0.18, 0.07,
		0.10, 0.15, 0.05,
		0.07, 0.12, 0.03,
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	observedSignature, err := matrix.NewPreparedDense(17, 3)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	err = observedSignature.Fill([]float64{
		0.04, 0.11, 0.02,
		0.07, 0.13, 0.03,
		0.09, 0.15, 0.03,
		0.13, 0.19, 0.05,
		0.22, 0.30, 0.08,
		0.33, 0.43, 0.11,
		0.47, 0.56, 0.17,
		0.70, 0.78, 0.29,
		0.90, 0.87, 0.41,
		0.96, 0.82, 0.43,
		0.87, 0.69, 0.35,
		0.63, 0.50, 0.25,
		0.39, 0.33, 0.16,
		0.24, 0.23, 0.11,
		0.15, 0.18, 0.07,
		0.10, 0.15, 0.05,
		0.07, 0.12, 0.03,
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	res, err := dtw.AlignMatrix(
		referenceSignature,
		observedSignature,
		dtw.WithWindow(4),
		dtw.WithSlopePenalty(0.015),
		dtw.WithReturnPath(true),
		dtw.WithReturnLocalCost(true),
	)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	stretchSteps := 0
	for i := 1; i < len(res.Path); i++ {
		if res.Path[i].I == res.Path[i-1].I || res.Path[i].J == res.Path[i-1].J {
			stretchSteps++
		}
	}

	fmt.Printf("vibration distance=%.4f\n", res.Distance)
	fmt.Printf("path-steps=%d stretch-steps=%d\n", len(res.Path), stretchSteps)
	fmt.Printf("anchors onset=%v peak=%v recovery=%v\n", res.Path[0], res.Path[8], res.Path[len(res.Path)-1])
	fmt.Printf("local-shape=%dx%d\n", res.LocalCost.Rows(), res.LocalCost.Cols())

	if res.Distance < 0.10 && stretchSteps <= 3 {
		fmt.Println("maintenance decision=signature match")
	} else {
		fmt.Println("maintenance decision=manual review")
	}

	// Output:
	// vibration distance=0.0776
	// path-steps=17 stretch-steps=3
	// anchors onset={0 0} peak={6 8} recovery={13 16}
	// local-shape=14x17
	// maintenance decision=signature match
}
