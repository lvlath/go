// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package main demonstrates aligning two voice-command amplitude envelopes
// using Dynamic Time Warping with lvlath/dtw.
package main

import (
	"fmt"

	"github.com/lvlath/go/dtw"
)

// ExampleDTW_audioAlignment aligns a clean voice-command envelope with a noisy stretched recording.
//
// Scenario:
//
//	A voice-control system stores a clean amplitude envelope for a short command.
//	The observed recording contains the same command, but the speaker holds one
//	vowel slightly longer and the microphone adds mild envelope noise. A strict
//	index-by-index comparison would over-penalize the stretched frames.
//
//	The example uses scalar DTW on amplitude-envelope samples. In production, the
//	same structure can be applied to MFCC-derived scores, neural embedding distances,
//	or a precomputed acoustic cost matrix through dtw.AlignCostMatrix.
//
// Playground: https://go.dev/play/p/yCVsUCFu9-b
//
// Implementation:
//   - Stage 1: Define a clean 13-frame command envelope.
//   - Stage 2: Define a 16-frame observed envelope with stretched attack/vowel/release phases.
//   - Stage 3: Run canonical dtw.Align with a bounded Sakoe-Chiba window.
//   - Stage 4: Request path tracking to inspect the temporal explanation.
//   - Stage 5: Count stretch steps and print semantic path anchors.
//   - Stage 6: Classify the recording as accepted or manual-review from distance and stretch count.
//
// Behavior highlights:
//   - Uses current public dtw API: Align, Result, PathOrError, Coord.I, Coord.J.
//   - Uses WithWindow(4) to permit realistic speech-speed drift.
//   - Uses WithSlopePenalty(0.05) so held frames are not completely free.
//   - Keeps the dataset fixed and deterministic for executable documentation.
//   - Demonstrates result consumption instead of panicking on ordinary errors.
//
// Inputs:
//   - None. The clean and observed envelopes are fixed in the example.
//
// Returns:
//   - None.
//
// Errors:
//   - Prints unexpected DTW or path-surface errors and returns early.
//
// Determinism:
//   - Stable for the same samples, option policy, row-major DP traversal, and
//     deterministic backtracking tie-break law.
//
// Complexity:
//   - Time O(N*M), where N=len(clean) and M=len(observed).
//   - Memory O(N*M), because WithReturnPath(true) requires accumulated matrix storage.
//
// Notes:
//   - This example uses amplitude envelopes for readability.
//   - For production acoustic systems, use AlignCostMatrix when another model has
//     already produced frame-to-frame costs.
//
// AI-Hints:
//   - Do not use dtw.DTWOptions; the current API uses Options for legacy DTW and
//     functional options for canonical Align.
//   - Do not index dtw.Coord as p[0] or p[1]; use p.I and p.J.
//   - Do not treat stretch as a failure by itself; controlled stretch is the reason
//     DTW is useful for speech alignment.
func ExampleDTW_audioAlignment() {
	clean := []float64{
		0.02,  // room tone before the command
		0.10,  // breath / onset
		0.28,  // consonant attack starts
		0.62,  // attack grows
		0.95,  // vowel peak
		0.72,  // vowel decay
		0.34,  // transition
		-0.10, // phase inversion / microphone envelope dip
		-0.55, // release valley
		-0.82, // deepest release
		-0.40, // recovery
		0.05,  // return toward silence
		0.12,  // trailing room tone
	}

	noisy := []float64{
		0.03,  // room tone
		0.08,  // softer onset
		0.18,  // extra onset frame
		0.31,  // attack begins later
		0.60,  // attack matches clean frame 3
		0.90,  // near vowel peak
		0.76,  // held vowel
		0.50,  // additional vowel decay frame
		0.22,  // transition stretch
		-0.12, // envelope dip
		-0.50, // release valley starts
		-0.78, // deepest release
		-0.58, // held release
		-0.28, // recovery stretch
		0.02,  // near silence
		0.10,  // trailing room tone
	}

	res, err := dtw.Align(
		clean,
		noisy,
		dtw.WithWindow(4),
		dtw.WithSlopePenalty(0.05),
		dtw.WithReturnPath(true),
	)
	if err != nil {
		fmt.Println("align:", err)
		return
	}

	path, err := res.PathOrError()
	if err != nil {
		fmt.Println("path:", err)
		return
	}

	// stretchSteps counts vertical or horizontal DTW path moves.
	//   - A stretch step means one side advanced while the other side was held.
	//   - In a market-pattern scenario, stretch steps reveal delayed or prolonged phases.
	stretchSteps := 0
	for i := 1; i < len(path); i++ {
		if path[i].I == path[i-1].I || path[i].J == path[i-1].J {
			stretchSteps++
		}
	}

	fmt.Printf("voice distance=%.2f\n", res.Distance)
	fmt.Printf("path-steps=%d stretch-steps=%d\n", len(path), stretchSteps)
	fmt.Printf("anchors onset=%v attack=%v vowel=%v release=%v\n", path[0], path[4], path[8], path[len(path)-1])

	for _, anchor := range []dtw.Coord{path[4], path[8], path[12]} {
		fmt.Printf("  clean[%02d]=%+.2f -> noisy[%02d]=%+.2f\n", anchor.I, clean[anchor.I], anchor.J, noisy[anchor.J])
	}

	if res.Distance <= 1.25 && stretchSteps <= 4 {
		fmt.Println("decision=same command with stretched pronunciation")
	} else {
		fmt.Println("decision=manual review")
	}

	// Output:
	// voice distance=1.14
	// path-steps=16 stretch-steps=3
	// anchors onset={0 0} attack={3 4} vowel={6 8} release={12 15}
	//   clean[03]=+0.62 -> noisy[04]=+0.60
	//   clean[06]=+0.34 -> noisy[08]=+0.22
	//   clean[10]=-0.40 -> noisy[12]=-0.58
	// decision=same command with stretched pronunciation
}
