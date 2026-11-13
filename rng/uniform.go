// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package rng implements the single fixed mapping from raw words to open uniforms.
// This file preserves exact midpoint arithmetic with no endpoint rejection draws.
package rng

const (
	// uniformDiscardBits removes the 12 least-significant bits to retain 52 bin bits.
	uniformDiscardBits = 12

	// uniformBinHalf selects the exact midpoint of each of the 2^52 bins.
	uniformBinHalf = 0.5

	// uniformBinScale is exactly 2^-52 in binary64; it never introduces rounding.
	uniformBinScale = 0x1p-52
)

// openFromRaw converts the high 52 bits into one strictly open uniform midpoint.
// The largest intermediate 2^52-0.5 is representable exactly in binary64.
//
// Implementation:
//   - Stage 1: discard the twelve lowest raw-word bits.
//   - Stage 2: convert the remaining integer to float64 and add exact midpoint 0.5.
//   - Stage 3: scale by exactly 2^-52, preserving both strict endpoint exclusions.
//
// Complexity:
//   - Time O(1), Space O(1); it is pure and consumes no generator words.
//
// AI-Hints:
//   - The 53-bit midpoint variant can round to 1.0 and is forbidden.
func openFromRaw(x uint64) float64 {
	return (float64(x>>uniformDiscardBits) + uniformBinHalf) * uniformBinScale
}

// UniformOpen returns an open-interval float64 from exactly one raw word.
// It uses the pinned midpoint map, not a 53-bit or inclusive conversion.
//
// Implementation:
//   - Stage 1: obtain exactly one next Uint64 from the mutable state.
//   - Stage 2: project the high 52 bits through the canonical openFromRaw helper.
//   - Stage 3: publish a finite value strictly between zero and one.
//
// Behavior highlights:
//   - The exact endpoints are 2^-53 and 1-2^-53 (both included in the image).
//   - It never clears the cached sine half of an earlier Box-Muller pair.
//
// Inputs:
//   - None: the current stream determines the next raw word.
//
// Returns:
//   - float64: strictly within (0,1), in increments of 2^-52 midpoints.
//
// Errors:
//   - None: every uint64 maps to a finite open-interval midpoint.
//
// Determinism:
//   - Exactly one PCG state advance in the current call order.
//
// Complexity:
//   - Time O(1), Space O(1), zero explicit hot-path allocation.
//
// Notes:
//   - Open-interval uniformity is a numeric range guarantee, not cryptographic quality.
//
// AI-Hints:
//   - Never use a 53-bit midpoint mapping: its top value can round to exactly 1.
//   - Do not introduce rejection loops; draw consumption is part of the ABI.
func (p *pcgStream) UniformOpen() float64 {
	return openFromRaw(p.Uint64())
}
