// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package atr provides exact True Range and Wilder Average True Range computations.
// This file implements first-bar-safe correctly rounded true range.
package atr

import "math"

// trueRange rounds the exact candle/previous-close envelope once.
// For low <= high, max(H-L, abs(H-Cprev), abs(L-Cprev)) equals max(H,Cprev)-min(L,Cprev).
//
// Implementation:
//   - Stage 1: Use only the current endpoints when no accepted previous close exists.
//   - Stage 2: Otherwise extend the endpoints to include the last accepted close.
//   - Stage 3: Subtract once, canonicalize zero, and reject an infinite required range.
//
// Behavior highlights:
//   - Endpoint comparisons select original floats exactly; no intermediate difference is rounded.
//
// Inputs:
//   - high, low: finite validated current interval.
//   - previousClose: finite last accepted close; ignored when hasPrevious is false.
//   - hasPrevious: true only after at least one successful transition.
//
// Returns:
//   - float64: finite nonnegative rounded TR, or +0 with an error.
//
// Errors:
//   - ErrOverflow: the required rounded range is infinite.
//
// Determinism:
//   - One subtraction after exact endpoint selection; zero output is canonical +0.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - A finite rounded shoulder above the exact MaxFloat64 value remains admissible.
//
// AI-Hints:
//   - Never substitute the current close, open, or zero for an absent historical close.
//   - Do not suppress TR overflow merely because a later average could be finite.
func trueRange(high, low, previousClose float64, hasPrevious bool) (float64, error) {
	// The first bar never consults the physically present but semantically absent close.
	if hasPrevious {
		// A downward gap extends the envelope above the current high.
		if previousClose > high {
			high = previousClose
		}
		// An upward gap extends the envelope below the current low.
		if previousClose < low {
			low = previousClose
		}
	}
	// Subtract exactly selected endpoints once; Abs only canonicalizes the zero sign.
	tr := math.Abs(high - low)
	// TR itself is mandatory, even when smoothing might reduce its magnitude.
	if math.IsInf(tr, 0) {
		return 0, ErrOverflow
	}
	return tr, nil
}
