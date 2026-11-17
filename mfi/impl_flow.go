// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mfi defines the shared, pure mathematical kernel for price and money flow.
// This file has no mutable state or difference between batch and stream execution.
package mfi

// typical evaluates the convex average of High, Low and Close without forming
// an overflowing three-term sum. The candle has already passed validateBar.
//
// Implementation:
//   - Stage 1: anchor the mean at High, the largest of its three operands.
//   - Stage 2: add two nonpositive one-third offsets, avoiding positive overflow.
//   - Stage 3: reject a nonfinite or nonpositive computed result.
//
// Behavior highlights:
//   - Both adjustment terms are nonpositive for a valid candle, avoiding a three-price sum overflow.
//
// Inputs:
//   - bar: already validated finite-positive OHLCV value; Open and Volume do not enter the mean.
//
// Returns:
//   - float64: finite, strictly positive typical price on success.
//   - error: ErrOverflow when the computed mean cannot be published.
//
// Errors:
//   - ErrOverflow: computed mean becomes nonfinite or nonpositive despite validated input.
//
// Determinism:
//   - Evaluates offsets in Low then Close order, with Go float64 semantics.
//
// Complexity:
//   - Time O(1), Space O(1), zero allocation.
//
// Notes:
//   - This stable expression still rounds to binary64; it does not claim exact rational means.
//
// AI-Hints:
//   - Never replace this formula with (High+Low+Close)/3 near MaxFloat64.
func typical(bar Bar) (float64, error) {
	// Since both offsets are nonpositive, no intermediate can exceed High.
	value := bar.High + (bar.Low-bar.High)/3 + (bar.Close-bar.High)/3
	if !finite(value) || value <= 0 {
		return 0, ErrOverflow
	}
	return value, nil
}

// classify produces exactly one signed money-flow contribution per transition.
// Flat or zero-volume transitions still occupy one ring position but contribute
// the explicit zero pair, preserving precise rolling-window age.
//
// Implementation:
//   - Stage 1: test exact typical-price equality and zero volume.
//   - Stage 2: multiply current typical price by volume and check representability.
//   - Stage 3: return the flow in the positive or negative component only.
//
// Behavior highlights:
//   - The amount is attached to the current bar and the direction compares adjacent typical prices.
//
// Inputs:
//   - previous: last accepted finite positive typical price.
//   - current: current finite positive typical price.
//   - volume: current finite nonnegative volume in the same base unit.
//
// Returns:
//   - positive, negative: mutually exclusive nonnegative flow contributions.
//   - error: ErrOverflow if a required current-price times volume product is nonfinite.
//
// Errors:
//   - ErrOverflow: positive-volume directional raw flow cannot be represented.
//
// Determinism:
//   - Comparisons use exact float64 ordering; equality is never rounded by tolerance.
//
// Complexity:
//   - Time O(1), Space O(1), zero allocation.
//
// Notes:
//   - The caller advances one window slot even when both contributions equal zero.
//
// AI-Hints:
//   - Even if both returns are zero, the caller must advance the ring cursor.
func classify(previous, current, volume float64) (positive, negative float64, err error) {
	// Preserve a zero pair without multiplying a potentially extreme volume.
	// The outer Stream still advances the window and remembers current typical.
	if current == previous || volume == 0 {
		return 0, 0, nil
	}
	flow := current * volume
	if !finite(flow) || flow < 0 {
		return 0, 0, ErrOverflow
	}
	if current > previous {
		return flow, 0, nil
	}
	return 0, flow, nil
}

// ratio computes MFI without forming an unbounded positive/negative money ratio.
// Its denominator is at most two because the smaller sum is divided by the larger.
//
// Implementation:
//   - Stage 1: require finite nonnegative effective window sums.
//   - Stage 2: handle all exact-zero cases with defined 50, 100 or 0 outputs.
//   - Stage 3: calculate the bounded ratio using the larger sum as divisor.
//
// Behavior highlights:
//   - Exactly zero positive and negative sums publish 50; one-sided sums publish 0 or 100.
//
// Inputs:
//   - pos, neg: finite, nonnegative effective window sums, including Neumaier correction.
//
// Returns:
//   - float64: finite MFI in [0,100] on success.
//   - error: ErrOverflow when the sums or computed ratio are invalid.
//
// Errors:
//   - ErrOverflow: negative, nonfinite or unpublishable effective input/ratio.
//
// Determinism:
//   - Branch selection uses exact float64 comparison; no clipping is permitted.
//
// Complexity:
//   - Time O(1), Space O(1), zero allocation.
//
// Notes:
//   - Tiny normalized quotients may underflow; no arbitrary clamp or epsilon is introduced.
//
// AI-Hints:
//   - Do not form pos+neg, which can overflow even when both sums are finite.
func ratio(pos, neg float64) (float64, error) {
	if !finite(pos) || !finite(neg) || pos < 0 || neg < 0 {
		return 0, ErrOverflow
	}
	switch {
	case pos == 0 && neg == 0:
		return 50, nil
	case neg == 0:
		return 100, nil
	case pos == 0:
		return 0, nil
	}

	var result float64
	// Keep the quotient <= 1 in both branches; pos+neg is never formed.
	if pos >= neg {
		result = 100 / (1 + neg/pos)
	} else {
		quotient := pos / neg
		result = 100 * (quotient / (1 + quotient))
	}
	if !finite(result) {
		return 0, ErrOverflow
	}
	return result, nil
}
