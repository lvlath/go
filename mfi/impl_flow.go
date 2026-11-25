// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mfi defines the shared, pure mathematical kernel for price and money flow.
// This file has no mutable state or difference between batch and stream execution.
package mfi

// typical evaluates the correctly rounded mean of the three validated prices.
// It delegates to the allocation-free exact binary-input kernel, so equal exact
// sums always give identical values regardless of candle geometry.
//
// Implementation:
//   - Stage 1: use canonicalTypical after validateBar has admitted the OHLC domain.
//   - Stage 2: return the single nearest-even binary64 rounding of the exact mean.
//
// Behavior highlights:
//   - The mean is finite and positive even for mixed subnormal and near-maximum prices.
//
// Inputs:
//   - bar: validated finite positive OHLC; Open and Volume are not mean operands.
//
// Returns:
//   - float64: canonical typical price, bounded by Low and High.
//
// Errors:
//   - None for validated prices; the exact convex mean is representable.
//
// Determinism:
//   - Equal exact sums yield identical bits; no order-dependent floating sum is used.
//
// Complexity:
//   - Time O(1), Space O(1), zero heap allocation.
//
// AI-Hints:
//   - Never round High/3, Low/3 and Close/3 independently or introduce an epsilon.
func typical(bar Bar) float64 {
	return canonicalTypical(bar.High, bar.Low, bar.Close)
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
	// The admitted operands are nonnegative; only nonfinite multiplication can fail.
	if !finite(flow) {
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
//   - error: ErrOverflow when the effective input sums are invalid.
//
// Errors:
//   - ErrOverflow: negative or nonfinite effective input sums.
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
	// The admitted nonnegative finite inputs give a denominator in [1,2]
	// and a quotient in [0,1]; the result cannot be nonfinite.
	return result, nil
}
