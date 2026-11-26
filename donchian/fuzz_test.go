// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package donchian_test fuzzes public input admission and exact midpoint behavior.
// This file uses only exported production operations and preserves explicit fixture ownership.
// AI-Hints: Bound independent oracle work; do not recover a public panic as an accepted fuzz outcome.
package donchian_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/donchian"
)

// FuzzDonchian_PublicBoundaries fuzzes metadata, price admission, and both named public modes.
// Bounded decoding keeps an independent direct-window oracle affordable for every admitted input.
//
// Implementation:
//   - Stage 1: register procedural seeds and decode bounded candidate observations.
//   - Stage 2: exercise the public API and independently classify the admitted domain.
//   - Stage 3: compare error identities or exact oracle outputs for each generated case.
//
// Behavior highlights:
//   - Bounded decoding keeps an independent direct-window oracle affordable for every admitted input.
//
// Inputs:
//   - procedural byte seeds, arbitrary periods, shape mutations, and both modes.
//
// Returns:
//   - No value; a contract violation produces a reproducible fuzz failure.
//
// Errors:
//   - Invalid input is an expected classified outcome, not a recovered panic.
//
// Determinism:
//   - Each candidate has deterministic expectations; fuzz exploration order is controlled by Go.
//
// Complexity:
//   - O(n*P) per bounded admitted case; n <= maxFuzzBars.
//
// Notes:
//   - Seed execution also participates in ordinary go test.
//
// AI-Hints:
//   - No panic recovery, private exports, external fixture loading, or epsilon comparisons are used.
func FuzzDonchian_PublicBoundaries(f *testing.F) {
	seeds := [][]byte{
		{},
		encodeBars([]float64{2, 4, 3}, []float64{1, 2, 0}),
		encodeBars([]float64{-1, 0}, []float64{-5, -3}),
		encodeBars([]float64{math.MaxFloat64, math.MaxFloat64}, []float64{math.MaxFloat64, -math.MaxFloat64}),
		encodeBars([]float64{math.NaN()}, []float64{0}),
		encodeBars([]float64{0}, []float64{1}),
	}
	for _, seed := range seeds {
		f.Add(seed, 2, false, uint8(0))
		f.Add(seed, math.MaxInt, true, uint8(0))
		f.Add(seed, 0, true, uint8(1))
		f.Add(seed, 1, false, uint8(2))
		f.Add(seed, 1, true, uint8(3))
	}
	f.Fuzz(func(t *testing.T, raw []byte, period int, exclude bool, shape uint8) {
		high, low := decodeBars(raw)
		switch shape % 5 {
		case 1:
			low = append(low, 0)
		case 2:
			high = nil
		case 3:
			low = nil
		case 4:
			high = high[:len(high)/2]
		}
		// Keep the original input bits, including NaN payloads and signed zeros.
		highCopy, lowCopy := append([]float64(nil), high...), append([]float64(nil), low...)
		opts := donchian.Options{Period: period, ExcludeCurrent: exclude}
		result, err := donchian.Compute(context.Background(), high, low, opts)
		// The named public mode must preserve every canonical error category and output bit.
		var wrapped *donchian.Result
		var wrappedErr error
		if exclude {
			wrapped, wrappedErr = donchian.ComputeExcludeCurrent(context.Background(), high, low, period)
		} else {
			wrapped, wrappedErr = donchian.ComputeIncludeCurrent(context.Background(), high, low, period)
		}
		assertErrorCategories(t, wrappedErr, err)
		assertFloats(t, high, highCopy)
		assertFloats(t, low, lowCopy)
		wantErr := expectedInputError(high, low, period)
		if wantErr != nil {
			if result != nil || wrapped != nil {
				t.Fatalf("result=%v error=%v, want %v", result, err, wantErr)
			}
			assertErrorCategories(t, err, wantErr)
			return
		}
		if err != nil {
			t.Fatal(err)
		}
		assertResult(t, result, exactOracle(high, low, opts))
		assertResult(t, wrapped, result)
	})
}

// FuzzDonchian_Midpoint fuzzes exact mean rounding through the public Period=1 path.
// Finite bit patterns are ordered into valid candles before comparison with a rational oracle.
//
// Implementation:
//   - Stage 1: decode two bit patterns and skip non-finite pairs.
//   - Stage 2: order the finite pair into a valid single candle and call Compute.
//   - Stage 3: compare Middle with one independently rounded exact rational mean.
//
// Behavior highlights:
//   - Finite bit patterns are ordered into valid candles before comparison with a rational oracle.
//
// Inputs:
//   - two arbitrary binary64 bit patterns with targeted boundary seeds.
//
// Returns:
//   - No value; a contract violation produces a reproducible fuzz failure.
//
// Errors:
//   - Non-finite pairs are excluded from this target; any failure on a finite pair is unexpected.
//
// Determinism:
//   - Each candidate has deterministic expectations; fuzz exploration order is controlled by Go.
//
// Complexity:
//   - O(1) work per binary64 pair, with fixed-domain rational costs.
//
// Notes:
//   - Seed execution also participates in ordinary go test.
//
// AI-Hints:
//   - Skipping non-finite pairs tests midpoint arithmetic, not the separately tested admission errors.
func FuzzDonchian_Midpoint(f *testing.F) {
	for _, pair := range [][2]float64{
		{0, 0}, {math.MaxFloat64, math.MaxFloat64}, {math.MaxFloat64, -math.MaxFloat64},
		{math.SmallestNonzeroFloat64, math.SmallestNonzeroFloat64},
		{0, -math.SmallestNonzeroFloat64}, {0x1p-1022, math.Nextafter(0x1p-1022, 1)},
		{1, math.Nextafter(1, 2)},
	} {
		f.Add(math.Float64bits(pair[0]), math.Float64bits(pair[1]))
	}
	f.Fuzz(func(t *testing.T, highBits, lowBits uint64) {
		high, low := math.Float64frombits(highBits), math.Float64frombits(lowBits)
		if math.IsNaN(high) || math.IsNaN(low) || math.IsInf(high, 0) || math.IsInf(low, 0) {
			return
		}
		if high < low {
			high, low = low, high
		}
		result := mustCompute(t, []float64{high}, []float64{low}, donchian.Options{Period: 1})
		assertFloats(t, result.Middle, []float64{exactMidpoint(high, low)})
	})
}
