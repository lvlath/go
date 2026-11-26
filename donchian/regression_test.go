// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package donchian_test checks cross-mode identities, revision horizons, and numeric edge cases.
// This file uses only exported production operations and preserves explicit fixture ownership.
// AI-Hints: A changed fixture must not relax the independently derived mathematical expectation.
package donchian_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/lvlath/go/donchian"
)

// TestDonchian_ExcludeEqualsShiftedInclude checks the algebraic one-row relationship between window modes.
// The previous-window row i must equal the inclusive row i-1, including readiness.
//
// Implementation:
//   - Stage 1: Build one signed history and periods covering readiness, n, n+1, and MaxInt.
//   - Stage 2: Call the named inclusive and exclusive wrappers on identical data.
//   - Stage 3: Compare exclude row i with include row i-1, including all mask and payload bits.
//
// Behavior highlights:
//   - Compare exclude row i with include row i-1, including all mask and payload bits.
//
// Inputs:
//   - a 257-row history and periods spanning one through MaxInt.
//
// Returns:
//   - No value; a violated assertion fails the owning test goroutine.
//
// Errors:
//   - Unexpected public errors fail the test; expected failures are classified with errors.Is.
//
// Determinism:
//   - For each fixed period, corresponding rows are compared in increasing index order.
//
// Complexity:
//   - O(k*n) for k tested periods.
//
// Notes:
//   - The checks use only public results and error identities; no private state is inspected.
//
// AI-Hints:
//   - Compare all channels and the mask, not just one extremum.
func TestDonchian_ExcludeEqualsShiftedInclude(t *testing.T) {
	high, low := syntheticBars(257)
	for _, period := range []int{1, 2, 14, 64, 257, 258, math.MaxInt} {
		included, err := donchian.ComputeIncludeCurrent(context.Background(), high, low, period)
		if err != nil {
			t.Fatal(err)
		}
		excluded, err := donchian.ComputeExcludeCurrent(context.Background(), high, low, period)
		if err != nil {
			t.Fatal(err)
		}
		// Excluding i is the same mathematical window as including i-1.
		for i := 1; i < len(high); i++ {
			if excluded.Valid[i] != included.Valid[i-1] {
				t.Fatalf("P=%d row=%d readiness shift", period, i)
			}
			assertFloats(t, []float64{excluded.Upper[i], excluded.Middle[i], excluded.Lower[i]},
				[]float64{included.Upper[i-1], included.Middle[i-1], included.Lower[i-1]})
		}
	}
}

// TestDonchian_RevisionInfluenceHorizon bounds which rows may change after one historical correction.
// Only ready windows actually containing the revised observation may change.
//
// Implementation:
//   - Stage 1: Choose revisions at the first, readiness-boundary, interior, and final observations.
//   - Stage 2: Insert strict opposite maximal extrema at one chosen observation and recompute.
//   - Stage 3: Check exact window membership, unchanged exterior rows, and unchanged readiness.
//
// Behavior highlights:
//   - Check exact window membership, unchanged exterior rows, and unchanged readiness.
//
// Inputs:
//   - five revision positions, both modes, and an extreme two-sided correction.
//
// Returns:
//   - No value; a violated assertion fails the owning test goroutine.
//
// Errors:
//   - Unexpected public errors fail the test; expected failures are classified with errors.Is.
//
// Determinism:
//   - Membership is derived from explicit window endpoints, not from the implementation queues.
//
// Complexity:
//   - O(n*P) oracle work per revision.
//
// Notes:
//   - The checks use only public results and error identities; no private state is inspected.
//
// AI-Hints:
//   - Do not recompute a shifted or unbounded correction influence interval.
func TestDonchian_RevisionInfluenceHorizon(t *testing.T) {
	const period = 14
	high, low := syntheticBars(75)
	for _, exclude := range []bool{false, true} {
		opts := donchian.Options{Period: period, ExcludeCurrent: exclude}
		before := mustCompute(t, high, low, opts)
		for _, revision := range []int{0, period - 1, period, 38, len(high) - 1} {
			t.Run(fmt.Sprintf("exclude=%t/revised=%d", exclude, revision), func(t *testing.T) {
				h, l := append([]float64(nil), high...), append([]float64(nil), low...)
				// A strict two-sided extremum makes inclusion of this observation observable.
				h[revision], l[revision] = math.MaxFloat64, -math.MaxFloat64
				after := mustCompute(t, h, l, opts)
				assertResult(t, after, exactOracle(h, l, opts))
				for i, valid := range after.Valid {
					end := i + 1
					if exclude {
						end = i
					}
					contains := valid && revision >= end-period && revision < end
					if contains {
						assertFloats(t, []float64{after.Upper[i], after.Middle[i], after.Lower[i]},
							[]float64{math.MaxFloat64, 0, -math.MaxFloat64})
					} else {
						assertFloats(t, []float64{after.Upper[i], after.Middle[i], after.Lower[i]},
							[]float64{before.Upper[i], before.Middle[i], before.Lower[i]})
					}
					if after.Valid[i] != before.Valid[i] {
						t.Fatal("revision changed readiness")
					}
				}
			})
		}
	}
}

// TestDonchian_ExhaustiveShortHistories enumerates every four-row history over a small signed candle alphabet.
// All six valid high/low pairs over {-1,0,1} cover equalities and mixed queue movement.
//
// Implementation:
//   - Stage 1: Enumerate all 6^4 sequences of valid candles over the alphabet {-1,0,1}.
//   - Stage 2: Run periods 1 through 5 and both modes for every four-observation history.
//   - Stage 3: Compare the complete artifact with the independent direct-window oracle.
//
// Behavior highlights:
//   - Compare the complete artifact with the independent direct-window oracle.
//
// Inputs:
//   - 1296 histories, five periods, and both window modes.
//
// Returns:
//   - No value; a violated assertion fails the owning test goroutine.
//
// Errors:
//   - Unexpected public errors fail the test; expected failures are classified with errors.Is.
//
// Determinism:
//   - Base-six enumeration covers each declared history exactly once in a fixed order.
//
// Complexity:
//   - O(6^4*k*n*P) fixed exhaustive test work.
//
// Notes:
//   - The checks use only public results and error identities; no private state is inspected.
//
// AI-Hints:
//   - This is exhaustive over the declared alphabet, not over all binary64 histories.
func TestDonchian_ExhaustiveShortHistories(t *testing.T) {
	// These six candles exhaust all ordered high/low pairs over {-1,0,1}.
	candles := [][2]float64{{-1, -1}, {0, -1}, {0, 0}, {1, -1}, {1, 0}, {1, 1}}
	const length = 4
	const histories = 6 * 6 * 6 * 6
	for code := 0; code < histories; code++ {
		high, low := make([]float64, length), make([]float64, length)
		remaining := code
		for i := range high {
			candle := candles[remaining%len(candles)]
			high[i], low[i] = candle[0], candle[1]
			remaining /= len(candles)
		}
		for period := 1; period <= length+1; period++ {
			for _, exclude := range []bool{false, true} {
				opts := donchian.Options{Period: period, ExcludeCurrent: exclude}
				assertResult(t, mustCompute(t, high, low, opts), exactOracle(high, low, opts))
			}
		}
	}
}

// TestDonchian_MidpointAdjacentPairsAllNormalExponents checks midpoint ties throughout the normal exponent range.
// Adjacent positive and negative numbers test nearest-even parity without random coverage gaps.
//
// Implementation:
//   - Stage 1: Visit all finite normal exponent fields and six significand-edge positions.
//   - Stage 2: Form finite adjacent pairs for both signs and run the public Period=1 path.
//   - Stage 3: Check nearest-even rounding against the independent exact rational midpoint.
//
// Behavior highlights:
//   - Check nearest-even rounding against the independent exact rational midpoint.
//
// Inputs:
//   - all 2046 finite normal exponent fields and six significand-edge positions.
//
// Returns:
//   - No value; a violated assertion fails the owning test goroutine.
//
// Errors:
//   - Unexpected public errors fail the test; expected failures are classified with errors.Is.
//
// Determinism:
//   - Exponent, fraction position, and sign are explicit nested loops rather than random samples.
//
// Complexity:
//   - O(e*k) bounded rational comparisons.
//
// Notes:
//   - The checks use only public results and error identities; no private state is inspected.
//
// AI-Hints:
//   - Do not test only exponent zero or only one sign of halfway cases.
func TestDonchian_MidpointAdjacentPairsAllNormalExponents(t *testing.T) {
	const fractionBits = 52
	const maximumFiniteExponent = 2046
	const fractionMask = uint64(1)<<fractionBits - 1
	for exponent := uint64(1); exponent <= maximumFiniteExponent; exponent++ {
		for _, fraction := range []uint64{0, 1, 2, fractionMask - 2, fractionMask - 1, fractionMask} {
			a := math.Float64frombits(exponent<<fractionBits | fraction)
			b := math.Nextafter(a, math.Inf(1))
			if math.IsInf(b, 1) {
				continue
			}
			// Adjacent finite numbers exercise exact midpoint ties throughout the exponent range.
			for _, bounds := range [][2]float64{{b, a}, {-a, -b}} {
				result := mustCompute(t, []float64{bounds[0]}, []float64{bounds[1]}, donchian.Options{Period: 1})
				assertFloats(t, result.Middle, []float64{exactMidpoint(bounds[0], bounds[1])})
			}
		}
	}
}

// TestDonchian_OutputCapacityAndAppendIsolation checks every output capacity and cross-channel append boundary.
// Numeric channels may share storage only as disjoint, fully capped regions.
//
// Implementation:
//   - Stage 1: Compute both modes and a separate expected artifact.
//   - Stage 2: Check each channel capacity and append through temporary numeric and mask views.
//   - Stage 3: Require the original artifact unchanged after every append.
//
// Behavior highlights:
//   - Require the original artifact unchanged after every append.
//
// Inputs:
//   - all numeric channels and the boolean mask in both modes.
//
// Returns:
//   - No value; a violated assertion fails the owning test goroutine.
//
// Errors:
//   - Unexpected public errors fail the test; expected failures are classified with errors.Is.
//
// Determinism:
//   - Upper, Middle, Lower, and Valid are checked separately in fixed order.
//
// Complexity:
//   - O(n*P) oracle work plus O(n) append copies.
//
// Notes:
//   - The checks use only public results and error identities; no private state is inspected.
//
// AI-Hints:
//   - Testing only Upper does not protect the Middle-to-Lower partition.
func TestDonchian_OutputCapacityAndAppendIsolation(t *testing.T) {
	high, low := syntheticBars(31)
	for _, exclude := range []bool{false, true} {
		opts := donchian.Options{Period: 3, ExcludeCurrent: exclude}
		want := exactOracle(high, low, opts)
		got := mustCompute(t, high, low, opts)
		for _, channel := range [][]float64{got.Upper, got.Middle, got.Lower} {
			if len(channel) != len(high) || cap(channel) != len(high) {
				t.Fatalf("numeric len/cap=%d/%d, want %d", len(channel), cap(channel), len(high))
			}
			// Appending through a temporary view must not enter the next packed channel.
			extended := append(channel, 999)
			if extended[len(channel)] != 999 {
				t.Fatal("append did not store the new value")
			}
			assertResult(t, got, want)
		}
		if cap(got.Valid) != len(high) {
			t.Fatal("mask has excess capacity")
		}
		extendedMask := append(got.Valid, true)
		if !extendedMask[len(got.Valid)] {
			t.Fatal("appended mask bit missing")
		}
		assertResult(t, got, want)
	}
}

// TestDonchian_SharedInputBacking admits read-only overlapping and identical input slices.
// Owning output must not depend on the caller providing disjoint price buffers.
//
// Implementation:
//   - Stage 1: Create overlapping high/low views and an identical-slice pair over one price array.
//   - Stage 2: Compute both modes without modifying either shared view.
//   - Stage 3: Compare complete results with the oracle and the backing array with its original bits.
//
// Behavior highlights:
//   - Compare complete results with the oracle and the backing array with its original bits.
//
// Inputs:
//   - overlapping monotone input views and an identical high/low slice.
//
// Returns:
//   - No value; a violated assertion fails the owning test goroutine.
//
// Errors:
//   - Unexpected public errors fail the test; expected failures are classified with errors.Is.
//
// Determinism:
//   - The same monotone backing array is used read-only for all calls.
//
// Complexity:
//   - O(n*P) oracle comparisons.
//
// Notes:
//   - The checks use only public results and error identities; no private state is inspected.
//
// AI-Hints:
//   - Aliasing between inputs is permitted; mutation during a call is not.
func TestDonchian_SharedInputBacking(t *testing.T) {
	// Overlapping input slices are valid: the library only reads them.
	prices := []float64{-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7}
	original := append([]float64(nil), prices...)
	for _, exclude := range []bool{false, true} {
		opts := donchian.Options{Period: 3, ExcludeCurrent: exclude}
		high, low := prices[1:], prices[:len(prices)-1]
		assertResult(t, mustCompute(t, high, low, opts), exactOracle(high, low, opts))
		assertResult(t, mustCompute(t, prices, prices, opts), exactOracle(prices, prices, opts))
		assertFloats(t, prices, original)
	}
}

// TestDonchian_NonFiniteAtScanBoundaries checks domain admission around cancellation polling boundaries.
// A later NaN or infinity must beat a remembered finite geometry failure even during warmup.
//
// Implementation:
//   - Stage 1: Inject each non-finite class near block edges into either input channel.
//   - Stage 2: Add an earlier finite geometry defect and run ready and wholly masked policies.
//   - Stage 3: Require ErrNonFinite and nil output regardless of poll-boundary or mode position.
//
// Behavior highlights:
//   - Require ErrNonFinite and nil output regardless of poll-boundary or mode position.
//
// Inputs:
//   - all non-finite classes in either channel near row-block boundaries.
//
// Returns:
//   - No value; a violated assertion fails the owning test goroutine.
//
// Errors:
//   - Unexpected public errors fail the test; expected failures are classified with errors.Is.
//
// Determinism:
//   - Index, non-finite class, channel, mode, and period are explicit nested case orders.
//
// Complexity:
//   - O(k*n) across the fixed boundary matrix.
//
// Notes:
//   - The checks use only public results and error identities; no private state is inspected.
//
// AI-Hints:
//   - The last excluded bar still belongs to the validated input, including Period=MaxInt.
func TestDonchian_NonFiniteAtScanBoundaries(t *testing.T) {
	const length = 2051
	for _, index := range []int{0, 1023, 1024, 2048, length - 1} {
		for _, bad := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
			for _, inHigh := range []bool{false, true} {
				for _, exclude := range []bool{false, true} {
					for _, period := range []int{1, math.MaxInt} {
						high, low := syntheticBars(length)
						// Finite bad geometry at zero must not hide a later non-finite value.
						high[0], low[0] = 0, 1
						if inHigh {
							high[index] = bad
						} else {
							low[index] = bad
						}
						got, err := donchian.Compute(context.Background(), high, low,
							donchian.Options{Period: period, ExcludeCurrent: exclude})
						if got != nil || !errors.Is(err, donchian.ErrNonFinite) {
							t.Fatalf("index=%d high=%t exclude=%t period=%d: result=%v error=%v", index, inHigh, exclude, period, got, err)
						}
					}
				}
			}
		}
	}
}

// TestDonchian_MarketFixtureIntegrity checks the frozen market fixture and its direct-window parity.
// Dates, OHLC geometry, and fresh storage protect executable examples from accidental fixture drift.
//
// Implementation:
//   - Stage 1: Check the frozen count, date endpoints, increasing dates, and finite OHLC geometry.
//   - Stage 2: Compare six periods in both modes with the direct-window oracle.
//   - Stage 3: Mutate one fixture copy and require a fresh call to retain its original values.
//
// Behavior highlights:
//   - Mutate one fixture copy and require a fresh call to retain its original values.
//
// Inputs:
//   - 58 sourced rows and six readiness-relevant periods in both modes.
//
// Returns:
//   - No value; a violated assertion fails the owning test goroutine.
//
// Errors:
//   - Unexpected public errors fail the test; expected failures are classified with errors.Is.
//
// Determinism:
//   - The oldest-first literal and explicit period list fix all comparisons.
//
// Complexity:
//   - O(n*P) per direct oracle plus O(n) fixture validation.
//
// Notes:
//   - The checks use only public results and error identities; no private state is inspected.
//
// AI-Hints:
//   - This validates internal fixture consistency, not exchange authenticity or historical availability.
func TestDonchian_MarketFixtureIntegrity(t *testing.T) {
	candles := ibmDailyCandles()
	const count = 58
	if len(candles) != count || candles[0].date != "2026-09-01" || candles[count-1].date != "2026-11-20" {
		t.Fatal("frozen vendor interval or count changed")
	}
	for i, candle := range candles {
		if i > 0 && candles[i-1].date >= candle.date {
			t.Fatal("session order is not strictly ascending")
		}
		for _, price := range []float64{candle.open, candle.high, candle.low, candle.close} {
			if math.IsNaN(price) || math.IsInf(price, 0) {
				t.Fatal("non-finite vendor price")
			}
		}
		if candle.open < candle.low || candle.open > candle.high || candle.close < candle.low || candle.close > candle.high {
			t.Fatalf("OHLC geometry at %s", candle.date)
		}
	}
	high, low := candleChannels(candles)
	for _, period := range []int{1, 10, 20, 50, 58, 59} {
		for _, exclude := range []bool{false, true} {
			opts := donchian.Options{Period: period, ExcludeCurrent: exclude}
			assertResult(t, mustCompute(t, high, low, opts), exactOracle(high, low, opts))
		}
	}
	// The fixed fixture is fresh per call, not mutable package-global history.
	candles[0].high = math.NaN()
	if math.IsNaN(ibmDailyCandles()[0].high) {
		t.Fatal("fixture calls share backing storage")
	}
}

// TestDonchian_SampledSpreadFixture protects simultaneous-leg aggregation in the spread example.
// Independently subtracting leg extrema must not replace the range of actual paired differences.
//
// Implementation:
//   - Stage 1: obtain the engineered paired-observation spread bars.
//   - Stage 2: check their exact expected high, low, and closing spread values.
//   - Stage 3: compare both channel modes with direct-window rational expectations.
//
// Behavior highlights:
//   - Both legs vary within a bar; incorrect independent-leg aggregation changes its bounds.
//
// Inputs:
//   - Fourteen engineered bars, with twelve baseline bars followed by two changed spreads.
//
// Returns:
//   - No value; fixture drift or a channel mismatch fails the owning test.
//
// Errors:
//   - Any public error on these finite ordered pairs is unexpected.
//
// Determinism:
//   - Expected quarter-unit values are literal, independent of the aggregation helper.
//
// Complexity:
//   - O(n*P) independent oracle work with n=14 and P=12; O(n) output storage.
//
// Notes:
//   - These are sampled marks, not all intrabar ticks, executable spreads, or vendor observations.
//
// AI-Hints:
//   - Do not seed signed extrema with zero or subtract separately aggregated leg highs/lows.
func TestDonchian_SampledSpreadFixture(t *testing.T) {
	const count = 14
	const baseline = 12
	high, low, closes := syntheticSpreadBars()
	if len(high) != count || len(low) != count || len(closes) != count {
		t.Fatal("sampled spread history shape changed")
	}
	for i := 0; i < baseline; i++ {
		assertFloats(t, []float64{high[i], low[i], closes[i]}, []float64{-1.75, -2.25, -2.25})
	}
	assertFloats(t, high[baseline:], []float64{-1, -1.25})
	assertFloats(t, low[baseline:], []float64{-1.5, -1.75})
	assertFloats(t, closes[baseline:], []float64{-1, -1.25})
	for _, exclude := range []bool{false, true} {
		opts := donchian.Options{Period: baseline, ExcludeCurrent: exclude}
		assertResult(t, mustCompute(t, high, low, opts), exactOracle(high, low, opts))
	}
}
