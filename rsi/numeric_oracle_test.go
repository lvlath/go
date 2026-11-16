// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package rsi_test compares Wilder RSI to a high-precision rational reference.
// The oracle computes exact seed sums and the algebraic weighted recurrence, never calling package internals.
package rsi_test

import (
	"math"
	"math/big"
	"testing"

	"github.com/lvlath/go/rsi"
)

// exactWilderReference calculates Wilder RSI as rationals from integer-valued prices.
// It is deliberately independent of float64 incremental averaging and branch selection.
func exactWilderReference(prices []int64, period int) ([]float64, []bool) {
	values := make([]float64, len(prices))
	valid := make([]bool, len(prices))
	gainSum, lossSum := new(big.Rat), new(big.Rat)
	averageGain, averageLoss := new(big.Rat), new(big.Rat)
	periodRat := new(big.Rat).SetInt64(int64(period))
	priorWeight := new(big.Rat).SetFrac64(int64(period-1), int64(period))
	currentWeight := new(big.Rat).SetFrac64(1, int64(period))

	for i := 1; i < len(prices); i++ {
		delta := prices[i] - prices[i-1]
		gain, loss := new(big.Rat), new(big.Rat)
		if delta > 0 {
			gain.SetInt64(delta)
		} else if delta < 0 {
			loss.SetInt64(-delta)
		}

		if i <= period {
			gainSum.Add(gainSum, gain)
			lossSum.Add(lossSum, loss)
			if i == period {
				averageGain.Quo(gainSum, periodRat)
				averageLoss.Quo(lossSum, periodRat)
			}
		} else {
			// Algebraically equivalent to Wilder's recurrence, but computed exactly
			// as ((P-1)/P)*old + (1/P)*new rather than subtracting in float64.
			averageGain.Add(new(big.Rat).Mul(averageGain, priorWeight), new(big.Rat).Mul(gain, currentWeight))
			averageLoss.Add(new(big.Rat).Mul(averageLoss, priorWeight), new(big.Rat).Mul(loss, currentWeight))
		}

		if i < period {
			continue
		}
		valid[i] = true
		sum := new(big.Rat).Add(averageGain, averageLoss)
		if sum.Sign() == 0 {
			values[i] = 50
			continue
		}
		ratio := new(big.Rat).Mul(averageGain, big.NewRat(100, 1))
		ratio.Quo(ratio, sum)
		values[i], _ = ratio.Float64()
	}
	return values, valid
}

// TestRSI_RationalOracle validates the numerical kernel against exact recurrence mathematics.
func TestRSI_RationalOracle(t *testing.T) {
	for _, tc := range []struct {
		name   string
		period int
		prices []int64
	}{
		{"seed-and-recurrence-P1", 1, []int64{3, 2, 2, 4, 4, 3, 5}},
		{"alternating-P2", 2, []int64{100, 104, 101, 107, 100, 108, 107, 106, 111, 110}},
		{"mixed-P7", 7, []int64{980, 986, 982, 990, 993, 989, 986, 995, 1002, 998, 991, 985, 990, 999, 1004, 996, 988, 1000, 1007, 1010, 1002, 1011}},
		{"mixed-P14", 14, []int64{1200, 1208, 1201, 1210, 1194, 1189, 1196, 1203, 1218, 1212, 1202, 1197, 1199, 1214, 1225, 1211, 1203, 1216, 1210, 1204, 1213, 1221, 1229, 1212, 1200, 1218, 1220, 1227, 1223, 1214}},
		{"flat-P4", 4, []int64{300, 300, 300, 300, 300, 300, 300, 300}},
		{"falling-P3", 3, []int64{500, 480, 469, 452, 441, 430, 417, 405}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			prices := make([]float64, len(tc.prices))
			for i, price := range tc.prices {
				prices[i] = float64(price)
			}
			wantValues, wantValid := exactWilderReference(tc.prices, tc.period)
			got, err := rsi.ComputePeriod(prices, tc.period)
			if err != nil {
				t.Fatal(err)
			}
			for i := range prices {
				mustEqualBool(t, got.Valid[i], wantValid[i], "oracle validity")
				if !got.Valid[i] {
					mustSameFloat(t, got.Values[i], 0)
					continue
				}
				if math.Abs(got.Values[i]-wantValues[i]) > 3e-12 {
					t.Fatalf("sample=%d got RSI=%.17g exact=%.17g", i, got.Values[i], wantValues[i])
				}
			}
		})
	}
}

// TestRSI_ExtremeFinitePrices exercises cancellation, convex averaging, and ratio scaling.
func TestRSI_ExtremeFinitePrices(t *testing.T) {
	for _, tc := range []struct {
		name     string
		closes   []float64
		period   int
		wantLast float64
	}{
		{"flat-near-max", []float64{math.MaxFloat64 / 2, math.MaxFloat64 / 2, math.MaxFloat64 / 2}, 2, 50},
		{"alternating-max-and-subnormal", []float64{math.MaxFloat64, math.SmallestNonzeroFloat64, math.MaxFloat64}, 2, 50},
		{"rise-near-max", []float64{math.MaxFloat64 / 4, math.MaxFloat64 / 2, math.MaxFloat64}, 2, 100},
		{"fall-near-max", []float64{math.MaxFloat64, math.MaxFloat64 / 2, math.MaxFloat64 / 4}, 2, 0},
		{"flat-subnormal", []float64{math.SmallestNonzeroFloat64, math.SmallestNonzeroFloat64, math.SmallestNonzeroFloat64}, 2, 50},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := rsi.ComputePeriod(tc.closes, tc.period)
			if err != nil {
				t.Fatal(err)
			}
			last := len(tc.closes) - 1
			mustEqualBool(t, got.Valid[last], true, "extreme valid")
			mustSameFloat(t, got.Values[last], tc.wantLast)
		})
	}
}

// TestRSI_AffineInvarianceWithRepresentableTicks checks mathematically equivalent price units.
func TestRSI_AffineInvarianceWithRepresentableTicks(t *testing.T) {
	base := []float64{120, 124, 121, 119, 127, 130, 128, 126, 133, 129, 125, 132, 140, 135, 139, 136, 143, 138, 141, 146, 144}
	// Integer-valued prices/increments and power-of-two scaling remain exact.
	shifted := make([]float64, len(base))
	for i, close := range base {
		shifted[i] = close*256 + math.Ldexp(1, 45)
	}
	for _, period := range []int{1, 2, 7, 14} {
		a, err := rsi.ComputePeriod(base, period)
		if err != nil {
			t.Fatal(err)
		}
		b, err := rsi.ComputePeriod(shifted, period)
		if err != nil {
			t.Fatal(err)
		}
		compareResultBits(t, a, b)
	}
}
