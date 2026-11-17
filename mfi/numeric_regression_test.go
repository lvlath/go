// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Independent arithmetic oracles and adversarial numeric transitions, external only.
package mfi_test

import (
	"math"
	"math/big"
	"testing"

	"github.com/lvlath/go/mfi"
)

// TestNumerics_ExactRationalOracle uses exact rational quantities from flat OHLC
// observations, not the package's classify, ratio or ring recurrence.
func TestNumerics_ExactRationalOracle(t *testing.T) {
	const count = 440
	prices, volume := make([]float64, count), make([]float64, count)
	for i := range prices {
		prices[i] = float64(20000 + (i*23)%67)
		volume[i] = float64((i * 11) % 97)
	}
	bars := sampleBars(prices, volume)
	for _, period := range []int{1, 2, 14, 73, 127} {
		computed, err := mfi.Compute(bars, mfi.Options{Period: period})
		if err != nil {
			t.Fatal(err)
		}
		for i := period; i < count; i++ {
			var positive, negative big.Rat
			for j := i - period + 1; j <= i; j++ {
				price := new(big.Rat).SetFloat64(prices[j])
				quantity := new(big.Rat).SetFloat64(volume[j])
				flow := new(big.Rat).Mul(price, quantity)
				switch {
				case prices[j] > prices[j-1]:
					positive.Add(&positive, flow)
				case prices[j] < prices[j-1]:
					negative.Add(&negative, flow)
				}
			}
			if !computed.Valid[i] {
				t.Fatalf("period=%d index=%d: missing valid window", period, i)
			}
			want := 50.0
			if positive.Sign() > 0 && negative.Sign() == 0 {
				want = 100
			} else if positive.Sign() == 0 && negative.Sign() > 0 {
				want = 0
			} else if positive.Sign() > 0 && negative.Sign() > 0 {
				total := new(big.Rat).Add(&positive, &negative)
				fraction := new(big.Rat).Quo(&positive, total)
				precise := new(big.Float).SetPrec(256).SetRat(fraction)
				f, _ := precise.Float64()
				want = 100 * f
			}
			if math.Abs(computed.Values[i]-want) > 2e-12 {
				t.Fatalf("period=%d index=%d: got %.17g, exact rational %.17g", period, i, computed.Values[i], want)
			}
		}
	}
}

func TestNumerics_IndependentRatioWithUnrepresentableTotal(t *testing.T) {
	// Positive and negative window sums are separately finite, but their sum
	// exceeds MaxFloat64. The ratio must still be returned without pos+neg.
	large := math.MaxFloat64
	bars := sampleBars([]float64{1, 2, 1}, []float64{0, large * 0.3, large * 0.6})
	result, err := mfi.Compute(bars, mfi.Options{Period: 2})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Valid[2] || !finiteTest(result.Values[2]) || math.Abs(result.Values[2]-50) > 1e-12 {
		t.Fatalf("unexpected bounded ratio: %+v", result)
	}
}

func TestNumerics_OverflowAtomicAfterWrap(t *testing.T) {
	stream, err := mfi.NewPeriod(2)
	if err != nil {
		t.Fatal(err)
	}
	for _, bar := range []mfi.Bar{
		sampleBar(1, 0),
		sampleBar(2, 1),
		sampleBar(3, 1),
		sampleBar(4, math.MaxFloat64*0.135),
	} {
		if _, _, err = stream.Push(bar); err != nil {
			t.Fatal(err)
		}
	}
	before := stream.Snapshot()
	// After the first successful wrap, the retained +4 bar contributes about
	// 0.54*MaxFloat64. Inserting +5 contributes another 0.8*MaxFloat64.
	// Both raw flows fit float64, but their joint positive window sum does not.
	_, _, err = stream.Push(sampleBar(5, math.MaxFloat64*0.16))
	mustErrorIs(t, err, mfi.ErrOverflow)
	mustSnapshotBits(t, stream.Snapshot(), before)
}

// TestNumerics_BothSignsOverflowAtomically covers negative/positive sum
// overflow on an unscheduled eviction/insertion, independently of raw flow
// overflow. Each signed raw flow fits; only the new window sum is too large.
func TestNumerics_BothSignsOverflowAtomically(t *testing.T) {
	maxN := math.MaxFloat64
	fixtures := []struct {
		name   string
		prices []float64
		volume []float64
	}{
		{"positive", []float64{1, 2, 3, 4, 5}, []float64{0, maxN * 0.15, maxN * 0.1, maxN * 0.075, maxN * 0.1}},
		{"negative", []float64{5, 4, 3, 2, 1}, []float64{0, maxN * 0.075, maxN * 0.1, maxN * 0.15, maxN * 0.5}},
	}
	for _, fixture := range fixtures {
		t.Run(fixture.name, func(t *testing.T) {
			stream, err := mfi.NewPeriod(3)
			if err != nil {
				t.Fatal(err)
			}
			for i := 0; i < 4; i++ {
				if _, _, err = stream.Push(sampleBar(fixture.prices[i], fixture.volume[i])); err != nil {
					t.Fatal(err)
				}
			}
			before := stream.Snapshot()
			value, valid, err := stream.Push(sampleBar(fixture.prices[4], fixture.volume[4]))
			mustErrorIs(t, err, mfi.ErrOverflow)
			if value != 0 || valid {
				t.Fatalf("invalid partial result value=%g valid=%t", value, valid)
			}
			mustSnapshotBits(t, stream.Snapshot(), before)
		})
	}
}

func TestNumerics_MaxTypicalAndTinyVolume(t *testing.T) {
	stream, err := mfi.NewPeriod(1)
	if err != nil {
		t.Fatal(err)
	}
	maxPrice := math.MaxFloat64
	for _, bar := range []mfi.Bar{
		sampleBar(maxPrice, 0),
		sampleBar(maxPrice, math.SmallestNonzeroFloat64),
	} {
		_, _, err = stream.Push(bar)
		if err != nil {
			t.Fatal(err)
		}
	}
	got := stream.Snapshot()
	mustEqualInt(t, len(got.FloatState), 7, "P=1 snapshot float shape")
	if got.Count != 2 {
		t.Fatalf("count=%d", got.Count)
	}
}
