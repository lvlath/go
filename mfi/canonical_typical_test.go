// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Canonical typical-price regressions through the published batch and stream API.
package mfi_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/lvlath/go/mfi"
)

// TestFIXMFI_EqualTypical reproduces false directional flow for equal sums.
// Reversing the observations must not reverse a nonexistent direction.
func TestFIXMFI_EqualTypical(t *testing.T) {
	a := mfi.Bar{Open: 2, High: 4, Low: 1, Close: 2, Volume: 1}
	b := mfi.Bar{Open: 2, High: 3, Low: 2, Close: 2, Volume: 1}
	for _, pair := range [][2]mfi.Bar{{a, b}, {b, a}} {
		batch, err := mfi.ComputeCandles(pair[:], mfi.Options{Period: 1})
		if err != nil {
			t.Fatal(err)
		}
		if !batch.Valid[1] || batch.Values[1] != 50 {
			t.Errorf("equal sum: batch=%v valid=%v", batch.Values, batch.Valid)
		}
		stream, err := mfi.NewPeriod(1)
		if err != nil {
			t.Fatal(err)
		}
		for i, bar := range pair {
			value, valid, err := stream.Push(bar)
			if err != nil {
				t.Fatal(err)
			}
			if valid != (i == 1) || (valid && value != 50) || math.Float64bits(value) != math.Float64bits(batch.Values[i]) {
				t.Errorf("bar %d: stream=(%g,%t), batch=%g", i, value, valid, batch.Values[i])
			}
		}
	}
}

// TestFIXMFI_EqualSumFamily exhausts all ordered pairs of distinct, valid
// integer candle shapes with equal H+L+C, not just one representative per sum.
func TestFIXMFI_EqualSumFamily(t *testing.T) {
	const maxHigh = 8
	var groups [3*maxHigh + 1][]mfi.Bar
	for h := 1; h <= maxHigh; h++ {
		for l := 1; l <= h; l++ {
			for c := l; c <= h; c++ {
				groups[h+l+c] = append(groups[h+l+c], mfi.Bar{Open: float64(l), High: float64(h), Low: float64(l), Close: float64(c), Volume: 1})
			}
		}
	}
	pairs, falseFlows := 0, 0
	for _, group := range groups {
		for i, a := range group {
			for j, b := range group {
				if i == j {
					continue
				}
				pairs++
				candles := [2]mfi.Bar{a, b}
				batch, err := mfi.ComputeCandles(candles[:], mfi.Options{Period: 1})
				if err != nil {
					t.Fatal(err)
				}
				if !batch.Valid[1] || batch.Values[1] != 50 {
					falseFlows++
				}
			}
		}
	}
	t.Logf("ordered equal-sum pairs=%d false flows=%d", pairs, falseFlows)
	if pairs == 0 || falseFlows != 0 {
		t.Fatalf("false directional flow in %d of %d ordered pairs", falseFlows, pairs)
	}
}

// TestMFI_NearMaxFloatCandles compares exact bits across the binary64 domain.
// It includes equal prices, subnormal/normal boundaries, exact ties, carry,
// compact/wide spans and operands separated by the entire exponent range.
func TestMFI_NearMaxFloatCandles(t *testing.T) {
	s, err := mfi.NewPeriod(1)
	if err != nil {
		t.Fatal(err)
	}
	tiny := math.SmallestNonzeroFloat64
	normal := math.Ldexp(1, -1022)
	largest := math.MaxFloat64
	below := math.Nextafter(largest, 0)
	values := []float64{tiny, 2 * tiny, 3 * tiny, math.Nextafter(normal, 0), normal, math.Nextafter(normal, math.Inf(1)), 0x1p-1000, 0x1p-64, 0x1p-10, 0x1p-9, 0.5, 1, math.Nextafter(1, 2), 2, math.Nextafter(2, 0), 3, 4, 0x1p63, 0x1p64, 0x1p65, 0x1p1022, below, largest}
	for _, a := range values {
		for _, b := range values {
			for _, c := range values {
				assertTypicalBits(t, s, sortedPriceBar(a, b, c))
			}
		}
	}
	// A low-limb addition crosses the next high limb and requires carry propagation.
	assertTypicalBits(t, s, sortedPriceBar(tiny,
		math.Ldexp(math.Nextafter(2, 1), -959),
		math.Ldexp(math.Nextafter(2, 1), -947)))
	// Exact halfway examples at a binade boundary: use an exact integer lattice
	// with unit 2^-52 so the wide/narrow paths encounter even and odd ties.
	for i := 0; i < 64; i++ {
		a := math.Float64frombits(math.Float64bits(1) + uint64(i))
		b := math.Float64frombits(math.Float64bits(4) + uint64(i))
		for j := 0; j < 12; j++ {
			assertTypicalBits(t, s, sortedPriceBar(a, b, math.Float64frombits(math.Float64bits(2)+uint64(j))))
		}
	}
}

// TestMFI_ExactTypicalRandomBits samples both arbitrary exponents and clustered
// mantissas, giving differential coverage of compact and wide integer paths.
func TestMFI_ExactTypicalRandomBits(t *testing.T) {
	s, err := mfi.NewPeriod(1)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(1702))
	const fracMask = uint64(1<<52) - 1
	for i := 0; i < 30000; i++ {
		var x [3]float64
		exponent := rng.Intn(2047)
		for j := range x {
			e := rng.Intn(2047)
			if i%2 == 0 {
				e = max(0, min(2046, exponent+rng.Intn(19)-9))
			}
			raw := uint64(e)<<52 | rng.Uint64()&fracMask
			if raw == 0 {
				raw = 1
			}
			x[j] = math.Float64frombits(raw)
		}
		assertTypicalBits(t, s, sortedPriceBar(x[0], x[1], x[2]))
	}
}

// TestMFI_AdjacentCanonicalPrices proves that no epsilon suppresses a real
// one-ULP directional movement, including at the subnormal boundary.
func TestMFI_AdjacentCanonicalPrices(t *testing.T) {
	for _, lo := range []float64{math.SmallestNonzeroFloat64, math.Ldexp(1, -1022), 1, math.Nextafter(math.MaxFloat64, 0)} {
		hi := math.Nextafter(lo, math.Inf(1))
		for _, tc := range []struct{ a, b, want float64 }{{lo, hi, 100}, {hi, lo, 0}} {
			bars := []mfi.Bar{sampleBar(tc.a, 1), sampleBar(tc.b, 1)}
			result, err := mfi.ComputeCandles(bars, mfi.Options{Period: 1})
			if err != nil || result == nil || !result.Valid[1] || result.Values[1] != tc.want {
				t.Fatalf("adjacent: %+v, %v; want %g", result, err, tc.want)
			}
		}
	}
	// The exact means differ, but round to the same value by design.
	a := sampleBar(1, 1)
	b := mfi.Bar{Open: 1, High: math.Nextafter(1, 2), Low: 1, Close: 1, Volume: 1}
	result, err := mfi.ComputeCandles([]mfi.Bar{a, b}, mfi.Options{Period: 1})
	if err != nil || result == nil || !result.Valid[1] || result.Values[1] != 50 {
		t.Fatalf("rounded collision: %+v %v", result, err)
	}
}

// TestStream_ZeroAllocsPush measures the nonflat compact and wide paths as well
// as regular scheduled rebases, rather than only identical-price candles.
func TestStream_ZeroAllocsPush(t *testing.T) {
	for _, bars := range [][2]mfi.Bar{
		{sortedPriceBar(99, 100, 101), sortedPriceBar(100, 101, 102)},
		{sortedPriceBar(math.SmallestNonzeroFloat64, 1, math.MaxFloat64/2), sortedPriceBar(math.SmallestNonzeroFloat64, 2, math.MaxFloat64)},
	} {
		// Nonzero volumes exercise classification and both sign accumulators.
		bars[0].Volume, bars[1].Volume = 0x1p-10, 0x1p-10
		s, err := mfi.NewPeriod(14)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 16; i++ {
			if _, _, err = s.Push(bars[i%2]); err != nil {
				t.Fatal(err)
			}
		}
		i := 0
		allocations := testing.AllocsPerRun(1000, func() {
			if _, _, err := s.Push(bars[i%2]); err != nil {
				t.Fatal(err)
			}
			i++
		})
		if allocations != 0 {
			t.Fatalf("Push allocations=%g", allocations)
		}
	}
}

// FuzzCanonicalTypical checks the price bits against independent exact rationals.
func FuzzCanonicalTypical(f *testing.F) {
	f.Add(uint64(1), math.Float64bits(math.MaxFloat64), math.Float64bits(1))
	f.Add(math.Float64bits(4), math.Float64bits(1), math.Float64bits(2))
	f.Fuzz(func(t *testing.T, a, b, c uint64) {
		const positiveMask = uint64(0x7fffffffffffffff)
		var prices [3]float64
		for i, x := range [...]uint64{a, b, c} {
			x &= positiveMask
			v := math.Float64frombits(x)
			if !finiteTest(v) || v == 0 {
				return
			}
			prices[i] = v
		}
		s, err := mfi.NewPeriod(1)
		if err != nil {
			t.Fatal(err)
		}
		assertTypicalBits(t, s, sortedPriceBar(prices[0], prices[1], prices[2]))
	})
}
