// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ad_test protects the Chaikin arithmetic with independent fixtures and invariants.
// These tests use only the exported ad facade; no production internals are exposed.
package ad_test

import (
	"errors"
	"math"
	"math/big"
	"math/rand"
	"testing"

	"github.com/lvlath/go/ad"
)

// TestAD01ThroughAD05_IndependentOracle anchors five exact hand-computed cumulative values.
func TestAD01ThroughAD05_IndependentOracle(t *testing.T) {
	candles := standardBars()
	want := []float64{3, 1, 1, 1, 5}
	result, err := ad.Compute(barsFrom(candles))
	if err != nil {
		t.Fatal(err)
	}
	mustResultValues(t, result, want)

	// A single valid first candle is not an OBV-style zero seed.
	first, err := ad.Compute(barsFrom(candles[:1]))
	if err != nil {
		t.Fatal(err)
	}
	mustResultValues(t, first, []float64{3})

	// Row-oriented facade must match the same independent oracle, not merely
	// agree with a potentially wrong column implementation.
	rows, err := ad.ComputeCandles(candles)
	if err != nil {
		t.Fatal(err)
	}
	mustResultValues(t, rows, want)
}

// TestAD_CandleGeometryEndpoints demonstrates closed-form signed-volume contributions.
func TestAD_CandleGeometryEndpoints(t *testing.T) {
	cases := []struct {
		name string
		bar  ad.Bar
		want float64
	}{
		{"closeAtHigh", ad.Bar{Open: 101, High: 105, Low: 99, Close: 105, Volume: 300}, 300},
		{"closeAtLow", ad.Bar{Open: 101, High: 105, Low: 99, Close: 99, Volume: 300}, -300},
		{"exactMidpoint", ad.Bar{Open: 101, High: 104, Low: 98, Close: 101, Volume: 300}, 0},
		{"flatHugeVolume", ad.Bar{Open: 100, High: 100, Low: 100, Close: 100, Volume: math.MaxFloat64}, 0},
		{"zeroVolumeAtLow", ad.Bar{Open: 101, High: 105, Low: 99, Close: 99, Volume: 0}, 0},
		{"quarterRange", ad.Bar{Open: 102, High: 104, Low: 100, Close: 101, Volume: 80}, -40},
		{"threeQuarterRange", ad.Bar{Open: 102, High: 104, Low: 100, Close: 103, Volume: 80}, 40},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := ad.ComputeCandles([]ad.Bar{tc.bar})
			if err != nil {
				t.Fatal(err)
			}
			mustResultValues(t, result, []float64{tc.want})
		})
	}
}

// TestAD06_AffinePriceInvariance uses an exact positive affine coordinate change.
func TestAD06_AffinePriceInvariance(t *testing.T) {
	candles := append(standardBars(), variedBars(127)...)
	transformed := append([]ad.Bar(nil), candles...)
	for index := range transformed {
		bar := &transformed[index]
		bar.Open = 8*bar.Open + 1024
		bar.High = 8*bar.High + 1024
		bar.Low = 8*bar.Low + 1024
		bar.Close = 8*bar.Close + 1024
	}
	original, err := ad.ComputeCandles(candles)
	if err != nil {
		t.Fatal(err)
	}
	scaled, err := ad.ComputeCandles(transformed)
	if err != nil {
		t.Fatal(err)
	}
	for index := range original.Values {
		mustBits(t, scaled.Values[index], original.Values[index])
		mustEqualBool(t, scaled.Valid[index], original.Valid[index], "affine validity")
	}
}

// TestAD07_ExtremePricesIndependentRationalOracle avoids naive 2*Close arithmetic.
func TestAD07_ExtremePricesIndependentRationalOracle(t *testing.T) {
	base := math.Ldexp(1, 450)
	step := math.Ldexp(1, 400)
	previous := math.Nextafter(math.MaxFloat64, 0)
	lowNearMax := math.Nextafter(previous, 0)
	cases := []struct {
		name string
		bar  ad.Bar
		want float64
	}{
		{"offset2Power450", ad.Bar{Open: base + step, Low: base, High: base + 4*step, Close: base + 3*step, Volume: 8}, 4},
		{"positiveNearMax", ad.Bar{Open: previous, Low: previous, High: math.MaxFloat64, Close: math.MaxFloat64, Volume: 3}, 3},
		{"negativeNearMax", ad.Bar{Open: previous, Low: previous, High: math.MaxFloat64, Close: previous, Volume: 3}, -3},
		{"midpointNearMax", ad.Bar{Open: previous, Low: lowNearMax, High: math.MaxFloat64, Close: previous, Volume: 8}, 0},
		{"smallestPositiveSpan", ad.Bar{Open: math.SmallestNonzeroFloat64, Low: math.SmallestNonzeroFloat64, High: 2 * math.SmallestNonzeroFloat64, Close: 2 * math.SmallestNonzeroFloat64, Volume: 1}, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			actual, err := ad.ComputeCandles([]ad.Bar{tc.bar})
			if err != nil {
				t.Fatal(err)
			}
			mustResultValues(t, actual, []float64{tc.want})
			oracle, _ := rationalContribution(tc.bar).Float64()
			mustWithinULP(t, actual.Values[0], oracle, 1)
		})
	}
}

// TestAD07_ExactRepresentableMultiplierGrid checks many price scales against big.Rat.
func TestAD07_ExactRepresentableMultiplierGrid(t *testing.T) {
	// Choosing powers of two and steps far above the ULP makes each OHLC
	// difference exactly representable; this is an analytic oracle, not parity.
	for exponent := -900; exponent <= 900; exponent += 37 {
		base := math.Ldexp(1, exponent)
		step := math.Ldexp(1, exponent-42)
		for position := 0; position <= 16; position++ {
			bar := ad.Bar{
				Open: base + 8*step, Low: base, High: base + 16*step,
				Close: base + float64(position)*step, Volume: float64(101 + position),
			}
			result, err := ad.ComputeCandles([]ad.Bar{bar})
			if err != nil {
				t.Fatalf("exp=%d position=%d: %v", exponent, position, err)
			}
			oracle, _ := rationalContribution(bar).Float64()
			mustBits(t, result.Values[0], oracle)
			if math.Abs(result.Values[0]) > bar.Volume {
				t.Fatalf("|contribution| exceeds Volume: %g > %g", math.Abs(result.Values[0]), bar.Volume)
			}
		}
	}
}

// TestAD08_ValidationAndPrecedence protects the global Nil->Empty->Shape->Finite->Domain gates.
func TestAD08_ValidationAndPrecedence(t *testing.T) {
	good := standardBars()[0]
	cases := []struct {
		name    string
		columns ad.Bars
		want    error
	}{
		{"nilAll", ad.Bars{}, ad.ErrNilInput},
		{"nilVolumeDespiteEmpty", ad.Bars{Open: []float64{}, High: []float64{}, Low: []float64{}, Close: []float64{}}, ad.ErrNilInput},
		{"emptyAllNonNil", ad.Bars{Open: []float64{}, High: []float64{}, Low: []float64{}, Close: []float64{}, Volume: []float64{}}, ad.ErrEmptyInput},
		{"emptyOpenDespiteShape", ad.Bars{Open: []float64{}, High: []float64{1}, Low: []float64{}, Close: []float64{}, Volume: []float64{}}, ad.ErrEmptyInput},
		{"shapeBeforeNonfinite", ad.Bars{Open: []float64{math.NaN()}, High: []float64{2, 2}, Low: []float64{1}, Close: []float64{1}, Volume: []float64{1}}, ad.ErrShape},
		{"nanOpen", barsFrom([]ad.Bar{{Open: math.NaN(), High: 2, Low: 1, Close: 2, Volume: 3}}), ad.ErrNonFinite},
		{"infiniteHigh", barsFrom([]ad.Bar{{Open: 1, High: math.Inf(1), Low: 1, Close: 1, Volume: 0}}), ad.ErrNonFinite},
		{"negativeInfiniteLow", barsFrom([]ad.Bar{{Open: 1, High: 2, Low: math.Inf(-1), Close: 1, Volume: 1}}), ad.ErrNonFinite},
		{"nanClose", barsFrom([]ad.Bar{{Open: 1, High: 2, Low: 1, Close: math.NaN(), Volume: 1}}), ad.ErrNonFinite},
		{"infiniteVolume", barsFrom([]ad.Bar{{Open: 1, High: 2, Low: 1, Close: 1, Volume: math.Inf(1)}}), ad.ErrNonFinite},
		{"closeAboveHigh", barsFrom([]ad.Bar{{Open: 1.5, High: 2, Low: 1, Close: 3, Volume: 1}}), ad.ErrDomain},
		{"closeBelowLow", barsFrom([]ad.Bar{{Open: 1.5, High: 2, Low: 1, Close: 0.5, Volume: 1}}), ad.ErrDomain},
		{"openBelowLow", barsFrom([]ad.Bar{{Open: 0.5, High: 2, Low: 1, Close: 1.5, Volume: 1}}), ad.ErrDomain},
		{"openAboveHigh", barsFrom([]ad.Bar{{Open: 3, High: 2, Low: 1, Close: 1.5, Volume: 1}}), ad.ErrDomain},
		{"negativeVolume", barsFrom([]ad.Bar{{Open: 1.5, High: 2, Low: 1, Close: 1.5, Volume: -1}}), ad.ErrDomain},
		{"zeroPrice", barsFrom([]ad.Bar{{Open: 1, High: 2, Low: 0, Close: 1.5, Volume: 1}}), ad.ErrDomain},
		{"reversedSpread", barsFrom([]ad.Bar{{Open: 2, High: 1, Low: 2, Close: 2, Volume: 1}}), ad.ErrDomain},
		{"invalidFlatOpen", barsFrom([]ad.Bar{{Open: 1.1, High: 1, Low: 1, Close: 1, Volume: 1}}), ad.ErrDomain},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			actual, err := ad.Compute(tc.columns)
			mustErrorIs(t, err, tc.want)
			mustNilState(t, actual, true, "invalid Compute")
		})
	}

	// Non-finite error precedence applies across the *whole* batch, not just
	// within an early row. This differs from a simple per-row short-circuit.
	earlyDomain := good
	earlyDomain.Open = 0
	lateNonFinite := good
	lateNonFinite.Volume = math.NaN()
	result, err := ad.Compute(barsFrom([]ad.Bar{earlyDomain, lateNonFinite}))
	mustErrorIs(t, err, ad.ErrNonFinite)
	mustNilState(t, result, true, "global precedence")
}

// TestAD_ComputeCandlesErrors ensures row-oriented adaptation has no extra defaults.
func TestAD_ComputeCandlesErrors(t *testing.T) {
	invalid := standardBars()[0]
	invalid.Close = math.Inf(1)
	cases := []struct {
		name    string
		candles []ad.Bar
		want    error
	}{
		{"nil", nil, ad.ErrNilInput},
		{"empty", []ad.Bar{}, ad.ErrEmptyInput},
		{"nonfinite", []ad.Bar{invalid}, ad.ErrNonFinite},
		{"domain", []ad.Bar{{Open: 3, High: 2, Low: 1, Close: 2, Volume: 1}}, ad.ErrDomain},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ad.ComputeCandles(tc.candles)
			mustNilState(t, got, true, "invalid ComputeCandles")
			mustErrorIs(t, err, tc.want)
		})
	}

	badDomain := standardBars()[0]
	badDomain.Volume = -1
	badNonfinite := standardBars()[0]
	badNonfinite.Close = math.NaN()
	got, err := ad.ComputeCandles([]ad.Bar{badDomain, badNonfinite})
	mustNilState(t, got, true, "row global precedence")
	mustErrorIs(t, err, ad.ErrNonFinite)
}

// TestAD09_CompensationIndependentOracle ensures signed massive bars preserve a small term.
func TestAD09_CompensationIndependentOracle(t *testing.T) {
	candles := []ad.Bar{
		{Open: 1.5, High: 2, Low: 1, Close: 2, Volume: 1e16},
		{Open: 1.5, High: 2, Low: 1, Close: 2, Volume: 1},
		{Open: 1.5, High: 2, Low: 1, Close: 1, Volume: 1e16},
	}
	result, err := ad.ComputeCandles(candles)
	if err != nil {
		t.Fatal(err)
	}
	mustBits(t, result.Values[2], 1)

	// Arbitrary-precision accumulation is an independent mathematical oracle:
	// every rational contribution here is exactly an integer (+1e16,+1,-1e16).
	oracle := new(big.Rat)
	for _, bar := range candles {
		oracle.Add(oracle, rationalContribution(bar))
	}
	want, _ := oracle.Float64()
	mustBits(t, result.Values[2], want)
}

// TestAD_ResultsAndInputsDetached ensures no externally owned arrays are aliased.
func TestAD_ResultsAndInputsDetached(t *testing.T) {
	columns := barsFrom(standardBars())
	original, err := ad.Compute(columns)
	if err != nil {
		t.Fatal(err)
	}
	columns.Open[0], columns.Volume[0] = 0, 400 // Mutation after return is safe.
	mustBits(t, original.Values[0], 3)

	original.Values[0], original.Valid[0] = 999, false
	independent, err := ad.ComputeCandles(standardBars())
	if err != nil {
		t.Fatal(err)
	}
	mustBits(t, independent.Values[0], 3)
	mustEqualBool(t, independent.Valid[0], true, "independent mask")
}

// TestAD_LatestProjections validates wrappers against independent expected values.
func TestAD_LatestProjections(t *testing.T) {
	candles := standardBars()
	fromColumns, validColumns, err := ad.Latest(barsFrom(candles))
	if err != nil {
		t.Fatal(err)
	}
	fromRows, validRows, err := ad.LatestCandles(candles)
	if err != nil {
		t.Fatal(err)
	}
	mustBits(t, fromColumns, 5)
	mustBits(t, fromRows, 5)
	mustEqualBool(t, validColumns, true, "Latest columns valid")
	mustEqualBool(t, validRows, true, "LatestCandles valid")

	fromColumns, validColumns, err = ad.Latest(ad.Bars{})
	mustErrorIs(t, err, ad.ErrNilInput)
	mustBits(t, fromColumns, 0)
	mustEqualBool(t, validColumns, false, "Latest columns error")

	fromRows, validRows, err = ad.LatestCandles([]ad.Bar{})
	mustErrorIs(t, err, ad.ErrEmptyInput)
	mustBits(t, fromRows, 0)
	mustEqualBool(t, validRows, false, "Latest rows error")
}

// TestResult_IsNilInterface checks typed-nil handling without reflect or unsafe.
func TestResult_IsNilInterface(t *testing.T) {
	var missing *ad.Result
	var nilable interface{ IsNil() bool } = missing
	mustEqualBool(t, nilable.IsNil(), true, "typed nil result")
	mustEqualBool(t, isNilLike(nilable), true, "typed nil interface")
	mustEqualBool(t, isNilLike(nil), true, "nil interface")

	present := &ad.Result{}
	mustNilState(t, present, false, "non-nil zero Result")
	mustEqualBool(t, isNilLike(present), false, "present interface")

	result, err := ad.ComputeCandles([]ad.Bar{standardBars()[0]})
	if err != nil {
		t.Fatal(err)
	}
	mustEqualBool(t, result.IsNil(), false, "computed result")
	mustBits(t, result.Values[0], 3)
}

// TestAD_SentinelIdentityAndNoUserNumbers guards the caller branching contract.
func TestAD_SentinelIdentityAndNoUserNumbers(t *testing.T) {
	for _, sentinel := range []error{ad.ErrNilInput, ad.ErrEmptyInput, ad.ErrShape,
		ad.ErrNonFinite, ad.ErrDomain, ad.ErrOverflow, ad.ErrStateVersion} {
		if sentinel == nil || !errors.Is(sentinel, sentinel) {
			t.Fatalf("invalid sentinel: %v", sentinel)
		}
	}
	invalid := ad.Bar{Open: 427198.25, High: 4, Low: 2, Close: 3, Volume: 1}
	_, err := ad.ComputeCandles([]ad.Bar{invalid})
	mustErrorIs(t, err, ad.ErrDomain)
	if err.Error() != ad.ErrDomain.Error() {
		t.Fatal("invalid domain diagnostic must not embed caller data")
	}
}

// TestExamples_BTCPrefixRationalOracle independently audits the scenario's documented output.
// It never calls the implementation to construct its expected half-hour balances.
func TestExamples_BTCPrefixRationalOracle(t *testing.T) {
	candles := illustrativeBTCBars()
	mustEqualInt(t, len(candles), 18, "BTC scenario candles")
	want := []float64{16, 37.5, 51}
	exact := new(big.Rat)
	for index, bar := range candles {
		exact.Add(exact, rationalContribution(bar))
		if (index+1)%6 == 0 {
			balance, _ := exact.Float64()
			mustBits(t, balance, want[(index+1)/6-1])
		}
	}
}

// TestAD_RandomizedScaledRationalBound compares individual multipliers with exact rationals.
// It probes broad binary64 exponent regimes without imposing an invalid relative near-zero bound.
func TestAD_RandomizedScaledRationalBound(t *testing.T) {
	generator := rand.New(rand.NewSource(19))
	const cases = 1_024
	absoluteErrorBound := 8 * (math.Nextafter(1.0, 2.0) - 1.0)
	for trial := 0; trial < cases; trial++ {
		// Power-of-two bases and varied spans are all positive and finite.
		// Roundoff is measured against the exact rational of the *represented*
		// input floats, not the idealized unrounded source coordinates.
		exponent := generator.Intn(1_800) - 900
		base := math.Ldexp(1, exponent)
		step := math.Ldexp(1, exponent-49+generator.Intn(51))
		low := base
		high := base + step
		position := generator.Float64()
		closePrice := low + (high-low)*position
		bar := ad.Bar{Open: low, Low: low, High: high, Close: closePrice, Volume: 1}
		result, err := ad.ComputeCandles([]ad.Bar{bar})
		if err != nil {
			t.Fatalf("trial=%d: %v", trial, err)
		}
		oracle, _ := rationalContribution(bar).Float64()
		if math.Abs(result.Values[0]-oracle) > absoluteErrorBound {
			t.Fatalf("trial=%d: multiplier=%g rational=%g error=%g", trial,
				result.Values[0], oracle, math.Abs(result.Values[0]-oracle))
		}
		if math.Abs(result.Values[0]) > 1 {
			t.Fatalf("trial=%d: multiplier outside [-1,1]", trial)
		}
	}
}

// TestAD09_RepeatedCancellationRetainsSmallBalances uses a long exact-integer oracle.
func TestAD09_RepeatedCancellationRetainsSmallBalances(t *testing.T) {
	const cycles = 4096
	var stream ad.Stream
	positive := ad.Bar{Open: 1.5, High: 2, Low: 1, Close: 2, Volume: 1e16}
	small := ad.Bar{Open: 1.5, High: 2, Low: 1, Close: 2, Volume: 1}
	negative := ad.Bar{Open: 1.5, High: 2, Low: 1, Close: 1, Volume: 1e16}
	for cycle := 1; cycle <= cycles; cycle++ {
		for _, bar := range []ad.Bar{positive, small, negative} {
			if _, _, err := stream.Push(bar); err != nil {
				t.Fatalf("cycle=%d: %v", cycle, err)
			}
		}
		balance := stream.Snapshot()
		if balance.Count != uint64(cycle*3) {
			t.Fatalf("cycle=%d Count=%d", cycle, balance.Count)
		}
		// Each mathematical triple is exactly 1, so cycle is an exact oracle.
		mustBits(t, balance.FloatState[0]+balance.FloatState[1], float64(cycle))
	}
}
