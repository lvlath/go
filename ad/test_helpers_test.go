// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ad_test provides reflection-free oracles for external contract tests.
// This file compares exact binary64 state, nilability and sentinel identity.
package ad_test

import (
	"errors"
	"math"
	"math/big"
	"testing"

	"github.com/lvlath/go/ad"
)

// Compile-time structural check for the same IsNil method required by core.Nilable.
var _ interface{ IsNil() bool } = (*ad.Result)(nil)

// mustErrorIs classifies errors by identity, not by diagnostic strings.
func mustErrorIs(t *testing.T, err, target error) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Fatalf("errors.Is(%v, %v) = false", err, target)
	}
}

// mustEqualBool checks an explicit boolean contract attribute.
func mustEqualBool(t *testing.T, got, want bool, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %v want %v", op, got, want)
	}
}

// mustEqualInt checks a shape, count or index without floating conversion.
func mustEqualInt(t *testing.T, got, want int, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %d want %d", op, got, want)
	}
}

// mustEqualString protects exact published metadata rather than searching error text.
func mustEqualString(t *testing.T, got, want, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %q want %q", op, got, want)
	}
}

// isNilLike exercises the reflect-free nilability protocol, including typed nil pointers.
func isNilLike(value interface{ IsNil() bool }) bool {
	return value == nil || value.IsNil()
}

// mustNilState verifies whether a result pointer is absent via its own public method.
func mustNilState(t *testing.T, value *ad.Result, want bool, op string) {
	t.Helper()
	if value.IsNil() != want {
		t.Fatalf("%s: IsNil()=%v want %v", op, value.IsNil(), want)
	}
}

// mustBits catches signed-zero, rounding and low-component differences invisible to ==.
func mustBits(t *testing.T, got, want float64) {
	t.Helper()
	if math.Float64bits(got) != math.Float64bits(want) {
		t.Fatalf("float bits differ: got %.17g (%016x), want %.17g (%016x)",
			got, math.Float64bits(got), want, math.Float64bits(want))
	}
}

// mustSnapshotEqual asserts transactional identity of every published checkpoint coordinate.
func mustSnapshotEqual(t *testing.T, got, want ad.Snapshot) {
	t.Helper()
	mustEqualString(t, got.Algorithm, want.Algorithm, "Algorithm")
	mustEqualString(t, got.Config, want.Config, "Config")
	if got.Count != want.Count {
		t.Fatalf("Count=%d want=%d", got.Count, want.Count)
	}
	mustEqualInt(t, len(got.FloatState), len(want.FloatState), "FloatState length")
	mustEqualInt(t, len(got.IntState), len(want.IntState), "IntState length")
	for i := range want.FloatState {
		mustBits(t, got.FloatState[i], want.FloatState[i])
	}
	for i := range want.IntState {
		if got.IntState[i] != want.IntState[i] {
			t.Fatalf("IntState[%d]=%d want=%d", i, got.IntState[i], want.IntState[i])
		}
	}
}

// mustResultValues checks ownership-independent numerical and validity-mask contracts.
func mustResultValues(t *testing.T, result *ad.Result, expected []float64) {
	t.Helper()
	mustNilState(t, result, false, "successful batch")
	mustEqualInt(t, len(result.Values), len(expected), "Values length")
	mustEqualInt(t, len(result.Valid), len(expected), "Valid length")
	for i, want := range expected {
		mustBits(t, result.Values[i], want)
		mustEqualBool(t, result.Valid[i], true, "Valid")
	}
}

// barsFrom creates five independent columns for a fixed row-oriented fixture.
func barsFrom(candles []ad.Bar) ad.Bars {
	columns := ad.Bars{
		Open: make([]float64, len(candles)), High: make([]float64, len(candles)),
		Low: make([]float64, len(candles)), Close: make([]float64, len(candles)),
		Volume: make([]float64, len(candles)),
	}
	for i, bar := range candles {
		columns.Open[i], columns.High[i], columns.Low[i] = bar.Open, bar.High, bar.Low
		columns.Close[i], columns.Volume[i] = bar.Close, bar.Volume
	}
	return columns
}

// standardBars is an exact arithmetic fixture, not a copy of the implementation.
// Expected cumulative A/D is [3,1,1,1,5] by direct range-position reasoning.
func standardBars() []ad.Bar {
	return []ad.Bar{
		{Open: 1.5, High: 2, Low: 1, Close: 2, Volume: 3},
		{Open: 1.5, High: 2, Low: 1, Close: 1, Volume: 2},
		{Open: 1, High: 1, Low: 1, Close: 1, Volume: 100},
		{Open: 1.5, High: 2, Low: 1, Close: 1.5, Volume: 100},
		{Open: 3, High: 4, Low: 2, Close: 3.5, Volume: 8},
	}
}

// variedBars creates deterministic nonflat data with signed contributions, midpoint
// zeros and occasional volume-zero observations for parity and ownership exercises.
func variedBars(count int) []ad.Bar {
	bars := make([]ad.Bar, count)
	for i := range bars {
		low := 96 + float64(i%17)
		closeOffset := float64((i*7 + 3) % 9) // Close in [Low, Low+8].
		volume := float64(100 + (i*37)%1700)
		if i%31 == 0 {
			volume = 0
		}
		bars[i] = ad.Bar{
			Open: low + 2, High: low + 8, Low: low,
			Close: low + closeOffset, Volume: volume,
		}
	}
	return bars
}

// rationalContribution computes the exact money-flow volume from the binary64
// input values as arbitrary-precision rationals, independently of the Go kernel.
func rationalContribution(bar ad.Bar) *big.Rat {
	if bar.High == bar.Low {
		return new(big.Rat)
	}
	closePrice := new(big.Rat).SetFloat64(bar.Close)
	low := new(big.Rat).SetFloat64(bar.Low)
	high := new(big.Rat).SetFloat64(bar.High)
	volume := new(big.Rat).SetFloat64(bar.Volume)
	left := new(big.Rat).Sub(closePrice, low)
	right := new(big.Rat).Sub(high, closePrice)
	numerator := new(big.Rat).Sub(left, right)
	spread := new(big.Rat).Sub(high, low)
	return new(big.Rat).Mul(new(big.Rat).Quo(numerator, spread), volume)
}

// orderedBits maps finite float64 values to monotonically increasing uint64 keys.
// This lets the test measure ULP distance without reflection or sign-branch errors.
func orderedBits(value float64) uint64 {
	bits := math.Float64bits(value)
	if (bits & (uint64(1) << 63)) != 0 {
		return ^bits
	}
	return bits | (uint64(1) << 63)
}

// mustWithinULP bounds the distance from an independent binary64 oracle.
func mustWithinULP(t *testing.T, got, want float64, tolerance uint64) {
	t.Helper()
	if math.IsNaN(got) || math.IsInf(got, 0) || math.IsNaN(want) || math.IsInf(want, 0) {
		t.Fatalf("nonfinite numeric oracle: got %g want %g", got, want)
	}
	a, b := orderedBits(got), orderedBits(want)
	if a < b {
		a, b = b, a
	}
	if a-b > tolerance {
		t.Fatalf("%.17g differs from oracle %.17g by %d ULP (> %d)", got, want, a-b, tolerance)
	}
}
