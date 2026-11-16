// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// File rsi_test.go anchors the independent arithmetic fixtures and validation.
package rsi_test

import (
	"errors"
	"math"
	"strconv"
	"testing"

	"github.com/lvlath/go/rsi"
)

// TestRSI01To05IndependentFixtures anchors known analytical outcomes independently of streaming parity.
// It checks seed boundaries, monotone losses, flat history and near-MaxFloat prices.
func TestRSI01To05IndependentFixtures(t *testing.T) {
	cases := []struct {
		name   string
		period int
		close  []float64
		values []float64
		valid  []bool
	}{
		{"RSI01", 2, []float64{1, 2, 3, 2}, []float64{0, 0, 100, 50}, []bool{false, false, true, true}},
		{"RSI02", 1, []float64{3, 2, 2, 4}, []float64{0, 0, 50, 100}, []bool{false, true, true, true}},
		{"RSI03", 2, []float64{5, 5, 5, 5}, []float64{0, 0, 50, 50}, []bool{false, false, true, true}},
		{"RSI04", 3, []float64{1, 2}, []float64{0, 0}, []bool{false, false}},
		{"RSI05", 2, []float64{math.MaxFloat64 / 2, math.MaxFloat64 / 2, math.MaxFloat64 / 2}, []float64{0, 0, 50}, []bool{false, false, true}},
		{"falling", 2, []float64{9, 8, 7, 6, 5}, []float64{0, 0, 0, 0, 0}, []bool{false, false, true, true, true}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, err := rsi.Compute(tc.close, rsi.Options{Period: tc.period})
			if err != nil {
				t.Fatal(err)
			}
			mustEqualInt(t, len(out.Values), len(tc.values), "length")
			for i := range tc.values {
				mustSameFloat(t, out.Values[i], tc.values[i])
				mustEqualBool(t, out.Valid[i], tc.valid[i], "valid")
			}
		})
	}
}

// TestResult_IsNilInterface protects the nilable convention when typed pointers enter interfaces.
func TestResult_IsNilInterface(t *testing.T) {
	var empty *rsi.Result
	var nilable interface{ IsNil() bool } = empty
	mustNilState(t, nilable, true, "nil result")
	mustNilState(t, &rsi.Result{}, false, "allocated result")
}

// TestInputErrorPrecedence distinguishes invalid options, shape failures and numeric input categories.
// Failed batch calculations must not publish partially filled results.
func TestInputErrorPrecedence(t *testing.T) {
	cases := []struct {
		a    []float64
		p    int
		want error
	}{
		{nil, -1, rsi.ErrInvalidOption}, {nil, 2, rsi.ErrNilInput}, {[]float64{}, 2, rsi.ErrEmptyInput},
		{[]float64{1, math.NaN(), 0}, 2, rsi.ErrNonFinite},
		{[]float64{1, 0}, 2, rsi.ErrDomain},
		{[]float64{1, -1}, 2, rsi.ErrDomain},
		{[]float64{1, math.Inf(1)}, 2, rsi.ErrNonFinite},
		{[]float64{1, math.Inf(-1)}, 2, rsi.ErrNonFinite},
		{[]float64{1}, int(^uint(0) >> 1), rsi.ErrInvalidOption},
	}
	for i, tc := range cases {
		out, err := rsi.Compute(tc.a, rsi.Options{Period: tc.p})
		if out != nil {
			t.Fatalf("case %d: unexpected partial result", i)
		}
		mustErrorIs(t, err, tc.want)
	}
}

// TestRSI09AffineTransform checks exact equivalence when scaled price differences remain representable.
// It does not claim cross-scale bitwise equality for arbitrary rounded affine inputs.
func TestRSI09AffineTransform(t *testing.T) {
	base := []float64{100, 103, 101, 105, 108, 104, 102, 107, 110, 111, 108, 109, 112, 115, 113, 116}
	transformed := make([]float64, len(base))
	for i, x := range base {
		transformed[i] = x*8 + 1<<20
	}
	a, e := rsi.Compute(base, rsi.Options{Period: 7})
	if e != nil {
		t.Fatal(e)
	}
	b, e := rsi.Compute(transformed, rsi.Options{Period: 7})
	if e != nil {
		t.Fatal(e)
	}
	for i := range a.Values {
		mustSameFloat(t, a.Values[i], b.Values[i])
		mustEqualBool(t, a.Valid[i], b.Valid[i], "affine valid")
	}
}

// TestDetachedResult confirms that public Result slices are caller-owned and not retained.
// Mutating one returned result cannot affect the next computation from unchanged prices.
func TestDetachedResult(t *testing.T) {
	closeR := []float64{5, 6, 5, 7, 8, 6, 9}
	original, e := rsi.Compute(closeR, rsi.Options{Period: 2})
	if e != nil {
		t.Fatal(e)
	}
	original.Values[2] = 99
	original.Valid[0] = true
	again, e := rsi.Compute(closeR, rsi.Options{Period: 2})
	if e != nil {
		t.Fatal(e)
	}
	mustSameFloat(t, again.Values[2], 50)
	mustEqualBool(t, again.Valid[0], false, "detached valid")
}

// TestSentinelIdentity prevents accidental collapse of distinct user input failure categories.
func TestSentinelIdentity(t *testing.T) {
	if errors.Is(rsi.ErrDomain, rsi.ErrNonFinite) {
		t.Fatal("distinct sentinels collapsed")
	}
}

// TestAPI_DefaultAndPeriodFacades prevents different period defaults or shadow recurrences.
func TestAPI_DefaultAndPeriodFacades(t *testing.T) {
	prices := []float64{
		120.1, 120.3, 119.8, 120.5, 121.2, 120.7, 121.4, 121.9,
		121.2, 120.6, 120.9, 121.7, 122.4, 122.1, 122.9, 122.2,
		122.7, 123.0, 123.4, 122.8, 123.1, 123.6, 123.3, 123.8,
	}

	canonicalDefault, err := rsi.Compute(prices, rsi.Options{Period: 14})
	if err != nil {
		t.Fatal(err)
	}
	wrappedDefault, err := rsi.ComputeDefault(prices)
	if err != nil {
		t.Fatal(err)
	}
	compareResultBits(t, canonicalDefault, wrappedDefault)

	for _, period := range []int{0, 1, 2, 7, 14, 23, 100} {
		t.Run("period="+strconv.Itoa(period), func(t *testing.T) {
			canonical, err := rsi.Compute(prices, rsi.Options{Period: period})
			if err != nil {
				t.Fatal(err)
			}
			wrapped, err := rsi.ComputePeriod(prices, period)
			if err != nil {
				t.Fatal(err)
			}
			compareResultBits(t, canonical, wrapped)

			latest, valid, err := rsi.Latest(prices, rsi.Options{Period: period})
			if err != nil {
				t.Fatal(err)
			}
			last := len(prices) - 1
			mustSameFloat(t, latest, canonical.Values[last])
			mustEqualBool(t, valid, canonical.Valid[last], "latest validity")

			stream, err := rsi.NewPeriod(period)
			if err != nil {
				t.Fatal(err)
			}
			for i, price := range prices {
				value, ready, err := stream.Push(price)
				if err != nil {
					t.Fatal(err)
				}
				mustSameFloat(t, value, canonical.Values[i])
				mustEqualBool(t, ready, canonical.Valid[i], "new-period parity")
			}
		})
	}

	defaultStream, err := rsi.NewDefault()
	if err != nil {
		t.Fatal(err)
	}
	explicitStream, err := rsi.NewPeriod(14)
	if err != nil {
		t.Fatal(err)
	}
	assertSnapshotEqual(t, defaultStream.Snapshot(), explicitStream.Snapshot())
	defaultAlias, err := rsi.New(rsi.Options{})
	if err != nil {
		t.Fatal(err)
	}
	assertSnapshotEqual(t, defaultStream.Snapshot(), defaultAlias.Snapshot())
}

// compareResultBits compares concrete result fields without reflection or floating tolerance.
func compareResultBits(t *testing.T, left, right *rsi.Result) {
	t.Helper()
	if left == nil || right == nil {
		t.Fatal("unexpected nil result")
	}
	mustEqualInt(t, len(left.Values), len(right.Values), "value length")
	mustEqualInt(t, len(left.Valid), len(right.Valid), "mask length")
	for i := range left.Values {
		mustSameFloat(t, left.Values[i], right.Values[i])
		mustEqualBool(t, left.Valid[i], right.Valid[i], "mask")
	}
}

// TestAPI_WrapperErrorIdentity verifies that shortcuts cannot bypass public validation.
func TestAPI_WrapperErrorIdentity(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for _, tc := range []struct {
		name   string
		prices []float64
		period int
		want   error
	}{
		{"invalid-option-before-nil", nil, -1, rsi.ErrInvalidOption},
		{"unrepresentable-warmup", []float64{1}, maxInt, rsi.ErrInvalidOption},
		{"nil", nil, 2, rsi.ErrNilInput},
		{"empty", []float64{}, 2, rsi.ErrEmptyInput},
		{"nonfinite-first", []float64{10, math.NaN(), 0}, 2, rsi.ErrNonFinite},
		{"domain-first", []float64{10, 0, math.NaN()}, 2, rsi.ErrDomain},
		{"negative", []float64{10, -0.25}, 2, rsi.ErrDomain},
	} {
		t.Run(tc.name, func(t *testing.T) {
			result, err := rsi.ComputePeriod(tc.prices, tc.period)
			if result != nil {
				t.Fatal("failed batch wrapper exposed partial result")
			}
			mustErrorIs(t, err, tc.want)

			value, valid, err := rsi.Latest(tc.prices, rsi.Options{Period: tc.period})
			mustErrorIs(t, err, tc.want)
			mustSameFloat(t, value, 0)
			mustEqualBool(t, valid, false, "failed Latest validity")
		})
	}

	result, err := rsi.ComputeDefault(nil)
	if result != nil {
		t.Fatal("ComputeDefault returned a result for nil input")
	}
	mustErrorIs(t, err, rsi.ErrNilInput)

	for _, period := range []int{-1, maxInt} {
		stream, err := rsi.NewPeriod(period)
		if stream != nil {
			t.Fatal("NewPeriod exposed a partially initialized stream")
		}
		mustErrorIs(t, err, rsi.ErrInvalidOption)
	}
}

// TestAPI_LatestWarmup protects the valid flag for a nonempty but insufficient history.
func TestAPI_LatestWarmup(t *testing.T) {
	value, valid, err := rsi.Latest([]float64{100, 101, 99}, rsi.Options{Period: 14})
	if err != nil {
		t.Fatal(err)
	}
	mustSameFloat(t, value, 0)
	mustEqualBool(t, valid, false, "warmup validity")
}

// TestAPI_IsNilDistinguishesTypedNil checks the core.Nilable shape through a public interface.
func TestAPI_IsNilDistinguishesTypedNil(t *testing.T) {
	var result *rsi.Result
	var nilable interface{ IsNil() bool } = result
	mustNilState(t, nilable, true, "typed nil")
	mustNilState(t, &rsi.Result{}, false, "non-nil zero Result")
}
