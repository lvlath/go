// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// External range tests verify exact half-open windows and argument boundaries.
// They reject partial tails and preserve non-nil empty successful output.

package series_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/series"
)

// TestWindowRanges_HalfOpenAndNoPartial protects complete range enumeration.
// It checks overlap, disjoint windows, exact-size and non-nil empty outputs.
func TestWindowRanges_HalfOpenAndNoPartial(t *testing.T) {
	for _, tc := range []struct {
		length, width, stride int
		want                  []series.Range
	}{
		{8, 3, 2, []series.Range{{Start: 0, End: 3}, {Start: 2, End: 5}, {Start: 4, End: 7}}},
		{8, 3, 3, []series.Range{{Start: 0, End: 3}, {Start: 3, End: 6}}},
		{8, 3, 4, []series.Range{{Start: 0, End: 3}, {Start: 4, End: 7}}},
		{3, 3, 1, []series.Range{{Start: 0, End: 3}}},
		{0, 3, 1, []series.Range{}},
		{2, 3, 1, []series.Range{}},
	} {
		got, err := series.WindowRanges(tc.length, tc.width, tc.stride)
		if err != nil {
			t.Fatalf("%+v err=%v", tc, err)
		}
		assertRanges(t, got, tc.want)
	}
}

// TestWindowRanges_InvalidAndBoundary protects option errors and integer extremes.
// A massive length remains legal when only one complete range is generated.
func TestWindowRanges_InvalidAndBoundary(t *testing.T) {
	for _, tc := range [][3]int{{-1, 1, 1}, {1, 0, 1}, {1, 1, 0}, {0, -1, 1}, {0, 1, -1}} {
		got, err := series.WindowRanges(tc[0], tc[1], tc[2])
		if got != nil || !errors.Is(err, series.ErrInvalidOption) {
			t.Fatalf("%v got=%v err=%v", tc, got, err)
		}
	}
	got, err := series.WindowRanges(math.MaxInt, math.MaxInt, math.MaxInt)
	if err != nil {
		t.Fatalf("boundary got=%v err=%v", got, err)
	}
	assertRanges(t, got, []series.Range{{Start: 0, End: math.MaxInt}})
	got, err = series.WindowRanges(math.MaxInt, math.MaxInt-1, math.MaxInt)
	if err != nil || len(got) != 1 {
		t.Fatalf("boundary got=%v err=%v", got, err)
	}
}

// TestWindowRanges_ExtendingInputPreservesCompletePrefix checks index causality.
// Windows that were already complete cannot change their boundaries when future
// samples are appended, although new complete windows may become available.
func TestWindowRanges_ExtendingInputPreservesCompletePrefix(t *testing.T) {
	prefix, err := series.WindowRanges(8, 3, 2)
	if err != nil {
		t.Fatal(err)
	}
	extended, err := series.WindowRanges(11, 3, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(extended) <= len(prefix) {
		t.Fatalf("extended input must admit additional windows: prefix=%v extended=%v", prefix, extended)
	}
	assertRanges(t, extended[:len(prefix)], prefix)
}
