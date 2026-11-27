// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package sessionlevels_test verifies that named facades are honest projections
// of the canonical Compute contract, without introducing implicit calendars.
package sessionlevels_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/sessionlevels"
)

// TestSessionLevels_WrapperCanonicalParity checks every field bit-for-bit,
// including signed-zero ties, warmup and incomplete final sessions.
func TestSessionLevels_WrapperCanonicalParity(t *testing.T) {
	cases := []struct {
		name string
		data fixture
	}{
		{"golden-incomplete", goldenFixture()},
		{"negative-spread", signedFixture()},
		{"signed-zero", zeroFixture(true)},
		{"empty", sizedFixture(0, 1)},
		{"daily", dailyFixture()},
		{"holiday-weeks", weeklyFixture()},
		{"large", generatedFixture(41, 4097)},
	}
	wrappers := []struct {
		name string
		call func(context.Context, []float64, []float64, []float64, []sessionlevels.Session) (*sessionlevels.Result, error)
	}{
		{"default", sessionlevels.ComputeDefault},
		{"daily", sessionlevels.ComputeDaily},
		{"weekly", sessionlevels.ComputeWeekly},
	}
	for _, scenario := range cases {
		t.Run(scenario.name, func(t *testing.T) {
			canonical := mustCompute(t, scenario.data)
			for _, wrapper := range wrappers {
				t.Run(wrapper.name, func(t *testing.T) {
					got, err := wrapper.call(context.Background(), scenario.data.high, scenario.data.low,
						scenario.data.closePrices, scenario.data.sessions)
					if err != nil || got == nil {
						t.Fatalf("wrapper returned %v, %v", got, err)
					}
					requireResult(t, got, canonical)
				})
			}
		})
	}
}

// TestSessionLevels_WrapperErrorsAreUnchanged makes sentinel precedence and
// cancellation contractual for every entry point, including an empty manifest.
func TestSessionLevels_WrapperErrorsAreUnchanged(t *testing.T) {
	wrappers := []struct {
		name string
		call func(context.Context, []float64, []float64, []float64, []sessionlevels.Session) (*sessionlevels.Result, error)
	}{
		{"default", sessionlevels.ComputeDefault},
		{"daily", sessionlevels.ComputeDaily},
		{"weekly", sessionlevels.ComputeWeekly},
	}
	for _, wrapper := range wrappers {
		t.Run(wrapper.name, func(t *testing.T) {
			f := goldenFixture()
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			for _, tc := range []struct {
				name string
				ctx  context.Context
				data fixture
				want error
			}{
				{"nil-context", nil, f, sessionlevels.ErrNilContext},
				{"canceled", ctx, f, context.Canceled},
				{"nil-price", context.Background(), fixture{}, sessionlevels.ErrNilInput},
				{"bad-session", context.Background(), func() fixture {
					bad := goldenFixture()
					bad.sessions[1].Start++
					return bad
				}(), sessionlevels.ErrSession},
				{"nonfinite", context.Background(), func() fixture {
					bad := goldenFixture()
					bad.high[3] = math.Inf(1)
					return bad
				}(), sessionlevels.ErrNonFinite},
			} {
				t.Run(tc.name, func(t *testing.T) {
					result, err := wrapper.call(tc.ctx, tc.data.high, tc.data.low, tc.data.closePrices, tc.data.sessions)
					if result != nil || !errors.Is(err, tc.want) {
						t.Fatalf("result=%v error=%v expected=%v", result, err, tc.want)
					}
				})
			}
		})
	}
}

// TestSessionLevels_DailyAndWeeklyAreCallerDefined protects against the
// accidental interpretation of named wrappers as automatic segmentation.
func TestSessionLevels_DailyAndWeeklyAreCallerDefined(t *testing.T) {
	f := weeklyFixture()
	weekly, err := sessionlevels.ComputeWeekly(context.Background(), f.high, f.low, f.closePrices, f.sessions)
	if err != nil {
		t.Fatal(err)
	}
	// One-bar sessions over the same observations deliberately produce different
	// summaries. Both named wrappers must use whichever partition is supplied.
	dailySegments := sizedFixture(len(f.high), 1).sessions
	daily, err := sessionlevels.ComputeDaily(context.Background(), f.high, f.low, f.closePrices, dailySegments)
	if err != nil {
		t.Fatal(err)
	}
	if !weekly.Valid[4] || !daily.Valid[4] {
		t.Fatal("both manifest choices must be ready after their first interval")
	}
	if weekly.PreviousHigh[4] != 115 || daily.PreviousHigh[4] != f.high[3] {
		t.Fatal("wrappers ignored caller-selected day or week boundaries")
	}
}
