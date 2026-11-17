// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Public-contract regression tests: per-bar error precedence and safe restore indexes.
// Only the exported mfi API is exercised; no private testing hooks are introduced.
package mfi_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/lvlath/go/mfi"
)

// TestValidation_FirstFailingBarPrecedesLaterNonFinite anchors ordered-input
// validation: shape is checked before elements, but candles are scanned by index.
// Within each candle nonfinite fields take precedence over domain predicates.
func TestValidation_FirstFailingBarPrecedesLaterNonFinite(t *testing.T) {
	const period = 2

	t.Run("earlier domain before later nonfinite", func(t *testing.T) {
		bars := sampleBars([]float64{10, 11, 12}, []float64{1, 1, 1})
		bars.Low[0] = -1
		bars.High[2] = math.NaN()
		result, err := mfi.ComputePeriod(bars, period)
		if result != nil {
			t.Fatal("failed batch exposed a partially computed Result")
		}
		mustErrorIs(t, err, mfi.ErrDomain)
	})

	t.Run("earlier nonfinite before later domain", func(t *testing.T) {
		bars := sampleBars([]float64{10, 11, 12}, []float64{1, 1, 1})
		bars.Open[0] = math.Inf(1)
		bars.Volume[2] = -1
		result, err := mfi.ComputePeriod(bars, period)
		if result != nil {
			t.Fatal("failed batch exposed a partially computed Result")
		}
		mustErrorIs(t, err, mfi.ErrNonFinite)
	})

	t.Run("same candle nonfinite before domain", func(t *testing.T) {
		bars := sampleBars([]float64{10, 11, 12}, []float64{1, 1, 1})
		bars.Close[1] = math.NaN()
		bars.Volume[1] = -1
		_, err := mfi.ComputePeriod(bars, period)
		mustErrorIs(t, err, mfi.ErrNonFinite)
	})

	t.Run("structural validation before candle access", func(t *testing.T) {
		bars := sampleBars([]float64{10, 11, 12}, []float64{1, 1, 1})
		bars.High = bars.High[:2]
		bars.Close[0] = math.NaN()
		_, err := mfi.ComputePeriod(bars, period)
		mustErrorIs(t, err, mfi.ErrShape)
	})

	t.Run("invalid option before nil columns", func(t *testing.T) {
		_, err := mfi.Compute(mfi.Bars{}, mfi.Options{Period: mfi.MaxPeriod + 1})
		mustErrorIs(t, err, mfi.ErrInvalidOption)
	})
}

// TestSnapshot_WireIndexesNeverWrapToNativeInt verifies hostile unsigned
// coordinates cannot be narrowed into a valid physical ring index.
// Restore rejects them without modifying already committed Stream state.
func TestSnapshot_WireIndexesNeverWrapToNativeInt(t *testing.T) {
	for _, period := range []int{1, 4, 14} {
		t.Run(fmt.Sprintf("period-%d", period), func(t *testing.T) {
			stream, err := mfi.NewPeriod(period)
			if err != nil {
				t.Fatal(err)
			}
			for i := 0; i < period+5; i++ {
				price := float64(100 + (i*7)%11)
				if _, _, err = stream.Push(sampleBar(price, float64(i%3+1))); err != nil {
					t.Fatal(err)
				}
			}
			before := stream.Snapshot()
			// A full ring legitimately has used==period; only values beyond the
			// period are invalid for used. head==period is always invalid.
			cases := []struct {
				name       string
				coordinate int
				value      uint64
			}{
				{"used-period-plus-one", 0, uint64(period + 1)},
				{"head-period", 1, uint64(period)},
				{"used-above-ceiling", 0, uint64(mfi.MaxPeriod + 1)},
				{"head-above-ceiling", 1, uint64(mfi.MaxPeriod + 1)},
				{"used-max-uint64", 0, math.MaxUint64},
				{"head-max-uint64", 1, math.MaxUint64},
			}
			for _, tc := range cases {
				t.Run(tc.name, func(t *testing.T) {
					bad := stream.Snapshot()
					bad.IntState[tc.coordinate] = tc.value
					mustErrorIs(t, stream.Restore(bad), mfi.ErrStateVersion)
					mustSnapshotBits(t, stream.Snapshot(), before)
				})
			}

			// A genuine checkpoint still decodes valid coordinates and replays faithfully.
			replica, err := mfi.NewPeriod(period)
			if err != nil {
				t.Fatal(err)
			}
			if err := replica.Restore(before); err != nil {
				t.Fatalf("rejected valid checkpoint: %v", err)
			}
			mustSnapshotBits(t, replica.Snapshot(), before)
		})
	}
}
