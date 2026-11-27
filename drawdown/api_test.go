// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package drawdown_test verifies scalar projection parity and atomic errors.
// These tests use only the public facades and independent expected values.
package drawdown_test

import (
	"context"
	"math"
	"slices"
	"testing"
	"time"

	"github.com/lvlath/go/drawdown"
)

// TestDrawdown_ComputeMaximum verifies exact historical maxima, including rounded
// collisions, no-drawdown, total loss and recovery followed by a smaller episode.
func TestDrawdown_ComputeMaximum(t *testing.T) {
	for _, equity := range [][]float64{
		{1}, {100, 50, 100}, {100, 120, 90, 120, 80}, {100, 0, 101},
		{1, 0x1p-55, 2, 0x1p-55, 2}, {3, math.Nextafter(3, 0)},
		{math.MaxFloat64, math.SmallestNonzeroFloat64, math.MaxFloat64},
	} {
		before := slices.Clone(equity)
		got, err := drawdown.ComputeMaximum(context.Background(), equity)
		mustErrorIs(t, err, nil)
		mustEqualBits(t, got, rationalOracle(equity).Maximum, "maximum oracle")
		mustEqualBits(t, got, mustCompute(t, equity).Maximum, "maximum projection")
		mustEqualSlice(t, equity, before)
	}
}

// TestDrawdown_ComputeCurrentDepth distinguishes the final observed loss from
// the historical maximum and validates every available prefix independently.
func TestDrawdown_ComputeCurrentDepth(t *testing.T) {
	equity := []float64{100, 125, 80, 100, 125, 140, 126, 140, 0, 140}
	before := slices.Clone(equity)
	for n := 1; n <= len(equity); n++ {
		prefix := equity[:n]
		got, err := drawdown.ComputeCurrentDepth(context.Background(), prefix)
		mustErrorIs(t, err, nil)
		mustEqualBits(t, got, rationalOracle(prefix).Depth[n-1], "current oracle")
		mustEqualBits(t, got, mustCompute(t, prefix).Depth[n-1], "current projection")
	}
	mustEqualSlice(t, equity, before)
}

// TestDrawdown_ProjectionFailureParity protects every scalar wrapper from
// indexing an absent result, dropping a sentinel, or publishing a partial value.
func TestDrawdown_ProjectionFailureParity(t *testing.T) {
	ctx := context.Background()
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	expired, stop := context.WithDeadline(ctx, time.Unix(0, 0))
	defer stop()
	for _, projection := range []struct {
		name string
		call func(context.Context, []float64) (float64, error)
	}{
		{"maximum", drawdown.ComputeMaximum}, {"current", drawdown.ComputeCurrentDepth},
	} {
		t.Run(projection.name, func(t *testing.T) {
			for _, tc := range []struct {
				name   string
				ctx    context.Context
				equity []float64
				want   error
			}{
				{"nil_context", nil, nil, drawdown.ErrNilContext},
				{"canceled", canceled, []float64{0}, context.Canceled},
				{"deadline", expired, nil, context.DeadlineExceeded},
				{"nil_equity", ctx, nil, drawdown.ErrInsufficient},
				{"empty_equity", ctx, []float64{}, drawdown.ErrInsufficient},
				{"initial_zero", ctx, []float64{math.Copysign(0, -1)}, drawdown.ErrEquity},
				{"initial_nonfinite", ctx, []float64{math.Inf(-1)}, drawdown.ErrNonFinite},
				{"later_negative", ctx, []float64{10, 5, -1, math.NaN()}, drawdown.ErrEquity},
				{"later_nonfinite", ctx, []float64{10, 5, math.NaN(), -1}, drawdown.ErrNonFinite},
			} {
				t.Run(tc.name, func(t *testing.T) {
					before := slices.Clone(tc.equity)
					got, err := projection.call(tc.ctx, tc.equity)
					mustErrorIs(t, err, tc.want)
					mustEqualBits(t, got, 0, "failed scalar is +0")
					mustEqualSlice(t, tc.equity, before)
				})
			}
			// An unrecovered trough creates a real suffix scan after the profile.
			equity := make([]float64, 2051)
			equity[0] = 100
			for _, poll := range []int{2, 4, 7} {
				// Entry; two profile polls; three suffix polls; final publication.
				got, err := projection.call(newPollContext(poll), equity)
				mustErrorIs(t, err, context.Canceled)
				mustEqualBits(t, got, 0, "canceled scalar is +0")
			}
		})
	}
}
