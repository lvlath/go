// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package rsi_test probes allocation-free streaming, large warmups, and precise receiver precedence.
// All tests use the exported Stream protocol and never inspect unexported state.
package rsi_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/rsi"
)

// TestStream_AllocationsWarmupSteadyAndFailure distinguishes three hot-path regimes.
func TestStream_AllocationsWarmupSteadyAndFailure(t *testing.T) {
	for _, tc := range []struct {
		name   string
		period int
		seed   int
		price  float64
	}{
		{"seed", 2000, 0, 101.0},
		{"steady-P1", 1, 5, 101.5},
		{"steady-P14", 14, 28, 102.0},
		{"rejected-domain", 14, 28, 0.0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			stream, err := rsi.NewPeriod(tc.period)
			if err != nil {
				t.Fatal(err)
			}
			for i := 0; i < tc.seed; i++ {
				if _, _, err := stream.Push(100 + float64(i%5)); err != nil {
					t.Fatal(err)
				}
			}
			var pushErr error
			allocs := testing.AllocsPerRun(1000, func() {
				_, _, pushErr = stream.Push(tc.price)
			})
			if tc.price == 0 {
				mustErrorIs(t, pushErr, rsi.ErrDomain)
			} else if pushErr != nil {
				t.Fatalf("%s unexpectedly rejected a positive price: %v", tc.name, pushErr)
			}
			if allocs != 0 {
				t.Fatalf("%s Push allocations/op=%.2f, want 0", tc.name, allocs)
			}
		})
	}
}

// TestStream_LargeSupportedPeriod requires no upfront period-sized buffer and no count cast corruption.
func TestStream_LargeSupportedPeriod(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	stream, err := rsi.NewPeriod(maxInt - 1)
	if err != nil {
		t.Fatal(err)
	}
	for _, price := range []float64{1, 2, 1, 3} {
		value, valid, err := stream.Push(price)
		if err != nil {
			t.Fatal(err)
		}
		mustSameFloat(t, value, 0)
		mustEqualBool(t, valid, false, "large-period warmup")
	}
	if count := stream.Snapshot().Count; count != 4 {
		t.Fatalf("Count=%d, want 4", count)
	}
}

// TestStream_ReceiverErrorPrecedence protects nil/uninitialized checks ahead of numeric input validation.
func TestStream_ReceiverErrorPrecedence(t *testing.T) {
	var absent *rsi.Stream
	_, _, err := absent.Push(math.NaN())
	mustErrorIs(t, err, rsi.ErrNilInput)
	var zero rsi.Stream
	_, _, err = zero.Push(math.NaN())
	mustErrorIs(t, err, rsi.ErrStateVersion)

	stream, err := rsi.NewPeriod(2)
	if err != nil {
		t.Fatal(err)
	}
	before := stream.Snapshot()
	for _, tc := range []struct {
		price float64
		want  error
	}{
		{math.Inf(1), rsi.ErrNonFinite},
		{math.Inf(-1), rsi.ErrNonFinite},
		{math.NaN(), rsi.ErrNonFinite},
		{math.Copysign(0, -1), rsi.ErrDomain},
		{0, rsi.ErrDomain},
		{-20, rsi.ErrDomain},
	} {
		value, valid, err := stream.Push(tc.price)
		mustErrorIs(t, err, tc.want)
		mustSameFloat(t, value, 0)
		mustEqualBool(t, valid, false, "rejected push validity")
		assertSnapshotEqual(t, before, stream.Snapshot())
	}
}
