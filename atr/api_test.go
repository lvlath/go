// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package atr_test verifies facade equivalence without reaching private implementation state.
// Every convenience function is checked against its canonical public operation.
package atr_test

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/lvlath/go/atr"
)

// TestAPI_PeriodWrappers protects bit parity, explicit periods, and detached output ownership.
// The largest int period is admitted without allocating a period-sized window.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - The largest int period is admitted without allocating a period-sized window.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - Preserve exact payload, mask, error identity, and ownership assertions.
func TestAPI_PeriodWrappers(t *testing.T) {
	high, low, closed := marketBars(48)
	for _, period := range []int{1, 2, 14, 64, int(^uint(0) >> 1)} {
		t.Run(fmt.Sprint(period), func(t *testing.T) {
			want := compute(t, high, low, closed, period)
			got, err := atr.ComputePeriod(context.Background(), high, low, closed, period)
			mustErrorIs(t, err, nil)
			mustResult(t, got, want)
			stream, err := atr.NewPeriod(period)
			mustErrorIs(t, err, nil)
			for i := range high {
				value := push(t, stream, candle{high[i], low[i], closed[i]})
				mustValue(t, value, atr.Value{TR: want.TR[i], ATR: want.ATR[i], Valid: want.Valid[i]})
			}
			got.TR[0] = -1
			got.ATR[len(got.ATR)-1] = -1
			got.Valid[0] = !got.Valid[0]
			mustResult(t, compute(t, high, low, closed, period), want)
		})
	}
}

// TestAPI_DefaultIsNamedPolicy proves that named convenience does not change Options{}.
// Default construction can restore the exact state produced by explicit Period=14.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - Default construction can restore the exact state produced by explicit Period=14.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - Preserve exact payload, mask, error identity, and ownership assertions.
func TestAPI_DefaultIsNamedPolicy(t *testing.T) {
	high, low, closed := marketBars(32)
	got, err := atr.ComputeDefault(context.Background(), high, low, closed)
	mustErrorIs(t, err, nil)
	mustResult(t, got, compute(t, high, low, closed, 14))
	stream, err := atr.NewDefault()
	mustErrorIs(t, err, nil)
	explicit := newStream(t, 14)
	for i := 0; i < 5; i++ {
		push(t, explicit, candle{high[i], low[i], closed[i]})
	}
	mustErrorIs(t, stream.Restore(explicit.Snapshot()), nil)
	for i := 5; i < len(high); i++ {
		mustValue(t, push(t, stream, candle{high[i], low[i], closed[i]}), push(t, explicit, candle{high[i], low[i], closed[i]}))
	}
	invalid, err := atr.New(atr.Options{})
	mustErrorIs(t, err, atr.ErrInvalidOption)
	if invalid != nil {
		t.Fatal("canonical constructor acquired an implicit default")
	}
	empty, err := atr.Compute(context.Background(), nil, nil, nil, atr.Options{})
	mustErrorIs(t, err, atr.ErrInvalidOption)
	if !empty.IsNil() {
		t.Fatal("invalid options published a result")
	}
}

// TestAPI_WrapperFailurePrecedence anchors the canonical error surface at each wrapper.
// Nil results must accompany every failure; empty successful outputs stay nonnil.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - Nil results must accompany every failure; empty successful outputs stay nonnil.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - Preserve exact payload, mask, error identity, and ownership assertions.
func TestAPI_WrapperFailurePrecedence(t *testing.T) {
	ctx := context.Background()
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	cases := []struct {
		name              string
		ctx               context.Context
		high, low, closed []float64
		period            int
		want              error
	}{
		{"nil-context", nil, nil, nil, nil, 0, atr.ErrNilContext},
		{"canceled", canceled, nil, nil, nil, 0, context.Canceled},
		{"period", ctx, nil, nil, nil, 0, atr.ErrInvalidOption},
		{"mixed-nil", ctx, nil, []float64{1}, []float64{1}, 14, atr.ErrNilInput},
		{"shape", ctx, []float64{}, []float64{1}, []float64{1}, 14, atr.ErrShape},
		{"nonfinite", ctx, []float64{math.NaN()}, []float64{2}, []float64{1}, 14, atr.ErrNonFinite},
		{"bar", ctx, []float64{1}, []float64{2}, []float64{1}, 14, atr.ErrBar},
		{"overflow", ctx, []float64{math.MaxFloat64}, []float64{-math.MaxFloat64}, []float64{0}, 14, atr.ErrOverflow},
		{"empty", ctx, nil, []float64{}, nil, 14, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := atr.ComputePeriod(tc.ctx, tc.high, tc.low, tc.closed, tc.period)
			mustErrorIs(t, err, tc.want)
			if (got == nil) != (tc.want != nil) {
				t.Fatal("period wrapper result/error mismatch")
			}
			if tc.period == atr.DefaultPeriod {
				other, defaultErr := atr.ComputeDefault(tc.ctx, tc.high, tc.low, tc.closed)
				mustErrorIs(t, defaultErr, tc.want)
				if (other == nil) != (tc.want != nil) {
					t.Fatal("default wrapper result/error mismatch")
				}
				if tc.want == nil {
					mustResult(t, other, got)
					if other.TR == nil || other.ATR == nil || other.Valid == nil {
						t.Fatal("empty slices are nil")
					}
				}
			}
		})
	}
	for _, period := range []int{-1, 0} {
		stream, err := atr.NewPeriod(period)
		mustErrorIs(t, err, atr.ErrInvalidOption)
		if stream != nil {
			t.Fatal("bad period published a stream")
		}
	}
	for _, invalidContext := range []context.Context{nil, canceled} {
		result, err := atr.ComputeDefault(invalidContext, nil, nil, nil)
		want := error(atr.ErrNilContext)
		if invalidContext != nil {
			want = context.Canceled
		}
		mustErrorIs(t, err, want)
		if result != nil {
			t.Fatal("failed default wrapper published a result")
		}
	}
}
