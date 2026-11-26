// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package atr_test exercises exported numeric and checkpoint boundaries under fuzzing.
// AI-Hints: a failure must preserve state; accepted continuation must match a rational oracle.
package atr_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/atr"
)

// FuzzATR_PublicBoundaries explores public numeric validation and rollback with independent arithmetic.
// fuzzed binary64 inputs, candle geometry, and finite ranges.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - fuzzed binary64 inputs, candle geometry, and finite ranges.
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
//   - Never call a private kernel from a supposedly independent oracle.
func FuzzATR_PublicBoundaries(f *testing.F) {
	for _, bar := range []candle{{10, 8, 9}, {-1, -3, -2}, {0, 0, 0}, {math.MaxFloat64, -math.MaxFloat64, 0}, {math.NaN(), 0, 0}, {1, 2, 0}, {math.SmallestNonzeroFloat64, 0, 0}} {
		f.Add(math.Float64bits(bar.high), math.Float64bits(bar.low), math.Float64bits(bar.close), 14)
	}
	f.Add(uint64(0), uint64(0), uint64(0), 0)
	f.Add(uint64(1), uint64(0), uint64(0), 1)
	f.Fuzz(func(t *testing.T, hb, lb, cb uint64, period int) {
		bar := candle{math.Float64frombits(hb), math.Float64frombits(lb), math.Float64frombits(cb)}
		opts := atr.Options{Period: period}
		result, batchErr := atr.Compute(context.Background(), []float64{bar.high}, []float64{bar.low}, []float64{bar.close}, opts)
		s, err := atr.New(opts)
		if period < 1 {
			mustErrorIs(t, err, atr.ErrInvalidOption)
			mustErrorIs(t, batchErr, atr.ErrInvalidOption)
			if result != nil || s != nil {
				t.Fatal("invalid options published state")
			}
			return
		}
		if err != nil {
			t.Fatal(err)
		}
		before := s.Snapshot()
		value, streamErr := s.Push(context.Background(), bar.high, bar.low, bar.close)
		var expected error
		switch {
		case math.IsNaN(bar.high) || math.IsNaN(bar.low) || math.IsNaN(bar.close) || math.IsInf(bar.high, 0) || math.IsInf(bar.low, 0) || math.IsInf(bar.close, 0):
			expected = atr.ErrNonFinite
		case bar.low > bar.high || bar.close < bar.low || bar.close > bar.high:
			expected = atr.ErrBar
		default:
			if math.IsInf(rationalTR(bar, 0, false), 0) {
				expected = atr.ErrOverflow
			}
		}
		mustErrorIs(t, batchErr, expected)
		mustErrorIs(t, streamErr, expected)
		if expected != nil {
			if result != nil {
				t.Fatal("failed batch published result")
			}
			mustValue(t, value, atr.Value{})
			mustSnapshot(t, s.Snapshot(), before)
			return
		}
		want := rationalHistory([]float64{bar.high}, []float64{bar.low}, []float64{bar.close}, period)
		mustResult(t, result, want)
		mustValue(t, value, atr.Value{TR: want.TR[0], ATR: want.ATR[0], Valid: want.Valid[0]})
	})
}

// FuzzSnapshot_PublicBoundaries explores snapshot grammar, metadata, and exact continuation.
// arbitrary bounded text, phase scalars, and independent big.Int validation.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - arbitrary bounded text, phase scalars, and independent big.Int validation.
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
//   - Structural acceptance does not authenticate the history.
func FuzzSnapshot_PublicBoundaries(f *testing.F) {
	f.Add("0", uint64(0), uint64(0), uint64(0), uint32(1), 14)
	f.Add("1", uint64(13), uint64(0), uint64(0), uint32(1), 14)
	f.Add("0", uint64(14), math.Float64bits(-3), math.Float64bits(2), uint32(1), 14)
	f.Add("00", uint64(1), uint64(0), uint64(0), uint32(1), 14)
	f.Add("0", ^uint64(0), uint64(0), uint64(0), uint32(1), 14)
	f.Fuzz(func(t *testing.T, seed string, count, previous, current uint64, version uint32, period int) {
		s := newStream(t, 14)
		push(t, s, candle{3, 1, 2})
		before := s.Snapshot()
		state := atr.Snapshot{Version: version, Algorithm: "atr/wilder-rn/v1", Options: atr.Options{Period: period}, Count: count, PreviousClose: math.Float64frombits(previous), Current: math.Float64frombits(current), SeedSum: seed}
		expectedValid := snapshotValidOracle(state, atr.Options{Period: 14})
		err := s.Restore(state)
		if (err == nil) != expectedValid {
			t.Fatalf("Restore validity: err=%v want-valid=%v", err, expectedValid)
		}
		if err != nil {
			mustErrorIs(t, err, atr.ErrStateVersion)
			mustSnapshot(t, s.Snapshot(), before)
			return
		}
		mustSnapshot(t, s.Snapshot(), state)
		next := candle{2, 0, 1}
		want, wantErr := rationalSnapshotStep(t, state, next)
		got, gotErr := s.Push(context.Background(), next.high, next.low, next.close)
		mustErrorIs(t, gotErr, wantErr)
		mustValue(t, got, want)
		if wantErr != nil {
			mustSnapshot(t, s.Snapshot(), state)
		}
	})
}
