// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package macd_test verifies batch/stream bit parity, rejected-price rollback,
// signed numeric boundaries, and allocation-free streaming through public methods.
package macd_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/macd"
)

// TestStream_BitwiseBatchParityAndRollback Compares every output bit and mask after deliberately interleaved invalid ticks.
func TestStream_BitwiseBatchParityAndRollback(t *testing.T) {
	prices := []float64{7, 9, 3, 11, 2, 13, 5, 17, 3, 19, 4, 20, 21}
	o := macd.Options{Fast: 2, Slow: 4, Signal: 3}
	r, e := macd.Compute(prices, o)
	if e != nil {
		t.Fatal(e)
	}
	s, e := macd.New(o)
	if e != nil {
		t.Fatal(e)
	}
	for i, x := range prices {
		before := s.Snapshot()
		_, e = s.Push(math.NaN())
		mustErrorIs(t, e, macd.ErrNonFinite)
		mustSnapshotsEqual(t, before, s.Snapshot())
		p, e := s.Push(x)
		if e != nil {
			t.Fatal(e)
		}
		mustBits(t, p.Line, r.Line.Values[i], "line")
		mustBits(t, p.Signal, r.Signal.Values[i], "signal")
		mustBits(t, p.Histogram, r.Histogram.Values[i], "hist")
		mustEqualBool(t, p.LineValid, r.Line.Valid[i], "line mask")
		mustEqualBool(t, p.SignalValid, r.Signal.Valid[i], "signal mask")
	}
}

// TestMACD07_FailedPushBeforeSlowSeed Asserts exact state preservation while Fast is seeded but Slow is still warming.
func TestMACD07_FailedPushBeforeSlowSeed(t *testing.T) {
	s, _ := macd.New(macd.Options{Fast: 2, Slow: 4, Signal: 2})
	for _, x := range []float64{1, 2, 3} {
		_, e := s.Push(x)
		if e != nil {
			t.Fatal(e)
		}
	}
	prior := s.Snapshot()
	for _, x := range []float64{math.NaN(), math.Inf(1), -2, 0} {
		_, e := s.Push(x)
		if e == nil {
			t.Fatal("accepted bad price")
		}
		mustSnapshotsEqual(t, prior, s.Snapshot())
	}
}

// TestNilAndZeroStream Checks nil-receiver and uninitialized-stream contracts without reflection or panic.
func TestNilAndZeroStream(t *testing.T) {
	var s *macd.Stream
	_, e := s.Push(1)
	mustErrorIs(t, e, macd.ErrNilInput)
	mustErrorIs(t, s.Restore(macd.Snapshot{}), macd.ErrNilInput)
	s.Reset()
	if s.Snapshot().Algorithm != "" {
		t.Fatal("nil snapshot")
	}
	z := &macd.Stream{}
	_, e = z.Push(1)
	mustErrorIs(t, e, macd.ErrStateVersion)
	mustErrorIs(t, z.Restore(macd.Snapshot{}), macd.ErrStateVersion)
}

// TestStream_ZeroAllocsPush Measures the steady-state hot path after New, excluding setup and snapshots.
func TestStream_ZeroAllocsPush(t *testing.T) {
	s, _ := macd.New(macd.Options{})
	allocs := testing.AllocsPerRun(10000, func() {
		if _, err := s.Push(101); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("Push allocations = %g", allocs)
	}
}

// FuzzStream_ParityNoPanic Varies valid positive histories to guard batch/stream bit parity and safety.
func FuzzStream_ParityNoPanic(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 5, 6, 7})
	f.Add([]byte{250, 0, 99, 42, 11, 200})
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) == 0 || len(data) > 300 {
			return
		}
		v := make([]float64, len(data))
		for i, x := range data {
			v[i] = float64(x) + 1
		}
		o := macd.Options{Fast: 2, Slow: 4, Signal: 3}
		r, e := macd.Compute(v, o)
		if e != nil {
			t.Fatal(e)
		}
		s, _ := macd.New(o)
		for i, x := range v {
			p, e := s.Push(x)
			if e != nil {
				t.Fatal(e)
			}
			mustBits(t, p.Line, r.Line.Values[i], "line")
			mustBits(t, p.Signal, r.Signal.Values[i], "signal")
			mustBits(t, p.Histogram, r.Histogram.Values[i], "hist")
		}
	})
}

// TestStream_HistogramOverflowDoesNotPartiallyCommit Uses a structurally valid adversarial state to exercise arithmetic rollback.
func TestStream_HistogramOverflowDoesNotPartiallyCommit(t *testing.T) {
	s, err := macd.New(macd.Options{Fast: 2, Slow: 3, Signal: 100})
	if err != nil {
		t.Fatal(err)
	}
	// Structurally valid but adversarial restored means stress the exact overflow gate.
	// This does not claim such a snapshot is reachable from ordinary positive histories.
	snap := s.Snapshot()
	snap.Count = 102
	snap.IntState = []uint64{2, 3, 100}
	snap.FloatState = []float64{1, math.MaxFloat64, math.MaxFloat64}
	if err = s.Restore(snap); err != nil {
		t.Fatal(err)
	}
	before := s.Snapshot()
	_, err = s.Push(1)
	mustErrorIs(t, err, macd.ErrOverflow)
	mustSnapshotsEqual(t, before, s.Snapshot())
}

// TestStream_SignedSignalOppositeMagnitude verifies that a valid opposite-sign
// signal update avoids overflowing sample-old when its convex sum is finite.
// Structurally valid restored states stress arithmetic independently of whether
// these extreme means arise from an ordinary positive close history.
func TestStream_SignedSignalOppositeMagnitude(t *testing.T) {
	for _, tc := range []struct {
		name       string
		fast, slow float64
		signal     float64
		close      float64
	}{
		{"positive_line_negative_signal", math.MaxFloat64, math.SmallestNonzeroFloat64, -math.MaxFloat64, math.MaxFloat64},
		{"negative_line_positive_signal", math.SmallestNonzeroFloat64, math.MaxFloat64, math.MaxFloat64, math.SmallestNonzeroFloat64},
	} {
		t.Run(tc.name, func(t *testing.T) {
			stream, err := macd.NewWithPeriods(2, 3, 2)
			if err != nil {
				t.Fatal(err)
			}
			state := stream.Snapshot()
			state.Count = 4
			state.IntState = []uint64{2, 3, 2}
			state.FloatState = []float64{tc.fast, tc.slow, tc.signal}
			if err = stream.Restore(state); err != nil {
				t.Fatal(err)
			}
			point, err := stream.Push(tc.close)
			if err != nil {
				t.Fatalf("representable signed update was rejected: %v", err)
			}
			if !point.LineValid || !point.SignalValid || math.IsInf(point.Signal, 0) || math.IsNaN(point.Signal) {
				t.Fatalf("unexpected signed signal point: %+v", point)
			}
			if stream.Snapshot().Count != 5 {
				t.Fatal("successful signed signal step did not commit")
			}
		})
	}
}
