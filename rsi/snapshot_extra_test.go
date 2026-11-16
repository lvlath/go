// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package rsi_test tests adversarial checkpoint boundaries that are invisible to black-box parity tests.
// All checks operate only through exported state transport and public stream methods.
package rsi_test

import (
	"math"
	"strings"
	"testing"

	"github.com/lvlath/go/rsi"
)

// TestSnapshot_RejectImpossibleSingleDifference is a regression for a forged two-sided first delta.
func TestSnapshot_RejectImpossibleSingleDifference(t *testing.T) {
	for _, period := range []int{1, 2, 14} {
		s, err := rsi.NewPeriod(period)
		if err != nil {
			t.Fatal(err)
		}
		before := s.Snapshot()

		// Count=2 means exactly one price change. No signed difference can be
		// both positive and negative, regardless of how the period is normalized.
		for _, gainLoss := range [][2]float64{{1, 0.1}, {math.SmallestNonzeroFloat64, math.MaxFloat64 / 2}} {
			bad := s.Snapshot()
			bad.Count = 2
			bad.FloatState = []float64{123.5, gainLoss[0], gainLoss[1]}
			mustErrorIs(t, s.Restore(bad), rsi.ErrStateVersion)
			assertSnapshotEqual(t, before, s.Snapshot())
		}

		// One gain or one loss is a legal first difference and must not be
		// rejected merely because the destination is still warming up.
		for _, legal := range [][2]float64{{1, 0}, {0, 1}, {0, 0}} {
			checkpoint := s.Snapshot()
			checkpoint.Count = 2
			checkpoint.FloatState = []float64{123.5, legal[0], legal[1]}
			if err := s.Restore(checkpoint); err != nil {
				t.Fatalf("period %d rejected valid one-delta shape: %v", period, err)
			}
			assertSnapshotEqual(t, checkpoint, s.Snapshot())
		}
	}
}

// TestSnapshot_StateDomainAndLengthMatrix protects the v1 ABI from malformed external checkpoints.
func TestSnapshot_StateDomainAndLengthMatrix(t *testing.T) {
	s, err := rsi.NewPeriod(14)
	if err != nil {
		t.Fatal(err)
	}
	for _, price := range []float64{105, 109, 107, 112, 110} {
		if _, _, err := s.Push(price); err != nil {
			t.Fatal(err)
		}
	}
	before := s.Snapshot()

	cases := []struct {
		name   string
		change func(*rsi.Snapshot)
	}{
		{"no-float-state", func(x *rsi.Snapshot) { x.FloatState = nil }},
		{"overlong-float-state", func(x *rsi.Snapshot) { x.FloatState = append(x.FloatState, 0) }},
		{"reserved-integer-state", func(x *rsi.Snapshot) { x.IntState = []uint64{1} }},
		{"wrong-algorithm", func(x *rsi.Snapshot) { x.Algorithm = "rsi/wilder/v2" }},
		{"uppercase-config", func(x *rsi.Snapshot) { x.Config = strings.ToUpper(x.Config) }},
		{"negative-loss", func(x *rsi.Snapshot) { x.FloatState[2] = -math.SmallestNonzeroFloat64 }},
		{"positive-inf-gain", func(x *rsi.Snapshot) { x.FloatState[1] = math.Inf(1) }},
		{"negative-inf-loss", func(x *rsi.Snapshot) { x.FloatState[2] = math.Inf(-1) }},
		{"nan-previous", func(x *rsi.Snapshot) { x.FloatState[0] = math.NaN() }},
		{"zero-previous", func(x *rsi.Snapshot) { x.FloatState[0] = 0 }},
		{"count-zero-with-previous", func(x *rsi.Snapshot) { x.Count = 0 }},
		{"count-one-with-averages", func(x *rsi.Snapshot) { x.Count = 1 }},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			candidate := s.Snapshot()
			tc.change(&candidate)
			mustErrorIs(t, s.Restore(candidate), rsi.ErrStateVersion)
			assertSnapshotEqual(t, before, s.Snapshot())
		})
	}
}

// TestSnapshot_ResetAndConfiguration verifies that reset does not rewrite the period fingerprint.
func TestSnapshot_ResetAndConfiguration(t *testing.T) {
	s, err := rsi.NewPeriod(7)
	if err != nil {
		t.Fatal(err)
	}
	oldConfig := s.Snapshot().Config
	for _, price := range []float64{99, 100, 104, 102, 98, 97, 102, 106, 105} {
		if _, _, err := s.Push(price); err != nil {
			t.Fatal(err)
		}
	}
	oldCheckpoint := s.Snapshot()
	s.Reset()
	resetCheckpoint := s.Snapshot()
	mustEqualString(t, resetCheckpoint.Config, oldConfig, "reset config")
	if resetCheckpoint.Count != 0 {
		t.Fatalf("reset Count=%d, want 0", resetCheckpoint.Count)
	}
	fresh, err := rsi.NewPeriod(7)
	if err != nil {
		t.Fatal(err)
	}
	assertSnapshotEqual(t, fresh.Snapshot(), resetCheckpoint)

	// A valid older checkpoint remains detached; restoring it later does not
	// retrospectively modify the fresh reset checkpoint already held by the caller.
	if err := s.Restore(oldCheckpoint); err != nil {
		t.Fatal(err)
	}
	assertSnapshotEqual(t, oldCheckpoint, s.Snapshot())
	assertSnapshotEqual(t, fresh.Snapshot(), resetCheckpoint)
}

// TestSnapshot_RejectMismatchedPeriod verifies that the digest binds normalized period, not the integer spelling.
func TestSnapshot_RejectMismatchedPeriod(t *testing.T) {
	defaultStream, err := rsi.New(rsi.Options{})
	if err != nil {
		t.Fatal(err)
	}
	explicit14, err := rsi.NewPeriod(14)
	if err != nil {
		t.Fatal(err)
	}
	if err := explicit14.Restore(defaultStream.Snapshot()); err != nil {
		t.Fatalf("equivalent normalized period unexpectedly rejected: %v", err)
	}
	wrong, err := rsi.NewPeriod(7)
	if err != nil {
		t.Fatal(err)
	}
	before := wrong.Snapshot()
	mustErrorIs(t, wrong.Restore(defaultStream.Snapshot()), rsi.ErrStateVersion)
	assertSnapshotEqual(t, before, wrong.Snapshot())
}

// TestSnapshot_MaxUint64RejectsNextPriceEvenWhenPriceIsValid verifies counter saturation before any commit.
func TestSnapshot_MaxUint64RejectsNextPriceEvenWhenPriceIsValid(t *testing.T) {
	s, err := rsi.NewPeriod(1)
	if err != nil {
		t.Fatal(err)
	}
	checkpoint := s.Snapshot()
	checkpoint.Count = math.MaxUint64
	checkpoint.FloatState = []float64{30, 1, 0}
	if err := s.Restore(checkpoint); err != nil {
		t.Fatal(err)
	}
	before := s.Snapshot()
	value, valid, err := s.Push(31)
	mustErrorIs(t, err, rsi.ErrOverflow)
	mustSameFloat(t, value, 0)
	mustEqualBool(t, valid, false, "overflow validity")
	assertSnapshotEqual(t, before, s.Snapshot())
}
