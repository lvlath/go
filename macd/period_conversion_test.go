// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package macd_test verifies that validated native-width periods retain their
// exact unsigned snapshot encoding without adding artificial 32-bit ceilings.
package macd_test

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"math/bits"
	"testing"

	"github.com/lvlath/go/macd"
)

// TestSnapshot_NativeIntPeriodEncoding compares the public configuration digest
// with independently encoded native-width periods, including values above 2^32
// on 64-bit hosts, without referring to private period conversion functions.
//
// Implementation:
//   - Stage 1: construct the largest admissible Fast/Slow horizons for this host.
//   - Stage 2: encode their exact LE64 integers independently of the MACD code.
//   - Stage 3: compare SHA-256 identities and replay two accepted prices.
//
// Behavior highlights:
//   - The maximum periods cannot complete warmup on these short test histories.
//   - A 32-bit-specific period clamp would change the configuration or reject New.
//
// Determinism:
//   - The expected digest covers exact ASCII+LE64 bytes, not normalized decimal text.
//
// Complexity:
//   - Time O(1), Space O(1), with a fixed number of tiny inputs.
//
// AI-Hints:
//   - This test proves preservation of the full period range, not performance on a
//     stream with a practically unrealistically long warmup horizon.
func TestSnapshot_NativeIntPeriodEncoding(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	options := macd.Options{Fast: maxInt - 1, Slow: maxInt, Signal: 1}

	active, err := macd.New(options)
	if err != nil {
		t.Fatalf("maximal native-width periods must remain valid: %v", err)
	}

	const algorithm = "macd/sma-ema/v2"
	const periodBytes = 8
	var wire [len(algorithm) + 3*periodBytes]byte
	copy(wire[:], algorithm)
	binary.LittleEndian.PutUint64(wire[len(algorithm):], uint64(options.Fast))
	binary.LittleEndian.PutUint64(wire[len(algorithm)+periodBytes:], uint64(options.Slow))
	binary.LittleEndian.PutUint64(wire[len(algorithm)+2*periodBytes:], uint64(options.Signal))
	digest := sha256.Sum256(wire[:])
	wantDigest := hex.EncodeToString(digest[:])

	initial := active.Snapshot()
	if initial.Config != wantDigest || initial.Algorithm != algorithm {
		t.Fatalf("LE64 policy mismatch: got=%q want=%q", initial.Config, wantDigest)
	}
	if bits.UintSize == 64 && uint64(options.Slow) <= math.MaxUint32 {
		t.Fatal("64-bit maximal period did not exceed the 32-bit limit")
	}

	standby, err := macd.New(options)
	if err != nil {
		t.Fatal(err)
	}
	if err := standby.Restore(initial); err != nil {
		t.Fatalf("compatible checkpoint rejected: %v", err)
	}

	for _, price := range []float64{93.125, 93.250} {
		leader, leaderErr := active.Push(price)
		follower, followerErr := standby.Push(price)
		if leaderErr != nil || followerErr != nil {
			t.Fatalf("unexpected update errors: leader=%v standby=%v", leaderErr, followerErr)
		}
		if leader.LineValid != follower.LineValid || leader.SignalValid != follower.SignalValid ||
			math.Float64bits(leader.Line) != math.Float64bits(follower.Line) ||
			math.Float64bits(leader.Signal) != math.Float64bits(follower.Signal) ||
			math.Float64bits(leader.Histogram) != math.Float64bits(follower.Histogram) {
			t.Fatal("replayed recurrence mismatch")
		}
	}

	latest := active.Snapshot()
	if latest.Count != 2 || latest.IntState[0] != 2 || latest.IntState[1] != 2 || latest.IntState[2] != 0 {
		t.Fatalf("unexpected warmup counters: %+v", latest)
	}
	mustSnapshotsEqual(t, latest, standby.Snapshot())
}
